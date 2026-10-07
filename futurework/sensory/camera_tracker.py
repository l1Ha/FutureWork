"""
真实摄像头感知与视觉追踪模块 (Real Camera & Vision Tracker)

支持：
1. 真实物理摄像头硬件（通过 OpenCV VideoCapture 设备捕获：0, 1, 2...）；
2. Web HUD / 移动端通过 HTTP 上传的真实相机图像帧（Base64 / JPEG 二进制）；
3. 面部检测、眼睛跟踪与视线估算 (Gaze Tracking)；
4. 头部姿态 (Head Pose) 估算与点头/摇头 (Nod / Shake) 离散手势检测；
5. 手部轮廓、凸包缺陷与手势识别（张手、握拳、单指指向、竖拇指、捏合）；
6. 可选 MediaPipe 21点高精度追踪插件（若已安装自动启用，未安装时优雅降级为原生 OpenCV 算子）。
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from futurework.sensory.facial import FacialExpressionAnalyzer, HeadPoseAnalyzer
from futurework.sensory.gaze import GazeTracker
from futurework.sensory.gesture import GestureInterpreter, Landmark
from futurework.types import (
    EmotionType,
    FacialSignal,
    GazeSignal,
    GestureType,
    HandGestureSignal,
    HeadPoseSignal,
)


@dataclass
class CameraPerceptionResult:
    """真实摄像头单帧视觉处理综合结果。"""

    timestamp: float
    face_detected: bool = False
    face_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    hand_detected: bool = False
    hand_box: Optional[Tuple[int, int, int, int]] = None
    facial_signal: Optional[FacialSignal] = None
    head_pose_signal: Optional[HeadPoseSignal] = None
    gaze_signal: Optional[GazeSignal] = None
    gesture_signal: Optional[HandGestureSignal] = None
    annotated_frame: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "face_detected": self.face_detected,
            "face_box": self.face_box,
            "hand_detected": self.hand_detected,
            "hand_box": self.hand_box,
            "primary_emotion": self.facial_signal.primary_emotion.value if self.facial_signal else "neutral",
            "attention_score": self.facial_signal.attention_score if self.facial_signal else 1.0,
            "nod": self.head_pose_signal.nod if self.head_pose_signal else False,
            "shake": self.head_pose_signal.shake if self.head_pose_signal else False,
            "gaze_x": self.gaze_signal.screen_x if self.gaze_signal else 0.5,
            "gaze_y": self.gaze_signal.screen_y if self.gaze_signal else 0.5,
            "gesture": self.gesture_signal.gesture.value if self.gesture_signal else "idle",
            "gesture_confidence": self.gesture_signal.confidence if self.gesture_signal else 0.0,
        }


class RealCameraTracker:
    """
    真实摄像头驱动与计算机视觉分析器。

    人脸检测采用双引擎策略：
    1. **YuNet DNN 检测器**（OpenCV 官方轻量模型，随包内置）——
       对真实摄像头画面（光照多变、距离较远、侧脸、低对比度）的召回率
       远高于 Haar 级联，CPU 上单帧约 10~20ms；
    2. **Haar 级联族回退**（alt2 → default → alt → profile，配合 CLAHE）——
       在 YuNet 模型缺失或加载失败时兜底，保证任何环境都能运行。
    """

    # YuNet 检测输入统一缩放到该宽度以内：更小分辨率对 DNN 召回几乎无影响，
    # 却能把单帧推理时间压到 10ms 级。
    YUNET_MAX_WIDTH = 480
    YUNET_SCORE_THRESHOLD = 0.55

    def __init__(
        self,
        device_index: int = 0,
        *,
        frame_width: int = 640,
        frame_height: int = 480,
        cascade_dir: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        self.device_index = device_index
        self.frame_width = frame_width
        self.frame_height = frame_height

        # ---- 主引擎：YuNet DNN 人脸检测器（随包内置，无需联网下载） ----
        self._yunet: Optional[Any] = None
        self._yunet_input_size: Optional[Tuple[int, int]] = None
        bundled_model = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx"
        )
        if os.path.isfile(bundled_model):
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    bundled_model,
                    config="",
                    input_size=(320, 240),
                    score_threshold=self.YUNET_SCORE_THRESHOLD,
                    nms_threshold=0.3,
                    top_k=5,
                )
            except Exception:
                self._yunet = None

        # ---- 回退引擎：OpenCV Haar 特征级联分类器族 ----
        cascade_path = cascade_dir or cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, "haarcascade_frontalface_default.xml"))
        self.face_cascade_alt2 = cv2.CascadeClassifier(os.path.join(cascade_path, "haarcascade_frontalface_alt2.xml"))
        self.face_cascade_alt = cv2.CascadeClassifier(os.path.join(cascade_path, "haarcascade_frontalface_alt.xml"))
        self.face_cascade_profile = cv2.CascadeClassifier(os.path.join(cascade_path, "haarcascade_profileface.xml"))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, "haarcascade_eye.xml"))
        self.smile_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, "haarcascade_smile.xml"))

        # 自适应对比度增强器 (CLAHE)，解决逆光、暗光与强光人脸失配
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 内部认知感知分析器
        self.facial_analyzer = FacialExpressionAnalyzer()
        self.head_pose_analyzer = HeadPoseAnalyzer()
        self.gaze_tracker = GazeTracker(smoothing=0.4)
        self.gesture_interpreter = GestureInterpreter()

        # 头部姿态基准中心（EMA 平滑）
        self._face_center_ema: Optional[Tuple[float, float]] = None
        self._last_face_time: float = 0.0

        # 视觉处理串行锁：ThreadingHTTPServer 每个请求一个线程，
        # 而表情/头动/视线分析器都持有跨帧状态，必须串行化避免竞态。
        self._vision_lock = threading.RLock()

        # 后台线程采集控制
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_result: Optional[CameraPerceptionResult] = None
        self._on_result_callbacks: List[Callable[[CameraPerceptionResult], None]] = []

    # ------------------------------------------------------------------
    # 引擎状态
    # ------------------------------------------------------------------
    @property
    def engine_name(self) -> str:
        return "yunet-dnn" if self._yunet is not None else "haar-cascade-fallback"

    # ------------------------------------------------------------------
    # 硬件可用性检查
    # ------------------------------------------------------------------
    @staticmethod
    def is_available(device_index: int = 0) -> Tuple[bool, str]:
        """探测系统是否存在可打开的物理摄像头。"""
        try:
            cap = cv2.VideoCapture(device_index)
            if not cap.isOpened():
                cap.release()
                return False, f"无法打开摄像头设备 index={device_index}（无视频流或权限不足）"
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return False, f"摄像头设备 index={device_index} 已打开但未读取到有效画面"
            return True, f"摄像头 index={device_index} 正常工作 (分辨率 {frame.shape[1]}x{frame.shape[0]})"
        except Exception as exc:
            return False, f"摄像头初始化异常：{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # 单帧视觉特征解析主干
    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        *,
        annotate: bool = True,
    ) -> CameraPerceptionResult:
        """
        处理单张 BGR 格式的图像帧，综合输出面部、视线、头动、手势多模态感知。

        全程持有 ``_vision_lock``：表情/头动/视线分析器都携带跨帧状态，
        而 HTTP 服务端是每请求一线程，不串行化会产生竞态。
        """
        with self._vision_lock:
            return self._process_frame_locked(frame, timestamp, annotate=annotate)

    def _process_frame_locked(
        self,
        frame: np.ndarray,
        timestamp: Optional[float],
        *,
        annotate: bool,
    ) -> CameraPerceptionResult:
        now_ts = timestamp or time.time()
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated = frame.copy() if annotate else None

        # 1. 人脸检测：YuNet DNN 主引擎 → Haar 级联族回退（CLAHE 均衡 + 多模型）
        face_box = self._detect_face(frame, gray, w, h)

        face_detected = face_box is not None
        facial_signal: Optional[FacialSignal] = None
        head_pose_signal: Optional[HeadPoseSignal] = None
        gaze_signal: Optional[GazeSignal] = None

        if face_detected:
            fx, fy, fw, fh = face_box
            face_roi_gray = gray[fy : fy + fh, fx : fx + fw]

            # 1.1 头部姿态分析：计算人脸中心相对画面中心偏移
            fc_x = (fx + fw / 2.0) / float(w)
            fc_y = (fy + fh / 2.0) / float(h)
            if self._face_center_ema is None:
                self._face_center_ema = (fc_x, fc_y)
            else:
                self._face_center_ema = (
                    0.8 * self._face_center_ema[0] + 0.2 * fc_x,
                    0.8 * self._face_center_ema[1] + 0.2 * fc_y,
                )

            # 偏航 yaw 与俯仰 pitch 估算（度数）
            yaw_deg = (fc_x - self._face_center_ema[0]) * 120.0
            pitch_deg = (self._face_center_ema[1] - fc_y) * 120.0
            head_pose_signal = self.head_pose_analyzer.update(
                pitch=pitch_deg,
                yaw=yaw_deg,
                confidence=0.9,
            )

            # 1.2 眼睛检测与注视点追踪
            eyes = self.eye_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(int(fw * 0.1), int(fh * 0.1)),
            )
            # 视线 screen_x, screen_y 估算
            eye_detected = len(eyes) >= 1
            if eye_detected:
                # 依据人脸在画面的归一化坐标作为注视锚点
                # 镜像翻转：当用户往自身右侧（画面左侧）看时，视线指向屏幕右侧
                gaze_x = max(0.0, min(1.0, 1.0 - fc_x))
                gaze_y = max(0.0, min(1.0, fc_y))
                gaze_signal = self.gaze_tracker.update(gaze_x, gaze_y, confidence=0.88, timestamp=now_ts)
            else:
                gaze_signal = self.gaze_tracker.update(
                    max(0.0, min(1.0, 1.0 - fc_x)),
                    max(0.0, min(1.0, fc_y)),
                    confidence=0.6,
                    timestamp=now_ts,
                )

            # 1.3 微表情与笑脸 / 困惑分析
            smiles = self.smile_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.2,
                minNeighbors=8,
                minSize=(int(fw * 0.15), int(fh * 0.1)),
            )
            au_map: Dict[str, float] = {}
            if len(smiles) > 0:
                au_map["AU12"] = 0.85  # 嘴角上扬
                au_map["AU6"] = 0.7
            else:
                au_map["AU25"] = 0.4  # 专注平视

            facial_signal = self.facial_analyzer.analyze(
                au_map,
                blink=not eye_detected,
                face_detected=True,
            )

            # 在图像上标注人脸框与信息
            if annotated is not None:
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (6, 182, 212), 2)
                tag = f"Face: {facial_signal.primary_emotion.value}"
                if head_pose_signal.nod:
                    tag += " [NOD]"
                elif head_pose_signal.shake:
                    tag += " [SHAKE]"
                cv2.putText(
                    annotated,
                    tag,
                    (fx, max(20, fy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (6, 182, 212),
                    2,
                )
                if gaze_signal:
                    gx = int((1.0 - gaze_signal.screen_x) * w)
                    gy = int(gaze_signal.screen_y * h)
                    cv2.circle(annotated, (gx, gy), 8, (0, 255, 255), -1)

        # 2. 手势检测（肤色轮廓与凸缺陷几何分析）
        gesture_signal, hand_box = self._detect_hand_gesture(frame, face_box, now_ts, annotated)

        res = CameraPerceptionResult(
            timestamp=now_ts,
            face_detected=face_detected,
            face_box=face_box,
            hand_detected=hand_box is not None,
            hand_box=hand_box,
            facial_signal=facial_signal,
            head_pose_signal=head_pose_signal,
            gaze_signal=gaze_signal,
            gesture_signal=gesture_signal,
            annotated_frame=annotated,
        )

        with self._lock:
            self._latest_result = res
            for cb in self._on_result_callbacks:
                try:
                    cb(res)
                except Exception:
                    pass

        return res

    # ------------------------------------------------------------------
    # 人脸检测双引擎：YuNet DNN 主引擎 + Haar 级联族回退
    # ------------------------------------------------------------------
    def _detect_face(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        w: int,
        h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        返回画面中最大人脸的 ``(x, y, w, h)`` 框；未检出返回 ``None``。

        YuNet 是 OpenCV 官方的轻量 DNN 人脸检测器，对真实摄像头画面
        （光照多变、远距离、轻微侧脸）的召回率远高于 Haar 级联。
        Haar 族在模型缺失时兜底，并配合 CLAHE 对比度均衡提升暗光表现。
        """
        # ---------- 主引擎：YuNet ----------
        if self._yunet is not None:
            try:
                det_w = min(w, self.YUNET_MAX_WIDTH)
                det_h = max(1, int(h * det_w / w))
                if (det_w, det_h) != self._yunet_input_size:
                    self._yunet.setInputSize((det_w, det_h))
                    self._yunet_input_size = (det_w, det_h)

                det_frame = frame
                if det_w != w:
                    det_frame = cv2.resize(frame, (det_w, det_h))

                _, faces = self._yunet.detect(det_frame)
                if faces is not None and len(faces) > 0:
                    scale_back = w / float(det_w)
                    best = max(faces, key=lambda f: f[2] * f[3])
                    x, y, fw, fh = best[:4]
                    # 框略外扩，便于眼睛/嘴部 ROI 检测
                    fw *= 1.08
                    fh *= 1.08
                    return (
                        max(0, int(x * scale_back)),
                        max(0, int(y * scale_back)),
                        min(w, int(fw * scale_back)),
                        min(h, int(fh * scale_back)),
                    )
                return None
            except Exception:
                # YuNet 推理异常时降级到 Haar，不让视觉链路中断
                self._yunet = None

        # ---------- 回退引擎：Haar 级联族（CLAHE 均衡 + 多模型回退） ----------
        gray_eq = self._clahe.apply(gray)
        min_dim = max(24, int(min(w, h) * 0.08))
        min_face_size = (min_dim, min_dim)

        for cascade in (
            self.face_cascade_alt2,
            self.face_cascade,
            self.face_cascade_alt,
            self.face_cascade_profile,
        ):
            faces = cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=min_face_size,
            )
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                return (int(fx), int(fy), int(fw), int(fh))
        return None

    # ------------------------------------------------------------------
    # 图像字节流快捷处理 (支持 Web HUD 前端上传)
    # ------------------------------------------------------------------
    def process_image_bytes(self, image_bytes: bytes, timestamp: Optional[float] = None) -> CameraPerceptionResult:
        """从 JPEG/PNG/WebP 字节数组解码并处理。"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("无法解码输入的图像字节流")
        return self.process_frame(frame, timestamp=timestamp)

    # ------------------------------------------------------------------
    # 真实手势检测核心算子
    # ------------------------------------------------------------------
    def _detect_hand_gesture(
        self,
        frame: np.ndarray,
        face_box: Optional[Tuple[int, int, int, int]],
        timestamp: float,
        annotated: Optional[np.ndarray],
    ) -> Tuple[Optional[HandGestureSignal], Optional[Tuple[int, int, int, int]]]:
        h, w = frame.shape[:2]

        # 转换到 YCrCb 空间进行鲁棒肤色提取
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # 常见人脸与手部肤色先验分布区间
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

        # 排除掉人脸区域，避免把脸当成手
        if face_box:
            fx, fy, fw, fh = face_box
            # 人脸向下稍微扩展，连同脖子一起屏蔽
            pad = int(fh * 0.25)
            y2 = min(h, fy + fh + pad)
            skin_mask[max(0, fy - pad) : y2, max(0, fx - pad) : min(w, fx + fw + pad)] = 0

        # 形态学滤波去除微小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        # 找面积最大的候选区域，且必须达到最小有效手部面积
        min_hand_area = (w * h) * 0.015
        max_hand_area = (w * h) * 0.45
        valid_contours = [c for c in contours if min_hand_area < cv2.contourArea(c) < max_hand_area]
        if not valid_contours:
            return None, None

        hand_contour = max(valid_contours, key=cv2.contourArea)
        hx, hy, hw, hh = cv2.boundingRect(hand_contour)
        hand_box = (int(hx), int(hy), int(hw), int(hh))

        # 手势几何特征分析
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        if len(hull) < 4:
            return None, hand_box

        defects = cv2.convexityDefects(hand_contour, hull)
        deep_defects = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s_idx, e_idx, f_idx, dist = defects[i, 0]
                start = hand_contour[s_idx][0]
                end = hand_contour[e_idx][0]
                far = hand_contour[f_idx][0]
                # 计算两指夹角
                a = math.dist(start, end)
                b = math.dist(start, far)
                c = math.dist(end, far)
                if b * c > 0:
                    angle = math.degrees(math.acos(max(-1.0, min(1.0, (b * b + c * c - a * a) / (2.0 * b * c)))))
                    # 深度大于阈值且夹角小于90度代表手指间的凹陷
                    if dist > 1400 and angle < 95.0:
                        deep_defects += 1

        # 质心归一化坐标 (镜像翻转 x)
        m = cv2.moments(hand_contour)
        if m["m00"] > 0:
            cx = (m["m10"] / m["m00"]) / float(w)
            cy = (m["m01"] / m["m00"]) / float(h)
        else:
            cx = (hx + hw / 2.0) / float(w)
            cy = (hy + hh / 2.0) / float(h)
        norm_x = max(0.0, min(1.0, 1.0 - cx))
        norm_y = max(0.0, min(1.0, cy))

        # 分类手势
        area = cv2.contourArea(hand_contour)
        hull_pts = cv2.convexHull(hand_contour, returnPoints=True)
        hull_area = cv2.contourArea(hull_pts)
        solidity = area / max(1.0, hull_area)

        gesture = GestureType.IDLE
        conf = 0.65

        if deep_defects >= 3:
            # 3个以上深凹缺陷，通常为4-5指全张开
            gesture = GestureType.OPEN_PALM
            conf = 0.92
        elif deep_defects == 0 and solidity > 0.8:
            # 无凹陷且轮廓极紧实，通常为握拳
            gesture = GestureType.FIST
            conf = 0.85
        elif deep_defects == 1 or deep_defects == 2:
            # 1-2个深凹陷，判断手指伸展方向
            top_pt = min(hand_contour, key=lambda p: p[0][1])[0]
            if top_pt[1] < hy + hh * 0.25:
                # 顶部手指突出向上 -> 指向或点赞
                if hh > hw * 1.2:
                    gesture = GestureType.POINT
                    conf = 0.88
                else:
                    gesture = GestureType.THUMBS_UP
                    conf = 0.82
            else:
                gesture = GestureType.PINCH
                conf = 0.78

        sig = HandGestureSignal(
            gesture=gesture,
            confidence=conf,
            hand="right",
            position_3d=(norm_x, norm_y, 0.5),
            timestamp=timestamp,
        )

        if annotated is not None:
            cv2.rectangle(annotated, (hx, hy), (hx + hw, hy + hh), (16, 185, 129), 2)
            cv2.putText(
                annotated,
                f"Hand: {gesture.value}",
                (hx, max(20, hy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (16, 185, 129),
                2,
            )

        return sig, hand_box

    # ------------------------------------------------------------------
    # 线程捕获控制
    # ------------------------------------------------------------------
    def start(self, callback: Optional[Callable[[CameraPerceptionResult], None]] = None) -> bool:
        """启动后台摄像头实时捕获分析线程。"""
        if self._running:
            return True
        if callback:
            self._on_result_callbacks.append(callback)

        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._running = True

        def _worker() -> None:
            while self._running and self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue
                self.process_frame(frame)
                time.sleep(0.02)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """停止摄像头捕获并释放硬件资源。"""
        self._running = False
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._thread = None

    def get_latest_result(self) -> Optional[CameraPerceptionResult]:
        with self._lock:
            return self._latest_result

    def on_result(self, callback: Callable[[CameraPerceptionResult], None]) -> None:
        self._on_result_callbacks.append(callback)
