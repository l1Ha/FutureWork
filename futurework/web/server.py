"""
FutureWork 可视化 HUD 与交互式 Web 控制台服务端 (Web HUD & Interaction Server)

基于 Python 原生轻量 HTTP 服务，零重型第三方依赖：
- 提供面向未来工作站的多模态状态监视器 (Sensory HUD)
- 实时视线 (Gaze)、手势 (Gesture)、面部表情 (Facial)、工作记忆 (Memory) 联动控制台
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from futurework.runtime.session import FutureWorkSession
from futurework.sensory.camera_tracker import RealCameraTracker
from futurework.types import GestureType


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _probe_futurework_status(host: str, port: int) -> bool:
    """探测指定端口上是否已有正在运行的 FutureWork Web HUD 实例。"""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        req = urllib.request.Request(f"http://{probe_host}:{port}/api/status")
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            data = json.loads(resp.read().decode())
            return isinstance(data, dict) and "adapters" in data
    except Exception:
        return False


class FutureWorkWebHandler(BaseHTTPRequestHandler):
    session: FutureWorkSession
    static_dir: str

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            index_file = os.path.join(self.static_dir, "index.html")
            if os.path.isfile(index_file):
                with open(index_file, "rb") as f:
                    content = f.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        if self.path == "/api/status":
            health = self.session.health()
            self._send_json(health)
            return

        if self.path == "/api/vision/status":
            avail, reason = self.camera_tracker.is_available()
            self._send_json({
                "camera_available": avail,
                "reason": reason,
                "engine": self.camera_tracker.engine_name,
            })
            return

        if self.path == "/api/files":
            workdir = self.session.orchestrator.workdir or os.getcwd()
            items = []
            try:
                for entry in sorted(os.listdir(workdir)):
                    if entry.startswith(".") and entry != ".futurework":
                        continue
                    full = os.path.join(workdir, entry)
                    items.append({
                        "name": entry,
                        "is_dir": os.path.isdir(full),
                        "size": os.path.getsize(full) if os.path.isfile(full) else None,
                    })
            except Exception:
                items = []
            self._send_json({"workdir": workdir, "files": items})
            return

        if self.path == "/api/todos":
            prod = self.session.orchestrator.registry.get("productivity")
            todos = prod._load_todos() if (prod and hasattr(prod, "_load_todos")) else []
            self._send_json({"todos": todos})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path == "/api/interact":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw_body)
            except Exception:
                payload = {}

            text = payload.get("text", "").strip()
            gaze = payload.get("gaze")
            look_at = (float(gaze[0]), float(gaze[1])) if (gaze and len(gaze) >= 2) else None

            # 动作模拟
            if payload.get("nod"):
                turn = self.session.nod()
                query_label = "[点头 Nod]"
            elif payload.get("shake"):
                turn = self.session.shake()
                query_label = "[摇头 Shake]"
            elif payload.get("pinch"):
                turn = self.session.pinch()
                query_label = "[捏合 Pinch]"
            else:
                gesture = GestureType(payload["gesture"]) if payload.get("gesture") in GestureType._value2member_map_ else None
                turn = self.session.say(text, gesture=gesture, look_at=look_at)
                query_label = text

            recent_mems = self.session.orchestrator.dialogue.memory.query_recent(limit=5)
            mem_data = [{"entity_id": m.entity_id, "label": m.label, "target_type": m.target_type} for m in recent_mems]

            resp = {
                "query": query_label,
                "status": turn.status.value,
                "feedback": turn.feedback.text,
                "intent_action": turn.intent.action if turn.intent else None,
                "confidence": turn.fused.confidence if turn.fused else None,
                "latency_ms": turn.latency_ms,
                "memory_entities": mem_data,
            }
            self._send_json(resp)
            return

        if self.path == "/api/vision/frame":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")

            image_bytes = None
            if "application/json" in content_type:
                try:
                    payload = json.loads(raw_body.decode("utf-8"))
                    data_url = payload.get("image", "")
                    if "," in data_url:
                        image_bytes = base64.b64decode(data_url.split(",", 1)[1])
                except Exception:
                    pass
            else:
                image_bytes = raw_body

            if not image_bytes:
                self._send_json({"error": "缺少有效图像帧数据"})
                return

            try:
                res = self.camera_tracker.process_image_bytes(image_bytes)
                # 注入会话多模态流中
                if res.facial_signal:
                    self.session.streaming_loop.feed_facial(res.facial_signal)
                if res.head_pose_signal:
                    self.session.streaming_loop.feed_head_pose(res.head_pose_signal)
                if res.gaze_signal:
                    self.session.streaming_loop.feed_gaze(res.gaze_signal)
                if res.gesture_signal:
                    self.session.streaming_loop.feed_gestures([res.gesture_signal])

                self._send_json(res.to_dict())
            except Exception as exc:
                self._send_json({"error": f"图像处理异常：{type(exc).__name__}: {exc}"})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _send_json(self, data: dict) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args) -> None:
        # 静默常规请求日志，避免刷屏
        pass


def run_web_hud(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    workdir: Optional[str] = None,
    blocking: bool = True,
    auto_open: bool = False,
    camera_index: int = 0,
) -> Optional[ThreadingHTTPServer]:
    """启动交互式 Web HUD 控制台服务器。"""
    session = FutureWorkSession(workdir=workdir)
    tracker = RealCameraTracker(device_index=camera_index)
    if hasattr(sys, "_MEIPASS"):
        static_dir = os.path.join(sys._MEIPASS, "futurework", "web", "static")
    else:
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    class BoundHandler(FutureWorkWebHandler):
        pass

    BoundHandler.session = session
    BoundHandler.camera_tracker = tracker
    BoundHandler.static_dir = static_dir

    server = None
    target_port = port
    try:
        server = ReusableThreadingHTTPServer((host, target_port), BoundHandler)
    except OSError as err:
        display_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
        # 1. 检查是否已经是正在运行中的 FutureWork 实例
        if _probe_futurework_status(host, target_port):
            print(f"✓ 检测到 FutureWork Web HUD 已经在运行中：http://{display_host}:{target_port}")
            if auto_open:
                try:
                    import webbrowser
                    webbrowser.open(f"http://{display_host}:{target_port}")
                except Exception:
                    pass
            print(f"提示：浏览器控制台已打开。如需重启或新建实例，请指定新端口（例如 --port {target_port + 1}）或先关闭已有进程。")
            return None

        # 2. 端口被其他非 FutureWork 程序占用，尝试自动顺延寻找空闲端口 (target_port + 1 ~ target_port + 20)
        for candidate in range(target_port + 1, target_port + 21):
            try:
                server = ReusableThreadingHTTPServer((host, candidate), BoundHandler)
                print(f"提示：端口 {target_port} 已被占用，已自动切换至空闲端口：http://{display_host}:{candidate}")
                target_port = candidate
                break
            except OSError:
                continue

        if server is None:
            print(f"错误：无法在端口 {target_port} 到 {target_port + 20} 之间绑定服务：{err}")
            return None

    display_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    if auto_open:
        def _open_browser() -> None:
            time.sleep(0.6)
            try:
                import webbrowser
                webbrowser.open(f"http://{display_host}:{target_port}")
            except Exception:
                pass
        threading.Thread(target=_open_browser, daemon=True).start()

    if blocking:
        print(f"FutureWork Web HUD 已启动：http://{display_host}:{target_port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n正在停止 FutureWork Web HUD 服务...")
            server.shutdown()
        return server
    else:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server
