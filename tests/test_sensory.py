"""感知层测试：语音、手势、表情、视线、融合。"""

from __future__ import annotations

import pytest

from futurework.sensory.facial import (
    FacialExpressionAnalyzer,
    HeadPoseAnalyzer,
    Landmark3D,
    detect_emotion_sequence,
)
from futurework.sensory.gaze import GazeTracker, UIElement, head_direction_to_screen
from futurework.sensory.gesture import GestureInterpreter, Landmark, PointingResolver
from futurework.sensory.speech import SpeechRecognizer, SimulatedMicrophone
from futurework.sensory.fusion import MultimodalFusionEngine, WorkingMemory
from futurework.types import (
    EmotionType,
    ModalityType,
    FacialSignal,
    GazeSignal,
    GestureType,
    HandGestureSignal,
    HeadPoseSignal,
    SpeechSignal,
)


# ============================================================================
# 语音
# ============================================================================
class TestSpeech:
    def test_removes_fillers(self):
        mic = SimulatedMicrophone()
        signal = mic.push("嗯那个打开浏览器吧")
        assert "嗯" not in signal.transcript
        assert "那个" not in signal.transcript
        assert "打开浏览器吧" in signal.transcript

    def test_prosody_detects_emphasis(self):
        mic = SimulatedMicrophone()
        calm = mic.push("打开文件")
        excited = mic.push("快点打开文件！")
        assert excited.energy > calm.energy
        assert excited.pitch_hz > calm.pitch_hz

    def test_noise_reduces_confidence(self):
        quiet = SimulatedMicrophone(noise_level=0.0).push("测试")
        noisy = SimulatedMicrophone(noise_level=0.8).push("测试")
        assert noisy.confidence < quiet.confidence

    def test_recognizer_dispatches_to_listeners(self):
        rec = SpeechRecognizer()
        seen = []
        rec.on_speech(seen.append)
        rec.submit("列出目录")
        assert len(seen) == 1
        assert seen[0].transcript == "列出目录"

    def test_listener_exception_does_not_break_chain(self):
        rec = SpeechRecognizer()
        good = []
        rec.on_speech(lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
        rec.on_speech(good.append)
        rec.submit("测试")
        assert len(good) == 1


# ============================================================================
# 手势
# ============================================================================
def _hand(extended: list[bool]) -> list[Landmark]:
    """
    构造符合 MediaPipe Hands 拓扑的 21 个关键点。

    索引：0=腕点，1-4=拇指，5-8=食指，9-12=中指，13-16=无名指，17-20=小指；
    每根手指依次为 MCP / PIP / DIP / TIP。
    ``extended[i]`` 决定第 i 根手指是否伸展。
    """
    tips_y = [0.25 if flag else 0.62 for flag in extended]  # 伸展=指尖远离掌心(y 更小)
    points: list[Landmark] = [Landmark(0.5, 0.90, 0.0)]  # 0 wrist

    for finger in range(5):
        x = 0.38 + 0.06 * finger
        mcp = Landmark(x, 0.70, 0.0)
        tip_y = tips_y[finger]
        pip = Landmark(x, (tip_y + 0.70) / 2 if extended[finger] else 0.66, 0.0)
        dip = Landmark(x, tip_y + (0.10 if extended[finger] else 0.02), 0.0)
        tip = Landmark(x, tip_y, 0.0)
        points.extend([mcp, pip, dip, tip])

    assert len(points) == 21
    return points


class TestGesture:
    def test_open_palm(self):
        interp = GestureInterpreter()
        signal = interp.classify(_hand([True, True, True, True, True]))
        assert signal.gesture is GestureType.OPEN_PALM

    def test_point(self):
        interp = GestureInterpreter()
        signal = interp.classify(_hand([False, True, False, False, False]))
        assert signal.gesture is GestureType.POINT
        assert signal.point_direction is not None

    def test_fist(self):
        interp = GestureInterpreter()
        signal = interp.classify(_hand([False, False, False, False, False]))
        assert signal.gesture is GestureType.FIST

    def test_insufficient_landmarks_raises(self):
        interp = GestureInterpreter()
        with pytest.raises(ValueError):
            interp.classify([Landmark(0.5, 0.5)])

    def test_pinched_fingers_detect_pinch(self):
        interp = GestureInterpreter()
        points = _hand([False, True, True, True, True])
        points[4] = Landmark(0.38, 0.50)   # 拇指尖移到食指尖旁
        points[8] = Landmark(0.39, 0.50)   # 食指尖
        signal = interp.classify(points)
        assert signal.gesture is GestureType.PINCH

    def test_pointing_resolver_combines_gaze(self):
        resolver = PointingResolver()
        resolver.feed(HandGestureSignal(gesture=GestureType.POINT, position_3d=(0.2, 0.2, 0.5)))
        target = resolver.resolve(screen_w=1920, screen_h=1080, gaze=(0.8, 0.6))
        assert target is not None
        assert 0.2 < target.bounding_box[0] < 0.8   # 视线权重应把结果拉向 0.8

    def test_resolver_returns_none_without_pointing(self):
        resolver = PointingResolver()
        assert resolver.resolve(screen_w=1920, screen_h=1080) is None


# ============================================================================
# 面部表情与头部姿态
# ============================================================================
class TestFacial:
    def test_detects_confusion_from_brow_au(self):
        analyzer = FacialExpressionAnalyzer()
        signal = analyzer.analyze({"AU4": 0.9, "AU9": 0.8})
        assert signal.primary_emotion is EmotionType.CONFUSED
        assert signal.confidence > 0.5

    def test_neutral_when_no_activity(self):
        analyzer = FacialExpressionAnalyzer()
        signal = analyzer.analyze({})
        assert signal.primary_emotion is EmotionType.NEUTRAL

    def test_missing_face_is_neutral(self):
        analyzer = FacialExpressionAnalyzer()
        assert analyzer.analyze({"AU4": 0.9}, face_detected=False).primary_emotion is EmotionType.NEUTRAL

    def test_blink_ear_closed_vs_open(self):
        analyzer = FacialExpressionAnalyzer()

        def eye(half_gap: float):
            """
            标准 EAR 六点（占据索引 1..6，索引 0 保留不用）：
            p1/p4 为内外眼角，p2·p6 与 p3·p5 为上下眼睑。
            """
            return [
                Landmark3D(0.00, 0.0),   # 索引 0（未使用）
                Landmark3D(0.00, 0.0),   # p1 眼角
                Landmark3D(0.01, half_gap),  # p2 上眼睑
                Landmark3D(0.03, half_gap),  # p3 上眼睑
                Landmark3D(0.04, 0.0),   # p4 眼角
                Landmark3D(0.03, -half_gap), # p5 下眼睑
                Landmark3D(0.01, -half_gap), # p6 下眼睑
            ]

        assert analyzer.classify_blink(eye(0.05)) is False   # 睁眼 EAR≈2.5
        assert analyzer.classify_blink(eye(0.002)) is True   # 闭眼 EAR≈0.1

    def test_attention_decays_on_frustration(self):
        analyzer = FacialExpressionAnalyzer()
        analyzer.analyze({"AU4": 0.9})
        analyzer.analyze({"AU4": 0.9})
        signal = analyzer.analyze({"AU4": 0.9})
        assert signal.attention_score < 0.95

    def test_emotion_sequence_needs_majority(self):
        frames = [FacialSignal(primary_emotion=EmotionType.CONFUSED) for _ in range(5)]
        frames.append(FacialSignal(primary_emotion=EmotionType.NEUTRAL))
        assert detect_emotion_sequence(frames) is EmotionType.CONFUSED


class TestHeadPose:
    def test_nod_requires_sweep(self):
        analyzer = HeadPoseAnalyzer()
        for pitch in (0.0, 15.0, 0.0, 15.0, 0.0):
            signal = analyzer.update(pitch=pitch, yaw=0.0)
        assert signal.nod is True

    def test_single_frame_not_nod(self):
        analyzer = HeadPoseAnalyzer()
        assert analyzer.update(pitch=20.0, yaw=0.0).nod is False

    def test_explicit_nod_override(self):
        analyzer = HeadPoseAnalyzer()
        assert analyzer.update(pitch=0.0, yaw=0.0, nod=True).nod is True

    def test_shake_detection(self):
        analyzer = HeadPoseAnalyzer()
        for yaw in (0.0, 20.0, -20.0, 0.0):
            signal = analyzer.update(pitch=0.0, yaw=yaw)
        assert signal.shake is True


# ============================================================================
# 视线
# ============================================================================
class TestGaze:
    def test_smooths_jitter(self):
        tracker = GazeTracker(smoothing=0.5)
        tracker.update(0.0, 0.0)
        signal = tracker.update(1.0, 1.0)
        assert 0.0 < signal.screen_x < 1.0

    def test_dwell_accumulates_when_still(self):
        tracker = GazeTracker()
        tracker.update(0.5, 0.5, timestamp=1000.0)
        signal = tracker.update(0.5, 0.5, timestamp=1001.0)
        assert signal.fixation_duration_ms >= 1000

    def test_dwell_resets_on_movement(self):
        tracker = GazeTracker()
        tracker.update(0.5, 0.5, timestamp=1000.0)
        tracker.update(0.9, 0.9, timestamp=1001.0)
        signal = tracker.update(0.9, 0.9, timestamp=1002.0)
        assert signal.fixation_duration_ms < 1000

    def test_element_hit_test(self):
        tracker = GazeTracker()
        tracker.register_element(UIElement("btn_ok", "确定", (0.4, 0.4, 0.1, 0.1), "设置"))
        signal = tracker.update(0.45, 0.45)
        assert signal.target_element_id == "btn_ok"
        assert tracker.to_target(signal).target_type == "element"

    def test_miss_returns_no_element(self):
        tracker = GazeTracker()
        tracker.register_element(UIElement("btn", "按钮", (0.0, 0.0, 0.1, 0.1)))
        assert tracker.update(0.9, 0.9).target_element_id is None

    def test_head_direction_fallback(self):
        x, y = head_direction_to_screen(yaw=15.0, pitch=0.0)
        assert 0.5 < x <= 1.0          # 头右转 → 视点右移
        assert y == pytest.approx(0.5)  # 无俯仰时垂直方向居中
        _, up = head_direction_to_screen(yaw=0.0, pitch=10.0)
        assert up < 0.5                 # 抬头 → 视点上移


# ============================================================================
# 多模态融合
# ============================================================================
class TestFusion:
    def test_speech_only_gives_reasonable_confidence(self):
        engine = MultimodalFusionEngine()
        frame = engine.assemble(speech=SpeechSignal(transcript="打开浏览器"))
        fused = engine.fuse(frame)
        assert 0.5 < fused.confidence <= 1.0

    def test_nod_agreement_raises_confidence(self):
        engine = MultimodalFusionEngine()
        alone = engine.fuse(engine.assemble(speech=SpeechSignal(transcript="确认")))
        engine2 = MultimodalFusionEngine()
        with_nod = engine2.fuse(engine2.assemble(
            speech=SpeechSignal(transcript="确认"),
            head_pose=HeadPoseSignal(nod=True),
        ))
        assert with_nod.confidence > alone.confidence
        assert ModalityType.HEAD_POSE in with_nod.agreeing_modalities

    def test_nod_vs_deny_is_a_conflict(self):
        engine = MultimodalFusionEngine()
        fused = engine.fuse(engine.assemble(
            speech=SpeechSignal(transcript="不要"),
            head_pose=HeadPoseSignal(nod=True),
        ))
        assert fused.conflicting_modalities
        assert fused.requires_confirmation is True

    def test_open_palm_is_a_stop_signal(self):
        engine = MultimodalFusionEngine()
        fused = engine.fuse(engine.assemble(
            speech=SpeechSignal(transcript="继续运行"),
            gestures=[HandGestureSignal(gesture=GestureType.PALM_STOP, confidence=0.98)],
        ))
        assert any("停止" in r for r in fused.rationale)

    def test_confusion_raises_ambiguity(self):
        engine = MultimodalFusionEngine()
        fused = engine.fuse(engine.assemble(
            speech=SpeechSignal(transcript="把这个弄一下"),
            facial=FacialSignal(primary_emotion=EmotionType.CONFUSED),
        ))
        assert fused.ambiguity_score >= 0.2

    def test_gaze_and_gesture_fuse_into_one_anchor(self):
        engine = MultimodalFusionEngine()
        fused = engine.fuse(engine.assemble(
            speech=SpeechSignal(transcript="移动这个"),
            gestures=[HandGestureSignal(gesture=GestureType.POINT, position_3d=(0.2, 0.2, 0.5))],
            gaze=GazeSignal(screen_x=0.8, screen_y=0.7),
        ))
        assert fused.target is not None
        assert "gaze" in fused.target.metadata["source"]

    def test_low_attention_triggers_question(self):
        engine = MultimodalFusionEngine()
        fused = engine.fuse(engine.assemble(
            speech=SpeechSignal(transcript="打开浏览器"),
            facial=FacialSignal(primary_emotion=EmotionType.TIRED, attention_score=0.1),
        ))
        assert fused.requires_confirmation is True

    def test_empty_frame_is_low_confidence(self):
        engine = MultimodalFusionEngine()
        fused = engine.fuse(engine.assemble())
        assert fused.confidence == 0.0
        assert fused.ambiguity_score > 0.0


class TestWorkingMemory:
    def test_capacity_bound(self):
        memory = WorkingMemory(capacity=3)
        for i in range(10):
            memory.remember(f"k{i}", 0.5)
        assert len(memory.recall(limit=10)) <= 3 * 3

    def test_prefix_recall(self):
        memory = WorkingMemory()
        memory.remember("speech:打开", 0.9)
        memory.remember("target:x", 0.9)
        assert memory.recall("speech:") == ["speech:打开"]

    def test_expired_entries_are_pruned(self):
        memory = WorkingMemory(retention=0.01)
        memory.remember("k", 0.5)
        import time as _t
        _t.sleep(0.05)
        assert memory.recall() == []