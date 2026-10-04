"""
高级特性单元与集成测试套件：
1. 条件控制流执行 (Conditional Flows)
2. 持久化工作记忆与跨会话指代消解 (Persistent Memory)
3. 办公与生产力套件适配器 (Productivity Suite Adapter)
4. 流式感知与时间窗口高斯对齐 (Temporal Alignment)
5. 交互式 Web HUD 控制台服务器 (Web HUD Server)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import pytest

from futurework.cognition.compound import ConditionalFlow, parse_conditional
from futurework.cognition.memory import MemoryEntity, PersistentMemoryStore
from futurework.runtime.session import FutureWorkSession, TextSession
from futurework.sensory.temporal_aligner import StreamingPerceptionLoop, TemporalAlignmentBuffer
from futurework.tools.productivity_adapter import ProductivitySuiteAdapter
from futurework.types import (
    EmotionType,
    ExecutionStatus,
    GazeSignal,
    GestureType,
    GroundingTarget,
    HandGestureSignal,
    SpeechSignal,
    ToolCommand,
)
from futurework.web.server import run_web_hud


# ============================================================================
# 1. 条件控制流测试
# ============================================================================
class TestConditionalFlow:
    @pytest.mark.parametrize("text,expected_initial,expected_then,expected_else", [
        (
            "如果跑测试成功了，就提交代码 feat: done，否则把失败信息写入 test_err.log",
            "跑测试", "提交代码 feat: done", "把失败信息写入 test_err.log"
        ),
        (
            "如果读取 missing.txt 失败了就创建目录 backup",
            "读取 missing.txt", "创建目录 backup", None
        ),
        (
            "if run tests then commit feat: ok else read test.log",
            "run tests", "commit feat: ok", "read test.log"
        ),
    ])
    def test_parse_conditional_patterns(self, text, expected_initial, expected_then, expected_else):
        flow = parse_conditional(text)
        assert flow is not None
        assert expected_initial in flow.initial_action
        assert expected_then in flow.then_branch
        if expected_else:
            assert flow.else_branch is not None and expected_else in flow.else_branch

    def test_conditional_success_triggers_then_branch(self, tmp_path):
        (tmp_path / "exist.txt").write_text("content")
        session = TextSession(workdir=str(tmp_path))
        turn = session.run("如果读取 exist.txt 成功了，就把 ok 写入 res.txt 否则把 err 写入 res.txt")
        assert turn.status is ExecutionStatus.SUCCESS
        assert (tmp_path / "res.txt").read_text() == "ok"

    def test_conditional_failure_triggers_else_branch(self, tmp_path):
        session = TextSession(workdir=str(tmp_path))
        turn = session.run("如果读取 nonexistent.txt 成功了，就把 ok 写入 res2.txt 否则把 fallback 写入 res2.txt")
        assert turn.status is ExecutionStatus.SUCCESS
        assert (tmp_path / "res2.txt").read_text() == "fallback"

    def test_conditional_failure_only_branch(self, tmp_path):
        session = TextSession(workdir=str(tmp_path))
        turn = session.run("如果读取 nonexistent.txt 失败了，就把 recovered 写入 res3.txt")
        assert turn.status is ExecutionStatus.SUCCESS
        assert (tmp_path / "res3.txt").read_text() == "recovered"


# ============================================================================
# 2. 持久化记忆与跨会话消歧测试
# ============================================================================
class TestPersistentMemory:
    def test_save_and_reload(self, tmp_path):
        mem_file = str(tmp_path / "memory.json")
        store1 = PersistentMemoryStore(storage_path=mem_file)
        target = GroundingTarget(target_type="file", target_id="/tmp/doc.txt", label="重要文档")
        store1.record_target(target, context_tags=["doc", "finance"])
        store1.flush()   # 延迟写：需显式收尾，否则磁盘上还没有数据

        # 重新从磁盘加载到另一个对象
        store2 = PersistentMemoryStore(storage_path=mem_file)
        recent = store2.query_recent(limit=5)
        assert len(recent) == 1
        assert recent[0].entity_id == "/tmp/doc.txt"
        assert recent[0].label == "重要文档"

    def test_memory_decay_calculation(self):
        entity = MemoryEntity(entity_id="x", target_type="file", label="x", last_accessed=1000.0)
        score_fresh = entity.current_score(now=1000.0)
        # 经过 24 小时后，衰减一半左右
        score_later = entity.current_score(now=1000.0 + 86400.0)
        assert score_later < score_fresh

    def test_cross_session_deictic_resolution(self, tmp_path):
        (tmp_path / "plan.md").write_text("# Project Plan")
        
        # Session 1: 交互
        s1 = TextSession(workdir=str(tmp_path))
        r1 = s1.run("读取 plan.md")
        assert r1.status is ExecutionStatus.SUCCESS
        del s1

        # Session 2: 全新会话对象（模拟重启）
        s2 = TextSession(workdir=str(tmp_path))
        r2 = s2.run("把 刚才那个文件 复制到 backup.md")
        assert r2.status is ExecutionStatus.SUCCESS
        assert (tmp_path / "backup.md").read_text() == "# Project Plan"


# ============================================================================
# 3. 办公与生产力套件测试
# ============================================================================
class TestProductivitySuite:
    @pytest.fixture
    def prod(self, tmp_path):
        return ProductivitySuiteAdapter(str(tmp_path))

    def test_csv_aggregation(self, prod, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("item,cost\nA,10\nB,25.5\nC,14.5\n")
        cmd = ToolCommand(
            tool_name="productivity",
            action="table_aggregate",
            args={"path": "data.csv", "column": "cost", "operation": "sum"},
        )
        res = prod.execute(cmd)
        assert res.status is ExecutionStatus.SUCCESS
        assert res.data["result"] == 50.0

    def test_csv_mean_and_count(self, prod, tmp_path):
        csv_file = tmp_path / "data2.csv"
        csv_file.write_text("score\n80\n90\n100\n")
        res = prod.execute(ToolCommand(
            tool_name="productivity", action="table_aggregate",
            args={"path": "data2.csv", "column": "score", "operation": "mean"}
        ))
        assert res.data["result"] == 90.0

    def test_generate_report_and_extract_outline(self, prod, tmp_path):
        cmd = ToolCommand(
            tool_name="productivity",
            action="generate_report",
            args={"title": "Q3 业务报告", "output_path": "q3.md"},
        )
        res = prod.execute(cmd)
        assert res.status is ExecutionStatus.SUCCESS
        assert (tmp_path / "q3.md").exists()

        out_cmd = ToolCommand(
            tool_name="productivity",
            action="extract_outline",
            args={"path": "q3.md"},
        )
        res_out = prod.execute(out_cmd)
        assert res_out.status is ExecutionStatus.SUCCESS
        assert any(item["title"] == "Q3 业务报告" for item in res_out.data)

    def test_todo_lifecycle(self, prod):
        # 1. 添加待办
        r_add = prod.execute(ToolCommand(
            tool_name="productivity", action="todo_add", args={"task": "编写安全测试报告"}
        ))
        assert r_add.status is ExecutionStatus.SUCCESS
        task_id = r_add.data["id"]

        # 2. 查询列表
        r_list = prod.execute(ToolCommand(tool_name="productivity", action="todo_list", args={}))
        assert len(r_list.data) == 1
        assert not r_list.data[0]["completed"]

        # 3. 完成待办
        r_done = prod.execute(ToolCommand(
            tool_name="productivity", action="todo_complete", args={"task_id_or_keyword": "安全测试"}
        ))
        assert r_done.status is ExecutionStatus.SUCCESS
        assert r_done.data["completed"] is True


# ============================================================================
# 4. 流式感知与时间对齐测试
# ============================================================================
class TestTemporalAlignment:
    def test_temporal_buffer_aligns_speech_and_gesture(self):
        buf = TemporalAlignmentBuffer()
        t0 = 1000.0

        # 手势在 t0 - 100ms
        g = HandGestureSignal(gesture=GestureType.POINT, timestamp=t0 - 0.1)
        buf.push_gestures([g], ts=t0 - 0.1)

        # 视线在 t0 - 50ms
        gaze = GazeSignal(screen_x=0.8, screen_y=0.2, timestamp=t0 - 0.05)
        buf.push_gaze(gaze, ts=t0 - 0.05)

        # 语音在 t0 断句
        speech = SpeechSignal(transcript="打开这个", timestamp=t0)
        buf.push_speech(speech, ts=t0)

        frame = buf.align_frame(anchor_timestamp=t0)
        assert frame.speech.transcript == "打开这个"
        assert len(frame.gestures) == 1
        assert frame.gaze.screen_x == 0.8

    def test_streaming_loop_triggers_on_final_speech(self):
        triggered = []
        loop = StreamingPerceptionLoop(on_aligned_frame=triggered.append)

        # 非 final 语音不触发
        loop.feed_speech(SpeechSignal(transcript="正在说", is_final=False))
        assert len(triggered) == 0

        # final 语音触发
        loop.feed_speech(SpeechSignal(transcript="说完了", is_final=True))
        assert len(triggered) == 1


# ============================================================================
# 5. 可视化 Web HUD 服务测试
# ============================================================================
class TestWebHudServer:
    def test_hud_page_and_api(self, tmp_path):
        port = 19876
        server = run_web_hud(host="127.0.0.1", port=port, workdir=str(tmp_path), blocking=False)
        time.sleep(0.3)

        try:
            # 1. 测试静态主页返回
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
                assert resp.status == 200
                html = resp.read().decode("utf-8")
                assert "FutureWork" in html
                assert "gazeCanvas" in html

            # 2. 测试 POST /api/interact
            payload = json.dumps({"text": "列出目录 ."}).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/interact",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["status"] == "success"
                assert "条目" in data["feedback"]

            # 3. 测试 GET /api/vision/status
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/vision/status") as resp:
                assert resp.status == 200
                v_data = json.loads(resp.read().decode("utf-8"))
                assert "camera_available" in v_data
        finally:
            server.shutdown()


# ============================================================================
# 6. 真实摄像头感知与视觉追踪测试
# ============================================================================
class TestCameraVisionTracking:
    def test_real_camera_tracker_processing(self):
        import cv2
        import numpy as np
        from futurework.sensory.camera_tracker import RealCameraTracker

        tracker = RealCameraTracker()
        
        # 构造带有标准几何特征的测试画面 (人脸 + 双眼 + 微笑嘴形)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.ellipse(img, (320, 240), (120, 160), 0, 0, 360, (200, 200, 200), -1)
        cv2.circle(img, (270, 200), 20, (50, 50, 50), -1)
        cv2.circle(img, (370, 200), 20, (50, 50, 50), -1)
        cv2.ellipse(img, (320, 300), (40, 20), 0, 0, 180, (50, 50, 50), -1)

        result = tracker.process_frame(img)
        assert result.face_detected is True
        assert result.face_box is not None
        assert result.facial_signal is not None
        assert result.head_pose_signal is not None
        assert result.gaze_signal is not None
        assert 0.0 <= result.gaze_signal.screen_x <= 1.0

        # 测试字典序列化
        res_dict = result.to_dict()
        assert res_dict["face_detected"] is True
        assert "primary_emotion" in res_dict
        assert "gaze_x" in res_dict

    def test_camera_process_jpeg_bytes(self):
        import cv2
        import numpy as np
        from futurework.sensory.camera_tracker import RealCameraTracker

        tracker = RealCameraTracker()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.ellipse(img, (320, 240), (120, 160), 0, 0, 360, (200, 200, 200), -1)
        cv2.circle(img, (270, 200), 20, (50, 50, 50), -1)
        cv2.circle(img, (370, 200), 20, (50, 50, 50), -1)

        _, jpg = cv2.imencode(".jpg", img)
        result = tracker.process_image_bytes(jpg.tobytes())
        assert result.face_detected is True

