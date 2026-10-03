"""
FutureWork 可视化 HUD 与交互式 Web 控制台服务端 (Web HUD & Interaction Server)

基于 Python 原生轻量 HTTP 服务，零重型第三方依赖：
- 提供面向未来工作站的多模态状态监视器 (Sensory HUD)
- 实时视线 (Gaze)、手势 (Gesture)、面部表情 (Facial)、工作记忆 (Memory) 联动控制台
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from futurework.runtime.session import FutureWorkSession
from futurework.types import GestureType


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
) -> ThreadingHTTPServer:
    """启动交互式 Web HUD 控制台服务器。"""
    session = FutureWorkSession(workdir=workdir)
    if hasattr(sys, "_MEIPASS"):
        static_dir = os.path.join(sys._MEIPASS, "futurework", "web", "static")
    else:
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    class BoundHandler(FutureWorkWebHandler):
        pass

    BoundHandler.session = session
    BoundHandler.static_dir = static_dir

    server = ThreadingHTTPServer((host, port), BoundHandler)

    if auto_open:
        def _open_browser() -> None:
            time.sleep(0.6)
            try:
                import webbrowser
                webbrowser.open(f"http://{host}:{port}")
            except Exception:
                pass
        threading.Thread(target=_open_browser, daemon=True).start()

    if blocking:
        print(f"FutureWork Web HUD 已启动：http://{host}:{port}")
        server.serve_forever()
    else:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server
