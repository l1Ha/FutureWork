"""
FutureWork — 面向未来人机自然交互的全平台生产力工具对接核心

设计哲学
--------
人类之间的沟通是"多模态 + 意图 + 情境"的：说话的同时手在指、脸在示意、
视线落在屏幕上某处。FutureWork 把这套交互范式抽象成一个可复用的软件接口：

    感知层 (Sensory)  →  认知层 (Cognition)  →  执行层 (Adapters)
         ↑                      ↓                      ↑
     多模态信号          意图 + 消歧 + 对话状态       全平台工具

任何硬件（麦克风、摄像头、眼动仪）只要能把信号转成 ``types.py`` 中的模型，
就能接入 FutureWork；任何工具（IDE、浏览器、Office、终端）只要实现
``ToolAdapter`` 协议，就能被自然语言指挥。
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "types",
    "sensory",
    "cognition",
    "tools",
    "resilience",
    "runtime",
]

from futurework import types  # noqa: F401