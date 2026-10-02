"""
多模态感知层：将人类信号（语音、手势、面部表情、视线、头部姿态）转为结构化信号。

每个感知模块都提供两条路径：
1. ``process_*`` —— 纯函数式信号处理，不依赖硬件，可离线测试与回放；
2. ``listen``     —— 连接真实硬件（麦克风/摄像头/眼动仪）的采集循环，
   硬件缺失时自动降级为 ``SimulatedSensor``，保证全平台可用。
"""

from __future__ import annotations

from futurework.sensory.speech import SpeechRecognizer, SimulatedMicrophone
from futurework.sensory.gesture import (
    GestureInterpreter,
    SimulatedHandTracker,
    PointingResolver,
)
from futurework.sensory.facial import (
    FacialExpressionAnalyzer,
    HeadPoseAnalyzer,
    SimulatedCamera,
)
from futurework.sensory.gaze import GazeTracker, SimulatedEyeTracker
from futurework.sensory.fusion import MultimodalFusionEngine, WorkingMemory

__all__ = [
    "SpeechRecognizer",
    "SimulatedMicrophone",
    "GestureInterpreter",
    "SimulatedHandTracker",
    "PointingResolver",
    "FacialExpressionAnalyzer",
    "HeadPoseAnalyzer",
    "SimulatedCamera",
    "GazeTracker",
    "SimulatedEyeTracker",
    "MultimodalFusionEngine",
    "WorkingMemory",
]