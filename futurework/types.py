"""
FutureWork Core Type Definitions and Data Models
Unified data representations for multimodal perception, cognitive intents,
tool commands, execution results, safety classification, and feedback.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ModalityType(str, Enum):
    """Sensory input modality types."""
    SPEECH = "speech"
    TEXT = "text"
    GESTURE = "gesture"
    FACIAL = "facial"
    GAZE = "gaze"
    HEAD_POSE = "head_pose"
    MULTIMODAL_FUSION = "multimodal_fusion"


class GestureType(str, Enum):
    """Recognized natural human hand gestures."""
    IDLE = "idle"
    POINT = "point"
    PINCH = "pinch"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PALM_STOP = "palm_stop"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    WAVE = "wave"
    TWO_FINGER_SCROLL = "two_finger_scroll"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


class EmotionType(str, Enum):
    """Facial emotional & cognitive state indicators."""
    NEUTRAL = "neutral"
    FOCUSED = "focused"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    SATISFIED = "satisfied"
    SURPRISED = "surprised"
    TIRED = "tired"


class SafetyLevel(int, Enum):
    """Risk stratification for tool operations."""
    READ_ONLY = 0           # Auto execute without prompt (e.g. read file, get status)
    SAFE_WRITE = 1          # Auto execute with automatic undo journal (e.g. append text)
    SENSITIVE_MODIFY = 2    # Light confirmation required (e.g. nod or high confidence)
    CRITICAL_DESTRUCTIVE = 3 # Strict confirmation required (e.g. explicit vocal 'yes' + nod)


class ExecutionStatus(str, Enum):
    """Tool command execution lifecycle states."""
    PENDING = "pending"
    PENDING_CONFIRMATION = "pending_confirmation"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED_BY_SAFETY = "rejected_by_safety"
    ROLLED_BACK = "rolled_back"


class IntentCategory(str, Enum):
    """Semantic domain categories for human intentions."""
    NAVIGATION = "navigation"
    WINDOW_MANAGEMENT = "window_management"
    DOCUMENT_EDIT = "document_edit"
    CODE_DEV = "code_dev"
    BROWSER_ACTION = "browser_action"
    SYSTEM_CONTROL = "system_control"
    CONFIRMATION = "confirmation"
    REJECTION = "rejection"
    CANCEL_UNDO = "cancel_undo"
    QUERY_EXPLAIN = "query_explain"
    UNKNOWN = "unknown"


# ============================================================================
# Sensory Perception Models
# ============================================================================

class SpeechSignal(BaseModel):
    """Acoustic speech / spoken natural language perception."""
    transcript: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_final: bool = True
    language: str = "zh-CN"
    pitch_hz: float = 180.0
    energy: float = 0.8
    speaking_rate: float = 1.0  # relative speed multiplier
    timestamp: float = Field(default_factory=time.time)


class HandGestureSignal(BaseModel):
    """Hand gesture perception with 3D spatial coordinate tracking."""
    gesture: GestureType = GestureType.IDLE
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    hand: str = "right"  # "left" or "right"
    position_3d: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # Normalized (x, y, z)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pinch_distance: Optional[float] = None
    point_direction: Optional[Tuple[float, float, float]] = None
    timestamp: float = Field(default_factory=time.time)


class FacialSignal(BaseModel):
    """Facial expression, action units and cognitive state."""
    primary_emotion: EmotionType = EmotionType.NEUTRAL
    emotion_scores: Dict[str, float] = Field(default_factory=dict)
    action_units: Dict[str, float] = Field(default_factory=dict)  # AU4, AU12, etc.
    eye_blink: bool = False
    attention_score: float = Field(default=0.9, ge=0.0, le=1.0)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    timestamp: float = Field(default_factory=time.time)


class HeadPoseSignal(BaseModel):
    """3D head pose and discrete head gestures (nod, shake)."""
    pitch: float = 0.0  # Up / down (nod axis)
    yaw: float = 0.0    # Left / right (shake axis)
    roll: float = 0.0   # Tilt axis
    nod: bool = False   # True if affirmative nod detected
    shake: bool = False # True if negative shake detected
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    timestamp: float = Field(default_factory=time.time)


class GazeSignal(BaseModel):
    """Eye tracking, screen fixation coordinates and visual focus."""
    screen_x: float = 0.5  # Normalized [0, 1] across primary display
    screen_y: float = 0.5  # Normalized [0, 1] across primary display
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    fixation_duration_ms: float = 250.0
    target_window_title: Optional[str] = None
    target_element_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class MultimodalFrame(BaseModel):
    """Synchronized multimodal sensory snapshot across all perception channels."""
    frame_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    speech: Optional[SpeechSignal] = None
    gestures: List[HandGestureSignal] = Field(default_factory=list)
    facial: Optional[FacialSignal] = None
    head_pose: Optional[HeadPoseSignal] = None
    gaze: Optional[GazeSignal] = None
    environment_noise_level: float = 0.1
    camera_active: bool = True
    microphone_active: bool = True


# ============================================================================
# Cognitive Grounding and Intent Models
# ============================================================================

class GroundingTarget(BaseModel):
    """Physical or virtual workstation target grounded by multimodal cues."""
    target_type: str = "window"  # "window", "element", "file", "text", "coordinate"
    target_id: str = ""
    label: str = ""
    bounding_box: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Intent(BaseModel):
    """Resolved human intent with semantic slots and execution parameters."""
    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: IntentCategory = IntentCategory.UNKNOWN
    action: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    source_modalities: List[ModalityType] = Field(default_factory=list)
    grounded_target: Optional[GroundingTarget] = None
    ambiguity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_confirmation: bool = False
    natural_explanation: str = ""
    timestamp: float = Field(default_factory=time.time)


# ============================================================================
# Universal Tool Commands and Execution Results
# ============================================================================

class ToolCommand(BaseModel):
    """Standardized tool invocation payload dispatched to platform adapters."""
    command_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool_name: str
    action: str
    args: Dict[str, Any] = Field(default_factory=dict)
    safety_level: SafetyLevel = SafetyLevel.READ_ONLY
    reversible: bool = False
    inverse_action: Optional[Dict[str, Any]] = None  # Used for undo
    timeout_seconds: float = 10.0
    created_at: float = Field(default_factory=time.time)


class ExecutionResult(BaseModel):
    """Result returned by a tool adapter after command execution."""
    command_id: str
    status: ExecutionStatus
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    rollback_available: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Feedback and System State Models
# ============================================================================

class SystemFeedback(BaseModel):
    """Natural feedback returned to the human across multimodal channels."""
    text: str
    speech_audio_text: Optional[str] = None
    visual_alert_level: str = "info"  # "info", "success", "warning", "error"
    haptic_pattern: Optional[str] = None
    sound_cue: Optional[str] = None
    screen_highlight_box: Optional[Tuple[float, float, float, float]] = None
    timestamp: float = Field(default_factory=time.time)
