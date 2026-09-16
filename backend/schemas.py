"""
Pydantic schemas for request/response validation.
"""
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field


# ---- Recognize ----
class RecognizeRequest(BaseModel):
    session_id: str
    mode: Literal["static", "temporal", "letters", "gesture_asl", "digits", "words"] = "static"
    landmarks: List[float]  # flat array: 63 for static/letters, or seq_len*63 for temporal/gesture_asl
    timestamp_ms: int = 0


class TopKPrediction(BaseModel):
    label: str
    prob: float


class RawPrediction(BaseModel):
    label: str
    confidence: float
    top_k: List[TopKPrediction]


class StablePrediction(BaseModel):
    emitted: bool
    label: Optional[str] = None
    confidence: Optional[float] = None


class DebugState(BaseModel):
    window: List[Any]
    agreement: float
    cooldown_remaining_ms: int
    modal_label: str = ""
    held_label: Optional[str] = None
    consecutive_unknown: int = 0
    buffer_size: int = 0


class RecognizeResponse(BaseModel):
    raw: RawPrediction
    stable: StablePrediction
    status: str
    message: str
    debug: DebugState
    head: Optional[str] = None       # digits | words | letters | gesture_asl
    language: Optional[str] = None   # asl | isl


# ---- Text-to-Sign ----
class TextToSignRequest(BaseModel):
    text: str


class SignToken(BaseModel):
    token: str
    available: bool
    asset: Optional[str] = None
    kind: Optional[str] = None  # word, digit
    reason: Optional[str] = None


class TextToSignResponse(BaseModel):
    tokens: List[SignToken]
    coverage: float
    notice: str


# ---- Messages ----
class MessageRequest(BaseModel):
    session_id: str
    sender: Literal["staff", "citizen"]
    text: str
    confidence: Optional[float] = None


class Message(BaseModel):
    sender: str
    text: str
    confidence: Optional[float] = None
    timestamp: float


class SessionResponse(BaseModel):
    messages: List[Message]
    started_at: float
    session_id: str


# ---- TTS ----
class SpeakRequest(BaseModel):
    text: str
    lang: str = "en"
