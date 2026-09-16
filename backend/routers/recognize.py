"""
POST /api/recognize — Landmark-based sign recognition endpoint.
Supports 4 heads across ASL and ISL:
1. digits (ASL) — static 63-dim landmark vector
2. words (ISL) — 32-frame sequence of 63-dim landmark vectors
3. letters (ASL) — static 63-dim landmark vector (A-Z)
4. gesture_asl (ASL) — 32-frame sequence of 63-dim landmark vectors (HELLO, NO, etc.)
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import numpy as np

from ..schemas import RecognizeRequest, RecognizeResponse
from ..services.classifier import (
    predict_static, predict_temporal, predict_letters, predict_gesture_asl,
    get_models_status
)
from ..services.smoothing import get_stabilizer

router = APIRouter()


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(req: RecognizeRequest):
    """Process landmarks and return prediction with temporal stabilization and language attribution."""
    
    models = get_models_status()
    head: str
    language: str
    result = None
    
    if req.mode in ["static", "digits"]:
        head = "digits"
        language = "asl"
        if not models.get("static", False):
            raise HTTPException(503, "Static digit model not loaded")
        if len(req.landmarks) != 63:
            raise HTTPException(400, f"Expected 63 landmarks, got {len(req.landmarks)}")
        result = predict_static(req.landmarks)
        
    elif req.mode in ["temporal", "words"]:
        head = "words"
        language = "isl"
        if not models.get("temporal", False):
            raise HTTPException(503, "Temporal word model not loaded")
        if len(req.landmarks) % 63 != 0:
            raise HTTPException(400, "Landmark array length must be multiple of 63")
        seq_len = len(req.landmarks) // 63
        sequence = [req.landmarks[i*63:(i+1)*63] for i in range(seq_len)]
        result = predict_temporal(sequence)
        
    elif req.mode == "letters":
        head = "letters"
        language = "asl"
        if not models.get("letters", False):
            raise HTTPException(503, "Letters model not loaded")
        if len(req.landmarks) != 63:
            raise HTTPException(400, f"Expected 63 landmarks, got {len(req.landmarks)}")
        result = predict_letters(req.landmarks)
        
    elif req.mode in ["gesture_asl", "gestures"]:
        head = "gesture_asl"
        language = "asl"
        if not models.get("gesture_asl", False):
            raise HTTPException(503, "ASL gesture model not loaded")
        if len(req.landmarks) % 63 != 0:
            raise HTTPException(400, "Landmark array length must be multiple of 63")
        seq_len = len(req.landmarks) // 63
        sequence = [req.landmarks[i*63:(i+1)*63] for i in range(seq_len)]
        result = predict_gesture_asl(sequence)
        
    else:
        raise HTTPException(400, f"Unknown mode: {req.mode}")
    
    if result is None:
        return RecognizeResponse(
            raw={"label": "unknown", "confidence": 0.0, "top_k": []},
            stable={"emitted": False},
            status="no_hand_detected",
            message="No hand detected. Please show your hand to the camera.",
            debug={"window": [], "agreement": 0, "cooldown_remaining_ms": 0},
            head=head,
            language=language
        )
    
    # Apply temporal stabilization
    stabilizer = get_stabilizer(req.session_id, mode=head)
    stable_result = stabilizer.push(
        label=result["label"],
        confidence=result["confidence"],
        timestamp_ms=req.timestamp_ms or None
    )
    
    return RecognizeResponse(
        raw={
            "label": result["label"],
            "confidence": result["confidence"],
            "top_k": result["top_k"]
        },
        stable={
            "emitted": stable_result["emitted"],
            "label": stable_result["label"],
            "confidence": stable_result["confidence"]
        },
        status=stable_result["status"],
        message=stable_result["message"],
        debug=stable_result["debug"],
        head=head,
        language=language
    )
