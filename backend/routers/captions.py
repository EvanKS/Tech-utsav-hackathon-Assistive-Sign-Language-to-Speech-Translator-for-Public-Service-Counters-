"""
POST /api/text-to-sign — Text to sign caption token endpoint.
POST /api/message — Conversation message endpoint.
GET  /api/session/{session_id} — Get session messages.
POST /api/session/{session_id}/reset — Reset session.
"""
from fastapi import APIRouter

from ..schemas import (
    TextToSignRequest, TextToSignResponse,
    MessageRequest, SessionResponse, Message
)
from ..services.vocabulary import text_to_sign_tokens
from ..services.session import get_session, add_message, reset_session

router = APIRouter()


@router.post("/text-to-sign", response_model=TextToSignResponse)
async def text_to_sign(req: TextToSignRequest):
    """Convert text to sign caption tokens."""
    result = text_to_sign_tokens(req.text)
    return TextToSignResponse(**result)


@router.post("/message", response_model=SessionResponse)
async def post_message(req: MessageRequest):
    """Add a message to the conversation log."""
    messages = add_message(
        session_id=req.session_id,
        sender=req.sender,
        text=req.text,
        confidence=req.confidence
    )
    session = get_session(req.session_id)
    return SessionResponse(
        messages=[Message(**m) for m in messages],
        started_at=session["started_at"],
        session_id=req.session_id
    )


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session_endpoint(session_id: str):
    """Get session conversation log."""
    session = get_session(session_id)
    return SessionResponse(
        messages=[Message(**m) for m in session["messages"]],
        started_at=session["started_at"],
        session_id=session_id
    )


@router.post("/session/{session_id}/reset")
async def reset_session_endpoint(session_id: str):
    """Reset session."""
    reset_session(session_id)
    return {"status": "ok", "session_id": session_id}
