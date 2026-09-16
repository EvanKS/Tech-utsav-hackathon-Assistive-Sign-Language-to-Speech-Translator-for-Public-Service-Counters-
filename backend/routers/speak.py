"""
POST /api/speak — Server-side TTS fallback using gTTS.
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import io
from gtts import gTTS

from ..schemas import SpeakRequest

router = APIRouter()


@router.post("/speak")
async def speak(req: SpeakRequest):
    """Generate TTS audio as MP3 stream (fallback when browser TTS unavailable)."""
    tts = gTTS(text=req.text, lang=req.lang)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    
    return StreamingResponse(
        audio_buffer,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"}
    )
