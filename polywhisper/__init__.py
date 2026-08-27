"""PolyWhisper — efficient multilingual Indic ASR via frozen Whisper + per-language LoRA."""

__version__ = "0.1.0"

from polywhisper.transcribe import transcribe, TranscriptionResult

__all__ = ["transcribe", "TranscriptionResult"]
