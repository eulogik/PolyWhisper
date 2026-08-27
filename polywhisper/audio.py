"""Audio loading, format conversion, and chunking utilities."""

import numpy as np
import soundfile as sf
from pathlib import Path

TARGET_SR = 16000
MAX_DURATION_SEC = 30.0
CHUNK_OVERLAP_SEC = 3.0


def load_audio(path, target_sr=TARGET_SR):
    """Load audio file, return (samples: np.ndarray[float32], sr: int).

    Supports wav, flac, mp3, ogg, m4a via soundfile/libsndfile.
    For formats not supported by soundfile, falls back to ffmpeg.
    """
    path = str(path)
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        # ffmpeg fallback for mp3/ogg/m4a
        import subprocess, io, tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(
                ["ffmpeg", "-i", path, "-ar", str(target_sr), "-ac", "1",
                 "-f", "wav", "-y", tmp.name],
                capture_output=True, check=True,
            )
            audio, sr = sf.read(tmp.name, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono
    if sr != target_sr:
        import resampy
        audio = resampy.resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def chunk_audio(audio, chunk_sec=MAX_DURATION_SEC, overlap_sec=CHUNK_OVERLAP_SEC, sr=TARGET_SR):
    """Split long audio into overlapping chunks. Returns list of (start_sample, end_sample)."""
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    n = len(audio)
    if n <= chunk_samples:
        return [(0, n)]
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_samples, n)
        chunks.append((start, end))
        if end == n:
            break
        start += chunk_samples - overlap_samples
    return chunks


def audio_to_chunks(audio_or_path, chunk_sec=MAX_DURATION_SEC, overlap_sec=CHUNK_OVERLAP_SEC, sr=TARGET_SR):
    """Load audio (if path) and return list of (chunk_np, start_sec, end_sec)."""
    if isinstance(audio_or_path, (str, Path)):
        audio, _ = load_audio(audio_or_path, sr)
    else:
        audio = np.asarray(audio_or_path, dtype="float32")
    spans = chunk_audio(audio, chunk_sec, overlap_sec, sr)
    return [(audio[s:e], s / sr, e / sr) for s, e in spans]
