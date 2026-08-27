"""High-level transcription API."""

import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union

from polywhisper.model import PolyWhisper, AVAILABLE_LANGS
from polywhisper.audio import load_audio, audio_to_chunks, TARGET_SR


@dataclass
class Segment:
    """A single transcription segment."""
    text: str
    start_sec: float
    end_sec: float
    tokens: List[int] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Full transcription result."""
    text: str
    lang: str
    segments: List[Segment] = field(default_factory=list)
    language_probs: dict = field(default_factory=dict)

    @property
    def duration_sec(self):
        if not self.segments:
            return 0.0
        return self.segments[-1].end_sec

    def to_dict(self):
        return {
            "text": self.text,
            "lang": self.lang,
            "segments": [
                {"text": s.text, "start": s.start_sec, "end": s.end_sec}
                for s in self.segments
            ],
        }


def _get_model(backbone="small", device="auto", model=None):
    """Return a PolyWhisper instance, reusing if already loaded."""
    if model is not None:
        return model
    return PolyWhisper(backbone=backbone, device=device)


def _prepare_features(model, audio):
    """Convert audio array to model input features."""
    feats = model.processor.feature_extractor(
        [audio], sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )["input_features"]
    if feats.shape[-1] < 3000:
        feats = torch.cat(
            [feats, torch.zeros(1, 80, 3000 - feats.shape[-1], dtype=feats.dtype)], -1
        )
    return feats.to(model.device)


def _decode_tokens(model, tokens):
    """Decode token IDs to text."""
    return model.processor.decode(tokens, skip_special_tokens=True).strip()


def transcribe(
    audio_input: Union[str, Path, np.ndarray],
    lang: Optional[str] = None,
    backbone: str = "small",
    variant: str = "prod",
    device: str = "auto",
    max_new_tokens: int = 256,
    num_beams: int = 1,
    chunk_sec: float = 30.0,
    model: Optional[PolyWhisper] = None,
    language_probs: bool = False,
) -> TranscriptionResult:
    """Transcribe audio file or numpy array.

    Args:
        audio_input: file path (str/Path) or numpy float32 array at 16kHz.
        lang: language code ("hi", "ta", "te", "bn", "mr"). If None, auto-detect.
        backbone: whisper backbone ("small" recommended).
        variant: adapter variant ("prod" or "base").
        device: "auto", "cuda", "mps", or "cpu".
        max_new_tokens: max generation length.
        num_beams: beam width (1 = greedy).
        chunk_sec: chunk long audio into this many seconds.
        model: pre-loaded PolyWhisper instance (avoids re-loading).
        language_probs: if lang is None, return probs for all languages.

    Returns:
        TranscriptionResult with text, segments, and metadata.
    """
    # Load audio
    if isinstance(audio_input, (str, Path)):
        audio, sr = load_audio(audio_input, TARGET_SR)
        source = str(audio_input)
    else:
        audio = np.asarray(audio_input, dtype=np.float32)
        source = "<array>"

    m = _get_model(backbone, device, model)

    # Auto-detect language
    if lang is None:
        lang, probs = _detect_language(m, audio, max_new_tokens=max_new_tokens)
        if language_probs:
            probs_str = ", ".join(f"{k}:{v:.1%}" for k, v in sorted(probs.items(), key=lambda x: -x[1]))
            print(f"Detected: {lang} ({probs_str})")
    else:
        probs = {}

    # Load adapter
    m.auto_load_adapter(lang, variant=variant)

    # Chunk and transcribe
    chunks = audio_to_chunks(audio, chunk_sec=chunk_sec)
    segments = []
    all_text = []

    for chunk_audio, start_sec, end_sec in chunks:
        feats = _prepare_features(m, chunk_audio)
        out = m.generate(
            feats, lang=lang, max_new_tokens=max_new_tokens,
            num_beams=num_beams, use_cache=True, task="transcribe",
        )
        text = _decode_tokens(m, out[0])
        if text:
            segments.append(Segment(text=text, start_sec=start_sec, end_sec=end_sec))
            all_text.append(text)

    full_text = " ".join(all_text)
    return TranscriptionResult(
        text=full_text, lang=lang, segments=segments, language_probs=probs,
    )


def _detect_language(model, audio, max_new_tokens=128):
    """Run all adapters, pick the one with highest confidence."""
    from polywhisper.model import ADAPTER_REGISTRY
    from polywhisper.audio import TARGET_SR
    import torch.nn.functional as F

    feats = _prepare_features(model, audio)
    scores = {}

    for lang in ADAPTER_REGISTRY:
        try:
            model.auto_load_adapter(lang, variant="prod")
        except FileNotFoundError:
            try:
                model.auto_load_adapter(lang, variant="base")
            except FileNotFoundError:
                continue

        model.set_language(lang)
        with torch.no_grad():
            # Get decoder logits for the first few tokens
            enc = model.whisper.model.encoder(feats).last_hidden_state
            # Start with BOS + lang token
            lang_token = model.processor.tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
            if lang_token is None:
                continue
            dec_input = torch.tensor([[50257, lang_token, 50359]], device=model.device)
            logits = model.whisper.model.decoder(dec_input, encoder_hidden_states=enc).last_hidden_state
            # Score: average log-prob of next-token predictions
            logprobs = F.log_softmax(logits[0, -1], dim=-1)
            top_k = logprobs.topk(5)
            scores[lang] = top_k.values.mean().item()

    if not scores:
        return "hi", {}

    # Normalize to probabilities
    total = sum(np.exp(v) for v in scores.values())
    probs = {k: np.exp(v) / total for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    best = max(scores, key=scores.get)
    return best, probs
