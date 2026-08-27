# PolyWhisper

Efficient multilingual Indic ASR via frozen Whisper backbone + per-language LoRA adapters.

## Install

```bash
pip install -e .
```

Requires Python 3.9+, torch, transformers, soundfile. For CPU-only inference, also install `onnxruntime`.

## Quick Start

```bash
# Transcribe a single file
polywhisper transcribe audio.wav --lang hi

# Auto-detect language
polywhisper transcribe audio.wav

# Batch transcribe a folder
polywhisper batch ./audio_folder/ --lang ta

# List available languages
polywhisper languages
```

## Python API

```python
from polywhisper import transcribe

result = transcribe("audio.wav", lang="hi")
print(result.text)          # "नमस्ते, आप कैसे हैं?"
print(result.segments)      # [Segment(text=..., start_sec=0.0, end_sec=5.2)]
print(result.duration_sec)  # 5.2

# Batch transcription
result = transcribe("long_meeting.mp3", lang="te", max_new_tokens=512)
```

## Supported Languages

| Language | Code | Script |
|----------|------|--------|
| Hindi | `hi` | Devanagari |
| Tamil | `ta` | Tamil |
| Telugu | `te` | Telugu |
| Bengali | `bn` | Bengali |
| Marathi | `mr` | Devanagari |

## Output Formats

```bash
# Plain text
polywhisper transcribe audio.wav --lang hi --format text

# JSON with timestamps
polywhisper transcribe audio.wav --lang hi --format json

# SRT subtitles
polywhisper transcribe audio.wav --lang hi --format srt
```

## Model Details

- **Backbone**: Whisper Small (244M params, frozen)
- **Adapters**: Per-language LoRA (rank 16, ~14MB each)
- **Total params**: ~258M (backbone) + ~3.5M (LoRA per language)
- **Inference**: ~3-5s per clip on MPS, ~20s on CPU via ONNX Runtime

## Export to ONNX (for CPU deployment)

```bash
polywhisper export --lang hi --variant prod --int8
polywhisper export --lang ta --variant prod --int8
```

ONNX models are saved to `export/onnx/` (or KIOXIA path).

## Project Structure

```
polywhisper/
├── __init__.py        # Public API (transcribe, TranscriptionResult)
├── __main__.py        # python -m polywhisper
├── model.py           # PolyWhisper model + adapter management
├── transcribe.py      # High-level transcribe() function
├── audio.py           # Audio loading, chunking, format conversion
├── cli.py             # CLI (transcribe, batch, export, languages)
└── onnx_backend.py    # ONNX Runtime inference backend
```

## Performance (FLEURS, normalized WER)

| Language | Whisper-Small (vanilla) | Expert (Base+LoRA) | Product (Small+LoRA) |
|----------|------------------------|---------------------|----------------------|
| HI | 62.3 | 38.6 | **37.2** |
| TA | 68.8 | 74.2 | **60.4** |
| TE | 129.6 | 91.9 | 105.7 |
| BN | 114.2 | 120.5 | 196.3 |
| MR | 118.8 | 65.0 | 167.3 |

Product matches/beats expert on Hindi and Tamil. Telugu/Bengali/Marathi require retraining with domain-matched data (scheduled for next Kaggle run).
