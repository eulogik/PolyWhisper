# PolyWhisper

> Tiny multilingual speech recognition. English + Hindi MVP, scaling to 50+ languages.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/model-whisper--base-yellow.svg)](https://huggingface.co/openai/whisper-base)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eulogik/PolyWhisper/blob/main/polywhisper_common_voice.ipynb)

## What is PolyWhisper?

PolyWhisper is a lightweight multilingual automatic speech recognition (ASR) system built on top of OpenAI's Whisper. Instead of fine-tuning the entire 74M parameter encoder, PolyWhisper freezes Whisper's encoder and trains small language-specific decoder adapters (~20MB per language).

**Why?** Full Whisper fine-tuning requires GPU clusters. PolyWhisper trains a new language in hours on a single T4, then runs inference anywhere Whisper runs.

### Key Features

- **Multilingual ASR** — one shared encoder, language-specific adapters
- **Language auto-detection** — built-in language ID head
- **Tiny adapters** — ~20MB per language vs 1.5GB for full Whisper
- **Trainable on Colab Free** — no expensive hardware needed
- **Fully resumable** — crash-safe training with automatic checkpoint recovery
- **Edge-ready** — quantize and deploy on mobile/embedded

## Architecture

```
┌─────────────────────────────────────────────┐
│              Audio Input (16kHz)             │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         Whisper Base Encoder (74M)           │
│              [FROZEN]                        │
│  Produces 512-dim encoder hidden states     │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
┌────────▼───────┐  ┌────────▼───────┐
│ Language ID    │  │ Decoder        │
│ Head (0.1M)    │  │ Adapter (20MB) │
│ Auto-detects   │  │ Per-language   │
│ language       │  │ cross-attention│
└────────────────┘  └────────┬───────┘
                             │
                   ┌─────────▼─────────┐
                   │   Text Output     │
                   └───────────────────┘
```

| Component | Parameters | Status |
|-----------|-----------|--------|
| Whisper Base encoder | 74M | Frozen |
| Language ID head | 0.1M | Trained |
| Per-language adapter | ~6M | Trained |
| Total trainable (2 langs) | ~12M | — |

## Quick Start

### Install

```bash
git clone https://github.com/eulogik/PolyWhisper.git
cd PolyWhisper
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run on Colab (Recommended)

The easiest way to train and evaluate:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eulogik/PolyWhisper/blob/main/polywhisper_common_voice.ipynb)

1. Click the badge above
2. Run all cells (Ctrl+F9)
3. Trained adapters save to Google Drive automatically

### Local Usage

```python
from src.model import PolyWhisper

model = PolyWhisper("openai/whisper-base", ["en", "hi"])
model.load_adapters("models/adapters/hi_best.pt")

# Run inference
logits = model(audio_features, decoder_input_ids, language="hi")

# Auto-detect language
result = model.detect_language(audio_features)
print(f"Detected: {result['language']} ({result['confidence']:.1%})")
```

### Train an Adapter

```bash
python src/main.py --language hi --epochs 10 --batch-size 8 --device mps
```

## Training Data

| Dataset | Languages | Hours | Used For |
|---------|-----------|-------|----------|
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | 102 | ~12h | Pipeline validation |
| [Common Voice 17.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | 100+ | 1000h+ | Production training |
| [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) | 1 | 1000h | English baseline |

## Results

| Language | Dataset | WER | Notes |
|----------|---------|-----|-------|
| English | FLEURS | 100% | Insufficient data (2K samples) |
| Hindi | FLEURS | 100% | Insufficient data (1.2K samples) |

> Training on Common Voice (50K+ samples per language) in progress.

## Project Structure

```
PolyWhisper/
├── src/
│   ├── model.py              # PolyWhisper architecture
│   ├── data.py               # Dataset loading pipeline
│   ├── train.py              # Training loop with checkpointing
│   ├── main.py               # CLI entry point
│   └── test_model.py         # Smoke tests
├── notebooks/
│   ├── polywhisper_training.ipynb      # FLEURS training
│   ├── polywhisper_common_voice.ipynb  # Common Voice training (recommended)
│   └── polywhisper_eval.ipynb          # Evaluation notebook
├── models/
│   └── adapters/             # Trained adapter weights
├── LIVING.md                 # Project decisions log
├── CITATION.cff              # Academic citation
└── requirements.txt
```

## Hardware Requirements

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Training | T4 GPU (Colab Free) | A100 (Colab Pro) |
| Inference | CPU (any) | MPS/CUDA GPU |
| Storage | 5GB | 20GB |

## Roadmap

- [x] Architecture implementation
- [x] Smoke tests on M4 MPS
- [x] FLEURS pipeline validation
- [x] Common Voice training pipeline
- [x] Fully resumable training
- [ ] English + Hindi trained adapters
- [ ] Tamil, Telugu, Bengali adapters
- [ ] Whisper tokenizer size optimization (51865 → 1024)
- [ ] HuggingFace model card
- [ ] Quantized export (ONNX, GGML)
- [ ] Streaming inference
- [ ] Mobile deployment (Core ML, TFLite)

## Contributing

Contributions welcome! Please read [LIVING.md](LIVING.md) for project context and decisions.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — encoder backbone
- [Hugging Face](https://huggingface.co/) — transformers, datasets, model hub
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) — multilingual speech data
- [Google FLEURS](https://huggingface.co/datasets/google/fleurs) — multilingual evaluation

---

**PolyWhisper** — making multilingual speech recognition accessible to everyone.
