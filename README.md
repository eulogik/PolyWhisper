# PolyWhisper

A tiny multilingual ASR model. English + Hindi MVP first.

## Architecture

- **Encoder**: Whisper Base (74M params, frozen) — provides universal acoustic features
- **Language ID Head** (0.1M) — classifies audio into target languages
- **Decoder Adapters** (~6M per language) — LoRA-inspired low-rank adapters with cross-attention to encoder

## Project Structure

```
├── src/
│   ├── model.py          # PolyWhisper model definition
│   ├── data.py           # Dataset and dataloader
│   ├── train.py          # Training loop and evaluation
│   ├── main.py           # CLI entry point
│   └── test_model.py     # Smoke tests
├── data/                 # Downloaded datasets
├── models/               # Saved adapter weights
├── LIVING.md             # Project story and decisions
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Test the model
```bash
python src/test_model.py
```

### Train an adapter
```bash
python src/main.py --language hi --epochs 10 --batch-size 8 --device mps
python src/main.py --language en --epochs 10 --batch-size 8 --device mps
```

### Load and use adapters
```python
from model import PolyWhisper

model = PolyWhisper("openai/whisper-base", ["en", "hi"])
model.load_adapters("models/adapters/hi_best.pt")
logits = model(audio_features, decoder_input_ids, language="hi")
```

## Key Parameters

| Component | Params |
|-----------|--------|
| Whisper Base encoder (frozen) | 74M |
| Language ID head | 0.1M |
| Per-language adapter | ~6M |
| Total trainable (2 langs) | ~12M |

## Device Support

- **MPS**: Apple Silicon GPU (tested on M4 Mac Mini)
- **CPU**: Fallback
- **CUDA**: Not yet tested
