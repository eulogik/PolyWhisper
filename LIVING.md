# LIVING.md — Development Plan & Walkthrough

> This document is the complete story of how this is built, why each decision was made, and where it's going. It is updated with every significant change. Anyone can read this to understand the project from any point in time.

---

## Project: PolyWhisper — English + Hindi MVP

**Started:** 2026-06-25
**Hardware:** M4 Mac Mini 16GB (development) + Google Colab Free/T4 (training)
**Goal:** Build a tiny ASR model for English and Hindi using shared Whisper Base encoder + language-specific decoder adapters.

---

## Phase 0: Architecture Decision (2026-06-25)

### Decision: Use Whisper Base as frozen encoder

**Why:**
- Training a Conformer encoder from scratch requires 500K+ hours and A100 — not feasible on our hardware
- Whisper Base (74M) already has strong English acoustic representations
- Fine-tuning it on Hindi data is possible on M4/Colab
- The adapter architecture still proves the concept — language-specific decoders on shared encoder

**Trade-off:**
- Larger total model (74M encoder vs 30M custom encoder)
- But much faster to train and still achieves quality targets

### Architecture (MVP)

```
Audio (16kHz) → Log-Mel Spectrogram (80 bins, 3000 frames)
                        ↓
            Whisper Base Encoder (frozen, 74M)
                        ↓
            Language ID Head (0.1M) → detects en/hi
                        ↓
            Decoder Adapter (~6M per lang)
                        ↓
            Transcription Output
```

---

## Phase 1: Environment Setup (2026-06-25)

### Hardware
- M4 Mac Mini, 16GB RAM, 228GB SSD (only 15GB free — cleaned up venv)
- MPS (Apple Silicon GPU) available and working

### Software
- Python 3.14, PyTorch 2.12.1 (CPU-only wheel from pytorch.org/whl/cpu)
- Transformers 5.12.1, Datasets 5.0.0, Librosa 0.11.0
- Virtual environment at `.venv/`

### Disk Space Warning
- System has only 15GB free — need to be careful with dataset downloads
- Will stream datasets rather than caching fully

---

## Phase 2: Model Implementation (2026-06-25)

### Key Design Decisions

1. **Whisper Base d_model = 512** (not 256 as originally planned in the doc)
   - Adapter cross-attention must project from 512 → 256
   - LowRankAdapter needs explicit input_dim parameter

2. **Adapter architecture:**
   - Token embedding (vocab_size → 256)
   - Position embedding (256 max positions)
   - 2 layers of: self-attn → cross-attn → FFN
   - Each attention has LoRA adapters on q and v projections
   - Output projection to vocab_size

3. **Language ID head:**
   - AdaptiveAvgPool1d + MLP classifier
   - Takes encoder hidden states, outputs language logits

4. **Save/load:**
   - Only adapter weights + LID head saved (not full Whisper)
   - Enables adding new languages without re-uploading 74M encoder

### Bugs Fixed During Implementation
- Whisper expects exactly 3000 frames (30s audio) — not arbitrary lengths
- MPS doesn't support integer-to-float linear projections — need embedding layer first
- AdapterLayer was passing wrong encoder_dim (256 instead of 512)
- LowRankAdapter needed explicit input_dim (512 for encoder outputs, 256 for self-attention)

### Smoke Test Results
```
Parameters: 6.0M per adapter, 0.1M LID head
Forward pass: ✓ (both en and hi)
Language ID: ✓ (random init, ~50/50 as expected)
Save/Load: ✓
Device: MPS working
```

---

## Phase 3: Data Pipeline (2026-06-25)

### Issue: Python 3.14 + dill incompatibility
- `datasets` library uses `dill` for pickling
- Python 3.14 changed pickle protocol (`_batch_setitems` signature)
- Tried multiple patches — none worked reliably
- **Solution**: Use synthetic data for M4 testing, use Colab for real training (different Python version)

### Datasets planned (for Colab training):
- **English**: Mozilla Common Voice 17 (en) — ~3K hours
- **Hindi**: Mozilla Common Voice 17 (hi) — ~12 hours
- **Fallback**: Google FLEURS (small, ~1K samples per language)

---

## Phase 4: Training (2026-06-25) — IN PROGRESS

### M4 Mac Mini Test Results (Synthetic Data)
- 200 synthetic samples, 5 epochs, batch_size=8
- Loss: 0.3954 → 0.0253 → 0.0026 → 0.0026 → 0.0026
- ✅ Model trains correctly on MPS
- ✅ Gradients flow through adapter layers
- ✅ Loss decreases as expected
- Speed: ~5 steps/sec on M4 MPS (with 30s audio)

### Next: Real training on Colab
- Full Common Voice data
- 10 epochs per language
- Checkpoint to Google Drive

---

## Phase 5: Evaluation (TODO)

### TODO: WER calculation and benchmarking

---

## Phase 6: Export & Release (TODO)

### TODO: HuggingFace model card and demo

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-25 | Use Whisper Base as encoder | Feasible on available hardware, still proves adapter concept |
| 2026-06-25 | English + Hindi MVP | Prove architecture works before scaling to 50 languages |
| 2026-06-25 | Freeze encoder entirely | M4 can't handle fine-tuning 74M params; Colab T4 barely can |
| 2026-06-25 | Custom adapter (not HuggingFace PEFT) | Full control over architecture, cross-attention, streaming-ready |
| 2026-06-25 | vocab_size=1024 per language | Small BPE vocab sufficient for MVP; can increase later |

---

## Current Status

🟢 **Phase 2 complete** — Model implemented and tested on MPS
🟡 **Phase 3 next** — Data pipeline and first training run

---

## What's Next

1. Test data loading (stream from HuggingFace datasets)
2. Do a short training run on M4 (1 epoch, 100 samples) to verify gradient flow
3. Then move to Colab for full training
