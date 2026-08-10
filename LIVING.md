# LIVING.md — Development Plan & Walkthrough

> This document is the complete story of how this is built, why each decision was made, and where it's going. It is updated with every significant change. Anyone can read this to understand the project from any point in time.

---

## Project: PolyWhisper — English + Hindi + Hinglish MVP

**Started:** 2026-06-25
**Hardware:** M4 Mac Mini 16GB (development + training)
**Goal:** Build a tiny ASR model for English, Hindi, and Hinglish using shared Whisper Base encoder + language-specific decoder adapters.
**Repo:** https://github.com/eulogik/PolyWhisper (private)

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
            Language ID Head (0.1M) → detects en/hi/hinglish
                        ↓
            Decoder Adapter (~65MB per lang with tied embeddings)
                        ↓
            Transcription Output
```

---

## Phase 1: Environment Setup (2026-06-25)

### Hardware
- M4 Mac Mini, 16GB RAM, 228GB SSD
- MPS (Apple Silicon GPU) available and working

### Software
- Python 3.11.14, PyTorch 2.12.1 (MPS)
- Transformers 4.44.2, Datasets 3.1.0, Librosa 0.11.0
- Virtual environment at `.venv/`

---

## Phase 2: Model Implementation (2026-06-25)

### Key Design Decisions

1. **Whisper Base d_model = 512** (not 256 as originally planned)
   - Adapter cross-attention projects from 512 → 256
   - LowRankAdapter needs explicit input_dim parameter

2. **Adapter architecture:**
   - Token embedding (vocab_size → 256)
   - Position embedding (256 max positions)
   - 3 layers of: self-attn (with causal mask) → cross-attn → FFN
   - Each attention has LoRA adapters on q and v projections
   - Output projection to vocab_size (tied with token embedding)

3. **Causal mask is mandatory** (fixed 2026-06-30):
   - Without it, bidirectional self-attention lets model peek at future tokens during training
   - Low training loss but 100% WER at inference
   - Triangular mask applied in SelfAttn.forward()

4. **Tied embeddings** (fixed 2026-06-28):
   - output_proj.weight = token_embedding.weight
   - Halves adapter size: 65MB vs 120MB per language

5. **Save/load:**
   - Only adapter weights saved (not full Whisper)
   - Enables adding new languages without re-uploading 74M encoder

---

## Phase 3: Data Pipeline (2026-06-25 to 2026-07-02)

### Datasets (final selection)
- **English**: LibriSpeech (openslr/librispeech_asr, clean) — 10K train, 1K test
- **Hindi**: IndicVoices-ST (ai4bharat/IndicVoices-ST, hindi) — 10K train (filtered from 54K), test from FLEURS (417)
- **Hinglish**: ujs/hinglish — 10K train, 3K test

### Hindi Upgrade: FLEURS (2K) → IndicVoices-ST (54K)
- FLEURS Hindi only has ~2K train samples (dataset limitation, not selection)
- IndicVoices-ST has 54,562 Hindi samples across 44K hours of 13 Indian languages
- Quality filter: `alignment_score > 0.8` removes noisy samples
- Gated dataset — requires HF login + contact info acceptance (done 2026-07-02)
- Test set kept as FLEURS (cached, 417 samples) for consistent benchmarking

### Why not Common Voice 17?
- Both repos broken on HuggingFace Hub ("doesn't contain any data files" / "Dataset scripts no longer supported")
- Newer `datasets` 3.x dropped loading script support
- Definitively abandoned for now

### Disk cleanup (2026-06-30)
- Freed ~25GB by clearing Xcode simulators (7.3GB), wallpaper cache (7.3GB), pip cache (251MB)
- Reduced LibriSpeech from 28K→10K samples to fit 228GB SSD (only ~27GB free)

---

## Phase 4: Training (2026-06-28 to 2026-06-30)

### Training completed: 30.2 hours, 55,280 steps

**Configuration:**
- 3 languages: en, hi, hinglish
- 20 epochs, batch_size=8, LR=5e-4 with warmup
- AdamW optimizer with weight_decay=0.01
- MPS on M4 Mac Mini
- Hindi training data: IndicVoices-ST (filtered, 10K) + FLEURS (2K) replaced by IndicVoices-ST alone

**Results:**

| Language | Best Val Loss | Adapter Size |
|----------|--------------|--------------|
| English | 0.738 | 64.9MB |
| Hindi | 1.393 | 64.9MB |
| Hinglish | 1.070 | 64.9MB |

**Adapters saved:**
- `polywhisper_output/adapters/en_best.pt` (64.9MB)
- `polywhisper_output/adapters/hi_best.pt` (64.9MB)
- `polywhisper_output/adapters/hinglish_best.pt` (64.9MB)
- `polywhisper_output/adapters/polywhisper_final.pt` (194.7MB combined)

### Bugs fixed during training

1. **Causal mask missing** (2026-06-30): Self-attention was bidirectional → model cheated during teacher-forced training → 100% WER
2. **Training state not updating** (2026-06-30): `state["lang"]` and `state["epoch"]` never updated in loop → checkpoints always named `en_ep0_last.pt`
3. **JSON serialization crash** (2026-06-30): optimizer state_dict contains tensors → not JSON-serializable → saved optimizer to separate `.opt.pt` file
4. **Download stall detection** (2026-06-29): no new samples in 60s → abort gracefully

---

## Phase 5: Evaluation (2026-06-30)

### WER Results (greedy decoding + repetition penalty)

| Language | WER | Samples | Notes |
|----------|-----|---------|-------|
| English | 209.9% | 50 | Gets sentence beginnings right, diverges after |
| Hindi | 102.6% | 50 | Produces Hindi script but wrong words |
| Hinglish | 206.5% | 50 | Partial captures on short phrases |

### Analysis

**What's working:**
- Model produces language-appropriate text (correct script, correct language)
- Repetition penalty fixed "OF THEM OF THEM..." loops
- Works better on short utterances
- Architecture is sound — model is learning the right patterns

**What needs improvement:**
- Undertrained: only 2K-10K samples per language × 20 epochs
- Greedy decoding is weak for ASR — beam search needed
- Model generates 128 tokens even for short sentences → adds garbage

### Next steps for evaluation
1. Train longer (50-100 epochs)
2. Add beam search decoding
3. Increase training data (FLEURS hi only has 2K samples)
4. Reduce max_new_tokens for shorter sentences

---

## Innovation Assessment (2026-07-02)

PolyWhisper's innovation is the **architectural synthesis** of proven ideas into a practical, deployable system:

### What Makes It Novel

1. **Frozen encoder + language-specific decoder** — clean separation of acoustic modeling (shared) from language modeling (independent). Architecturally distinct from:
   - Full Whisper fine-tuning (modifies all 74M params)
   - HuggingFace PEFT/LoRA (adds adapters *within* Whisper's decoder)
   - Moonshine (separate encoder+decoder per language, English-only)
   - Traditional multilingual fine-tuning (one model, cross-language interference)

2. **LoRA within custom decoders** — rank-8/16 bottleneck on Q+V projections, zero-initialized so adapters start as identity and learn incrementally. Applied to both self-attention and cross-attention.

3. **Tied embeddings for compression** — output projection shares weights with token embedding, halving adapter size (65MB vs 120MB per language) without quality loss.

4. **Per-language runtime swapping** — adapters are independent modules, hot-swappable, independently deployable without loading Whisper decoder.

5. **Causal mask correctness** — critical discovery: missing causal mask causes 100% WER despite low training loss (model cheats by peeking at future tokens).

### Research Gaps Addressed

| Gap | PolyWhisper Solution |
|-----|---------------------|
| Whisper "mediocre everywhere" on low-resource languages | Per-language specialization without sacrificing shared acoustic backbone |
| Moonshine (English-only tiny model) | Multilingual adapters on shared encoder |
| Full fine-tuning too expensive | 65MB adapter, 30h training on consumer hardware |
| Model bloat for multi-language | Sub-linear storage scaling (encoder shared, +65MB per language) |
| Runtime language switching | Clean adapter swapping without cross-language interference |

### Practical Impact

- **Democratizes ASR** — individual developers in low-resource communities can train adapters
- **Edge-ready** — 65MB adapters fit mobile/embedded devices
- **India-focused** — Voice-Bharat: 22 scheduled languages, no existing tiny multilingual ASR for Indian languages

### Current State

**Proof-of-concept stage.** Architecture is sound, infrastructure is mature, but WER is far from production (209% en, 102% hi, 206% hinglish). The bottleneck is training data and epochs, solvable with more training.

---

## Architecture Discussion: Whisper Base vs Custom Conformer (2026-07-02)

**Decision: Using Whisper Base (74M) is better for MVP, deferred custom encoder to Phase 2.**

| Dimension | Whisper Base (74M) | Custom Conformer (30M) |
|-----------|-------------------|----------------------|
| Quality | Excellent — pretrained on 680K hrs, 96 langs | Unknown — needs months of training |
| Training cost | ~$0 (M4) | Enormous (cluster required) |
| Inference size | 74M (larger) | 30M (smaller) |
| Deployment | Needs quantization for edge | Fits edge natively |

**Verdict:** Whisper Base is the right choice for MVP. The custom encoder is only needed later for edge deployment where 74M is too large. Whisper's multilingual pretraining actually benefits the adapters — it already knows Hindi acoustically from its 680K-hour training.

---

## Data Limitations: FLEURS Hindi (2026-07-02)

FLEURS Hindi (`hi_in`) has only ~2K train samples — that's the dataset's fixed size, not a sampling choice. To get 10K+ Hindi samples, options:
- **indicvoices/IndicVoices-ST** (44K hours, 13 Indian languages) — gated on HF, needs license acceptance
- **ULCA** — another Indian language dataset
- **Combine FLEURS + ULCA** for more coverage

Currently blocked on IndicVoices-ST (gated). Future training should pursue license access.

---

## Current Status

✅ **Phase 1-4 complete** — Architecture, data, training all working (30.2h, 55K steps)
✅ **Phase 5 partial** — Evaluation done, results show model works but needs more training
💾 **All training saved and pushed** — adapters, state, data all backed up on GitHub

---

## What's Next

1. **Train further** — 50-100 epochs to improve WER (resume with `python train_local.py`)
2. **Add beam search** — greedy decoding is weak for ASR
3. **More Hindi data** — pursue IndicVoices-ST license access (gated, 44K hours)
4. **Voice-Bharat expansion** — add ta, te, bn, mr, gu + tanglish
5. **HuggingFace release** — model card, demo, integration
6. **Quantize and export** — for edge deployment (ONNX, GGML)

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-25 | Use Whisper Base as encoder | Feasible on available hardware, still proves adapter concept |
| 2026-06-25 | English + Hindi MVP | Prove architecture works before scaling to 50 languages |
| 2026-06-25 | Freeze encoder entirely | M4 can't handle fine-tuning 74M params |
| 2026-06-25 | Custom adapter (not HuggingFace PEFT) | Full control over architecture, cross-attention, streaming-ready |
| 2026-06-25 | Use `len(tokenizer)` not `vocab_size` | Whisper special tokens need 51865 not 50258 |
| 2026-06-28 | Tie embeddings | output_proj.weight = token_embedding.weight → halves adapter size |
| 2026-06-28 | Python 3.11 over 3.14 | Python 3.14 on macOS 26 causes segfaults |
| 2026-06-28 | Switch from Common Voice to LibriSpeech | Common Voice Hub is broken on HuggingFace |
| 2026-06-30 | Add hinglish as 3rd language | Voice-Bharat vision for India market |
| 2026-06-30 | No external SSD for training | Risk of disconnection over 30h run |
| 2026-06-30 | No runtime limit on training | User wants full epochs to complete |
| 2026-06-30 | Causal mask mandatory | Without it, model cheats and gets 100% WER |
| 2026-06-30 | Indian English not added | FLEURS has no en_in config; IndicVoices-ST gated |
| 2026-07-02 | Keep hinglish as 3rd language | Needed for Voice-Bharat India market; pipeline works |
| 2026-07-02 | Whisper Base is better for MVP | 680K-hr pretraining quality > custom Conformer size savings; defer encoder to Phase 2 |
| 2026-07-02 | FLEURS Hindi 2K is dataset limit, not selection | Need IndicVoices-ST (gated) or ULCA for 10K+ |
| 2026-07-02 | Switch Hindi train to IndicVoices-ST (54K samples) | Quality filter alignment_score > 0.8; keep FLEURS for test benchmark |
| 2026-07-02 | Resume training with 48h run target | NUM_EPOCHS=80, batch=8, IndicVoices-ST Hindi data |
| 2026-07-27 | v2 rewrite: LoRA inside Whisper Base decoder | v1 custom decoder too weak; pretrained decoder + LoRA keeps 680K-hr knowledge |
| 2026-07-27 | Per-language LoRA on q/k/v + out_proj (self + cross attn) | Full attention adaptation without touching frozen weights |
| 2026-07-28 | Training first token = BOS (50257), not START (50258) | Matches `generate()`; START caused 124% WER |
| 2026-07-30 | Abandoned v2 run 1 (BOS mismatch) — full retrain | Adapters unusable; retraining cheaper than fixing |
| 2026-07-31 | Abandoned v2 epochs 8-10 (plateau) | No val improvement since epoch 6; ~8h saved |
| 2026-07-31 | Hybrid: vanilla for en + LoRA for hi/hinglish | Vanilla 5.6% en vs LoRA 44.9%; LoRA rescues Indic (44.8/50.0) |
| 2026-07-31 | Custom normalized WER (not jiwer raw) | jiwer 4.0 `wer()` is case-sensitive; `wer_standardize()` broken |
| 2026-07-31 | v3: LR 1e-4 + full 28.5K LibriSpeech for en retrain | LR 1e-3 + 10K samples caused catastrophic overfit (44.9%) |
| 2026-07-31 | Track B: token-level code-switch LoRA router | Hinglish mixes languages mid-sentence; single adapter is a blunt tool |
| 2026-07-31 | On-device (CoreML/ONNX) as first deployment target | 74M backbone + ~9MB adapters fits low-end phones; offline Indic ASR gap |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/model.py` | Full model architecture (tied embeddings, per-language save/load) |
| `train_local.py` | Bulletproof training script (M4 MPS, 3 languages, fully resumable) |
| `eval_local.py` | WER evaluation script (with repetition penalty + beam search) |
| `polywhisper_output/adapters/*.pt` | Trained adapter weights |
| `polywhisper_output/data/*.json` | Cached dataset manifests |
| `polywhisper_output/training.log` | Full training history (~47h) |
| `polywhisper_output/eval_results.json` | Latest evaluation results |
| `train_v2.py` | v2 training (LoRA in Whisper Base decoder, resumable) |
| `eval_v2.py` | v2 eval (LoRA adapters, normalized WER) |
| `eval_v2_baseline.py` | Vanilla Whisper Base baseline eval |
| `eval_v2_hybrid.py` | Hybrid eval (vanilla en + LoRA hi/hinglish) |
| `polywhisper_output/adapters_v2/*.pt` | v2 per-language LoRA adapters (best + per-epoch) |
| `LIVING.md` | This document |

---

## Phase 6: Label Bug Fix + Scheduled Sampling (2026-07-20)

### Root Cause Discovery: Label-Start Mismatch

**The bug:** The Whisper tokenizer silently adds `<|startoftranscript|>` (50258) and `<|notimestamps|>` (50363) before the text and `<|endoftext|>` (50257) after it. The Collator's old `lbls[lbls == pad_id] = -100` masked ALL `<|endoftext|>` tokens to -100 — including the **real** EOS — because `pad_id = bos_id = eos_id = 50257`. The model never learned to predict EOS.

Additionally, inference started with `tok.bos_token_id` (50257 = `<|endoftext|>`) but training's first input token was `<|startoftranscript|>` (50258). These don't match.

**Symptoms that made sense after discovery:**
- Model always predicted `<|notimestamps|>` (50363) as first output token with 99% probability
- Never emitted EOS → overgenerated 128 tokens of garbage
- Binary WER pattern (26% or 200%+) — correct first text token → good; wrong first token → cascade

**Fixes:**
1. **Collator fix** (`train_local.py`): Use `attention_mask` from tokenizer to track `real_len`. Only mask tokens BEYOND `real_len` to -100. This preserves the real EOS in the loss, teaching the model when to stop.
2. **Inference start token** (`eval_local.py`): Use `START_TOKEN = 50258` (`<|startoftranscript|>`) instead of `tok.bos_token_id` (50257), matching the training label format.
3. **Scheduled sampling**: Linear decay from 0.1→0 over 15 epochs (epochs 57-72). On each step with probability `p`, replace ground-truth prefix tokens with model's own predictions. This teaches recovery from exposure bias.

**Impact on training:** 
- Labels now: `[50258, 50363, ...text..., 50257, -100, ...]` (EOS preserved)
- Training input starts with 50258 (train-inference match)
- Model will learn to predict EOS at end of text
- Scheduled sampling at mild rate (0.1→0) shouldn't spike loss

**Training resumed** from epoch 56, ~24 epochs remaining, estimated ~33h.
Best pre-fix losses: en=0.632, hi=0.574, hinglish=1.040 (from 57 epochs)

---

## Phase 7: v2 Rewrite — LoRA Inside Whisper Base Decoder (2026-07-27 to 2026-07-30)

### Why rewrite
The v1 custom adapter decoder (Phase 2-6) maxed out at ~100%+ WER. The custom 3-layer decoder was too weak. Instead of building a decoder from scratch, we injected **per-language LoRA adapters directly into Whisper Base's pretrained decoder** — keeping all 680K-hour pretraining knowledge and only adapting a few projection layers.

### Architecture (v2)
```
Audio (16kHz) → Log-Mel (80 bins, 3000 frames)
                        ↓
        Whisper Base Encoder (frozen, 74M)
                        ↓
        Whisper Base Decoder (frozen, 74M)
        + LoRA rank-16 on q/k/v + out_proj
          of self-attn AND cross-attn, per layer
                        ↓
        Transcription Output (proj_out, frozen)
```
- **~2.36M trainable params total** across all 3 languages (~786K per language, ~3MB adapter files)
- LoRA B matrices zero-initialized → adapters start as identity
- Language switched at runtime via forward hooks; only one language's adapters active at a time
- Files: `train_v2.py`, `eval_v2.py`, `clean_datasets.py`

### Data (audited + cleaned 2026-07-27)
- **English**: LibriSpeech train.100 — capped at 10K samples (SSD constraint), 1K test
- **Hindi**: IndicVoices-ST (alignment_score > 0.8) — deduped 29K → 19.5K train, FLEURS test (264 after dedup)
- **Hinglish**: MUCS — template boilerplate removed + dedup 52K → 42K train, 3K test
- Template keywords (SpokenTutorial boilerplate) filtered; exact-duplicate text removed

### Training config (v2)
- Batch 4, LR 1e-3 cosine, warmup 100, weight decay 0.01
- Forced prefix: `[BOS=50257, lang_token, task=50359, no_timestamps=50363]` — **BOS, not START** (see bug below)
- Scheduled sampling: epochs 3-13, prob 0.15 → 0 linear
- Runtime ~1.3-1.6 steps/s on MPS; ~30-35h for 10 epochs

### Critical bugs found & fixed

1. **BOS vs START mismatch (2026-07-28)** — First v2 run used START (50258) as training's first token, but `generate()` starts with BOS (50257). Result: **124% WER** (repetitive garbage). Fixed: forced prefix uses BOS (50257). Retrained from scratch.

2. **`load_state()` reset step to 0 (2026-07-31)** — On resume, step always reset to 0 → mid-epoch progress lost. Removed the reset; resume now continues from exact step.

3. **Resume loaded only `_best.pt` adapters (2026-07-31)** — Languages mid-first-epoch had no `_best.pt` yet (only `_ep{ep}_last.pt`); on restart their LoRA would reinit to zeros, silently losing all training. Fixed: fall back to `_last.pt` for current epoch.

4. **jiwer 4.0 silent API change (2026-07-31)** — `jiwer.wer()` is now case-sensitive with NO normalization, and `wer_standardize()` is broken (`Compose.__call__` TypeError). First evals used raw `wer()`: vanilla model output mixed-case/punctuated text vs uppercase references → reported 98.5% (artifact, not real). Fixed: custom normalized WER (lowercase, strip punctuation, DP edit distance).

### v2 Training run (2026-07-30 to 2026-07-31)
- Resumed after BOS fix; ran 8 of 10 epochs (paused at epoch 8, Hindi, step 200/4896)
- **Best val losses: en 0.368, hi 0.428, hinglish 0.433** — all set in epochs 1-6; plateaued after
- Training abandoned at epoch 8/10: post-plateau epochs yield no val improvement (~8h saved)

---

## Phase 8: v2 Evaluation Findings (2026-07-31)

### Final v2 WERs (200 samples/lang, beam 5, normalized WER)

| Language | LoRA v2 | Vanilla Base | Verdict |
|----------|---------|--------------|---------|
| English | 44.9% | **5.6%** | LoRA destroyed English (overfit to 10K samples) |
| Hindi | **44.8%** | 114.7% | LoRA rescued Hindi (vanilla transliterates to roman text) |
| Hinglish | **50.0%** | 141.8% | LoRA rescued Hinglish |
| **Avg** | 46.6% | 87.4% | |

### Key insights
1. **Vanilla Whisper Base is near-perfect on English (5.6%)** — English LoRA is unnecessary and harmful. The en adapter overfit to 10K LibriSpeech samples at LR 1e-3, destroying the pretrained English ability.
2. **Vanilla Whisper Base catastrophically fails on Hindi/Hinglish (115-142% WER)** — outputs romanized/transliterated garbage instead of Devanagari. This is the real problem space for Indic ASR.
3. **LoRA on the decoder fixes script + language** — 2.5-2.8x WER improvement for Indian languages. Devanagari output restored.
4. **Plateau by epoch 6** — decoder-only LoRA on Base hit its ceiling; val loss flat despite training loss decreasing.

### Hybrid model (production baseline)
Use vanilla for English + LoRA for hi/hinglish → **en 5.6%, hi 44.8%, hinglish 50.0%, avg 33.5%** (confirmed by `eval_v2_hybrid.py`).

---

## Phase 9: v3 Plan — Product + Research (2026-07-31)

### Track A — Product baseline (~2-3 days)
| Step | Work | Target |
|------|------|--------|
| A3 | CER metric + per-sample JSON in evals | Devanagari-friendly metrics |
| A1 | English LoRA retrain: full LibriSpeech 100h (28.5K), LR 1e-4, 3 epochs, SS from ep 1 | en WER <10% with LoRA |
| A2 | Encoder LoRA ablation (hi): add encoder attn LoRA, 2 epochs | beat 44.8% |
| A4 | On-device packaging: CoreML/ONNX export + lang-detect wrapper | mobile deployable |

### Track B — Research: token-level code-switch router (~1-2 weeks)
The hardest unsolved Indic ASR problem: Hinglish code-switches mid-sentence; one "hi" adapter handles everything today.

| Step | Work |
|------|------|
| B1 | Script-detect per-token en/hi labels on 42K hinglish (Devanagari→hi, Latin→en) — confirmed viable, MUCS is script-mixed |
| B2 | Router MLP on decoder hidden states → per-token soft selection of en/hi LoRA experts (~100K params, MoE-lite) |
| B3 | Phase 1: freeze adapters, train router on token labels. Phase 2: joint fine-tune |
| B4 | Eval: hinglish WER vs 50.0%, token-label agreement, script robustness |

**Sequencing**: A3 (15 min) → A1 kicks off (~5h unattended) → B1 + A2 parallel → B2/B3 → A4.

**Artifacts**: v2 adapters frozen as baseline in `adapters_v2/` (on external SSD via symlink `polywhisper_output -> /Volumes/KIOXIA 1TB/polywhisper_output`). New runs write to `adapters_v3/`.

---

## Phase 10: Track B — Router Implementation & Validation (2026-07-31)

### Architecture (B2, implemented in `train_router.py`)
```
decoder embed_tokens output (B, T, 512)
            ↓ forward hook
        Router MLP (512 → 64 → 2, softmax)
            ↓ per-token weights w_en, w_hi
per-projection hook: out = base(x) + w_en·Δen(x) + w_hi·Δhi(x)
```
- Whisper Base frozen; two LoRA rank-16 experts (en, hi) active **simultaneously**, mixed per token
- Router weights computed by an embed_tokens hook → identical behavior in teacher-forced training AND beam-search generation
- Phase 1 (router only): adapters frozen, CE on per-token language labels; Phase 2 (joint): adapters + router, ASR loss + λ·router loss
- Labels: `build_codeswitch_labels.py` → script detection (Devanagari→hi, Latin→en), 84.8% train / 80.3% test code-switched

### Bugs found & fixed during validation
1. **Router hook init order (2026-07-31)** — `_install_expert_hooks()` ran in `__init__` before `add_language()` → `KeyError: 'en'`. Fixed: call hooks after adding languages.
2. **Cross-attention k/v shape mismatch (2026-07-31)** — cross-attn `k_proj`/`v_proj` receive *encoder* states (1500 positions), not decoder tokens (255) → per-token weights crashed (`tensor a (255) must match tensor b (1500)`). Fixed: sequence-level mix (mean over decoder positions, per-sample scalar) for encoder-state projections; per-token mixing for decoder-state projections.
3. **Label alignment failed 0/200 (2026-07-31)** — word-by-word tokenization ≠ context tokenization (BPE splits chars mid-token; byte-escaped mojibake in slow tokenizer). Fixed: align via `WhisperTokenizerFast` `return_offsets_mapping` (verified id-identical to slow tokenizer 300/300), map char spans → word spans → labels. Now 200/200.
4. **Import-time side effects (2026-07-31)** — training loop executed on import. Fixed: wrapped in `if __name__ == "__main__":`; `align_labels` moved to module scope for testing.

### Validation results (smoke tests)
- Forward pass: logits (1, 4, 51865), router weights (1, 4, 2), softmax ~0.5/0.5 pre-training ✓
- Phase-1 step: router CE 0.6933 (random-init baseline) ✓
- Beam-search generation (num_beams=4) with mixing hooks ✓

### Test infrastructure
- `eval_router.py` (B4): hinglish test WER/CER + router token accuracy + per-sample JSON (`eval_router_samples.json`), args for router/adapter paths
- B2/B3 training awaits A1's `adapters_v3/en_best.pt` (router phase 1 uses final v3 en expert; hi expert = v2 `hi_best.pt`)

---

## Current Status

✅ **Phase 1-4 complete** — v1 architecture, data, training (30.2h, 55K steps)
✅ **Phase 5 partial** — v1 eval showed architecture works, needs training
✅ **Phase 6 complete** — label-start bug fixed, scheduled sampling added
✅ **Phase 7 complete** — v2 rewrite (LoRA in Whisper Base decoder), BOS fix, resume fixes; 8/10 epochs trained; best losses en 0.368 / hi 0.428 / hinglish 0.433
✅ **Phase 8 complete** — v2 eval: hybrid = **33.5% avg WER** (en vanilla 5.6%, hi LoRA 44.8%, hinglish LoRA 50.0%)
🔄 **Phase 9 in progress** — v3: Track A (product) + Track B (code-switch router research)
💾 **All training saved** — adapters on external SSD (`polywhisper_output/adapters_v2/`), per-epoch backups

## What's Next

1. **A1 (running)**: English LoRA retrain, full LibriSpeech 28.5K, LR 1e-4 → download ~85% → train ~5h → target <10% en WER
2. **B2/B3**: code-switch router training (phase 1, ~6h) — blocked on A1 `en_best.pt`
3. **A2**: encoder LoRA ablation for hi (needs MPS slot after A1/router)
4. **B4**: router eval (`eval_router.py`) vs hinglish 50.0% baseline
5. **A4**: on-device packaging (coremltools/onnx not installed)
6. **A4**: on-device CoreML/ONNX packaging
