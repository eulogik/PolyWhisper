# PolyWhisper — Code-Switch ASR with LoRA Routing (Hinglish)

> **PolyWhisper** is a tiny, edge-ready speech recognition system that turns a frozen Whisper-Base encoder into a **code-switching Hindi–English (Hinglish) transcriber** using **language-expert LoRA adapters** and a **per-token Mixture-of-Experts router**. It trains on a single Mac Mini, and beats vanilla Whisper-Base by **+7.8 WER points** while cutting hallucinated repetition by **~20×**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Org: eulogik](https://img.shields.io/badge/org-eulogik-181717.svg)](https://github.com/eulogik)
[![Hugging Face: eulogik](https://img.shields.io/badge/HuggingFace-eulogik-yellow.svg)](https://huggingface.co/eulogik)
[![Model: whisper-base](https://img.shields.io/badge/backbone-whisper--base-yellow.svg)](https://huggingface.co/openai/whisper-base)
[![Contributors](https://img.shields.io/github/contributors/eulogik/PolyWhisper.svg)](https://github.com/eulogik/PolyWhisper/graphs/contributors)

---

## ⚡ TL;DR — Why PolyWhisper

- **Code-switch ASR done right** — a router that picks *per token* between an English LoRA expert and a Hindi LoRA expert, instead of one model trying to do both.
- **+13.3 WER pts over a static 50/50 expert mix** — proves per-token routing is *real*, not decoration.
- **+7.8 WER pts over vanilla Whisper-Base** on the same 3,129-utterance code-switched test set.
- **~20× fewer hallucinations** — 13 vs 279 repetitive-loop errors vs vanilla.
- **Trains in hours on a Mac Mini (M4, 16GB, MPS)** — no GPU cluster required.
- **Fully open**: MIT License, weights on Hugging Face, code on GitHub — by **[Eulogik](https://github.com/eulogik)**.

| System | WER ↓ | FuzzyWER ↓ | CER ↓ | Hallucination events |
|---|---|---|---|---|
| **PolyWhisper (v5 router)** 🏆 | **58.8%** | **57.3%** | **57.9%** | **13** |
| Vanilla Whisper-Base | 66.6% | 63.3% | 67.5% | 279 |
| Static 50/50 expert mix | 72.1% | 70.4% | 71.1% | 655 |

<sub>Full 3,129-utterance code-switched Hinglish test set, greedy decoding, `language="hi"` prefix.</sub>

---

## 🧠 What problem does it solve?

Real-world speech in India is **Hinglish** — Hindi and English mixed mid-sentence:

> *"इस tutorial में हम impress window के भागों के बारे में सीखेंगे"*
> ("In this tutorial we learn about the parts of the Impress window")

Whisper-Base, tuned for clean English, stumbles on this: it **hallucinates loops**, **drops English words**, and **mangles Devanagari**. Full Whisper fine-tuning on Hinglish is expensive and still leaves a single model trying to reconcile two very different token-spaces.

PolyWhisper's answer: **keep the shared Whisper encoder frozen, attach two tiny LoRA experts (one per language), and let a learned router decide at every token which expert should predict the next word.**

---

## 🏗️ Architecture

```
        Audio (16 kHz)
             │
             ▼
   ┌──────────────────────┐
   │  Whisper-Base Encoder │   frozen (74M)
   │  (shared, frozen)     │
   └──────────┬───────────┘
              │  encoder hidden states
              ▼
   ┌──────────────────────┐
   │   PolyWhisper Router  │   33K params — per-token expert gating
   │  (2-layer MLP)        │
   └──────┬───────────┬───┘
          │ w_en      │ w_hi
          ▼           ▼
   ┌──────────┐  ┌──────────┐
   │  EN LoRA  │  │  HI LoRA  │   ~3M params each (rank-8 decoder adapters)
   │  expert   │  │  expert   │
   └─────┬────┘  └─────┬────┘
         └──────┬──────┘
                ▼
        Decoder (Whisper-Base, frozen)
                ▼
         Hinglish text out
```

- **Encoders stay frozen** — only the LoRA adapters (~3M) and router (33K) are trained.
- **Training is two-phase**: (1) train each expert on language-masked data; (2) joint phase where the router learns language selection *and* adapters adapt to being routed.
- **Hallucination control** — routing stops the decoder from collapsing into one language's repetitive loop.

---

## 🚀 Quickstart

```bash
pip install torch transformers soundfile
```

```python
import torch
from transformers import WhisperProcessor
from model import PolyWhisperRouter          # from this repo

model = PolyWhisperRouter().to("mps" if torch.backends.mps.is_available() else "cpu")
model.add_language("en").add_language("hi")
model.load_adapter("en", "en_router_best_v5.pt")
model.load_adapter("hi", "hi_router_best_v5.pt")
model.load_router("router_best_v5.pt")

audio, sr = sf.read("hinglish_sample.wav")
feats = processor.feature_extractor([audio], sampling_rate=16000, return_tensors="pt").input_features
out = model.generate(feats, max_new_tokens=128, use_cache=False, language="hi", task="transcribe")
print(processor.decode(out[0], skip_special_tokens=True).strip())
```

> 💾 Weights: `eulogik/polywhisper-hinglish-router` on [Hugging Face](https://huggingface.co/eulogik) · code: [github.com/eulogik/PolyWhisper](https://github.com/eulogik/PolyWhisper)

---

## 📊 Benchmarks (reproducible)

- **Test set**: 3,129 code-switched Hinglish utterances (MUCS / IndicVoices-ST derived, Devanagari+Latin orthography, ortho-normalized).
- **Protocol**: greedy (beam=1), 128 max tokens, `language="hi"` prefix, `use_cache=False` — the honest, memory-fair setting.
- **Per-sample WER/`FuzzyWER`/CER** are logged to `results/` with the full 3,129-sample output for independent re-scoring.

| System | WER | FuzzyWER | CER | router token-acc |
|---|---|---|---|---|
| **PolyWhisper v5** | **58.8%** | **57.3%** | **57.9%** | **89.1%** |
| Vanilla Whisper-Base | 66.6% | 63.3% | 67.5% | — |
| Static 50/50 mix | 72.1% | 70.4% | 71.1% | — |

**By code-switch intensity** (word-level):

| System | no CS | CS utterances | routing value |
|---|---|---|---|
| PolyWhisper v5 | 63.4% | **58.3%** | — |
| Vanilla | 68.4% | 66.4% | — |
| Static 50/50 | 74.8% | 71.8% | +13.3 pts |

**Hallucination audit** (repeated 3-grams / ≥4-word loops / empty outputs / length blowups):

| System | rep-3gram | run≥4 | empty | length-blowup |
|---|---|---|---|---|
| PolyWhisper v5 | 0 | **13** | 0 | 1 |
| Vanilla | 0 | 279 | 0 | 21 |
| Static 50/50 | 0 | 655 | 0 | 47 |

---

## 🧪 Reproduce the pipeline

```bash
# 1. Experts (per-language masked training)
.venv/bin/python train_v3.py --langs en --mask-lang en --epochs 3 --tag _v5
.venv/bin/python train_v3.py --langs hi --encoder-lora --epochs 4 --tag _v5

# 2. Router phase 1 (learn language selection) + phase 2 (joint)
.venv/bin/python train_router.py --phase 1 --epochs 2 --tag _v5 \
    --en-adapter polywhisper_output/adapters_v3/en_best_v5.pt \
    --hi-adapter polywhisper_output/adapters_v3/hi_best_v5.pt
.venv/bin/python train_router.py --phase 2 --epochs 2 --lr 1e-4 --lambda-router 1.0 --tag _v5 \
    --router-init polywhisper_output/adapters_v3/router_best_v5.pt

# 3. Evaluate
.venv/bin/python eval_router.py --router polywhisper_output/adapters_v3/router_best_v5.pt \
    --en-adapter polywhisper_output/adapters_v3/en_router_best_v5.pt \
    --hi-adapter polywhisper_output/adapters_v3/hi_router_best_v5.pt
```

The whole v5 pipeline is orchestrated by `watch_and_launch_router_v5.py` (crash-safe, resumable, auto-eval).

---

## ❓ FAQ

**What is Hinglish / code-switching?**
Hinglish is the Hindi–English mix spoken daily in India (e.g., "मैं file खोल रहा हूँ"). Code-switching means speakers switch language mid-sentence — the core challenge PolyWhisper tackles.

**How is this different from fine-tuning Whisper on Hinglish?**
PolyWhisper never retrains the 74M-parameter Whisper model. It adds ~6M parameters of LoRA experts and a 33K router, which can be trained on a 16GB Mac in ~a day. Fine-tuning Whisper needs GPUs and produces a single-language-blob model.

**What does "+7.8 WER points" mean?**
On the identical 3,129-utterance test set, PolyWhisper's word error rate is 58.8% vs 66.6% for vanilla Whisper-Base — that's a ~12% relative reduction in errors, with ~20× fewer hallucinations.

**Does it run on a Mac / CPU?**
Yes — training runs on Apple Silicon MPS (M4 16GB used here); inference is plain PyTorch.

**What's the license?**
MIT — code, weights, and results are free to use commercially. Made by [Eulogik](https://github.com/eulogik).

**What's next?**
More language pairs (e.g., Tamil–English), orthography normalization for Devanagari, and bigger Hindi-expert capacity. See [Roadmap](#roadmap).

---

## 🗺️ Roadmap

- [x] Hinglish MVP — v5 router beats vanilla +7.8 WER, static-mix +13.3 WER
- [x] Per-token routing validated (router token accuracy 89.1%)
- [ ] Whisper-small / medium / large baselines
- [ ] Second language pair (Tamil–English, Bengali–English)
- [ ] Orthography-normalized training references
- [ ] Larger Hindi expert (52K cleaned hours, rank-32)
- [ ] ONNX / CoreML edge export

---

## 📜 License & Attribution

MIT License. © 2026 **[Eulogik](https://github.com/eulogik)** — [Eulogik](https://huggingface.co/eulogik) on Hugging Face.

Data derived from MUCS / IndicVoices-ST (see [LIVING.md](LIVING.md) for sources & attribution). Whisper is OpenAI's model. The PolyWhisper code, adapter weights, router, and evaluation results are original work by Eulogik.

---

## 📚 Docs

- [LIVING.md](LIVING.md) — full engineering walkthrough & decisions
- [PolyWhisper-Project-Document.md](PolyWhisper-Project-Document.md) — project spec
- [results/](results/) — all evaluation JSONs (per-sample, rescorable)
- [Hugging Face — eulogik/polywhisper-hinglish-router](https://huggingface.co/eulogik/polywhisper-hinglish-router)
