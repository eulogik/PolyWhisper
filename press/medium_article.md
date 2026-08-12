# PolyWhisper: Hinglish ASR That Routed Its Way Past Whisper-Base — by Eulogik

*Code-switching speech recognition with per-token LoRA routing, trained on a Mac Mini.*

## TL;DR

We built **PolyWhisper** — a Hinglish (Hindi-English code-switched) speech recognition
system on top of a frozen Whisper-Base encoder, using two tiny LoRA language experts and a
**33K-parameter router that decides per token which expert predicts the next word**.

Result (3,129-utterance code-switched test set):

- **+7.8 WER points vs vanilla Whisper-Base** (58.8% vs 66.6%)
- **+13.3 WER points vs a static 50/50 expert mix** (72.1%) — the routing itself earns the gain
- **~20× fewer hallucinations** (13 vs 279 repetitive-loop errors)
- Trains in ~a day on a 16GB Apple Silicon Mac — no GPU cluster, MIT licensed

## The problem

India's most-spoken "language" is actually two languages at once: **Hinglish**.

> "इस tutorial में हम impress window के भागों के बारे में सीखेंगे"

Vanilla Whisper-Base — trained on clean English — breaks down here: it hallucinates
repetitive loops, drops English words, and mangles Devanagari. Fine-tuning the whole model
is expensive and forces one model to reconcile two token-spaces.

## The idea

Keep the shared encoder frozen. Attach **one LoRA expert per language** (English, Hindi).
Add a **router** that reads decoder state and gates, at every token, between experts.

## What actually happened (the honest story)

Building this taught us a lesson worth sharing: **our first "working" router was broken.**

The label collator compared int labels against the string `"en"` — always false — so every
token collapsed to one class. The v4 router silently routed *everything* through the English
expert. Our +17-point routing claim was, in effect, "en-expert-only" — routing was never
tested. A routing-introspection script caught it.

We fixed the labels, retrained (v5), and *then* measured real routing:

| System | WER | Hallucinations |
|---|---|---|
| **PolyWhisper v5** | **58.8%** | **13** |
| Vanilla Whisper-Base | 66.6% | 279 |
| Static 50/50 mix | 72.1% | 655 |

Router per-token language accuracy: **89.1%**. And the ablation — static 50/50 mix — proves
the router's selection is what buys the +13.3 points.

## Why it matters

- **Edge/low-resource ASR**: trains and runs on consumer hardware
- **Code-switching** is the norm in most of the world, not the exception
- **Reproducible**: full per-utterance results published, rescorable
- **Open**: MIT, weights on HF, code on GitHub

## What's next

Whisper-small/medium/large baselines, more language pairs (Tamil-English, Bengali-English),
orthography normalization, bigger Hindi expert capacity.

## Links

- GitHub: https://github.com/eulogik/PolyWhisper
- Model: https://huggingface.co/eulogik/polywhisper-hinglish-router
- By **Eulogik** — https://github.com/eulogik · https://huggingface.co/eulogik

*MIT License. Whisper is OpenAI's model; Hinglish data derived from MUCS (CC-BY-SA) and IndicVoices-ST (CC-BY) corpora.*
