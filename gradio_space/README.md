---
title: PolyWhisper
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
license: mit
---

# PolyWhisper — Multilingual Indic ASR

Efficient speech recognition for **Hindi, Tamil, Telugu, Bengali, and Marathi** using frozen Whisper Small + per-language LoRA adapters.

## Features

- 🎤 Upload audio or record from microphone
- 🌐 5 Indian languages with auto-selection
- 📋 Copy-friendly text output
- 🎬 SRT subtitle export
- ⚡ ~3-5s per clip on CPU

## Benchmark

| Language | WER (FLEURS) |
|----------|-------------|
| Hindi | **46.3** |
| Tamil | **70.1** |
| Telugu | 100.1 |
| Bengali | 130.2 |
| Marathi | 96.7 |

## Model

- **Backbone**: Whisper Small (244M params, frozen)
- **Adapters**: LoRA rank 16 (~14MB per language)
- **Total**: 244M + 3.5M adapter = ~248M active params

## Links

- [Model Card](https://huggingface.co/eulogik/polywhisper)
- [GitHub](https://github.com/eulogik/PolyWhisper)
- [Training Data](https://huggingface.co/datasets/ai4bharat/indicvoices-st)
