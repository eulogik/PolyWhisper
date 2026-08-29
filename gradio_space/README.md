---
title: PolyWhisper
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
license: apache-2.0
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
| Hindi | **37.2** |
| Tamil | **60.4** |
| Telugu | 105.7 |
| Bengali | 196.3 |
| Marathi | 167.3 |

## Model

- **Backbone**: Whisper Small (244M params, frozen)
- **Adapters**: LoRA rank 16 (~14MB per language)
- **Total**: 244M + 3.5M adapter = ~248M active params

## Links

- [Model Card](https://huggingface.co/eulogik/polywhisper)
- [GitHub](https://github.com/eulogikdeveloper/PolyWhisper)
- [Training Data](https://huggingface.co/datasets/ai4bharat/indicvoices-st)
