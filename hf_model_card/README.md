---
language:
- hi
- en
license: mit
library_name: pytorch
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- hinglish
- code-switching
- hindi
- english
- asr
- speech-recognition
- lora
- mixture-of-experts
- router
- low-resource
- edge-asr
- devanagari
- indic
metrics:
- wer
- cer
- fuzzy-wer
model-index:
- name: polywhisper-hinglish-router
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: PolyWhisper Hinglish Test (MUCS/IndicVoices-ST derived)
      type: eulogik/polywhisper-hinglish
      split: test
      args: 3129 utterances, ortho-normalized
    metrics:
    - type: wer
      value: 58.8
      name: WER
    - type: cer
      value: 57.9
      name: CER
    - type: fuzzy-wer
      value: 57.3
      name: FuzzyWER
---
# PolyWhisper Hinglish Router

A **code-switch (Hindi–English) speech recognition** model by **[Eulogik](https://huggingface.co/eulogik)** — frozen Whisper-Base encoder + two rank-8 LoRA language experts + a 33K-parameter per-token router.

**This is the v5 release.** The router was trained with corrected per-token language labels, and it is the first PolyWhisper checkpoint where per-token routing is genuinely learned and measured.

## Why it matters

Speakers in India switch between Hindi and English mid-sentence (Hinglish). Single-language ASR models degrade on this. PolyWhisper:

- Routes **every token** to an English-expert or Hindi-expert LoRA,
- Adds **+13.3 WER points over a static 50/50 expert mix** (58.8% vs 72.1%),
- Adds **+7.8 WER points over vanilla Whisper-Base** (66.6%),
- Cuts hallucinated repetition loops **~20×** (13 vs 279 events),
- Trains in ~a day on a 16GB Apple Silicon Mac — no GPU cluster.

## Results (3,129-utterance code-switched test set)

| System | WER | FuzzyWER | CER | Hallucinations |
|---|---|---|---|---|
| **PolyWhisper v5 (this model)** | **58.8%** | **57.3%** | **57.9%** | **13** |
| Vanilla Whisper-Base | 66.6% | 63.3% | 67.5% | 279 |
| Static 50/50 expert mix | 72.1% | 70.4% | 71.1% | 655 |

Router per-token language accuracy: **89.1%** (99,663 / 111,815 tokens).

## Files

| File | Contents |
|---|---|
| `en_router_best_v5.pt` | English LoRA expert (rank-8 decoder adapters) |
| `hi_router_best_v5.pt` | Hindi LoRA expert (rank-8 decoder adapters) |
| `router_best_v5.pt` | Per-token router (33K params) |
| `eval_router_v5_samples.json` | Full 3,129-sample per-utterance results |
| `eval_static5050_v5_samples.json` | Static 50/50 ablation results |
| `eval_vanilla_samples.json` | Vanilla Whisper-Base results |
| `hinglish_codeswitch_test_ortho.json` | Ortho-normalized test set |
| `README.md` | This card |

## Usage

```python
import torch, soundfile as sf
from transformers import WhisperProcessor
from model import PolyWhisperRouter   # see github.com/eulogik/PolyWhisper

model = PolyWhisperRouter().to("mps" if torch.backends.mps.is_available() else "cpu")
model.add_language("en").add_language("hi")
model.load_adapter("en", "en_router_best_v5.pt")
model.load_adapter("hi", "hi_router_best_v5.pt")
model.load_router("router_best_v5.pt")

audio, _ = sf.read("hinglish.wav")
feats = WhisperProcessor.from_pretrained("openai/whisper-base").feature_extractor(
    [audio], sampling_rate=16000, return_tensors="pt").input_features
out = model.generate(feats, max_new_tokens=128, use_cache=False, language="hi", task="transcribe")
```

## Training

- Backbone: `openai/whisper-base` (frozen encoder + frozen base decoder)
- Experts: rank-8 LoRA on decoder cross-attention (K/V) — ~3M params each
- Router: 2-layer MLP over decoder hidden states (33K params)
- Stage A1: EN expert, masked to English tokens (3 epochs, ~9h on M4 MPS)
- Stage A2: HI expert + encoder-LoRA (4 epochs, ~9h)
- Stage P1: router-only language selection (2 epochs)
- Stage P2: joint router + expert adaptation (2 epochs, λ_router=1.0)
- Data: MUCS / IndicVoices-ST derived code-switched Hinglish (~42K train)

## Limitations

- Word error rate is high in absolute terms (~58.8%) — acceptable for edge/low-resource use, not parity with large models
- Tested on Hinglish tutorial-style speech; robustness to spontaneous/overlapping speech untested
- Devanagari orthography variance still inflates WER (see per-sample errors)
- No privacy guarantees; data is public corpora

## License

MIT. © 2026 [Eulogik](https://huggingface.co/eulogik). Whisper is OpenAI's model. Derived data from MUCS (CC-BY-SA) and IndicVoices-ST (CC-BY) — see the [GitHub repo](https://github.com/eulogik/PolyWhisper) for attribution.

## Citation

```bibtex
@misc{eulogik2026polywhisper,
  title={PolyWhisper: Code-Switch ASR with Per-Token LoRA Routing for Hinglish},
  author={Eulogik},
  year={2026},
  howpublished={\url{https://huggingface.co/eulogik/polywhisper-hinglish-router}},
  note={MIT licensed; benchmarked on 3,129-utterance code-switched test set}
}
```
