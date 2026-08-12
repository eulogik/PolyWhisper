# Reddit r/MachineLearning post — paste as text submission

---

**[P] PolyWhisper — Hinglish ASR with per-token LoRA routing (+7.8 WER over vanilla Whisper-Base, MIT, trained on a Mac Mini)**

We built a code-switching (Hindi–English) ASR system: frozen Whisper-Base encoder + two
rank-8 LoRA experts (EN/HI) + a 33K-parameter router that gates per token.

**Results** (3,129-utterance code-switched Hinglish test, ortho-normalized, greedy decode):

| System | WER | Hallucinations (run≥4) |
|---|---|---|
| PolyWhisper v5 | **58.8%** | **13** |
| Vanilla Whisper-Base | 66.6% | 279 |
| Static 50/50 mix | 72.1% | 655 |

- Routing adds **+13.3 WER pts** over the static 50/50 mix (the ablation that matters)
- Router per-token language accuracy: **89.1%**
- Trains in ~a day on 16GB M4 Mac Mini (MPS)

**The interesting bit — a real gotcha we hit:** our first "working" router was silently broken.
The collator compared int labels to the string "en" → every token's label collapsed to one
class → the router learned to route *everything* to the EN expert. WER still improved (because
the EN expert was a good full-utterance adapter), so the metric looked fine. A router-weight
introspection script exposed it. After fixing labels and retraining, the static-50/50 ablation
finally showed real routing value.

Takeaway: for MoE-style routing, ablations against static mixing + weight introspection are
the only honest validation.

Repo: https://github.com/eulogik/PolyWhisper (MIT)
Model: https://huggingface.co/eulogik/polywhisper-hinglish-router
Full per-utterance eval JSONs in the repo for independent re-scoring.

By **Eulogik**. Happy to discuss limitations — absolute WER is far from large-model SOTA;
the contribution is parameter-efficient routing for code-switched low-resource speech.
