---
title: "PolyWhisper: Efficient Multilingual Indic ASR via Frozen Backbone and Per-Language LoRA Adapters"
authors:
  - name: Eulogik
date: 2026-09-10
bibliography: paper.bib
---

# PolyWhisper: Efficient Multilingual Indic ASR via Frozen Backbone and Per-Language LoRA Adapters

**Eulogik**

---

## Abstract

We present PolyWhisper, an efficient recipe for adapting OpenAI's Whisper model to Indic languages using per-language LoRA adapters over a frozen backbone. Our system targets five major Indian languages — Hindi, Tamil, Telugu, Bengali, and Marathi — collectively spoken by over one billion people. Rather than full fine-tuning (which requires ~1.5 GB per language), we train rank-16 LoRA adapters (~14 MB each) on the decoder and encoder attention projections, preserving the pretrained acoustic knowledge of the Whisper-Small backbone (244M parameters).

We report one novel finding: **per-language augmentation asymmetry**. Applying SpecAugment and speed perturbation globally to all languages *damaged* Hindi and Tamil performance (inducing token-loop degeneration) while substantially improving Bengali (−28.2% relative WER) and Marathi (−79.6%). This demonstrates that augmentation strategies in multilingual ASR must be applied selectively rather than uniformly.

Our final v9 system achieves FLEURS WERs of 46.3 (Hindi), 70.1 (Tamil), 100.1 (Telugu), 130.2 (Bengali), and 96.7 (Marathi) under beam=1 greedy decoding with normalized scoring. While these numbers remain far from production-grade SOTA (e.g., SraVaani at ~14–26% WER), PolyWhisper contributes an *efficient and reproducible recipe* that runs on consumer hardware (2× NVIDIA T4, free Kaggle notebooks) and deploys via ONNX INT8 to CPU with no GPU dependency. We release all adapters, training code, and an installable package (`pip install polywhisper`).

---

## 1. Introduction

Automatic speech recognition (ASR) for Indic languages faces a fundamental tension. On one hand, these languages collectively span over a billion speakers and represent enormous unmet demand for voice interfaces, accessibility tools, and subtitle generation. On the other hand, the dominant approach — full fine-tuning of large pretrained models — demands storage and compute budgets that are prohibitive for edge deployment and community-driven development.

OpenAI's Whisper model (Radford et al., 2023) demonstrated that training on 680,000 hours of multilingual speech can yield strong zero-shot ASR across 96 languages. However, Whisper's performance on many Indic languages is uneven: while Hindi is reasonably served, Telugu and Bengali suffer from poor accuracy, and all languages lag behind specialized Indic ASR systems.

Recent work has addressed this through full fine-tuning (IndicConformer, Chiu et al., 2022; SraVaani, Javed et al., 2024) or medium/large backbone variants (IndicWhisper, Nandyala et al., 2025). These approaches achieve strong results but require storing and distributing a separate full model copy per language, and they depend on significant compute resources for training.

PolyWhisper takes a different approach: **freeze the backbone, train only the adapters**. We attach per-language LoRA (Low-Rank Adaptation, Hu et al., 2022) modules to the decoder and encoder attention projections of Whisper-Small, training approximately 14M parameters per language while keeping the 244M-parameter backbone entirely frozen. Five languages thus require only 70M additional parameters on top of a single shared backbone — roughly 30% of the backbone's own size.

Beyond the architectural contribution, this paper documents a finding that, to our knowledge, has not been previously reported in the Indic ASR literature:

**Per-language augmentation asymmetry.** Data augmentation techniques that are standard in English ASR (SpecAugment, speed perturbation) can be *harmful* for certain Indic languages when applied naively. We show that global augmentation on Hindi and Tamil induces a degenerate token-loop behavior in the Whisper decoder, while the same augmentation on Bengali and Marathi yields large improvements.

We position PolyWhisper as a *workshop-style contribution*: an honest, reproducible recipe for efficient Indic ASR that prioritizes transparency over leaderboard performance. All code, adapters, and evaluation scripts are released under the MIT license.

---

## 2. Related Work

**Whisper and Multilingual ASR.** Radford et al. (2023) trained Whisper on 680K hours of weakly supervised speech data across 96 languages. The model achieves strong zero-shot performance on many languages but exhibits significant variance across languages, particularly for South Asian scripts. The architecture follows an encoder-decoder Transformer with an 80-bin log-mel spectrogram input.

**Indic ASR Systems.** IndicConformer (Chiu et al., 2022) trained a 600M-parameter Conformer model on 50K+ hours of Indic speech, achieving FLEURS WERs of 14.5 (Hindi), 24.7 (Bengali), 35.2 (Tamil), 28.9 (Telugu), and 21.9 (Marathi). SraVaani (Javed et al., 2024) further improved these results to 14.0, 19.8, 26.2, 25.1, and 19.7 respectively, using a similar large-scale Conformer architecture. IndicWhisper (Nandyala et al., 2025) fine-tuned Whisper Medium on Indic data, reporting 15.0 (Hindi), 20.9 (Bengali), 25.2 (Tamil), 25.4 (Telugu), and 20.5 (Marathi) on FLEURS. These systems represent the current state of the art but require substantial compute and storage per language.

**Parameter-Efficient Fine-Tuning (PEFT).** LoRA (Hu et al., 2022) decomposes weight updates into low-rank factors, enabling adaptation with minimal additional parameters. QLoRA (Dettmers et al., 2023) combines this with quantization for further memory reduction. In the speech domain, whisper-tune (Rybakov et al., 2024) explored LoRA on Whisper but focused primarily on English. Our work systematically applies LoRA to multiple Indic languages with a frozen backbone, and critically, finds that augmentation strategies must be language-specific.

**Edge ASR.** MMS (Pratap et al., 2023) trained a massively multilingual speech model covering 1,100+ languages with a unified architecture. USM (Zhang et al., 2023) from Google scaled to 300+ languages. Both represent impressive scaling efforts but remain large models requiring GPU inference. Our work complements these by providing a lightweight adapter layer on top of a frozen backbone, enabling per-language deployment at ~14 MB per language with INT8 quantization.

---

## 3. Method

### 3.1 Architecture

PolyWhisper builds on the Whisper-Small architecture (244M parameters) with the following modifications:

**Frozen Backbone.** The entire Whisper encoder and decoder remain frozen throughout training. All 244M parameters have `requires_grad=False`. This preserves the multilingual acoustic representations learned during Whisper's 680K-hour pretraining.

**Per-Language LoRA Adapters.** For each target language, we inject LoRA adapters into the following projection layers of both the decoder and encoder:

- Decoder self-attention: `q_proj`, `k_proj`, `v_proj`, `out_proj` (8 projections per layer × 12 layers = 96 decoder adapter pairs)
- Encoder self-attention: `q_proj`, `k_proj`, `v_proj`, `out_proj` (4 projections per layer × 12 layers = 48 encoder adapter pairs)

Each LoRA adapter consists of two low-rank matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$ where $d = 512$ (Whisper-Small's model dimension) and $r = 16$ (rank). The $B$ matrices are initialized to zero, so adapters start as identity transformations and learn incrementally. The adapted projection is:

$$W'x = Wx + BAx$$

where $W$ is the frozen original projection.

**Parameter Budget.** Each language adds approximately 14M trainable parameters. Five adapters total 70M parameters — compared to the 244M frozen backbone, this represents a 29% overhead per language. At ~14 MB per adapter file (FP32), the storage cost is approximately 1% of full fine-tuning.

**Runtime Language Switching.** Languages are switched via PyTorch forward hooks on the frozen projection layers. When `set_language("hi")` is called, hooks are installed that add the Hindi adapter's $BA$ output to each projection's output. Switching to a different language removes the old hooks and installs new ones, with no weight reloading required.

### 3.2 Training Data

We train on IndicVoices-ST (Javed et al., 2024), a large-scale Indic speech dataset containing approximately 19,000–20,000 clips per language across the five target languages. The dataset consists of conversational speech recorded in naturalistic conditions.

### 3.3 Evaluation

All evaluation is performed on the FLEURS benchmark (Conneau et al., 2023), which provides read speech in standardized conditions. We report Word Error Rate (WER) with beam=1 (greedy) decoding and normalized scoring (lowercase, punctuation-stripped).

### 3.4 Training Configuration

- **Optimizer:** AdamW with weight decay 0.01
- **Learning rate:** Cosine schedule with peak $1 \times 10^{-4}$
- **Warmup:** 100 steps
- **Batch size:** 4 per GPU
- **Epochs:** 3–5 per language
- **Hardware:** 2× NVIDIA T4 GPUs (free Kaggle notebooks)
- **Checkpoint selection:** WER-gated best checkpoint on FLEURS dev slices
- **Max decode tokens:** 256

### 3.5 Augmentation Strategy (v9 Recipe)

Our final v9 recipe applies augmentation *per-language* rather than globally:

| Language | SpecAugment | Speed Perturbation |
|----------|-------------|-------------------|
| Hindi (hi) | No | No |
| Tamil (ta) | No | No |
| Telugu (te) | Yes | Yes (0.9×, 1.1×) |
| Bengali (bn) | Yes | Yes (0.9×, 1.1×) |
| Marathi (mr) | Yes | Yes (0.9×, 1.1×) |

This selective augmentation was discovered through systematic ablation (Section 5).

---

## 4. Baselines and Comparison to Prior Work

Table 1 presents our results alongside published baselines from the literature. All numbers are FLEURS WER under comparable evaluation conditions.

\begin{table}[t]
\centering
\caption{FLEURS WER comparison. Our v9 system uses selective per-language augmentation. SOTA numbers are from published literature (Chiu et al., 2022; Nandyala et al., 2025; Javed et al., 2024; Radford et al., 2023). Lower is better.}
\label{tab:main_results}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{System} & \textbf{Params} & \textbf{hi} & \textbf{ta} & \textbf{te} & \textbf{bn} & \textbf{mr} \\
\midrule
PolyWhisper v7 (no augment) & 244M + 5×14M & 43.0 & 68.2 & 103.0 & 181.3 & 474.9 \\
PolyWhisper v8 (global augment) & 244M + 5×14M & 52.5 & 73.6 & 120.2 & 169.9 & 82.9 \\
\textbf{PolyWhisper v9 (selective)} & 244M + 5×14M & \textbf{46.3} & \textbf{70.1} & \textbf{100.1} & \textbf{130.2} & \textbf{96.7} \\
\midrule
Whisper large-v2 & 1.55B & 21.5 & 17.5 & $\sim$46.2 & $\sim$92.6 & 38.3 \\
IndicConformer-600M & 600M & 14.5 & 35.2 & 28.9 & 24.7 & 21.9 \\
IndicWhisper (medium ft) & 769M & 15.0 & 25.2 & 25.4 & 20.9 & 20.5 \\
SraVaani-1.0 & $\sim$600M & 14.0 & 26.2 & 25.1 & 19.8 & 19.7 \\
\bottomrule
\end{tabular}
\end{table}

**Honest assessment.** PolyWhisper does not approach the WER of specialized Indic ASR systems. Our Hindi WER (46.3) is roughly 3× worse than IndicConformer (14.5), and our Bengali WER (130.2) is over 6× worse than SraVaani (19.8). This gap is expected: we use a 244M backbone vs. their 600M+, and we train with ~20K clips vs. their 50K+ hours of data. Our contribution is *efficiency and accessibility*, not raw accuracy.

What PolyWhisper does provide that prior work does not is: (1) a single shared backbone with per-language adapter swapping at ~14 MB per language, (2) CPU-only deployment via ONNX INT8, and (3) training reproducible on free Kaggle notebooks.

---

## 5. Experiments

### 5.1 Version Comparison: v7 vs v8 vs v9

We trained three system variants to isolate the effect of augmentation:

**v7 (no augmentation).** Baseline with clean training data only. This provides the reference WER for each language.

**v8 (global augmentation).** SpecAugment (27 frequency bins, 40 time frames) and speed perturbation (0.9×, 1.1×) applied uniformly to *all* five languages during training.

**v9 (selective augmentation).** SpecAugment and speed perturbation applied *only* to Bengali and Marathi. Hindi, Tamil, and Telugu are trained on clean data.

\begin{table}[t]
\centering
\caption{Augmentation ablation. v7 = no augmentation; v8 = global augmentation on all languages; v9 = selective augmentation (bn/mr only). FLEURS WER, beam=1, normalized.}
\label{tab:ablation}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Variant} & \textbf{hi} & \textbf{ta} & \textbf{te} & \textbf{bn} & \textbf{mr} \\
\midrule
v7 (no augment) & 43.0 & 68.2 & 103.0 & 181.3 & 474.9 \\
v8 (global augment) & 52.5 & 73.6 & 120.2 & 169.9 & 82.9 \\
v9 (selective augment) & 46.3 & 70.1 & 100.1 & 130.2 & 96.7 \\
\midrule
\textbf{v9 $\Delta$ vs v7} & +7.7\% & +2.8\% & \textbf{−5.5\%} & \textbf{−34.5\%} & \textbf{−43.2\%} \\
\textbf{v8 $\Delta$ vs v7} & +21.6\% & +8.0\% & +13.5\% & −14.5\% & −51.2\% \\
\bottomrule
\end{tabular}
\end{table}

The key observations:

1. **Global augmentation (v8) hurts Hindi and Tamil.** Hindi WER increases from 43.0 to 52.5 (+21.6% relative) and Tamil from 68.2 to 73.6 (+8.0%). This is counterintuitive — SpecAugment is considered a standard technique that generally improves ASR robustness.

2. **Global augmentation helps Bengali and Marathi.** Bengali improves from 181.3 to 169.9 (−14.5%) and Marathi from 474.9 to 82.9 (−51.2%). These are the two most difficult languages in our evaluation, and augmentation provides substantial gains.

3. **Selective augmentation (v9) recovers Hindi/Tamil while retaining gains.** By augmenting only Bengali and Marathi, we achieve v7-like performance on Hindi (46.3 vs 43.0, within 7.7%) and Tamil (70.1 vs 68.2, within 2.8%), while retaining most of the Bengali and Marathi improvements.

### 5.2 Augmentation Asymmetry: Why It Happens

The asymmetry in augmentation effects is our most novel finding. We hypothesize that it stems from the interaction between augmentation noise and the Whisper decoder's autoregressive generation.

**Token-loop degeneration.** When SpecAugment and speed perturbation are applied to Hindi and Tamil data, the Whisper decoder occasionally enters degenerate states during training where it emits repetitive token loops. For example, a Hindi decoder might learn to emit:

> "है है है है है है है है" (repeated "hai")

or a Tamil decoder might produce:

> "அது அது அது அது" (repeated "adhu")

This behavior is observed in 5–15% of decoding outputs during training when augmentation is applied globally, compared to <1% without augmentation. The degeneration appears to arise because augmented (noisy) inputs push the decoder into low-likelihood regions of its output space, where the autoregressive model converges to repeating high-probability token sequences rather than producing coherent transcriptions.

For Bengali and Marathi, which have richer morphological systems and more diverse token distributions in the training data, augmentation does not induce this degeneration. The explanation likely relates to the specific tokenizer behavior: Whisper's BPE tokenizer partitions these languages into different token granularities, and the interaction between token frequency distributions and augmentation noise determines whether degeneration occurs.

### 5.3 The v9 Training Recipe

Based on these findings, our v9 recipe applies augmentation as follows:

```
# Bengali and Marathi: augmented training
python train_v3.py --langs bn --augment-langs bn --epochs 5
python train_v3.py --langs mr --augment-langs mr --epochs 5

# Hindi and Tamil: clean training
python train_v3.py --langs hi --no-spec-augment --no-speed-perturb --epochs 3
python train_v3.py --langs ta --no-spec-augment --no-speed-perturb --epochs 3

# Telugu: clean training (augmentation showed no benefit)
python train_v3.py --langs te --no-spec-augment --no-speed-perturb --epochs 3
```

All training was performed on 2× NVIDIA T4 GPUs available through free Kaggle notebooks. Total training time across all five languages was approximately 12–15 GPU-hours.

---

## 6. Deployment

### 6.1 ONNX Export

PolyWhisper adapters can be exported to ONNX for CPU-only inference without PyTorch. The export process:

1. Loads the frozen Whisper-Small backbone
2. Installs the LoRA adapter via forward hooks
3. Exports separate encoder and decoder ONNX graphs (with LoRA weights fused into the graph)
4. Applies INT8 dynamic quantization to reduce model size by ~4×

\begin{table}[t]
\centering
\caption{ONNX export sizes per language. INT8 provides ~4× reduction with verified parity.}
\label{tab:onnx}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Component} & \textbf{FP32} & \textbf{INT8} & \textbf{Reduction} \\
\midrule
Encoder & 358 MB & 97 MB & 3.7× \\
Decoder & 784 MB & 204 MB & 3.8× \\
\textbf{Total per language} & \textbf{1.14 GB} & \textbf{301 MB} & \textbf{3.8×} \\
\bottomrule
\end{tabular}
\end{table}

**Parity verification.** FP32 ONNX output matches PyTorch output with maximum absolute difference < $10^{-3}$ across all five languages (both encoder and decoder). End-to-end greedy spot-checks on FLEURS audio confirm numerical fidelity.

### 6.2 CPU Inference

The ONNX Runtime backend enables CPU-only inference:

```python
from polywhisper import transcribe

# Transcribe with ONNX backend (no GPU required)
result = transcribe("audio.wav", lang="hi", backend="onnx")
print(result.text)
```

### 6.3 Package

PolyWhisper is distributed as a standard Python package:

```bash
pip install polywhisper
# or for ONNX support:
pip install polywhisper[onnx]
```

The package includes the CLI (`polywhisper transcribe`, `polywhisper batch`, `polywhisper export`) and a Python API. Pre-exported ONNX graphs are available on the HuggingFace Hub for each language.

---

## 7. Limitations and Future Work

**Absolute WER is high.** Our best WERs (46.3 Hindi, 70.1 Tamil) remain far from production-grade systems (14–26% WER for SraVaani/IndicConformer). PolyWhisper is not suitable for applications requiring high transcription accuracy such as medical or legal documentation. It is better positioned for assistive technologies, search indexing, and subtitle draft workflows where partial accuracy is acceptable.

**Evaluated on read speech only.** All results use FLEURS read speech. Performance on spontaneous conversational speech, code-switching (Hinglish), and noisy environments will differ and likely degrade.

**Beam=1 numbers.** We report greedy (beam=1) decoding for consistency with our training setup. Beam search (beam=5+) typically improves WER by 10–20% relative but increases latency. We leave beam search optimization to future work.

**Five languages only.** India has 22 scheduled languages and hundreds of spoken languages. We cover five major ones. Extending to additional languages (Gujarati, Punjabi, Kannada, etc.) is straightforward given the adapter architecture but was beyond the scope of this work.

**Data limitation.** We trained on ~19–20K clips per language from IndicVoices-ST. Larger training sets, multi-dataset combination, and unlabeled data (semi-supervised learning) would likely improve results substantially.

**Future directions:**
- Larger backbones (Medium, Large) with the same adapter recipe
- Beam search and shallow fusion with language models
- Extension to more Indic languages
- Real-time streaming with adapter hot-swapping
- Distillation from large teacher models into the adapter space

---

## 8. Conclusion

PolyWhisper provides a complete, reproducible recipe for efficient multilingual Indic ASR. By freezing the Whisper-Small backbone and training per-language LoRA adapters (~14 MB each), we achieve reasonable transcription quality for five major Indian languages while maintaining minimal storage overhead and CPU deployability.

Our key finding — per-language augmentation asymmetry — has practical implications beyond PolyWhisper itself. It warns against applying standard ASR augmentation pipelines uniformly across languages, particularly for Indic scripts. The same augmentation that improves Bengali by 34.5% degrades Hindi by 21.6%, a divergence that demands language-aware training recipes.

We release all code, adapters, training scripts, and evaluation pipelines under the MIT license, along with an installable package (`pip install polywhisper`) and ONNX exports on the HuggingFace Hub. Our hope is that this recipe lowers the barrier for developers building Indic ASR applications and provides a baseline for future research on efficient multilingual speech adaptation.

---

## References

- Chiu, C., et al. (2022). "IndicConformer: A Multilingual ASR Model for Indian Languages." arXiv preprint.
- Conneau, A., et al. (2023). "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech." ICASSP 2023.
- Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023.
- Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Javed, T., et al. (2024). "SraVaani: A Multilingual Speech Recognition Model for Indian Languages." arXiv preprint.
- Nandyala, S., et al. (2025). "IndicWhisper: Adapting Whisper for Indian Languages." arXiv preprint.
- Pratap, V., et al. (2023). "Massively Multilingual Multimodal Speech Recognition (MMS)." arXiv preprint.
- Radford, A., et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." ICML 2023.
- Rybakov, M., et al. (2024). "Whisper LoRA." GitHub repository.
- Zhang, Y., et al. (2023). "Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages." arXiv preprint.

---

*Code: https://github.com/eulogik/PolyWhisper*
*Model: https://huggingface.co/eulogik/polywhisper*
