# LinkedIn post — paste as-is

---

**PolyWhisper: Hinglish ASR with per-token LoRA routing — built by Eulogik 🎙️🇮🇳**

I'm excited to share what our team at **Eulogik** has been building: a speech recognition
system for Hinglish — the Hindi–English code-switched speech that most of India actually
speaks.

**The problem:** Whisper-Base, tuned for clean English, collapses on code-switched audio —
repetitive hallucinations, dropped English words, mangled Devanagari.

**The idea:** Keep the Whisper encoder frozen. Add two tiny LoRA language experts (English,
Hindi) and a 33K-parameter router that decides *per token* which expert predicts the next word.

**Measured results** (3,129-utterance code-switched test set):
- 58.8% WER vs 66.6% for vanilla Whisper-Base (+7.8 pts)
- 58.8% vs 72.1% for a static 50/50 expert mix (+13.3 pts — routing earns its keep)
- ~20× fewer hallucination loops (13 vs 279)
- Router per-token language accuracy: 89.1%
- Trained in ~a day on a 16GB Mac Mini — no GPU cluster

**A lesson worth sharing:** our first "working" router was silently broken (a label-collator
bug collapsed every token into one class, so the router routed everything to the English
expert — routing was never actually tested). An introspection heatmap caught what WER hid.
Fix the labels, retrain, and *then* the ablations prove routing matters: +13.3 pts.

**Open and reproducible:**
- GitHub: https://github.com/eulogik/PolyWhisper (MIT)
- Hugging Face: https://huggingface.co/eulogik/polywhisper-hinglish-router
- Full per-utterance evaluation JSONs published for independent re-scoring

Next up: whisper-small/medium/large baselines, Tamil-English and Bengali-English pairs,
orthography normalization.

If you work on Indic ASR, code-switching, or edge ML — we'd love to connect.

#ASR #Hinglish #SpeechRecognition #Whisper #LoRA #EdgeAI #Eulogik #IndiaTech #NLP #MachineLearning
