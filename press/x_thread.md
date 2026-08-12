# X / Twitter thread — paste as 8-10 tweets

---

**1/**
We built PolyWhisper: Hinglish (Hindi-English code-switched) ASR using a frozen Whisper-Base
encoder + 2 tiny LoRA experts + a 33K-param router that picks per token which expert speaks next.
Trained on a Mac Mini. MIT. Results: 58.8% WER vs 66.6% vanilla.

**2/**
India's most spoken language is two languages at once:
"इस tutorial में हम impress window के भागों के बारे में सीखेंगे"
Vanilla Whisper-Base hallucinates loops and mangles Devanagari here. One model can't serve two token-spaces.

**3/**
Our fix: don't retrain the 74M model. Keep the encoder frozen. Add rank-8 LoRA adapters —
one English expert, one Hindi expert — and a router that reads decoder state every token.

**4/**
+7.8 WER pts over vanilla Whisper-Base. +13.3 over a static 50/50 expert mix — the routing
itself earns the gain. ~20× fewer hallucination loops (13 vs 279).

**5/**
But here's the honest story: our first "working" router was broken. A label collator compared
ints to a string, collapsed every token to one class, and the router silently routed everything
through the English expert. Routing was never actually tested until we fixed it and retrained.

**6/**
Lesson: introspection scripts > benchmarks. A heatmap of router weights caught what WER alone hid.

**7/**
Reproducible: full 3,129-utterance per-sample results published, ortho-normalized test set,
crash-safe pipeline, MIT license.

**8/**
Next: whisper-small/medium/large baselines, Tamil-English + Bengali-English pairs,
orthography normalization.

**9/**
GitHub → https://github.com/eulogik/PolyWhisper
HF → https://huggingface.co/eulogik/polywhisper-hinglish-router
By @EulogikDev — open, edge-ready, low-resource ASR for code-switched speech.

#Eulogik #PolyWhisper #ASR #Hinglish #SpeechRecognition #Whisper #LoRA #EdgeAI #NLP
