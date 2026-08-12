# Hacker News post — text submission

**Title:** PolyWhisper — Hinglish ASR with per-token LoRA routing, trained on a Mac Mini

**Text:**

We built a code-switch (Hindi–English / Hinglish) speech recognition system: frozen
Whisper-Base encoder + two rank-8 LoRA language experts + a 33K-parameter router that gates
per token.

Results on a 3,129-utterance code-switched test set:
- 58.8% WER vs 66.6% vanilla Whisper-Base (+7.8 pts)
- 58.8% vs 72.1% static 50/50 expert mix (+13.3 pts — routing is what earns it)
- ~20× fewer hallucination loops (13 vs 279)
- Router language accuracy 89.1%

The honest part: our first "working" router was broken. A label collator compared ints to a
string, collapsed every label to one class, and the router silently routed everything through
the English expert. WER looked great; routing was never actually tested. A weight-heatmap
introspection script caught it. We fixed the labels, retrained, and only then did the ablation
show real routing value.

Everything is MIT, reproducible (per-utterance JSONs published), and trains in ~a day on a
16GB M4 Mac Mini — no GPU cluster.

Links: https://github.com/eulogik/PolyWhisper · https://huggingface.co/eulogik/polywhisper-hinglish-router
By Eulogik (github.com/eulogik).

Happy to answer questions about the routing mechanics, the label-bug archaeology, or the
data (MUCS CC-BY-SA / IndicVoices-ST CC-BY derived).
