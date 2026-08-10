"""
Rebuild the English test cache: the v2 test wavs in audio_librispeech_en/ were
clobbered by A1's 28.5K train download (same dir, same 000000.wav naming).
Labels in librispeech_en_test.json are still correct; re-download the matching
audio into a dedicated dir and rewrite the json wav paths (labels unchanged,
so WER stays comparable to v2's 44.9% baseline).
"""

import json
import soundfile as sf
from datasets import load_dataset
from pathlib import Path

DATA_DIR = Path("./polywhisper_output/data")
SRC_JSON = DATA_DIR / "librispeech_en_test.json"
OUT_DIR = DATA_DIR / "audio_librispeech_en_test"
OUT_JSON = DATA_DIR / "librispeech_en_test.json"

OUT_DIR.mkdir(exist_ok=True)
recs = json.load(open(SRC_JSON))
texts = {r["text"] for r in recs}
print(f"Loading librispeech test split ({len(texts)} texts to find)...", flush=True)

ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
by_text = {}
for item in ds:
    t = item["text"].strip().upper()
    if t in texts and t not in by_text:
        by_text[t] = item
    if len(by_text) == len(texts):
        break
print(f"Matched {len(by_text)}/{len(texts)} texts", flush=True)

missing = [r for r in recs if r["text"] not in by_text]
if missing:
    print(f"WARNING: {len(missing)} texts unmatched, keeping old wav refs for them", flush=True)

fixed = 0
for r in recs:
    item = by_text.get(r["text"])
    if item is None:
        continue
    idx = recs.index(r)
    out_wav = OUT_DIR / f"{idx:06d}.wav"
    audio = item["audio"]["array"]
    sf.write(out_wav, audio, item["audio"]["sampling_rate"])
    r["wav"] = str(out_wav)
    fixed += 1

json.dump(recs, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
print(f"Rebuilt {SRC_JSON}: {fixed} wav refs rewritten to {OUT_DIR}", flush=True)
