"""Download/cache a FLEURS test split as {wav, text} json (same format as train_v3).

Usage: python fetch_testset.py <lang>   (e.g. te_in, bn_in, mr_in)
Writes polywhisper_output/data/fleurs_<lang>_test.json and audio wavs.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset

os.environ.setdefault("HF_TOKEN", "")
DATA_DIR = Path("polywhisper_output/data")


def resample(audio, sr):
    from scipy import signal
    if sr == 16000:
        return np.asarray(audio, dtype=np.float32)
    n = int(len(audio) * 16000 / sr)
    return signal.resample(np.asarray(audio, dtype=np.float64), n).astype(np.float32)


def process_sample(audio_array, sr, text, adir, idx):
    audio = resample(audio_array, sr)
    if len(audio) < 1600 or len(audio) > int(30.0 * 16000):
        return None
    text = text.strip()
    if not text or len(text) < 3:
        return None
    wav_path = adir / f"{idx:06d}.wav"
    sf.write(str(wav_path), audio, 16000)
    return {"wav": str(wav_path), "text": text}


def main():
    lang = sys.argv[1]
    cache = DATA_DIR / f"fleurs_{lang}_test.json"
    if cache.exists():
        print(f"Cached: {cache} ({len(json.load(open(cache)))} records)")
        return
    adir = DATA_DIR / f"audio_fleurs_{lang}"
    adir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading fleurs {lang} (test)...")
    ds = load_dataset("google/fleurs", lang, split="test", streaming=True)
    recs, stall, last = [], 0, 0
    for i, item in enumerate(tqdm(ds)):
        a = item["audio"]
        r = process_sample(a["array"], a["sampling_rate"], item["transcription"], adir, len(recs))
        if r:
            recs.append(r)
        if i > 0 and i % 50 == 0:
            if len(recs) == last:
                stall += 30
                if stall > 60:
                    print(f"STALL at {i} items, aborting ({len(recs)} saved)")
                    break
            else:
                stall = 0
            last = len(recs)
    if recs:
        json.dump(recs, open(cache, "w"))
        print(f"Saved {len(recs)} records -> {cache}")
    else:
        print("FAILED: no records")
        sys.exit(1)


if __name__ == "__main__":
    main()