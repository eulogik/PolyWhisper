"""
B1: Build code-switch labeled dataset for the router experiment.
Labels each token of MUCS Hinglish transcripts as 'hi' (Devanagari script) or 'en' (Latin).
Saves: polywhisper_output/data/hinglish_codeswitch_train.json / _test.json
"""

import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path("./polywhisper_output/data")
OUT_TRAIN = DATA_DIR / "hinglish_codeswitch_train.json"
OUT_TEST = DATA_DIR / "hinglish_codeswitch_test.json"

DEVANAGARI = re.compile(r'[\u0900-\u097F]')
LATIN = re.compile(r'[A-Za-z]')


def label_token(token):
    has_dev = bool(DEVANAGARI.search(token))
    has_lat = bool(LATIN.search(token))
    if has_dev and has_lat:
        dev_len = len(DEVANAGARI.findall(token))
        lat_len = len(LATIN.findall(token))
        return 'hi' if dev_len >= lat_len else 'en'
    if has_dev:
        return 'hi'
    if has_lat:
        return 'en'
    return None  # punctuation/numbers only


def process(data_path):
    data = json.load(open(data_path))
    records = []
    stats = Counter()
    total_tokens = 0
    cs_samples = 0
    for i, item in enumerate(data):
        text = item["text"].strip()
        tokens = re.findall(r'[\w\u0900-\u097F]+', text)
        if not tokens:
            continue
        labels = []
        for t in tokens:
            lab = label_token(t)
            if lab is None:
                lab = 'en' if i % 2 == 0 else 'hi'  # tie-break deterministically
            labels.append(lab)
            total_tokens += 1
        n_hi = labels.count('hi')
        n_en = labels.count('en')
        if n_hi and n_en:
            cs_samples += 1
        stats['samples'] += 1
        stats['hi_tokens'] += n_hi
        stats['en_tokens'] += n_en
        stats['cs_samples'] += (1 if (n_hi and n_en) else 0)
        records.append({
            "wav": item["wav"],
            "text": text,
            "tokens": [{"t": t, "lang": lab} for t, lab in zip(tokens, labels)],
            "cs": 1 if (n_hi and n_en) else 0,
        })
    return records, stats, total_tokens, cs_samples


for split, out in [("train", OUT_TRAIN), ("test", OUT_TEST)]:
    src = DATA_DIR / f"mucs_hinglish_hinglish_{split}_cleaned.json"
    if not src.exists():
        src = DATA_DIR / f"mucs_hinglish_hinglish_{split}.json"
    print(f"Loading {src}...")
    records, stats, total_tokens, cs_samples = process(src)
    json.dump(records, open(out, "w"), indent=1, ensure_ascii=False)
    pct = stats['cs_samples'] / stats['samples'] * 100 if stats['samples'] else 0
    print(f"  -> {out.name}: {stats['samples']} samples, {total_tokens} tokens")
    print(f"     hi={stats['hi_tokens']} ({stats['hi_tokens']/total_tokens*100:.1f}%), "
          f"en={stats['en_tokens']} ({stats['en_tokens']/total_tokens*100:.1f}%)")
    print(f"     code-switched samples: {stats['cs_samples']} ({pct:.1f}%)")
