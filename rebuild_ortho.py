"""
Rebuild test refs to match TRAIN orthography (Latin loanwords) for a fair WER.
- Builds a frequency table of Latin words from the train set.
- For each Devanagari test word, transliterates it and matches against train
  Latin words (edit-distance). On a confident match, replaces with the train
  spelling; otherwise keeps the word.
- Rebuilds the word-level `tokens` (script-based lang labels) like
  build_codeswitch_labels.py.
Saves: polywhisper_output/data/hinglish_codeswitch_test_ortho.json
"""

import json
import re
from collections import Counter
from pathlib import Path

from eval_metrics import translit, _edit_distance

DATA_DIR = Path("./polywhisper_output/data")
TRAIN_JSON = DATA_DIR / "hinglish_codeswitch_train.json"
TEST_JSON = DATA_DIR / "hinglish_codeswitch_test.json"
OUT_JSON = DATA_DIR / "hinglish_codeswitch_test_ortho.json"

DEVANAGARI = re.compile(r'[\u0900-\u097F]')
LATIN = re.compile(r'[A-Za-z]')

STOPLIST = {
    "में", "है", "हैं", "हैं", "और", "के", "की", "का", "यह", "यहाँ", "एक", "इस", "हम",
    "तो", "पर", "से", "नहीं", "नही", "वह", "वे", "जो", "हो", "है।", "होता", "होते", "होती",
    "कर", "करें", "करते", "करना", "करने", "क्या", "कैसे", "कहाँ", "जब", "तब", "अब", "फिर",
    "बहुत", "आप", "मैं", "हमारे", "हमारा", "आदि", "रूप", "बारे", "में।", "लिए", "लिये",
    "बाद", "पहले", "साथ", "द्वारा", "वाले", "वाला", "वाली", "देखते", "देखें", "देख",
}

# FIXME: 'हैं' duplicated above; keep set semantics anyway (harmless).


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
    return None


def main():
    train = json.load(open(TRAIN_JSON))
    latin_vocab = Counter()
    for item in train:
        for t in re.findall(r'[A-Za-z]+', item["text"]):
            latin_vocab[t.lower()] += 1
    candidates = [w for w, c in latin_vocab.items() if c >= 2 and len(w) >= 4]
    by_len = {}
    for w in candidates:
        by_len.setdefault(len(w), []).append(w)

    test = json.load(open(TEST_JSON))
    dev_words = set()
    for item in test:
        for raw in re.findall(r'\S+', item["text"]):
            token = re.sub(r'[^\w\u0900-\u097F]', '', raw)
            if not token or token in STOPLIST:
                continue
            if DEVANAGARI.search(token) and not LATIN.search(token):
                dev_words.add(token)

    mapping = {}
    for token in dev_words:
        d = translit(token, final_a=False)
        if len(d) < 4:
            continue
        best, best_d = None, 99
        for L in range(max(4, len(d) - 2), len(d) + 3):
            for w in by_len.get(L, []):
                dd = _edit_distance(d, w)
                if dd < best_d:
                    best, best_d = w, dd
        if best and ((best_d <= 1 and len(d) >= 5 and len(best) >= 5)
                     or (best_d <= 2 and len(d) >= 8 and len(best) >= 8)):
            mapping[token] = best

    n_replaced = 0
    n_total_dev = 0
    stats = Counter()
    records = []
    for item in test:
        text = item["text"].strip()
        out_words = []
        for raw in re.findall(r'\S+', text):
            token = re.sub(r'[^\w\u0900-\u097F]', '', raw)
            if not token:
                out_words.append(raw)
                continue
            if DEVANAGARI.search(token) and not LATIN.search(token) and token not in STOPLIST:
                n_total_dev += 1
                if token in mapping:
                    out_words.append(mapping[token])
                    n_replaced += 1
                    continue
            out_words.append(raw)
        new_text = ' '.join(out_words)

        tokens = re.findall(r'[\w\u0900-\u097F]+', new_text)
        labels = []
        for t in tokens:
            lab = label_token(t)
            if lab is None:
                lab = 'en'
            labels.append(lab)
        n_hi = labels.count('hi')
        n_en = labels.count('en')
        records.append({
            "wav": item["wav"],
            "text": new_text,
            "tokens": [{"t": t, "lang": l} for t, l in zip(tokens, labels)],
            "cs": int(n_hi > 0 and n_en > 0),
        })
        stats['cs_samples'] += 1 if (n_hi and n_en) else 0

    json.dump(records, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)
    print(f"Saved {OUT_JSON}: {len(records)} records")
    print(f"Replaced {n_replaced}/{n_total_dev} Devanagari words with train Latin spellings")
    print(f"cs_samples: {stats['cs_samples']}/{len(records)}")


if __name__ == "__main__":
    main()
