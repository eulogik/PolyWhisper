"""
Dataset cleaning for PolyWhisper v2
- Exact dedup (keep 1 copy per unique text)
- Remove Hinglish template boilerplate (keyword match)
- Remove ultra-short samples
- Save cleaned versions
"""

import json, re, collections
from pathlib import Path

DATA_DIR = Path("./polywhisper_output/data")

# ============ HINGLISH TEMPLATE KEYWORDS ============
# Phrases that identify template boilerplate from Spoken Tutorial platform
TEMPLATE_KEYWORDS = [
    "spoken tutorial",
    "bandwidth",
    "download",
    "link पर",
    "link पर उपलब्ध video",
    "mission पर अधिक जानकारी",
    "प्रमाणपत्र",
    "नियतकार्य",
    "सारांशित",
    "contact @spokentutorial",
    "online test",
    "हमसे जुड़ने के लिए धन्यवाद",
    "close पर click",
    "ok पर click",
    "save पर click",
    "ok button पर click",
    "save button पर click",
    "next button पर click",
    "enter दबाएं",
    "enter दबाएँ",
    "आई आई टी बॉम्बे से मैं श्रुति आर्य",
    "आई आई टी बॉम्बे की ओर से मैं",
    "आईआईटी बॉम्बे की ओर से मैं",
    "यह स्क्रिप्ट प्रभाकर द्वारा अनुवादित",
    "यह स्क्रिप्ट विकास द्वारा अनुवादित",
    "यह स्क्रिप्ट बिंदु पांडे द्वारा अनुवादित",
    "iit iit टी बॉम्बे",
    "spoken hyphen tutorial",
    "spokentutorial",
    "talktoa teacher",
    "talktoateacher",
    "workshops भी conduct करते",
    "workshops भी conduct",
    "workshops का आयोजन",
    "कार्यशालाएं चलाती है",
    "कार्यशालाएँ भी चलाती है",
    "कार्यशालाएँ आयोजित",
    "कार्यशालाओं का आयोजन",
    "nmeict",
    "mhrd",
    "एमएचआरडी",
]

def is_template(text):
    text_lower = text.lower()
    for kw in TEMPLATE_KEYWORDS:
        if kw.lower() in text_lower:
            return True
    return False

def clean_dataset(name, lang, split, max_samples=None):
    path = DATA_DIR / f"{name}_{lang}_{split}.json"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return None

    data = json.load(open(path))
    original_count = len(data)
    print(f"\n{'='*60}")
    print(f"{name}/{lang}/{split}: {original_count} samples")

    # Track stats
    removed_template = 0
    removed_short = 0
    removed_non_ascii_hi = 0

    # Step 1: Remove template boilerplate (Hinglish only)
    if "hinglish" in name or "hinglish" in lang:
        template_texts = []
        non_template = []
        for d in data:
            if is_template(d["text"]):
                template_texts.append(d)
            else:
                non_template.append(d)
        removed_template = len(template_texts)
        data = non_template
        print(f"  Template boilerplate: {removed_template} removed ({removed_template/original_count*100:.1f}%)")

    # Step 2: Remove ultra-short texts (<3 chars)
    pre_short = len(data)
    data = [d for d in data if len(d["text"].strip()) >= 3]
    removed_short = pre_short - len(data)
    if removed_short:
        print(f"  Ultra-short (<3 chars): {removed_short} removed")

    # Step 3: Exact dedup (keep first occurrence of each unique text)
    seen = set()
    deduped = []
    for d in data:
        text = d["text"].strip().lower()
        if text not in seen:
            seen.add(text)
            deduped.append(d)
    removed_dedup = len(data) - len(deduped)
    print(f"  Exact dedup: {removed_dedup} removed ({removed_dedup/original_count*100:.1f}%)")
    print(f"  FINAL: {len(deduped)} samples (removed {original_count - len(deduped)} total)")

    # Save cleaned version
    output_path = DATA_DIR / f"{name}_{lang}_{split}_cleaned.json"
    json.dump(deduped, open(output_path, "w"), indent=2)
    print(f"  Saved: {output_path}")
    return len(deduped)

def main():
    # Only clean training datasets. Test sets remain untouched (dedup would remove valid multi-speaker samples).
    datasets = [
        ("librispeech", "en", "train.100"),
        ("indicvoices_st", "hindi", "hindi"),
        ("mucs_hinglish", "hinglish", "train"),
    ]
    for name, lang, split in datasets:
        clean_dataset(name, lang, split)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, lang, split in datasets:
        orig = Path(DATA_DIR / f"{name}_{lang}_{split}.json")
        clean = Path(DATA_DIR / f"{name}_{lang}_{split}_cleaned.json")
        if orig.exists() and clean.exists():
            o = len(json.load(open(orig)))
            c = len(json.load(open(clean)))
            pct = (o - c) / o * 100
            print(f"  {name}/{lang}/{split}: {o} -> {c} ({pct:.1f}% removed)")

if __name__ == "__main__":
    main()
