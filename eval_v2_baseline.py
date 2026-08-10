"""
PolyWhisper v2 Vanilla Baseline Evaluation
Plain Whisper Base (no LoRA) on same test sets, same inference settings.
"""

import torch
import numpy as np
import json
import time
import random
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

WHISPER_MODEL = "openai/whisper-base"
WHISPER_EXPECTED_LEN = 3000
LANGUAGES = ["en", "hi", "hinglish"]
MAX_NEW_TOKENS = 128
MAX_SAMPLES = 200
BATCH_SIZE = 4

SAVE_DIR = Path("./polywhisper_output")
DATA_DIR = SAVE_DIR / "data"
EVAL_FILE = SAVE_DIR / "eval_v2_baseline_results.json"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
tok = processor.tokenizer

model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(DEVICE)
model.eval()
print(f"Loaded {WHISPER_MODEL} (vanilla, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

def load_test_data(lang):
    cache_map = {
        "en": ["librispeech_en_test.json"],
        "hi": ["fleurs_hi_in_test.json"],
        "hinglish": ["mucs_hinglish_hinglish_test.json"],
    }
    files = cache_map.get(lang, [])
    data = []
    for fname in files:
        cf = DATA_DIR / fname
        if cf.exists():
            records = json.load(open(cf))
            print(f"  Loaded {len(records)} samples from {cf.name}")
            data.extend(records)
        else:
            print(f"  WARNING: {cf} not found")
    return data

def process_audio(wav_path):
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = np.array(audio, dtype=np.float32)
    audio = audio[:int(30.0 * 16000)]
    inputs = processor.feature_extractor(
        [audio], sampling_rate=16000, return_tensors="pt", padding=True
    )
    feat = inputs["input_features"].to(DEVICE)
    if feat.shape[-1] < WHISPER_EXPECTED_LEN:
        pad = torch.zeros(feat.shape[0], feat.shape[1], WHISPER_EXPECTED_LEN - feat.shape[-1], dtype=feat.dtype, device=DEVICE)
        feat = torch.cat([feat, pad], dim=-1)
    else:
        feat = feat[:, :, :WHISPER_EXPECTED_LEN]
    return feat

def compute_wer(references, hypotheses):
    import re as _re
    def _norm(text):
        text = text.lower()
        text = _re.sub(r'[^\w\s]', '', text)
        return _re.sub(r'\s+', ' ', text).strip()
    total_wer = 0.0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        ref_w = _norm(ref).split()
        hyp_w = _norm(hyp).split()
        if not ref_w:
            continue
        n = len(ref_w)
        m = len(hyp_w)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_w[i - 1] == hyp_w[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        total_wer += dp[n][m]
        total_words += n
    return total_wer / total_words if total_words else 1.0

print("\n" + "=" * 60)
print("VANILLA WHISPER BASE BASELINE EVALUATION")
print("=" * 60)

results = {}
all_examples = []

for lang in LANGUAGES:
    print(f"\n--- {lang.upper()} ---")
    test_data = load_test_data(lang)
    if not test_data:
        results[lang] = {"wer": 1.0, "samples": 0, "error": "no test data"}
        continue
    if len(test_data) > MAX_SAMPLES:
        random.seed(42)
        test_data = random.sample(test_data, MAX_SAMPLES)
        print(f"  Sampled {MAX_SAMPLES} from {len(test_data)} total")

    references = []
    hypotheses = []
    examples = []
    start_time = time.time()

    for i, item in enumerate(tqdm(test_data, desc=f"  {lang}")):
        wav_path = item["wav"]
        ref_text = item["text"]
        if not Path(wav_path).exists():
            continue
        try:
            feat = process_audio(wav_path)
            forced_ids = processor.get_decoder_prompt_ids(language=lang if lang != "hinglish" else "hi", task="transcribe")
            out = model.generate(feat,
                forced_decoder_ids=forced_ids,
                num_beams=5,
                max_length=MAX_NEW_TOKENS,
            )
            hyp_text = tok.decode(out[0].tolist(), skip_special_tokens=True)
            references.append(ref_text)
            hypotheses.append(hyp_text)
            examples.append({"ref": ref_text, "hyp": hyp_text})
            if len(references) <= 3:
                print(f"    REF: {ref_text[:80]}")
                print(f"    HYP: {hyp_text[:80]}")
                print()
        except Exception as e:
            print(f"    ERROR on sample {i}: {e}")
            continue

    elapsed = time.time() - start_time
    if references:
        wer_score = compute_wer(references, hypotheses)
        results[lang] = {"wer": wer_score, "samples": len(references), "elapsed_sec": elapsed}
        all_examples.extend([(lang, ex) for ex in examples])
        print(f"  WER: {wer_score * 100:.1f}%")
        print(f"  Samples: {len(references)}")
        print(f"  Time: {elapsed:.1f}s")
    else:
        results[lang] = {"wer": 1.0, "samples": 0, "error": "no valid samples"}

print("\n" + "=" * 60)
print("BASELINE RESULTS SUMMARY")
print("=" * 60)
print(f"{'Language':<12} {'WER':>8} {'Samples':>8}")
print("-" * 30)
for lang in LANGUAGES:
    r = results.get(lang, {})
    wer = r.get("wer", 1.0) * 100
    samples = r.get("samples", 0)
    print(f"{lang:<12} {wer:>7.1f}% {samples:>7d}")

total_samples = sum(r.get("samples", 0) for r in results.values())
if total_samples > 0:
    avg_wer = sum(r.get("wer", 1.0) * r.get("samples", 0) for r in results.values()) / total_samples
    print(f"\nAvg WER: {avg_wer * 100:.1f}% ({total_samples} samples)")
else:
    print("\nNo samples evaluated.")

if all_examples:
    print("\n" + "=" * 60)
    print("DETAILED EXAMPLES (first 5 per language)")
    print("=" * 60)
    for lang in LANGUAGES:
        lang_examples = [ex for ex in all_examples if ex[0] == lang]
        print(f"\n--- {lang.upper()} ---")
        for i, (l, ex) in enumerate(lang_examples[:5]):
            print(f"  {i+1}. REF: {ex['ref'][:100]}")
            print(f"     HYP: {ex['hyp'][:100]}")

EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(EVAL_FILE, "w"), indent=2)
print(f"\nResults saved to {EVAL_FILE}")
