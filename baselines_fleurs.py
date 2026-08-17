"""Whisper baselines (small/medium/large) on FLEURS Indic test sets.

Designed to run on Colab T4 GPU (~3h total for all sizes).
Can also run locally on MPS but will be 3-5x slower.

Usage on Colab:
  !pip install datasets transformers torch soundfile
  !python baselines_fleurs.py --size small --output-dir ./baselines

Usage locally:
  .venv/bin/python baselines_fleurs.py --size large --output-dir ./baselines
"""

import argparse
import json
import time
from pathlib import Path

import torch
import soundfile as sf
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

LANGUAGES = {
    "hi": "hi_in", "ta": "ta_in", "te": "te_in",
    "bn": "bn_in", "mr": "mr_in",
}
WHISPER_SIZES = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
}


def compute_wer(ref, hyp):
    rw, hw = ref.split(), hyp.split()
    if not rw:
        return 0.0 if not hw else 1.0
    n, m = len(rw), len(hw)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                         prev + (rw[i - 1] != hw[j - 1]))
            prev = temp
    return dp[m] / n


def compute_cer(ref, hyp):
    r, h = list(ref), list(hyp)
    if not r:
        return 0.0 if not h else 1.0
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                         prev + (r[i - 1] != h[j - 1]))
            prev = temp
    return dp[m] / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(WHISPER_SIZES.keys()), required=True)
    parser.add_argument("--langs", nargs="+", default=list(LANGUAGES.keys()))
    parser.add_argument("--output-dir", type=str, default="baselines")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {WHISPER_SIZES[args.size]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = WhisperProcessor.from_pretrained(WHISPER_SIZES[args.size])
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_SIZES[args.size]).to(device)
    model.eval()

    all_results = {}
    for lang in args.langs:
        fleurs_split = LANGUAGES[lang]
        print(f"\n{'='*60}")
        print(f"Baseline {args.size}: {lang} ({fleurs_split})")
        print(f"{'='*60}")

        ds = load_dataset("google/fleurs", fleurs_split, split="test", trust_remote_code=True)
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"Samples: {len(ds)}")

        tot_w, tot_c, err_w, err_c = 0, 0, 0, 0
        samples = []
        t0 = time.time()

        for i, ex in enumerate(ds):
            audio = ex["audio"]["array"]
            sr = ex["audio"]["sampling_rate"]
            ref = ex["transcription"]
            if sr != 16000:
                import torchaudio
                audio = torchaudio.functional.resample(torch.tensor(audio), sr, 16000).numpy()

            feats = proc.feature_extractor([audio], sampling_rate=16000,
                                           return_tensors="pt", padding=True)["input_features"].to(device)
            out = model.generate(feats, max_new_tokens=256, num_beams=1,
                                 language=lang, task="transcribe")
            hyp = proc.decode(out[0], skip_special_tokens=True).strip()

            wer = compute_wer(ref, hyp)
            cer = compute_cer(ref, hyp)
            nw = max(1, len(ref.split()))
            tot_w += nw
            tot_c += 1
            err_w += wer * nw
            err_c += cer
            samples.append({"ref": ref, "hyp": hyp, "wer": wer, "cer": cer})

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(ds) - i - 1) / rate
                print(f"  [{i+1}/{len(ds)}] WER={err_w/tot_w*100:.1f}% "
                      f"CER={err_c/tot_c*100:.1f}% ({rate:.1f} samples/s, ETA {eta:.0f}s)")

        elapsed = time.time() - t0
        wer_pct = err_w / max(1, tot_w) * 100
        cer_pct = err_c / max(1, tot_c) * 100
        print(f"  FINAL: WER {wer_pct:.1f}%  CER {cer_pct:.1f}%  ({elapsed:.0f}s)")

        result = {"lang": lang, "model": args.size, "wer": wer_pct, "cer": cer_pct,
                  "n": len(ds), "time_s": elapsed, "samples": samples}
        all_results[lang] = result

        out_file = out_dir / f"baseline_{args.size}_{lang}.json"
        json.dump(result, open(out_file, "w"), indent=1, ensure_ascii=False)
        print(f"  Saved: {out_file}")

    summary = {lang: {"wer": r["wer"], "cer": r["cer"]} for lang, r in all_results.items()}
    json.dump(summary, open(out_dir / f"summary_{args.size}.json", "w"), indent=2)
    print(f"\nSummary: {out_dir / f'summary_{args.size}.json'}")


if __name__ == "__main__":
    main()