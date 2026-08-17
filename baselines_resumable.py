"""Resumable whisper baselines on FLEURS Indic test sets.

Designed for Colab T4 (disconnect-safe). Persists state to HF Hub after each
language completes — nothing is lost on session restart.

State flow:
  1. On start: download baselines_state.json from HF Hub (or create empty)
  2. For each (size, lang) not in state["completed"]:
     a. Run eval
     b. Save result locally
     c. Upload result to HF Hub
     d. Update state, upload state to HF Hub
  3. Generate summary, upload to HF Hub

Resume: re-run the cell. It skips everything already in state["completed"].

Usage (standalone):
  python baselines_resumable.py --repo eulogik/polywhisper --size small

Usage (Colab):
  !python baselines_resumable.py --repo eulogik/polywhisper --size large --hf-token $HF_TOKEN
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import soundfile as sf
from huggingface_hub import hf_hub_download, HfApi
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Suppress the harmless "max_new_tokens and max_length both set" advisory warning
try:
    from transformers.utils.logging import set_verbosity_error
    set_verbosity_error()
except Exception:
    pass


# ======================== CONFIG ========================

LANGUAGES = {
    "hi": "hi_in", "ta": "ta_in", "te": "te_in",
    "bn": "bn_in", "mr": "mr_in",
}

WHISPER_SIZES = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
}

STATE_FILE = "baselines_state.json"
MAX_NEW_TOKENS = 256


# ======================== METRICS ========================

def compute_wer(ref, hyp):
    rw, hw = ref.split(), hyp.split()
    if not rw:
        return 0.0 if not hw else 1.0
    n, m = len(rw), len(hw)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
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
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                         prev + (r[i - 1] != h[j - 1]))
            prev = temp
    return dp[m] / n


# ======================== STATE ========================

def load_state(repo_id):
    """Download state from HF Hub, or return empty state."""
    try:
        path = hf_hub_download(repo_id=repo_id, filename=STATE_FILE,
                               repo_type="model")
        state = json.load(open(path))
        print(f"Loaded state from HF: {len(state['completed'])} completed")
        return state
    except Exception:
        print("No existing state found — starting fresh")
        return {"completed": {}, "last_update": None}


def save_state(state, repo_id, local_dir):
    """Save state locally + upload to HF Hub."""
    state["last_update"] = datetime.now().isoformat()
    local_path = local_dir / STATE_FILE
    json.dump(state, open(local_path, "w"), indent=2)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=STATE_FILE,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"state update: {len(state['completed'])} completed",
    )


def save_and_upload(result, repo_id, local_dir):
    """Save result locally + upload to HF Hub."""
    size = result["model"]
    lang = result["lang"]
    filename = f"baselines_{size}_{lang}.json"
    local_path = local_dir / filename
    json.dump(result, open(local_path, "w"), indent=1, ensure_ascii=False)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"baseline {size}/{lang}: WER={result['wer']:.1f} CER={result['cer']:.1f}",
    )
    print(f"  Uploaded: {filename}")


def save_summary(all_results, repo_id, local_dir):
    """Save summary locally + upload to HF Hub."""
    size = all_results[list(all_results.keys())[0]]["model"]
    summary = {lang: {"wer": r["wer"], "cer": r["cer"], "n": r["n"]}
               for lang, r in all_results.items()}
    filename = f"baselines_summary_{size}.json"
    local_path = local_dir / filename
    json.dump(summary, open(local_path, "w"), indent=2)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"baseline {size} summary",
    )
    print(f"Summary: {filename}")


# ======================== EVAL ========================

def eval_language(model, proc, lang, fleurs_split, device, max_samples=0):
    """Run baseline eval for one language. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"  {model.config.model_type} — {lang} ({fleurs_split})")
    print(f"{'='*60}")

    ds = load_dataset("google/fleurs", fleurs_split, split="test", trust_remote_code=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    tot_w, tot_c, err_w, err_c = 0, 0, 0, 0
    samples = []
    t0 = time.time()

    for i, ex in enumerate(ds):
        audio = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        ref = ex["transcription"]
        if sr != 16000:
            import torchaudio
            audio = torchaudio.functional.resample(
                torch.tensor(audio, dtype=torch.float32), sr, 16000
            ).numpy()

        feats = proc.feature_extractor(
            [audio], sampling_rate=16000, return_tensors="pt", padding=True
        )["input_features"].to(device)

        out = model.generate(feats, max_new_tokens=MAX_NEW_TOKENS, num_beams=1,
                             language=lang, task="transcribe")
        hyp = proc.decode(out[0], skip_special_tokens=True).strip()

        wer = compute_wer(ref, hyp)
        cer = compute_cer(ref, hyp)
        nw = max(1, len(ref.split()))
        tot_w += nw; tot_c += 1
        err_w += wer * nw; err_c += cer
        samples.append({"ref": ref, "hyp": hyp, "wer": round(wer, 4), "cer": round(cer, 4)})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(ds) - i - 1) / rate
            print(f"    [{i+1}/{len(ds)}] WER={err_w/tot_w*100:.1f}% "
                  f"CER={err_c/tot_c*100:.1f}% ({rate:.1f}/s ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    wer_pct = round(err_w / max(1, tot_w) * 100, 2)
    cer_pct = round(err_c / max(1, tot_c) * 100, 2)
    print(f"  DONE: WER {wer_pct:.1f}% CER {cer_pct:.1f}% ({elapsed:.0f}s)")

    return {
        "model": size_name(model.config._name_or_path),
        "lang": lang,
        "wer": wer_pct, "cer": cer_pct,
        "n": len(ds), "time_s": round(elapsed),
        "samples": samples,
    }


def size_name(model_path):
    """Map model path to short name."""
    for k in WHISPER_SIZES:
        if k in model_path:
            return k
    return model_path.split("/")[-1]


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="eulogik/polywhisper")
    parser.add_argument("--size", choices=list(WHISPER_SIZES.keys()), required=True)
    parser.add_argument("--langs", nargs="+", default=list(LANGUAGES.keys()))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--output-dir", default="baselines")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default=None, help="force device: cuda/mps/cpu")
    args = parser.parse_args()

    # flatten comma-separated langs (Colab passes "hi,ta,te,bn,mr" as one string)
    args.langs = [l.strip() for item in args.langs for l in item.split(",") if l.strip()]

    if args.hf_token:
        HfApi(token=args.hf_token)  # validate token early

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    local_dir = Path(args.output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {WHISPER_SIZES[args.size]}")
    print(f"Repo: {args.repo}")
    print(f"Languages: {args.langs}")

    # Load state from HF Hub
    state = load_state(args.repo)
    completed = state["completed"]

    # Filter to pending languages
    pending = [l for l in args.langs if f"{args.size}_{l}" not in completed]
    if not pending:
        print("\nAll done! Nothing to run.")
        return

    print(f"Completed: {len(completed)} | Pending: {pending}")

    # Load model once
    model_path = WHISPER_SIZES[args.size]
    print(f"\nLoading {model_path}...")
    proc = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    all_results = {}
    for lang in pending:
        fleurs_split = LANGUAGES[lang]
        try:
            result = eval_language(model, proc, lang, fleurs_split, device, args.max_samples)
            all_results[lang] = result

            # Persist after each language
            save_and_upload(result, args.repo, local_dir)
            completed[f"{args.size}_{lang}"] = {
                "wer": result["wer"], "cer": result["cer"],
                "n": result["n"], "time_s": result["time_s"],
            }
            state["completed"] = completed
            save_state(state, args.repo, local_dir)
            print(f"  State saved ({len(completed)} completed)")

        except Exception as e:
            print(f"\n  ERROR on {lang}: {e}")
            traceback.print_exc()
            print(f"  Saving state and continuing...")
            state["completed"] = completed
            save_state(state, args.repo, local_dir)
            continue

    if all_results:
        save_summary(all_results, args.repo, local_dir)

    print(f"\n{'='*60}")
    print(f"ALL DONE — {len(completed)} total completed")
    for lang, r in all_results.items():
        print(f"  {lang}: WER {r['wer']:.1f}% CER {r['cer']:.1f}%")


if __name__ == "__main__":
    main()