#!/usr/bin/env python3
"""Kaggle eval-only pass: product adapters are already on HF from the training
session; this restores them, fetches FLEURS test audio, runs the 5 evals
(256 tokens, small+encLoRA) and uploads results. No training."""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = "eulogik/polywhisper"
TAG = "_prod"
LANGS = {"hi": "gpu0", "te": "gpu0", "bn": "gpu0", "ta": "gpu1", "mr": "gpu1"}
FLEURS_CODE = {"hi": "hi_in", "te": "te_in", "bn": "bn_in", "ta": "ta_in", "mr": "mr_in"}

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    raise SystemExit("pip install huggingface_hub")

api = HfApi()
api.token = os.environ.get("HF_TOKEN", "")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def restore_from_hf():
    """Pull data index jsons + adapters into the expected save-dir layout."""
    files = [f for f in api.list_repo_files(REPO, repo_type="model")
             if f.startswith("polywhisper_output_gpu")]
    for f in sorted(files):
        dest = Path(f)
        if dest.exists():
            continue
        try:
            p = hf_hub_download(REPO, f, repo_type="model")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)
            log(f"  restored {f}")
        except Exception as e:
            log(f"  skip {f}: {str(e)[:100]}")


def ensure_test_audio(lang):
    """Recreate FLEURS test FLACs + index json if audio is missing on disk."""
    code = FLEURS_CODE[lang]
    save_dir = f"polywhisper_output_gpu{LANGS[lang][3]}"
    data_dir = Path(save_dir) / "data"
    jpath = data_dir / f"fleurs_{code}_test.json"
    adir = data_dir / f"audio_fleurs_{code}"
    adir.mkdir(parents=True, exist_ok=True)

    if jpath.exists():
        recs = json.load(open(jpath))
        n_audio = len(list(adir.iterdir())) if adir.exists() else 0
        if n_audio >= int(len(recs) * 0.9):
            log(f"  {lang}: test set cached ({len(recs)} samples)")
            return str(jpath)
        log(f"  {lang}: stale cache ({n_audio}/{len(recs)} audio) — refetching")

    import soundfile as sf
    from datasets import load_dataset
    ds = load_dataset("google/fleurs", code, split="test", streaming=True,
                      trust_remote_code=True)
    recs = []
    for item in ds:
        try:
            a = item["audio"]
            arr = np.asarray(a["array"], dtype=np.float32)
            if a["sampling_rate"] != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=a["sampling_rate"], target_sr=16000)
            text = (item.get("transcription") or "").strip()
            if len(arr) < 1600 or len(arr) > 30 * 16000 or len(text) < 3:
                continue
            wav = adir / f"{len(recs):06d}.flac"
            sf.write(str(wav), arr, 16000, format="FLAC")
            recs.append({"wav": str(wav), "text": text})
        except Exception:
            continue
        if len(recs) % 100 == 0 and recs:
            log(f"    {lang}: {len(recs)} clips")
    json.dump(recs, open(jpath, "w"))
    log(f"  {lang}: fetched {len(recs)} test clips -> {jpath}")
    # disposable parquet cache
    try:
        root = Path(os.environ.get("HF_DATASETS_CACHE", ""))
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass
    return str(jpath)


def main():
    log("Restoring adapters + data indexes from HF...")
    restore_from_hf()

    log("Fetching FLEURS test audio...")
    test_jsons = {}
    for lang in LANGS:
        test_jsons[lang] = ensure_test_audio(lang)

    log("Running evals (small + encLoRA, 256 tokens)...")
    ok = []
    for lang in LANGS:
        sd = f"polywhisper_output_gpu{LANGS[lang][3]}"
        out = f"{sd}/eval_{lang}_pure_fleurs.json"
        if Path(out).exists():
            log(f"  [eval] {lang}: already done")
            ok.append(lang)
            continue
        adapter = Path(sd) / "adapters_v3" / f"{lang}_best{TAG}.pt"
        if not adapter.exists():
            log(f"  [eval] {lang}: adapter missing, SKIP")
            continue
        t0 = time.time()
        cmd = [
            sys.executable, "eval_lang_pure.py",
            "--lang", lang, "--model-size", "small",
            "--adapter", f"{lang}_best{TAG}.pt",
            "--adapter-dir", f"{sd}/adapters_v3",
            "--test-json", test_jsons[lang],
            "--out", out,
            "--max-new-tokens", "256", "--encoder-lora",
        ]
        r = subprocess.run(cmd)
        dt = time.time() - t0
        if r.returncode == 0 and Path(out).exists():
            wer = json.load(open(out)).get("wer")
            log(f"  [eval] {lang}: WER {wer:.1f}% ({dt:.0f}s)")
            ok.append(lang)
            try:
                api.upload_file(path_or_fileobj=out, path_in_repo=out,
                                repo_id=REPO, repo_type="model",
                                commit_message=f"eval {lang}: WER {wer:.1f}")
                log(f"  [upload] {out}")
            except Exception as e:
                log(f"  [upload fail] {out}: {str(e)[:120]}")
        else:
            log(f"  [eval] {lang} FAILED rc={r.returncode}")

    log(f"EVAL DONE ({len(ok)}/{len(LANGS)}): {ok}")
    log("Next: pull evals locally -> polywhisper_output_gpu{0,1}/ -> make_results_table.py")


if __name__ == "__main__":
    main()