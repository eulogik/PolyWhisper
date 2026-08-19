#!/usr/bin/env python3
"""Kaggle 2xT4 product-training orchestrator (resumable, disconnect-safe).

Strategy (product target):
  - Backbone: whisper-small (frozen) + per-language LoRA experts (r16, encoder LoRA)
  - Data: IndicVoices-ST train for hi/ta/te/bn/mr (real conversational speech)
  - Eval: FLEURS test per language (standard benchmark), ortho-normalized offline
  - Two GPUs in parallel: GPU0 = hi,te,bn | GPU1 = ta,mr (bn/te hardest, split)
  - State + adapters persisted to HF Hub after every epoch; nothing lost on
    session death. Re-run the notebook cell to resume.

Flow:
  1. Download state+adapters from HF (if any) into save dirs
  2. Spawn two training processes (one per GPU, isolated --save-dir)
  3. Monitor: upload new adapters/state to HF every 30s
  4. When both finish: run FLEURS evals, upload results
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = "eulogik/polywhisper"
LANG_SPLIT = {"gpu0": ["hi", "te", "bn"], "gpu1": ["ta", "mr"]}
SAVE_DIRS = {"gpu0": "polywhisper_output_gpu0", "gpu1": "polywhisper_output_gpu1"}
EPOCHS = 3
BATCH = 8
TAG = "_prod"

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    raise SystemExit("pip install huggingface_hub")

api = HfApi()


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def remote_files():
    try:
        return set(api.list_repo_files(REPO, repo_type="model"))
    except Exception:
        return set()


def fetch_remote(dirname):
    """Download state + adapters from HF for this save dir (if any)."""
    local = Path(dirname)
    local.mkdir(parents=True, exist_ok=True)
    for f in remote_files():
        if not f.startswith(dirname + "/"):
            continue
        dest = local / Path(f).name
        if dest.exists():
            continue
        try:
            p = hf_hub_download(repo_id=REPO, filename=f, repo_type="model")
            shutil.copy(p, dest)
            log(f"  restored {f}")
        except Exception as e:
            log(f"  skip {f}: {e}")


def upload_tree(dirname):
    """Upload all .pt / .json artifacts in dirname to HF, if changed."""
    local = Path(dirname)
    if not local.exists():
        return
    for p in sorted(local.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix not in (".pt", ".json"):
            continue
        rel = f"{dirname}/{p.name}"
        try:
            if not remote_files() or rel in remote_files():
                info = api.model_info(REPO, files_metadata=True, repo_type="model")
                sizes = {s.rfilename: s.size for s in info.siblings}
                if rel in sizes and sizes[rel] == p.stat().st_size:
                    continue
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                            repo_id=REPO, repo_type="model",
                            commit_message=f"ckpt {rel}")
            log(f"  uploaded {rel}")
        except Exception as e:
            log(f"  upload fail {rel}: {e}")


def run_gpu(gpu, langs, save_dir):
    cmd = [
        sys.executable, "train_v3.py",
        "--langs", ",".join(langs),
        "--model-size", "small",
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH),
        "--encoder-lora",
        "--tag", TAG,
        "--save-dir", save_dir,
        "--max-runtime-hours", "10",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log(f"GPU{gpu}: {cmd}")
    return subprocess.Popen(cmd, env=env,
                            stdout=open(f"{save_dir}_run.log", "a"),
                            stderr=subprocess.STDOUT)


def eval_lang(lang, save_dir):
    """FLEURS test eval for one lang after training. Returns result json path."""
    from train_v3 import LANG_TOKENS
    test_json = f"{save_dir}/data/fleurs_{lang}_in_test.json"
    if not Path(test_json).exists():
        log(f"  [eval] {lang}: no cached fleurs test, skipping")
        return
    adapter = f"{lang}_best{TAG}.pt"
    out = f"{save_dir}/eval_{lang}_pure_fleurs.json"
    if Path(out).exists():
        log(f"  [eval] {lang}: already done")
        return
    cmd = [
        sys.executable, "eval_lang_pure.py",
        "--lang", lang, "--model-size", "small",
        "--adapter", adapter,
        "--test-json", test_json,
        "--out", out,
        "--max-new-tokens", "256", "--encoder-lora",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(cmd, env=env, check=True)
    log(f"  [eval] {lang}: {out}")
    return out


def main():
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if not HF_TOKEN:
        log("WARNING: HF_TOKEN not set — cannot persist to HF. Add Kaggle secret 'HF_TOKEN'.")
    else:
        api.token = HF_TOKEN

    log("Restoring state from HF...")
    for gpu, d in SAVE_DIRS.items():
        fetch_remote(d)

    log("Launching 2 GPU jobs...")
    procs = {}
    for gpu, langs in LANG_SPLIT.items():
        d = SAVE_DIRS[gpu]
        Path(d).mkdir(parents=True, exist_ok=True)
        procs[gpu] = run_gpu(gpu, langs, d)

    log("Monitoring (uploading checkpoints to HF every 30s)...")
    while any(p.poll() is None for p in procs.values()):
        time.sleep(30)
        for d in SAVE_DIRS.values():
            upload_tree(d)

    for gpu, p in procs.items():
        code = p.wait()
        log(f"GPU {gpu} exited rc={code}")
    log("Both GPU jobs finished.")

    for gpu, d in SAVE_DIRS.items():
        upload_tree(d)

    log("Running FLEURS evals...")
    for gpu, langs in LANG_SPLIT.items():
        for lang in langs:
            try:
                eval_lang(lang, SAVE_DIRS[gpu])
            except Exception as e:
                log(f"  [eval] {lang} FAILED: {e}")

    for gpu, d in SAVE_DIRS.items():
        upload_tree(d)

    log("ALL DONE. Re-run make_results_table.py locally for the final table.")


if __name__ == "__main__":
    main()
