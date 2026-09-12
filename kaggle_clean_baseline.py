#!/usr/bin/env python3
"""Kaggle v7 CLEAN baseline: train te/bn/mr without augmentation.

Purpose: get verified FLEURS WER numbers for clean (no augment) training,
to confirm the augmentation asymmetry finding in the paper.

Strategy:
  - GPU0: te, bn (2 langs)
  - GPU1: mr (1 lang)
  - NO SpecAugment, NO speed perturbation
  - 3 epochs per language (matches v9 clean hi/ta recipe)
  - Eval on FLEURS after training
  - Persist to HF Hub
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
LANG_SPLIT = {"gpu0": ["te", "bn"], "gpu1": ["mr"]}
SAVE_DIRS = {"gpu0": "polywhisper_output_gpu0_clean", "gpu1": "polywhisper_output_gpu1_clean"}
CUDA_IDX = {"gpu0": "0", "gpu1": "1"}
EPOCHS = 3
BATCH = 4
TAG = "_clean"

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
    local = Path(dirname)
    local.mkdir(parents=True, exist_ok=True)
    for f in remote_files():
        if not f.startswith(dirname + "/"):
            continue
        rel = f[len(dirname) + 1:]
        dest = local / rel
        if dest.exists():
            continue
        try:
            p = hf_hub_download(repo_id=REPO, filename=f, repo_type="model")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)
            log(f"  restored {f}")
        except Exception as e:
            log(f"  skip {f}: {e}")


_upload_times = {}


def upload_tree(dirname, remote_snapshot=None, force=False):
    local = Path(dirname)
    if not local.exists():
        return
    now = time.time()
    for p in sorted(local.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix not in (".pt", ".json"):
            continue
        if not artifact_ok(p):
            log(f"  SKIP upload {p} (corrupt/incomplete artifact)")
            continue
        rel = f"{dirname}/{p.relative_to(local)}"
        try:
            if remote_snapshot is not None and rel in remote_snapshot:
                if remote_snapshot[rel] == p.stat().st_size:
                    continue
            if not force and now - _upload_times.get(rel, 0) < 600:
                continue
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                            repo_id=REPO, repo_type="model",
                            commit_message=f"clean baseline {rel}")
            _upload_times[rel] = now
            log(f"  uploaded {rel}")
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                import re as _re
                m = _re.search(r"Retry after (\d+)", msg)
                wait = min(int(m.group(1)) + 5, 300) if m else 60
                log(f"  rate-limited on {rel}; sleeping {wait}s")
                time.sleep(wait)
                try:
                    api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                                    repo_id=REPO, repo_type="model",
                                    commit_message=f"clean baseline {rel}")
                    _upload_times[rel] = time.time()
                    log(f"  uploaded {rel} (after backoff)")
                except Exception as e2:
                    log(f"  upload fail {rel} after backoff: {str(e2)[:150]}")
            else:
                log(f"  upload fail {rel}: {msg[:200]}")


def remote_snapshot():
    try:
        from huggingface_hub import list_repo_tree
        out = {}
        for f in list_repo_tree(REPO, recursive=True, expand=True, repo_type="model"):
            size = getattr(f, "size", None)
            if size is not None:
                out[f.path] = size
        return out
    except Exception:
        try:
            info = api.model_info(REPO, files_metadata=True, repo_type="model")
            return {s.rfilename: s.size for s in info.siblings
                    if getattr(s, "size", None) is not None}
        except Exception:
            return {}


def artifact_ok(p):
    try:
        if p.suffix == ".pt":
            import zipfile
            return any(n.endswith("/data/0") for n in zipfile.ZipFile(p).namelist())
        if p.suffix == ".json":
            with open(p) as f:
                j = json.load(f)
            return isinstance(j, (dict, list)) and len(j) > 0
    except Exception:
        return False
    return False


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
        "--max-runtime-hours", "11",
        "--wer-eval-every", "500",
        "--no-spec-augment",
        "--no-speed-perturb",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = CUDA_IDX[gpu]
    env["HF_DATASETS_CACHE"] = f"/kaggle/working/hf_datasets_cache_{gpu}"
    log(f"GPU{gpu} (cuda:{CUDA_IDX[gpu]}): CLEAN training {langs}")
    log(f"  cmd: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env,
                            stdout=open(f"{save_dir}_run.log", "a"),
                            stderr=subprocess.STDOUT)


def eval_lang(lang, save_dir):
    test_json = f"{save_dir}/data/fleurs_{lang}_in_test.json"
    if not Path(test_json).exists():
        log(f"  [eval] {lang}: no cached fleurs test, skipping")
        return
    adapter_wer = f"{lang}_best_wer{TAG}.pt"
    adapter_ce = f"{lang}_best{TAG}.pt"
    adapter_path = Path(save_dir) / "adapters_v3"
    adapter = adapter_wer if (adapter_path / adapter_wer).exists() else adapter_ce
    adapter_full = adapter_path / adapter
    out = f"{save_dir}/eval_{lang}_pure_fleurs_clean.json"
    if Path(out).exists():
        log(f"  [eval] {lang}: already done")
        return
    if not adapter_full.exists() or not artifact_ok(adapter_full):
        log(f"  [eval] {lang}: adapter missing/corrupt — skipping")
        return
    cmd = [
        sys.executable, "eval_lang_pure.py",
        "--lang", lang, "--model-size", "small",
        "--adapter", adapter,
        "--adapter-dir", f"{save_dir}/adapters_v3",
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
        log("WARNING: HF_TOKEN not set — cannot persist to HF.")
    else:
        api.token = HF_TOKEN

    log("=== v7 CLEAN BASELINE: te/bn/mr, no augmentation, 3 epochs ===")
    log("Restoring state from HF...")
    for gpu, d in SAVE_DIRS.items():
        fetch_remote(d)

    log("Launching 2 GPU jobs (CLEAN training)...")
    procs = {}
    for gpu, langs in LANG_SPLIT.items():
        d = SAVE_DIRS[gpu]
        Path(d).mkdir(parents=True, exist_ok=True)
        procs[gpu] = run_gpu(gpu, langs, d)

    log("Monitoring (uploading checkpoints every 30s)...")
    last_snap = 0.0
    snap = {}
    while any(p.poll() is None for p in procs.values()):
        time.sleep(30)
        if time.time() - last_snap > 300:
            snap = remote_snapshot()
            last_snap = time.time()
        for d in SAVE_DIRS.values():
            upload_tree(d, snap)

    for gpu, p in procs.items():
        code = p.wait()
        log(f"GPU {gpu} exited rc={code}")
        if code != 0:
            log(f"  >>> GPU {gpu} FAILED — run log tail:")
            try:
                for line in open(f"{SAVE_DIRS[gpu]}_run.log", errors="replace").read().splitlines()[-40:]:
                    log(f"  | {line}")
            except Exception as e:
                log(f"  | (no run log: {e})")
            else:
                try:
                    api.upload_file(path_or_fileobj=f"{SAVE_DIRS[gpu]}_run.log",
                                    path_in_repo=f"{SAVE_DIRS[gpu]}_run.log",
                                    repo_id=REPO, repo_type="model",
                                    commit_message=f"clean runlog {gpu}")
                except Exception:
                    pass

    log("Both GPU jobs finished.")
    snap = remote_snapshot()
    for gpu, d in SAVE_DIRS.items():
        upload_tree(d, snap, force=True)

    log("Running FLEURS evals...")
    for gpu, langs in LANG_SPLIT.items():
        for lang in langs:
            try:
                eval_lang(lang, SAVE_DIRS[gpu])
            except Exception as e:
                log(f"  [eval] {lang} FAILED: {e}")

    for gpu, d in SAVE_DIRS.items():
        upload_tree(d, remote_snapshot(), force=True)

    log("=== CLEAN BASELINE DONE ===")
    log("Expected WERs (approximate, from paper): te~106, bn~199, mr~170")
    log("If actual WERs differ significantly, the augmentation asymmetry finding changes.")


if __name__ == "__main__":
    main()
