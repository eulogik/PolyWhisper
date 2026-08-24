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
CUDA_IDX = {"gpu0": "0", "gpu1": "1"}
EPOCHS = 3
# T4 (14.5GB) OOMs at batch 8 with small+encLoRA @ seq 3000 (v6 run died at 65min,
# both GPUs, in cross-attn k_proj hook). Batch 4 fits comfortably; ~0.6-1.0 steps/s.
BATCH = 4
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


def upload_tree(dirname, remote_snapshot=None):
    """Upload all .pt / .json artifacts in dirname to HF, if changed.
    Rate-limit aware: HF free tier caps repo commits at 128/hour. The monitor
    loop calls this every 30s; we only push files whose size actually changed
    (snapshot check) AND back off when HF returns 429, so a 2-3h session stays
    far below the cap while still persisting every checkpoint within minutes."""
    local = Path(dirname)
    if not local.exists():
        return
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
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                            repo_id=REPO, repo_type="model",
                            commit_message=f"ckpt {rel}")
            log(f"  uploaded {rel}")
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                # parse "Retry after N seconds" and sleep once (max 120s here;
                # the outer 30s loop will retry on next tick)
                import re as _re
                m = _re.search(r"Retry after (\d+)", msg)
                wait = min(int(m.group(1)) + 5, 120) if m else 60
                log(f"  rate-limited on {rel}; sleeping {wait}s")
                time.sleep(wait)
                try:
                    api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                                    repo_id=REPO, repo_type="model",
                                    commit_message=f"ckpt {rel}")
                    log(f"  uploaded {rel} (after backoff)")
                    continue
                except Exception as e2:
                    log(f"  upload fail {rel} after backoff: {e2}")
            else:
                log(f"  upload fail {rel}: {msg[:200]}")


def remote_snapshot():
    """{path_in_repo: size} for all files in the HF repo ({} if unreachable)."""
    try:
        from huggingface_hub import list_repo_tree
        return {f.path: f.size for f in
                list_repo_tree(REPO, recursive=True, expand=True, repo_type="model")}
    except Exception:
        try:
            info = api.model_info(REPO, files_metadata=True, repo_type="model")
            return {s.rfilename: s.size for s in info.siblings}
        except Exception:
            return {}


def artifact_ok(p):
    """True if p is a complete, loadable artifact (rejects truncated saves)."""
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
        "--max-runtime-hours", "10",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = CUDA_IDX[gpu]
    # isolate datasets cache per GPU to avoid 2× peak and rm race
    env["HF_DATASETS_CACHE"] = f"/kaggle/working/hf_datasets_cache_{gpu}"
    log(f"GPU{gpu} (cuda:{CUDA_IDX[gpu]}): {cmd}  HF_DATASETS_CACHE={env['HF_DATASETS_CACHE']}")
    return subprocess.Popen(cmd, env=env,
                            stdout=open(f"{save_dir}_run.log", "a"),
                            stderr=subprocess.STDOUT)


def eval_lang(lang, save_dir):
    """FLEURS test eval for one lang after training. Returns result json path."""
    test_json = f"{save_dir}/data/fleurs_{lang}_in_test.json"
    if not Path(test_json).exists():
        log(f"  [eval] {lang}: no cached fleurs test, skipping")
        return
    adapter = f"{lang}_best{TAG}.pt"
    adapter_path = Path(save_dir) / "adapters_v3" / adapter
    out = f"{save_dir}/eval_{lang}_pure_fleurs.json"
    if Path(out).exists():
        log(f"  [eval] {lang}: already done")
        return
    if not adapter_path.exists() or not artifact_ok(adapter_path):
        log(f"  [eval] {lang}: adapter missing/corrupt — skipping (was training interrupted?)")
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


def clean_datasets_cache():
    """Delete disposable datasets parquet cache (re-downloadable; frees GBs after a dead session)."""
    root = Path(os.environ.get(
        "HF_DATASETS_CACHE", str(Path.home() / ".cache" / "huggingface" / "datasets")))
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
        log(f"Cleared disposable datasets cache: {root}")


def main():
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if not HF_TOKEN:
        log("WARNING: HF_TOKEN not set — cannot persist to HF. Add Kaggle secret 'HF_TOKEN'.")
    else:
        api.token = HF_TOKEN

    clean_datasets_cache()

    log("Restoring state from HF...")
    for gpu, d in SAVE_DIRS.items():
        fetch_remote(d)

    log(f"Disk free before launch: {shutil.disk_usage('.')[2]/1e9:.1f} GB "
        f"(audio ~3.4GB/lang; datasets cache is freed per language)")

    log("Launching 2 GPU jobs...")
    procs = {}
    for gpu, langs in LANG_SPLIT.items():
        d = SAVE_DIRS[gpu]
        Path(d).mkdir(parents=True, exist_ok=True)
        procs[gpu] = run_gpu(gpu, langs, d)

    log("Monitoring (uploading checkpoints to HF every 30s)...")
    while any(p.poll() is None for p in procs.values()):
        time.sleep(30)
        snap = remote_snapshot()
        for d in SAVE_DIRS.values():
            upload_tree(d, snap)
            snap = remote_snapshot()  # refresh after each dir's uploads

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
                # also mirror the run log to HF so a dead session is diagnosable
                try:
                    api.upload_file(path_or_fileobj=f"{SAVE_DIRS[gpu]}_run.log",
                                    path_in_repo=f"{SAVE_DIRS[gpu]}_run.log",
                                    repo_id=REPO, repo_type="model",
                                    commit_message=f"runlog {gpu}")
                except Exception:
                    pass
            try:
                crash = Path(SAVE_DIRS[gpu]) / "crash_v3.log"
                if crash.exists():
                    log(f"  >>> GPU {gpu} crash_v3.log tail:")
                    for line in crash.read_text(errors="replace").splitlines()[-25:]:
                        log(f"  | {line}")
            except Exception as e:
                log(f"  | (no crash log: {e})")
    log("Both GPU jobs finished.")
    log(f"Disk free now: {shutil.disk_usage('.')[2]/1e9:.1f} GB")

    snap = remote_snapshot()
    for gpu, d in SAVE_DIRS.items():
        upload_tree(d, snap)

    log("Running FLEURS evals...")
    for gpu, langs in LANG_SPLIT.items():
        for lang in langs:
            try:
                eval_lang(lang, SAVE_DIRS[gpu])
            except Exception as e:
                log(f"  [eval] {lang} FAILED: {e}")

    for gpu, d in SAVE_DIRS.items():
        upload_tree(d, remote_snapshot())

    log("ALL DONE. Re-run make_results_table.py locally for the final table.")


if __name__ == "__main__":
    main()
