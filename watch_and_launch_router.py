"""
Auto-pipeline: runs the full 15h unattended schedule.

  A1 (en retrain, running)  ->  A2 (hi encoder-LoRA, 2ep)  ->  router phase 1 (2ep)
  ->  router phase 2 (2ep, joint)  ->  done flag.

Stage logs: nohup_v3_a2.log, nohup_router_phase1.log, nohup_router_phase2.log
Guards: double-launch flag file; FATAL-crash detection per stage; per-stage timeouts.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SAVE_DIR = Path("./polywhisper_output")
WATCHER_LOG = SAVE_DIR / "nohup_watcher.log"
FLAG = SAVE_DIR / "PIPELINE_DONE.flag"
EN_BEST = SAVE_DIR / "adapters_v3" / "en_best.pt"
PYTHON = str(Path(".venv/bin/python").absolute())

STAGES = [
    {"log": SAVE_DIR / "nohup_v3.log", "done": "TRAINING COMPLETE", "timeout_h": 8,
     "cmd": None},  # A1 already running
    {"log": SAVE_DIR / "nohup_v3_a2.log", "done": "TRAINING COMPLETE", "timeout_h": 5,
     "cmd": ["train_v3.py", "--langs", "hi", "--encoder-lora", "--epochs", "2", "--lr", "1e-4"]},
    {"log": SAVE_DIR / "nohup_router_phase1.log", "done": "ROUTER TRAINING COMPLETE", "timeout_h": 6,
     "cmd": ["train_router.py", "--phase", "1", "--epochs", "2"]},
    {"log": SAVE_DIR / "nohup_router_phase2.log", "done": "ROUTER TRAINING COMPLETE", "timeout_h": 6,
     "cmd": ["train_router.py", "--phase", "2", "--epochs", "2", "--lr", "1e-4",
             "--lambda-router", "1.0", "--router-init", str(SAVE_DIR / "adapters_v3" / "router_best.pt")]},
]


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(WATCHER_LOG, "a") as f:
        f.write(line + "\n")


def launch(cmd, out_log):
    # truncate stale log so crash detection never sees content from a previous run
    out_log.unlink(missing_ok=True)
    pid = subprocess.Popen([PYTHON] + cmd, stdout=open(out_log, "a"), stderr=subprocess.STDOUT).pid
    log(f"Launched: {' '.join(cmd)} (pid {pid}, log {out_log})")
    return pid


def wait_stage(stage):
    """Wait until stage log contains its completion string, or a FATAL appears. Returns True on completion."""
    t0 = time.time()
    last_size = 0
    while time.time() - t0 < stage["timeout_h"] * 3600:
        if stage["log"].exists():
            text = stage["log"].read_text(errors="ignore")
            if stage["done"] in text:
                return True
            if "FATAL" in text or "Traceback (most recent call last)" in text:
                log(f"STAGE FAILED: {stage['log'].name} shows a fatal error")
                return False
        time.sleep(60)
    log(f"STAGE TIMEOUT after {stage['timeout_h']}h: {stage['log'].name}")
    return False


if __name__ == "__main__":
    log("Watcher started (A1 -> A2 -> router P1 -> router P2).")
    if FLAG.exists():
        log("Pipeline already completed. Exiting.")
        sys.exit(0)

    if not wait_stage(STAGES[0]):
        log("A1 failed or timed out. Aborting pipeline.")
        sys.exit(1)
    time.sleep(30)
    if not EN_BEST.exists():
        log(f"FATAL: {EN_BEST} missing after A1. Aborting.")
        sys.exit(1)
    log("A1 complete. Starting A2 (hi encoder-LoRA).")

    for stage in STAGES[1:]:
        launch(stage["cmd"], stage["log"])
        if not wait_stage(stage):
            log(f"Pipeline aborted at stage: {stage['log'].name}")
            sys.exit(1)
        time.sleep(30)

    FLAG.write_text(datetime.now().isoformat())
    log("PIPELINE COMPLETE: A1 + A2 + router phases 1-2 all finished.")
