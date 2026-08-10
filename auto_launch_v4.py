"""
Waits for the full eval (pid 55028) to finish, then self-launches the v4
retrain pipeline watcher (A1 -> A2 -> P1 -> P2).

Eval is considered done when BOTH:
  1. the eval process no longer exists, and
  2. its log contains the completion marker "Saved per-sample results".
If the eval crashes (process gone, no marker) the launcher aborts.

Log: polywhisper_output/nohup_auto_launch_v4.log
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SAVE_DIR = Path("./polywhisper_output")
EVAL_LOG = SAVE_DIR / "nohup_eval_final.log"
EVAL_PID = 55028
MARKER = "Saved per-sample results"
PYTHON = str(Path(".venv/bin/python").absolute())
LAUNCHER_LOG = SAVE_DIR / "nohup_auto_launch_v4.log"
WATCHER_LOG = SAVE_DIR / "nohup_watcher_v4.log"


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LAUNCHER_LOG, "a") as f:
        f.write(line + "\n")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def wait_for_eval():
    while pid_alive(EVAL_PID):
        time.sleep(60)
    time.sleep(10)  # flush grace for the eval's final writes
    if EVAL_LOG.exists() and MARKER in EVAL_LOG.read_text(errors="ignore"):
        return True
    log("FATAL: eval process ended WITHOUT completion marker — aborting (no auto-launch).")
    if EVAL_LOG.exists():
        log("Eval log tail:")
        log("\n".join(EVAL_LOG.read_text(errors="ignore").splitlines()[-15:]))
    return False


if __name__ == "__main__":
    log("Auto-launcher v4 waiting for eval pid %d..." % EVAL_PID)
    if EVAL_LOG.exists() and MARKER in EVAL_LOG.read_text(errors="ignore") and not pid_alive(EVAL_PID):
        log("Eval already complete.")
    elif not wait_for_eval():
        sys.exit(1)
    log("Eval complete. Launching v4 pipeline watcher...")
    WATCHER_LOG.unlink(missing_ok=True)
    pid = subprocess.Popen([PYTHON, "watch_and_launch_router_v4.py"],
                           stdout=open(WATCHER_LOG, "a"), stderr=subprocess.STDOUT).pid
    log(f"Watcher v4 launched (pid {pid}, log {WATCHER_LOG})")
