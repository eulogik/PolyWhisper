"""
Waits for P2 (v4 jet router, pid 58661 / marker in nohup_v4_phase2.log) to finish,
then launches the full eval of the v4 router on the 3129-sample ortho test set.

Eval args pass the v4 checkpoints explicitly (eval_router.py defaults point at v3).
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

SAVE_DIR = Path("./polywhisper_output")
ADAPTERS = SAVE_DIR / "adapters_v3"
P2_LOG = SAVE_DIR / "nohup_v4_phase2.log"
P2_PID_CHECK = "train_router.py --phase 2"
MARKER = "ROUTER TRAINING COMPLETE"
PYTHON = str(Path(".venv/bin/python").absolute())
LAUNCHER_LOG = SAVE_DIR / "nohup_auto_eval_v4.log"
EVAL_LOG = SAVE_DIR / "nohup_eval_v4.log"


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LAUNCHER_LOG, "a") as f:
        f.write(line + "\n")


def phase2_alive():
    try:
        out = subprocess.run(["pgrep", "-f", "train_router.py --phase 2"],
                             capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return True


def wait_p2():
    while phase2_alive():
        time.sleep(60)
    time.sleep(10)
    if P2_LOG.exists() and MARKER in P2_LOG.read_text(errors="ignore"):
        return True
    log("FATAL: P2 ended without completion marker — not launching eval.")
    return False


if __name__ == "__main__":
    log("Auto-eval v4 armed: waiting for P2 to finish...")
    if P2_LOG.exists() and MARKER in P2_LOG.read_text(errors="ignore") and not phase2_alive():
        log("P2 already complete.")
    elif not wait_p2():
        raise SystemExit(1)
    log("P2 complete. Launching v4 eval...")
    EVAL_LOG.unlink(missing_ok=True)
    cmd = ["eval_router.py",
           "--test-json", str(SAVE_DIR / "data" / "hinglish_codeswitch_test_ortho.json"),
           "--router", str(ADAPTERS / "router_best_v4.pt"),
           "--en-adapter", str(ADAPTERS / "en_router_best_v4.pt"),
           "--hi-adapter", str(ADAPTERS / "hi_router_best_v4.pt"),
           "--beams", "1"]
    pid = subprocess.Popen([PYTHON] + cmd, stdout=open(EVAL_LOG, "a"), stderr=subprocess.STDOUT).pid
    log(f"Eval v4 launched (pid {pid}, log {EVAL_LOG}): {' '.join(cmd)}")