"""Run te pure-expert eval once te training truly completes.

Guard: only reacts to TRAINING COMPLETE lines timestamped after the real te
launch (15:27:37 IST 2026-08-12) — the botched first run's stale line is ignored.
Logs to polywhisper_output/wait_te_eval.log.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SAVE = Path("polywhisper_output")
TLOG = SAVE / "nohup_te.log"
CUTOFF = datetime(2026, 8, 12, 15, 27, 37)


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(SAVE / "wait_te_eval.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def tail_done_stamp():
    if not TLOG.exists():
        return None
    stamps = []
    for ln in TLOG.read_text(errors="ignore").splitlines():
        if "TRAINING COMPLETE" not in ln:
            continue
        try:
            ts = datetime.strptime(ln[1:20], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        stamps.append(ts)
    return max(stamps) if stamps else None


def main():
    log(f"watcher: waiting for te TRAINING COMPLETE after {CUTOFF}")
    while True:
        done = tail_done_stamp()
        if done and done > CUTOFF:
            log(f"te TRAINING COMPLETE at {done}. Running pure eval...")
            r = subprocess.run([
                sys.executable, "eval_lang_pure.py", "--lang", "te",
                "--adapter", "te_best_te.pt",
                "--test-json", str(SAVE / "data" / "fleurs_te_in_test.json"),
                "--out", str(SAVE / "eval_te_pure_fleurs.json")],
                capture_output=True, text=True, timeout=7200)
            log(f"pure eval exit={r.returncode}: {(r.stdout or r.stderr)[-400:].replace(chr(10), ' | ')}")
            log("DONE")
            return
        if TLOG.exists() and time.time() - TLOG.stat().st_mtime > 3600:
            log("WARN: te log silent >1h — training may be dead, aborting watcher.")
            return
        time.sleep(120)


if __name__ == "__main__":
    main()