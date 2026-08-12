"""Watch Tamil v3 training; when it finishes, run the FLEURS-ta evals.

1. Waits for "TRAINING COMPLETE" in the ta log (or 15 min of log silence beyond ETA).
2. Runs vanilla whisper-base on fleurs_ta_in_test.json.
3. Runs the pure ta LoRA expert on the same test json.

Logs to polywhisper_output/auto_eval_ta.log.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SAVE = Path("polywhisper_output")
LOG = SAVE / "nohup_ta.log"
VANILLA_JSON = SAVE / "data" / "fleurs_ta_in_test.json"
OUT_VANILLA = SAVE / "eval_vanilla_fleurs_ta.json"
OUT_PURE = SAVE / "eval_ta_pure_fleurs.json"
ADAPTER = "ta_best_ta.pt"
E_LOG = SAVE / "auto_eval_ta.log"


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(E_LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run(cmd):
    log("RUN " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"exit={r.returncode}")
    if r.stdout:
        log("  stdout tail: " + r.stdout[-500:].replace("\n", " | "))
    if r.returncode != 0 and r.stderr:
        log("  stderr tail: " + r.stderr[-500:].replace("\n", " | "))
    return r.returncode


def main():
    log("Watcher started. Waiting for Tamil training to complete...")
    done = False
    while not done:
        if LOG.exists():
            txt = LOG.read_text(errors="ignore")
            if "TRAINING COMPLETE" in txt:
                done = True
            elif "CRASH" in txt or "Traceback" in txt:
                log("FATAL: training crashed. Aborting watcher.")
                sys.exit(1)
        mtime = LOG.stat().st_mtime if LOG.exists() else 0
        if time.time() - mtime > 3600:
            log("WARNING: training log silent >1h — assuming stalled or killed. Aborting.")
            sys.exit(2)
        time.sleep(120)
    log("Training complete! Running evals...")

    if not VANILLA_JSON.exists():
        log(f"FATAL: missing {VANILLA_JSON}")
        sys.exit(3)

    run([sys.executable, "eval_vanilla.py",
         "--language", "ta", "--test-json", str(VANILLA_JSON),
         "--out", str(OUT_VANILLA)])
    run([sys.executable, "eval_lang_pure.py",
         "--lang", "ta", "--adapter", ADAPTER,
         "--test-json", str(VANILLA_JSON), "--out", str(OUT_PURE)])
    log("ALL DONE")


if __name__ == "__main__":
    main()