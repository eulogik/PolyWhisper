"""Overnight chain: train te -> bn -> mr with v3 LoRA, auto-eval each on FLEURS.

ta is already running (handled by auto_eval_ta.py). This orchestrator, for each
remaining language: caches the FLEURS test set (so per-epoch val works), launches
batch-4 fp16 training detached under caffeinate, waits for TRAINING COMPLETE,
then runs vanilla whisper-base + pure LoRA expert evals.

Logs to polywhisper_output/auto_pipeline.log. Crash-tolerant: a failed language
is logged and skipped; the chain continues.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SAVE = Path("polywhisper_output")
LANGS = ["te", "bn", "mr"]
FLEURS = {"te": "te_in", "bn": "bn_in", "mr": "mr_in"}
ETA_HOURS = 5.0

E_LOG = SAVE / "auto_pipeline.log"


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(E_LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run(cmd, timeout=None):
    log("RUN " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    log(f"exit={r.returncode}")
    if r.stdout:
        log("  out: " + r.stdout[-400:].replace("\n", " | "))
    if r.returncode != 0 and r.stderr:
        log("  err: " + r.stderr[-400:].replace("\n", " | "))
    return r.returncode


def launch_training(lang):
    tlog = SAVE / f"nohup_{lang}.log"
    cmd = ["caffeinate", "-i", "-s", sys.executable, "train_v3.py",
           "--langs", lang, "--tag", f"_{lang}", "--batch-size", "4", "--epochs", "3"]
    p = subprocess.Popen(cmd, stdout=open(tlog, "a"), stderr=subprocess.STDOUT)
    log(f"Launched {lang} training pid={p.pid} -> {tlog.name}")
    return p


def wait_done(lang, proc):
    tlog = SAVE / f"nohup_{lang}.log"
    t0 = time.time()
    while True:
        if tlog.exists():
            txt = tlog.read_text(errors="ignore")
            if "TRAINING COMPLETE" in txt:
                log(f"{lang}: TRAINING COMPLETE")
                return True
            if "Traceback" in txt:
                log(f"{lang}: FATAL traceback in log. Aborting this language.")
                return False
        if proc.poll() is not None:
            log(f"{lang}: training process exited early (rc={proc.returncode})")
            return False
        if time.time() - t0 > ETA_HOURS * 3600:
            log(f"{lang}: TIMEOUT after {ETA_HOURS}h. Aborting this language.")
            return False
        time.sleep(60)


def evaluate(lang):
    fleurs = FLEURS[lang]
    tjson = SAVE / "data" / f"fleurs_{fleurs}_test.json"
    if not tjson.exists():
        log(f"{lang}: test json missing ({tjson}); fetching")
        run([sys.executable, "fetch_testset.py", fleurs])
    if not tjson.exists():
        log(f"{lang}: still no test json — skipping evals")
        return
    run([sys.executable, "eval_vanilla.py", "--language", lang,
         "--test-json", str(tjson), "--out", str(SAVE / f"eval_vanilla_fleurs_{lang}.json")])
    run([sys.executable, "eval_lang_pure.py", "--lang", lang,
         "--adapter", f"{lang}_best_{lang}.pt",
         "--test-json", str(tjson), "--out", str(SAVE / f"eval_{lang}_pure_fleurs.json")])


def main():
    log("Pipeline started: waiting 10 min for ta to fully detach before te fetch")
    time.sleep(600)
    for lang in LANGS:
        log(f"===== {lang} =====")
        evaluate(lang)
        p = launch_training(lang)
        if wait_done(lang, p):
            evaluate(lang)
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()