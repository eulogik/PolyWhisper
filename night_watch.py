"""Overnight supervisor: restart crashed trainings + takeover bn/mr if pipeline dies.

Responsibilities (poll every 4 min, log to night_audit.log):
  1. ta/te/bn/mr: if a training process died WITHOUT a fresh TRAINING COMPLETE,
     relaunch it (resume-safe: state file + batch-size guard, log rotated).
  2. mr takeover: if auto_pipeline.py is dead AND bn is complete, run bn's
     post-training evals then launch mr, and mr's evals when mr completes.
  3. Every poll: write a heartbeat snapshot for morning triage.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SAVE = Path("polywhisper_output")
AUDIT = SAVE / "night_audit.log"
PY = sys.executable
ENV = dict(os.environ)
for line in Path(".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        ENV.setdefault(k, v)

CMDS = {
    "ta": ["train_v3.py", "--langs", "ta", "--tag", "_ta", "--batch-size", "4", "--epochs", "3"],
    "te": ["train_v3.py", "--langs", "te", "--tag", "_te", "--batch-size", "4", "--epochs", "3"],
    "bn": ["train_v3.py", "--langs", "bn", "--tag", "_bn", "--batch-size", "4", "--epochs", "3"],
    "mr": ["train_v3.py", "--langs", "mr", "--tag", "_mr", "--batch-size", "4", "--epochs", "3"],
}
FLEURS = {"te": "te_in", "bn": "bn_in", "mr": "mr_in"}


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(AUDIT, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def proc_running(name):
    r = subprocess.run(["pgrep", "-f", f"train_v3.py --langs {name} "], capture_output=True, text=True)
    return r.returncode == 0


def log_has(tag, text):
    p = SAVE / f"nohup_{tag}.log"
    return p.exists() and text in p.read_text(errors="ignore")


def rel_tail(tag, n=1):
    p = SAVE / f"nohup_{tag}.log"
    if not p.exists():
        return "(no log)"
    lines = [l for l in p.read_text(errors="ignore").splitlines() if l.strip()]
    return lines[-n][-140:] if lines else "(empty)"


def relaunch(tag):
    tlog = SAVE / f"nohup_{tag}.log"
    if tlog.exists():
        old = SAVE / f"nohup_{tag}.log.prev"
        log(f"{tag}: rotating stale log before relaunch")
        tlog.rename(old)
    cmd = ["caffeinate", "-i", "-s", PY, *CMDS[tag]]
    p = subprocess.Popen(cmd, stdout=open(tlog, "a"), stderr=subprocess.STDOUT, env=ENV)
    log(f"{tag}: RELAUNCHED pid={p.pid} (resume via state file)")
    return p


def run_eval(tag):
    fleurs = FLEURS[tag]
    tjson = SAVE / "data" / f"fleurs_{fleurs}_test.json"
    out_v = SAVE / f"eval_vanilla_fleurs_{tag}.json"
    out_p = SAVE / f"eval_{tag}_pure_fleurs.json"
    if not tjson.exists():
        log(f"{tag}: missing test json, fetching")
        subprocess.run([PY, "fetch_testset.py", fleurs], env=ENV, timeout=3600)
    if not out_v.exists() and tjson.exists():
        log(f"{tag}: vanilla eval")
        subprocess.run([PY, "eval_vanilla.py", "--language", tag, "--test-json", str(tjson),
                        "--out", str(out_v)], env=ENV, timeout=7200)
    if not out_p.exists() and tjson.exists():
        log(f"{tag}: pure eval")
        subprocess.run([PY, "eval_lang_pure.py", "--lang", tag, "--adapter", f"{tag}_best_{tag}.pt",
                        "--test-json", str(tjson), "--out", str(out_p)], env=ENV, timeout=7200)


def pipeline_alive():
    r = subprocess.run(["pgrep", "-f", "auto_pipeline.py"], capture_output=True, text=True)
    return r.returncode == 0


def main():
    log("night supervisor started")
    takeover_bn = False
    while True:
        try:
            for tag in ["ta", "te"]:
                if proc_running(tag):
                    log(f"{tag}: live | {rel_tail(tag)}")
                elif log_has(tag, "TRAINING COMPLETE"):
                    log(f"{tag}: complete (no action)")
                else:
                    log(f"{tag}: DEAD without completion -> relaunch")
                    relaunch(tag)
            if not pipeline_alive():
                log("PIPELINE DEAD -> supervisor takeover mode")
                for tag in ["bn", "mr"]:
                    if log_has(tag, "TRAINING COMPLETE"):
                        log(f"{tag}: complete")
                        run_eval(tag)
                    elif proc_running(tag):
                        log(f"{tag}: live | {rel_tail(tag)}")
                    else:
                        relaunch(tag)
                        break
            else:
                for tag in ["bn", "mr"]:
                    if proc_running(tag):
                        log(f"{tag}: live | {rel_tail(tag)}")
                    else:
                        log(f"{tag}: waiting (pipeline owns this stage)")
        except Exception as e:
            log(f"SUPERVISOR ERROR: {e}")
        time.sleep(240)


if __name__ == "__main__":
    main()