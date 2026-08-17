"""Launch bn_v2 retraining after te_v2 completes."""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SAVE = Path("polywhisper_output")
LOG = SAVE / "nohup_te_v2.log"
BN_LOG = SAVE / "nohup_bn_v2.log"
PY = sys.executable
ENV = dict(__import__("os").environ)
for line in Path(".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        ENV.setdefault(k, v)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log("Waiting for te_v2 to complete before launching bn_v2...")
while True:
    if LOG.exists() and "TRAINING COMPLETE" in LOG.read_text(errors="ignore"):
        log("te_v2 complete. Launching bn_v2...")
        cmd = ["caffeinate", "-i", "-s", PY, "train_v3.py",
               "--langs", "bn", "--tag", "_bn_v2", "--batch-size", "4", "--epochs", "3"]
        subprocess.Popen(cmd, stdout=open(BN_LOG, "a"), stderr=subprocess.STDOUT, env=ENV)
        log("bn_v2 launched.")
        break
    time.sleep(60)