"""Re-eval all 4 languages (vanilla + pure) with max_new_tokens=256.
Writes to polywhisper_output/eval_{lang}_{vanilla|pure}_fleurs_256.json.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

PY = sys.executable
SAVE = Path("polywhisper_output")
FLEURS = {"ta": "ta_in", "te": "te_in", "bn": "bn_in", "mr": "mr_in"}
MAX_TOK = 256


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


PURE_ADAPTERS = {
    "ta": "ta_best_ta.pt",
    "te": "te_best_te_v2.pt",
    "bn": "bn_best_bn_v2.pt",
    "mr": "mr_best_mr.pt",
}

def run(cmd, label):
    log(f"START {label}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.stdout:
        for ln in r.stdout.splitlines()[-3:]:
            log(f"  {ln}")
    if r.returncode != 0 and r.stderr:
        for ln in r.stderr.splitlines()[-3:]:
            log(f"  ERR {ln}")
    log(f"DONE  {label} exit={r.returncode}")
    return r.returncode


def main():
    for lang, fleurs in FLEURS.items():
        tjson = SAVE / "data" / f"fleurs_{fleurs}_test.json"
        run([PY, "eval_vanilla.py",
             "--language", lang,
             "--test-json", str(tjson),
             "--out", str(SAVE / f"eval_vanilla_fleurs_{lang}_256.json"),
             "--max-new-tokens", str(MAX_TOK)],
            f"vanilla-{lang}-256")
        run([PY, "eval_lang_pure.py",
             "--lang", lang,
             "--adapter", PURE_ADAPTERS[lang],
             "--test-json", str(tjson),
             "--out", str(SAVE / f"eval_{lang}_pure_fleurs_256.json"),
             "--max-new-tokens", str(MAX_TOK)],
            f"pure-{lang}-256")
    log("ALL RE-EVALS DONE")


if __name__ == "__main__":
    main()