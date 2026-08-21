"""
B4: Evaluate the code-switch LoRA router on the Hinglish test set.
Metrics:
  - WER / CER (normalized, via eval_metrics)
  - Router token accuracy vs script-detected language labels
  - Per-sample JSON for error analysis
Run: HF_TOKEN=... .venv/bin/python eval_router.py [--adapter path] [--num-samples N]
"""

import argparse
import json
import torch
import soundfile as sf
from pathlib import Path
from datetime import datetime

from train_router import PolyWhisperRouter, DEVICE, processor, MAX_LABEL_LEN
from train_router import align_labels
from eval_metrics import compute_wer, compute_cer, compute_fuzzy_wer

WHISPER_MODEL = "openai/whisper-base"
SAVE_DIR = Path("./polywhisper_output")
DATA_DIR = SAVE_DIR / "data"
TEST_JSON = DATA_DIR / "hinglish_codeswitch_test_ortho.json"
ADAPTER_DIR = SAVE_DIR / "adapters_v3"
LANG_TOKENS = {"en": 50259, "hi": 50276}

parser = argparse.ArgumentParser()
parser.add_argument("--test-json", type=str, default=str(DATA_DIR / "hinglish_codeswitch_test_ortho.json"))
parser.add_argument("--router", type=str, default=str(ADAPTER_DIR / "router_best.pt"))
parser.add_argument("--en-adapter", type=str, default="")
parser.add_argument("--hi-adapter", type=str, default="")
parser.add_argument("--num-samples", type=int, default=0, help="0 = full test set")
parser.add_argument("--max-new-tokens", type=int, default=256,
                         help="256 for FLEURS/long; 128 truncates")
parser.add_argument("--beams", type=int, default=1)
ARGS = parser.parse_args()

if not ARGS.en_adapter:
    ARGS.en_adapter = str(ADAPTER_DIR / "en_router_best.pt") if (ADAPTER_DIR / "en_router_best.pt").exists() \
        else str(ADAPTER_DIR / "en_best.pt")
if not ARGS.hi_adapter:
    ARGS.hi_adapter = str(ADAPTER_DIR / "hi_router_best.pt") if (ADAPTER_DIR / "hi_router_best.pt").exists() \
        else "polywhisper_output/adapters_v2/hi_best.pt"

log_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{log_ts}] {msg}", flush=True)


def main():
    log("Loading router model...")
    model = PolyWhisperRouter().to(DEVICE)
    model.add_language("en")
    model.add_language("hi")
    model._install_expert_hooks()
    if Path(ARGS.en_adapter).exists():
        model.load_adapter("en", ARGS.en_adapter)
        log(f"  en expert: {ARGS.en_adapter}")
    else:
        log(f"  WARNING: missing {ARGS.en_adapter} (falling back to untrained expert)")
    model.load_adapter("hi", ARGS.hi_adapter)
    log(f"  hi expert: {ARGS.hi_adapter}")
    if Path(ARGS.router).exists():
        model.load_router(ARGS.router)
        log(f"  router: {ARGS.router}")
    model.eval()

    records = json.load(open(ARGS.test_json))
    if ARGS.num_samples > 0:
        records = records[:ARGS.num_samples]
    log(f"Test samples: {len(records)}")

    refs, hyps = [], []
    tot_w, tot_c, err_w, err_c = 0, 0, 0, 0
    tot_fw, err_fw = 0, 0
    r_acc, r_tok = 0, 0
    n_route_agree_cs = n_cs = 0
    samples = []

    with torch.no_grad():
        for i, r in enumerate(records):
            audio, _ = sf.read(r["wav"])
            feats = processor.feature_extractor(
                [audio], sampling_rate=16000, return_tensors="pt", padding=True
            )["input_features"]
            B, C, T = feats.shape
            if T < 3000:
                pad = torch.zeros(B, C, 3000 - T, dtype=feats.dtype)
                feats = torch.cat([feats, pad], dim=-1)
            feats = feats.to(DEVICE)

            out = model.generate(feats, max_new_tokens=ARGS.max_new_tokens, num_beams=ARGS.beams,
                                 use_cache=False, language="hi", task="transcribe")
            hyp = processor.decode(out[0], skip_special_tokens=True).strip()
            ref = r["text"].strip()

            aligned = align_labels(ref, r["tokens"])
            if aligned is not None:
                sub, labels = aligned
                labels = torch.tensor(labels).to(DEVICE)

            w_ag, w_tk = 0, 0
            if aligned is not None:
                hyp_tokens = out[0]
                if hyp_tokens.numel() > 1:
                    with torch.no_grad():
                        model(feats, hyp_tokens[:-1].unsqueeze(0))
                w = model._router_weights
                if w is not None:
                    n = min(len(labels), w.shape[1] - 4)
                    if n > 0:
                        w_s = w[0, 4 : 4 + n]
                        pred = w_s.argmax(-1)
                        lab = 1 - labels[:n]  # align_labels 1=en -> router 1=hi
                        valid = lab != -100
                        if valid.any():
                            w_ag = (pred[valid] == lab[valid]).sum().item()
                            w_tk = valid.sum().item()
                r_acc += w_ag
                r_tok += w_tk
                if r["cs"] and w_tk:
                    n_cs += 1
                    n_route_agree_cs += 1

            wer = compute_wer(ref, hyp)
            cer = compute_cer(ref, hyp)
            f_wer = compute_fuzzy_wer(ref, hyp)
            nw = max(1, len(ref.split()))
            tot_w += nw
            tot_fw += nw
            err_w += wer * nw
            err_c += cer
            err_fw += f_wer * nw
            refs.append(ref)
            hyps.append(hyp)
            samples.append({"ref": ref, "hyp": hyp, "wer": wer, "cer": cer})

            if (i + 1) % 50 == 0:
                log(f"  [{i+1}/{len(records)}] WER so far: {err_w/tot_w*100:.1f}%")

    wer_pct = err_w / max(1, tot_w) * 100
    cer_pct = err_c / max(1, len(refs)) * 100
    f_wer_pct = err_fw / max(1, tot_fw) * 100
    r_acc_pct = r_acc / max(1, r_tok) * 100
    log("=" * 60)
    log(f"Hinglish (router): WER {wer_pct:.1f}%  FuzzyWER {f_wer_pct:.1f}%  CER {cer_pct:.1f}%")
    log(f"Router token accuracy: {r_acc_pct:.1f}% ({r_acc}/{r_tok})")
    log("=" * 60)

    out_json = SAVE_DIR / "eval_router_samples.json"
    json.dump({"wer": wer_pct, "fuzzy_wer": f_wer_pct, "cer": cer_pct, "router_acc": r_acc_pct, "samples": samples},
              open(out_json, "w"), indent=1, ensure_ascii=False)
    log(f"Saved per-sample results: {out_json}")


if __name__ == "__main__":
    main()
