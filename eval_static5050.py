"""
Ablation: static 50/50 en-hi mix with the v4 P2 experts, same harness as the
router eval (use_cache=False, ortho refs, identical metric aggregation).

The router eval already produced router hyps (results/eval_router_samples.json);
this run produces the "no routing" control: a fixed hook forces w = 0.5 for
every token after the model's own embed hook computes router weights.

Run: .venv/bin/python eval_static5050.py
Output: polywhisper_output/eval_static5050_v5_samples.json
"""

import json
import torch
import soundfile as sf
from pathlib import Path
from datetime import datetime

from train_router import PolyWhisperRouter, DEVICE, processor
from eval_metrics import compute_wer, compute_cer, compute_fuzzy_wer

WHISPER_MODEL = "openai/whisper-base"
SAVE_DIR = Path("./polywhisper_output")
DATA_DIR = SAVE_DIR / "data"
ADAPTER_DIR = SAVE_DIR / "adapters_v3"

log_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{log_ts}] {msg}", flush=True)


def main():
    log(f"Device: {DEVICE}")
    test_json = DATA_DIR / "hinglish_codeswitch_test_ortho.json"
    records = json.load(open(test_json))
    log(f"Test samples: {len(records)}")

    model = PolyWhisperRouter().to(DEVICE)
    model.add_language("en")
    model.add_language("hi")
    model._install_expert_hooks()
    model.load_adapter("en", str(ADAPTER_DIR / "en_router_best_v5.pt"))
    model.load_adapter("hi", str(ADAPTER_DIR / "hi_router_best_v5.pt"))

    def fixed_hook(mod, inp, out):
        B, S = out.shape[0], out.shape[1]
        model._router_weights = torch.full((B, S, 2), 0.5, device=out.device)

    model.whisper.model.decoder.embed_tokens.register_forward_hook(fixed_hook)
    log("Static 50/50 hook registered (after router embed hook -> overrides w).")
    model.eval()

    tot_w, tot_c, err_w, err_c = 0, 0, 0, 0
    tot_fw, err_fw = 0, 0
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
            feats = feats[:, :, :3000].to(DEVICE)

            out = model.generate(feats, max_new_tokens=128, num_beams=1,
                                 use_cache=False, language="hi", task="transcribe")
            hyp = processor.decode(out[0], skip_special_tokens=True).strip()
            ref = r["text"].strip()

            wer = compute_wer(ref, hyp)
            cer = compute_cer(ref, hyp)
            f_wer = compute_fuzzy_wer(ref, hyp)
            nw = max(1, len(ref.split()))
            tot_w += nw
            tot_fw += nw
            err_w += wer * nw
            err_c += cer
            err_fw += f_wer * nw
            samples.append({"ref": ref, "hyp": hyp, "wer": wer, "cer": cer})

            if (i + 1) % 50 == 0:
                log(f"  [{i+1}/{len(records)}] WER so far: {err_w/tot_w*100:.1f}%")

    wer_pct = err_w / max(1, tot_w) * 100
    cer_pct = err_c / max(1, len(records)) * 100
    f_wer_pct = err_fw / max(1, tot_fw) * 100
    log("=" * 60)
    log(f"Static 50/50 mix (v5 experts): WER {wer_pct:.1f}%  FuzzyWER {f_wer_pct:.1f}%  CER {cer_pct:.1f}%")
    log("=" * 60)

    json.dump({"wer": wer_pct, "fuzzy_wer": f_wer_pct, "cer": cer_pct, "samples": samples},
              open(SAVE_DIR / "eval_static5050_v5_samples.json", "w"), indent=1, ensure_ascii=False)
    log(f"Saved: {SAVE_DIR / 'eval_static5050_v5_samples.json'}")


if __name__ == "__main__":
    main()