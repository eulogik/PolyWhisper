"""
Vanilla whisper-base baseline on the full ortho test set — same metrics and
aggregation as eval_router.py (compute_wer/compute_cer/compute_fuzzy_wer).
No LoRA, no router hooks, greedy decode, use_cache=True (safe: no hooks).
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from eval_metrics import compute_wer, compute_cer, compute_fuzzy_wer

WHISPER_MODEL = "openai/whisper-base"
DATA_DIR = Path("./polywhisper_output/data")
SAVE_DIR = Path("./polywhisper_output")

log_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{log_ts}] {msg}", flush=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", type=str, default=str(DATA_DIR / "hinglish_codeswitch_test_ortho.json"))
    parser.add_argument("--use-cache", action="store_true", default=False)
    parser.add_argument("--language", type=str, default="hi")
    parser.add_argument("--out", type=str, default=str(SAVE_DIR / "eval_vanilla_samples.json"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    A = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"Device: {device}")
    proc = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(device)
    model.eval()

    records = json.load(open(A.test_json))
    log(f"Test samples: {len(records)}")

    tot_w, tot_c, err_w, err_c = 0, 0, 0, 0
    tot_fw, err_fw = 0, 0
    samples = []

    with torch.no_grad():
        for i, r in enumerate(records):
            audio, _ = sf.read(r["wav"])
            feats = proc.feature_extractor([audio], sampling_rate=16000, return_tensors="pt", padding=True)["input_features"]
            B, C, T = feats.shape
            if T < 3000:
                pad = torch.zeros(B, C, 3000 - T, dtype=feats.dtype)
                feats = torch.cat([feats, pad], dim=-1)
            feats = feats[:, :, :3000].to(device)
            out = model.generate(feats, max_new_tokens=A.max_new_tokens, num_beams=1, use_cache=A.use_cache,
                                 language=A.language, task="transcribe")
            hyp = proc.decode(out[0], skip_special_tokens=True).strip()
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
    log(f"Vanilla whisper-base: WER {wer_pct:.1f}%  FuzzyWER {f_wer_pct:.1f}%  CER {cer_pct:.1f}%")
    log("=" * 60)

    json.dump({"wer": wer_pct, "fuzzy_wer": f_wer_pct, "cer": cer_pct, "samples": samples},
              open(A.out, "w"), indent=1, ensure_ascii=False)
    log(f"Saved per-sample results: {SAVE_DIR / 'eval_vanilla_samples.json'}")

if __name__ == "__main__":
    main()