"""Vanilla Whisper-Small zero-shot baseline on FLEURS test sets.

Same methodology as the paper's adapter evals: greedy beam-1 decode,
lang token forced, max 256 tokens, normalized scoring via normalize_ortho.
No adapters, no hooks.

Usage:
  .venv/bin/python3 eval_vanilla_small.py --langs hi,ta,te,bn,mr
  .venv/bin/python3 eval_vanilla_small.py --langs bn --out results/vanilla_small_bn_fleurs.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent))
from normalize_ortho import NORMS, normalize, wer_from

KIOXIA_DATA = Path("/Volumes/KIOXIA 1TB/polywhisper_output/data")
LOCAL_DATA = Path("polywhisper_output/data")
LANG_JSON = {
    "hi": "fleurs_hi_in_test.json",
    "ta": "fleurs_ta_in_test.json",
    "te": "fleurs_te_in_test.json",
    "bn": "fleurs_bn_in_test.json",
    "mr": "fleurs_mr_in_test.json",
}


def resolve_wav(rel):
    for base in (Path("."), Path("/Volumes/KIOXIA 1TB")):
        p = base / rel
        if p.exists():
            return str(p)
    raise FileNotFoundError(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hi,ta,te,bn,mr")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    model.eval()

    for lang in [l.strip() for l in args.langs.split(",")]:
        for base in (LOCAL_DATA, KIOXIA_DATA):
            jp = base / LANG_JSON[lang]
            if jp.exists():
                break
        records = json.load(open(jp))
        print(f"[{lang}] {len(records)} samples from {jp}", flush=True)
        samples = []
        tot_w, err_w = 0, 0
        t0 = time.time()
        with torch.no_grad():
            for i, r in enumerate(records):
                audio, _ = sf.read(resolve_wav(r["wav"]))
                feats = proc.feature_extractor(
                    [audio], sampling_rate=16000, return_tensors="pt", padding=True
                )["input_features"]
                b, c, t = feats.shape
                if t < 3000:
                    feats = torch.cat([feats, torch.zeros(b, c, 3000 - t)], dim=-1)
                feats = feats[:, :, :3000].to(device)
                out = model.generate(
                    feats, max_new_tokens=args.max_new_tokens,
                    num_beams=1, language=lang, task="transcribe",
                )
                hyp = proc.decode(out[0], skip_special_tokens=True).strip()
                ref = r["text"].strip()
                rf, hf = normalize(ref, NORMS[lang]), normalize(hyp, NORMS[lang])
                w = wer_from(rf, hf)
                n = len(rf.split())
                tot_w += n
                err_w += w * n
                samples.append({"ref": ref, "hyp": hyp, "wer": w})
                if (i + 1) % 50 == 0:
                    dt = time.time() - t0
                    print(f"  [{lang} {i+1}/{len(records)}] WER so far: {err_w/max(1,tot_w)*100:.1f}% ({dt:.0f}s)", flush=True)
        wer = err_w / max(1, tot_w) * 100
        print(f"[{lang}] vanilla whisper-small WER: {wer:.1f}%", flush=True)
        out = args.out or f"results/vanilla_small_{lang}_fleurs.json"
        json.dump({"lang": lang, "wer": wer, "n": len(samples), "samples": samples},
                  open(out, "w"), indent=1, ensure_ascii=False)
        print(f"[{lang}] saved {out}", flush=True)


if __name__ == "__main__":
    main()
