"""Pure single-language eval of a v3 LoRA expert on a cached FLEURS test json.

Loads the ta/te/bn/mr-style expert from adapters_v3 (PolyWhisperV3, encoder+decoder
LoRA or decoder-only — whichever the adapter was trained with) and transcribes
with the lang token forced. Router-free: measures the expert alone.

Usage:
  python eval_lang_pure.py --lang ta --adapter ta_best_ta.pt \
      --test-json polywhisper_output/data/fleurs_ta_in_test.json \
      --out polywhisper_output/eval_ta_pure_fleurs.json
"""

import argparse
import json
import sys
import time
import torch
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_v3 import PolyWhisperV3, DEVICE, processor, log, MODEL_SIZES  # noqa: E402

ADAPTERS = Path("polywhisper_output/adapters_v3")


def compute_wer(ref, hyp):
    rw, hw = ref.split(), hyp.split()
    if not rw:
        return 0.0 if not hw else 1.0
    dp = [[0] * (len(hw) + 1) for _ in range(len(rw) + 1)]
    for i in range(len(rw) + 1):
        dp[i][0] = i
    for j in range(len(hw) + 1):
        dp[0][j] = j
    for i in range(1, len(rw) + 1):
        for j in range(1, len(hw) + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (rw[i - 1] != hw[j - 1]))
    return dp[len(rw)][len(hw)] / len(rw)


def main():
    A = argparse.ArgumentParser()
    A.add_argument("--lang", required=True)
    A.add_argument("--adapter", required=True)
    A.add_argument("--adapter-dir", default="polywhisper_output/adapters_v3",
                   help="dir containing the adapter (default polywhisper_output/adapters_v3)")
    A.add_argument("--test-json", required=True)
    A.add_argument("--out", required=True)
    A.add_argument("--max-new-tokens", type=int, default=256,
                    help="max tokens to generate (256 for FLEURS; 128 truncates long utterances)")
    A.add_argument("--model-size", type=str, default="base", choices=["base", "small"],
                   help="backbone the adapter was trained on")
    A.add_argument("--encoder-lora", action="store_true",
                   help="adapter was trained with encoder LoRA (e.g. hi_best_v5)")
    a = A.parse_args()

    records = json.load(open(a.test_json))
    log(f"Records: {len(records)} | expert: {a.adapter} | backbone: {a.model_size}")

    model = PolyWhisperV3(whisper_name=MODEL_SIZES[a.model_size],
                          encoder_lora=a.encoder_lora).to(DEVICE)
    model.add_language(a.lang)
    model.load_adapter(a.lang, str(Path(a.adapter_dir) / a.adapter))
    model.eval()

    tot_w, err_w = 0, 0
    samples = []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(records):
            audio, _ = sf.read(r["wav"])
            feats = processor.feature_extractor([audio], sampling_rate=16000,
                                                return_tensors="pt", padding=True)["input_features"]
            B, C, T = feats.shape
            if T < 3000:
                pad = torch.zeros(B, C, 3000 - T, dtype=feats.dtype)
                feats = torch.cat([feats, pad], dim=-1)
            feats = feats[:, :, :3000].to(DEVICE)
            out = model.generate(feats, lang=a.lang, max_new_tokens=a.max_new_tokens, num_beams=1,
                                 use_cache=False, task="transcribe")
            hyp = processor.decode(out[0], skip_special_tokens=True).strip()
            ref = r["text"].strip()
            w = compute_wer(ref, hyp)
            n = len(ref.split())
            err_w += w * n
            tot_w += n
            samples.append({"ref": ref, "hyp": hyp, "wer": w})
            if (i + 1) % 50 == 0:
                log(f"  [{i+1}/{len(records)}] WER so far: {100*err_w/max(1,tot_w):.1f}%")
    wer = 100 * err_w / max(1, tot_w)
    log("=" * 60)
    log(f"{a.lang} (pure {a.lang}): WER {wer:.1f}%  (n={len(records)})  [{time.time()-t0:.0f}s]")
    log("=" * 60)
    json.dump({"lang": a.lang, "adapter": a.adapter, "wer": wer, "samples": samples},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    log(f"Saved: {a.out}")


if __name__ == "__main__":
    main()