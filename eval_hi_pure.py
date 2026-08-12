"""Pure-Hindi eval of the hi LoRA expert on FLEURS hi (417 cached test records).

Forces w_hi = 1.0 (router disabled) so this measures the expert alone —
the utterance-level "Hindi-only" scenario from the original plan.
"""

import json
import sys
import time
import torch
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_router import PolyWhisperRouter, DEVICE, log  # noqa: E402

DATA_DIR = Path("polywhisper_output/data")
ADAPTERS = Path("polywhisper_output/adapters_v3")
OUT = Path("polywhisper_output/eval_hi_fleurs_samples.json")

from transformers import WhisperProcessor  # noqa: E402
processor = WhisperProcessor.from_pretrained("openai/whisper-base")


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
    records = json.load(open(DATA_DIR / "fleurs_hi_in_test.json"))
    log(f"FLEURS hi test records: {len(records)}")
    model = PolyWhisperRouter().to(DEVICE)
    model.add_language("en")
    model.add_language("hi")
    model._install_expert_hooks()

    def fixed_hi_hook(mod, inp, out):
        B, S = out.shape[0], out.shape[1]
        model._router_weights = torch.full((B, S, 2), 0.0, device=out.device)
        model._router_weights[..., 1] = 1.0
        return out

    model.whisper.model.decoder.embed_tokens.register_forward_hook(fixed_hi_hook)
    model.load_adapter("hi", str(ADAPTERS / "hi_best_v5.pt"))
    log("Loaded hi_best_v5.pt, forced w_hi=1.0 (no routing)")

    tot_w, err_w = 0, 0
    samples = []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(records):
            a, _ = sf.read(r["wav"])
            feats = processor.feature_extractor([a], sampling_rate=16000, return_tensors="pt",
                                                padding=True)["input_features"]
            B, C, T = feats.shape
            if T < 3000:
                pad = torch.zeros(B, C, 3000 - T, dtype=feats.dtype)
                feats = torch.cat([feats, pad], dim=-1)
            feats = feats[:, :, :3000].to(DEVICE)
            out = model.generate(feats, max_new_tokens=128, num_beams=1,
                                 use_cache=False, language="hi", task="transcribe")
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
    log(f"FLEURS hi (pure Hindi): WER {wer:.1f}%  (n={len(records)})  [{time.time()-t0:.0f}s]")
    log("=" * 60)
    json.dump({"wer": wer, "samples": samples},
              open(OUT, "w"), indent=1, ensure_ascii=False)
    log(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
