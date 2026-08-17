"""v5 router introspection: per-token routing weights, heatmaps, and diagnostics.

Runs the v5 PolyWhisperRouter over the Hinglish test set, captures per-token
w_en / w_hi router weights via the embed hook, and produces:
  - heatmap PNGs (routing weight per token) for code-switched samples
  - aggregate diagnostics (separation, mean w_en at en/hi, top misrouted)
  - results/v5_routing_analysis.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from train_router import PolyWhisperRouter, processor, DEVICE
from train_router import align_labels

SAVE_DIR = Path("./polywhisper_output")
ADAPTER_DIR = SAVE_DIR / "adapters_v3"
RESULTS_DIR = Path("./results")
TEST_JSON = SAVE_DIR / "data" / "hinglish_codeswitch_test_ortho.json"

parser = argparse.ArgumentParser()
parser.add_argument("--router", type=str, default=str(ADAPTER_DIR / "router_best_v5.pt"))
parser.add_argument("--en-adapter", type=str, default=str(ADAPTER_DIR / "en_router_best_v5.pt"))
parser.add_argument("--hi-adapter", type=str, default=str(ADAPTER_DIR / "hi_router_best_v5.pt"))
parser.add_argument("--test-json", type=str, default=str(TEST_JSON))
parser.add_argument("--num-samples", type=int, default=0, help="0 = full test set")
parser.add_argument("--num-heatmaps", type=int, default=8)
parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "v5_routing_analysis.json"))
ARGS = parser.parse_args()


def load_model():
    model = PolyWhisperRouter().to(DEVICE)
    model.add_language("en").add_language("hi")
    model.load_adapter("en", ARGS.en_adapter)
    model.load_adapter("hi", ARGS.hi_adapter)
    model.load_router(ARGS.router)
    model.eval()
    return model


def main():
    model = load_model()
    records = json.load(open(ARGS.test_json))
    if ARGS.num_samples:
        records = records[: ARGS.num_samples]
    print(f"Introspecting {len(records)} samples with v5 router")

    pred_all, lab_all = [], []
    w_at_en, w_at_hi = [], []
    misrouted, heatmaps = [], []
    w_en_rows = []

    for i, r in enumerate(records):
        audio_path = r.get("audio", "")
        if not audio_path or not Path(audio_path).exists():
            audio_path = SAVE_DIR / "data" / (r.get("audio_path") or "")
        if not Path(audio_path).exists():
            continue
        audio, _ = sf.read(str(audio_path))
        feats = processor.feature_extractor([audio], sampling_rate=16000,
                                            return_tensors="pt").input_features.to(DEVICE)

        out = model.generate(feats, max_new_tokens=128, num_beams=1,
                             use_cache=False, language="hi", task="transcribe")
        hyp_tokens = out[0]
        if hyp_tokens.numel() > 1:
            with torch.no_grad():
                model(feats, hyp_tokens[:-1].unsqueeze(0))
        w = model._router_weights
        if w is None:
            continue
        aligned = align_labels(r["text"], r["tokens"])
        if aligned is None:
            continue
        sub, labels = aligned
        labels = torch.tensor(labels).to(DEVICE)
        n = min(len(labels), w.shape[1] - 4)
        if n <= 0:
            continue
        w_s = w[0, 4 : 4 + n]
        pred = w_s.argmax(-1)
        lab = 1 - labels[:n]  # align_labels 1=en -> router 1=hi
        valid = lab != -100

        if valid.any():
            p_v, l_v = pred[valid], lab[valid]
            pred_all.extend(p_v.tolist())
            lab_all.extend(l_v.tolist())
            we_v = w_s[valid, 0]
            w_at_en.extend(we_v[l_v == 1].tolist())
            w_at_hi.extend(we_v[l_v == 0].tolist())

            for pos in valid.nonzero().flatten().tolist():
                if pred[pos] != lab[pos]:
                    tok_text = r["tokens"][pos]["t"] if pos < len(r["tokens"]) else "?"
                    misrouted.append({"pos": pos, "pred": "en" if pred[pos] == 1 else "hi",
                                      "true": "en" if lab[pos] == 1 else "hi",
                                      "w_en": round(w_s[pos, 0].item(), 3), "text": tok_text})

        if r.get("cs") and len(heatmaps) < ARGS.num_heatmaps:
            toks = [t["t"] for t in r["tokens"]]
            labs = [t["lang"] for t in r["tokens"]]
            w_en_row = [round(x, 3) for x in w_s[: len(toks), 0].tolist()]
            heatmaps.append({"text": r["text"][:150], "tokens": toks, "labels": labs,
                             "w_en": w_en_row, "cs": 1})
            w_en_rows.append(w_en_row)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(records)}] processed")

    pred_t = torch.tensor(pred_all)
    lab_t = torch.tensor(lab_all)
    correct = (pred_t == lab_t).sum().item()
    total = max(1, len(pred_t))
    acc = 100 * correct / total
    mean_en = sum(w_at_en) / max(1, len(w_at_en))
    mean_hi = sum(w_at_hi) / max(1, len(w_at_hi))
    top_mis = Counter(m["text"] for m in misrouted).most_common(20)

    print("=" * 60)
    print("ROUTING DIAGNOSTIC (v5, full test set)")
    print("=" * 60)
    print(f"Token-level acc: {acc:.1f}% ({correct}/{total})")
    print(f"Mean w_en at en tokens: {mean_en:.3f}  at hi tokens: {mean_hi:.3f}")
    print(f"Separation (en - hi): {mean_en - mean_hi:.3f}")
    print(f"Misrouted tokens: {len(misrouted)} / {total} ({100*len(misrouted)/total:.1f}%)")
    print("Top misrouted words:")
    for word, count in top_mis[:10]:
        print(f"  {word}: {count}")

    out = {
        "acc": acc, "total": total, "misrouted": len(misrouted),
        "mean_w_en_at_en": mean_en, "mean_w_en_at_hi": mean_hi,
        "separation": mean_en - mean_hi,
        "top_misrouted_words": [{"word": w, "count": c} for w, c in top_mis],
        "heatmap_samples": heatmaps,
        "model": "v5", "n_samples": len(records),
    }
    json.dump(out, open(ARGS.out, "w"), indent=1, ensure_ascii=False)
    print(f"Saved: {ARGS.out}")

    heat_out = RESULTS_DIR / "v5_routing_heatmaps.png"
    if w_en_rows:
        fig, axs = plt.subplots(len(w_en_rows), 1, figsize=(12, 1.4 * len(w_en_rows)),
                                squeeze=False)
        for ax, (row, hm) in zip(axs[:, 0], zip(w_en_rows, heatmaps)):
            ax.imshow([row], cmap="RdBu", aspect="auto", vmin=0, vmax=1)
            ax.set_yticks([])
            toks = hm["tokens"]
            ax.set_xticks(range(len(toks)))
            ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=7)
            for j, (lbl, v) in enumerate(zip(hm["labels"], row)):
                color = "blue" if lbl == "en" else "green"
                ax.annotate(f"{v:.2f}", (j, 0), ha="center", va="center",
                            fontsize=6, color=color)
            ax.set_title(f'CS "{hm["text"][:60]}..."', fontsize=8)
        fig.tight_layout()
        fig.savefig(heat_out, dpi=150)
        print(f"Heatmaps saved: {heat_out}")


if __name__ == "__main__":
    main()