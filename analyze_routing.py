"""
Routing diagnostic: load the v4 P2 router, run train set forward,
report w_en distributions, confusion matrix, misroutes, heatmaps.
"""
import json, torch, soundfile as sf
from pathlib import Path
from datetime import datetime

from train_router import PolyWhisperRouter, DEVICE, processor
from train_router import START_TOKEN, LANG_TOKENS, TASK_TOKEN, NO_TIME_TOKEN, align_labels

FORCED_PREFIX = [START_TOKEN, LANG_TOKENS["hinglish"], TASK_TOKEN, NO_TIME_TOKEN]

SAVE_DIR = Path("./polywhisper_output")
DATA_DIR = SAVE_DIR / "data"
ADAPTER_DIR = SAVE_DIR / "adapters_v3"
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

log_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg):
    print(f"[{log_ts}] {msg}", flush=True)

def main():
    log(f"Device: {DEVICE}")
    model = PolyWhisperRouter().to(DEVICE)
    model.add_language("en"); model.add_language("hi")
    model._install_expert_hooks()
    model.load_adapter("en", str(ADAPTER_DIR / "en_router_best_v4.pt"))
    model.load_adapter("hi", str(ADAPTER_DIR / "hi_router_best_v4.pt"))
    model.load_router(str(ADAPTER_DIR / "router_best_v4.pt"))
    model.eval()
    log("Loaded v4 P2 router + experts.")

    records = json.load(open(DATA_DIR / "hinglish_codeswitch_train.json"))
    N = 2000
    records = records[:N]
    log(f"Routing stats on first {N} train samples...")

    w_en_at_en, w_en_at_hi = [], []
    pred_all, lab_all = [], []
    misrouted = []
    heatmap_samples = []

    with torch.no_grad():
        for i, r in enumerate(records):
            aligned = align_labels(r["text"], r["tokens"])
            if aligned is None:
                continue
            sub, labels = aligned

            audio, _ = sf.read(r["wav"])
            feats = processor.feature_extractor(
                [audio], sampling_rate=16000, return_tensors="pt", padding=True
            )["input_features"]
            B, C, T = feats.shape
            if T < 3000:
                pad = torch.zeros(B, C, 3000 - T, dtype=feats.dtype)
                feats = torch.cat([feats, pad], dim=-1)
            feats = feats[:, :, :3000].to(DEVICE)

            full = FORCED_PREFIX + sub + [50257]
            lbls = torch.tensor([full], dtype=torch.long).to(DEVICE)
            logits, w = model(feats, lbls[:, :-1])
            if w is None:
                continue

            # align_labels returns integer labels: 1=en, 0=hi
            lab = torch.tensor(labels, dtype=torch.long).to(DEVICE)
            n = min(len(labels), w.shape[1] - len(FORCED_PREFIX))
            if n <= 0:
                continue

            w_s = w[0, len(FORCED_PREFIX) : len(FORCED_PREFIX) + n]
            pred = w_s.argmax(-1)
            w_en = w_s[:, 0].cpu().tolist()

            for p, l, we, pos in zip(
                pred.cpu().tolist(), lab[:n].cpu().tolist(), w_en, range(n)
            ):
                if l == 1:
                    w_en_at_en.append(we)
                else:
                    w_en_at_hi.append(we)
                pred_all.append(p)
                lab_all.append(l)
                if p != l:
                    tok_text = r["tokens"][pos]["t"] if pos < len(r["tokens"]) else "?"
                    misrouted.append({
                        "pos": pos, "pred": "en" if p == 1 else "hi",
                        "true": "en" if l == 1 else "hi",
                        "w_en": round(we, 3), "text": tok_text,
                    })

            if r["cs"] and len(heatmap_samples) < 5:
                heatmap_samples.append({
                    "text": r["text"][:120],
                    "tokens": [t["t"] for t in r["tokens"]],
                    "labels": [t["lang"] for t in r["tokens"]],
                    "w_en": [round(x, 3) for x in w_en[:len(r["tokens"])]],
                    "cs": r["cs"],
                })

            if (i + 1) % 500 == 0:
                log(f"  [{i+1}/{len(records)}] processed")

    pred_t = torch.tensor(pred_all)
    lab_t = torch.tensor(lab_all)
    correct = (pred_t == lab_t).sum().item()
    total = len(pred_t)
    acc = 100 * correct / max(1, total)

    tp = ((pred_t == 1) & (lab_t == 1)).sum().item()
    fp = ((pred_t == 1) & (lab_t == 0)).sum().item()
    fn = ((pred_t == 0) & (lab_t == 1)).sum().item()
    tn = ((pred_t == 0) & (lab_t == 0)).sum().item()

    mean_en = sum(w_en_at_en) / max(1, len(w_en_at_en))
    mean_hi = sum(w_en_at_hi) / max(1, len(w_en_at_hi))

    # Top misrouted words
    from collections import Counter
    misroute_words = Counter(m["text"] for m in misrouted)
    top_misroute = misroute_words.most_common(20)

    print("=" * 60)
    print("ROUTING DIAGNOSTIC (v4 P2, 2000 train samples)")
    print("=" * 60)
    print(f"Token-level acc: {acc:.1f}% ({correct}/{total})")
    print(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  en precision: {tp/(tp+fp)*100:.1f}%  en recall: {tp/(tp+fn)*100:.1f}%")
    print(f"  hi precision: {tn/(tn+fn)*100:.1f}%  hi recall: {tn/(tn+fp)*100:.1f}%")
    print(f"Mean w_en (index0) at en tokens: {mean_en:.3f}")
    print(f"Mean w_en (index0) at hi tokens: {mean_hi:.3f}")
    print(f"Separation (en - hi): {mean_en - mean_hi:.3f}")
    print(f"Misrouted tokens: {len(misrouted)} / {total} ({100*len(misrouted)/max(1,total):.1f}%)")
    print(f"  hi->en (false en): {fp} | en->hi (false hi): {fn}")
    print(f"\nTop misrouted words:")
    for word, count in top_misroute:
        print(f"  {word}: {count}")

    print(f"\nHeatmap samples (5 code-switched):")
    for s in heatmap_samples:
        pairs = list(zip(s["tokens"][:20], s["labels"][:20], s["p_hi"][:20]))
        vis = " ".join(f"{t}({'E' if l=='en' else 'H'}:{w:.2f})" for t, l, w in pairs)
        print(f"  {vis}")

    out = {
        "acc": acc, "total": total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mean_w_en_at_en": mean_en, "mean_w_en_at_hi": mean_hi,
        "separation": mean_en - mean_hi,
        "n_misrouted": len(misrouted),
        "top_misrouted_words": top_misroute,
        "heatmap_samples": heatmap_samples,
    }
    json.dump(out, open(RESULTS / "routing_analysis.json", "w"), indent=1, ensure_ascii=False)
    print(f"\nSaved: {RESULTS / 'routing_analysis.json'}")

if __name__ == "__main__":
    main()
