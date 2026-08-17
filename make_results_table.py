#!/usr/bin/env python3
"""Aggregate PolyWhisper expert results + whisper baselines into one paper table.

Reads:
  polywhisper_output/fleurs_normalized_results.json   (our experts: vanilla + pure)
  baselines_{small,medium,large}_{hi,ta,te,bn,mr}.json  (from HF Hub eulogik/polywhisper)

Outputs a markdown table + JSON summary.
"""
import json
import os
from huggingface_hub import HfApi, hf_hub_download

REPO = "eulogik/polywhisper"
LANGS = ["hi", "ta", "te", "bn", "mr"]
SIZES = ["small", "medium", "large"]
LOCAL = os.path.join(os.path.dirname(__file__), "polywhisper_output")
CACHE = os.path.join(LOCAL, "baselines_cache")
os.makedirs(CACHE, exist_ok=True)


def load_experts():
    with open(os.path.join(LOCAL, "fleurs_normalized_results.json")) as f:
        return json.load(f)


def load_baselines():
    """Download + cache baseline JSONs from HF; return {(size,lang): dict}."""
    api = HfApi()
    out = {}
    for size in SIZES:
        for lang in LANGS:
            fname = f"baselines_{size}_{lang}.json"
            local = os.path.join(CACHE, fname)
            if not os.path.exists(local):
                try:
                    local = hf_hub_download(REPO, fname, repo_type="model",
                                            local_dir=CACHE)
                except Exception as e:
                    print(f"  [skip] {fname}: {e}")
                    continue
            with open(local) as f:
                out[(size, lang)] = json.load(f)
    return out


def fmt(v):
    return f"{v:.1f}" if v is not None else "—"


def main():
    experts = load_experts()
    baselines = load_baselines()
    print(f"Loaded {len(baselines)} baseline files from HF.")

    # header
    cols = (["Lang"]
            + [f"Whisper-Base\n(vanilla)" for _ in range(1)]
            + [f"Whisper-{s.capitalize()}\n(vanilla)" for s in SIZES]
            + ["PolyWhisper\nExpert (Base+LoRA)"])
    print("\n" + "| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")

    summary = {}
    for lang in LANGS:
        ex = experts.get(lang, {})
        van = ex.get("vanilla", {}).get("all", {})
        pur = ex.get("pure", {}).get("all", {})
        pur_scr = ex.get("pure", {}).get("script_matched", {}).get("rate")

        cells = [lang.upper()]
        # base vanilla
        cells.append(f"{fmt(van.get('wer'))}/{fmt(van.get('cer'))}")
        # baselines
        for size in SIZES:
            b = baselines.get((size, lang))
            if b:
                cells.append(f"{fmt(b['wer'])}/{fmt(b['cer'])}")
            else:
                cells.append("—/—")
        # our expert
        scr_note = f" (scr {pur_scr:.0f}%)" if pur_scr is not None else ""
        cells.append(f"**{fmt(pur.get('wer'))}/{fmt(pur.get('cer'))}**{scr_note}")

        print("| " + " | ".join(cells) + " |")
        summary[lang] = {
            "base_vanilla": {"wer": van.get("wer"), "cer": van.get("cer")},
            "expert": {"wer": pur.get("wer"), "cer": pur.get("cer"),
                       "script_rate": pur_scr},
            "baselines": {s: (baselines.get((s, lang)) or {}).get("wer") for s in SIZES},
        }

    out_path = os.path.join(LOCAL, "paper_results_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")

    # also a plain-text WER matrix for the paper
    print("\nWER-only matrix (lower is better):")
    print("Lang | Base(van) | Small | Medium | Large | Ours")
    for lang in LANGS:
        s = summary[lang]
        b = s["baselines"]
        print(f"{lang.upper():4} | {fmt(s['base_vanilla']['wer']):>8} | "
              f"{fmt(b['small']):>5} | {fmt(b['medium']):>6} | {fmt(b['large']):>5} | "
              f"{fmt(s['expert']['wer']):>5}")


if __name__ == "__main__":
    main()
