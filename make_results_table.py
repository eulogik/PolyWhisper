#!/usr/bin/env python3
"""Aggregate PolyWhisper expert results + whisper baselines into one paper table.

Reads:
  polywhisper_output/fleurs_normalized_results.json   (our experts: vanilla + pure)
  baselines_{small,medium,large}_{hi,ta,te,bn,mr}.json  (from HF Hub eulogik/polywhisper)
  polywhisper_output_gpu{0,1}/eval_{lang}_pure_fleurs.json  (product run: Small+LoRA, from Kaggle)

Outputs a markdown table + JSON summary.
"""
import argparse
import json
import os
import re
from huggingface_hub import HfApi, hf_hub_download
from normalize_ortho import normalize, wer_from, cer_from, NORMS

REPO = "eulogik/polywhisper"
LANGS = ["hi", "ta", "te", "bn", "mr"]
SIZES = ["small", "medium", "large"]
LOCAL = os.path.join(os.path.dirname(__file__), "polywhisper_output")
CACHE = os.path.join(LOCAL, "baselines_cache")
PROD_DIRS = ["polywhisper_output_gpu0", "polywhisper_output_gpu1"]
os.makedirs(CACHE, exist_ok=True)


def load_experts():
    p = os.path.join(LOCAL, "fleurs_normalized_results.json")
    if not os.path.exists(p):
        print(f"  [warn] missing {p} — expert columns will be —/—")
        return {}
    with open(p) as f:
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


# --- offline script-match detection (reused from normalize_ortho.py) ---
_SCRIPT_RANGES = {
    "\u0900-\u097F": "hi", "\u0B80-\u0BFF": "ta",
    "\u0C00-\u0C7F": "te", "\u0980-\u09FF": "bn",
    "\u0600-\u06FF": "ar", "A-Za-z": "en",
}
_PATS = [(re.compile("[" + r + "]"), n) for r, n in _SCRIPT_RANGES.items()]


def dom_script(s):
    best, n = None, 0
    for p, name in _PATS:
        c = len(p.findall(s))
        if c > n:
            best, n = name, c
    return best


def baseline_script_match(b):
    """Compute % of baseline samples whose hyp is in the ref's dominant script."""
    if not b or "samples" not in b:
        return None
    matched = 0
    for s in b["samples"]:
        if dom_script(s.get("ref", "")) == dom_script(s.get("hyp", "")):
            matched += 1
    return 100.0 * matched / max(1, len(b["samples"]))


def score_baseline_norm(b, lang):
    """Normalized WER/CER for a HF baseline (apples-to-apples with expert column)."""
    if not b or "samples" not in b or lang not in NORMS:
        return None
    table = NORMS[lang]
    tot_w, err_w, err_c, matched = 0, 0, 0, 0
    for s in b["samples"]:
        rf, hf = normalize(s.get("ref", ""), table), normalize(s.get("hyp", ""), table)
        n = len(rf.split())
        err_w += wer_from(rf, hf) * n
        err_c += cer_from(rf, hf)
        tot_w += n
        if dom_script(rf) == dom_script(hf):
            matched += 1
    return {
        "wer_norm": 100 * err_w / max(1, tot_w),
        "cer_norm": 100 * err_c / max(1, len(b["samples"])),
        "wer_raw": b.get("wer"),
        "cer_raw": b.get("cer"),
        "script_rate": 100 * matched / max(1, len(b["samples"])),
        "n": len(b["samples"]),
    }


def load_prod_evals():
    """Load product-run eval jsons (Small+LoRA on FLEURS) from the GPU save dirs."""
    out = {}
    for d in PROD_DIRS:
        for lang in LANGS:
            p = os.path.join(d, f"eval_{lang}_pure_fleurs.json")
            if not os.path.exists(p):
                continue
            try:
                with open(p) as f:
                    out[lang] = json.load(f)
            except Exception as e:
                print(f"  [skip] {p}: {e}")
    return out


def score_prod(ev):
    """Normalized WER/CER (all clips) + script-matched rate, like the expert column."""
    samples = ev.get("samples", [])
    if not samples or ev.get("lang") not in NORMS:
        return None
    table = NORMS[ev["lang"]]
    tot_w, err_w, err_c, matched = 0, 0, 0, 0
    for s in samples:
        rf, hf = normalize(s.get("ref", ""), table), normalize(s.get("hyp", ""), table)
        n = len(rf.split())
        err_w += wer_from(rf, hf) * n
        err_c += cer_from(rf, hf)
        tot_w += n
        if dom_script(rf) == dom_script(hf):
            matched += 1
    return {
        "wer": 100 * err_w / max(1, tot_w),
        "cer": 100 * err_c / max(1, len(samples)),
        "script_rate": 100 * matched / max(1, len(samples)),
        "n": len(samples),
    }


def main():
    experts = load_experts()
    baselines = load_baselines()
    prods = load_prod_evals()
    print(f"Loaded {len(baselines)} baseline files from HF, {len(prods)} product evals.")
    # warn on FLEURS version drift (local 417 vs HF live 418 for hi)
    for lang in LANGS:
        ex_n = experts.get(lang, {}).get("pure", {}).get("n")
        for size in SIZES:
            b = baselines.get((size, lang))
            if b and ex_n and b.get("n") != ex_n:
                print(f"  [note] {size}/{lang} baseline n={b.get('n')} vs expert n={ex_n} — FLEURS version drift (1-sample Δ, <0.3% WER impact)")

    # header
    cols = (["Lang"]
            + [f"Whisper-Base\n(vanilla)" for _ in range(1)]
            + [f"Whisper-{s.capitalize()}\n(vanilla)" for s in SIZES]
            + ["PolyWhisper\nExpert (Base+LoRA)"]
            + ["Product\n(Small+LoRA)"])
    print("\n" + "| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")

    summary = {}
    for lang in LANGS:
        ex = experts.get(lang, {})
        van = ex.get("vanilla", {}).get("all", {})
        pur = ex.get("pure", {}).get("all", {})
        pur_scr = ex.get("pure", {}).get("script_matched", {}).get("rate")
        prd = score_prod(prods.get(lang, {}))

        cells = [lang.upper()]
        # base vanilla (already normalized in fleurs_normalized_results.json)
        cells.append(f"{fmt(van.get('wer'))}/{fmt(van.get('cer'))}")
        # baselines — now normalized apples-to-apples (same ortho tables as expert)
        for size in SIZES:
            b = baselines.get((size, lang))
            sb = score_baseline_norm(b, lang) if b else None
            if sb:
                cells.append(f"{fmt(sb['wer_norm'])}/{fmt(sb['cer_norm'])} (scr {sb['script_rate']:.0f}%)")
            else:
                cells.append("—/—")
        # our expert
        scr_note = f" (scr {pur_scr:.0f}%)" if pur_scr is not None else ""
        cells.append(f"**{fmt(pur.get('wer'))}/{fmt(pur.get('cer'))}**{scr_note}")
        # product expert (Small+LoRA)
        if prd:
            cells.append(f"**{fmt(prd['wer'])}/{fmt(prd['cer'])}** (scr {prd['script_rate']:.0f}%)")
        else:
            cells.append("—/—")

        print("| " + " | ".join(cells) + " |")
        # keep raw for audit, norm is the paper number
        norm_map = {}
        raw_map = {}
        for s in SIZES:
            sb = score_baseline_norm(baselines.get((s, lang)), lang)
            norm_map[s] = sb["wer_norm"] if sb else None
            raw_map[s] = sb["wer_raw"] if sb else None
        summary[lang] = {
            "base_vanilla": {"wer": van.get("wer"), "cer": van.get("cer")},
            "expert": {"wer": pur.get("wer"), "cer": pur.get("cer"),
                       "script_rate": pur_scr},
            "product": prd,
            "baselines": raw_map,
            "baselines_norm": norm_map,
            "baselines_raw": raw_map,
        }

    out_path = os.path.join(LOCAL, "paper_results_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")

    # also a plain-text WER matrix for the paper (all normalized, apples-to-apples)
    print("\nWER-only matrix — all normalized (lower is better):")
    print("Lang | Base(van) | Small | Medium | Large | Ours | Product")
    for lang in LANGS:
        s = summary[lang]
        bn = s.get("baselines_norm", s["baselines"])
        p = s["product"]
        print(f"{lang.upper():4} | {fmt(s['base_vanilla']['wer']):>8} | "
              f"{fmt(bn['small']):>5} | {fmt(bn['medium']):>6} | {fmt(bn['large']):>5} | "
              f"{fmt(s['expert']['wer']):>5} | {fmt(p['wer'] if p else None):>8}")
    print("\nRaw baseline WER (for audit, not paper):")
    print("Lang | Small(raw) | Medium(raw)")
    for lang in LANGS:
        s = summary[lang]
        br = s.get("baselines_raw", {})
        print(f"{lang.upper():4} | {fmt(br.get('small')):>10} | {fmt(br.get('medium')):>11}")


if __name__ == "__main__":
    main()
