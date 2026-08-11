"""
Error analysis + ablation view over eval result JSons.

Merges per-sample results (router / vanilla / static-50/50 when present) with
the ortho test records (cs flag) and reports:
  - overall WER, cs-split WER (code-switched vs pure)
  - hallucination heuristics (repeated 3-grams, >10x ref length, empty hyps)
  - worst samples per model (by error words, ref >= 6 words)
  - delta table vs vanilla

Run: .venv/bin/python analyze_results.py
Output: results/analysis_summary.json (+ printed tables)
"""

import json
from collections import Counter
from pathlib import Path

SAVE_DIR = Path("./polywhisper_output")
RESULTS_DIR = Path("results")
ORTHO_JSON = SAVE_DIR / "data" / "hinglish_codeswitch_test_ortho.json"

RESULT_FILES = {
    "router": RESULTS_DIR / "eval_router_v5_samples.json",
    "vanilla": RESULTS_DIR / "eval_vanilla_samples.json",
    "static5050": SAVE_DIR / "eval_static5050_v5_samples.json",
}


def repeated_ngram(hyp, n=3):
    words = hyp.split()
    if len(words) < n * 2:
        return None
    for i in range(len(words) - n):
        ng = tuple(words[i:i + n])
        if ng in tuple(words[j:j + n] for j in range(i + 1, len(words) - n + 1)):
            return " ".join(ng)
    return None


def max_word_run(hyp):
    words = hyp.split()
    if not words:
        return 0
    best = cur = 1
    for a, b in zip(words, words[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best


def analyze(name, samples, records):
    by_cs = {0: [0.0, 0], 1: [0.0, 0]}
    errs = {}
    hall = {"repeated_3gram": 0, "word_run>=4": 0, "empty": 0, "sunny_len": 0}
    n = len(samples)
    for s, r in zip(samples, records):
        ref, hyp = s["ref"], s["hyp"]
        cs = r["cs"]
        nw = max(1, len(ref.split()))
        by_cs[cs][0] += s["wer"] * nw
        by_cs[cs][1] += nw
        errs[s["wer"] * nw] = (s, r)
        if repeated_ngram(hyp):
            hall["repeated_3gram"] += 1
        if max_word_run(hyp) >= 4:
            hall["word_run>=4"] += 1
        if not hyp.strip() or len(hyp.split()) == 0:
            hall["empty"] += 1
        if len(hyp.split()) > 10 * max(1, len(ref.split())):
            hall["sunny_len"] += 1
    out = {
        "model": name,
        "wer": 100 * sum(s["wer"] * max(1, len(s["ref"].split())) for s in samples)
               / max(1, sum(max(1, len(s["ref"].split())) for s in samples)),
        "samples": n,
        "cs_split": {f"cs={k}": (100 * v[0] / max(1, v[1]), v[1]) for k, v in by_cs.items()},
        "hallucination": hall,
    }
    worst = sorted(errs.items(), key=lambda kv: -kv[0])[:15]
    out["worst"] = [
        {"ref": r["text"], "hyp": s["hyp"], "err_words": round(e, 1)}
        for e, (s, r) in worst
    ]
    return out


def main():
    records = json.load(open(ORTHO_JSON))
    print(f"Ortho test records: {len(records)}")
    summary = {}
    for name, path in RESULT_FILES.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        data = json.load(open(path))
        print(f"[use] {name}: {data.get('wer')}% (n={len(data['samples'])})")
        summary[name] = analyze(name, data["samples"], records)

    print("\n================ SUMMARY ================")
    for name, a in summary.items():
        cs = ", ".join(f"cs={k} WER {v[0]:.1f}% (w={v[1]})" for k, v in a["cs_split"].items())
        h = a["hallucination"]
        print(f"{name:10s} WER {a['wer']:.1f}% | {cs} | hall {h['repeated_3gram']} rep3, "
              f"{h['word_run>=4']} run4+, {h['empty']} empty, {h['sunny_len']} len")

    if "vanilla" in summary and "router" in summary:
        dv = summary["vanilla"]["wer"] - summary["router"]["wer"]
        print(f"\nrouter vs vanilla: {dv:+.1f} pts WER")
    if "router" in summary and "static5050" in summary:
        ds = summary["static5050"]["wer"] - summary["router"]["wer"]
        print(f"router vs static_5050: {ds:+.1f} pts WER (routing value)")

    print("\n================ WORST SAMPLES (router) ================")
    if "router" in summary:
        for i, w in enumerate(summary["router"]["worst"]):
            print(f"{i:2d} [{w['err_words']:.0f} err] ref: {w['ref'][:80]}")
            print(f"     hyp: {w['hyp'][:80]}")

    Results_dir = RESULTS_DIR
    Results_dir.mkdir(exist_ok=True)
    json.dump(summary, open(Results_dir / "analysis_summary.json", "w"),
              indent=1, ensure_ascii=False)
    print(f"\nSaved: {Results_dir / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()