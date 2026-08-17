"""Bootstrap significance tests for PolyWhisper vs vanilla WER/CER differences.

Reads per-sample eval JSONs and computes:
  - Mean WER/CER difference
  - 95% bootstrap CI (10K resamples)
  - p-value (proportion of bootstrap samples where difference <= 0)
"""

import json
import random
import sys
from pathlib import Path

SAVE = Path("polywhisper_output")
N_BOOT = 10000
random.seed(42)


def load_samples(path):
    d = json.load(open(path))
    return d.get("samples", d) if isinstance(d, dict) else d


def bootstrap_ci(a, b, n_boot=N_BOOT):
    """Return (mean_diff, ci_lo, ci_hi, p_value) for a - b."""
    a, b = list(a), list(b)
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    diffs = [sum(x) / len(x) for x in zip(a, b)]
    mean_diff = sum(diffs) / len(diffs)
    n = len(a)
    boot_diffs = []
    for _ in range(n_boot):
        idxs = [random.randint(0, n - 1) for _ in range(n)]
        boot_diffs.append(sum(a[i] - b[i] for i in idxs) / n)
    boot_diffs.sort()
    lo = boot_diffs[int(0.025 * n_boot)]
    hi = boot_diffs[int(0.975 * n_boot)]
    p = sum(1 for d in boot_diffs if d <= 0) / n_boot
    return mean_diff, lo, hi, p


def test_pair(vanilla_json, pure_json, lang):
    v_samps = load_samples(vanilla_json)
    p_samps = load_samples(pure_json)
    # match by ref text (order may differ if samples were filtered)
    v_by_ref = {s["ref"]: s for s in v_samps}
    p_by_ref = {s["ref"]: s for s in p_samps}
    common = sorted(set(v_by_ref.keys()) & set(p_by_ref.keys()))
    n = len(common)
    if n == 0:
        print(f"  {lang}: no paired samples — skipped")
        return
    v_wer = [v_by_ref[r]["wer"] for r in common]
    p_wer = [p_by_ref[r]["wer"] for r in common]
    has_cer = "cer" in p_by_ref[common[0]]
    if has_cer:
        v_cer = [v_by_ref[r]["cer"] for r in common]
        p_cer = [p_by_ref[r]["cer"] for r in common]

    wer_diff, wer_lo, wer_hi, wer_p = bootstrap_ci(v_wer, p_wer)
    sig_wer = "***" if wer_p < 0.001 else "**" if wer_p < 0.01 else "*" if wer_p < 0.05 else "ns"
    line = f"  {lang:4s} | WER {wer_diff:+.1f} [{wer_lo:+.1f}, {wer_hi:+.1f}] p={wer_p:.4f} {sig_wer} | n={n}"
    r = {"lang": lang, "n": n, "wer_diff": wer_diff, "wer_ci": [wer_lo, wer_hi], "wer_p": wer_p}
    if has_cer:
        cer_diff, cer_lo, cer_hi, cer_p = bootstrap_ci(v_cer, p_cer)
        sig_cer = "***" if cer_p < 0.001 else "**" if cer_p < 0.01 else "*" if cer_p < 0.05 else "ns"
        line += f" | CER {cer_diff:+.1f} [{cer_lo:+.1f}, {cer_hi:+.1f}] p={cer_p:.4f} {sig_cer}"
        r.update({"cer_diff": cer_diff, "cer_ci": [cer_lo, cer_hi], "cer_p": cer_p})
    print(line)
    return r


def main():
    print("Bootstrap significance tests (10K resamples, 95% CI)")
    print("vanilla - pure = positive = pure is better")
    print("=" * 80)

    tests = [
        ("hi", "eval_vanilla_fleurs_hi.json", "eval_hi_pure_fleurs.json"),
        ("ta", "eval_vanilla_fleurs_ta.json", "eval_ta_pure_fleurs.json"),
        ("te", "eval_vanilla_fleurs_te.json", "eval_te_pure_fleurs.json"),
        ("bn", "eval_vanilla_fleurs_bn.json", "eval_bn_pure_fleurs.json"),
        ("mr", "eval_vanilla_fleurs_mr.json", "eval_mr_pure_fleurs.json"),
    ]

    results = []
    for lang, v_file, p_file in tests:
        v_path = SAVE / v_file
        p_path = SAVE / p_file
        if not v_path.exists() or not p_path.exists():
            print(f"  {lang}: MISSING FILES — skipped")
            continue
        r = test_pair(v_path, p_path, lang)
        if r:
            results.append(r)

    if results:
        json.dump(results, open(SAVE / "significance_tests.json", "w"), indent=2)
        print(f"\nSaved: {SAVE / 'significance_tests.json'}")


if __name__ == "__main__":
    main()