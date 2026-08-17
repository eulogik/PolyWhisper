"""Orthographic normalization for FLEURS ta/te/bn/mr, then WER/CER re-scoring.

Standard Indic ASR practice (cf. IndicWER): fold spelling variants to a
canonical form on BOTH ref and hyp (symmetric), so that same-pronunciation
spellings do not count as errors. Steps per language:

  - NFC unicode normalization
  - script-specific char folding (e.g. Tam iL ன/ண/ந -> ந, sa ச/ஶ/ஷ/ஸ -> ச)
  - drop virama/halant (் ్ ্ ्) and nukta (় ़)
  - fold vowel-length signs (ஆ->அ, ీ->ి, ா-drop, ...)

Re-scores the saved eval jsons offline (no model rerun):
  eval_vanilla_fleurs_{lang}.json  vs  eval_{lang}_pure_fleurs.json
Output: polywhisper_output/fleurs_normalized_results.json + console table.
"""

import json
import re
import unicodedata
from pathlib import Path

SAVE = Path("polywhisper_output")

# ---------- fold tables (ord(cp) -> replacement str) ----------

def _fold(*pairs):
    return {ord(a): b for a, b in pairs}

TAMIL = _fold(
    ("ஶ", "ச"), ("ஷ", "ச"), ("ஸ", "ச"),
    ("ன", "ந"), ("ண", "ந"),
    ("ள", "ல"), ("ழ", "ல"),
    ("ற", "ர"),
    ("ா", ""), ("ீ", "ி"), ("ூ", "ு"), ("ே", "ெ"), ("ோ", "ொ"),
    ("ௌ", "ொ"), ("ை", "ெ"),
    ("்", ""), ("ஂ", ""), ("ஃ", ""),
)
TELUGU = _fold(
    ("శ", "స"), ("ష", "స"),
    ("ణ", "న"), ("ళ", "ల"), ("ఱ", "ర"),
    ("ా", ""), ("ీ", "ి"), ("ూ", "ు"), ("ే", "ె"), ("ో", "ొ"),
    ("ై", "ె"), ("ౌ", "ొ"),
    ("్", ""), ("ం", ""),
)
BENGALI = _fold(
    ("শ", "স"), ("ষ", "স"),
    ("ণ", "ন"),
    ("া", ""), ("ী", "ি"), ("ূ", "ু"), ("ৈ", "ে"), ("ৌ", "ো"),
    ("্", ""), ("়", ""), ("ঁ", ""), ("ং", ""),
)
MARATHI = _fold(
    ("श", "स"), ("ष", "स"),
    ("ऱ", "र"),
    ("ा", ""), ("ी", "ि"), ("ू", "ु"), ("ै", "े"), ("ौ", "ो"),
    ("्", ""), ("़", ""), ("ँ", ""), ("ऽ", ""),
)
HINDI = _fold(
    ("श", "स"), ("ष", "स"),
    ("ा", ""), ("ी", "ि"), ("ू", "ु"), ("ै", "े"), ("ौ", "ो"),
    ("्", ""), ("़", ""), ("ँ", ""), ("ऽ", ""),
)
# multi-char folds applied before the per-codepoint table
HINDI_PRE = (("क्ष", "कस"), ("ज्ञ", "गय"))

NORMS = {"ta": TAMIL, "te": TELUGU, "bn": BENGALI, "mr": MARATHI, "hi": HINDI}


def normalize(text, table):
    text = unicodedata.normalize("NFC", text.strip())
    for a, b in HINDI_PRE:
        text = text.replace(a, b)
    out = text.translate(table)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def wer_from(ref, hyp):
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


def cer_from(ref, hyp):
    r, h = list(ref), list(hyp)
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return dp[len(r)][len(h)] / max(1, len(r))


def score(samples, table, lang):
    tot_w, err_w, err_c, matched = 0, 0, 0, 0
    m_tot_w, m_err_w, m_err_c = 0, 0, 0
    script = {
        "\u0900-\u097F": "hi", "\u0B80-\u0BFF": "ta",
        "\u0C00-\u0C7F": "te", "\u0980-\u09FF": "bn",
        "\u0600-\u06FF": "ar", "A-Za-z": "en",
    }
    pats = [(re.compile("[" + r + "]"), n) for r, n in script.items()]

    def dom_script(s):
        best, n = None, 0
        for p, name in pats:
            c = len(p.findall(s))
            if c > n:
                best, n = name, c
        return best

    for s in samples:
        rf, hf = normalize(s["ref"], table), normalize(s["hyp"], table)
        n = len(rf.split())
        err_w += wer_from(rf, hf) * n
        err_c += cer_from(rf, hf)
        tot_w += n
        if dom_script(rf) == dom_script(hf):
            matched += 1
            m_err_w += wer_from(rf, hf) * n
            m_err_c += cer_from(rf, hf)
            m_tot_w += n
    return {
        "all": {"wer": 100 * err_w / max(1, tot_w),
                "cer": 100 * err_c / max(1, len(samples))},
        "script_matched": {"rate": 100 * matched / max(1, len(samples)),
                           "wer": 100 * m_err_w / max(1, m_tot_w),
                           "cer": 100 * m_err_c / max(1, matched)},
        "n": len(samples),
    }


def main():
    results = {}
    print(f"{'lang':<4} | {'vanilla':>18} | {'pure expert':>18}")
    print(f"{'':<4} | {'scr%':>5} {'WER':>6} {'CER':>6} | {'scr%':>5} {'WER':>6} {'CER':>6}")
    print("-" * 56)
    for lang, table in NORMS.items():
        v = json.load(open(SAVE / f"eval_vanilla_fleurs_{lang}.json"))
        p = json.load(open(SAVE / f"eval_{lang}_pure_fleurs.json"))
        v_s = score(v["samples"], table, lang)
        p_s = score(p["samples"], table, lang)
        results[lang] = {"vanilla": v_s, "pure": p_s}
        print(f"{lang:<4} | "
              f"{v_s['script_matched']['rate']:5.1f} {v_s['script_matched']['wer']:6.1f} {v_s['script_matched']['cer']:6.1f} | "
              f"{p_s['script_matched']['rate']:5.1f} {p_s['script_matched']['wer']:6.1f} {p_s['script_matched']['cer']:6.1f}")
    json.dump(results, open(SAVE / "fleurs_normalized_results.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\nScores shown on script-matched clips only (all n in json). "
          f"Saved: {SAVE / 'fleurs_normalized_results.json'}")


if __name__ == "__main__":
    main()