"""
Shared evaluation metrics for PolyWhisper.
Normalized WER + CER. jiwer 4.0's wer() is case-sensitive and wer_standardize() is broken,
so we use our own normalization (lowercase, strip punctuation, collapse whitespace).
"""

import re as _re


def norm_text(text):
    text = text.lower()
    text = _re.sub(r'\uFFFD', '', text)
    text = _re.sub(r'[!-/:-@\[-`{-~]', '', text)
    return _re.sub(r'\s+', ' ', text).strip()


def _edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def wer_single(ref, hyp):
    ref_w = norm_text(ref).split()
    hyp_w = norm_text(hyp).split()
    if not ref_w:
        return 1.0 if hyp_w else 0.0
    n, m = len(ref_w), len(hyp_w)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_w[i - 1] == hyp_w[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def cer_single(ref, hyp):
    ref_c = _re.sub(r'\s+', '', norm_text(ref))
    hyp_c = _re.sub(r'\s+', '', norm_text(hyp))
    if not ref_c:
        return 1.0 if hyp_c else 0.0
    return _edit_distance(ref_c, hyp_c) / len(ref_c)


def compute_wer(references, hypotheses):
    if not references:
        return 1.0
    return sum(wer_single(r, h) for r, h in zip(references, hypotheses)) / len(references)


def compute_cer(references, hypotheses):
    if not references:
        return 1.0
    return sum(cer_single(r, h) for r, h in zip(references, hypotheses)) / len(references)


_DEVA_VOWELS = {'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu',
                'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऍ': 'e', 'ऑ': 'o', 'ऒ': 'o'}
_DEVA_MATRAS = {'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u', 'ृ': 'ri',
                'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', 'ँ': 'n', 'ः': 'h'}
_DEVA_CONS = {'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
              'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
              'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
              'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
              'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
              'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
              'क़': 'q', 'ख़': 'kh', 'ग़': 'g', 'ज़': 'z', 'ड़': 'r', 'ढ़': 'rh', 'फ़': 'f'}
_VIrama = '्'
_SKIP = set('्\u200c\u200d')


def translit(text, final_a=True):
    out = []
    chars = list(text)
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in _SKIP:
            i += 1
            continue
        if c in _DEVA_VOWELS:
            out.append(_DEVA_VOWELS[c])
            i += 1
            continue
        if c in _DEVA_MATRAS:
            out.append(_DEVA_MATRAS[c])
            i += 1
            continue
        if c in _DEVA_CONS:
            out.append(_DEVA_CONS[c])
            nxt = chars[i + 1] if i + 1 < len(chars) else ''
            if nxt not in (_VIrama,) and nxt not in _DEVA_MATRAS:
                if final_a or nxt:
                    out.append('a')
            i += 1
            continue
        if 'a' <= c <= 'z' or 'A' <= c <= 'Z':
            out.append(c.lower())
            i += 1
            continue
        i += 1
    return ''.join(out)


def fuzzy_wer_single(ref, hyp, thr=0.35):
    ref_w = translit(norm_text(ref)).split()
    hyp_w = translit(norm_text(hyp)).split()
    if not ref_w:
        return 1.0 if hyp_w else 0.0
    n, m = len(ref_w), len(hyp_w)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = ref_w[i - 1], hyp_w[j - 1]
            if a == b or _edit_distance(a, b) <= thr * max(len(a), len(b)):
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def compute_fuzzy_wer(references, hypotheses, thr=0.35):
    if not references:
        return 1.0
    return sum(fuzzy_wer_single(r, h, thr) for r, h in zip(references, hypotheses)) / len(references)
