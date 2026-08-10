"""
PolyWhisper Local Evaluation Script
Matches train_local.py architecture exactly (including causal mask).
Uses cached test data from polywhisper_output/data/.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import math
import time
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

# ============ CONFIG ============
WHISPER_MODEL = "openai/whisper-base"
WHISPER_EXPECTED_LEN = 3000
LANGUAGES = ["en", "hi", "hinglish"]
MAX_LABEL_LEN = 256
AD_HID = 256
AD_LAYERS = 3
AD_HEADS = 4
AD_FFN = 1024
AD_RANK = 16
MAX_NEW_TOKENS = 128
MAX_SAMPLES = 50
BATCH_SIZE = 8
UNFREEZE_LAYERS = [4, 5]

# ============ PATHS ============
SAVE_DIR = Path("./polywhisper_output")
ADAPTER_DIR = SAVE_DIR / "adapters"
DATA_DIR = SAVE_DIR / "data"
EVAL_FILE = SAVE_DIR / "eval_results.json"

# ============ DEVICE ============
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

# ============ MODEL (exact match with train_local.py) ============
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
ENCODE_DIM = 512
VOCAB_SIZE = len(processor.tokenizer)
tok = processor.tokenizer

class LowRank(nn.Module):
    def __init__(self, i, o, r):
        super().__init__()
        self.a = nn.Linear(i, r, bias=False)
        self.b = nn.Linear(r, o, bias=False)
        nn.init.zeros_(self.b.weight)
    def forward(self, x): return self.b(self.a(x))

class SelfAttn(nn.Module):
    def __init__(self, d, h, r):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.qa = LowRank(d, d, r)
        self.va = LowRank(d, d, r)
        self.h, self.dh = h, d // h
        self.sc = math.sqrt(self.dh)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x) + self.qa(x)
        k, v = self.k(x), self.v(x) + self.va(x)
        def rs(t): return t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = rs(q), rs(k), rs(v)
        scores = (q @ k.transpose(-2, -1)) / self.sc
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=scores.dtype), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        a = self.drop(torch.softmax(scores, dim=-1))
        return self.o((a @ v).transpose(1, 2).contiguous().view(B, T, -1))

class CrossAttn(nn.Module):
    def __init__(self, d, ed, h, r):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(ed, d)
        self.v = nn.Linear(ed, d)
        self.o = nn.Linear(d, d)
        self.qa = LowRank(d, d, r)
        self.va = LowRank(ed, d, r)
        self.h, self.dh = h, d // h
        self.sc = math.sqrt(self.dh)
        self.drop = nn.Dropout(0.1)
    def forward(self, x, enc):
        B, Td, _ = x.shape
        Te = enc.size(1)
        q = self.q(x) + self.qa(x)
        k, v = self.k(enc), self.v(enc) + self.va(enc)
        def rs(t, T): return t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = rs(q, Td), rs(k, Te), rs(v, Te)
        a = self.drop(torch.softmax((q @ k.transpose(-2, -1)) / self.sc, dim=-1))
        return self.o((a @ v).transpose(1, 2).contiguous().view(B, Td, -1))

class AdaptLayer(nn.Module):
    def __init__(self, d, ed, h, ffn, r):
        super().__init__()
        self.sa = SelfAttn(d, h, r)
        self.ca = CrossAttn(d, ed, h, r)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(0.1), nn.Linear(ffn, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.1)
    def forward(self, x, enc):
        x = x + self.drop(self.sa(self.n1(x)))
        x = x + self.drop(self.ca(self.n2(x), enc))
        x = x + self.drop(self.ff(self.n3(x)))
        return x

class LangAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(VOCAB_SIZE, AD_HID)
        self.pe = nn.Embedding(MAX_LABEL_LEN, AD_HID)
        self.layers = nn.ModuleList([AdaptLayer(AD_HID, ENCODE_DIM, AD_HEADS, AD_FFN, AD_RANK) for _ in range(AD_LAYERS)])
        self.out = nn.Linear(AD_HID, VOCAB_SIZE, bias=False)
        self.out.weight = self.te.weight
        self.norm = nn.LayerNorm(AD_HID)
        self.drop = nn.Dropout(0.1)
    def forward(self, enc, ids):
        ids = ids.clamp(min=0, max=VOCAB_SIZE - 1)
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.drop(self.te(ids) + self.pe(pos))
        for l in self.layers:
            x = l(x, enc)
        return self.out(self.norm(x))

class PolyWhisper(nn.Module):
    def __init__(self):
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
        for p in self.whisper.model.encoder.parameters():
            p.requires_grad = False
        for p in self.whisper.model.decoder.parameters():
            p.requires_grad = False
        for i in UNFREEZE_LAYERS:
            for p in self.whisper.model.encoder.layers[i].parameters():
                p.requires_grad = True
        self.adapters = nn.ModuleDict({l: LangAdapter() for l in LANGUAGES})
    def encode(self, feat):
        return self.whisper.model.encoder(feat).last_hidden_state
    def forward(self, feat, dec_ids, lang):
        enc = self.encode(feat)
        return self.adapters[lang](enc, dec_ids)
    def load_ckpt(self, path):
        st = torch.load(path, map_location=DEVICE, weights_only=True)
        for n, s in st.items():
            if n in self.adapters:
                self.adapters[n].load_state_dict(s)
        if '_encoder' in st:
            for k, v in st['_encoder'].items():
                idx = int(k)
                self.whisper.model.encoder.layers[idx].load_state_dict(v)

print("Loading model...")
model = PolyWhisper().to(DEVICE)

# Load all adapters
loaded = []
for lang in LANGUAGES:
    p = ADAPTER_DIR / f"{lang}_best.pt"
    if p.exists():
        model.load_ckpt(str(p))
        loaded.append(lang)
        print(f"  Loaded {lang}: {p.name}")
    else:
        print(f"  WARNING: No adapter for {lang}")

if not loaded:
    print("ERROR: No adapters found!")
    exit(1)

model.eval()

# ============ GENERATION (autoregressive with causal mask) ============
START_TOKEN = 50258  # <|startoftranscript|>, NOT tok.bos_token_id (50257=<|endoftext|>)

@torch.no_grad()
def generate(model, audio_feat, lang, max_len=MAX_NEW_TOKENS, temperature=1.0, top_k=50, rep_penalty=1.3):
    """Greedy decoding with repetition penalty and top-k sampling."""
    enc = model.encode(audio_feat)
    device = audio_feat.device
    dec_ids = torch.tensor([[START_TOKEN]], device=device)
    for _ in range(max_len):
        logits = model.adapters[lang](enc, dec_ids)
        next_logits = logits[:, -1, :] / temperature

        # Repetition penalty
        if rep_penalty != 1.0:
            seen = set(dec_ids[0].cpu().tolist())
            for t in seen:
                if next_logits[0, t] > 0:
                    next_logits[0, t] /= rep_penalty
                else:
                    next_logits[0, t] *= rep_penalty

        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = next_logits.topk(top_k, dim=-1)
            next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')

        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        dec_ids = torch.cat([dec_ids, next_token], dim=1)
        if next_token.item() == tok.eos_token_id:
            break
    return tok.decode(dec_ids[0].cpu().tolist(), skip_special_tokens=True)

@torch.no_grad()
def generate_beam(model, audio_feat, lang, beam_width=5, max_len=MAX_NEW_TOKENS, rep_penalty=1.2):
    """Beam search decoding with repetition penalty."""
    enc = model.encode(audio_feat)
    device = audio_feat.device
    beams = [(torch.tensor([[START_TOKEN]], device=device), 0.0)]

    for step in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq.size(1) > 1 and seq[0, -1].item() == tok.eos_token_id:
                candidates.append((seq, score))
                continue

            logits = model.adapters[lang](enc, seq)
            next_logits = logits[:, -1, :].clone()

            # Repetition penalty
            seen = set(seq[0].cpu().tolist())
            for t in seen:
                if next_logits[0, t] > 0:
                    next_logits[0, t] /= rep_penalty
                else:
                    next_logits[0, t] *= rep_penalty

            probs = torch.log_softmax(next_logits, dim=-1)
            top_k_probs, top_k_idx = probs.topk(beam_width, dim=-1)

            for k in range(beam_width):
                new_seq = torch.cat([seq, top_k_idx[:, k:k+1]], dim=1)
                new_score = score + top_k_probs[0, k].item()
                candidates.append((new_seq, new_score))

        if not candidates:
            break

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        if all(b[0, -1].item() == tok.eos_token_id for b, _ in beams):
            break

    return tok.decode(beams[0][0][0].tolist(), skip_special_tokens=True)

# ============ TEST DATA ============
def load_test_data(lang):
    """Load cached test data from training - exact filename match only."""
    cache_map = {
        "en": ["librispeech_en_test.json"],
        "hi": ["fleurs_hi_in_test.json"],
        "hinglish": ["mucs_hinglish_hinglish_test.json"],
    }
    files = cache_map.get(lang, [])
    data = []
    for fname in files:
        cf = DATA_DIR / fname
        if cf.exists():
            records = json.load(open(cf))
            print(f"  Loaded {len(records)} samples from {cf.name}")
            data.extend(records)
        else:
            print(f"  WARNING: {cf} not found")
    return data

def process_audio(wav_path):
    """Load and process audio file to mel features."""
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = np.array(audio, dtype=np.float32)
    audio = audio[:int(30.0 * 16000)]

    inputs = processor.feature_extractor(
        [audio], sampling_rate=16000, return_tensors="pt",
        padding=True, max_length=int(30 * 16000)
    )
    feat = inputs["input_features"].to(DEVICE)

    if feat.shape[-1] < WHISPER_EXPECTED_LEN:
        pad = torch.zeros(feat.shape[0], feat.shape[1], WHISPER_EXPECTED_LEN - feat.shape[-1], dtype=feat.dtype, device=DEVICE)
        feat = torch.cat([feat, pad], dim=-1)
    else:
        feat = feat[:, :, :WHISPER_EXPECTED_LEN]

    return feat

# ============ WER ============
def compute_wer(references, hypotheses):
    """Compute Word Error Rate using jiwer."""
    from jiwer import wer as jiwer_wer
    return jiwer_wer(references, hypotheses)

# ============ EVALUATION ============
print("\n" + "=" * 60)
print("POLYWHISPER EVALUATION")
print("=" * 60)

results = {}
all_examples = []

for lang in LANGUAGES:
    print(f"\n--- {lang.upper()} ---")
    test_data = load_test_data(lang)

    if not test_data:
        results[lang] = {"wer": 1.0, "samples": 0, "error": "no test data"}
        continue

    # Sample if too many
    if len(test_data) > MAX_SAMPLES:
        import random
        random.seed(42)
        test_data = random.sample(test_data, MAX_SAMPLES)
        print(f"  Sampled {MAX_SAMPLES} from {len(test_data)} total")

    references = []
    hypotheses = []
    examples = []

    start_time = time.time()

    for i, item in enumerate(tqdm(test_data, desc=f"  {lang}")):
        wav_path = item["wav"]
        ref_text = item["text"]

        if not Path(wav_path).exists():
            continue

        try:
            feat = process_audio(wav_path)
            hyp_text = generate_beam(model, feat, lang, beam_width=5)

            references.append(ref_text)
            hypotheses.append(hyp_text)
            examples.append({"ref": ref_text, "hyp": hyp_text})

            # Show first 3 examples inline
            if len(references) <= 3:
                print(f"    REF: {ref_text[:80]}")
                print(f"    HYP: {hyp_text[:80]}")
                print()

        except Exception as e:
            print(f"    ERROR on sample {i}: {e}")
            continue

    elapsed = time.time() - start_time

    if references:
        wer_score = compute_wer(references, hypotheses)
        results[lang] = {
            "wer": wer_score,
            "samples": len(references),
            "elapsed_sec": elapsed
        }
        all_examples.extend([(lang, ex) for ex in examples])

        print(f"  WER: {wer_score * 100:.1f}%")
        print(f"  Samples: {len(references)}")
        print(f"  Time: {elapsed:.1f}s")
    else:
        results[lang] = {"wer": 1.0, "samples": 0, "error": "no valid samples"}

# ============ SUMMARY ============
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Language':<12} {'WER':>8} {'Samples':>8}")
print("-" * 30)
for lang in LANGUAGES:
    r = results.get(lang, {})
    wer = r.get("wer", 1.0) * 100
    samples = r.get("samples", 0)
    print(f"{lang:<12} {wer:>7.1f}% {samples:>7d}")

# Average WER (weighted by samples)
total_samples = sum(r.get("samples", 0) for r in results.values())
if total_samples > 0:
    avg_wer = sum(r.get("wer", 1.0) * r.get("samples", 0) for r in results.values()) / total_samples
    print(f"\nAvg WER: {avg_wer * 100:.1f}% ({total_samples} samples)")
else:
    print("\nNo samples evaluated.")

# ============ DETAILED EXAMPLES ============
if all_examples:
    print("\n" + "=" * 60)
    print("DETAILED EXAMPLES (first 5 per language)")
    print("=" * 60)
    for lang in LANGUAGES:
        lang_examples = [ex for ex in all_examples if ex[0] == lang]
        print(f"\n--- {lang.upper()} ---")
        for i, (l, ex) in enumerate(lang_examples[:5]):
            print(f"  {i+1}. REF: {ex['ref'][:100]}")
            print(f"     HYP: {ex['hyp'][:100]}")

# ============ SAVE ============
EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(EVAL_FILE, "w"), indent=2)
print(f"\nResults saved to {EVAL_FILE}")
