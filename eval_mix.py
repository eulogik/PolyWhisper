"""Eval: old adapters (frozen encoder) + new hinglish adapter on original encoder"""
import torch, json, math, time, random
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf, numpy as np

DEVICE = "mps"
WHISPER_MODEL = "openai/whisper-base"
UNFREEZE_LAYERS = [4, 5]
START_TOKEN = 50258
AD_HID = 256; AD_LAYERS = 3; AD_HEADS = 4; AD_FFN = 1024; AD_RANK = 16
MAX_LABEL_LEN = 256; ENCODE_DIM = 512; MAX_NEW_TOKENS = 128
WHISPER_EXPECTED_LEN = 3000

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
VOCAB_SIZE = len(processor.tokenizer)
tok = processor.tokenizer

import sys; sys.path.insert(0, ".")
# Copy model classes from eval_local content
exec(Path("eval_local.py").read_text().split("# ============ TEST DATA")[0].strip())

print("\n=== TEST 1: Original Whisper encoder + old en adapter ===")
model = PolyWhisper().to(DEVICE)
# Keep original encoder (don't load any _encoder)
# Load old English adapter (no encoder overwrite)
old_en = torch.load("polywhisper_output/adapters/en_ep79_last.pt", weights_only=True)
model.adapters["en"].load_state_dict(old_en["en"])
print("  Loaded old en adapter (ep79)")

# Quick English eval
random.seed(42)
en_data = json.load(open("polywhisper_output/data/librispeech_en_test.json"))
en_data = random.sample(en_data, 50)
refs, hyps = [], []
for item in en_data:
    audio, sr = sf.read(item["wav"])
    if sr != 16000:
        import librosa; audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = np.array(audio, dtype=np.float32)[:int(30.0 * 16000)]
    inp = processor.feature_extractor([audio], sampling_rate=16000, return_tensors="pt", padding=True)
    feat = inp["input_features"].to(DEVICE)
    if feat.shape[-1] < WHISPER_EXPECTED_LEN:
        pad = torch.zeros(*feat.shape[:2], WHISPER_EXPECTED_LEN - feat.shape[-1], dtype=feat.dtype, device=DEVICE)
        feat = torch.cat([feat, pad], dim=-1)
    else: feat = feat[:, :, :WHISPER_EXPECTED_LEN]
    
    enc = model.encode(feat)
    dec_ids = torch.tensor([[START_TOKEN]], device=DEVICE)
    for _ in range(MAX_NEW_TOKENS):
        logits = model.adapters["en"](enc, dec_ids)[:, -1, :]
        seen = set(dec_ids[0].cpu().tolist())
        for t in seen:
            if logits[0, t] > 0: logits[0, t] /= 1.2
            else: logits[0, t] *= 1.2
        topk_vals, _ = logits.topk(50, dim=-1)
        logits[logits < topk_vals[:, -1:]] = float('-inf')
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        dec_ids = torch.cat([dec_ids, nxt], dim=1)
        if nxt.item() == tok.eos_token_id: break
    refs.append(item["text"]); hyps.append(tok.decode(dec_ids[0].cpu().tolist(), skip_special_tokens=True))

from jiwer import wer as jiwer_wer
en_wer = jiwer_wer(refs, hyps)
print(f"  EN WER (greedy): {en_wer*100:.1f}%")

print("\n=== TEST 2: Original encoder + new hinglish adapter ===")
model2 = PolyWhisper().to(DEVICE)
ckpt = torch.load("polywhisper_output/adapters/hinglish_best.pt", weights_only=True)
model2.adapters["hinglish"].load_state_dict(ckpt["hinglish"])
# Don't load _encoder — keep original Whisper encoder
print("  Loaded new hinglish adapter with ORIGINAL encoder")

random.seed(42)
hi_data = json.load(open("polywhisper_output/data/mucs_hinglish_hinglish_test.json"))
hi_data = random.sample(hi_data, 30)
refs2, hyps2 = [], []
for item in hi_data:
    audio, sr = sf.read(item["wav"])
    import librosa; audio = librosa.resample(np.array(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
    audio = audio[:int(30.0 * 16000)]
    inp = processor.feature_extractor([audio], sampling_rate=16000, return_tensors="pt", padding=True)
    feat = inp["input_features"].to(DEVICE)
    if feat.shape[-1] < WHISPER_EXPECTED_LEN:
        pad = torch.zeros(*feat.shape[:2], WHISPER_EXPECTED_LEN - feat.shape[-1], dtype=feat.dtype, device=DEVICE)
        feat = torch.cat([feat, pad], dim=-1)
    else: feat = feat[:, :, :WHISPER_EXPECTED_LEN]
    
    enc = model2.encode(feat)
    # Beam search
    beams = [(torch.tensor([[START_TOKEN]], device=DEVICE), 0.0)]
    for step in range(MAX_NEW_TOKENS):
        candidates = []
        for seq, score in beams:
            if seq.size(1) > 1 and seq[0, -1].item() == tok.eos_token_id:
                candidates.append((seq, score)); continue
            logits = model2.adapters["hinglish"](enc, seq)[:, -1, :].clone()
            seen = set(seq[0].cpu().tolist())
            for t in seen:
                if logits[0, t] > 0: logits[0, t] /= 1.2
                else: logits[0, t] *= 1.2
            probs = torch.log_softmax(logits, dim=-1)
            topk_probs, topk_idx = probs.topk(5, dim=-1)
            for k in range(5):
                candidates.append((torch.cat([seq, topk_idx[:, k:k+1]], dim=1), score + top_k_probs[0, k].item()))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
        if all(b[0, -1].item() == tok.eos_token_id for b, _ in beams): break
    refs2.append(item["text"]); hyps2.append(tok.decode(beams[0][0][0].tolist(), skip_special_tokens=True))

hi_wer = jiwer_wer(refs2, hyps2)
print(f"  Hinglish WER (beam-5, original encoder): {hi_wer*100:.1f}%")

print(f"\n=== SUMMARY ===")
print(f"  EN (old adapter, original encoder): {en_wer*100:.1f}% greedy")
print(f"  Hinglish (new adapter, original encoder): {hi_wer*100:.1f}% beam-5")
