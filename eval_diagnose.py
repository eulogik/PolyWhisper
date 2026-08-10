"""Standalone diagnostic"""
import sys, torch, torch.nn as nn, numpy as np, json, math, time, random
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

DEVICE = "mps"
WHISPER_MODEL = "openai/whisper-base"
START_TOKEN = 50258
AD_HID = 256; AD_LAYERS = 3; AD_HEADS = 4; AD_FFN = 1024; AD_RANK = 16
MAX_LABEL_LEN = 256; ENCODE_DIM = 512; MAX_NEW_TOKENS = 128
WHISPER_EXPECTED_LEN = 3000

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
VOCAB_SIZE = len(processor.tokenizer)
tok = processor.tokenizer

# Reuse model classes from eval_local.py by importing its classes only (not the eval code)
# First define everything we need inline
exec(open("eval_local.py").read().split("# ============ TEST DATA")[0])

# Just keep the class definitions, discard the model instantiation
import gc
# Create a fresh model
model = PolyWhisper().to(DEVICE)

def load_audio(wav_path):
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        import librosa; audio = librosa.resample(np.array(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
    audio = np.array(audio, dtype=np.float32)[:int(30.0 * 16000)]
    inp = processor.feature_extractor([audio], sampling_rate=16000, return_tensors="pt", padding=True)
    feat = inp["input_features"].to(DEVICE)
    if feat.shape[-1] < WHISPER_EXPECTED_LEN:
        pad = torch.zeros(*feat.shape[:2], WHISPER_EXPECTED_LEN - feat.shape[-1], dtype=feat.dtype, device=DEVICE)
        feat = torch.cat([feat, pad], dim=-1)
    else: feat = feat[:, :, :WHISPER_EXPECTED_LEN]
    return feat

# TEST: Combined best from different runs
print("=== LOAD OLD EN + NEW HINGLISH ADAPTERS WITH ORIGINAL ENCODER ===")

# Load old English adapter (no encoder overwrite)
old_en = torch.load("polywhisper_output/adapters/en_ep79_last.pt", weights_only=True)
model.load_ckpt_with_adapter_only = lambda lang, sd: model.adapters[lang].load_state_dict(sd)
model.adapters["en"].load_state_dict(old_en["en"])
print("  Loaded old en (ep79)")

random.seed(42)
en_data = json.load(open("polywhisper_output/data/librispeech_en_test.json"))
en_data = random.sample(en_data, 20)
refs, hyps = [], []
for item in en_data:
    feat = load_audio(item["wav"])
    hyps.append(generate_beam(model, feat, "en"))
    refs.append(item["text"])
print(f"  EN beam-5 (old adapter, original encoder): {compute_wer(refs, hyps)*100:.1f}%")

