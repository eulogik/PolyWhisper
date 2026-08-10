import sys, json, random, time
from pathlib import Path

# Minimal imports
import torch
import numpy as np
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WHISPER_MODEL = "openai/whisper-base"
UNFREEZE_LAYERS = [4, 5]
START_TOKEN = 50258

# Import model classes from eval_local
sys.path.insert(0, ".")
from eval_local import PolyWhisper, process_audio, generate_beam, compute_wer, LANGUAGES, AD_HID, AD_LAYERS, AD_HEADS, AD_FFN, AD_RANK, MAX_LABEL_LEN, ENCODE_DIM, VOCAB_SIZE

ADAPTER_DIR = Path("./polywhisper_output/adapters")
DATA_DIR = Path("./polywhisper_output/data")

print("Loading model...")
model = PolyWhisper().to(DEVICE)
model.load_ckpt(str(ADAPTER_DIR / "hinglish_best.pt"))
model.eval()

random.seed(42)
data = json.load(open(DATA_DIR / "mucs_hinglish_hinglish_test.json"))
data = random.sample(data, 30)

refs, hyps = [], []
t0 = time.time()
for i, item in enumerate(data):
    feat = process_audio(item["wav"])
    hyp = generate_beam(model, feat, "hinglish", beam_width=5)
    refs.append(item["text"])
    hyps.append(hyp)
    print(f"{i+1}/30 REF: {item['text'][:70]}")
    print(f"        HYP: {hyp[:70]}")

wer = compute_wer(refs, hyps)
print(f"\nHINGLISH WER: {wer*100:.1f}% ({len(refs)} samples, {time.time()-t0:.0f}s)")
