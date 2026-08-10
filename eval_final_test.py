import torch, json, random
from transformers import WhisperProcessor, WhisperForConditionalGeneration as WFCG

DEVICE = "mps"
tok = WhisperProcessor.from_pretrained("openai/whisper-base").tokenizer

# Load model classes + generation functions from eval_local (skip test data & eval loop)
exec(open("eval_local.py").read().split("# ============ TEST DATA")[0])
import gc

# Fresh model with ORIGINAL encoder (no encoder override)
model = PolyWhisper().to(DEVICE)
print(f"Loaded PolyWhisper with original encoder. Adapter keys: {list(model.adapters.keys())}")

# Load old English adapter ONLY (no _encoder, so encoder stays original)
old_en = torch.load("polywhisper_output/adapters/en_ep79_last.pt", weights_only=True)
model.adapters["en"].load_state_dict(old_en["en"])
print("Loaded old en adapter (ep79)")

# Load new hinglish adapter ONLY (no _encoder)
new_hi = torch.load("polywhisper_output/adapters/hinglish_best.pt", weights_only=True)
model.adapters["hinglish"].load_state_dict(new_hi["hinglish"])
print("Loaded new hinglish adapter")

# Load old hi adapter
old_hi = torch.load("polywhisper_output/adapters/hi_ep79_last.pt", weights_only=True)
model.adapters["hi"].load_state_dict(old_hi["hi"])
print("Loaded old hi adapter (ep79)")

# Quick test: 10 English samples
def test_one(wav_path):
    import soundfile as sf, librosa, numpy as np
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        audio = librosa.resample(np.array(audio, np.float32), orig_sr=sr, target_sr=16000)
    audio = np.array(audio, np.float32)[:480000]
    inp = WhisperProcessor.from_pretrained("openai/whisper-base").feature_extractor([audio], sampling_rate=16000, return_tensors="pt", padding=True)
    feat = inp["input_features"].to(DEVICE)
    n = 3000 - feat.shape[-1]
    if n > 0: feat = torch.cat([feat, torch.zeros(1, 80, n, device=DEVICE)], dim=-1)
    else: feat = feat[:,:,:3000]
    return feat

random.seed(42)
en_data = json.load(open("polywhisper_output/data/librispeech_en_test.json"))
en_data = random.sample(en_data, 10)
for item in en_data:
    feat = test_one(item["wav"])
    hyp = generate_beam(model, feat, "en")
    beam5 = (lambda: None)
    print(f" REF: {item['text'][:60]}")
    print(f" HYP: {hyp[:60]}")
    print()

print("Now run eval_local.py with these mixed adapters for full results")
print("Model has: old en, old hi, new hinglish — all on ORIGINAL encoder")
