"""Final eval: OLD en+hi adapters + NEW hinglish adapter, all on ORIGINAL encoder"""
import torch, json, random, time
exec(open("eval_local.py").read().split("# ============ TEST DATA")[0])

import gc
from jiwer import wer as jiwer_wer

model = PolyWhisper().to(DEVICE)

# Load old English adapter (ep79, no encoder overwrite)
old_en = torch.load("polywhisper_output/adapters/en_ep79_last.pt", weights_only=True)
model.adapters["en"].load_state_dict(old_en["en"])
print("Loaded old en (ep79) — original encoder preserved")

# Load old Hindi adapter (ep79, no encoder overwrite)
old_hi = torch.load("polywhisper_output/adapters/hi_ep79_last.pt", weights_only=True)
model.adapters["hi"].load_state_dict(old_hi["hi"])
print("Loaded old hi (ep79) — original encoder preserved")

# Load new hinglish adapter (NO encoder)
new_hi = torch.load("polywhisper_output/adapters/hinglish_best.pt", weights_only=True)
model.adapters["hinglish"].load_state_dict(new_hi["hinglish"])
print("Loaded new hinglish — original encoder preserved")

model.eval()
print(f"\nEncoder layers 4-5: original Whisper weights (frozen)\n")

random.seed(42)
results = {}

for lang, cache_file, n_samples in [
    ("en", "librispeech_en_test.json", 50),
    ("hi", "fleurs_hi_in_test.json", 50),
    ("hinglish", "mucs_hinglish_hinglish_test.json", 30),
]:
    data = json.load(open(f"polywhisper_output/data/{cache_file}"))
    data = random.sample(data, min(n_samples, len(data)))
    refs, hyps = [], []
    t0 = time.time()
    for item in data:
        feat = process_audio(item["wav"])
        hyps.append(generate_beam(model, feat, lang))
        refs.append(item["text"])
    wer = jiwer_wer(refs, hyps)
    results[lang] = {"wer": wer, "samples": len(refs), "time": time.time()-t0}
    print(f"{lang.upper():>8}: {wer*100:5.1f}%  ({len(refs)} samples, {results[lang]['time']:.0f}s)")
    for i in range(min(2, len(data))):
        print(f"  REF: {refs[i][:70]}")
        print(f"  HYP: {hyps[i][:70]}")
    print()

print("="*50)
print("SUMMARY — OLD adapters + ORIGINAL encoder")
for lang in ["en", "hi", "hinglish"]:
    r = results[lang]
    print(f"  {lang:>8}: {r['wer']*100:5.1f}%  ({r['samples']} samples)")
