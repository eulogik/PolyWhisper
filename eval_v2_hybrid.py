"""
PolyWhisper v2 Hybrid Evaluation
Vanilla Whisper Base for English, LoRA adapters for hi/hinglish.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import random
import re as _re
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

WHISPER_MODEL = "openai/whisper-base"
WHISPER_EXPECTED_LEN = 3000
LANGUAGES = ["en", "hi", "hinglish"]
MAX_NEW_TOKENS = 128
MAX_SAMPLES = 200
RANK = 16

SAVE_DIR = Path("./polywhisper_output")
ADAPTER_DIR = SAVE_DIR / "adapters_v2"
DATA_DIR = SAVE_DIR / "data"
EVAL_FILE = SAVE_DIR / "eval_v2_hybrid_results.json"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
tok = processor.tokenizer

PROJ_MAP = [
    ('self_attn', 'self_q'), ('self_attn', 'self_k'), ('self_attn', 'self_v'),
    ('encoder_attn', 'cross_q'), ('encoder_attn', 'cross_k'), ('encoder_attn', 'cross_v'),
    ('self_attn', 'self_out'), ('encoder_attn', 'cross_out'),
]

class PolyWhisperV2(nn.Module):
    def __init__(self, whisper_name=WHISPER_MODEL, rank=RANK):
        super().__init__()
        self.rank = rank
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        for p in self.whisper.parameters():
            p.requires_grad = False
        self.d_model = self.whisper.config.d_model
        self.decoder_layers = self.whisper.config.decoder_layers
        self.lora_adapters = nn.ModuleDict()
        self._lora_hooks = []
        self._current_lang = None

    def add_language(self, lang):
        d, r = self.d_model, self.rank
        n = self.decoder_layers
        dev = next(self.whisper.parameters()).device
        adpt = nn.ModuleDict()
        for i in range(n):
            for _, base_name in PROJ_MAP:
                adpt[f'{i}_{base_name}_a'] = nn.Linear(d, r, bias=False).to(dev)
                adpt[f'{i}_{base_name}_b'] = nn.Linear(r, d, bias=False).to(dev)
                nn.init.zeros_(adpt[f'{i}_{base_name}_b'].weight)
        self.lora_adapters[lang] = adpt
        return adpt

    def set_language(self, lang):
        if self._current_lang == lang:
            return
        self._remove_lora_hooks()
        if lang not in self.lora_adapters:
            self.add_language(lang)
        lora = self.lora_adapters[lang]
        self._lora_hooks = []
        for i, layer in enumerate(self.whisper.model.decoder.layers):
            for attn_name, base_name in PROJ_MAP:
                a = lora[f'{i}_{base_name}_a']
                b = lora[f'{i}_{base_name}_b']
                def make_hook(a, b):
                    def hook(mod, inp, out):
                        return out + b(a(inp[0]))
                    return hook
                if 'out' in base_name:
                    proj_name = 'out_proj'
                elif 'q' in base_name:
                    proj_name = 'q_proj'
                elif 'k' in base_name:
                    proj_name = 'k_proj'
                else:
                    proj_name = 'v_proj'
                proj = getattr(getattr(layer, attn_name), proj_name)
                self._lora_hooks.append(proj.register_forward_hook(make_hook(a, b)))
        self._current_lang = lang

    def _remove_lora_hooks(self):
        for handle in self._lora_hooks:
            handle.remove()
        self._lora_hooks = []

    def load_adapter(self, lang, path):
        if lang not in self.lora_adapters:
            self.add_language(lang)
        st = torch.load(path, map_location="cpu", weights_only=True)
        self.lora_adapters[lang].load_state_dict(st)

    @torch.no_grad()
    def generate(self, feat, lang, **gen_kwargs):
        self.set_language(lang)
        return self.whisper.generate(feat, **gen_kwargs)

print("Loading model...")
model = PolyWhisperV2().to(DEVICE)
vanilla = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(DEVICE)
vanilla.eval()
print(f"Loaded LoRA (en) + vanilla (hi/hinglish)")

# en uses vanilla; hi/hinglish use LoRA
for lang in ["hi", "hinglish"]:
    p = ADAPTER_DIR / f"{lang}_best.pt"
    if p.exists():
        model.load_adapter(lang, str(p))
        print(f"  Loaded LoRA adapter: {lang} ({p.stat().st_size/1024:.0f}KB)")
    else:
        print(f"  WARNING: No adapter for {lang}")

model.eval()

def load_test_data(lang):
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
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = np.array(audio, dtype=np.float32)
    audio = audio[:int(30.0 * 16000)]
    inputs = processor.feature_extractor(
        [audio], sampling_rate=16000, return_tensors="pt", padding=True
    )
    feat = inputs["input_features"].to(DEVICE)
    if feat.shape[-1] < WHISPER_EXPECTED_LEN:
        pad = torch.zeros(feat.shape[0], feat.shape[1], WHISPER_EXPECTED_LEN - feat.shape[-1], dtype=feat.dtype, device=DEVICE)
        feat = torch.cat([feat, pad], dim=-1)
    else:
        feat = feat[:, :, :WHISPER_EXPECTED_LEN]
    return feat

from eval_metrics import compute_wer, compute_cer

print("\n" + "=" * 60)
print("POLYWHISPER V2 HYBRID EVALUATION")
print("=" * 60)

results = {}
all_examples = []

for lang in LANGUAGES:
    print(f"\n--- {lang.upper()} ({'VANILLA' if lang == 'en' else 'LORA'}) ---")
    test_data = load_test_data(lang)
    if not test_data:
        results[lang] = {"wer": 1.0, "samples": 0, "error": "no test data"}
        continue
    if len(test_data) > MAX_SAMPLES:
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
            forced_ids = processor.get_decoder_prompt_ids(language=lang if lang != "hinglish" else "hi", task="transcribe")
            if lang == "en":
                out = vanilla.generate(feat,
                    forced_decoder_ids=forced_ids,
                    num_beams=5,
                    max_length=MAX_NEW_TOKENS,
                )
            else:
                out = model.generate(feat, lang,
                    forced_decoder_ids=forced_ids,
                    num_beams=5,
                    max_length=MAX_NEW_TOKENS,
                )
            hyp_text = tok.decode(out[0].tolist(), skip_special_tokens=True)
            references.append(ref_text)
            hypotheses.append(hyp_text)
            examples.append({"ref": ref_text, "hyp": hyp_text})
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
        cer_score = compute_cer(references, hypotheses)
        results[lang] = {"wer": wer_score, "cer": cer_score, "samples": len(references), "elapsed_sec": elapsed}
        all_examples.extend([(lang, ex) for ex in examples])
        print(f"  WER: {wer_score * 100:.1f}%")
        print(f"  CER: {cer_score * 100:.1f}%")
        print(f"  Samples: {len(references)}")
        print(f"  Time: {elapsed:.1f}s")
    else:
        results[lang] = {"wer": 1.0, "cer": 1.0, "samples": 0, "error": "no valid samples"}

print("\n" + "=" * 60)
print("HYBRID RESULTS SUMMARY")
print("=" * 60)
print(f"{'Language':<12} {'WER':>8} {'CER':>8} {'Samples':>8}")
print("-" * 38)
for lang in LANGUAGES:
    r = results.get(lang, {})
    wer = r.get("wer", 1.0) * 100
    cer = r.get("cer", 1.0) * 100
    samples = r.get("samples", 0)
    print(f"{lang:<12} {wer:>7.1f}% {cer:>7.1f}% {samples:>7d}")

total_samples = sum(r.get("samples", 0) for r in results.values())
if total_samples > 0:
    avg_wer = sum(r.get("wer", 1.0) * r.get("samples", 0) for r in results.values()) / total_samples
    avg_cer = sum(r.get("cer", 1.0) * r.get("samples", 0) for r in results.values()) / total_samples
    print(f"\nAvg WER: {avg_wer * 100:.1f}% | Avg CER: {avg_cer * 100:.1f}% ({total_samples} samples)")
else:
    print("\nNo samples evaluated.")

EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(EVAL_FILE, "w"), indent=2)

SAMPLES_FILE = SAVE_DIR / "eval_v2_hybrid_samples.json"
samples_out = {lang: [ex for l, ex in all_examples if l == lang] for lang in LANGUAGES}
json.dump(samples_out, open(SAMPLES_FILE, "w"), indent=2, ensure_ascii=False)
print(f"\nResults saved to {EVAL_FILE}")
print(f"Samples saved to {SAMPLES_FILE}")
