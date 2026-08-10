"""
PolyWhisper v2 — LoRA on Whisper Small decoder
- Whisper Small (244M) fully frozen
- Per-language LoRA on decoder q/k/v projections (rank 16)
- ~1.8M trainable params per language (~3s/step on MPS)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import math
import gc
import time
import signal
import sys
import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

# ============ CONFIG ============
import os
os.environ.setdefault("HF_TOKEN", "")

WHISPER_MODEL = "openai/whisper-base"
WHISPER_EXPECTED_LEN = 3000  # max, but we don't pad short audio to this
LANGUAGES = ["en", "hi", "hinglish"]
NUM_EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
RANK = 16
MAX_LABEL_LEN = 256
MAX_SAMPLES_PER_LANG = 20000
MAX_SAMPLES_HINGLISH = 42000

# Whisper token IDs for forced prefix
# Position 0 = BOS (50257), same token used at inference start
# Positions 1-3 = language, task, no_timestamps
START_TOKEN = 50257
LANG_TOKENS = {"en": 50259, "hi": 50276, "hinglish": 50276}
TASK_TOKEN = 50359
NO_TIME_TOKEN = 50363
BOS_EOS = 50257
SAVE_EVERY = 5
MAX_RUNTIME_HOURS = 999

# ============ PATHS ============
SAVE_DIR = Path("./polywhisper_output")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = SAVE_DIR / "adapters_v2"
ADAPTER_DIR.mkdir(exist_ok=True)
DATA_DIR = SAVE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = SAVE_DIR / "training_v2.log"
STATE_FILE = SAVE_DIR / "training_state_v2.json"
HEARTBEAT_FILE = SAVE_DIR / "heartbeat_v2.txt"
CRASH_LOG = SAVE_DIR / "crash_v2.log"

# ============ LOGGING ============
def log(msg, also_print=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)

def log_crash(e, tb):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CRASH_LOG, "a") as f:
        f.write(f"\n{'='*60}\n[{ts}] CRASH: {e}\n{tb}\n")

def update_heartbeat(msg="alive"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    HEARTBEAT_FILE.write_text(f"{ts} | {msg}")

def check_runtime(start_time):
    elapsed = time.time() - start_time
    if elapsed > MAX_RUNTIME_HOURS * 3600:
        log(f"  Runtime limit reached ({MAX_RUNTIME_HOURS}h). Stopping gracefully.")
        return False
    return True

# ============ DEVICE ============
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
log(f"Device: {DEVICE}")

USE_FP16 = (DEVICE == "cuda")
scaler = torch.amp.GradScaler("cuda") if USE_FP16 else None

# ============ PROCESSOR ============
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
VOCAB_SIZE = len(processor.tokenizer)
tok = processor.tokenizer
log(f"Vocab: {VOCAB_SIZE}")

# ============ MODEL ============
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

    PROJ_MAP = [
        ('self_attn', 'self_q'), ('self_attn', 'self_k'), ('self_attn', 'self_v'),
        ('encoder_attn', 'cross_q'), ('encoder_attn', 'cross_k'), ('encoder_attn', 'cross_v'),
        ('self_attn', 'self_out'), ('encoder_attn', 'cross_out'),
    ]

    def add_language(self, lang):
        d, r = self.d_model, self.rank
        n = self.decoder_layers
        dev = next(self.whisper.parameters()).device
        adpt = nn.ModuleDict()
        for i in range(n):
            for _, base_name in self.PROJ_MAP:
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
            for attn_name, base_name in self.PROJ_MAP:
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

    def forward(self, feat, dec_ids, lang):
        self.set_language(lang)
        enc = self.whisper.model.encoder(feat).last_hidden_state
        dec_out = self.whisper.model.decoder(dec_ids, encoder_hidden_states=enc, output_hidden_states=False)
        return self.whisper.proj_out(dec_out.last_hidden_state)

    @torch.no_grad()
    def generate(self, feat, lang, **gen_kwargs):
        self.set_language(lang)
        return self.whisper.generate(feat, **gen_kwargs)

    def save_adapter(self, lang, path):
        torch.save(self.lora_adapters[lang].state_dict(), path)

    def load_adapter(self, lang, path):
        if lang not in self.lora_adapters:
            self.add_language(lang)
        st = torch.load(path, map_location="cpu", weights_only=True)
        self.lora_adapters[lang].load_state_dict(st)

log("Building model...")
model = PolyWhisperV2().to(DEVICE)
for lang in LANGUAGES:
    model.add_language(lang)
    log(f"  Added LoRA adapter: {lang} ({sum(p.numel() for p in model.lora_adapters[lang].parameters())/1e3:.0f}K params)")

total_lora = sum(p.numel() for p in model.lora_adapters.parameters() if p.requires_grad)
log(f"Total trainable (LoRA): {total_lora/1e6:.2f}M params")

# ============ LR SCHEDULER ============
def get_lr(step, warmup=WARMUP_STEPS, total_steps=100000):
    if step < warmup:
        return LR * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return LR * max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

# Scheduled sampling: linear decay from SS_START_PROB to 0 over SS_EPOCHS
SS_START_EPOCH = 3
SS_EPOCHS = 10
SS_START_PROB = 0.15
def get_ss_prob(ep):
    if ep < SS_START_EPOCH or ep >= SS_START_EPOCH + SS_EPOCHS:
        return 0.0
    progress = (ep - SS_START_EPOCH) / SS_EPOCHS
    return SS_START_PROB * (1 - progress)

# ============ DATA ============
import soundfile as sf
from datasets import load_dataset

def resample(audio, orig_sr):
    if orig_sr != 16000:
        import librosa
        audio = librosa.resample(np.array(audio, dtype=np.float32), orig_sr=orig_sr, target_sr=16000)
    return np.array(audio, dtype=np.float32)

def process_sample(audio_array, sr, text, adir, idx):
    audio = resample(audio_array, sr)
    if len(audio) < 1600 or len(audio) > int(30.0 * 16000):
        return None
    text = text.strip()
    if not text or len(text) < 3:
        return None
    wav_path = adir / f"{idx:06d}.wav"
    sf.write(str(wav_path), audio, 16000)
    return {"wav": str(wav_path), "text": text}

class AudioDataset(Dataset):
    def __init__(self, name, lang, split, max_samples, text_key, audio_key="audio"):
        self.name = name
        self.lang = lang
        self.cache = DATA_DIR / f"{name}_{lang}_{split}.json"
        self.adir = DATA_DIR / f"audio_{name}_{lang}"
        self.adir.mkdir(parents=True, exist_ok=True)
        self.data = self._load(split, max_samples, text_key, audio_key)

    # Template keywords for Hinglish cleaning
    _TEMPLATE_KW = [
        "spoken tutorial", "bandwidth", "download", "link पर",
        "mission पर अधिक जानकारी", "प्रमाणपत्र", "नियतकार्य", "सारांशित",
        "contact @spokentutorial", "online test",
        "हमसे जुड़ने के लिए धन्यवाद", "close पर click", "ok पर click",
        "save पर click", "ok button पर click", "save button पर click",
        "enter दबाएं", "enter दबाएँ",
        "आई आई टी बॉम्बे से मैं श्रुति आर्य", "आई आई टी बॉम्बे की ओर से मैं",
        "यह स्क्रिप्ट प्रभाकर द्वारा अनुवादित",
        "यह स्क्रिप्ट विकास द्वारा अनुवादित",
        "spoken hyphen tutorial", "talktoa teacher", "talktoateacher",
        "कार्यशालाएं चलाती", "कार्यशालाएँ", "nmeict", "mhrd", "एमएचआरडी",
    ]

    @staticmethod
    def _is_template(text):
        tl = text.lower()
        return any(kw in tl for kw in AudioDataset._TEMPLATE_KW)

    def _load(self, split, max_samples, text_key, audio_key):
        # Check for cleaned version first
        cleaned = DATA_DIR / f"{self.name}_{self.lang}_{split}_cleaned.json"
        if cleaned.exists():
            self.cache = cleaned
            log(f"  Cleaned cache: {self.name}/{self.lang}/{split}")
            return json.load(open(cleaned))
        # Check for regular cache
        if self.cache.exists():
            log(f"  Cached {self.name}/{self.lang}/{split}")
            data = json.load(open(self.cache))
            # Apply max_samples limit
            if len(data) > max_samples:
                log(f"  Truncating to {max_samples} (cached has {len(data)})")
                data = data[:max_samples]
            # Dedup on load
            seen = set()
            deduped = []
            removed_template = 0
            for d in data:
                text = d["text"].strip()
                if "hinglish" in self.lang and self._is_template(text):
                    removed_template += 1
                    continue
                key = text.lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(d)
            if removed_template or len(deduped) < len(data):
                log(f"  Cleaned: templates={removed_template} dedup={len(data)-len(deduped)-removed_template} -> {len(deduped)}")
                return deduped
            return data
        log(f"  Downloading {self.name} {self.lang} ({split})...")
        ds = None
        try:
            if self.name == "librispeech":
                ds = load_dataset("openslr/librispeech_asr", "clean", split=split, streaming=True)
            elif self.name == "fleurs":
                ds = load_dataset("google/fleurs", self.lang, split=split, streaming=True)
            elif self.name == "mucs_hinglish":
                ds = load_dataset("dianavdavidson/MUCS-Hinglish-traintestblindsplit", split=split, streaming=True)
            elif self.name == "indicvoices_st":
                ds = load_dataset("ai4bharat/IndicVoices-ST", "indic2en", split=split, streaming=True, token=os.environ["HF_TOKEN"])
        except Exception as e:
            log(f"  FAILED to load {self.name}/{self.lang}: {e}")
            return []
        if ds is None:
            log(f"  WARNING: Could not load {self.name}/{self.lang}")
            return []
        recs = []
        stall_time = 0
        last_count = 0
        for i, item in enumerate(tqdm(ds, desc=f"  {self.name}/{self.lang}")):
            if i >= max_samples:
                break
            try:
                if self.name == "indicvoices_st" and item.get("alignment_score", 0) < 0.8:
                    continue
                a = item[audio_key]
                if isinstance(a, dict):
                    audio_array, sr = a["array"], a["sampling_rate"]
                else:
                    audio_array, sr = a["array"], a["sampling_rate"]
                result = process_sample(audio_array, sr, item[text_key], self.adir, len(recs))
                if result:
                    recs.append(result)
            except Exception:
                continue
            if i > 0 and i % 50 == 0:
                if len(recs) == last_count:
                    stall_time += 30
                    if stall_time > 60:
                        log(f"  STALL detected at {i} items, aborting. Got {len(recs)} samples.")
                        break
                else:
                    stall_time = 0
                last_count = len(recs)
        if len(recs) > 0:
            json.dump(recs, open(self.cache, "w"))
        log(f"  Saved {len(recs)} samples")
        return recs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        a, _ = sf.read(r["wav"])
        return {"audio": a, "text": r["text"]}

class Collator:
    def __init__(self, proc, max_lbl, lang):
        self.proc = proc
        self.max_lbl = max_lbl
        self.forced_prefix = [START_TOKEN, LANG_TOKENS[lang], TASK_TOKEN, NO_TIME_TOKEN]
    def __call__(self, batch):
        auds = [item["audio"] for item in batch]
        ft = self.proc.feature_extractor(auds, sampling_rate=16000, return_tensors="pt", padding=True)
        feats = ft["input_features"]
        B, C, T = feats.shape
        if T < WHISPER_EXPECTED_LEN:
            pad = torch.zeros(B, C, WHISPER_EXPECTED_LEN - T, dtype=feats.dtype)
            feats = torch.cat([feats, pad], dim=-1)
        else:
            feats = feats[:, :, :WHISPER_EXPECTED_LEN]
        lbls = []
        for item in batch:
            toks = self.proc.tokenizer(item["text"])["input_ids"]
            toks = [t for t in toks if t != BOS_EOS]
            full = self.forced_prefix + toks + [BOS_EOS]
            if len(full) > self.max_lbl:
                full = full[:self.max_lbl]
            while len(full) < self.max_lbl:
                full.append(-100)
            lbls.append(full)
        return {"feats": feats, "lbls": torch.tensor(lbls, dtype=torch.long)}

def make_loader(dataset, lang, batch_size=BATCH_SIZE, shuffle=True):
    if len(dataset) == 0:
        return []
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=Collator(processor, MAX_LABEL_LEN, lang),
        num_workers=0, pin_memory=(DEVICE == "cuda"),
    )

# ============ RESUME STATE ============
def new_state():
    return {
        "epoch": 0, "lang": LANGUAGES[0], "step": 0, "best_loss": {}, "lang_done": {},
        "global_step": 0, "start_time": time.time(),
    }

def load_state():
    if not STATE_FILE.exists():
        log("  Starting fresh")
        return new_state()
    try:
        st = json.load(open(STATE_FILE))
    except Exception as e:
        log(f"  Corrupt state: {e}, starting fresh")
        return new_state()
    # Load adapters: best first, fall back to last checkpoint for current epoch
    cur_ep = st.get("epoch", 0)
    for l in LANGUAGES:
        bp = ADAPTER_DIR / f"{l}_best.pt"
        if bp.exists():
            model.load_adapter(l, str(bp))
            log(f"  Loaded best LoRA adapter: {l}")
        else:
            lp = ADAPTER_DIR / f"{l}_ep{cur_ep}_last.pt"
            if lp.exists():
                model.load_adapter(l, str(lp))
                log(f"  Loaded last LoRA adapter: {l} (epoch {cur_ep})")
            else:
                log(f"  No adapter for {l}")
    log(f"  Resuming epoch {st['epoch']} lang {st['lang']} step {st['step']}")
    return st

def save_state(state):
    lang = state["lang"]
    ep = state["epoch"]
    ckpt = ADAPTER_DIR / f"{lang}_ep{ep}_last.pt"
    model.save_adapter(lang, str(ckpt))
    state_to_save = {k: v for k, v in state.items()}
    json.dump(state_to_save, open(STATE_FILE, "w"), indent=2)

def save_best(state):
    lang = state["lang"]
    p = ADAPTER_DIR / f"{lang}_best.pt"
    model.save_adapter(lang, str(p))

# ============ OPTIMIZER HELPERS ============
def make_optimizer(lang):
    return torch.optim.AdamW(model.lora_adapters[lang].parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ============ TRAINING ============
log("="*60)
log("PolyWhisper v2 Training — LoRA on Whisper Small Decoder")
log(f"Languages: {LANGUAGES}")
log(f"Epochs: {NUM_EPOCHS}, LR: {LR}, Rank: {RANK}")
log(f"Batch size: {BATCH_SIZE}, Max runtime: {MAX_RUNTIME_HOURS}h")
log("="*60)

log("Loading datasets...")
train_sets = {}
test_sets = {}
TRAIN_CONFIG = {
    "en": ("librispeech", "en", "text", "audio", "train.100"),
    "hi": ("indicvoices_st", "hindi", "text", "chunked_audio_filepath", "hindi"),
    "hinglish": ("mucs_hinglish", "hinglish", "transcript", "audio", "train"),
}
TEST_CONFIG = {
    "en": ("librispeech", "en", "text", "audio", "test"),
    "hi": ("fleurs", "hi_in", "transcription", "audio", "test"),
    "hinglish": ("mucs_hinglish", "hinglish", "transcript", "audio", "test"),
}
for lang in LANGUAGES:
    log(f"Loading {lang}...")
    max_samples = MAX_SAMPLES_HINGLISH if lang == "hinglish" else MAX_SAMPLES_PER_LANG
    name, lang_code, text_key, audio_key, split = TRAIN_CONFIG[lang]
    train_sets[lang] = AudioDataset(name, lang_code, split, max_samples, text_key, audio_key)
    name, lang_code, text_key, audio_key, split = TEST_CONFIG[lang]
    test_sets[lang] = AudioDataset(name, lang_code, split, max_samples, text_key, audio_key)
    log(f"  {lang}: {len(train_sets[lang])} train, {len(test_sets[lang])} test")

state = load_state()
start_time = time.time()
total_steps_done = state.get("global_step", 0)

try:
    for ep in range(state["epoch"], NUM_EPOCHS):
        for li, lang in enumerate(LANGUAGES):
            if not check_runtime(start_time):
                break
            done_key = f"{ep}_{lang}"
            if state["lang_done"].get(done_key, False):
                log(f"  Skipping {lang} epoch {ep+1} (done)")
                continue
            if len(train_sets[lang]) == 0:
                log(f"  Skipping {lang} (no data)")
                state["lang_done"][done_key] = True
                save_state(state)
                continue
            state_key = f"{ep}_{lang}"
            if state.get("current_key") != state_key:
                state["step"] = 0
                state["current_key"] = state_key
                state["lang"] = lang
                state["epoch"] = ep

            # ---- TRAINING ----
            log(f"  Setting LoRA to {lang}")
            model.set_language(lang)
            loader = make_loader(train_sets[lang], lang, batch_size=BATCH_SIZE)
            if not loader:
                state["lang_done"][done_key] = True
                save_state(state)
                continue
            opt = make_optimizer(lang)
            model.train()
            total, steps = 0.0, 0
            skip_to = state["step"]
            epoch_start = time.time()
            log(f"\nEpoch {ep+1}/{NUM_EPOCHS} | Training {lang.upper()} from step {skip_to} ({len(train_sets[lang])} samples)")

            for batch in loader:
                if not check_runtime(start_time):
                    save_state(state)
                    break
                if steps < skip_to:
                    steps += 1
                    total_steps_done += 1
                    continue
                try:
                    feats = batch["feats"].to(DEVICE)
                    lbls = batch["lbls"].to(DEVICE)
                    sched_lr = get_lr(total_steps_done)
                    for pg in opt.param_groups:
                        pg["lr"] = sched_lr
                    ss_prob = get_ss_prob(ep)

                    if ss_prob > 0 and not USE_FP16:
                        # Scheduled sampling on MPS (fp32)
                        enc = model.whisper.model.encoder(feats).last_hidden_state
                        with torch.no_grad():
                            tf_logits = model.whisper.model.decoder(lbls[:, :-1], encoder_hidden_states=enc).last_hidden_state
                            tf_logits = model.whisper.proj_out(tf_logits)
                            pred_tokens = tf_logits.argmax(dim=-1)
                        mix_mask = (torch.rand_like(lbls[:, :-1].float()) < ss_prob)
                        mix_mask[:, :2] = False  # keep START and lang token
                        mixed_input = torch.where(mix_mask, pred_tokens, lbls[:, :-1])
                        logits = model.whisper.model.decoder(mixed_input, encoder_hidden_states=enc).last_hidden_state
                        logits = model.whisper.proj_out(logits)
                    elif ss_prob > 0 and USE_FP16:
                        with torch.amp.autocast("cuda"):
                            enc = model.whisper.model.encoder(feats).last_hidden_state
                            with torch.no_grad():
                                tf_logits = model.whisper.model.decoder(lbls[:, :-1], encoder_hidden_states=enc).last_hidden_state
                                tf_logits = model.whisper.proj_out(tf_logits)
                                pred_tokens = tf_logits.argmax(dim=-1)
                            mix_mask = (torch.rand_like(lbls[:, :-1].float()) < ss_prob)
                            mix_mask[:, :2] = False
                            mixed_input = torch.where(mix_mask, pred_tokens, lbls[:, :-1])
                            logits = model.whisper.model.decoder(mixed_input, encoder_hidden_states=enc).last_hidden_state
                            logits = model.whisper.proj_out(logits)
                    elif USE_FP16:
                        with torch.amp.autocast("cuda"):
                            logits = model(feats, lbls[:, :-1], lang)
                    else:
                        logits = model(feats, lbls[:, :-1], lang)

                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, VOCAB_SIZE), lbls[:, 1:].reshape(-1), ignore_index=-100
                    )
                    opt.zero_grad()
                    if USE_FP16:
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.lora_adapters[lang].parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.lora_adapters[lang].parameters(), 1.0)
                        opt.step()
                    total += loss.item()
                    steps += 1
                    total_steps_done += 1
                    state["step"] = steps
                    state["global_step"] = total_steps_done
                    if steps % SAVE_EVERY == 0:
                        avg = total / SAVE_EVERY
                        elapsed = time.time() - epoch_start
                        rate = steps / elapsed if elapsed > 0 else 0
                        remaining = (len(loader) - steps) / rate if rate > 0 else 0
                        lr_now = opt.param_groups[0]["lr"]
                        mem = torch.mps.current_allocated_memory() / 1e9 if DEVICE == "mps" else 0
                        log(f"  [{steps}/{len(loader)}] loss={avg:.4f} lr={lr_now:.2e} {rate:.1f} steps/s ETA={remaining/60:.0f}m mem={mem:.1f}GB")
                        total = 0.0
                        save_state(state)
                        update_heartbeat(f"v2 training {lang} ep{ep+1} step {steps}/{len(loader)} loss={avg:.4f}")
                except Exception as e:
                    tb = traceback.format_exc()
                    log(f"\n  CRASH during training: {e}")
                    log_crash(e, tb)
                    save_state(state)
                    raise

            state["step"] = 0
            save_state(state)
            del loader, opt
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

            if not check_runtime(start_time):
                break

            # ---- VALIDATION ----
            log(f"\n  Validating {lang.upper()}...")
            model.set_language(lang)
            val_loader = make_loader(test_sets[lang], lang, batch_size=BATCH_SIZE, shuffle=False)
            if not val_loader:
                state["lang_done"][done_key] = True
                save_state(state)
                continue
            model.eval()
            val_loss, val_n = 0.0, 0
            try:
                for i, batch in enumerate(val_loader):
                    feats = batch["feats"].to(DEVICE)
                    lbls = batch["lbls"].to(DEVICE)
                    logits = model(feats, lbls[:, :-1], lang)
                    val_loss += nn.functional.cross_entropy(
                        logits.reshape(-1, VOCAB_SIZE), lbls[:, 1:].reshape(-1), ignore_index=-100
                    ).item()
                    val_n += 1
                val_loss /= max(val_n, 1)
                log(f"  Val {lang}: loss={val_loss:.4f}")
                prev_best = state["best_loss"].get(lang, float("inf"))
                if val_loss < prev_best:
                    state["best_loss"][lang] = val_loss
                    save_best(state)
                    log(f"  New best!")
                else:
                    log(f"  No improvement (best: {prev_best:.4f})")
            except Exception as e:
                tb = traceback.format_exc()
                log(f"\n  CRASH during val: {e}")
                log_crash(e, tb)
                save_state(state)
                raise
            finally:
                del val_loader
                gc.collect()
                if DEVICE == "mps":
                    torch.mps.empty_cache()
            state["lang_done"][done_key] = True
            save_state(state)
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            done = sum(1 for v in state["lang_done"].values() if v)
            total_jobs = NUM_EPOCHS * len(LANGUAGES)
            log(f"  Progress: {done}/{total_jobs} ({done/total_jobs*100:.0f}%) | Elapsed: {hours:.1f}h | Global steps: {total_steps_done}")

except KeyboardInterrupt:
    log("\n  Interrupted! Saving...")
    save_state(state)
except Exception as e:
    tb = traceback.format_exc()
    log(f"\n  FATAL: {e}")
    log_crash(e, tb)
    save_state(state)
    raise

# ============ SUMMARY ============
elapsed = time.time() - start_time
log(f"\n{'='*60}")
log(f"TRAINING COMPLETE")
log(f"Total time: {elapsed/3600:.1f} hours")
log(f"Total steps: {total_steps_done}")
for lang in LANGUAGES:
    p = ADAPTER_DIR / f"{lang}_best.pt"
    if p.exists():
        size_kb = p.stat().st_size / 1024
        log(f"  {lang}: {p.name} ({size_kb:.0f}KB)")
log(f"Best losses: {state.get('best_loss', {})}")
log("="*60)
update_heartbeat("v2 training complete")
