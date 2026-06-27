"""
PolyWhisper Local Training — Mac Mini M4
Fully resumable. Run it, leave it, re-run to resume.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import math
import gc
import time
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

# ============ CONFIG ============
WHISPER_MODEL = "openai/whisper-base"
WHISPER_EXPECTED_LEN = 3000
LANGUAGES = ["en", "hi"]
NUM_EPOCHS = 5
BATCH_SIZE = 4
LR = 5e-4
MAX_LABEL_LEN = 256
AD_HID = 256
AD_LAYERS = 3
AD_HEADS = 4
AD_FFN = 1024
AD_RANK = 16
MAX_SAMPLES_PER_LANG = 50000
SAVE_EVERY = 20

# ============ PATHS ============
SAVE_DIR = Path("./polywhisper_output")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = SAVE_DIR / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)
DATA_DIR = SAVE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = SAVE_DIR / "training_state.json"

# ============ DEVICE ============
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

# ============ PROCESSOR ============
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
ENCODE_DIM = 512
VOCAB_SIZE = len(processor.tokenizer)
tok = processor.tokenizer

# ============ MODEL ============
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
        a = self.drop(torch.softmax((q @ k.transpose(-2, -1)) / self.sc, dim=-1))
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

print("Loading model...")
model = PolyWhisper().to(DEVICE)
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable/1e6:.1f}M")

# ============ DATA ============
import soundfile as sf

class CommonVoiceLocal(Dataset):
    def __init__(self, lang, split="train"):
        self.lang = lang
        self.split = split
        self.cache = DATA_DIR / f"cv_{lang}_{split}.json"
        self.adir = DATA_DIR / f"audio_{lang}_{split}"
        self.adir.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self):
        if self.cache.exists():
            print(f"  Cached {self.lang}/{self.split}")
            return json.load(open(self.cache))

        from datasets import load_dataset
        cfg_map = {"en": "en", "hi": "hi"}
        print(f"  Downloading Common Voice {self.lang}...")
        ds = None
        for repo in ["mozilla-foundation/common_voice_17_0", "fsicoli/common_voice_17_0"]:
            try:
                ds = load_dataset(
                    repo, cfg_map[self.lang],
                    split=self.split, streaming=True, trust_remote_code=True,
                )
                print(f"  Using {repo}")
                break
            except Exception as e:
                print(f"  {repo} failed: {e}")

        if ds is None:
            raise RuntimeError(f"Could not load Common Voice {self.lang}")

        recs = []
        for i, item in enumerate(tqdm(ds, desc=f"  {self.lang}")):
            if i >= MAX_SAMPLES_PER_LANG:
                break
            try:
                audio = item["audio"]["array"]
                sr = item["audio"]["sampling_rate"]
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(np.array(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
                audio = np.array(audio, dtype=np.float32)
                if len(audio) < 1600 or len(audio) > int(30.0 * 16000):
                    continue
                text = item["sentence"].strip()
                if not text or len(text) < 3:
                    continue
                wav_path = self.adir / f"{i:06d}.wav"
                sf.write(str(wav_path), audio, 16000)
                recs.append({"wav": str(wav_path), "text": text.lower() if self.lang == "en" else text})
            except Exception:
                continue

        json.dump(recs, open(self.cache, "w"))
        print(f"  Saved {len(recs)} samples")
        return recs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        a, _ = sf.read(r["wav"])
        return {"audio": a, "text": r["text"]}


class Collator:
    def __init__(self, proc, max_lbl):
        self.proc = proc
        self.max_lbl = max_lbl
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
            toks = self.proc.tokenizer(
                item["text"], padding="max_length", max_length=self.max_lbl, truncation=True
            )
            lbls.append(toks["input_ids"])
        lbls = torch.tensor(lbls, dtype=torch.long)
        lbls[lbls == tok.pad_token_id] = -100
        return {"feats": feats, "lbls": lbls}


def make_loader(dataset, batch_size=4, shuffle=True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=Collator(processor, MAX_LABEL_LEN),
        num_workers=0, pin_memory=(DEVICE == "cuda"),
    )

# ============ RESUME STATE ============
def new_state():
    return {
        "epoch": 0, "phase": "train", "lang": LANGUAGES[0],
        "step": 0, "opt": None, "best_loss": {}, "lang_done": {},
    }

def load_state():
    if STATE_FILE.exists():
        st = json.load(open(STATE_FILE))
        if st.get("opt"):
            opt.load_state_dict(st["opt"])
        lang = st["lang"]
        ep = st["epoch"]
        ckpt = ADAPTER_DIR / f"{lang}_ep{ep}_last.pt"
        if ckpt.exists():
            model.load_ckpt(str(ckpt))
            print(f"  Loaded: {ckpt.name}")
        print(f"  Resumed: {st['phase']} {st['lang']} epoch {st['epoch']} step {st['step']}")
        return st
    return new_state()

def save_state(state):
    state["opt"] = opt.state_dict()
    lang = state["lang"]
    ep = state["epoch"]
    ckpt = ADAPTER_DIR / f"{lang}_ep{ep}_last.pt"
    torch.save({n: a.state_dict() for n, a in model.adapters.items()}, str(ckpt))
    json.dump(state, open(STATE_FILE, "w"), indent=2)

def save_best(state):
    lang = state["lang"]
    p = ADAPTER_DIR / f"{lang}_best.pt"
    torch.save({lang: model.adapters[lang].state_dict()}, str(p))

# ============ TRAINING ============
print("Loading datasets...")
train_sets = {}
test_sets = {}
for lang in LANGUAGES:
    print(f"Loading {lang}...")
    train_sets[lang] = CommonVoiceLocal(lang, "train")
    test_sets[lang] = CommonVoiceLocal(lang, "test")
    print(f"  {lang}: {len(train_sets[lang])} train, {len(test_sets[lang])} test")

state = load_state()
start_time = time.time()

try:
    for ep in range(state["epoch"], NUM_EPOCHS):
        for li, lang in enumerate(LANGUAGES):
            done_key = f"{ep}_{lang}"
            if state["lang_done"].get(done_key, False):
                print(f"  Skipping {lang} epoch {ep+1} (done)")
                continue

            state_key = f"{ep}_{lang}"
            if state.get("current_key") != state_key:
                state["step"] = 0
                state["current_key"] = state_key
                state["phase"] = "train"

            # ---- TRAINING ----
            if state["phase"] == "train":
                loader = make_loader(train_sets[lang], batch_size=BATCH_SIZE)
                model.train()
                total, steps = 0.0, 0
                skip_to = state["step"]
                epoch_start = time.time()
                print(f"\nEpoch {ep+1}/{NUM_EPOCHS} | Training {lang.upper()} from step {skip_to}")

                for batch in loader:
                    if steps < skip_to:
                        steps += 1
                        continue
                    try:
                        feats = batch["feats"].to(DEVICE)
                        lbls = batch["lbls"].to(DEVICE)
                        logits = model(feats, lbls[:, :-1], lang)
                        loss = nn.functional.cross_entropy(
                            logits.reshape(-1, VOCAB_SIZE), lbls[:, 1:].reshape(-1), ignore_index=-100
                        )
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        total += loss.item()
                        steps += 1
                        state["step"] = steps

                        if steps % SAVE_EVERY == 0:
                            avg = total / SAVE_EVERY
                            elapsed = time.time() - epoch_start
                            rate = steps / elapsed if elapsed > 0 else 0
                            print(f"  [{steps}/{len(loader)}] loss={avg:.4f} ({rate:.1f} steps/sec)")
                            total = 0.0
                            save_state(state)
                    except Exception as e:
                        print(f"\n  CRASH: {e}")
                        save_state(state)
                        raise

                state["phase"] = "val"
                state["step"] = 0
                save_state(state)
                del loader
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            # ---- VALIDATION ----
            if state["phase"] == "val":
                print(f"\n  Validating {lang.upper()}...")
                val_loader = make_loader(test_sets[lang], batch_size=BATCH_SIZE, shuffle=False)
                model.eval()
                val_loss, val_n = 0.0, 0
                skip_to = state["step"]

                try:
                    for i, batch in enumerate(val_loader):
                        if i < skip_to:
                            continue
                        feats = batch["feats"].to(DEVICE)
                        lbls = batch["lbls"].to(DEVICE)
                        logits = model(feats, lbls[:, :-1], lang)
                        val_loss += nn.functional.cross_entropy(
                            logits.reshape(-1, VOCAB_SIZE), lbls[:, 1:].reshape(-1), ignore_index=-100
                        ).item()
                        val_n += 1
                        state["step"] = i + 1
                        if (i + 1) % 50 == 0:
                            save_state(state)

                    val_loss /= max(val_n, 1)
                    print(f"  Val {lang}: loss={val_loss:.4f}")
                    prev_best = state["best_loss"].get(lang, float("inf"))
                    if val_loss < prev_best:
                        state["best_loss"][lang] = val_loss
                        save_best(state)
                        print(f"  New best!")

                    del val_loader
                    gc.collect()
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n  CRASH during val: {e}")
                    save_state(state)
                    raise

                state["lang_done"][done_key] = True
                state["phase"] = "train"
                state["step"] = 0
                save_state(state)

                # Progress
                elapsed = time.time() - start_time
                hours = elapsed / 3600
                done = sum(1 for v in state["lang_done"].values() if v)
                total_jobs = NUM_EPOCHS * len(LANGUAGES)
                print(f"  Progress: {done}/{total_jobs} ({done/total_jobs*100:.0f}%) | Elapsed: {hours:.1f}h")

except KeyboardInterrupt:
    print("\n  Interrupted! Saving...")
    save_state(state)
except Exception as e:
    print(f"\n  FATAL: {e}")
    save_state(state)
    raise

elapsed = time.time() - start_time
print(f"\n{'='*50}")
print(f"TRAINING COMPLETE")
print(f"Total time: {elapsed/3600:.1f} hours")
for lang in LANGUAGES:
    p = ADAPTER_DIR / f"{lang}_best.pt"
    if p.exists():
        size_mb = p.stat().st_size / (1024*1024)
        print(f"  {lang}: {p.name} ({size_mb:.1f}MB)")
print(f"Best losses: {state.get('best_loss', {})}")
