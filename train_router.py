"""
B2/B3: Token-level code-switch LoRA router for Hinglish.

Architecture: Whisper Base frozen. Two LoRA experts (en, hi) active simultaneously.
A small router MLP reads the decoder input embeddings (per token) and outputs
soft weights over {en, hi}; every LoRA projection delta is mixed per-token:
    out = base(x) + w_en * delta_en(x) + w_hi * delta_hi(x)

The router weights are computed by a forward hook on the decoder's embed_tokens,
so they apply identically during teacher-forced training AND beam-search generation.

Phase 1 (router only): adapters frozen, CE loss on per-token language labels
Phase 2 (joint): adapter + router fine-tuned, ASR loss + lambda * router loss

Labels come from build_codeswitch_labels.py (script-detected, word-level),
aligned to subword tokens via character-offset mapping (context-aware).
"""

import argparse
import re
import torch
import torch.nn as nn
import numpy as np
import json
import gc
import time
import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

import os
os.environ.setdefault("HF_TOKEN", "")

WHISPER_MODEL = "openai/whisper-base"
WHISPER_EXPECTED_LEN = 3000
BATCH_SIZE = 4
RANK = 16
MAX_LABEL_LEN = 256

START_TOKEN = 50257
LANG_TOKENS = {"en": 50259, "hi": 50276, "hinglish": 50276}
TASK_TOKEN = 50359
NO_TIME_TOKEN = 50363
BOS_EOS = 50257
SAVE_EVERY = 5

SS_START_PROB = 0.10

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lambda-router", type=float, default=1.0)
parser.add_argument("--hi-adapter", type=str, default="polywhisper_output/adapters_v2/hi_best.pt")
parser.add_argument("--en-adapter", type=str, default="polywhisper_output/adapters_v3/en_best.pt")
parser.add_argument("--router-init", type=str, default="")
parser.add_argument("--tag", type=str, default="",
                    help="Suffix for checkpoints/state, e.g. _v4 (isolates this run from stale state)")
ARGS = None

SAVE_DIR = Path("./polywhisper_output")
ADAPTER_DIR = SAVE_DIR / "adapters_v3"
DATA_DIR = SAVE_DIR / "data"
LOG_FILE = SAVE_DIR / "training_router.log"
HEARTBEAT_FILE = SAVE_DIR / "heartbeat_router.txt"
CRASH_LOG = SAVE_DIR / "crash_router.log"
ROUTER_STATE_FILE = SAVE_DIR / "training_state_router.json"


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


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
log(f"Device: {DEVICE}")

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
VOCAB_SIZE = len(processor.tokenizer)
tok = processor.tokenizer
log(f"Vocab: {VOCAB_SIZE}")

PROJ_MAP = [
    ('self_attn', 'self_q'), ('self_attn', 'self_k'), ('self_attn', 'self_v'),
    ('encoder_attn', 'cross_q'), ('encoder_attn', 'cross_k'), ('encoder_attn', 'cross_v'),
    ('self_attn', 'self_out'), ('encoder_attn', 'cross_out'),
]


class PolyWhisperRouter(nn.Module):
    def __init__(self, whisper_name=WHISPER_MODEL, rank=RANK):
        super().__init__()
        self.rank = rank
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        for p in self.whisper.parameters():
            p.requires_grad = False
        self.d_model = self.whisper.config.d_model
        self.decoder_layers = self.whisper.config.decoder_layers
        self.lora_adapters = nn.ModuleDict()
        self._router_weights = None

        self.router = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        self.router.apply(self._init_router)

        self._embed_hook = self.whisper.model.decoder.embed_tokens.register_forward_hook(self._embed_hook_fn)

    @staticmethod
    def _init_router(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def _embed_hook_fn(self, mod, inp, out):
        w = torch.softmax(self.router(out), dim=-1)
        self._router_weights = w

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

    @staticmethod
    def _proj_name(base_name):
        if 'out' in base_name:
            return 'out_proj'
        if 'q' in base_name:
            return 'q_proj'
        if 'k' in base_name:
            return 'k_proj'
        return 'v_proj'

    def _install_expert_hooks(self):
        """Hooks for BOTH en and hi experts with per-token router mixing.

        v4 fix: cross k/v projections (encoder-state projections) are NOT hooked.
        They were previously mixed once with prefix-mean router weights and cached
        by the KV cache, diverging from the per-token mixing of later steps and
        collapsing generation. All decoder-state projections (self q/k/v/out,
        cross q/out) keep per-token mixing, which is KV-cache-safe.
        """
        for i, layer in enumerate(self.whisper.model.decoder.layers):
            for attn_name, base_name in PROJ_MAP:
                if base_name in ("cross_k", "cross_v"):
                    continue
                a_en = self.lora_adapters['en'][f'{i}_{base_name}_a']
                b_en = self.lora_adapters['en'][f'{i}_{base_name}_b']
                a_hi = self.lora_adapters['hi'][f'{i}_{base_name}_a']
                b_hi = self.lora_adapters['hi'][f'{i}_{base_name}_b']

                def make_hook(a_en, b_en, a_hi, b_hi):
                    def hook(mod, inp, out):
                        x = inp[0]
                        w = self._router_weights
                        d_en = b_en(a_en(x))
                        d_hi = b_hi(a_hi(x))
                        if w is not None:
                            w_en = w[:, :, 0:1]
                            w_hi = w[:, :, 1:2]
                            if w_en.shape[0] != x.shape[0]:
                                reps = (x.shape[0] + w_en.shape[0] - 1) // w_en.shape[0]
                                w_en = w_en.repeat(reps, 1, 1)[:x.shape[0]]
                                w_hi = w_hi.repeat(reps, 1, 1)[:x.shape[0]]
                            delta = w_en * d_en + w_hi * d_hi
                        else:
                            delta = d_hi
                        return out + delta
                    return hook

                proj = getattr(getattr(layer, attn_name), self._proj_name(base_name))
                proj.register_forward_hook(make_hook(a_en, b_en, a_hi, b_hi))

    def forward(self, feat, dec_ids):
        enc = self.whisper.model.encoder(feat).last_hidden_state
        self._router_weights = None
        dec_out = self.whisper.model.decoder(dec_ids, encoder_hidden_states=enc, output_hidden_states=False)
        logits = self.whisper.proj_out(dec_out.last_hidden_state)
        w = self._router_weights
        return logits, w

    @torch.no_grad()
    def generate(self, feat, **gen_kwargs):
        return self.whisper.generate(feat, **gen_kwargs)

    def load_adapter(self, lang, path):
        if lang not in self.lora_adapters:
            self.add_language(lang)
        st = torch.load(path, map_location="cpu", weights_only=True)
        # v3 checkpoints use d{i}_ / e{i}_ prefixes (train_v3); router uses {i}_ (v2 naming).
        # Strip encoder keys (router has no encoder LoRA) and normalize d{i}_ -> {i}_.
        st = {re.sub(r"^d(\d+)_", r"\1_", k): v for k, v in st.items() if not k.startswith("e")}
        self.lora_adapters[lang].load_state_dict(st)

    def save_router(self, path):
        torch.save(self.router.state_dict(), path)

    def load_router(self, path):
        st = torch.load(path, map_location="cpu", weights_only=True)
        self.router.load_state_dict(st)

    def save_all(self, tag):
        self.save_router(str(ADAPTER_DIR / f"router_{tag}.pt"))
        for lang in self.lora_adapters:
            torch.save(self.lora_adapters[lang].state_dict(),
                       str(ADAPTER_DIR / f"{lang}_router_{tag}.pt"))

    def load_all(self, tag):
        rp = ADAPTER_DIR / f"router_{tag}.pt"
        if rp.exists():
            self.load_router(str(rp))
            for lang in self.lora_adapters:
                lp = ADAPTER_DIR / f"{lang}_router_{tag}.pt"
                if lp.exists():
                    self.lora_adapters[lang].load_state_dict(
                        torch.load(str(lp), map_location="cpu", weights_only=True))
            return True
        return False



SPECIALS = {50257, 50258, 50359, 50363}  # eos, bos, task, no-timestamps

tok_fast = None


def _get_fast_tokenizer():
    global tok_fast
    if tok_fast is None:
        from transformers import WhisperTokenizerFast
        tok_fast = WhisperTokenizerFast.from_pretrained(WHISPER_MODEL)
    return tok_fast


def align_labels(text, words):
    """Subword-level language labels via char offsets (context-aware tokenization)."""
    enc = _get_fast_tokenizer()(text, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    spans = []
    pos = 0
    for w in words:
        start = text.find(w["t"], pos)
        if start < 0:
            return None
        spans.append((start, start + len(w["t"])))
        pos = start + len(w["t"])
    sub, labels = [], []
    wi = 0
    for tid, o in zip(ids, offs):
        s, e = o
        if tid in SPECIALS:
            continue
        while wi < len(spans) - 1 and spans[wi][1] <= s:
            wi += 1
        sub.append(tid)
        labels.append(1 if words[wi]["lang"] == "en" else 0)
    return sub, labels



if __name__ == "__main__":
    ARGS = parser.parse_args()
    TAG = ARGS.tag
    ROUTER_STATE_FILE = SAVE_DIR / f"training_state_router{TAG}.json"
    log("Building model...")
    model = PolyWhisperRouter().to(DEVICE)
    model.add_language("en")
    model.add_language("hi")
    model._install_expert_hooks()
    if Path(ARGS.en_adapter).exists():
        model.load_adapter("en", ARGS.en_adapter)
        log(f"  Loaded en expert: {ARGS.en_adapter}")
    else:
        log(f"  WARNING: en adapter not found: {ARGS.en_adapter}")
    if Path(ARGS.hi_adapter).exists():
        model.load_adapter("hi", ARGS.hi_adapter)
        log(f"  Loaded hi expert: {ARGS.hi_adapter}")
    else:
        log(f"  WARNING: hi adapter not found: {ARGS.hi_adapter}")

    router_params = sum(p.numel() for p in model.router.parameters())
    log(f"Router params: {router_params/1e3:.0f}K")

    # ============ DATA ============
    import soundfile as sf

    CS_DATA = DATA_DIR / "hinglish_codeswitch_train.json"
    CS_TEST = DATA_DIR / "hinglish_codeswitch_test.json"

    FORCED_PREFIX = [START_TOKEN, LANG_TOKENS["hinglish"], TASK_TOKEN, NO_TIME_TOKEN]


    class CodeSwitchDataset(Dataset):
        def __init__(self, path):
            self.records = json.load(open(path))

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            r = self.records[i]
            a, _ = sf.read(r["wav"])
            return {"audio": a, "text": r["text"], "words": r["tokens"]}




    class RouterCollator:
        def __init__(self, max_lbl):
            self.max_lbl = max_lbl
            self.forced_prefix = FORCED_PREFIX

        def __call__(self, batch):
            auds = [item["audio"] for item in batch]
            ft = processor.feature_extractor(auds, sampling_rate=16000, return_tensors="pt", padding=True)
            feats = ft["input_features"]
            B, C, T = feats.shape
            if T < WHISPER_EXPECTED_LEN:
                pad = torch.zeros(B, C, WHISPER_EXPECTED_LEN - T, dtype=feats.dtype)
                feats = torch.cat([feats, pad], dim=-1)
            else:
                feats = feats[:, :, :WHISPER_EXPECTED_LEN]

            lbls = []
            lang_labels = []
            for item in batch:
                aligned = align_labels(item["text"], item["words"])
                if aligned is None:
                    toks = tok(item["text"])["input_ids"]
                    toks = [t for t in toks if t not in SPECIALS]
                    sub, labels = toks, None
                else:
                    sub, labels = aligned
                full = self.forced_prefix + sub + [BOS_EOS]
                if len(full) > self.max_lbl:
                    full = full[:self.max_lbl]
                    if labels is not None:
                        labels = labels[:self.max_lbl - len(self.forced_prefix) - 1]
                while len(full) < self.max_lbl:
                    full.append(-100)
                lbls.append(full)
                if labels is not None:
                    ll = [1] * len(self.forced_prefix)  # prefix tokens: hi (label 1 = hi)
                    ll += [1 - lab for lab in labels]  # align_labels: 1=en,0=hi -> router: 1=hi,0=en
                    ll = ll[:self.max_lbl]
                    while len(ll) < self.max_lbl:
                        ll.append(-100)
                    lang_labels.append(ll)
                else:
                    lang_labels.append([-100] * self.max_lbl)
            return {
                "feats": feats,
                "lbls": torch.tensor(lbls, dtype=torch.long),
                "lang_labels": torch.tensor(lang_labels, dtype=torch.long),
            }


    def make_loader(ds, shuffle=True, batch_size=BATCH_SIZE):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=RouterCollator(MAX_LABEL_LEN), num_workers=0)


    def get_ss_prob(ep):
        if ARGS.phase == 1 or ep >= 2:
            return 0.0
        return SS_START_PROB * (1 - ep / 2)


    log("Loading datasets...")
    train_ds = CodeSwitchDataset(CS_DATA)
    test_ds = CodeSwitchDataset(CS_TEST)
    log(f"  train: {len(train_ds)}, test: {len(test_ds)}")

    # ============ TRAINING ============
    log("=" * 60)
    log(f"Router training — phase {ARGS.phase}, epochs {ARGS.epochs}, lr {ARGS.lr}")
    log("=" * 60)

    for p in model.lora_adapters.parameters():
        p.requires_grad = (ARGS.phase == 2)
    for p in model.router.parameters():
        p.requires_grad = True

    trainable = list(model.router.parameters())
    if ARGS.phase == 2:
        trainable += [p for p in model.lora_adapters.parameters() if p.requires_grad]

    # ---- resume: same phase + checkpoint files present -> continue where we left off ----
    start_epoch, start_step = 0, 0
    if ROUTER_STATE_FILE.exists():
        st = json.load(open(ROUTER_STATE_FILE))
        if st.get("phase") == ARGS.phase and model.load_all(f"last{TAG}"):
            start_epoch = st.get("epoch", 0)
            start_step = st.get("step", 0)
            log(f"  RESUMED: phase {ARGS.phase}, epoch {start_epoch+1}, step {start_step} "
                f"(from router_last.pt / {{lang}}_last.pt)")
    if start_epoch == 0 and start_step == 0 and ARGS.router_init:
        rp = Path(ARGS.router_init)
        if rp.exists():
            model.load_router(str(rp))
            log(f"  Initialized router from: {rp}")

    opt = torch.optim.AdamW(trainable, lr=ARGS.lr, weight_decay=0.01)

    train_loader = make_loader(train_ds, shuffle=False)
    val_loader = make_loader(test_ds, shuffle=False)

    try:
        for ep in range(start_epoch, ARGS.epochs):
            model.train()
            total, steps = 0.0, 0
            n_router_acc, n_router_tok = 0, 0
            epoch_start = time.time()
            ss_prob = get_ss_prob(ep)
            log(f"\nEpoch {ep+1}/{ARGS.epochs} | SS={ss_prob:.2f}")
            for bi, batch in enumerate(train_loader):
                if ep == start_epoch and bi < start_step:
                    continue
                feats = batch["feats"].to(DEVICE)
                lbls = batch["lbls"].to(DEVICE)
                ll = batch["lang_labels"].to(DEVICE)

                if ss_prob > 0:
                    with torch.no_grad():
                        tf_logits, _ = model(feats, lbls[:, :-1])
                        pred_tokens = tf_logits.argmax(dim=-1)
                    mix_mask = (torch.rand_like(lbls[:, :-1].float()) < ss_prob)
                    mix_mask[:, :2] = False
                    mixed_input = torch.where(mix_mask, pred_tokens, lbls[:, :-1])
                    logits, w = model(feats, mixed_input)
                else:
                    logits, w = model(feats, lbls[:, :-1])

                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), lbls[:, 1:].reshape(-1), ignore_index=-100)

                router_loss = nn.functional.cross_entropy(
                    w.reshape(-1, 2), ll[:, :-1].reshape(-1), ignore_index=-100)
                total_loss = loss + ARGS.lambda_router * router_loss

                opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()

                valid = ll[:, :-1] != -100
                if valid.any():
                    pred = w.argmax(-1)
                    n_router_acc += (pred[valid] == ll[:, :-1][valid]).sum().item()
                    n_router_tok += valid.sum().item()

                total += total_loss.item()
                steps += 1
                if steps % SAVE_EVERY == 0:
                    avg = total / SAVE_EVERY
                    elapsed = time.time() - epoch_start
                    rate = steps / elapsed
                    remaining = (len(train_loader) - steps) / rate
                    mem = torch.mps.current_allocated_memory() / 1e9 if DEVICE == "mps" else 0
                    racc = n_router_acc / max(1, n_router_tok) * 100
                    log(f"  [{steps}/{len(train_loader)}] loss={avg:.4f} r_acc={racc:.1f}% {rate:.1f} steps/s ETA={remaining/60:.0f}m mem={mem:.1f}GB")
                    total = 0.0
                    model.save_all(f"last{TAG}")
                    json.dump({"phase": ARGS.phase, "epoch": ep, "step": steps},
                              open(ROUTER_STATE_FILE, "w"))
                    HEARTBEAT_FILE.write_text(f"{datetime.now()} | router phase{ARGS.phase} ep{ep+1} step {steps}")

            model.eval()
            val_loss, val_n = 0.0, 0
            v_acc, v_tok = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    feats = batch["feats"].to(DEVICE)
                    lbls = batch["lbls"].to(DEVICE)
                    ll = batch["lang_labels"].to(DEVICE)
                    logits, w = model(feats, lbls[:, :-1])
                    vl = nn.functional.cross_entropy(
                        logits.reshape(-1, VOCAB_SIZE), lbls[:, 1:].reshape(-1), ignore_index=-100)
                    rl = nn.functional.cross_entropy(w.reshape(-1, 2), ll[:, :-1].reshape(-1), ignore_index=-100)
                    valid = ll[:, :-1] != -100
                    if valid.any():
                        pred = w.argmax(-1)
                        v_acc += (pred[valid] == ll[:, :-1][valid]).sum().item()
                        v_tok += valid.sum().item()
                    val_loss += (vl + ARGS.lambda_router * rl).item()
                    val_n += 1
            val_loss /= max(val_n, 1)
            v_acc_pct = v_acc / max(1, v_tok) * 100
            log(f"  Val: loss={val_loss:.4f} router_acc={v_acc_pct:.1f}%")
            model.save_all(f"best{TAG}")
            json.dump({"phase": ARGS.phase, "epoch": ep + 1, "step": 0},
                      open(ROUTER_STATE_FILE, "w"))
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

    except KeyboardInterrupt:
        log("Interrupted, saving router...")
        model.save_all(f"last{TAG}")
        json.dump({"phase": ARGS.phase, "epoch": ep, "step": steps},
                  open(ROUTER_STATE_FILE, "w"))
    except Exception as e:
        tb = traceback.format_exc()
        log(f"FATAL: {e}")
        log_crash(e, tb)
        raise

    log("=" * 60)
    log("ROUTER TRAINING COMPLETE")
    log("=" * 60)

