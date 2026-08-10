import torch, torch.nn as nn, numpy as np, json, math, time
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WHISPER_MODEL = "openai/whisper-base"
UNFREEZE_LAYERS = [4, 5]
START_TOKEN = 50258
AD_HID = 256; AD_LAYERS = 3; AD_HEADS = 4; AD_FFN = 1024; AD_RANK = 16
MAX_LABEL_LEN = 256; ENCODE_DIM = 512
MAX_NEW_TOKENS = 128

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
VOCAB_SIZE = len(processor.tokenizer)
tok = processor.tokenizer
WHISPER_EXPECTED_LEN = 3000

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
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.qa = LowRank(d, d, r); self.va = LowRank(d, d, r)
        self.h, self.dh = h, d // h; self.sc = math.sqrt(self.dh); self.drop = nn.Dropout(0.1)
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x) + self.qa(x); k, v = self.k(x), self.v(x) + self.va(x)
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
        self.q = nn.Linear(d, d); self.k = nn.Linear(ed, d); self.v = nn.Linear(ed, d); self.o = nn.Linear(d, d)
        self.qa = LowRank(d, d, r); self.va = LowRank(ed, d, r)
        self.h, self.dh = h, d // h; self.sc = math.sqrt(self.dh); self.drop = nn.Dropout(0.1)
    def forward(self, x, enc):
        B, Td, _ = x.shape; Te = enc.size(1)
        q = self.q(x) + self.qa(x); k, v = self.k(enc), self.v(enc) + self.va(enc)
        def rs(t, T): return t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = rs(q, Td), rs(k, Te), rs(v, Te)
        a = self.drop(torch.softmax((q @ k.transpose(-2, -1)) / self.sc, dim=-1))
        return self.o((a @ v).transpose(1, 2).contiguous().view(B, Td, -1))

class AdaptLayer(nn.Module):
    def __init__(self, d, ed, h, ffn, r):
        super().__init__()
        self.sa = SelfAttn(d, h, r); self.ca = CrossAttn(d, ed, h, r)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(0.1), nn.Linear(ffn, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d); self.drop = nn.Dropout(0.1)
    def forward(self, x, enc):
        x = x + self.drop(self.sa(self.n1(x))); x = x + self.drop(self.ca(self.n2(x), enc)); x = x + self.drop(self.ff(self.n3(x)))
        return x

class LangAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(VOCAB_SIZE, AD_HID); self.pe = nn.Embedding(MAX_LABEL_LEN, AD_HID)
        self.layers = nn.ModuleList([AdaptLayer(AD_HID, ENCODE_DIM, AD_HEADS, AD_FFN, AD_RANK) for _ in range(AD_LAYERS)])
        self.out = nn.Linear(AD_HID, VOCAB_SIZE, bias=False); self.out.weight = self.te.weight
        self.norm = nn.LayerNorm(AD_HID); self.drop = nn.Dropout(0.1)
    def forward(self, enc, ids):
        ids = ids.clamp(min=0, max=VOCAB_SIZE - 1); B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.drop(self.te(ids) + self.pe(pos))
        for l in self.layers: x = l(x, enc)
        return self.out(self.norm(x))

class PolyWhisper(nn.Module):
    def __init__(self):
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
        for p in self.whisper.model.encoder.parameters(): p.requires_grad = False
        for p in self.whisper.model.decoder.parameters(): p.requires_grad = False
        for i in UNFREEZE_LAYERS:
            for p in self.whisper.model.encoder.layers[i].parameters(): p.requires_grad = True
        self.adapters = nn.ModuleDict({l: LangAdapter() for l in ["hinglish"]})
    def encode(self, feat):
        return self.whisper.model.encoder(feat).last_hidden_state
    def load_ckpt(self, path):
        st = torch.load(path, map_location=DEVICE, weights_only=True)
        if "hinglish" in st: self.adapters["hinglish"].load_state_dict(st["hinglish"])
        if "_encoder" in st:
            for k, v in st["_encoder"].items():
                self.whisper.model.encoder.layers[int(k)].load_state_dict(v)

print("Loading model...")
model = PolyWhisper().to(DEVICE)
model.load_ckpt("./polywhisper_output/adapters/hinglish_best.pt")
model.eval()

def process_audio(wav_path):
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = np.array(audio, dtype=np.float32)[:int(30.0 * 16000)]
    inputs = processor.feature_extractor([audio], sampling_rate=16000, return_tensors="pt", padding=True)
    feat = inputs["input_features"].to(DEVICE)
    if feat.shape[-1] < WHISPER_EXPECTED_LEN:
        pad = torch.zeros(feat.shape[0], feat.shape[1], WHISPER_EXPECTED_LEN - feat.shape[-1], dtype=feat.dtype, device=DEVICE)
        feat = torch.cat([feat, pad], dim=-1)
    else:
        feat = feat[:, :, :WHISPER_EXPECTED_LEN]
    return feat

@torch.no_grad()
def generate_beam(model, audio_feat, beam_width=5, max_len=MAX_NEW_TOKENS, rep_penalty=1.2):
    enc = model.encode(audio_feat)
    beams = [(torch.tensor([[START_TOKEN]], device=audio_feat.device), 0.0)]
    for step in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq.size(1) > 1 and seq[0, -1].item() == tok.eos_token_id:
                candidates.append((seq, score)); continue
            logits = model.adapters["hinglish"](enc, seq)[:, -1, :].clone()
            seen = set(seq[0].cpu().tolist())
            for t in seen:
                if logits[0, t] > 0: logits[0, t] /= rep_penalty
                else: logits[0, t] *= rep_penalty
            probs = torch.log_softmax(logits, dim=-1)
            top_k_probs, top_k_idx = probs.topk(beam_width, dim=-1)
            for k in range(beam_width):
                candidates.append((torch.cat([seq, top_k_idx[:, k:k+1]], dim=1), score + top_k_probs[0, k].item()))
        if not candidates: break
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(b[0, -1].item() == tok.eos_token_id for b, _ in beams): break
    return tok.decode(beams[0][0][0].tolist(), skip_special_tokens=True)

from jiwer import wer as jiwer_wer
import random
random.seed(42)

data = json.load(open("./polywhisper_output/data/mucs_hinglish_hinglish_test.json"))
data = random.sample(data, 30)

refs, hyps = [], []
t0 = time.time()
for i, item in enumerate(data):
    feat = process_audio(item["wav"])
    hyp = generate_beam(model, feat)
    refs.append(item["text"]); hyps.append(hyp)
    print(f"{i+1}/30 REF: {item['text'][:70]}")
    print(f"        HYP: {hyp[:70]}")

print(f"\nHINGLISH WER: {jiwer_wer(refs, hyps)*100:.1f}% ({len(refs)} samples, {time.time()-t0:.0f}s)")
