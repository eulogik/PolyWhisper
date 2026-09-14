"""PolyWhisper model — frozen Whisper backbone + per-language LoRA adapters."""

import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import io
import torch
import torch.nn as nn
import logging
import warnings
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

WHISPER_SMALL = "openai/whisper-small"
RANK = 16
LANG_TOKENS = {
    "en": 50259, "hi": 50276, "ta": 50287,
    "te": 50299, "bn": 50302, "mr": 50320,
}
BOS_EOS = 50257

ADAPTER_REGISTRY = {
    "hi": {"base": "hi_best_v5.pt", "prod": "hi_best_prod.pt"},
    "ta": {"base": "ta_best_ta.pt", "prod": "ta_best_prod.pt"},
    "te": {"base": "te_best_te.pt", "prod": "te_best_prod.pt"},
    "bn": {"base": "bn_best_bn.pt", "prod": "bn_best_prod.pt"},
    "mr": {"base": "mr_best_mr.pt", "prod": "mr_best_prod.pt"},
}

AVAILABLE_LANGS = list(ADAPTER_REGISTRY.keys())

# Optimal decoding per language (FLEURS beam-5 + rep_penalty=1.3 eval, Sep 2026).
# Beam-5 helps hi/ta/bn/mr by 2-5% relative; te degenerates under beam search
# (100.1 -> 120.5 WER, repeated-token loops), so te stays greedy.
OPTIMAL_BEAMS = {"hi": 5, "ta": 5, "te": 1, "bn": 5, "mr": 5}
DEFAULT_REPETITION_PENALTY = 1.3


def optimal_beams(lang):
    """Return the validated optimal beam width for a language (default 1)."""
    return OPTIMAL_BEAMS.get(lang, 1)


def _get_processor(backbone="small"):
    """Get WhisperProcessor for the given backbone."""
    from transformers import WhisperProcessor
    return WhisperProcessor.from_pretrained(f"openai/whisper-{backbone}")


def _adapter_search_paths():
    """Return candidate directories containing adapters_v3/."""
    candidates = [
        Path(__file__).parent.parent / "polywhisper_output" / "adapters_v3",
        Path.home() / "polywhisper_output" / "adapters_v3",
        Path("/Volumes/KIOXIA 1TB/polywhisper_output/adapters_v3"),
    ]
    return [p for p in candidates if p.exists()]


class PolyWhisper(nn.Module):
    """Whisper backbone with frozen weights and per-language LoRA adapters.

    Usage:
        model = PolyWhisper(backbone="small", device="auto")
        model.load_adapter("hi", "path/to/hi_best_prod.pt")
        model.set_language("hi")
        tokens = model.generate(features, max_new_tokens=256)
    """

    PROJ_MAP = [
        ("self_attn", "self_q"), ("self_attn", "self_k"), ("self_attn", "self_v"),
        ("encoder_attn", "cross_q"), ("encoder_attn", "cross_k"), ("encoder_attn", "cross_v"),
        ("self_attn", "self_out"), ("encoder_attn", "cross_out"),
    ]
    ENC_PROJ_MAP = [
        ("self_attn", "enc_q"), ("self_attn", "enc_k"), ("self_attn", "enc_v"),
        ("self_attn", "enc_out"),
    ]

    def __init__(self, backbone="small", rank=RANK, device="auto"):
        super().__init__()
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.rank = rank
        self._current_lang = None
        self._hooks = []

        whisper_name = f"openai/whisper-{backbone}"
        import warnings as _warn
        _warn.filterwarnings("ignore")
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        for p in self.whisper.parameters():
            p.requires_grad = False
        self.whisper.to(self.device)

        self.d_model = self.whisper.config.d_model
        self.decoder_layers = self.whisper.config.decoder_layers
        self.encoder_layers = self.whisper.config.encoder_layers
        self.lora_adapters = nn.ModuleDict()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            self.processor = WhisperProcessor.from_pretrained(whisper_name)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def add_language(self, lang):
        d, r = self.d_model, self.rank
        dev = self.device
        adpt = nn.ModuleDict()
        for i in range(self.decoder_layers):
            for _, base_name in self.PROJ_MAP:
                adpt[f"d{i}_{base_name}_a"] = nn.Linear(d, r, bias=False).to(dev)
                adpt[f"d{i}_{base_name}_b"] = nn.Linear(r, d, bias=False).to(dev)
                nn.init.zeros_(adpt[f"d{i}_{base_name}_b"].weight)
        if lang not in self.lora_adapters or not any(
            k.startswith("e0_") for k in self.lora_adapters.get(lang, {}).keys()
        ):
            pass  # decoder-only by default; encoder LoRA loaded from checkpoint
        self.lora_adapters[lang] = adpt
        return adpt

    @staticmethod
    def _proj_name(base_name):
        if "out" in base_name:
            return "out_proj"
        if "q" in base_name:
            return "q_proj"
        if "k" in base_name:
            return "k_proj"
        return "v_proj"

    def set_language(self, lang):
        if self._current_lang == lang:
            return
        self._remove_hooks()
        if lang not in self.lora_adapters:
            self.add_language(lang)
        lora = self.lora_adapters[lang]
        self._hooks = []
        for i, layer in enumerate(self.whisper.model.decoder.layers):
            for attn_name, base_name in self.PROJ_MAP:
                a = lora[f"d{i}_{base_name}_a"]
                b = lora[f"d{i}_{base_name}_b"]

                def make_hook(a, b):
                    def hook(mod, inp, out):
                        return out + b(a(inp[0]))
                    return hook

                proj = getattr(getattr(layer, attn_name), self._proj_name(base_name))
                self._hooks.append(proj.register_forward_hook(make_hook(a, b)))
        # encoder LoRA hooks (only if adapter has encoder keys)
        if any(k.startswith("e0_") for k in lora.keys()):
            for i, layer in enumerate(self.whisper.model.encoder.layers):
                for attn_name, base_name in self.ENC_PROJ_MAP:
                    a = lora[f"e{i}_{base_name}_a"]
                    b = lora[f"e{i}_{base_name}_b"]
                    proj = getattr(getattr(layer, attn_name), self._proj_name(base_name))
                    self._hooks.append(proj.register_forward_hook(make_hook(a, b)))
        self._current_lang = lang

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def load_adapter(self, lang, path):
        path = str(path)
        st = torch.load(path, map_location="cpu", weights_only=True)
        d, r = self.d_model, self.rank
        dev = self.device
        if lang not in self.lora_adapters:
            self.lora_adapters[lang] = nn.ModuleDict()
        adpt = self.lora_adapters[lang]
        # create decoder layers if present in checkpoint
        dec_keys = [k for k in st if k.startswith("d0_")]
        if dec_keys:
            for i in range(self.decoder_layers):
                for _, base_name in self.PROJ_MAP:
                    a_key = f"d{i}_{base_name}_a"
                    b_key = f"d{i}_{base_name}_b"
                    if a_key not in adpt:
                        adpt[a_key] = nn.Linear(d, r, bias=False).to(dev)
                        adpt[b_key] = nn.Linear(r, d, bias=False).to(dev)
        # create encoder LoRA layers if present in checkpoint
        enc_keys = [k for k in st if k.startswith("e0_")]
        if enc_keys:
            for i in range(self.encoder_layers):
                for _, base_name in self.ENC_PROJ_MAP:
                    a_key = f"e{i}_{base_name}_a"
                    b_key = f"e{i}_{base_name}_b"
                    if a_key not in adpt:
                        adpt[a_key] = nn.Linear(d, r, bias=False).to(dev)
                        adpt[b_key] = nn.Linear(r, d, bias=False).to(dev)
        adpt.load_state_dict(st)

    def auto_load_adapter(self, lang, variant="prod"):
        """Load adapter by language name, searching standard paths."""
        reg = ADAPTER_REGISTRY.get(lang)
        if reg is None:
            raise ValueError(f"Unknown language: {lang}. Available: {AVAILABLE_LANGS}")
        if isinstance(reg, dict):
            fname = reg.get(variant, reg.get("prod", list(reg.values())[0]))
        else:
            fname = reg
        for search_dir in _adapter_search_paths():
            path = search_dir / fname
            if path.exists():
                self.load_adapter(lang, path)
                return str(path)
        raise FileNotFoundError(
            f"Adapter {fname} not found. Searched: {[str(p) for p in _adapter_search_paths()]}"
        )

    def forward(self, feat, dec_ids, lang):
        self.set_language(lang)
        if (dec_ids == -100).any():
            dec_ids = torch.where(dec_ids == -100, torch.tensor(BOS_EOS, device=dec_ids.device), dec_ids)
        enc = self.whisper.model.encoder(feat).last_hidden_state
        dec_out = self.whisper.model.decoder(dec_ids, encoder_hidden_states=enc)
        return self.whisper.proj_out(dec_out.last_hidden_state)

    @torch.no_grad()
    def generate(self, feat, lang, **kwargs):
        self.set_language(lang)
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        result = self.whisper.generate(feat, **kwargs)
        sys.stderr = old_stderr
        return result

    def save_adapter(self, lang, path):
        torch.save(self.lora_adapters[lang].state_dict(), path)
