import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from dataclasses import dataclass
from typing import Optional, Dict
import math


@dataclass
class AdapterConfig:
    encoder_hidden_size: int = 512
    adapter_hidden_size: int = 256
    adapter_layers: int = 2
    adapter_heads: int = 4
    adapter_ffn_dim: int = 1024
    vocab_size: int = 51865
    max_target_positions: int = 128
    dropout: float = 0.1
    rank: int = 8
    tie_embeddings: bool = True


class LowRankAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 8):
        super().__init__()
        self.lora_a = nn.Linear(input_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, output_dim, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.lora_b(self.lora_a(x))


class AdapterAttention(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.adapter_hidden_size, config.adapter_hidden_size)
        self.k_proj = nn.Linear(config.adapter_hidden_size, config.adapter_hidden_size)
        self.v_proj = nn.Linear(config.adapter_hidden_size, config.adapter_hidden_size)
        self.out_proj = nn.Linear(config.adapter_hidden_size, config.adapter_hidden_size)

        self.q_adapter = LowRankAdapter(config.adapter_hidden_size, config.adapter_hidden_size, config.rank)
        self.v_adapter = LowRankAdapter(config.adapter_hidden_size, config.adapter_hidden_size, config.rank)

        self.num_heads = config.adapter_heads
        self.head_dim = config.adapter_hidden_size // config.adapter_heads
        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x) + self.q_adapter(x)
        k = self.k_proj(x)
        v = self.v_proj(x) + self.v_adapter(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class AdapterCrossAttention(nn.Module):
    def __init__(self, config: AdapterConfig, encoder_dim: int):
        super().__init__()
        self.config = config
        self.encoder_dim = encoder_dim

        self.q_proj = nn.Linear(config.adapter_hidden_size, config.adapter_hidden_size)
        self.k_proj = nn.Linear(encoder_dim, config.adapter_hidden_size)
        self.v_proj = nn.Linear(encoder_dim, config.adapter_hidden_size)
        self.out_proj = nn.Linear(config.adapter_hidden_size, config.adapter_hidden_size)

        self.q_adapter = LowRankAdapter(config.adapter_hidden_size, config.adapter_hidden_size, config.rank)
        self.v_adapter = LowRankAdapter(encoder_dim, config.adapter_hidden_size, config.rank)

        self.num_heads = config.adapter_heads
        self.head_dim = config.adapter_hidden_size // self.num_heads
        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output, encoder_mask=None):
        B, T_dec, _ = x.shape
        _, T_enc, _ = encoder_output.shape

        q = self.q_proj(x) + self.q_adapter(x)
        k = self.k_proj(encoder_output)
        v = self.v_proj(encoder_output) + self.v_adapter(encoder_output)

        q = q.view(B, T_dec, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_enc, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_enc, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if encoder_mask is not None:
            attn = attn.masked_fill(~encoder_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_dec, -1)
        return self.out_proj(out)


class AdapterFFN(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.adapter_hidden_size, config.adapter_ffn_dim)
        self.fc2 = nn.Linear(config.adapter_ffn_dim, config.adapter_hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class AdapterLayer(nn.Module):
    def __init__(self, config: AdapterConfig, encoder_dim: int):
        super().__init__()
        self.self_attn = AdapterAttention(config)
        self.cross_attn = AdapterCrossAttention(config, encoder_dim)
        self.ffn = AdapterFFN(config)
        self.norm1 = nn.LayerNorm(config.adapter_hidden_size)
        self.norm2 = nn.LayerNorm(config.adapter_hidden_size)
        self.norm3 = nn.LayerNorm(config.adapter_hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output, self_mask=None, encoder_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), self_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), encoder_output, encoder_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class LanguageAdapter(nn.Module):
    def __init__(self, config: AdapterConfig, encoder_dim: int):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.adapter_hidden_size)
        self.position_embedding = nn.Embedding(config.max_target_positions, config.adapter_hidden_size)
        self.layers = nn.ModuleList([
            AdapterLayer(config, encoder_dim) for _ in range(config.adapter_layers)
        ])
        self.output_proj = nn.Linear(config.adapter_hidden_size, config.vocab_size, bias=False)
        self.norm = nn.LayerNorm(config.adapter_hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        if config.tie_embeddings:
            self.output_proj.weight = self.token_embedding.weight

    def forward(self, encoder_output, decoder_input_ids, encoder_mask=None, decoder_mask=None):
        x = self.token_embedding(decoder_input_ids)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, decoder_mask, encoder_mask)

        x = self.norm(x)
        return self.output_proj(x)


class LanguageIdentifier(nn.Module):
    def __init__(self, encoder_dim: int, num_languages: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 4),
            nn.GELU(),
            nn.Linear(encoder_dim // 4, num_languages)
        )

    def forward(self, encoder_output):
        x = encoder_output.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class PolyWhisper(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        languages: list[str] = ["en", "hi"],
        adapter_config: Optional[AdapterConfig] = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.languages = languages
        self.config = adapter_config or AdapterConfig()

        self.whisper = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.whisper.model.encoder.parameters():
                param.requires_grad = False

        encoder_dim = self.whisper.config.d_model

        self.language_id_head = LanguageIdentifier(encoder_dim, len(languages))
        self.adapters = nn.ModuleDict({
            lang: LanguageAdapter(self.config, encoder_dim) for lang in languages
        })

        self._adapter_param_count = sum(p.numel() for p in self.adapters.parameters())
        self._lid_param_count = sum(p.numel() for p in self.language_id_head.parameters())

    def forward(
        self,
        audio_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        language: str,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        encoder_output = self.whisper.model.encoder(
            audio_features,
            attention_mask=attention_mask,
        ).last_hidden_state

        logits = self.adapters[language](encoder_output, decoder_input_ids)
        return logits

    def generate(self, audio_features: torch.Tensor, language: str, max_length: int = 256, **kwargs):
        encoder_output = self.whisper.model.encoder(audio_features).last_hidden_state

        decoder_start = torch.tensor(
            [[self.processor.tokenizer.bos_token_id]],
            device=audio_features.device,
        ).repeat(audio_features.size(0), 1)

        generated = decoder_start
        for _ in range(max_length):
            logits = self.adapters[language](encoder_output, generated)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == self.processor.tokenizer.eos_token_id:
                break

        return generated

    def detect_language(self, audio_features: torch.Tensor):
        encoder_output = self.whisper.model.encoder(audio_features).last_hidden_state
        logits = self.language_id_head(encoder_output)
        probs = torch.softmax(logits, dim=-1)
        lang_idx = probs[0].argmax(dim=-1).item()
        return {
            "language": self.languages[lang_idx],
            "confidence": probs[0, lang_idx].item(),
            "all_probs": {lang: probs[0, i].item() for i, lang in enumerate(self.languages)},
        }

    def get_trainable_params(self):
        return {
            "adapters": sum(p.numel() for p in self.adapters.parameters() if p.requires_grad),
            "language_id": sum(p.numel() for p in self.language_id_head.parameters() if p.requires_grad),
            "encoder": sum(p.numel() for p in self.whisper.model.encoder.parameters() if p.requires_grad),
        }

    def get_adapter_size_mb(self, lang: str = None):
        """Get size of adapter(s) in MB."""
        if lang:
            params = sum(p.numel() for p in self.adapters[lang].parameters())
            return params * 4 / (1024 * 1024)
        else:
            params = sum(p.numel() for p in self.adapters.parameters())
            return params * 4 / (1024 * 1024)

    def save_adapters(self, path: str):
        state = {
            "adapters": {name: adapter.state_dict() for name, adapter in self.adapters.items()},
            "language_id": self.language_id_head.state_dict(),
            "languages": self.languages,
            "config": self.config,
        }
        torch.save(state, path)

    def save_adapter(self, lang: str, path: str):
        torch.save({
            "adapter": self.adapters[lang].state_dict(),
            "language_id": self.language_id_head.state_dict(),
            "lang": lang,
            "config": self.config,
        }, path)

    def load_adapter(self, path: str, device: str = "cpu"):
        state = torch.load(path, map_location=device, weights_only=False)
        lang = state["lang"]
        if lang not in self.adapters:
            self.adapters[lang] = LanguageAdapter(state["config"], self.whisper.config.d_model).to(device)
        self.adapters[lang].load_state_dict(state["adapter"])
        self.language_id_head.load_state_dict(state["language_id"])
        return lang

    def load_adapters(self, path: str, device: str = "cpu"):
        state = torch.load(path, map_location=device, weights_only=False)
        self.languages = state["languages"]
        for name, adapter_state in state["adapters"].items():
            if name not in self.adapters:
                self.adapters[name] = LanguageAdapter(
                    state["config"],
                    self.whisper.config.d_model,
                ).to(device)
            self.adapters[name].load_state_dict(adapter_state)
        self.language_id_head.load_state_dict(state["language_id"])
