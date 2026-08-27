"""ONNX Runtime inference backend — no torch required at inference time.

Exports per-language ONNX models (encoder+decoder with LoRA baked in),
then runs autoregressive decoding via onnxruntime.
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from polywhisper.audio import load_audio, TARGET_SR

START_TOKEN = 50257
LANG_TOKENS = {
    "en": 50259, "hi": 50276, "ta": 50287,
    "te": 50299, "bn": 50302, "mr": 50320,
}
TASK_TOKEN = 50359
NO_TIME_TOKEN = 50363


def _export_dir():
    candidates = [
        Path(__file__).parent.parent / "export" / "onnx",
        Path.home() / "polywhisper_output" / "export" / "onnx",
        Path("/Volumes/KIOXIA 1TB/polywhisper_output/export/onnx"),
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


class OnnxBackend:
    """ONNX Runtime inference for PolyWhisper (no torch needed)."""

    def __init__(self, export_dir: Optional[Path] = None, lang: str = "hi",
                 int8: bool = False, device: str = "cpu"):
        if ort is None:
            raise ImportError("onnxruntime is required: pip install onnxruntime")
        self.export_dir = Path(export_dir) if export_dir else _export_dir()
        self.lang = lang
        self.int8 = int8

        suffix = "_int8" if int8 else ""
        enc_path = self.export_dir / f"{lang}_encoder{suffix}.onnx"
        dec_path = self.export_dir / f"{lang}_decoder{suffix}.onnx"

        if not enc_path.exists() or not dec_path.exists():
            raise FileNotFoundError(
                f"ONNX models not found for '{lang}'. "
                f"Expected: {enc_path.name}, {dec_path.name}\n"
                f"Run: python export_onnx.py --lang {lang} --adapter {lang}_best_prod.pt "
                f"--adapter-dir polywhisper_output/adapters_v3 --int8"
            )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]

        self.enc_session = ort.InferenceSession(str(enc_path), opts, providers=providers)
        self.dec_session = ort.InferenceSession(str(dec_path), opts, providers=providers)

    @staticmethod
    def _mel(audio, processor):
        """Compute mel features from audio array."""
        import torch
        feats = processor.feature_extractor(
            [audio], sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )["input_features"]
        if feats.shape[-1] < 3000:
            feats = torch.cat(
                [feats, torch.zeros(1, 80, 3000 - feats.shape[-1], dtype=feats.dtype)], -1
            )
        return feats.numpy()

    def transcribe(self, audio_input, processor, max_new_tokens=256,
                   num_beams=1, language=None) -> str:
        """Transcribe audio via ONNX Runtime (greedy only for now)."""
        lang = language or self.lang

        # Load audio
        if isinstance(audio_input, (str, Path)):
            audio, _ = load_audio(str(audio_input), TARGET_SR)
        else:
            audio = np.asarray(audio_input, dtype=np.float32)

        # Compute features
        feats = self._mel(audio, processor)

        # Encode
        enc_hidden = self.enc_session.run(None, {"feats": feats})[0]

        # Decode (greedy autoregressive)
        lang_token = LANG_TOKENS.get(lang, LANG_TOKENS["hi"])
        ids = np.array([[START_TOKEN, lang_token, TASK_TOKEN, NO_TIME_TOKEN]], dtype=np.int64)

        for _ in range(max_new_tokens):
            logits = self.dec_session.run(None, {
                "input_ids": ids,
                "enc_hidden": enc_hidden,
            })[0]
            next_token = int(np.argmax(logits[0, -1]))
            ids = np.concatenate([ids, [[next_token]]], axis=1)
            if next_token == START_TOKEN:
                break

        return processor.decode(ids[0], skip_special_tokens=True).strip()


def export_language(lang, adapter_path, backbone="small", out_dir=None, int8=True):
    """Export a single language to ONNX with LoRA baked in."""
    import torch
    import torch.nn as nn
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if ort is None:
        raise ImportError("onnxruntime required for export")

    out_dir = Path(out_dir) if out_dir else _export_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    whisper_name = f"openai/whisper-{backbone}"
    processor = WhisperProcessor.from_pretrained(whisper_name)

    # Load model with LoRA
    from polywhisper.model import PolyWhisper
    model = PolyWhisper(backbone=backbone, device="cpu")
    model.load_adapter(lang, adapter_path)
    model.set_language(lang)

    # Freeze everything for export
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.lora_adapters[lang].parameters():
        p.requires_grad = False

    # Wrapper classes matching export_onnx.py
    class EncoderWrapper(nn.Module):
        def __init__(self, whisper):
            super().__init__()
            self.w = whisper
        def forward(self, feats):
            return self.w.model.encoder(feats).last_hidden_state

    class DecoderWrapper(nn.Module):
        def __init__(self, whisper):
            super().__init__()
            self.w = whisper
        def forward(self, input_ids, enc_hidden):
            out = self.w.model.decoder(input_ids, encoder_hidden_states=enc_hidden).last_hidden_state
            return self.w.proj_out(out)

    feats = torch.randn(1, 80, 3000)
    ids = torch.randint(0, 100, (1, 10))
    d = model.d_model
    enc_hidden = torch.randn(1, 1500, d)

    # Export encoder
    enc_w = EncoderWrapper(model.whisper)
    enc_path = out_dir / f"{lang}_encoder.onnx"
    torch.onnx.export(
        enc_w, (feats,), str(enc_path),
        input_names=["feats"], output_names=["enc_hidden"],
        dynamic_axes={"feats": {0: "batch", 2: "time"}, "enc_hidden": {0: "batch", 1: "enc_seq"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"  encoder: {enc_path}")

    # Export decoder
    dec_w = DecoderWrapper(model.whisper)
    dec_path = out_dir / f"{lang}_decoder.onnx"
    torch.onnx.export(
        dec_w, (ids, enc_hidden), str(dec_path),
        input_names=["input_ids", "enc_hidden"], output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "enc_hidden": {0: "batch", 1: "enc_seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"  decoder: {dec_path}")

    # Int8 quantization
    if int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        for part in ("encoder", "decoder"):
            src = out_dir / f"{lang}_{part}.onnx"
            dst = out_dir / f"{lang}_{part}_int8.onnx"
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
            print(f"  int8: {dst}")

    # Parity check
    s_enc = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    s_dec = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        e_t = enc_w(feats).numpy()
        d_t = dec_w(ids, enc_hidden).numpy()
    e_o = s_enc.run(None, {"feats": feats.numpy()})[0]
    d_o = s_dec.run(None, {"input_ids": ids.numpy(), "enc_hidden": enc_hidden.numpy()})[0]
    enc_diff = float(np.abs(e_t - e_o).max())
    dec_diff = float(np.abs(d_t - d_o).max())
    print(f"  parity: enc={enc_diff:.2e} dec={dec_diff:.2e} {'OK' if enc_diff < 1e-3 and dec_diff < 1e-3 else 'FAIL'}")

    return enc_path, dec_path
