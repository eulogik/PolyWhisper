#!/usr/bin/env python3
"""Export a PolyWhisper LoRA expert to ONNX (encoder + decoder-step), verify parity,
optionally int8-quantize, and spot-check end-to-end greedy decoding vs torch.

Usage:
  python export_onnx.py --model-size base --adapter hi_best_v5.pt --lang hi
                        --adapter-dir polywhisper_output/adapters_v3 --out-dir export/onnx
  python export_onnx.py --int8 ...   (also writes *_int8.onnx + e2e spot-check)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from train_v3 import PolyWhisperV3, processor, MODEL_SIZES, LANG_TOKENS, TASK_TOKEN, NO_TIME_TOKEN  # noqa: E402

START_TOKEN = 50257  # BOS/SOT (also EOS)
MAX_NEW = 256  # must match eval (128 truncates FLEURS long utterances)


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


def export(wrapper, sample_in, names, dyn_axes, path):
    with torch.no_grad():
        torch.onnx.export(
            wrapper, sample_in, str(path),
            input_names=names[0], output_names=[names[1]],
            dynamic_axes=dyn_axes, opset_version=17, do_constant_folding=True,
            dynamo=False,
        )
    print(f"  wrote {path}")
    return path


def max_diff(a, b):
    return float(np.abs(a - b).max())


def main():
    A = argparse.ArgumentParser()
    A.add_argument("--model-size", default="base", choices=["base", "small"])
    A.add_argument("--adapter", required=True)
    A.add_argument("--adapter-dir", default="polywhisper_output/adapters_v3")
    A.add_argument("--lang", required=True)
    A.add_argument("--out-dir", default="export/onnx")
    A.add_argument("--int8", action="store_true", help="also write int8-quantized onnx")
    A.add_argument("--encoder-lora", action="store_true", help="adapter was trained with encoder LoRA (hi_v5)")
    A.add_argument("--spot-check", type=int, default=20,
                    help="samples for e2e greedy check (uses FLEURS hi test json if present)")
    a = A.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.lang}_{Path(a.adapter).stem}"

    print(f"Loading {a.model_size} backbone + adapter {a.adapter} ...")
    t0 = time.time()
    model = PolyWhisperV3(whisper_name=MODEL_SIZES[a.model_size],
                          encoder_lora=a.encoder_lora).to("cpu").eval()
    model.add_language(a.lang)
    model.load_adapter(a.lang, str(Path(a.adapter_dir) / a.adapter))
    model.set_language(a.lang)
    print(f"  loaded in {time.time()-t0:.0f}s")

    # adapter params live outside model.whisper; freeze them so the legacy
    # tracer can constant-fold the LoRA weights into the exported graph
    for p in model.lora_adapters[a.lang].parameters():
        p.requires_grad = False

    d = model.d_model
    enc_w = EncoderWrapper(model.whisper)
    dec_w = DecoderWrapper(model.whisper)

    feats = torch.randn(1, 80, 3000)
    enc_hidden = torch.randn(1, 1500, d)
    ids = torch.randint(0, 100, (1, 10))

    # ---- export encoder ----
    print("Exporting encoder...")
    export(enc_w, (feats,), (["feats"], "enc_hidden"),
           {"feats": {0: "batch", 2: "time"}, "enc_hidden": {0: "batch", 1: "enc_seq"}},
           out_dir / f"{tag}_encoder.onnx")

    # ---- export decoder-step ----
    print("Exporting decoder...")
    export(dec_w, (ids, enc_hidden), (["input_ids", "enc_hidden"], "logits"),
           {"input_ids": {0: "batch", 1: "seq"},
            "enc_hidden": {0: "batch", 1: "enc_seq"},
            "logits": {0: "batch", 1: "seq"}},
           out_dir / f"{tag}_decoder.onnx")

    # ---- parity check (fp32) ----
    import onnxruntime as ort
    print("Parity check (fp32 onnx vs torch)...")
    s_enc = ort.InferenceSession(str(out_dir / f"{tag}_encoder.onnx"),
                                 providers=["CPUExecutionProvider"])
    s_dec = ort.InferenceSession(str(out_dir / f"{tag}_decoder.onnx"),
                                 providers=["CPUExecutionProvider"])
    with torch.no_grad():
        e_t = enc_w(feats).numpy()
        d_t = dec_w(ids, enc_hidden).numpy()
    e_o = s_enc.run(None, {"feats": feats.numpy()})[0]
    d_o = s_dec.run(None, {"input_ids": ids.numpy(),
                           "enc_hidden": enc_hidden.numpy()})[0]
    print(f"  encoder max diff: {max_diff(e_t, e_o):.2e}")
    print(f"  decoder max diff: {max_diff(d_t, d_o):.2e}")
    assert max_diff(e_t, e_o) < 1e-3 and max_diff(d_t, d_o) < 1e-3, "fp32 parity FAILED"

    int8_dec = None
    if a.int8:
        print("Quantizing int8 (dynamic)...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        for part in ("encoder", "decoder"):
            src = out_dir / f"{tag}_{part}.onnx"
            dst = out_dir / f"{tag}_{part}_int8.onnx"
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
            print(f"  wrote {dst}")
        s_dec8 = ort.InferenceSession(str(out_dir / f"{tag}_decoder_int8.onnx"),
                                      providers=["CPUExecutionProvider"])
        int8_dec = s_dec8
        d_o8 = s_dec8.run(None, {"input_ids": ids.numpy(),
                                 "enc_hidden": enc_hidden.numpy()})[0]
        print(f"  decoder int8 max diff vs fp32 onnx: {max_diff(d_o, d_o8):.2e}")

    # ---- end-to-end greedy spot check ----
    test_json = Path("polywhisper_output/data/fleurs_hi_in_test.json")
    if not test_json.exists():
        print("No fleurs test json — skipping e2e spot check.")
        return
    records = json.load(open(test_json))[: a.spot_check]

    def mel(audio):
        f = processor.feature_extractor([audio], sampling_rate=16000,
                                        return_tensors="pt", padding=True)["input_features"]
        if f.shape[-1] < 3000:
            f = torch.cat([f, torch.zeros(1, 80, 3000 - f.shape[-1], dtype=f.dtype)], dim=-1)
        return f

    def decode_ort(s_enc_, s_dec_, audio, max_new=MAX_NEW):
        eh = s_enc_.run(None, {"feats": mel(audio).numpy()})[0]
        ids_ = np.array([[START_TOKEN, LANG_TOKENS[a.lang], TASK_TOKEN, NO_TIME_TOKEN]],
                        dtype=np.int64)
        for _ in range(max_new):
            lg = s_dec_.run(None, {"input_ids": ids_,
                                   "enc_hidden": eh})[0][0, -1]
            nxt = int(np.argmax(lg))
            ids_ = np.concatenate([ids_, [[nxt]]], axis=1)
            if nxt == START_TOKEN:
                break
        return processor.decode(ids_[0], skip_special_tokens=True).strip()

    def compute_wer(ref, hyp):
        rw, hw = ref.split(), hyp.split()
        dp = [[0] * (len(hw) + 1) for _ in range(len(rw) + 1)]
        for i in range(len(rw) + 1):
            dp[i][0] = i
        for j in range(len(hw) + 1):
            dp[0][j] = j
        for i in range(1, len(rw) + 1):
            for j in range(1, len(hw) + 1):
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                               dp[i - 1][j - 1] + (rw[i - 1] != hw[j - 1]))
        return dp[len(rw)][len(hw)] / len(rw)

    print(f"\nE2E greedy spot-check on {len(records)} FLEURS hi samples:")
    tot_w_t, err_w_t, tot_w_o, err_w_o = 0, 0, 0, 0
    for r in records:
        audio, _ = sf.read(r["wav"])
        with torch.no_grad():
            torch_hyp = processor.decode(
                model.generate(mel(audio), lang=a.lang, max_new_tokens=MAX_NEW,
                    num_beams=1, use_cache=False, task="transcribe")[0],
                skip_special_tokens=True).strip()
        ort_hyp = decode_ort(s_enc, int8_dec or s_dec, audio)
        ref = r["text"].strip()
        n = len(ref.split())
        w_t, w_o = compute_wer(ref, torch_hyp), compute_wer(ref, ort_hyp)
        err_w_t += w_t * n; err_w_o += w_o * n; tot_w_t += n; tot_w_o += n
        if w_t != w_o:
            print(f"  diff: torch='{torch_hyp[:40]}' onnx='{ort_hyp[:40]}'")
    print(f"  torch WER: {100*err_w_t/max(1,tot_w_t):.1f}%  onnx WER: {100*err_w_o/max(1,tot_w_o):.1f}%")
    if int8_dec is not None:
        print("  (onnx numbers above are INT8)")


if __name__ == "__main__":
    main()