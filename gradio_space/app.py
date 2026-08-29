#!/usr/bin/env python3
"""PolyWhisper — Gradio demo for multilingual Indic ASR.

Run locally:
    python app.py                    # local only
    python app.py --share            # public URL via gradio.live
    python app.py --share --port 7860
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
import numpy as np
import time
import json
from pathlib import Path

# Global model cache
_model = None
_processor = None

LANGUAGES = {
    "Hindi (हिन्दी)": "hi",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Bengali (বাংলা)": "bn",
    "Marathi (मराठी)": "mr",
}

SAMPLE_AUDIO_DIR = Path(__file__).parent / "samples"


def load_model():
    global _model, _processor
    if _model is None:
        from polywhisper.model import PolyWhisper
        _model = PolyWhisper(backbone="small", device="cpu")
        # Pre-load all adapters
        for lang in ["hi", "ta", "te", "bn", "mr"]:
            try:
                _model.auto_load_adapter(lang, variant="prod")
            except FileNotFoundError:
                pass
        _processor = _model.processor
    return _model, _processor


def transcribe_audio(audio, language, max_tokens, beam_width, output_format):
    """Transcribe uploaded or recorded audio."""
    if audio is None:
        return "Please upload or record audio first.", "", None

    lang_code = LANGUAGES.get(language, "hi")
    model, processor = load_model()

    # Load audio
    import soundfile as sf
    if isinstance(audio, tuple):
        # Gradio returns (sample_rate, numpy_array) for recorded audio
        sr, audio_array = audio
        audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sr != 16000:
            import resampy
            audio_array = resampy.resample(audio_array, sr, 16000)
    else:
        # File path
        audio_array, sr = sf.read(str(audio), dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sr != 16000:
            import resampy
            audio_array = resampy.resample(audio_array, sr, 16000)

    # Transcribe
    t0 = time.time()
    result = _transcribe(model, processor, audio_array, lang_code, max_tokens, int(beam_width))
    dt = time.time() - t0

    # Format output
    text = result["text"]
    duration = result["duration"]

    # Build output based on format
    if output_format == "Plain Text":
        output_text = text
        srt_output = None
    elif output_format == "JSON":
        output_text = json.dumps(result, ensure_ascii=False, indent=2)
        srt_output = None
    elif output_format == "SRT Subtitles":
        output_text = text
        srt_lines = []
        for i, seg in enumerate(result.get("segments", []), 1):
            start = _fmt_srt(seg["start"])
            end = _fmt_srt(seg["end"])
            srt_lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
        srt_output = "\n".join(srt_lines)
    else:
        output_text = text
        srt_output = None

    # Stats
    stats = f"Language: {language} | Duration: {duration:.1f}s | Processed in: {dt:.1f}s | Speed: {duration/dt:.1f}× realtime"

    return output_text, stats, srt_output


def _transcribe(model, processor, audio, lang, max_new_tokens=256, num_beams=1):
    """Core transcription function."""
    import torch
    model.set_language(lang)

    feats = processor.feature_extractor(
        [audio], sampling_rate=16000, return_tensors="pt", padding=True
    )["input_features"]
    if feats.shape[-1] < 3000:
        feats = torch.cat(
            [feats, torch.zeros(1, 80, 3000 - feats.shape[-1], dtype=feats.dtype)], -1
        )

    out = model.generate(
        feats, lang=lang, max_new_tokens=max_new_tokens,
        num_beams=num_beams, use_cache=True, task="transcribe",
    )
    text = processor.decode(out[0], skip_special_tokens=True).strip()

    return {
        "text": text,
        "lang": lang,
        "duration": len(audio) / 16000,
        "segments": [{"text": text, "start": 0.0, "end": len(audio) / 16000}],
    }


def _fmt_srt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ===== Custom CSS =====
CUSTOM_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.5em;
}
.main-title h1 {
    font-size: 2.2em;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2em;
}
.main-title p {
    font-size: 1.1em;
    color: #666;
    margin-top: 0;
}
.lang-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 500;
    margin: 2px;
}
.stats-box {
    background: #f0f7ff;
    border: 1px solid #b3d4fc;
    border-radius: 8px;
    padding: 8px 12px;
    font-family: monospace;
    font-size: 0.9em;
}
footer { display: none !important; }
"""


def build_app():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="PolyWhisper — Indic ASR Demo",
    ) as app:
        # Header
        gr.HTML("""
        <div class="main-title">
            <h1>PolyWhisper</h1>
            <p>Efficient Multilingual Indic Speech Recognition</p>
            <p style="font-size:0.9em; color:#888;">
                Hindi · Tamil · Telugu · Bengali · Marathi
                &nbsp;|&nbsp; Powered by Whisper Small + LoRA
            </p>
        </div>
        """)

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload or Record Audio",
                    sources=["upload", "microphone"],
                    type="numpy",
                    waveform_options={"waveform_color": "#667eea"},
                )

                with gr.Row():
                    language = gr.Dropdown(
                        choices=list(LANGUAGES.keys()),
                        value="Hindi (हिन्दी)",
                        label="Language",
                        scale=2,
                    )
                    output_format = gr.Dropdown(
                        choices=["Plain Text", "JSON", "SRT Subtitles"],
                        value="Plain Text",
                        label="Output Format",
                        scale=1,
                    )

                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=64, maximum=512, value=256, step=64,
                        label="Max Tokens",
                    )
                    beam_width = gr.Slider(
                        minimum=1, maximum=5, value=1, step=1,
                        label="Beam Width",
                    )

                transcribe_btn = gr.Button(
                    "Transcribe",
                    variant="primary",
                    size="lg",
                    elem_id="transcribe-btn",
                )

            # Right column: Output
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Transcription",
                    lines=8,
                    elem_id="output-text",
                )
                stats = gr.Textbox(
                    label="Stats",
                    interactive=False,
                    elem_classes=["stats-box"],
                )
                srt_output = gr.Textbox(
                    label="SRT Subtitles",
                    lines=8,
                    visible=False,
                )

        # Wire up output format visibility
        output_format.change(
            fn=lambda fmt: gr.update(visible=fmt == "SRT Subtitles"),
            inputs=[output_format],
            outputs=[srt_output],
        )

        # Wire up transcribe button
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, language, max_tokens, beam_width, output_format],
            outputs=[output_text, stats, srt_output],
        )

        # Examples
        gr.Examples(
            examples=[
                ["polywhisper_output/data/audio_fleurs_hi_in/000000.wav", "Hindi (हिन्दी)"],
                ["polywhisper_output/data/audio_fleurs_ta_in/000000.wav", "Tamil (தமிழ்)"],
            ],
            inputs=[audio_input, language],
            label="Try these examples",
        )

        # Footer
        gr.HTML("""
        <div style="text-align:center; margin-top:2em; padding:1em; color:#888; font-size:0.85em;">
            <p><b>PolyWhisper</b> — Frozen Whisper Small + Per-Language LoRA Adapters</p>
            <p>
                <a href="https://huggingface.co/eulogik/polywhisper" target="_blank">🤗 Model</a> ·
                <a href="https://github.com/eulogikdeveloper/PolyWhisper" target="_blank"> GitHub</a> ·
                <a href="https://huggingface.co/papers" target="_blank">📄 Paper</a>
            </p>
            <p>Hindi WER 37.2 | Tamil WER 60.4 on FLEURS benchmark</p>
        </div>
        """)

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public URL via gradio.live")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
    )
