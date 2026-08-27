#!/usr/bin/env python3
"""PolyWhisper CLI — transcribe audio files from the command line.

Usage:
    polywhisper transcribe audio.wav --lang hi
    polywhisper transcribe audio.wav               # auto-detect language
    polywhisper batch ./audio_folder/ --lang te --output results.json
    polywhisper languages
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sys
import time
from pathlib import Path


def cmd_transcribe(args):
    from polywhisper import transcribe
    from polywhisper.model import PolyWhisper
    import sys, io, os, warnings

    # Suppress everything
    warnings.filterwarnings("ignore")
    old_stderr = sys.stderr
    old_stdout = sys.stdout
    sys.stderr = io.StringIO()
    sys.stdout = io.StringIO()
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = PolyWhisper(backbone=args.backbone, device=args.device)

    t0 = time.time()
    result = transcribe(
        args.audio,
        lang=args.lang,
        backbone=args.backbone,
        variant=args.variant,
        device=args.device,
        max_new_tokens=args.max_tokens,
        num_beams=args.beams,
        model=model,
        backend=args.backend,
    )
    dt = time.time() - t0

    sys.stderr = old_stderr
    sys.stdout = old_stdout

    if args.format == "text":
        print(result.text)
    elif args.format == "json":
        out = result.to_dict()
        out["duration_sec"] = result.duration_sec
        out["process_time_sec"] = round(dt, 2)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.format == "srt":
        for i, seg in enumerate(result.segments, 1):
            start = _fmt_srt(seg.start_sec)
            end = _fmt_srt(seg.end_sec)
            print(f"{i}\n{start} --> {end}\n{seg.text}\n")

    print(f"[{dt:.1f}s, {result.lang}, {len(result.segments)} segments]", file=old_stderr)


def cmd_batch(args):
    from polywhisper import transcribe
    from polywhisper.model import PolyWhisper
    import sys as _sys, io as _io, os as _os

    _os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    _os.environ["TOKENIZERS_PARALLELISM"] = "false"

    audio_dir = Path(args.audio_dir)
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm"}
    files = sorted(f for f in audio_dir.iterdir() if f.suffix.lower() in exts)
    if not files:
        print(f"No audio files found in {audio_dir}")
        return

    old_stderr = _sys.stderr
    _sys.stderr = _io.StringIO()
    print(f"Loading model (backbone={args.backbone})...", file=old_stderr)
    model = PolyWhisper(backbone=args.backbone, device=args.device)

    results = []
    total_t0 = time.time()
    for i, f in enumerate(files, 1):
        t0 = time.time()
        result = transcribe(
            f, lang=args.lang, backbone=args.backbone, variant=args.variant,
            device=args.device, max_new_tokens=args.max_tokens, num_beams=args.beams,
            model=model,
        )
        dt = time.time() - t0
        entry = result.to_dict()
        entry["file"] = str(f)
        entry["process_time_sec"] = round(dt, 2)
        results.append(entry)
        print(f"  [{i}/{len(files)}] {f.name} → {result.lang} ({dt:.1f}s)", file=old_stderr)

    total_dt = time.time() - total_t0
    out = {"results": results, "total_files": len(results), "total_time_sec": round(total_dt, 2)}

    _sys.stderr = old_stderr
    if args.output:
        Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"Wrote {args.output} ({len(results)} files)", file=old_stderr)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_languages(args):
    from polywhisper.model import AVAILABLE_LANGS, ADAPTER_REGISTRY, _adapter_search_paths
    from pathlib import Path

    search = _adapter_search_paths()
    print("Available languages:")
    for lang in AVAILABLE_LANGS:
        reg = ADAPTER_REGISTRY[lang]
        if isinstance(reg, dict):
            variants = list(reg.keys())
        else:
            variants = ["base"]
        found = []
        for variant in variants:
            fname = reg[variant] if isinstance(reg, dict) else reg
            for d in search:
                if (d / fname).exists():
                    found.append(f"{variant} ({fname})")
                    break
        status = ", ".join(found) if found else "NOT FOUND locally"
        print(f"  {lang:4s}  adapters: {status}")
    print(f"\nSearch paths: {[str(p) for p in search]}")


def _fmt_srt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cmd_export(args):
    from polywhisper.onnx_backend import export_language
    from polywhisper.model import ADAPTER_REGISTRY, _adapter_search_paths

    # Resolve adapter path
    reg = ADAPTER_REGISTRY.get(args.lang)
    if reg is None:
        print(f"Unknown language: {args.lang}. Available: {list(ADAPTER_REGISTRY.keys())}")
        return
    fname = reg.get(args.variant, reg.get("prod", list(reg.values())[0])) if isinstance(reg, dict) else reg
    adapter_path = None
    for d in _adapter_search_paths():
        p = d / fname
        if p.exists():
            adapter_path = p
            break
    if adapter_path is None:
        print(f"Adapter not found: {fname}")
        return

    print(f"Exporting {args.lang} ({args.variant}) to ONNX...")
    enc_path, dec_path = export_language(
        args.lang, str(adapter_path), backbone=args.backbone,
        out_dir=args.out_dir, int8=args.int8,
    )
    print(f"Done. Encoder: {enc_path}\nDecoder: {dec_path}")


def main():
    p = argparse.ArgumentParser(
        prog="polywhisper",
        description="PolyWhisper — efficient multilingual Indic ASR",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # transcribe
    t = sub.add_parser("transcribe", help="Transcribe a single audio file")
    t.add_argument("audio", help="Path to audio file")
    t.add_argument("--lang", "-l", default=None, help="Language code (hi/ta/te/bn/mr)")
    t.add_argument("--backbone", "-b", default="small", help="Whisper backbone (default: small)")
    t.add_argument("--variant", "-v", default="prod", help="Adapter variant (prod/base)")
    t.add_argument("--device", "-d", default="auto", help="Device (auto/cuda/mps/cpu)")
    t.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    t.add_argument("--beams", type=int, default=1, help="Beam width")
    t.add_argument("--format", "-f", default="text", choices=["text", "json", "srt"],
                    help="Output format")
    t.add_argument("--backend", default="auto", choices=["auto", "torch", "onnx"],
                    help="Inference backend (auto: ONNX if available, else torch)")
    t.set_defaults(func=cmd_transcribe)

    # batch
    b = sub.add_parser("batch", help="Transcribe all audio files in a directory")
    b.add_argument("audio_dir", help="Directory containing audio files")
    b.add_argument("--lang", "-l", default=None, help="Language code")
    b.add_argument("--backbone", "-b", default="small")
    b.add_argument("--variant", "-v", default="prod")
    b.add_argument("--device", "-d", default="auto")
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--beams", type=int, default=1)
    b.add_argument("--output", "-o", default=None, help="Output JSON file")
    b.set_defaults(func=cmd_batch)

    # export
    e = sub.add_parser("export", help="Export language to ONNX (CPU inference)")
    e.add_argument("--lang", "-l", required=True, help="Language to export")
    e.add_argument("--variant", "-v", default="prod", help="Adapter variant")
    e.add_argument("--backbone", "-b", default="small")
    e.add_argument("--out-dir", "-o", default="export/onnx", help="Output directory")
    e.add_argument("--int8", action="store_true", help="Also quantize to int8")
    e.set_defaults(func=cmd_export)

    # languages
    sub.add_parser("languages", help="List available languages and adapters").set_defaults(func=cmd_languages)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
