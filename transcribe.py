#!/usr/bin/env python3
"""Transcribe speech in an MP4 file to a plain-text file using Whisper."""

from __future__ import annotations

import argparse
from pathlib import Path

import whisper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a transcript from an MP4 file and save it as .txt."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input MP4 file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .txt path (default: same basename as input)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="base",
        metavar="NAME",
        help="Whisper model name, e.g. tiny, base, small, medium, large (default: base).",
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="CODE",
        help="Force language, e.g. en, es (ISO 639-1). Omit for auto-detect.",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    out_path = args.output
    if out_path is None:
        out_path = input_path.with_suffix(".txt")
    else:
        out_path = out_path.expanduser().resolve()

    print(f"Loading model {args.model!r}…")
    model = whisper.load_model(args.model)

    transcribe_kw: dict = {}
    if args.language:
        transcribe_kw["language"] = args.language

    print(f"Transcribing {input_path}…")
    result = model.transcribe(str(input_path), **transcribe_kw)
    text = (result.get("text") or "").strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote transcript to {out_path}")


if __name__ == "__main__":
    main()
