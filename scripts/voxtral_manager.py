#!/usr/bin/env python3
"""Local-first Voxtral helper for setup, model download, STT/TTS orchestration, and hotkey integration."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

DEFAULT_VENV = Path(__file__).resolve().parent / "venvs" / "voxtral"
DEFAULT_MODELS = Path(__file__).resolve().parent / "models" / "voxtral"


def _python_in_venv(venv_path: Path) -> str:
    if os.name == "nt":
        return str(venv_path / "Scripts" / "python.exe")
    return str(venv_path / "bin" / "python")


def create_venv(venv_path: Path) -> str:
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    return f"Created venv at {venv_path}"


def install_dependencies(venv_path: Path, with_gpu: bool, with_clone: bool) -> str:
    py = _python_in_venv(venv_path)
    base = [
        "pip",
        "setuptools",
        "wheel",
        "requests>=2.31",
        "huggingface_hub>=0.24",
        "soundfile>=0.12",
        "pydub>=0.25",
        "numpy>=1.24,<2.0",
    ]
    stt_tts = [
        "faster-whisper>=1.1.0",
        "ctranslate2>=4.5",
        "TTS>=0.22",
        "torch>=2.3",
        "torchaudio>=2.3",
    ]
    if with_gpu:
        # User can later replace index URL with specific CUDA build if needed.
        stt_tts.append("onnxruntime-gpu>=1.18")
    else:
        stt_tts.append("onnxruntime>=1.18")
    if with_clone:
        stt_tts.extend(["resemblyzer>=0.1.4", "librosa>=0.10"]) 

    commands = [
        [py, "-m", "pip", "install", "--upgrade", *base],
        [py, "-m", "pip", "install", "--upgrade", *stt_tts],
    ]
    logs: List[str] = []
    for cmd in commands:
        done = subprocess.run(cmd, capture_output=True, text=True)
        logs.append(f"$ {' '.join(cmd)}\n{done.stdout}{done.stderr}".strip())
        if done.returncode != 0:
            raise RuntimeError("\n\n".join(logs))
    return "\n\n".join(logs)


def search_hf_gguf(query: str, limit: int = 20) -> str:
    if requests is None:
        raise RuntimeError("requests non disponibile")
    url = "https://huggingface.co/api/models"
    params = {"search": query, "limit": str(limit), "full": "true"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    models = resp.json() if isinstance(resp.json(), list) else []
    lines: List[str] = []
    for model in models:
        model_id = model.get("id", "")
        siblings = model.get("siblings") or []
        ggufs = [s.get("rfilename", "") for s in siblings if str(s.get("rfilename", "")).lower().endswith(".gguf")]
        if ggufs:
            lines.append(f"{model_id} :: {len(ggufs)} GGUF -> {', '.join(ggufs[:4])}")
    if not lines:
        return f"Nessun GGUF trovato per query '{query}'."
    return "\n".join(lines)


def download_file(url: str, output_path: Path) -> str:
    if requests is None:
        raise RuntimeError("requests non disponibile")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=90) as resp:
        resp.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return f"Downloaded model to {output_path}"


def run_stt(args: argparse.Namespace) -> str:
    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments, info = model.transcribe(
        args.input,
        language=args.language or None,
        task=args.task,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        word_timestamps=args.word_timestamps,
    )
    lines: List[str] = [f"Detected language: {info.language} ({info.language_probability:.3f})"]
    text_parts: List[str] = []
    for segment in segments:
        row = f"[{segment.start:7.2f}s -> {segment.end:7.2f}s] {segment.text.strip()}"
        lines.append(row)
        text_parts.append(segment.text.strip())

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(text_parts), encoding="utf-8")
        lines.append(f"Saved transcription to {out}")
    return "\n".join(lines)


def run_tts(args: argparse.Namespace) -> str:
    from TTS.api import TTS  # type: ignore

    tts = TTS(model_name=args.model, progress_bar=False, gpu=args.device.startswith("cuda"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "text": args.text,
        "file_path": str(out),
        "speaker_wav": args.speaker_wav or None,
        "language": args.language or None,
        "speed": args.speed,
    }
    tts.tts_to_file(**{k: v for k, v in kwargs.items() if v is not None})
    return f"Generated audio at {out}"


def generate_hotkey(mode: str, hotkey: str, output: Path, tool_root: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode == "windows-ahk-stt":
        content = textwrap.dedent(
            f"""
            ; Auto-generated Voxtral hotkey launcher
            ; Requires AutoHotkey v2 installed.
            ^+Space::
            {{
                Run '"{tool_root / 'start.bat'}" --voxtral-stt'
            }}
            """
        ).strip() + "\n"
    elif mode == "windows-ahk-tts":
        content = textwrap.dedent(
            f"""
            ; Auto-generated Voxtral hotkey launcher
            ^+T::
            {{
                Run '"{tool_root / 'start.bat'}" --voxtral-tts'
            }}
            """
        ).strip() + "\n"
    else:
        content = textwrap.dedent(
            f"""
            #!/usr/bin/env bash
            # Linux/macOS sample shortcut target.
            # Bind this script to a system hotkey.
            python "{Path(__file__).resolve()}" stt --input /tmp/in.wav --model small
            """
        ).strip() + "\n"

    output.write_text(content, encoding="utf-8")
    return f"Generated hotkey helper at {output} ({platform.system()})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voxtral local toolkit helper")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup")
    setup.add_argument("--venv", default=str(DEFAULT_VENV))
    setup.add_argument("--with-gpu", action="store_true")
    setup.add_argument("--with-clone", action="store_true")

    search = sub.add_parser("search-gguf")
    search.add_argument("--query", default="voxtral")
    search.add_argument("--limit", type=int, default=30)

    dl = sub.add_parser("download")
    dl.add_argument("--url", required=True)
    dl.add_argument("--output", required=True)

    stt = sub.add_parser("stt")
    stt.add_argument("--input", required=True)
    stt.add_argument("--model", default="small")
    stt.add_argument("--language", default="")
    stt.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    stt.add_argument("--device", default="cpu")
    stt.add_argument("--compute-type", default="int8")
    stt.add_argument("--beam-size", type=int, default=5)
    stt.add_argument("--vad-filter", action="store_true")
    stt.add_argument("--word-timestamps", action="store_true")
    stt.add_argument("--output", default="")

    tts = sub.add_parser("tts")
    tts.add_argument("--text", required=True)
    tts.add_argument("--output", required=True)
    tts.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    tts.add_argument("--speaker-wav", default="")
    tts.add_argument("--language", default="it")
    tts.add_argument("--speed", type=float, default=1.0)
    tts.add_argument("--device", default="cpu")

    hk = sub.add_parser("hotkey")
    hk.add_argument("--mode", default="windows-ahk-stt", choices=["windows-ahk-stt", "windows-ahk-tts", "linux-script"])
    hk.add_argument("--hotkey", default="ctrl+shift+space")
    hk.add_argument("--output", required=True)
    hk.add_argument("--tool-root", default=str(Path(__file__).resolve().parents[1]))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "setup":
            venv_path = Path(args.venv).expanduser()
            create_venv(venv_path)
            print(install_dependencies(venv_path, with_gpu=args.with_gpu, with_clone=args.with_clone))
            return 0
        if args.command == "search-gguf":
            print(search_hf_gguf(args.query, args.limit))
            return 0
        if args.command == "download":
            print(download_file(args.url, Path(args.output).expanduser()))
            return 0
        if args.command == "stt":
            print(run_stt(args))
            return 0
        if args.command == "tts":
            print(run_tts(args))
            return 0
        if args.command == "hotkey":
            print(generate_hotkey(args.mode, args.hotkey, Path(args.output).expanduser(), Path(args.tool_root).expanduser()))
            return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
