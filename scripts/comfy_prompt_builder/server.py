"""FastAPI server for the simplified prompt creator."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SYSTEM_PROMPT_PATH = Path(
    os.getenv("COMFY_SYSTEM_PROMPT")
    or BASE_DIR / "comfy_diffusion_image_prompt.txt"
)
DEFAULT_KOBOLD_HOST = os.getenv("KOBOLD_HOST", "http://127.0.0.1:5001").rstrip("/")


def _load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return (
            "You are an expert creative writer. Take any rough idea and expand it "
            "into a polished text-to-image prompt. Write flowing sentences that "
            "cover the subject, surrounding scene, artistic style, and technical "
            "touches. Keep the tone vivid and cohesive."
        )


SYSTEM_PROMPT = _load_system_prompt()


class GenerateRequest(BaseModel):
    idea: str
    kobold_url: Optional[str] = None


app = FastAPI(title="Prompt Creator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def _normalise_base_url(raw: Optional[str]) -> str:
    base = (raw or "").strip() or DEFAULT_KOBOLD_HOST
    parsed = urlparse(base)
    if not parsed.scheme:
        base = f"http://{base}"
    return base.rstrip("/")


def _format_prompt(idea: str) -> str:
    clean_idea = idea.strip()
    if not clean_idea:
        raise HTTPException(status_code=400, detail="Idea cannot be empty")
    user_message = f"Create a new prompt from this idea: {clean_idea}"
    sections = [
        f"System:\n{SYSTEM_PROMPT}",
        f"User:\n{user_message}",
        "Assistant:\n",
    ]
    return "\n\n".join(sections)


def _call_kobold(base: str, idea: str) -> Dict[str, Optional[str]]:
    payload = {
        "prompt": _format_prompt(idea),
        "max_length": 600,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False,
        "stop_sequence": ["\nUser:", "\nSystem:"],
    }
    try:
        response = requests.post(
            f"{base}/api/v1/generate", json=payload, timeout=(10, 180)
        )
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - depends on external service
        raise HTTPException(status_code=502, detail=f"Kobold connection failed: {exc}")
    data = response.json()
    results = data.get("results")
    if not results:
        raise HTTPException(status_code=502, detail="Kobold returned no content")
    text = (results[0] or {}).get("text", "").strip()
    if text.lower().startswith("assistant:"):
        text = text.split(":", 1)[1].strip()
    if not text:
        raise HTTPException(status_code=502, detail="Kobold response was empty")
    return {"content": text, "model": data.get("model")}


@app.get("/api/kobold/status")
def kobold_status(
    url: Optional[str] = Query(default=None, description="Base URL of Kobold server")
) -> Dict[str, Optional[str] | bool]:
    base = _normalise_base_url(url)
    result: Dict[str, Optional[str] | bool] = {"url": base, "online": False, "model": None}
    try:
        response = requests.get(f"{base}/api/v1/model", timeout=5)
        response.raise_for_status()
        data = response.json()
        model_name = (
            data.get("result")
            or data.get("model")
            or data.get("name")
            or data.get("model_name")
        )
        result["online"] = True
        result["model"] = model_name
    except Exception as exc:  # pragma: no cover - depends on external service
        result["error"] = str(exc)
    return result


@app.post("/api/generate_prompt")
def generate_prompt(request: GenerateRequest) -> Dict[str, Optional[str]]:
    base = _normalise_base_url(request.kobold_url)
    return _call_kobold(base, request.idea)


__all__ = ["app"]
