"""FastAPI server for the simplified prompt creator."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SYSTEM_PROMPT_PATH = Path(
    os.getenv("COMFY_SYSTEM_PROMPT")
    or BASE_DIR / "comfy_diffusion_image_prompt.txt"
)


def _coerce_base(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    base = raw.strip()
    if not base:
        return None
    parsed = urlparse(base)
    if not parsed.scheme:
        base = f"http://{base}"
    return base.rstrip("/")


DEFAULT_KOBOLD_HOST = (
    _coerce_base(os.getenv("KOBOLD_HOST") or "http://127.0.0.1:5001")
    or "http://127.0.0.1:5001"
)


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

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _candidate_bases(preferred: Optional[str]) -> Iterable[str]:
    raw_candidates = [
        preferred,
        DEFAULT_KOBOLD_HOST,
        "http://127.0.0.1:5001",
        "http://localhost:5001",
        "http://0.0.0.0:5001",
    ]
    seen = set()
    for raw in raw_candidates:
        base = _coerce_base(raw)
        if base and base not in seen:
            seen.add(base)
            yield base


def _format_prompt(idea: str) -> str:
    clean_idea = idea.strip()
    if not clean_idea:
        raise HTTPException(status_code=400, detail="Idea cannot be empty")
    user_message = f"Create a new prompt from this idea:{clean_idea}"
    sections = [
        f"System:\n{SYSTEM_PROMPT}",
        f"User:\n{user_message}",
        "Assistant:\n",
    ]
    return "\n\n".join(sections)


def _extract_model_name(data: Dict[str, object]) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    direct_keys = ("model", "model_name", "name")
    for key in direct_keys:
        value = data.get(key)
        if value:
            return str(value)
    nested = data.get("result")
    if isinstance(nested, dict):
        for key in direct_keys:
            value = nested.get(key)
            if value:
                return str(value)
    return None


def _probe_model(base: str) -> Tuple[bool, Optional[str], Optional[str]]:
    endpoints = [
        "/api/v1/model",
        "/api/model",
        "/v1/model",
        "/api/v1/status",
    ]
    errors = []
    for path in endpoints:
        url = f"{base}{path}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            model_name = _extract_model_name(data)
            return True, model_name, None
        except Exception as exc:  # pragma: no cover - depends on external service
            errors.append(f"{url}: {exc}")
    return False, None, "; ".join(errors) if errors else None


def _call_kobold(base: str, idea: str) -> Dict[str, Optional[str]]:
    payload = {"prompt": _format_prompt(idea)}
    endpoints = [
        f"{base}/api/v1/generate",
        f"{base}/api/generate",
        f"{base}/v1/generate",
    ]
    errors = []
    for url in endpoints:
        try:
            response = requests.post(url, json=payload, timeout=(10, 180))
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - depends on external service
            errors.append(f"{url}: {exc}")
            continue
        data = response.json()
        text: Optional[str] = None
        results = data.get("results")
        if isinstance(results, list) and results:
            first = results[0] or {}
            if isinstance(first, dict):
                text = str(first.get("text", "")).strip()
        elif "text" in data:
            candidate = data.get("text")
            if isinstance(candidate, str):
                text = candidate.strip()
        if text and text.lower().startswith("assistant:"):
            text = text.split(":", 1)[1].strip()
        if text:
            return {"content": text, "model": _extract_model_name(data)}
        errors.append(f"{url}: Kobold response was empty")
    raise HTTPException(
        status_code=502,
        detail="; ".join(errors) if errors else "Kobold response was empty",
    )


@app.get("/api/kobold/status")
def kobold_status(
    url: Optional[str] = Query(default=None, description="Base URL of Kobold server")
) -> Dict[str, Optional[str] | bool]:
    candidates = list(_candidate_bases(url))
    result: Dict[str, Optional[str] | bool] = {
        "url": candidates[0] if candidates else None,
        "online": False,
        "model": None,
    }
    errors = []
    for base in candidates:
        ok, model_name, error = _probe_model(base)
        if ok:
            result.update({"url": base, "online": True, "model": model_name})
            return result
        if error:
            errors.append(error)
    if errors:
        result["error"] = "; ".join(errors)
    return result


@app.post("/api/generate_prompt")
def generate_prompt(request: GenerateRequest) -> Dict[str, Optional[str]]:
    errors = []
    for base in _candidate_bases(request.kobold_url):
        try:
            return _call_kobold(base, request.idea)
        except HTTPException as exc:
            errors.append(f"{base}: {exc.detail}")
    raise HTTPException(status_code=502, detail="; ".join(errors))


__all__ = ["app"]
