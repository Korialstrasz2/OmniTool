"""FastAPI server for ComfyUI prompt builder."""
from __future__ import annotations

import os
import platform
import string
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:  # pragma: no cover - optional dependency for local models
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    Llama = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_SYSTEM_PROMPT = (
    "You are a prompt engineer for Stable Diffusion (ComfyUI). "
    "Transform a short idea into a clean SD/ComfyUI prompt.\n"
    "Output JSON with fields:\n"
    "{\n"
    '  "positive": "comma-separated descriptive prompt",\n'
    '  "negative": "things to avoid, artifacts",\n'
    '  "extras": { "style_tags": [], "sampler": "DPM++ 2M Karras", "cfg": 6.5, "steps": 28 }\n'
    "}\n"
    "Rules: concise, powerful tags; include subject, composition, lighting, lens/camera, style; "
    "avoid repetition; keep negative realistic. If the idea names a model, tag appropriately.\n"
    "Also include a flat string field 'prompt' equal to positive."
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
GGUF_ENV_PATHS = os.getenv("GGUF_SEARCH_PATHS") or os.getenv("GGUF_ROOTS")
GGUF_DEFAULT_CONTEXT = int(os.getenv("GGUF_CONTEXT", "4096"))
GGUF_THREADS = int(os.getenv("GGUF_THREADS", "0"))
GGUF_N_GPU_LAYERS = os.getenv("GGUF_N_GPU_LAYERS")
GGUF_N_GPU_LAYERS_INT = int(GGUF_N_GPU_LAYERS) if GGUF_N_GPU_LAYERS else None


class GenerateRequest(BaseModel):
    backend: str = "ollama"
    model: Optional[str] = None
    gguf_path: Optional[str] = None
    idea: str
    system: Optional[str] = None
    examples: Optional[str] = None
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    seed: Optional[int] = None
    max_tokens: Optional[int] = 600


class DirectoryListing(BaseModel):
    path: Optional[str]
    parent: Optional[str]
    directories: List[Dict[str, str]]
    files: List[Dict[str, str]]


@dataclass
class LocalModel:
    path: str
    llama: "Llama"


class LocalModelManager:
    """Load and reuse llama.cpp models on demand."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._current: Optional[LocalModel] = None

    @property
    def current_path(self) -> Optional[str]:
        return self._current.path if self._current else None

    def _load_model(self, model_path: Path) -> "Llama":
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it to use local GGUF models."
            )
        kwargs = {
            "model_path": str(model_path),
            "n_ctx": GGUF_DEFAULT_CONTEXT,
        }
        if GGUF_THREADS:
            kwargs["n_threads"] = GGUF_THREADS
        if GGUF_N_GPU_LAYERS_INT is not None:
            kwargs["n_gpu_layers"] = GGUF_N_GPU_LAYERS_INT
        return Llama(**kwargs)

    def get_model(self, model_path: str) -> "Llama":
        target = str(Path(model_path).expanduser().resolve())
        with self._lock:
            if self._current and self._current.path == target:
                return self._current.llama
            llama = self._load_model(Path(target))
            self._current = LocalModel(path=target, llama=llama)
            return llama

    def generate(self, request: GenerateRequest, messages: List[Dict[str, str]]) -> Dict[str, str]:
        llama = self.get_model(request.gguf_path or "")
        options: Dict[str, float | int] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.top_k is not None:
            options["top_k"] = request.top_k
        if request.repeat_penalty is not None:
            options["repeat_penalty"] = request.repeat_penalty
        if request.seed is not None:
            options["seed"] = request.seed
        if request.max_tokens is not None:
            options["max_tokens"] = request.max_tokens
        with self._lock:
            result = llama.create_chat_completion(messages=messages, **options)
        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError("Empty response from local model")
        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise RuntimeError("Local model returned empty content")
        return {
            "backend": "local",
            "model": self.current_path,
            "content": content,
        }


local_models = LocalModelManager()
app = FastAPI(title="Comfy Prompt Builder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def _build_messages(request: GenerateRequest) -> List[Dict[str, str]]:
    system_msg = request.system.strip() if request.system else DEFAULT_SYSTEM_PROMPT
    if request.examples:
        examples = request.examples.strip()
        if examples:
            system_msg += "\n\nFew-shot examples for style and format:\n" + examples
    idea = request.idea.strip()
    if not idea:
        raise HTTPException(status_code=400, detail="Idea cannot be empty")
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": idea},
    ]


def _get_roots() -> List[Path]:
    roots: List[Path] = []
    if GGUF_ENV_PATHS:
        for raw in GGUF_ENV_PATHS.split(os.pathsep):
            p = Path(raw).expanduser()
            if p.exists():
                roots.append(p)
    if not roots:
        home = Path.home()
        roots.append(home)
        if os.name != "nt":
            roots.append(Path("/"))
        else:
            for letter in string.ascii_uppercase:
                drive = Path(f"{letter}:\\")
                if drive.exists():
                    roots.append(drive)
    seen = set()
    unique_roots = []
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        unique_roots.append(resolved)
        seen.add(resolved)
    return unique_roots


@app.get("/api/ollama_models")
def list_ollama_models() -> Dict[str, List[str]]:
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [item.get("name") for item in data.get("models", []) if item.get("name")]
        return {"models": models}
    except Exception as exc:  # pragma: no cover - depends on external service
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama: {exc}")


@app.get("/api/local/status")
def local_status() -> Dict[str, Optional[str] | bool]:
    available = Llama is not None
    return {
        "available": available,
        "current_path": local_models.current_path if available else None,
        "context": GGUF_DEFAULT_CONTEXT,
        "threads": GGUF_THREADS,
        "n_gpu_layers": GGUF_N_GPU_LAYERS_INT,
        "implementation": platform.platform(),
    }


@app.get("/api/filetree", response_model=DirectoryListing)
def browse_filetree(path: Optional[str] = Query(default=None, description="Directory to list")) -> DirectoryListing:
    if path:
        target = Path(path).expanduser()
        if not target.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        target = target.resolve()
        if not target.is_dir():
            raise HTTPException(status_code=400, detail="Path must be a directory")
        directories: List[Dict[str, str]] = []
        files: List[Dict[str, str]] = []
        try:
            for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if entry.is_dir():
                    directories.append({"name": entry.name, "path": str(entry.resolve())})
                elif entry.is_file() and entry.suffix.lower() == ".gguf":
                    files.append({"name": entry.name, "path": str(entry.resolve())})
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=f"Permission denied: {exc}")
        parent = str(target.parent) if target.parent != target else None
        return DirectoryListing(
            path=str(target),
            parent=parent,
            directories=directories,
            files=files,
        )
    roots = [
        {"name": str(root), "path": str(root)}
        for root in _get_roots()
    ]
    return DirectoryListing(path=None, parent=None, directories=roots, files=[])


def _call_ollama(request: GenerateRequest, messages: List[Dict[str, str]]) -> Dict[str, str]:
    if not request.model:
        raise HTTPException(status_code=400, detail="Model name is required for Ollama requests")
    payload = {
        "model": request.model,
        "messages": messages,
        "stream": False,
        "options": {
            key: value
            for key, value in {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repeat_penalty": request.repeat_penalty,
                "seed": request.seed,
            }.items()
            if value is not None
        },
    }
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat", json=payload, timeout=(10, 180)
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Empty content from Ollama")
        return {"backend": "ollama", "model": request.model, "content": content}
    except Exception as exc:  # pragma: no cover - depends on external service
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}")


@app.post("/api/generate_prompt")
def generate_prompt(request: GenerateRequest) -> Dict[str, str]:
    backend = (request.backend or "ollama").lower()
    messages = _build_messages(request)
    if backend == "local":
        return local_models.generate(request, messages)
    if backend == "ollama":
        return _call_ollama(request, messages)
    raise HTTPException(status_code=400, detail=f"Unknown backend '{request.backend}'")


__all__ = ["app"]
