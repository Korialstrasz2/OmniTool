"""FastAPI server for ComfyUI prompt builder."""
from __future__ import annotations

import os
import platform
import string
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
from urllib.parse import urlparse

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
SYSTEM_PROMPT_PATH = (
    os.getenv("COMFY_SYSTEM_PROMPT")
    or str(BASE_DIR / "comfy_diffusion_image_prompt.txt")
)


def _load_system_prompt() -> str:
    path = Path(SYSTEM_PROMPT_PATH)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            pass
    return (
        "You are Flux, a seasoned prompt engineer for FLUX.1 text-to-image models. "
        "Expand any short, unstructured idea into a polished Flux prompt.\n\n"
        "Return a single text string—no JSON, bullet lists, code fences, or parameter blocks. "
        "Write a cohesive prompt that clearly covers four parts in order:\n"
        "1. Subject – Identify the focus of the image with vivid specificity.\n"
        "2. Scene / Environment – Describe the setting, atmosphere, or surrounding action.\n"
        "3. Style / Medium – Indicate the artistic treatment, camera approach, or medium.\n"
        "4. Technical modifiers – Add lighting, mood, lens, composition, texture, or resolution cues.\n\n"
        "Blend these parts into flowing sentences separated by periods. "
        "Never mention or infer samplers, CFG, steps, seeds, aspect ratios, or other generation parameters. "
        "Keep the language punchy, avoid repetition, and honour any artists, styles, or materials referenced by the user.\n\n"
        "Examples:\n"
        "Idea: \"flower. sunset\"\n"
        "Prompt:\n"
        "A single vivid wildflower standing tall in the foreground of a golden meadow at dusk. "
        "The horizon glows with warm orange and lavender clouds while distant hills fade into soft haze. "
        "Style: cinematic photography with shallow depth of field and tactile macro detail. "
        "Technical: golden hour glow, gentle lens flare, creamy bokeh, ultra crisp textures, 8k clarity.\n\n"
        "Idea: \"snowy mountain cabin\"\n"
        "Prompt:\n"
        "A hand-built log cabin with smoke curling from the chimney nestled beside frost-laden pines. "
        "Moonlight reflects off untouched alpine snowdrifts and a star-studded peak towering behind. "
        "Style: hyperreal winter landscape photography with subtle film grain. "
        "Technical: midnight blue palette, long exposure shimmer, wide-angle composition, crystalline highlights, 8k definition.\n\n"
        "Idea: \"retro-futuristic city skyline\"\n"
        "Prompt:\n"
        "A sweeping skyline of retro-futuristic skyscrapers crowned with chrome spires and neon billboards. "
        "Elevated highways weave between hovering transit pods above a sunset-lit harbor. "
        "Style: vibrant digital illustration blending art deco and synthwave aesthetics. "
        "Technical: magenta and cyan bloom lighting, atmospheric haze, cinematic framing, ultra detailed reflections, 32mm lens perspective."
    )


DEFAULT_SYSTEM_PROMPT = _load_system_prompt()

KOBOLD_HOST = os.getenv("KOBOLD_HOST", "http://127.0.0.1:5001").rstrip("/")
GGUF_ENV_PATHS = os.getenv("GGUF_SEARCH_PATHS") or os.getenv("GGUF_ROOTS")
GGUF_DEFAULT_CONTEXT = int(os.getenv("GGUF_CONTEXT", "4096"))
GGUF_THREADS = int(os.getenv("GGUF_THREADS", "0"))
GGUF_N_GPU_LAYERS = os.getenv("GGUF_N_GPU_LAYERS")
GGUF_N_GPU_LAYERS_INT = int(GGUF_N_GPU_LAYERS) if GGUF_N_GPU_LAYERS else None


class GenerateRequest(BaseModel):
    backend: str = "koboldcpp"
    kobold_url: Optional[str] = None
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


def _normalise_base_url(raw: Optional[str]) -> str:
    base = (raw or "").strip() or KOBOLD_HOST
    parsed = urlparse(base)
    if not parsed.scheme:
        base = f"http://{base}"
    return base.rstrip("/")


@app.get("/api/kobold/status")
def kobold_status(
    url: Optional[str] = Query(default=None, description="Base URL of KoboldCpp server")
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


def _format_messages_for_kobold(messages: List[Dict[str, str]]) -> str:
    system_parts: List[str] = []
    user_parts: List[str] = []
    assistant_parts: List[str] = []
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)
        else:
            user_parts.append(content)
    sections = []
    if system_parts:
        sections.append("System:\n" + "\n\n".join(system_parts))
    if user_parts:
        sections.append("User:\n" + "\n\n".join(user_parts))
    if assistant_parts:
        sections.append("Assistant:\n" + "\n\n".join(assistant_parts))
    sections.append("Assistant:\n")
    return "\n\n".join(sections)


def _call_kobold(request: GenerateRequest, messages: List[Dict[str, str]]) -> Dict[str, str]:
    base = _normalise_base_url(request.kobold_url)
    prompt = _format_messages_for_kobold(messages)
    payload: Dict[str, object] = {
        "prompt": prompt,
        "max_length": request.max_tokens or 600,
        "temperature": request.temperature if request.temperature is not None else 0.6,
        "top_p": request.top_p if request.top_p is not None else 0.9,
        "stream": False,
        "stop_sequence": ["\nUser:", "\nSystem:"],
    }
    if request.top_k is not None:
        payload["top_k"] = request.top_k
    if request.repeat_penalty is not None:
        payload["repetition_penalty"] = request.repeat_penalty
    if request.seed is not None:
        payload["seed"] = request.seed
    try:
        response = requests.post(
            f"{base}/api/v1/generate", json=payload, timeout=(10, 180)
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results")
        if not results:
            raise RuntimeError("Empty response from KoboldCpp")
        text = results[0].get("text", "")
        content = text.strip()
        if content.lower().startswith("assistant:"):
            content = content.split(":", 1)[1].strip()
        if not content:
            raise RuntimeError("KoboldCpp returned empty content")
        return {"backend": "koboldcpp", "model": data.get("model"), "content": content}
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - depends on external service
        raise HTTPException(status_code=502, detail=f"KoboldCpp error: {exc}")


@app.post("/api/generate_prompt")
def generate_prompt(request: GenerateRequest) -> Dict[str, str]:
    backend = (request.backend or "koboldcpp").lower()
    messages = _build_messages(request)
    if backend == "local":
        return local_models.generate(request, messages)
    if backend in {"kobold", "koboldcpp"}:
        return _call_kobold(request, messages)
    raise HTTPException(status_code=400, detail=f"Unknown backend '{request.backend}'")


__all__ = ["app"]
