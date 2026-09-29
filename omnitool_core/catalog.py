"""Validated, file-backed tool catalog; no execution during discovery."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
FLAG = re.compile(r"--[a-z][a-z0-9-]*\Z")
MAX_MANIFEST = 64 * 1024


def relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("Invalid relative path")
    parts = value.split("/")
    if "\\" in value or ":" in value or any(p in {"", ".", ".."} for p in parts):
        raise ValueError("Paths must be relative, portable, and traversal-free")
    if any(any(ord(c) < 32 for c in p) or p.endswith((" ", ".")) for p in parts):
        raise ValueError("Unsafe path")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(p.split(".")[0].upper() in reserved for p in parts):
        raise ValueError("Reserved filename")
    return PurePosixPath(value)


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict) or spec.get("schema_version", 1) != 1:
        raise ValueError("Unsupported tool manifest")
    if not ID.fullmatch(str(spec.get("id", ""))):
        raise ValueError("Invalid tool ID")
    for key, limit in (("name", 120), ("description", 1000), ("folder", 160)):
        if key == "name" and not spec.get(key):
            raise ValueError("Tool name is required")
        if not isinstance(spec.get(key, ""), str) or len(spec.get(key, "")) > limit:
            raise ValueError(f"Invalid {key}")
    result = dict(spec)
    result.setdefault("folder", "Unsorted")
    result.setdefault("description", "")
    result.setdefault("kind", "python")
    result.setdefault("status", "ready")
    result.setdefault("tags", [])
    result.setdefault("parameters", [])
    result.setdefault("args", [])
    result.setdefault("requires", [])
    result.setdefault("platforms", [])
    if result["kind"] not in {"python", "page"}:
        raise ValueError("Only Python tools and registered pages are supported")
    if result["kind"] == "python":
        entry = relative_path(result.get("entrypoint", ""))
        if entry.suffix != ".py":
            raise ValueError("Entrypoint must be a Python file")
    else:
        if result.get("page") not in {"/csv-editor", "/lyrics-embedder", "/media-harvester"}:
            raise ValueError("Unregistered tool page")
    if not isinstance(result["parameters"], list) or len(result["parameters"]) > 32:
        raise ValueError("Invalid parameter list")
    for key in ("args", "tags", "requires", "platforms"):
        if not isinstance(result[key], list) or len(result[key]) > 64 or not all(isinstance(x, str) and len(x) < 4096 for x in result[key]):
            raise ValueError(f"Invalid {key}")
    seen = set()
    for p in result["parameters"]:
        if not isinstance(p, dict) or not ID.fullmatch(str(p.get("name", ""))) or p["name"] in seen:
            raise ValueError("Invalid or duplicate parameter")
        seen.add(p["name"])
        if p.get("type", "text") not in {"text", "path", "integer", "boolean", "choice"}:
            raise ValueError("Unknown parameter type")
        if p.get("flag") and not FLAG.fullmatch(p["flag"]):
            raise ValueError("Invalid parameter flag")
        if p.get("type") == "boolean" and not p.get("flag"):
            raise ValueError("Boolean parameters require a flag")
        if p.get("type") == "choice" and (not isinstance(p.get("choices"), list) or not p["choices"]):
            raise ValueError("Choice parameters require choices")
    return result


def arguments(spec: dict[str, Any], values: dict[str, Any]) -> list[str]:
    if not isinstance(values, dict):
        raise ValueError("Parameters must be an object")
    known = {p["name"] for p in spec["parameters"]}
    if values.keys() - known:
        raise ValueError("Unknown parameter")
    args = list(spec.get("args", []))
    for p in spec["parameters"]:
        value = values.get(p["name"], p.get("default", ""))
        kind = p.get("type", "text")
        if kind == "boolean":
            if not isinstance(value, bool):
                if value == "":
                    value = False
                else:
                    raise ValueError(f"{p['name']} must be true or false")
            if value:
                args.append(p["flag"])
            continue
        if value is None or value == "":
            if p.get("required"):
                raise ValueError(f"{p['name']} is required")
            continue
        if kind == "integer":
            if isinstance(value, bool) or not re.fullmatch(r"-?\d+", str(value)):
                raise ValueError(f"{p['name']} must be an integer")
            number = int(value)
            if not p.get("min", -2**31) <= number <= p.get("max", 2**31 - 1):
                raise ValueError(f"{p['name']} is out of range")
            value = str(number)
        if not isinstance(value, str) or len(value) > 4096 or "\0" in value:
            raise ValueError("Invalid parameter value")
        if kind == "choice" and value not in p["choices"]:
            raise ValueError("Invalid choice")
        # Do not let a positional value turn into an option for the target script.
        if value.startswith("-") and not p.get("flag"):
            raise ValueError("Positional arguments cannot start with '-'")
        if p.get("flag"):
            args.append(p["flag"])
        args.append(value)
    return args


class Catalog:
    def __init__(self, base: Path):
        self.base = base.resolve()
        self.errors: list[str] = []

    def load(self) -> list[dict[str, Any]]:
        tools, seen, self.errors = [], set(), []
        sources: list[tuple[dict, Path]] = []
        try:
            sources.extend((x, self.base) for x in json.loads((self.base / "tools.json").read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            self.errors.append("Unable to read tools.json")
        root = self.base / "tools"
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(directory) / d).is_symlink())
            path = Path(directory) / "tool.json"
            if "tool.json" not in files:
                continue
            try:
                if path.is_symlink() or path.stat().st_size > MAX_MANIFEST:
                    raise ValueError("Invalid manifest file")
                sources.append((json.loads(path.read_text(encoding="utf-8")), path.parent))
            except (OSError, ValueError):
                self.errors.append(f"Invalid manifest: {path.relative_to(self.base)}")
        for raw, root in sources:
            try:
                tool = validate(raw)
                if tool["id"] in seen:
                    raise ValueError("Duplicate tool ID")
                seen.add(tool["id"])
                tool["_root"] = root
                if tool["kind"] == "python":
                    entry = (root / tool["entrypoint"]).resolve()
                    if not entry.is_relative_to(root.resolve()) or not entry.is_file():
                        raise ValueError("Entrypoint is missing or outside its tool directory")
                tool["availability"] = self.availability(tool)
                tools.append(tool)
            except (ValueError, TypeError, KeyError) as exc:
                self.errors.append(f"{raw.get('id', '?') if isinstance(raw, dict) else '?'}: {exc}")
        return tools

    @staticmethod
    def availability(tool: dict) -> str:
        if tool.get("disabled"):
            return "disabled"
        if tool.get("platforms") and sys.platform not in tool["platforms"]:
            return "unsupported-platform"
        for module in tool.get("requires", []):
            try:
                if importlib.util.find_spec(module) is None:
                    return "needs-setup"
            except (ImportError, ModuleNotFoundError, ValueError):
                return "needs-setup"
        return "ready"


def public_spec(tool: dict) -> dict:
    return {k: v for k, v in tool.items() if not k.startswith("_") and k not in {"entrypoint", "args", "requires"}}
