#!/usr/bin/env python3
"""Media Harvester: preview and download page media with multi-tool fallbacks."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import requests

MEDIA_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
    ".avif",
    ".mp4",
    ".webm",
    ".mkv",
    ".mov",
    ".avi",
    ".mp3",
    ".m4a",
    ".wav",
    ".ogg",
    ".flac",
    ".aac",
    ".m3u8",
    ".mpd",
    ".ts",
}

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
]


@dataclass
class MediaItem:
    index: int
    url: str
    media_type: str
    source: str


def has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(cmd: Sequence[str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, (proc.stdout + proc.stderr).strip()
    except subprocess.CalledProcessError as exc:
        return False, ((exc.stdout or "") + (exc.stderr or "")).strip()


def doctor(auto_install: bool) -> int:
    needed_python = ["requests", "bs4"]
    missing_py = []
    for mod in needed_python:
        try:
            __import__(mod if mod != "bs4" else "bs4")
        except Exception:
            missing_py.append(mod)

    cli_tools = ["yt-dlp", "ffmpeg", "gallery-dl", "aria2c"]
    status = {tool: has_tool(tool) for tool in cli_tools}

    print("=== Launch check ===")
    print("Python deps:")
    print("  requests: OK")
    print("  beautifulsoup4:", "OK" if "bs4" not in missing_py else "MISSING")
    print("CLI deps:")
    for tool, ok in status.items():
        print(f"  {tool}: {'OK' if ok else 'MISSING'}")

    if auto_install and missing_py:
        print("\nInstalling missing Python packages...")
        ok, out = run_cmd([sys.executable, "-m", "pip", "install", *missing_py])
        print(out)
        if not ok:
            return 1

    if missing_py:
        print("\nInstall missing Python packages with:")
        print(f"  {sys.executable} -m pip install {' '.join(missing_py)}")

    print(
        "\nPrivacy notice: this tool can reduce tracking signals (custom UA, optional SOCKS proxy/Tor), "
        "but cannot guarantee complete anonymity or malware-proof downloads."
    )
    return 0


def build_session(proxy: Optional[str], timeout: int) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": random.choice(UA_POOL), "Accept": "*/*"})
    if proxy:
        sess.proxies.update({"http": proxy, "https": proxy})
    sess.timeout = timeout
    return sess


def classify(url: str) -> str:
    path = urlparse(url).path.lower()
    ext = Path(path).suffix
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".avif"}:
        return "image"
    if ext in {".m3u8", ".mpd", ".ts"}:
        return "stream"
    if ext in {".mp4", ".webm", ".mkv", ".mov", ".avi"}:
        return "video"
    if ext in {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac"}:
        return "audio"
    return "unknown"


def extract_media_links(page_url: str, html: str) -> List[MediaItem]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("beautifulsoup4 is required. Run --doctor for setup steps.") from exc

    soup = BeautifulSoup(html, "html.parser")
    found: List[Tuple[str, str]] = []

    attrs = ["src", "href", "data-src", "data-original", "poster"]
    tags = ["img", "video", "audio", "source", "a"]
    for tag in tags:
        for element in soup.find_all(tag):
            for attr in attrs:
                value = element.get(attr)
                if not value:
                    continue
                absolute = urljoin(page_url, value)
                if Path(urlparse(absolute).path).suffix.lower() in MEDIA_EXTENSIONS:
                    found.append((absolute, f"{tag}[{attr}]"))

    regex_hits = re.findall(r"https?://[^\s\"'<>]+(?:\.m3u8|\.mpd|\.ts)", html)
    for hit in regex_hits:
        found.append((hit, "html-regex"))

    dedup: Dict[str, str] = {}
    for url, source in found:
        dedup.setdefault(url, source)

    items = [
        MediaItem(index=i + 1, url=url, media_type=classify(url), source=src)
        for i, (url, src) in enumerate(dedup.items())
    ]
    return items


def safe_filename(url: str, fallback: str) -> str:
    raw = Path(urlparse(url).path).name or fallback
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    return clean[:180] or fallback


def save_manifest(folder: Path, page_url: str, items: List[MediaItem]) -> Path:
    manifest = {
        "page_url": page_url,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "items": [asdict(item) for item in items],
    }
    out = folder / "media_preview.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def parse_selection(selection: str, max_index: int) -> List[int]:
    chosen: set[int] = set()
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            for i in range(int(start), int(end) + 1):
                if 1 <= i <= max_index:
                    chosen.add(i)
        else:
            i = int(part)
            if 1 <= i <= max_index:
                chosen.add(i)
    return sorted(chosen)


def download_with_requests(session: requests.Session, item: MediaItem, out_dir: Path, timeout: int) -> bool:
    filename = safe_filename(item.url, f"item_{item.index}")
    target = out_dir / filename
    with session.get(item.url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if target.suffix == "":
            ext = mimetypes.guess_extension(ctype.split(";")[0].strip() or "") or ""
            target = target.with_suffix(ext)
        with target.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=512 * 1024):
                if chunk:
                    f.write(chunk)
    return True


def try_download(item: MediaItem, out_dir: Path, session: requests.Session, timeout: int) -> Tuple[bool, str]:
    common_ydl = ["yt-dlp", "--no-progress", "--no-part", "-o", str(out_dir / "%(title)s.%(ext)s")]
    if has_tool("yt-dlp"):
        ok, out = run_cmd([*common_ydl, item.url])
        if ok:
            return True, "yt-dlp"

    if item.media_type in {"stream", "video", "audio"} and has_tool("ffmpeg"):
        target = out_dir / safe_filename(item.url, f"stream_{item.index}.mp4")
        ok, out = run_cmd(["ffmpeg", "-y", "-i", item.url, "-c", "copy", str(target)])
        if ok:
            return True, "ffmpeg"

    if item.media_type == "image" and has_tool("gallery-dl"):
        ok, out = run_cmd(["gallery-dl", "-d", str(out_dir), item.url])
        if ok:
            return True, "gallery-dl"

    try:
        download_with_requests(session, item, out_dir, timeout)
        return True, "requests"
    except Exception as exc:
        return False, f"requests failed: {exc}"


def preview(url: str, out_dir: Path, session: requests.Session, timeout: int) -> Tuple[List[MediaItem], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html = session.get(url, timeout=timeout).text
    items = extract_media_links(url, html)
    manifest = save_manifest(out_dir, url, items)
    print(f"Found {len(items)} media candidates")
    for item in items:
        print(f"[{item.index:03}] {item.media_type:7} {item.url} ({item.source})")
    print(f"Preview manifest: {manifest}")
    return items, manifest


def default_output(url: str) -> Path:
    host = (urlparse(url).hostname or "page").replace(".", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "downloads" / f"{host}_{stamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Professional media harvester with preview and fallback download engines.")
    parser.add_argument("url", nargs="?", help="Page URL to scan for media")
    parser.add_argument("--output", "-o", type=Path, help="Output folder (new folder is recommended)")
    parser.add_argument("--preview-only", action="store_true", help="Only scan and generate preview manifest")
    parser.add_argument("--select", help="Select indexes (e.g. 1,3-8). If omitted, downloads all found items.")
    parser.add_argument("--proxy", help="Proxy URL. Example: socks5h://127.0.0.1:9050 for Tor.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--doctor", action="store_true", help="Run dependency and privacy capability checks")
    parser.add_argument("--auto-install", action="store_true", help="Install missing Python packages during --doctor")

    args = parser.parse_args()

    if args.doctor:
        return doctor(args.auto_install)

    if not args.url:
        parser.error("url is required unless --doctor is used")

    out_dir = args.output or default_output(args.url)
    session = build_session(args.proxy, args.timeout)

    items, _ = preview(args.url, out_dir, session, args.timeout)
    if args.preview_only:
        return 0
    if not items:
        print("No downloadable media found.")
        return 0

    selected = parse_selection(args.select, len(items)) if args.select else [item.index for item in items]
    selected_items = [item for item in items if item.index in selected]
    print(f"\nDownloading {len(selected_items)} item(s) into {out_dir} ...")

    ok_count = 0
    for item in selected_items:
        success = False
        detail = ""
        for attempt in range(1, args.retries + 2):
            success, detail = try_download(item, out_dir, session, args.timeout)
            if success:
                ok_count += 1
                print(f"[OK] #{item.index} via {detail}")
                break
            time.sleep(min(2 * attempt, 5))
        if not success:
            print(f"[FAIL] #{item.index} {item.url} ({detail})")

    print(f"Completed: {ok_count}/{len(selected_items)} successful")
    print("Tip: use --proxy socks5h://127.0.0.1:9050 with Tor for stronger network privacy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
