#!/usr/bin/env python3
"""Privacy-first media harvester with multiple fallback download methods.

This tool is designed for authorized downloads only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import requests

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
MEDIA_EXTENSIONS = {
    ".mp4", ".webm", ".mkv", ".mov", ".m4v", ".avi", ".mp3", ".aac", ".m4a",
    ".wav", ".flac", ".ogg", ".opus", ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".bmp", ".svg", ".ts", ".m3u8",
}
TOOL_INSTALL_GUIDE = {
    "yt-dlp": ["python -m pip install --upgrade yt-dlp"],
    "ffmpeg": ["sudo apt-get install ffmpeg", "brew install ffmpeg", "winget install ffmpeg"],
    "curl": ["sudo apt-get install curl", "brew install curl", "winget install cURL.cURL"],
    "aria2c": ["sudo apt-get install aria2", "brew install aria2", "winget install aria2.aria2"],
    "gallery-dl": ["python -m pip install --upgrade gallery-dl"],
}


@dataclass(frozen=True)
class Candidate:
    url: str
    kind: str
    source: str


@dataclass(frozen=True)
class ToolStatus:
    name: str
    available: bool
    version: str
    install_hint: str


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def get_tool_version(name: str) -> str:
    commands = {
        "python": [sys.executable, "--version"],
        "yt-dlp": ["yt-dlp", "--version"],
        "ffmpeg": ["ffmpeg", "-version"],
        "curl": ["curl", "--version"],
        "aria2c": ["aria2c", "--version"],
        "gallery-dl": ["gallery-dl", "--version"],
    }
    cmd = commands.get(name)
    if not cmd:
        return "n/a"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        line = (proc.stdout or proc.stderr or "").strip().splitlines()
        return line[0][:120] if line else "unknown"
    except Exception:
        return "unknown"


def tool_statuses() -> list[ToolStatus]:
    tools = ["python", "yt-dlp", "ffmpeg", "curl", "aria2c", "gallery-dl"]
    statuses: list[ToolStatus] = []
    for tool in tools:
        available = True if tool == "python" else command_exists(tool)
        hints = TOOL_INSTALL_GUIDE.get(tool, ["Manual install required"])
        statuses.append(ToolStatus(tool, available, get_tool_version(tool) if available else "missing", " | ".join(hints)))
    return statuses


def print_tool_checklist() -> bool:
    print("\n=== Download tool checklist ===")
    statuses = tool_statuses()
    all_ok = True
    for status in statuses:
        marker = "OK" if status.available else "MISSING"
        print(f"- {status.name:<10} : {marker:<7} | {status.version}")
        if not status.available:
            all_ok = False
            print(f"  Install: {status.install_hint}")
    return all_ok


def auto_install_yt_dlp() -> bool:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return completed.returncode == 0
    except Exception:
        return False


def auto_install_gallery_dl() -> bool:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "gallery-dl"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return completed.returncode == 0
    except Exception:
        return False


def auto_install_missing() -> None:
    print("\n=== Auto-install attempt ===")
    if not command_exists("yt-dlp"):
        print("Attempting to install yt-dlp via pip...")
        print("yt-dlp install: OK" if auto_install_yt_dlp() else "yt-dlp install: FAILED")
    if not command_exists("gallery-dl"):
        print("Attempting to install gallery-dl via pip...")
        print("gallery-dl install: OK" if auto_install_gallery_dl() else "gallery-dl install: FAILED")


def build_session(proxy: str | None, timeout: float) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.request_timeout = timeout  # type: ignore[attr-defined]
    return session


def fetch_page(session: requests.Session, url: str, timeout: float) -> str:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def normalize_candidate(url: str, base_url: str) -> str:
    joined = urljoin(base_url, html.unescape(url.strip()))
    parsed = urlparse(joined)
    cleaned = parsed._replace(fragment="").geturl()
    return cleaned


def extract_from_html(page_html: str, base_url: str) -> list[Candidate]:
    candidates: list[Candidate] = []

    patterns = {
        "video/img tags": r"<(?:video|source|img|audio)[^>]+(?:src|data-src)=['\"]([^'\"]+)['\"]",
        "srcset": r"srcset=['\"]([^'\"]+)['\"]",
        "meta": r"<meta[^>]+content=['\"]([^'\"]+)['\"]",
        "json": r"https?://[^\s'\"<>]+",
        "hls": r"https?://[^\s'\"<>]+\.m3u8(?:\?[^'\"\s<>]*)?",
    }

    for source, pattern in patterns.items():
        for match in re.findall(pattern, page_html, flags=re.IGNORECASE):
            if source == "srcset":
                for chunk in match.split(","):
                    part = chunk.strip().split(" ")[0].strip()
                    if part:
                        candidates.append(Candidate(normalize_candidate(part, base_url), "media", source))
                continue
            candidates.append(Candidate(normalize_candidate(match, base_url), "media", source))

    filtered: list[Candidate] = []
    seen: set[str] = set()
    for item in candidates:
        lower_url = item.url.lower()
        if any(ext in lower_url for ext in MEDIA_EXTENSIONS) or "m3u8" in lower_url or "format=" in lower_url:
            if item.url not in seen:
                filtered.append(item)
                seen.add(item.url)
    return filtered


def resolution_hint(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    for key in ("res", "resolution", "quality", "height"):
        if key in qs and qs[key]:
            return qs[key][0]
    match = re.search(r"(\d{3,4}p)", url.lower())
    return match.group(1) if match else "unknown"


def safe_filename_from_url(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or fallback
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:180] or fallback


def probe_content_type(session: requests.Session, url: str, timeout: float) -> str | None:
    try:
        resp = session.head(url, timeout=timeout, allow_redirects=True)
        if resp.ok:
            return resp.headers.get("content-type")
    except requests.RequestException:
        return None
    return None


def looks_like_media(url: str, content_type: str | None) -> bool:
    lower = url.lower()
    if any(ext in lower for ext in MEDIA_EXTENSIONS):
        return True
    if content_type and any(kind in content_type for kind in ("image/", "video/", "audio/", "application/vnd.apple.mpegurl")):
        return True
    return False


def download_file(session: requests.Session, url: str, out_dir: Path, timeout: float) -> tuple[bool, str]:
    filename = safe_filename_from_url(url, "downloaded_media")
    target = out_dir / filename
    suffix_counter = 1
    while target.exists():
        target = out_dir / f"{target.stem}_{suffix_counter}{target.suffix}"
        suffix_counter += 1

    try:
        with session.get(url, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            if not target.suffix:
                guess = mimetypes.guess_extension(resp.headers.get("content-type", "").split(";")[0].strip())
                if guess:
                    target = target.with_suffix(guess)
            with open(target, "wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        out.write(chunk)
        return True, f"saved {target.name}"
    except Exception as exc:
        return False, f"direct failed: {exc}"


def run_yt_dlp(url: str, out_dir: Path, timeout: float, proxy: str | None, fmt: str, debug: bool, list_formats: bool) -> tuple[bool, str]:
    if not command_exists("yt-dlp"):
        return False, "yt-dlp not installed"

    output_template = str(out_dir / "%(title).120s_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp", "--no-overwrites", "--restrict-filenames", "--write-info-json", "--write-thumbnail",
        "--output", output_template,
    ]
    if list_formats:
        cmd.append("--list-formats")
    else:
        cmd.extend(["--format", fmt])
    if command_exists("ffmpeg"):
        cmd.extend(["--merge-output-format", "mp4"])
    if proxy:
        cmd.extend(["--proxy", proxy])
    if debug:
        cmd.extend(["--verbose", "--print", "after_move:filepath"])
    cmd.append(url)

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        if completed.returncode == 0:
            return True, output[-1200:] if output else "yt-dlp success"
        return False, output[-1200:] if output else "yt-dlp failed"
    except subprocess.TimeoutExpired:
        return False, "yt-dlp timeout"


def collect_candidates(session: requests.Session, url: str, timeout: float) -> list[Candidate]:
    page_html = fetch_page(session, url, timeout)
    return extract_from_html(page_html, url)


def preview(candidates: Iterable[Candidate], limit: int = 40) -> None:
    print("\n=== Media selector preview ===")
    for idx, item in enumerate(candidates):
        if idx >= limit:
            print(f"... and more ({idx + 1}+ entries)")
            break
        name = safe_filename_from_url(item.url, "media")
        res = resolution_hint(item.url)
        print(f"[{idx+1:03}] {name:<40} | res={res:<8} | {item.url}  ({item.source})")


def anonymize_notice(proxy: str | None) -> None:
    print("\n=== Privacy notice ===")
    print("This tool minimizes local metadata and supports proxy routing, but cannot guarantee complete anonymity.")
    if proxy:
        print(f"Proxy in use: {proxy}")
    else:
        print("No proxy configured. Consider --proxy socks5h://127.0.0.1:9050 with Tor.")


def write_debug_report(out_dir: Path, payload: dict) -> None:
    target = out_dir / "media_harvester_debug.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Debug report saved: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download media from a webpage with fallback strategies.")
    parser.add_argument("url", nargs="?", help="Target page URL")
    parser.add_argument("--output", default="media_dump", help="Output directory")
    parser.add_argument("--proxy", default=None, help="Proxy URL (example: socks5h://127.0.0.1:9050)")
    parser.add_argument("--timeout", type=float, default=45.0, help="Request timeout in seconds")
    parser.add_argument("--workers", type=int, default=6, help="Parallel download workers")
    parser.add_argument("--preview-only", action="store_true", help="Only discover and preview media")
    parser.add_argument("--include-page-with-ytdlp", action="store_true", help="Also run yt-dlp directly on page URL")
    parser.add_argument("--auto-install", action="store_true", help="Try to auto-install missing supported tools")
    parser.add_argument("--tool-check", action="store_true", help="Only print tool checklist and exit")
    parser.add_argument("--preview-limit", type=int, default=50, help="Preview line limit")
    parser.add_argument("--ytdlp-format", default="bestvideo*+bestaudio/best", help="Primary yt-dlp format selector")
    parser.add_argument("--fallback-formats", default="best/bv*+ba/b", help="Comma separated fallback yt-dlp formats")
    parser.add_argument("--max-downloads", type=int, default=0, help="Cap number of candidate downloads (0 = no cap)")
    parser.add_argument("--skip-direct", action="store_true", help="Skip direct HTTP file download attempts")
    parser.add_argument("--skip-ytdlp", action="store_true", help="Skip yt-dlp attempts")
    parser.add_argument("--list-formats", action="store_true", help="Use yt-dlp --list-formats for each URL")
    parser.add_argument("--debug", action="store_true", help="Enable verbose diagnostics and write debug report")
    args = parser.parse_args()

    if args.tool_check:
        all_ok = print_tool_checklist()
        if args.auto_install:
            auto_install_missing()
            all_ok = print_tool_checklist()
        return 0 if all_ok else 2

    if not args.url:
        print("No URL provided. Example:")
        print("python media_harvester.py https://example.com --include-page-with-ytdlp")
        return 1

    anonymize_notice(args.proxy)
    all_ok = print_tool_checklist()
    if not all_ok and args.auto_install:
        auto_install_missing()
        print_tool_checklist()

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session(proxy=args.proxy, timeout=args.timeout)
    try:
        candidates = collect_candidates(session, args.url, args.timeout)
    except Exception as exc:
        print(f"Failed to parse page: {exc}")
        candidates = []

    preview(candidates, limit=max(args.preview_limit, 1))
    print(f"\nDiscovered {len(candidates)} potential media URLs.")

    if args.max_downloads > 0:
        candidates = candidates[:args.max_downloads]
        print(f"Applying max-downloads cap: {len(candidates)}")

    if args.preview_only:
        return 0

    success_count = 0
    failure_count = 0
    failures: list[dict[str, str]] = []

    def worker(candidate: Candidate) -> tuple[str, bool, str]:
        content_type = probe_content_type(session, candidate.url, args.timeout)
        if not looks_like_media(candidate.url, content_type):
            return candidate.url, False, "not media"

        errors: list[str] = []
        if not args.skip_direct:
            ok, msg = download_file(session, candidate.url, out_dir, args.timeout)
            if ok:
                return candidate.url, True, msg
            errors.append(msg)

        if not args.skip_ytdlp:
            formats = [args.ytdlp_format] + [f.strip() for f in args.fallback_formats.split(",") if f.strip()]
            for fmt in formats:
                ok2, msg2 = run_yt_dlp(
                    candidate.url,
                    out_dir,
                    max(args.timeout * 2, 90),
                    args.proxy,
                    fmt=fmt,
                    debug=args.debug,
                    list_formats=args.list_formats,
                )
                if ok2:
                    return candidate.url, True, f"yt-dlp({fmt}) -> {msg2[-220:]}"
                errors.append(f"ytdlp({fmt}): {msg2[-220:]}")

        return candidate.url, False, "; ".join(errors) if errors else "no downloader strategy enabled"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        results = list(pool.map(worker, candidates)) if candidates else []

    for url, ok, message in results:
        if ok:
            success_count += 1
            print(f"[OK] {url} -> {message}")
        else:
            failure_count += 1
            failures.append({"url": url, "error": message})
            print(f"[FAIL] {url} -> {message}")

    if args.include_page_with_ytdlp and not args.skip_ytdlp:
        page_formats = [args.ytdlp_format] + [f.strip() for f in args.fallback_formats.split(",") if f.strip()]
        page_ok = False
        page_msg = ""
        for fmt in page_formats:
            ok, msg = run_yt_dlp(args.url, out_dir, max(args.timeout * 2, 90), args.proxy, fmt=fmt, debug=args.debug, list_formats=args.list_formats)
            if ok:
                page_ok, page_msg = True, f"yt-dlp({fmt}) -> {msg[-220:]}"
                break
            page_msg = msg
        if page_ok:
            success_count += 1
            print(f"[OK] page-url -> {page_msg}")
        else:
            failure_count += 1
            failures.append({"url": args.url, "error": f"page-url: {page_msg}"})
            print(f"[FAIL] page-url -> {page_msg}")

    print("\n=== Summary ===")
    print(f"Output folder : {out_dir}")
    print(f"Successful    : {success_count}")
    print(f"Failed        : {failure_count}")
    print("Use only on content you are allowed to download.")

    if args.debug:
        write_debug_report(
            out_dir,
            {
                "url": args.url,
                "output": str(out_dir),
                "candidate_count": len(candidates),
                "success_count": success_count,
                "failure_count": failure_count,
                "failures": failures,
                "tool_status": [status.__dict__ for status in tool_statuses()],
                "settings": {
                    "timeout": args.timeout,
                    "workers": args.workers,
                    "ytdlp_format": args.ytdlp_format,
                    "fallback_formats": args.fallback_formats,
                    "list_formats": args.list_formats,
                    "skip_direct": args.skip_direct,
                    "skip_ytdlp": args.skip_ytdlp,
                },
            },
        )

    return 0 if success_count > 0 or not candidates else 2


if __name__ == "__main__":
    sys.exit(main())
