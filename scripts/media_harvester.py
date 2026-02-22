#!/usr/bin/env python3
"""Privacy-first media harvester with multiple fallback download methods.

This tool is designed for authorized downloads only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
MEDIA_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mkv",
    ".mov",
    ".m4v",
    ".avi",
    ".mp3",
    ".aac",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
    ".ts",
    ".m3u8",
}


@dataclass(frozen=True)
class Candidate:
    url: str
    kind: str
    source: str


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_launch_check(quiet: bool = False) -> bool:
    checks = {
        "python": True,
        "yt-dlp": command_exists("yt-dlp"),
        "ffmpeg": command_exists("ffmpeg"),
        "curl": command_exists("curl"),
    }
    if not quiet:
        print("\\n=== Launch check ===")
        for name, ok in checks.items():
            status = "OK" if ok else "MISSING"
            print(f"- {name:<8} : {status}")
        if not checks["yt-dlp"]:
            print("  ! yt-dlp missing: video extraction fallback is limited.")
        if not checks["ffmpeg"]:
            print("  ! ffmpeg missing: HLS merging may fail.")
    return all(checks.values())


def auto_install_yt_dlp() -> bool:
    """Attempt to install yt-dlp into the active Python environment."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0
    except Exception:
        return False


def auto_install_dependencies() -> dict[str, bool]:
    """Try to install recommended download dependencies."""
    results: dict[str, bool] = {}
    packages = ["yt-dlp", "ffmpeg-python"]
    for package in packages:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", package],
                capture_output=True,
                text=True,
            )
            results[package] = completed.returncode == 0
        except Exception:
            results[package] = False
    return results


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
        if any(ext in lower_url for ext in MEDIA_EXTENSIONS) or "m3u8" in lower_url:
            if item.url not in seen:
                filtered.append(item)
                seen.add(item.url)
    return filtered


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


def run_yt_dlp(url: str, out_dir: Path, timeout: float, proxy: str | None) -> tuple[bool, str]:
    if not command_exists("yt-dlp"):
        return False, "yt-dlp not installed"

    output_template = str(out_dir / "%(title).120s_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-overwrites",
        "--no-playlist",
        "--restrict-filenames",
        "--write-info-json",
        "--write-thumbnail",
        "--output",
        output_template,
        url,
    ]
    if command_exists("ffmpeg"):
        cmd.extend(["--merge-output-format", "mp4"])
    if proxy:
        cmd.extend(["--proxy", proxy])

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if completed.returncode == 0:
            return True, "yt-dlp success"
        return False, (completed.stderr or completed.stdout).strip()[-250:]
    except subprocess.TimeoutExpired:
        return False, "yt-dlp timeout"


def run_yt_dlp_all_resolutions(url: str, out_dir: Path, timeout: float, proxy: str | None) -> tuple[int, int, list[str]]:
    """Download every available video resolution by enumerating yt-dlp formats."""
    if not command_exists("yt-dlp"):
        return 0, 0, ["yt-dlp not installed"]

    list_cmd = ["yt-dlp", "--list-formats", url]
    if proxy:
        list_cmd.extend(["--proxy", proxy])

    listed = subprocess.run(list_cmd, capture_output=True, text=True, timeout=max(timeout, 60))
    if listed.returncode != 0:
        return 0, 1, [f"format listing failed: {(listed.stderr or listed.stdout).strip()[-300:]}"]

    lines = listed.stdout.splitlines()
    pattern = re.compile(r"^\s*(\S+)\s+\S+\s+(\d+x\d+|audio only)")
    selected_formats: dict[str, str] = {}
    debug_lines: list[str] = ["yt-dlp format listing:"]
    for line in lines:
        if line.strip():
            debug_lines.append(line)
        match = pattern.search(line)
        if not match:
            continue
        fmt_id = match.group(1)
        resolution = match.group(2)
        if resolution == "audio only":
            continue
        selected_formats[resolution] = fmt_id

    success = 0
    failed = 0
    for resolution, fmt_id in sorted(selected_formats.items()):
        output_template = str(out_dir / f"%(title).120s_%(id)s_{resolution}.%(ext)s")
        cmd = [
            "yt-dlp",
            "--no-overwrites",
            "--restrict-filenames",
            "--write-info-json",
            "--write-thumbnail",
            "-f",
            f"{fmt_id}+bestaudio/best",
            "--output",
            output_template,
            url,
        ]
        if command_exists("ffmpeg"):
            cmd.extend(["--merge-output-format", "mp4"])
        if proxy:
            cmd.extend(["--proxy", proxy])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(timeout, 120))
        if result.returncode == 0:
            success += 1
            debug_lines.append(f"[OK] {resolution} via format {fmt_id}")
        else:
            failed += 1
            err = (result.stderr or result.stdout).strip()[-220:]
            debug_lines.append(f"[FAIL] {resolution} via format {fmt_id}: {err}")

    if not selected_formats:
        failed += 1
        debug_lines.append("No video formats discovered from yt-dlp output.")
    return success, failed, debug_lines


def collect_candidates(session: requests.Session, url: str, timeout: float) -> list[Candidate]:
    page_html = fetch_page(session, url, timeout)
    return extract_from_html(page_html, url)


def preview(candidates: Iterable[Candidate], limit: int = 40) -> None:
    print("\\n=== Preview ===")
    for idx, item in enumerate(candidates):
        if idx >= limit:
            print(f"... and more ({idx + 1}+ entries)")
            break
        print(f"[{idx+1:03}] {safe_filename_from_url(item.url, 'media_item')} :: {item.url}  ({item.source})")


def anonymize_notice(proxy: str | None) -> None:
    print("\\n=== Privacy notice ===")
    print("This tool minimizes local metadata and supports proxy routing, but cannot guarantee complete anonymity.")
    if proxy:
        print(f"Proxy in use: {proxy}")
    else:
        print("No proxy configured. Consider --proxy socks5h://127.0.0.1:9050 with Tor.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download media from a webpage with fallback strategies.")
    parser.add_argument("url", nargs="?", help="Target page URL")
    parser.add_argument("--output", default="media_dump", help="Output directory")
    parser.add_argument("--proxy", default=None, help="Proxy URL (example: socks5h://127.0.0.1:9050)")
    parser.add_argument("--timeout", type=float, default=45.0, help="Request timeout in seconds")
    parser.add_argument("--workers", type=int, default=6, help="Parallel download workers")
    parser.add_argument("--preview-only", action="store_true", help="Only discover and preview media")
    parser.add_argument("--include-page-with-ytdlp", action="store_true", help="Also run yt-dlp directly on page URL")
    parser.add_argument("--skip-launch-check", action="store_true", help="Skip dependency check output")
    parser.add_argument("--auto-install", action="store_true", help="Try to auto-install missing yt-dlp")
    parser.add_argument("--install-dependencies", action="store_true", help="Install recommended downloader dependencies")
    parser.add_argument("--list-tools", action="store_true", help="Print a checklist of downloader tools")
    parser.add_argument("--diagnostics", action="store_true", help="Print detailed diagnostics and format information")
    parser.add_argument("--download-all-resolutions", action="store_true", help="Download each available video resolution with yt-dlp")
    args = parser.parse_args()

    if args.install_dependencies:
        install_results = auto_install_dependencies()
        print("\n=== Dependency installation ===")
        for pkg, ok in install_results.items():
            print(f"- {pkg:<14}: {'INSTALLED' if ok else 'FAILED'}")
        print("Note: ffmpeg binary may still need OS package manager install.")

    if args.list_tools:
        print("\n=== Tool checklist ===")
        print("- yt-dlp  :", "OK" if command_exists("yt-dlp") else "MISSING")
        print("- ffmpeg  :", "OK" if command_exists("ffmpeg") else "MISSING")
        print("- curl    :", "OK" if command_exists("curl") else "MISSING")
        print("- python  : OK")
        print("Install notes:")
        print("  * python -m pip install -U yt-dlp")
        print("  * Linux: sudo apt install ffmpeg")
        print("  * macOS: brew install ffmpeg")
        print("  * Windows: winget install Gyan.FFmpeg")

    if not args.url and not (args.install_dependencies or args.list_tools):
        print("No URL provided. Example:")
        print("python media_harvester.py https://example.com --include-page-with-ytdlp")
        return 1
    if not args.url:
        return 0

    anonymize_notice(args.proxy)
    if not args.skip_launch_check:
        all_ok = run_launch_check()
        if not all_ok and args.auto_install and not command_exists("yt-dlp"):
            print("Attempting auto-install of yt-dlp...")
            if auto_install_yt_dlp():
                print("yt-dlp installed successfully.")
            else:
                print("Auto-install failed. Continue with limited fallback support.")

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session(proxy=args.proxy, timeout=args.timeout)
    try:
        candidates = collect_candidates(session, args.url, args.timeout)
    except Exception as exc:
        print(f"Failed to parse page: {exc}")
        candidates = []

    preview(candidates)
    print(f"\\nDiscovered {len(candidates)} potential media URLs.")

    if args.preview_only:
        return 0

    success_count = 0
    failure_count = 0

    def worker(candidate: Candidate) -> tuple[str, bool, str]:
        content_type = probe_content_type(session, candidate.url, args.timeout)
        if not looks_like_media(candidate.url, content_type):
            return candidate.url, False, "not media"

        if candidate.url.lower().endswith(".m3u8") or "m3u8" in candidate.url.lower():
            ok, msg = run_yt_dlp(candidate.url, out_dir, args.timeout, args.proxy)
            if ok:
                return candidate.url, True, msg

        ok, msg = download_file(session, candidate.url, out_dir, args.timeout)
        if ok:
            return candidate.url, True, msg

        ok2, msg2 = run_yt_dlp(candidate.url, out_dir, args.timeout, args.proxy)
        return candidate.url, ok2, msg2 if ok2 else f"{msg}; ytdlp: {msg2}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        results = list(pool.map(worker, candidates)) if candidates else []

    for url, ok, message in results:
        if ok:
            success_count += 1
            print(f"[OK] {url} -> {message}")
        else:
            failure_count += 1
            print(f"[FAIL] {url} -> {message}")

    if args.include_page_with_ytdlp:
        ok, msg = run_yt_dlp(args.url, out_dir, max(args.timeout * 2, 90), args.proxy)
        if ok:
            success_count += 1
            print(f"[OK] page-url -> {msg}")
        else:
            failure_count += 1
            print(f"[FAIL] page-url -> {msg}")

    if args.download_all_resolutions:
        res_ok, res_fail, debug_lines = run_yt_dlp_all_resolutions(args.url, out_dir, max(args.timeout * 2, 120), args.proxy)
        success_count += res_ok
        failure_count += res_fail
        for line in debug_lines:
            print(line)

    if args.diagnostics:
        print("\n=== Diagnostics ===")
        print(f"Target URL     : {args.url}")
        print(f"Workers        : {args.workers}")
        print(f"Timeout        : {args.timeout}")
        print(f"Proxy          : {args.proxy or 'none'}")
        print(f"Candidates     : {len(candidates)}")
        if command_exists("yt-dlp"):
            probe = subprocess.run(["yt-dlp", "--list-formats", args.url], capture_output=True, text=True, timeout=max(args.timeout, 90))
            print("Format probe rc:", probe.returncode)
            print((probe.stdout or probe.stderr)[-4000:])

    print("\\n=== Summary ===")
    print(f"Output folder : {out_dir}")
    print(f"Successful    : {success_count}")
    print(f"Failed        : {failure_count}")
    print("Use only on content you are allowed to download.")
    return 0 if success_count > 0 or not candidates else 2


if __name__ == "__main__":
    sys.exit(main())
