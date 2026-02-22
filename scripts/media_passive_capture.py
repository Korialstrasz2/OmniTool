#!/usr/bin/env python3
"""Passive stream capture helper for authorized media playback."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MEDIA_MARKERS = (
    ".m3u8",
    ".mpd",
    ".mp4",
    ".webm",
    "videoplayback",
    "mime=video",
    "audio/mp4",
)


@dataclass(frozen=True)
class StreamHit:
    url: str
    source: str


def looks_like_stream(url: str, content_type: str) -> bool:
    lowered = url.lower()
    if any(marker in lowered for marker in MEDIA_MARKERS):
        return True
    ctype = (content_type or "").lower()
    return any(token in ctype for token in ("video/", "audio/", "mpegurl", "dash+xml"))


def capture_with_playwright(url: str, capture_seconds: int, idle_seconds: int, headless: bool) -> list[StreamHit]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Playwright is required for passive capture. Install with: python -m pip install playwright && python -m playwright install chromium"
        ) from exc

    discovered: list[StreamHit] = []
    seen: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        last_new_at = time.time()

        def record(resource_url: str, source: str, content_type: str = "") -> None:
            nonlocal last_new_at
            if resource_url in seen:
                return
            if looks_like_stream(resource_url, content_type):
                seen.add(resource_url)
                discovered.append(StreamHit(resource_url, source))
                last_new_at = time.time()
                print(f"[captured] {resource_url}")

        page.on("request", lambda request: record(request.url, "request", request.headers.get("content-type", "")))
        page.on("response", lambda response: record(response.url, "response", response.headers.get("content-type", "")))

        print(f"Opening: {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        print("Page opened. Start playback in the browser if needed.")

        started = time.time()
        while time.time() - started < capture_seconds:
            page.wait_for_timeout(800)
            if discovered and (time.time() - last_new_at) >= idle_seconds:
                print(f"Idle timeout reached ({idle_seconds}s) after latest stream capture.")
                break

        browser.close()

    return discovered


def sanitize_filename(url: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", url.split("?")[0].split("/")[-1] or fallback)
    return cleaned[:100] or fallback


def run_yt_dlp(stream_url: str, output_dir: Path) -> tuple[bool, str]:
    output_template = str(output_dir / "%(title).120s_%(id)s.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-overwrites",
        "--restrict-filenames",
        "--output",
        output_template,
        stream_url,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 15)
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return completed.returncode == 0, output[-2000:]


def run_ffmpeg_passthrough(stream_url: str, output_dir: Path) -> tuple[bool, str]:
    out_file = output_dir / f"{sanitize_filename(stream_url, 'stream_capture')}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        stream_url,
        "-c",
        "copy",
        str(out_file),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 15)
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return completed.returncode == 0, output[-2000:]


def download_hits(hits: list[StreamHit], output_dir: Path, mode: str) -> tuple[int, int, list[str]]:
    debug: list[str] = []
    success = 0
    failed = 0

    targets = hits
    if mode == "fast":
        preferred = [hit for hit in hits if any(token in hit.url.lower() for token in (".m3u8", ".mpd", "videoplayback", "mime=video"))]
        targets = preferred or hits

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        for hit in targets:
            futures.append(pool.submit(run_yt_dlp, hit.url, output_dir))

        for hit, future in zip(targets, futures):
            ok, msg = future.result()
            if ok:
                success += 1
                debug.append(f"[OK] yt-dlp -> {hit.url}")
                continue
            ff_ok, ff_msg = run_ffmpeg_passthrough(hit.url, output_dir)
            if ff_ok:
                success += 1
                debug.append(f"[OK] ffmpeg copy -> {hit.url}")
            else:
                failed += 1
                debug.append(f"[FAIL] {hit.url}\nyt-dlp:\n{msg}\nffmpeg:\n{ff_msg}")

    return success, failed, debug


def main() -> int:
    parser = argparse.ArgumentParser(description="Passive stream detector and downloader (authorized use only).")
    parser.add_argument("--url", default="", help="Page URL to open for passive capture")
    parser.add_argument("--output", default="media_dump", help="Output directory")
    parser.add_argument("--mode", choices=["slow", "fast"], default="slow", help="slow: capture while playback happens, fast: aggressively fetch full stream from intercepted URLs")
    parser.add_argument("--capture-seconds", type=int, default=90, help="Maximum capture window")
    parser.add_argument("--idle-seconds", type=int, default=12, help="Stop after this idle period once stream URLs are found")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--auto-download", action="store_true", help="Immediately download captured stream URLs")
    args = parser.parse_args()

    if not args.url:
        print("No URL provided. Provide --url to enable passive capture.")
        return 1

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Authorized-use reminder: capture content you have rights to access/download.")
    print(f"Mode: {args.mode}")

    try:
        hits = capture_with_playwright(args.url, args.capture_seconds, args.idle_seconds, args.headless)
    except Exception as exc:
        print(f"Passive capture failed: {exc}")
        return 1

    print("\n=== Captured stream candidates ===")
    for idx, hit in enumerate(hits, start=1):
        print(f"[{idx:03}] {hit.url} ({hit.source})")
    print(f"Total captured: {len(hits)}")

    if not args.auto_download:
        print("Auto-download disabled. Re-run with --auto-download to fetch captured streams.")
        return 0

    success, failed, debug = download_hits(hits, out_dir, args.mode)
    print("\n=== Download summary ===")
    print(f"Success: {success}")
    print(f"Failed : {failed}")
    if debug:
        print("\n".join(debug))

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
