#!/usr/bin/env python3
"""Passive stream capture helper for authorized media playback."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

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


def safe_text(text: object) -> str:
    """Return text that can always be encoded by the active stdout encoding."""
    value = str(text)
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


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
        try:
            browser = p.chromium.launch(headless=headless)
        except Exception as exc:
            message = str(exc)
            if "Executable doesn't exist" in message and "playwright install" in message:
                raise RuntimeError(
                    "Playwright browser binaries are missing. Run: python -m playwright install chromium"
                ) from exc
            raise
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
    parsed = urlparse(stream_url)
    referer = f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme and parsed.netloc else ""
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-overwrites",
        "--restrict-filenames",
        "--socket-timeout",
        "30",
        "--retries",
        "15",
        "--fragment-retries",
        "20",
        "--retry-sleep",
        "exp=1:8",
        "--concurrent-fragments",
        "8",
        "--hls-prefer-native",
        "--extractor-retries",
        "6",
        "--output",
        output_template,
        "--add-header",
        "User-Agent:Mozilla/5.0",
        stream_url,
    ]
    if referer:
        cmd[cmd.index(stream_url):cmd.index(stream_url)] = ["--add-header", f"Referer:{referer}"]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 30)
        output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        return completed.returncode == 0, output[-2000:]
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
        return False, f"yt-dlp timeout after 30m\n{output[-1800:]}"


def run_yt_dlp_resilient(stream_url: str, output_dir: Path) -> tuple[bool, str]:
    """Attempt yt-dlp with fallback argument profiles for hard-to-fetch streams."""
    output_template = str(output_dir / "%(title).120s_%(id)s.%(ext)s")
    strategies = [
        ["--hls-prefer-native"],
        ["--hls-use-mpegts", "--downloader", "ffmpeg"],
        ["--force-generic-extractor", "--hls-prefer-native"],
        ["--force-generic-extractor", "--hls-use-mpegts", "--downloader", "ffmpeg"],
    ]
    failures: list[str] = []
    for extra_args in strategies:
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-overwrites",
            "--restrict-filenames",
            "--socket-timeout",
            "30",
            "--retries",
            "18",
            "--fragment-retries",
            "24",
            "--extractor-retries",
            "8",
            "--retry-sleep",
            "exp=1:10",
            "--concurrent-fragments",
            "8",
            "--output",
            output_template,
            *extra_args,
            "--add-header",
            "User-Agent:Mozilla/5.0",
            stream_url,
        ]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 35)
            output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
            if completed.returncode == 0:
                return True, output[-2000:]
            failures.append(output[-450:] or "yt-dlp failed")
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
            failures.append(f"yt-dlp timeout after 35m\n{output[-420:]}")
    return False, "\n---\n".join(failures)[-2500:]


def run_ffmpeg_passthrough(stream_url: str, output_dir: Path) -> tuple[bool, str]:
    out_file = output_dir / f"{sanitize_filename(stream_url, 'stream_capture')}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-protocol_whitelist",
        "file,http,https,tcp,tls,crypto",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "8",
        "-i",
        stream_url,
        "-c",
        "copy",
        "-bsf:a",
        "aac_adtstoasc",
        str(out_file),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 25)
        output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        return completed.returncode == 0, output[-2000:]
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
        return False, f"ffmpeg copy timeout after 25m\n{output[-1800:]}"


def run_ffmpeg_reencode(stream_url: str, output_dir: Path) -> tuple[bool, str]:
    """Fallback for snippet/fragment sources that fail stream copy."""
    out_file = output_dir / f"{sanitize_filename(stream_url, 'stream_capture')}_reencode.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-protocol_whitelist",
        "file,http,https,tcp,tls,crypto",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "8",
        "-i",
        stream_url,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        str(out_file),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 30)
        output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        return completed.returncode == 0, output[-2000:]
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
        return False, f"ffmpeg reencode timeout after 30m\n{output[-1800:]}"


def run_curl_fetch(stream_url: str, output_dir: Path) -> tuple[bool, str]:
    """Last-resort direct fetch for segment/snippet style URLs."""
    curl_path = shutil.which("curl")
    if not curl_path:
        return False, "curl not installed"
    out_file = output_dir / f"{sanitize_filename(stream_url, 'stream_capture')}.bin"
    cmd = [
        curl_path,
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "1",
        "--connect-timeout",
        "20",
        "--output",
        str(out_file),
        stream_url,
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 15)
        output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        if completed.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
            return True, output[-2000:]
        return False, output[-2000:] or "curl failed"
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
        return False, f"curl timeout after 15m\n{output[-1800:]}"


def download_hits(hits: list[StreamHit], output_dir: Path, mode: str) -> tuple[int, int, list[str]]:
    debug: list[str] = []
    success = 0
    failed = 0

    targets = hits
    if mode == "fast":
        preferred = [hit for hit in hits if any(token in hit.url.lower() for token in (".m3u8", ".mpd", "videoplayback", "mime=video"))]
        targets = preferred or hits

    if mode == "elusive-sites":
        seen_urls: set[str] = set()
        ordered_targets: list[StreamHit] = []
        for hit in hits:
            if hit.url in seen_urls:
                continue
            ordered_targets.append(hit)
            seen_urls.add(hit.url)
        # Prioritize adaptive manifests first (often reference all snippets/fragments).
        targets = sorted(
            ordered_targets,
            key=lambda hit: 0 if any(token in hit.url.lower() for token in (".m3u8", ".mpd")) else 1,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        for hit in targets:
            if mode == "elusive-sites":
                futures.append(pool.submit(run_yt_dlp_resilient, hit.url, output_dir))
            else:
                futures.append(pool.submit(run_yt_dlp, hit.url, output_dir))

        for hit, future in zip(targets, futures):
            ok, msg = future.result()
            if ok:
                success += 1
                strategy = "yt-dlp resilient" if mode == "elusive-sites" else "yt-dlp"
                debug.append(f"[OK] {strategy} -> {hit.url}")
                continue
            ff_ok, ff_msg = run_ffmpeg_passthrough(hit.url, output_dir)
            if ff_ok:
                success += 1
                debug.append(f"[OK] ffmpeg copy -> {hit.url}")
            else:
                if mode == "elusive-sites":
                    reenc_ok, reenc_msg = run_ffmpeg_reencode(hit.url, output_dir)
                    if reenc_ok:
                        success += 1
                        debug.append(f"[OK] ffmpeg reencode -> {hit.url}")
                        continue
                    curl_ok, curl_msg = run_curl_fetch(hit.url, output_dir)
                    if curl_ok:
                        success += 1
                        debug.append(f"[OK] curl fallback -> {hit.url}")
                        continue
                    failed += 1
                    debug.append(
                        f"[FAIL] {hit.url}\n"
                        f"yt-dlp:\n{msg}\n"
                        f"ffmpeg-copy:\n{ff_msg}\n"
                        f"ffmpeg-reencode:\n{reenc_msg}\n"
                        f"curl:\n{curl_msg}"
                    )
                else:
                    failed += 1
                    debug.append(f"[FAIL] {hit.url}\nyt-dlp:\n{msg}\nffmpeg:\n{ff_msg}")

    return success, failed, debug


def main() -> int:
    parser = argparse.ArgumentParser(description="Passive stream detector and downloader (authorized use only).")
    parser.add_argument("--url", default="", help="Page URL to open for passive capture")
    parser.add_argument("--output", default="media_dump", help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["slow", "fast", "elusive-sites"],
        default="slow",
        help="slow: capture while playback happens, fast: aggressively fetch full stream, elusive-sites: multi-strategy recovery for hard streaming pages",
    )
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
        print(safe_text(f"Passive capture failed: {exc}"))
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
