#!/usr/bin/env python3
"""Privacy-first media discovery and downloader with multi-tool fallbacks.

This script is intentionally focused on legitimate archival/downloading use-cases.
It does not guarantee anonymity or safety against malicious content.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0 Safari/537.36"
)

MEDIA_ATTRS = {"src", "href", "data-src", "data-href", "poster", "content"}
MEDIA_HINT_RE = re.compile(r"(m3u8|mpd|mp4|webm|mov|mkv|mp3|aac|wav|flac|jpg|jpeg|png|gif|webp)", re.I)


@dataclass
class MediaItem:
    item_id: str
    source_url: str
    resolved_url: str
    media_type: str
    method_hint: str
    tag: str
    note: str = ""


class MediaLinkParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.found_urls: List[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        for attr in MEDIA_ATTRS:
            value = attr_map.get(attr)
            if value and MEDIA_HINT_RE.search(value):
                self.found_urls.append((tag, value))


def _safe_name(url: str, fallback: str = "download") -> str:
    parsed = urlparse(url)
    raw = Path(parsed.path).name or fallback
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    return stem[:120] or fallback


def _guess_media_type(url: str) -> str:
    lowered = url.lower()
    if any(ext in lowered for ext in [".m3u8", ".mpd"]):
        return "stream"
    if any(ext in lowered for ext in [".mp4", ".webm", ".mov", ".mkv"]):
        return "video"
    if any(ext in lowered for ext in [".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus"]):
        return "audio"
    if any(ext in lowered for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".svg"]):
        return "image"
    return "file"


def check_dependencies() -> Dict[str, Dict[str, str | bool]]:
    deps = {
        "python_requests": {"available": True, "path": "stdlib/pip"},
        "yt-dlp": {"available": bool(shutil.which("yt-dlp")), "path": shutil.which("yt-dlp") or ""},
        "ffmpeg": {"available": bool(shutil.which("ffmpeg")), "path": shutil.which("ffmpeg") or ""},
        "curl": {"available": bool(shutil.which("curl")), "path": shutil.which("curl") or ""},
        "wget": {"available": bool(shutil.which("wget")), "path": shutil.which("wget") or ""},
    }
    deps["summary"] = {
        "available": True,
        "path": "Core mode works with Python + requests; extra binaries improve fallback coverage.",
    }
    return deps


def fetch_page(url: str, timeout: float, user_agent: str, verify_tls: bool = True) -> str:
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
        "Referer": "",
    }
    resp = requests.get(url, headers=headers, timeout=timeout, verify=verify_tls)
    resp.raise_for_status()
    return resp.text


def preview_media(url: str, timeout: float = 20.0, user_agent: str = DEFAULT_USER_AGENT) -> Dict[str, object]:
    html = fetch_page(url, timeout=timeout, user_agent=user_agent)
    parser = MediaLinkParser(base_url=url)
    parser.feed(html)

    candidates: Dict[str, MediaItem] = {}
    for tag, raw_url in parser.found_urls:
        resolved = urljoin(url, raw_url)
        mtype = _guess_media_type(resolved)
        method_hint = "direct"
        if mtype in {"video", "audio", "stream"}:
            method_hint = "yt-dlp"
        item = MediaItem(
            item_id=f"html-{len(candidates)+1}",
            source_url=raw_url,
            resolved_url=resolved,
            media_type=mtype,
            method_hint=method_hint,
            tag=tag,
        )
        candidates[resolved] = item

    ytdlp_items = _extract_with_ytdlp(url)
    for item in ytdlp_items:
        candidates[item.resolved_url] = item

    return {
        "count": len(candidates),
        "items": [asdict(v) for v in candidates.values()],
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }


def _extract_with_ytdlp(url: str) -> List[MediaItem]:
    if not shutil.which("yt-dlp"):
        return []
    cmd = ["yt-dlp", "--dump-single-json", "--skip-download", url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout)
    except Exception:
        return []

    items: List[MediaItem] = []
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if isinstance(entries, list):
        iterable: Iterable[dict] = [e for e in entries if isinstance(e, dict)]
    else:
        iterable = [payload] if isinstance(payload, dict) else []

    for idx, entry in enumerate(iterable, start=1):
        webpage = entry.get("webpage_url") or url
        direct = entry.get("url") or entry.get("original_url") or webpage
        ext = (entry.get("ext") or "").lower()
        media_type = "video" if ext not in {"mp3", "aac", "wav", "flac", "opus"} else "audio"
        items.append(
            MediaItem(
                item_id=f"ytdlp-{idx}",
                source_url=webpage,
                resolved_url=direct,
                media_type=media_type,
                method_hint="yt-dlp",
                tag="yt-dlp",
                note=str(entry.get("title") or ""),
            )
        )
    return items


def _run(cmd: List[str], cwd: Optional[Path] = None) -> tuple[bool, str]:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None, check=True)
        return True, (completed.stdout + completed.stderr).strip()
    except subprocess.CalledProcessError as exc:
        return False, ((exc.stdout or "") + (exc.stderr or "")).strip()


def download_item(item: Dict[str, str], output_dir: Path, user_agent: str, timeout: float) -> Dict[str, object]:
    url = item["resolved_url"]
    media_type = item.get("media_type", "file")
    preferred = item.get("method_hint", "direct")
    filename = _safe_name(url)
    target = output_dir / filename

    attempts: List[str] = []

    if preferred == "yt-dlp" and shutil.which("yt-dlp"):
        out_tpl = str(output_dir / "%(title).120s-%(id)s.%(ext)s")
        ok, log = _run(["yt-dlp", "-o", out_tpl, "--no-playlist", url])
        attempts.append(f"yt-dlp: {'ok' if ok else 'fail'}")
        if ok:
            return {"ok": True, "url": url, "method": "yt-dlp", "output": out_tpl, "log": log[-1500:]}

    if media_type in {"stream", "video", "audio"} and shutil.which("ffmpeg"):
        ff_target = output_dir / f"{target.stem}.mp4"
        ok, log = _run([
            "ffmpeg", "-y", "-loglevel", "error", "-user_agent", user_agent,
            "-i", url, "-c", "copy", str(ff_target),
        ])
        attempts.append(f"ffmpeg: {'ok' if ok else 'fail'}")
        if ok:
            return {"ok": True, "url": url, "method": "ffmpeg", "output": str(ff_target), "log": log[-1500:]}

    ok_direct, log_direct = _direct_download(url, target, user_agent, timeout)
    attempts.append(f"requests: {'ok' if ok_direct else 'fail'}")
    if ok_direct:
        return {"ok": True, "url": url, "method": "requests", "output": str(target), "log": log_direct[-1500:]}

    if shutil.which("curl"):
        ok, log = _run(["curl", "-L", "-A", user_agent, "-o", str(target), url])
        attempts.append(f"curl: {'ok' if ok else 'fail'}")
        if ok:
            return {"ok": True, "url": url, "method": "curl", "output": str(target), "log": log[-1500:]}

    if shutil.which("wget"):
        ok, log = _run(["wget", "-U", user_agent, "-O", str(target), url])
        attempts.append(f"wget: {'ok' if ok else 'fail'}")
        if ok:
            return {"ok": True, "url": url, "method": "wget", "output": str(target), "log": log[-1500:]}

    return {
        "ok": False,
        "url": url,
        "method": "none",
        "attempts": attempts,
        "log": (log_direct if isinstance(log_direct, str) else "")[-2500:],
    }


def _direct_download(url: str, target: Path, user_agent: str, timeout: float) -> tuple[bool, str]:
    headers = {"User-Agent": user_agent, "DNT": "1", "Referer": ""}
    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").split(";")[0]
            if target.suffix == "":
                guess = mimetypes.guess_extension(content_type) or ""
                if guess:
                    target = target.with_suffix(guess)
            with open(target, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)
        return True, f"Saved to {target}"
    except Exception as exc:
        return False, str(exc)


def bulk_download(items: List[Dict[str, str]], output_dir: Path, user_agent: str, timeout: float) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [download_item(item, output_dir=output_dir, user_agent=user_agent, timeout=timeout) for item in items]
    success = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    return {
        "ok": True,
        "output_dir": str(output_dir),
        "downloaded": len(success),
        "failed": len(failed),
        "results": results,
    }


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Media harvester with preview + fallback download pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python scripts/media_harvester.py check
              python scripts/media_harvester.py preview --url https://example.com
              python scripts/media_harvester.py download --url https://example.com --out downloads/example
            """
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Show dependency readiness")

    preview = sub.add_parser("preview", help="Preview media URLs on a page")
    preview.add_argument("--url", required=True)
    preview.add_argument("--timeout", type=float, default=20.0)
    preview.add_argument("--user-agent", default=DEFAULT_USER_AGENT)

    dl = sub.add_parser("download", help="Preview and download everything in one go")
    dl.add_argument("--url", required=True)
    dl.add_argument("--out", required=True)
    dl.add_argument("--timeout", type=float, default=20.0)
    dl.add_argument("--user-agent", default=DEFAULT_USER_AGENT)

    return parser


def main() -> int:
    parser = make_arg_parser()
    args = parser.parse_args()

    if args.command == "check":
        print(json.dumps(check_dependencies(), indent=2))
        return 0

    if args.command == "preview":
        print(json.dumps(preview_media(url=args.url, timeout=args.timeout, user_agent=args.user_agent), indent=2))
        return 0

    if args.command == "download":
        preview = preview_media(url=args.url, timeout=args.timeout, user_agent=args.user_agent)
        out = bulk_download(
            items=preview["items"],
            output_dir=Path(args.out),
            user_agent=args.user_agent,
            timeout=args.timeout,
        )
        print(json.dumps({"preview": preview, "download": out}, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
