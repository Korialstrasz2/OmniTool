#!/usr/bin/env python3
"""Privacy-first multi-strategy media downloader for a webpage.

This tool discovers media links from a page, previews them, and downloads
selected items with layered fallback methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from html.parser import HTMLParser

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
    "Referer": "",
}

COMMON_UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

HLS_HINTS = (".m3u8", ".ts", "application/vnd.apple.mpegurl", "application/x-mpegurl")
MEDIA_EXTS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg", ".avif", ".tif", ".tiff"},
    "video": {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".ts", ".m3u8"},
    "audio": {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".flac"},
}


@dataclass
class MediaCandidate:
    index: int
    url: str
    source: str
    media_type: str
    extension: str
    content_type: str = ""
    size_hint: Optional[int] = None


@dataclass
class DownloadResult:
    candidate_index: int
    url: str
    output_file: str
    method: str
    success: bool
    error: str = ""




class MediaHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.discovered: List[Tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        amap = {k.lower(): (v or "") for k, v in attrs}
        watch = {
            "img": ["src", "data-src", "srcset"],
            "video": ["src"],
            "source": ["src"],
            "audio": ["src"],
            "a": ["href"],
        }
        if tag not in watch:
            return
        for key in watch[tag]:
            val = amap.get(key, "").strip()
            if not val:
                continue
            if key == "srcset":
                for entry in val.split(","):
                    part = entry.strip().split(" ")[0]
                    if part:
                        self.discovered.append((part, f"{tag}[srcset]"))
            else:
                self.discovered.append((val, f"{tag}[{key}]"))

class MediaHarvester:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_root = Path(args.output).expanduser().resolve()
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.session.headers["User-Agent"] = args.user_agent or random.choice(COMMON_UA)
        self.session.verify = not args.insecure
        self.proxies = self._build_proxies(args.proxy)
        if self.proxies:
            self.session.proxies.update(self.proxies)

    @staticmethod
    def _build_proxies(proxy: str) -> Dict[str, str]:
        if not proxy:
            return {}
        return {"http": proxy, "https": proxy}

    def run_doctor(self) -> int:
        checks = {
            "python": shutil.which("python") or shutil.which("python3"),
            "yt-dlp": shutil.which("yt-dlp"),
            "ffmpeg": shutil.which("ffmpeg"),
            "curl": shutil.which("curl"),
            "wget": shutil.which("wget"),
            "gallery-dl": shutil.which("gallery-dl"),
        }
        print("== Launch Check (Doctor) ==")
        for name, path in checks.items():
            status = "OK" if path else "MISSING"
            print(f"- {name:<10} {status} {path or ''}")

        print("\nPrivacy note: this tool reduces metadata leakage (headers/proxy options),")
        print("but cannot guarantee complete anonymity against advanced tracking.")
        return 0

    def fetch_page(self, url: str) -> str:
        response = self.session.get(url, timeout=self.args.timeout)
        response.raise_for_status()
        return response.text

    def discover(self, page_url: str, html: str) -> List[MediaCandidate]:
        seen: Set[str] = set()
        discovered: List[Tuple[str, str]] = []

        parser = MediaHTMLParser()
        parser.feed(html)
        for raw, src in parser.discovered:
            discovered.append((urljoin(page_url, raw), src))

        script_urls = re.findall(r"https?://[^\"'\s)]+", html)
        for raw in script_urls:
            discovered.append((raw, "script-regex"))

        results: List[MediaCandidate] = []
        idx = 1
        for raw_url, source in discovered:
            normalized = self._normalize_url(raw_url)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            media_type, ext = self._classify(normalized)
            if media_type == "other" and not self.args.include_other:
                continue
            results.append(MediaCandidate(index=idx, url=normalized, source=source, media_type=media_type, extension=ext))
            idx += 1

        if self.args.probe:
            for item in results:
                self._probe_item(item)
        return results

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = url.strip()
        if not url or url.startswith("data:") or url.startswith("blob:"):
            return ""
        return url

    def _probe_item(self, item: MediaCandidate) -> None:
        try:
            head = self.session.head(item.url, timeout=self.args.timeout, allow_redirects=True)
            item.content_type = head.headers.get("Content-Type", "")
            clen = head.headers.get("Content-Length")
            if clen and clen.isdigit():
                item.size_hint = int(clen)
            if item.media_type == "other" and item.content_type:
                item.media_type = self._classify_from_content_type(item.content_type, item.extension)
        except Exception:
            return

    @staticmethod
    def _classify(url: str) -> Tuple[str, str]:
        path = urlparse(url).path.lower()
        ext = Path(path).suffix
        for media_type, extset in MEDIA_EXTS.items():
            if ext in extset:
                return media_type, ext
        if any(h in url.lower() for h in HLS_HINTS):
            return "video", ext or ".m3u8"
        return "other", ext

    @staticmethod
    def _classify_from_content_type(content_type: str, ext: str) -> str:
        c = content_type.lower()
        if c.startswith("image/"):
            return "image"
        if c.startswith("video/") or "mpegurl" in c:
            return "video"
        if c.startswith("audio/"):
            return "audio"
        return "other"

    @staticmethod
    def _human_size(size: Optional[int]) -> str:
        if not size:
            return "?"
        unit = ["B", "KB", "MB", "GB"]
        fsize = float(size)
        for u in unit:
            if fsize < 1024 or u == unit[-1]:
                return f"{fsize:.1f}{u}"
            fsize /= 1024
        return "?"

    def preview(self, items: List[MediaCandidate]) -> List[MediaCandidate]:
        if not items:
            print("No media candidates found.")
            return []

        print("\n== Media Preview ==")
        print(f"Found {len(items)} candidate URLs")
        for item in items:
            print(
                f"[{item.index:03}] {item.media_type:<6} {self._human_size(item.size_hint):>8} "
                f"{item.source:<12} {item.url}"
            )

        if self.args.non_interactive:
            return items

        print("\nSelect what to download:")
        print("- all")
        print("- comma list: 1,2,9")
        print("- ranges: 1-5,7,10-12")
        print("- type filters: type:image type:video type:audio")
        selection = input("> ").strip().lower()
        if not selection or selection == "all":
            return items

        if selection.startswith("type:"):
            wanted = {part.split(":", 1)[1] for part in selection.split() if ":" in part}
            return [item for item in items if item.media_type in wanted]

        chosen = self._parse_indices(selection, len(items))
        return [item for item in items if item.index in chosen]

    @staticmethod
    def _parse_indices(selection: str, max_index: int) -> Set[int]:
        out: Set[int] = set()
        for part in selection.split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                a, b = token.split("-", 1)
                if a.isdigit() and b.isdigit():
                    lo, hi = sorted((int(a), int(b)))
                    out.update(i for i in range(lo, hi + 1) if 1 <= i <= max_index)
            elif token.isdigit():
                value = int(token)
                if 1 <= value <= max_index:
                    out.add(value)
        return out

    def download_all(self, items: List[MediaCandidate]) -> List[DownloadResult]:
        folder = self._prepare_output_folder()
        print(f"\nOutput folder: {folder}")
        results: List[DownloadResult] = []
        for item in items:
            print(f"\n--> [{item.index}] {item.url}")
            result = self._download_with_fallbacks(item, folder)
            results.append(result)
            if result.success:
                print(f"    OK via {result.method}: {result.output_file}")
            else:
                print(f"    FAIL: {result.error}")

        manifest_path = folder / "manifest.json"
        payload = {
            "source": self.args.url,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "options": {
                "proxy": bool(self.args.proxy),
                "probe": self.args.probe,
                "non_interactive": self.args.non_interactive,
            },
            "downloads": [asdict(r) for r in results],
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote manifest: {manifest_path}")
        return results

    def _prepare_output_folder(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        host = urlparse(self.args.url).netloc.replace(":", "_") or "page"
        folder = self.output_root / f"media_{host}_{stamp}"
        for sub in ("images", "videos", "audio", "other"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        return folder

    def _download_with_fallbacks(self, item: MediaCandidate, folder: Path) -> DownloadResult:
        output_path = self._build_filename(item, folder)
        methods = [self._download_via_ytdlp, self._download_via_requests, self._download_via_ffmpeg, self._download_via_curl_or_wget]
        errors: List[str] = []
        for method in methods:
            ok, message = method(item, output_path)
            if ok:
                return DownloadResult(item.index, item.url, str(output_path), method.__name__, True)
            errors.append(f"{method.__name__}: {message}")
        return DownloadResult(item.index, item.url, str(output_path), "", False, " | ".join(errors))

    def _build_filename(self, item: MediaCandidate, folder: Path) -> Path:
        subfolder = {
            "image": "images",
            "video": "videos",
            "audio": "audio",
        }.get(item.media_type, "other")
        parsed = urlparse(item.url)
        basename = Path(parsed.path).name or "asset"
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", basename)[:140]
        if "." not in safe and item.extension:
            safe = f"{safe}{item.extension}"
        if "." not in safe:
            safe = f"{safe}.bin"
        digest = hashlib.sha1(item.url.encode("utf-8")).hexdigest()[:8]
        return folder / subfolder / f"{item.index:03}_{digest}_{safe}"

    def _download_via_ytdlp(self, item: MediaCandidate, output_path: Path) -> Tuple[bool, str]:
        if not shutil.which("yt-dlp"):
            return False, "yt-dlp not installed"
        template = str(output_path.with_suffix("")) + ".%(ext)s"
        cmd = [
            "yt-dlp",
            "--no-progress",
            "--no-playlist",
            "--restrict-filenames",
            "-o",
            template,
            item.url,
        ]
        if self.args.proxy:
            cmd.extend(["--proxy", self.args.proxy])
        return self._run_command(cmd)

    def _download_via_requests(self, item: MediaCandidate, output_path: Path) -> Tuple[bool, str]:
        try:
            with self.session.get(item.url, timeout=self.args.timeout, stream=True) as response:
                response.raise_for_status()
                ctype = response.headers.get("Content-Type", "")
                if item.media_type == "other" and ctype:
                    item.media_type = self._classify_from_content_type(ctype, item.extension)
                with open(output_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            fh.write(chunk)
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def _download_via_ffmpeg(self, item: MediaCandidate, output_path: Path) -> Tuple[bool, str]:
        if not shutil.which("ffmpeg"):
            return False, "ffmpeg not installed"
        if "m3u8" not in item.url and not any(item.url.lower().endswith(ext) for ext in (".ts",)):
            return False, "not hls/ts-like"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            item.url,
            "-c",
            "copy",
            str(output_path.with_suffix(".mp4")),
        ]
        return self._run_command(cmd)

    def _download_via_curl_or_wget(self, item: MediaCandidate, output_path: Path) -> Tuple[bool, str]:
        if shutil.which("curl"):
            cmd = ["curl", "-L", "--fail", "-o", str(output_path), item.url]
            if self.args.proxy:
                cmd.extend(["--proxy", self.args.proxy])
            ok, msg = self._run_command(cmd)
            if ok:
                return True, msg
        if shutil.which("wget"):
            cmd = ["wget", "-O", str(output_path), item.url]
            if self.args.proxy:
                cmd.extend(["-e", f"use_proxy=yes", "-e", f"https_proxy={self.args.proxy}"])
            return self._run_command(cmd)
        return False, "curl/wget not installed"

    @staticmethod
    def _run_command(cmd: List[str]) -> Tuple[bool, str]:
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True, (completed.stdout + completed.stderr).strip()
        except subprocess.CalledProcessError as exc:
            return False, ((exc.stdout or "") + (exc.stderr or "")).strip()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover, preview, and download media from a web page using fallback download strategies.",
    )
    parser.add_argument("url", nargs="?", help="Page URL to inspect")
    parser.add_argument("--output", default="downloads", help="Base output directory (default: downloads)")
    parser.add_argument("--proxy", default="", help="Proxy URL, e.g. socks5h://127.0.0.1:9050")
    parser.add_argument("--user-agent", default="", help="Override User-Agent header")
    parser.add_argument("--timeout", type=int, default=25, help="Network timeout in seconds")
    parser.add_argument("--probe", action="store_true", help="Probe candidate URLs with HEAD requests for content type/size")
    parser.add_argument("--include-other", action="store_true", help="Include unknown/non-media links")
    parser.add_argument("--non-interactive", action="store_true", help="Download everything without selection prompt")
    parser.add_argument("--doctor", action="store_true", help="Run installation and runtime checks, then exit")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    harvester = MediaHarvester(args)

    if args.doctor:
        return harvester.run_doctor()

    if not args.url:
        print("error: url is required unless --doctor is used", file=sys.stderr)
        return 2

    print("Privacy mode enabled: minimized headers, no cookie persistence, optional proxy routing.")
    print("Security note: no tool can guarantee complete anonymity or be fully virus-proof.")

    try:
        html = harvester.fetch_page(args.url)
        items = harvester.discover(args.url, html)
        selected = harvester.preview(items)
        if not selected:
            print("Nothing selected. Exiting.")
            return 0
        results = harvester.download_all(selected)
    except requests.RequestException as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        return 1

    success = sum(1 for r in results if r.success)
    print(f"\nCompleted: {success}/{len(results)} downloads successful")
    return 0 if success == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
