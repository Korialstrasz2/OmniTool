#!/usr/bin/env python3
"""Privacy-first media harvester with multiple fallback download methods.

This tool is designed for authorized downloads only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import importlib.util
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULTS_FILE = Path(__file__).with_name("media_harvester_defaults.json")

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
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
}

SENSITIVE_QUERY_KEYS = {
    "token",
    "access_token",
    "auth",
    "authorization",
    "session",
    "sessionid",
    "sid",
    "key",
    "apikey",
    "api_key",
    "password",
    "passwd",
    "email",
    "phone",
}

SOCIAL_REFERER = "https://www.instagram.com/"


@dataclass(frozen=True)
class Candidate:
    url: str
    kind: str
    source: str


def normalize_target_url(raw_url: str, site_mode: str) -> tuple[str, str | None]:
    """Normalize user-supplied URL for site-specific known patterns."""
    normalized = raw_url.strip()
    if site_mode != "instagram-public":
        return normalized, None

    parsed = urlparse(normalized)
    host = parsed.netloc.lower()
    if not host.endswith("instagram.com"):
        return normalized, None

    match = re.match(r"^/popular/([^/?#]+)/?$", parsed.path, flags=re.IGNORECASE)
    if match:
        username = match.group(1)
        cleaned_username = re.sub(r"[^a-zA-Z0-9._]", "", username)
        if cleaned_username:
            username = cleaned_username
        rewritten = parsed._replace(path=f"/{username}/", query="", fragment="").geturl()
        return rewritten, f"Normalized Instagram URL to profile path: {rewritten}"

    profile_match = re.match(r"^/([^/?#]+)/?$", parsed.path, flags=re.IGNORECASE)
    if profile_match:
        raw_username = profile_match.group(1)
        if raw_username.lower() not in {"p", "reel", "explore", "stories", "tv"}:
            cleaned_username = re.sub(r"[^a-zA-Z0-9._]", "", raw_username)
            if cleaned_username and cleaned_username != raw_username:
                rewritten = parsed._replace(path=f"/{cleaned_username}/", query="", fragment="").geturl()
                return rewritten, f"Normalized Instagram username to valid charset: {rewritten}"
    return normalized, None


def load_defaults() -> dict[str, object]:
    if not DEFAULTS_FILE.exists():
        return {}
    try:
        data = json.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_defaults(values: dict[str, object]) -> tuple[bool, str]:
    try:
        DEFAULTS_FILE.write_text(json.dumps(values, indent=2), encoding="utf-8")
        return True, f"defaults saved to {DEFAULTS_FILE}"
    except Exception as exc:
        return False, f"failed to save defaults: {exc}"


def bool_default(defaults: dict[str, object], key: str, fallback: bool) -> bool:
    value = defaults.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return fallback


def collect_candidates_from_scrolling(url: str, timeout: float, proxy: str | None) -> list[Candidate]:
    """Capture media-like URLs discovered while auto-scrolling a page.

    Uses Playwright when installed. Falls back to an empty list if unavailable.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("Playwright is not installed; skipping all-scroll discovery.")
        return []

    discovered: list[Candidate] = []
    seen: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        if proxy:
            print("Note: Playwright scrolling currently ignores proxy setting.")
        page = context.new_page()

        def capture(resource_url: str, source: str) -> None:
            normalized = normalize_candidate(resource_url, url)
            lower = normalized.lower()
            if normalized in seen:
                return
            if any(ext in lower for ext in MEDIA_EXTENSIONS) or any(x in lower for x in ("/image", "/video", "/audio", ".pdf")):
                discovered.append(Candidate(normalized, "media", source))
                seen.add(normalized)

        page.on("response", lambda r: capture(r.url, "scroll-response"))
        page.goto(url, wait_until="networkidle", timeout=int(timeout * 1000))

        same_height_count = 0
        previous_height = 0
        for _ in range(30):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(900)
            height = page.evaluate("document.body.scrollHeight")
            if height == previous_height:
                same_height_count += 1
            else:
                same_height_count = 0
            previous_height = height
            if same_height_count >= 3:
                break

        html_content = page.content()
        for item in extract_from_html(html_content, url):
            if item.url not in seen:
                discovered.append(item)
                seen.add(item.url)

        browser.close()

    return discovered


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def yt_dlp_command() -> list[str] | None:
    """Resolve an executable command for yt-dlp in this runtime.

    Some environments install yt-dlp as a Python module without exposing a
    `yt-dlp` binary on PATH, so we support both entrypoint and module modes.
    """
    binary = shutil.which("yt-dlp")
    if binary:
        return [binary]
    if importlib.util.find_spec("yt_dlp"):
        return [sys.executable, "-m", "yt_dlp"]
    return None


def yt_dlp_available() -> bool:
    cmd = yt_dlp_command()
    if not cmd:
        return False
    try:
        probe = subprocess.run(cmd + ["--version"], capture_output=True, text=True, timeout=15)
        return probe.returncode == 0
    except Exception:
        return False


def run_launch_check(quiet: bool = False) -> bool:
    checks = {
        "python": True,
        "yt-dlp": yt_dlp_available(),
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
        if completed.returncode == 0 and yt_dlp_available():
            return True
        # Fallback for restricted PATH systems.
        user_install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--user", "yt-dlp"],
            capture_output=True,
            text=True,
        )
        return user_install.returncode == 0 and yt_dlp_available()
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
    # Verify yt-dlp can actually execute from this runtime, not just pip install success.
    results["yt-dlp-runtime-check"] = yt_dlp_available()
    return results


def auto_install_yt_dlp_impersonation() -> bool:
    """Install yt-dlp impersonation extras when Cloudflare/challenge pages block requests."""
    commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp[curl-cffi]"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "--user", "yt-dlp[curl-cffi]"],
    ]
    for cmd in commands:
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode == 0:
                return True
        except Exception:
            continue
    return False


def list_impersonate_targets() -> list[str]:
    """Return yt-dlp impersonation targets available in this runtime."""
    cmd_prefix = yt_dlp_command()
    if not cmd_prefix:
        return []
    try:
        completed = subprocess.run(
            cmd_prefix + ["--list-impersonate-targets"],
            capture_output=True,
            text=True,
            timeout=25,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    targets: list[str] = []
    valid_target = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    likely_prefixes = (
        "chrome",
        "chromium",
        "edge",
        "safari",
        "firefox",
        "opera",
        "brave",
        "vivaldi",
        "ios",
        "android",
    )
    for line in completed.stdout.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue

        # yt-dlp may prefix rows with markers like "[info]" or list bullets.
        if cleaned.startswith("["):
            cleaned = cleaned.split("]", 1)[-1].strip()
        if cleaned.startswith(("-", "*")):
            cleaned = cleaned[1:].strip()

        if not cleaned:
            continue

        token = cleaned.split()[0].strip().strip(",")
        lowered = token.lower()
        if lowered in {"available", "targets", "impersonate"}:
            continue
        if valid_target.match(token) and lowered.startswith(likely_prefixes):
            targets.append(token)
    # Keep stable order while deduplicating.
    return list(dict.fromkeys(targets))


def build_session(proxy: str | None, timeout: float, site_mode: str = "general") -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if site_mode == "instagram-public":
        session.headers.update(
            {
                "Referer": SOCIAL_REFERER,
                "Origin": SOCIAL_REFERER.rstrip("/"),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
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
    cleaned = html.unescape(url.strip())
    cleaned = (
        cleaned.replace("\\/", "/")
        .replace("\\u0026", "&")
        .replace("\\u002F", "/")
        .replace("\\u003D", "=")
    )
    joined = urljoin(base_url, cleaned)
    parsed = urlparse(joined)
    cleaned = parsed._replace(fragment="").geturl()
    return cleaned


def sanitize_url_for_logs(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if not parsed.query:
        return raw_url
    sanitized: list[str] = []
    for item in parsed.query.split("&"):
        if "=" not in item:
            sanitized.append(item)
            continue
        key, value = item.split("=", 1)
        if key.lower() in SENSITIVE_QUERY_KEYS:
            sanitized.append(f"{key}=REDACTED")
        else:
            sanitized.append(f"{key}={value}")
    return parsed._replace(query="&".join(sanitized)).geturl()


def url_violates_privacy_policy(raw_url: str) -> bool:
    parsed = urlparse(raw_url)
    if parsed.username or parsed.password:
        return True
    for item in parsed.query.split("&"):
        if "=" not in item:
            continue
        key, _value = item.split("=", 1)
        if key.lower() in SENSITIVE_QUERY_KEYS:
            return True
    return False


def extract_from_html(page_html: str, base_url: str, include_all_html_media: bool = False) -> list[Candidate]:
    candidates: list[Candidate] = []

    patterns = {
        "video/img tags": r"<(?:video|source|img|audio)[^>]+(?:src|data-src)=['\"]([^'\"]+)['\"]",
        "srcset": r"srcset=['\"]([^'\"]+)['\"]",
        "meta": r"<meta[^>]+content=['\"]([^'\"]+)['\"]",
        "json": r"https?://[^\s'\"<>]+",
        "json-escaped": r"https?:\\\\/\\\\/[^\s'\"<>]+",
        "hls": r"https?://[^\s'\"<>]+\.m3u8(?:\?[^'\"\s<>]*)?",
        "css-url": r"url\((['\"]?)(https?://[^)'\"]+)\1\)",
        "link-href": r"<link[^>]+href=['\"]([^'\"]+)['\"]",
    }

    for source, pattern in patterns.items():
        for match in re.findall(pattern, page_html, flags=re.IGNORECASE):
            if isinstance(match, tuple):
                match = match[-1]
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
        if "static.cdninstagram.com/rsrc.php/" in lower_url:
            continue
        media_match = any(ext in lower_url for ext in MEDIA_EXTENSIONS) or "m3u8" in lower_url
        if include_all_html_media and (
            any(token in lower_url for token in ("/image", "/video", "/audio"))
            or any(ext in lower_url for ext in (".avif", ".heic", ".jfif"))
        ):
            media_match = True
        if media_match:
            if item.url not in seen:
                filtered.append(item)
                seen.add(item.url)
    return filtered


def extract_instagram_post_candidates(page_html: str, profile_url: str) -> list[Candidate]:
    """Extract Instagram post/reel/tv URLs from a profile page payload."""
    parsed = urlparse(profile_url)
    profile_root = parsed._replace(path="/", params="", query="", fragment="").geturl().rstrip("/")
    patterns = [
        r"https?://(?:www\.)?instagram\.com/(?:p|reel|tv)/[A-Za-z0-9_-]+/?",
        r"/(?:p|reel|tv)/[A-Za-z0-9_-]+/?",
        r'"shortcode"\s*:\s*"([A-Za-z0-9_-]+)"',
    ]

    discovered: list[Candidate] = []
    seen: set[str] = set()

    for full_url in re.findall(patterns[0], page_html, flags=re.IGNORECASE):
        normalized = normalize_candidate(full_url, profile_url)
        if normalized not in seen:
            discovered.append(Candidate(normalized, "instagram-post", "instagram-profile-html"))
            seen.add(normalized)

    for partial in re.findall(patterns[1], page_html, flags=re.IGNORECASE):
        normalized = normalize_candidate(urljoin(profile_root + "/", partial.lstrip("/")), profile_url)
        if normalized not in seen:
            discovered.append(Candidate(normalized, "instagram-post", "instagram-profile-html"))
            seen.add(normalized)

    for shortcode in re.findall(patterns[2], page_html, flags=re.IGNORECASE):
        normalized = normalize_candidate(f"{profile_root}/p/{shortcode}/", profile_url)
        if normalized not in seen:
            discovered.append(Candidate(normalized, "instagram-post", "instagram-shortcode"))
            seen.add(normalized)

    return discovered


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
    if content_type and any(kind in content_type for kind in ("image/", "video/", "audio/", "application/vnd.apple.mpegurl", "application/pdf")):
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
        if command_exists("curl"):
            curl_name = safe_filename_from_url(url, "downloaded_media")
            curl_target = out_dir / curl_name
            if not curl_target.suffix:
                curl_target = curl_target.with_suffix(".bin")
            curl_cmd = ["curl", "-L", "--fail", "--silent", "--show-error", "--output", str(curl_target), url]
            try:
                curl_completed = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=max(timeout, 90))
                if curl_completed.returncode == 0 and curl_target.exists() and curl_target.stat().st_size > 0:
                    return True, f"saved {curl_target.name} (curl fallback)"
                curl_err = (curl_completed.stderr or curl_completed.stdout).strip()[-180:]
                return False, f"direct failed: {exc}; curl failed: {curl_err}"
            except Exception as curl_exc:
                return False, f"direct failed: {exc}; curl failed: {curl_exc}"
        return False, f"direct failed: {exc}"


def run_ffmpeg_hls(url: str, out_dir: Path, timeout: float, proxy: str | None) -> tuple[bool, str]:
    if not command_exists("ffmpeg"):
        return False, "ffmpeg not installed"
    target = out_dir / f"{safe_filename_from_url(url, 'stream')}.mp4"
    if target.exists():
        target = out_dir / f"{target.stem}_{int(time.time())}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        url,
        "-c",
        "copy",
        str(target),
    ]
    env = os.environ.copy()
    if proxy:
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=max(timeout, 120), env=env)
        if completed.returncode == 0 and target.exists() and target.stat().st_size > 0:
            return True, f"saved {target.name} (ffmpeg hls fallback)"
        return False, (completed.stderr or completed.stdout).strip()[-220:]
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timeout"


def run_yt_dlp(url: str, out_dir: Path, timeout: float, proxy: str | None) -> tuple[bool, str]:
    cmd_prefix = yt_dlp_command()
    if not cmd_prefix:
        return False, "yt-dlp not installed"

    output_template = str(out_dir / "%(title).120s_%(id)s.%(ext)s")
    cmd = cmd_prefix + [
        "--no-overwrites",
        "--no-playlist",
        "--socket-timeout",
        str(max(timeout, 20)),
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
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=max(timeout, 180))
        if completed.returncode == 0:
            return True, "yt-dlp success"
        return False, (completed.stderr or completed.stdout).strip()[-250:]
    except subprocess.TimeoutExpired:
        return False, "yt-dlp timeout"


def run_yt_dlp_with_strategies(
    url: str,
    out_dir: Path,
    timeout: float,
    proxy: str | None,
    access_strategy: str,
    site_mode: str,
    impersonation_targets: list[str] | None = None,
    allow_auto_install: bool = True,
) -> tuple[bool, str]:
    """Run yt-dlp with progressive, authorized-access strategies.

    These strategies are intended for publicly accessible or otherwise
    authorized content only. They do not attempt account takeover,
    paywall bypass, or bot-protection circumvention.
    """
    cmd_prefix = yt_dlp_command()
    if not cmd_prefix:
        return False, "yt-dlp not installed"

    output_template = str(out_dir / "%(title).120s_%(id)s.%(ext)s")
    base_cmd = cmd_prefix + [
        "--no-overwrites",
        "--no-playlist",
        "--restrict-filenames",
        "--write-info-json",
        "--write-thumbnail",
        "--socket-timeout",
        str(max(timeout, 20)),
        "--output",
        output_template,
    ]
    if command_exists("ffmpeg"):
        base_cmd.extend(["--merge-output-format", "mp4"])
    if proxy:
        base_cmd.extend(["--proxy", proxy])
    if site_mode == "instagram-public":
        base_cmd.extend(
            [
                "--referer",
                SOCIAL_REFERER,
                "--add-header",
                "Origin:https://www.instagram.com",
                "--add-header",
                "Accept-Language:en-US,en;q=0.9",
            ]
        )

    if impersonation_targets is None:
        impersonation_targets = list_impersonate_targets()

    strategies: list[tuple[str, list[str]]] = [("default", [])]
    if access_strategy == "resilient":
        strategies.append(("generic-extractor", ["--force-generic-extractor"]))
        strategies.append(
            (
                "generic-network-retry",
                ["--force-generic-extractor", "--retries", "12", "--retry-sleep", "exp=1:16"],
            )
        )
        preferred_targets = ["chrome", "edge", "safari", "firefox"]
        available_targets = [target for target in preferred_targets if target in impersonation_targets]
        if not available_targets and impersonation_targets:
            available_targets = impersonation_targets[:1]
        for target in available_targets:
            strategies.append((f"impersonate-{target}", ["--impersonate", target]))
            strategies.append(
                (
                    f"network-retry-{target}",
                    ["--impersonate", target, "--retries", "12", "--retry-sleep", "exp=1:16"],
                )
            )
            strategies.append(
                (
                    f"generic-impersonate-{target}",
                    ["--force-generic-extractor", "--impersonate", target],
                )
            )
        if not available_targets:
            strategies.append(("network-retry", ["--retries", "12", "--retry-sleep", "exp=1:16"]))

    failures: list[str] = []
    for label, extra in strategies:
        cmd = base_cmd + extra + [url]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=max(timeout, 180))
            if completed.returncode == 0:
                return True, f"yt-dlp success via {label} strategy"
            error_text = (completed.stderr or completed.stdout).strip()
            err = error_text.splitlines()[-1:] or ["unknown error"]
            failures.append(f"{label}: {err[0]}")
        except subprocess.TimeoutExpired:
            failures.append(f"{label}: timeout")

    combined_failure = " | ".join(failures)
    if (
        allow_auto_install
        and access_strategy == "resilient"
        and needs_impersonation_dependency(combined_failure)
    ):
        if auto_install_yt_dlp_impersonation():
            refreshed_targets = list_impersonate_targets()
            return run_yt_dlp_with_strategies(
                url,
                out_dir,
                timeout,
                proxy,
                "resilient",
                site_mode,
                impersonation_targets=refreshed_targets,
                allow_auto_install=False,
            )
    return False, combined_failure[-320:]




def needs_impersonation_dependency(error_text: str) -> bool:
    lower = error_text.lower()
    return (
        "--list-impersonate-targets" in lower
        or "cloudflare anti-bot challenge" in lower
        or "install the required impersonation dependency" in lower
        or "impersonate target" in lower
    )


def suggest_authorized_next_steps(error_text: str) -> str | None:
    """Return actionable guidance for common download failure classes.

    Guidance focuses on authorized-access fixes (cookies, retries, extractor mode),
    not bypassing access controls.
    """
    lower = error_text.lower()
    if "unsupported url" in lower:
        return (
            "tip: page was not matched by a dedicated extractor; use --all-page "
            "or keep --access-strategy resilient for generic extractor fallback"
        )
    if "403" in lower or "forbidden" in lower or "anti-bot" in lower or "cloudflare" in lower:
        return (
            "tip: server rejected non-browser requests; use public media URLs, lower request rate, "
            "and only enable authenticated cookie workflows if your runtime policy explicitly allows it"
        )
    if "429" in lower or "too many requests" in lower:
        return "tip: rate-limited by host; reduce parallelism (--workers 1-2) and retry later"
    return None


def run_yt_dlp_all_resolutions(
    url: str,
    out_dir: Path,
    timeout: float,
    proxy: str | None,
    access_strategy: str,
    site_mode: str,
) -> tuple[int, int, list[str]]:
    """Download every available video resolution by enumerating yt-dlp formats."""
    cmd_prefix = yt_dlp_command()
    if not cmd_prefix:
        return 0, 0, ["yt-dlp not installed"]

    impersonation_targets = list_impersonate_targets() if access_strategy == "resilient" else []
    impersonation_args: list[str] = []
    if impersonation_targets:
        impersonation_args = ["--impersonate", impersonation_targets[0]]

    list_cmd = cmd_prefix + ["--list-formats"] + impersonation_args + [url]
    if site_mode == "instagram-public":
        list_cmd.extend(["--referer", SOCIAL_REFERER])
    if proxy:
        list_cmd.extend(["--proxy", proxy])

    listed = subprocess.run(list_cmd, capture_output=True, text=True, timeout=max(timeout, 60))
    if listed.returncode != 0:
        listing_error = (listed.stderr or listed.stdout).strip()
        if (
            access_strategy == "resilient"
            and needs_impersonation_dependency(listing_error)
            and auto_install_yt_dlp_impersonation()
        ):
            impersonation_targets = list_impersonate_targets()
            if impersonation_targets:
                impersonation_args = ["--impersonate", impersonation_targets[0]]
                list_cmd = cmd_prefix + ["--list-formats"] + impersonation_args + [url]
                if site_mode == "instagram-public":
                    list_cmd.extend(["--referer", SOCIAL_REFERER])
                if proxy:
                    list_cmd.extend(["--proxy", proxy])
                listed = subprocess.run(list_cmd, capture_output=True, text=True, timeout=max(timeout, 60))
                if listed.returncode != 0:
                    listing_error = (listed.stderr or listed.stdout).strip()
        if listed.returncode != 0:
            return 0, 1, [f"format listing failed: {listing_error[-300:]}"]

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
        cmd = cmd_prefix + impersonation_args + [
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
        if site_mode == "instagram-public":
            cmd.extend(["--referer", SOCIAL_REFERER])

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


def collect_candidates(
    session: requests.Session,
    url: str,
    timeout: float,
    include_all_html_media: bool,
    site_mode: str,
) -> list[Candidate]:
    page_html = fetch_page(session, url, timeout)
    extracted = extract_from_html(page_html, url, include_all_html_media=include_all_html_media)
    if site_mode == "instagram-public":
        existing = {item.url for item in extracted}
        for item in extract_instagram_post_candidates(page_html, url):
            if item.url not in existing:
                extracted.append(item)
                existing.add(item.url)
    return extracted


def preview(candidates: Iterable[Candidate], limit: int = 40) -> None:
    print("\\n=== Preview ===")
    for idx, item in enumerate(candidates):
        if idx >= limit:
            print(f"... and more ({idx + 1}+ entries)")
            break
        print(f"[{idx+1:03}] {safe_filename_from_url(item.url, 'media_item')} :: {sanitize_url_for_logs(item.url)}  ({item.source})")


def anonymize_notice(proxy: str | None) -> None:
    print("\\n=== Privacy notice ===")
    print("This tool minimizes local metadata and supports proxy routing, but cannot guarantee complete anonymity.")
    if proxy:
        print(f"Proxy in use: {proxy}")
    else:
        print("No proxy configured. Consider --proxy socks5h://127.0.0.1:9050 with Tor.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download media from a webpage with fallback strategies.")
    defaults = load_defaults()
    parser.add_argument("url", nargs="?", help="Target page URL")
    parser.add_argument("--output", default=str(defaults.get("output", "media_dump")), help="Output directory")
    parser.add_argument("--proxy", default=defaults.get("proxy"), help="Proxy URL (example: socks5h://127.0.0.1:9050)")
    parser.add_argument("--timeout", type=float, default=float(defaults.get("timeout", 45.0)), help="Request timeout in seconds")
    parser.add_argument("--workers", type=int, default=int(defaults.get("workers", 6)), help="Parallel download workers")
    parser.add_argument("--preview-only", action="store_true", default=bool_default(defaults, "preview_only", False), help="Only discover and preview media")
    parser.add_argument("--include-page-with-ytdlp", action="store_true", default=bool_default(defaults, "include_page_with_ytdlp", False), help="Also run yt-dlp directly on page URL")
    parser.add_argument("--skip-launch-check", action="store_true", help="Skip dependency check output")
    parser.add_argument("--auto-install", action="store_true", help="Try to auto-install missing yt-dlp")
    parser.add_argument("--install-dependencies", action="store_true", help="Install recommended downloader dependencies")
    parser.add_argument("--list-tools", action="store_true", help="Print a checklist of downloader tools")
    parser.add_argument("--diagnostics", action="store_true", default=bool_default(defaults, "diagnostics", False), help="Print detailed diagnostics and format information")
    parser.add_argument("--download-all-resolutions", action="store_true", default=bool_default(defaults, "download_all_resolutions", False), help="Download each available video resolution with yt-dlp")
    parser.add_argument("--all-page", action="store_true", help="Capture page-level assets and run page extraction")
    parser.add_argument("--all-scroll", action="store_true", help="Auto-scroll page (Playwright) and capture media URLs")
    parser.add_argument(
        "--site-mode",
        choices=["general", "instagram-public", "all-media-html-sources"],
        default=str(defaults.get("site_mode", "general")),
        help="Site profile preset. Use instagram-public for public social media pages.",
    )
    parser.add_argument(
        "--access-strategy",
        choices=["standard", "resilient"],
        default=str(defaults.get("access_strategy", "standard")),
        help="Authorized access strategy for yt-dlp (resilient adds retries and browser impersonation).",
    )
    parser.add_argument("--save-defaults", action="store_true", help="Save current options as defaults")
    args = parser.parse_args()

    if args.save_defaults:
        ok, msg = save_defaults({
            "output": args.output,
            "proxy": args.proxy,
            "timeout": args.timeout,
            "workers": args.workers,
            "preview_only": args.preview_only,
            "include_page_with_ytdlp": args.include_page_with_ytdlp,
            "diagnostics": args.diagnostics,
            "download_all_resolutions": args.download_all_resolutions,
            "access_strategy": args.access_strategy,
            "site_mode": args.site_mode,
        })
        print(msg)
        if not ok:
            return 1

    if args.install_dependencies:
        install_results = auto_install_dependencies()
        print("\n=== Dependency installation ===")
        for pkg, ok in install_results.items():
            print(f"- {pkg:<14}: {'INSTALLED' if ok else 'FAILED'}")
        print("Note: ffmpeg binary may still need OS package manager install.")

    if args.list_tools:
        yt_cmd = yt_dlp_command()
        print("\n=== Tool checklist ===")
        print("- yt-dlp  :", "OK" if yt_dlp_available() else "MISSING")
        if yt_cmd:
            print("  resolver:", " ".join(yt_cmd))
        print("  python  :", sys.executable)
        print("  PATH    :", os.environ.get("PATH", ""))
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
    if url_violates_privacy_policy(args.url):
        print("POLICY_BLOCKED: URL contains disallowed credentials or sensitive query keys.")
        print("Remove token/session/password-style data and retry.")
        return 3

    if args.site_mode == "instagram-public":
        if args.access_strategy != "resilient":
            print("Site mode instagram-public overrides access strategy to resilient.")
        args.access_strategy = "resilient"
        args.include_page_with_ytdlp = True
        args.workers = min(args.workers, 2)
        normalized_url, note = normalize_target_url(args.url, args.site_mode)
        if note:
            print(note)
        args.url = normalized_url

    anonymize_notice(args.proxy)
    if not args.skip_launch_check:
        all_ok = run_launch_check()
        if not all_ok and args.auto_install and not yt_dlp_available():
            print("Attempting auto-install of yt-dlp...")
            if auto_install_yt_dlp():
                print("yt-dlp installed successfully.")
            else:
                print("Auto-install failed. Continue with limited fallback support.")

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    include_all_html_media = args.site_mode == "all-media-html-sources"
    session = build_session(proxy=args.proxy, timeout=args.timeout, site_mode=args.site_mode)
    try:
        candidates = collect_candidates(
            session,
            args.url,
            args.timeout,
            include_all_html_media=include_all_html_media,
            site_mode=args.site_mode,
        )
    except Exception as exc:
        print(f"Failed to parse page: {exc}")
        candidates = []

    if args.all_scroll:
        try:
            scrolled = collect_candidates_from_scrolling(args.url, args.timeout, args.proxy)
            existing = {c.url for c in candidates}
            for item in scrolled:
                if item.url not in existing:
                    candidates.append(item)
                    existing.add(item.url)
            print(f"All-scroll discovery added {len(scrolled)} entries.")
        except Exception as exc:
            print(f"All-scroll discovery failed: {exc}")

    preview(candidates)
    print(f"\\nDiscovered {len(candidates)} potential media URLs.")

    if args.preview_only:
        return 0

    success_count = 0
    failure_count = 0

    def worker(candidate: Candidate) -> tuple[str, bool, str]:
        if candidate.kind == "instagram-post":
            ok, msg = run_yt_dlp_with_strategies(
                candidate.url,
                out_dir,
                max(args.timeout * 2, 90),
                args.proxy,
                args.access_strategy,
                args.site_mode,
            )
            return candidate.url, ok, msg if ok else f"instagram post extraction failed: {msg}"

        if url_violates_privacy_policy(candidate.url):
            return candidate.url, False, "policy-blocked candidate URL with sensitive token/credentials"
        content_type = probe_content_type(session, candidate.url, args.timeout)
        if not looks_like_media(candidate.url, content_type):
            return candidate.url, False, "not media"

        if candidate.url.lower().endswith(".m3u8") or "m3u8" in candidate.url.lower():
            ok, msg = run_yt_dlp_with_strategies(
                candidate.url,
                out_dir,
                args.timeout,
                args.proxy,
                args.access_strategy,
                args.site_mode,
            )
            if ok:
                return candidate.url, True, msg
            ff_ok, ff_msg = run_ffmpeg_hls(candidate.url, out_dir, args.timeout, args.proxy)
            if ff_ok:
                return candidate.url, True, ff_msg
            return candidate.url, False, f"{msg}; ffmpeg: {ff_msg}"

        ok, msg = download_file(session, candidate.url, out_dir, args.timeout)
        if ok:
            return candidate.url, True, msg

        ok2, msg2 = run_yt_dlp_with_strategies(
            candidate.url,
            out_dir,
            args.timeout,
            args.proxy,
            args.access_strategy,
            args.site_mode,
        )
        return candidate.url, ok2, msg2 if ok2 else f"{msg}; ytdlp: {msg2}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        results = list(pool.map(worker, candidates)) if candidates else []

    for url, ok, message in results:
        if ok:
            success_count += 1
            print(f"[OK] {sanitize_url_for_logs(url)} -> {message}")
        else:
            failure_count += 1
            tip = suggest_authorized_next_steps(message)
            if tip:
                print(f"[FAIL] {sanitize_url_for_logs(url)} -> {message} | {tip}")
            else:
                print(f"[FAIL] {sanitize_url_for_logs(url)} -> {message}")

    run_page_level_yt_dlp = args.include_page_with_ytdlp or args.all_page
    if args.site_mode == "instagram-public":
        has_instagram_post_candidates = any(item.kind == "instagram-post" for item in candidates)
        if has_instagram_post_candidates and not args.all_page:
            run_page_level_yt_dlp = False
            print("Skipping direct profile URL extraction because post links were discovered.")

    if run_page_level_yt_dlp:
        ok, msg = run_yt_dlp_with_strategies(
            args.url,
            out_dir,
            max(args.timeout * 2, 90),
            args.proxy,
            args.access_strategy,
            args.site_mode,
        )
        if ok:
            success_count += 1
            print(f"[OK] page-url -> {msg}")
        else:
            failure_count += 1
            tip = suggest_authorized_next_steps(msg)
            if tip:
                print(f"[FAIL] page-url -> {msg} | {tip}")
            else:
                print(f"[FAIL] page-url -> {msg}")

    if args.download_all_resolutions:
        res_ok, res_fail, debug_lines = run_yt_dlp_all_resolutions(
            args.url,
            out_dir,
            max(args.timeout * 2, 120),
            args.proxy,
            args.access_strategy,
            args.site_mode,
        )
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
        print(f"Site mode      : {args.site_mode}")
        print(f"Access strategy: {args.access_strategy}")
        print(f"Candidates     : {len(candidates)}")
        cmd_prefix = yt_dlp_command()
        if cmd_prefix:
            probe = subprocess.run(cmd_prefix + ["--list-formats", args.url], capture_output=True, text=True, timeout=max(args.timeout, 90))
            print("Format probe rc:", probe.returncode)
            print((probe.stdout or probe.stderr)[-4000:])

    print("\\n=== Summary ===")
    print(f"Output folder : {out_dir}")
    print(f"Successful    : {success_count}")
    print(f"Failed        : {failure_count}")
    print("Use only on content you are allowed to download.")
    session.cookies.clear()
    session.close()
    return 0 if success_count > 0 or not candidates else 2


if __name__ == "__main__":
    sys.exit(main())
