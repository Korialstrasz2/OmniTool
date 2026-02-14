#!/usr/bin/env python3
"""Fetch and embed lyrics into audio files with optional AI metadata recovery."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import requests
from mutagen import File as MutagenFile
from mutagen.flac import FLAC
from mutagen.id3 import ID3, ID3NoHeaderError, USLT
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore

AUDIO_EXTS_DEFAULT = {".mp3", ".flac", ".m4a", ".mp4", ".aac", ".ogg", ".opus"}
LRCLIB_GET = "https://lrclib.net/api/get"
LYRICS_OVH_HOST = "https://api.lyrics.ovh/v1"
CHARTLYRICS_DIRECT = "http://api.chartlyrics.com/apiv1.asmx/SearchLyricDirect"
CACHE_FILE = ".lyrics_cache.json"
OK_LOG = "lyrics_embedded.txt"
FAILED_LOG = "lyrics_failed.txt"
CACHE_VERSION = 3


def norm(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return re.sub(r"\s+", " ", s.strip()) or None


def safe_get_first(tagval: Any) -> Optional[str]:
    if tagval is None:
        return None
    if hasattr(tagval, "text"):
        text = getattr(tagval, "text", None)
        if isinstance(text, (list, tuple)) and text:
            return str(text[0])
        return str(text) if text else None
    if isinstance(tagval, (list, tuple)):
        return str(tagval[0]) if tagval else None
    return str(tagval)


def parse_artist_title_from_filename(path: Path) -> Tuple[Optional[str], Optional[str]]:
    stem = path.stem
    for sep in (" - ", " – ", " — "):
        if sep in stem:
            artist, title = stem.split(sep, 1)
            return norm(artist), norm(title)
    return None, norm(stem)


def clean_synced_lrc_to_plain(synced: str) -> str:
    lines = []
    for line in synced.splitlines():
        cleaned = re.sub(r"\[[0-9:.]+\]", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()


def atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_metadata(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    artist = title = album = None
    duration_ms: Optional[int] = None
    audio = None
    try:
        audio = MutagenFile(path)
    except Exception:
        audio = None

    if audio is not None:
        try:
            if getattr(audio, "info", None) and getattr(audio.info, "length", None):
                duration_ms = int(round(audio.info.length * 1000))
        except Exception:
            duration_ms = None

        try:
            if isinstance(audio, MP3):
                id3 = audio.tags
                if id3:
                    title = safe_get_first(id3.get("TIT2"))
                    artist = safe_get_first(id3.get("TPE1")) or safe_get_first(id3.get("TPE2"))
                    album = safe_get_first(id3.get("TALB"))
            elif isinstance(audio, FLAC):
                title = safe_get_first(audio.get("title"))
                artist = safe_get_first(audio.get("artist"))
                album = safe_get_first(audio.get("album"))
            elif isinstance(audio, MP4):
                tags = audio.tags or {}
                title = safe_get_first(tags.get("\xa9nam"))
                artist = safe_get_first(tags.get("\xa9ART")) or safe_get_first(tags.get("aART"))
                album = safe_get_first(tags.get("\xa9alb"))
            elif isinstance(audio, (OggVorbis, OggOpus)):
                title = safe_get_first(audio.get("title"))
                artist = safe_get_first(audio.get("artist"))
                album = safe_get_first(audio.get("album"))
        except Exception:
            pass

    if not title or not artist:
        file_artist, file_title = parse_artist_title_from_filename(path)
        artist = artist or file_artist
        title = title or file_title

    return norm(artist), norm(title), norm(album), duration_ms


def has_existing_lyrics(path: Path) -> bool:
    try:
        audio = MutagenFile(path)
    except Exception:
        return False
    if audio is None:
        return False

    try:
        if isinstance(audio, MP3):
            try:
                id3 = ID3(path)
            except ID3NoHeaderError:
                return False
            return any(isinstance(getattr(frame, "text", ""), str) and getattr(frame, "text", "").strip() for frame in id3.getall("USLT"))
        if isinstance(audio, FLAC):
            lyrics = audio.get("LYRICS") or audio.get("lyrics")
            value = safe_get_first(lyrics) if lyrics else None
            return bool(value and value.strip())
        if isinstance(audio, MP4):
            lyrics = (audio.tags or {}).get("\xa9lyr")
            value = safe_get_first(lyrics) if lyrics else None
            return bool(value and value.strip())
        if isinstance(audio, (OggVorbis, OggOpus)):
            lyrics = audio.get("LYRICS") or audio.get("lyrics")
            value = safe_get_first(lyrics) if lyrics else None
            return bool(value and value.strip())
    except Exception:
        return False
    return False


def embed_lyrics(path: Path, lyrics: str) -> None:
    lyrics = lyrics.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not lyrics:
        raise RuntimeError("Cannot embed empty lyrics")

    audio = MutagenFile(path)
    if audio is None:
        raise RuntimeError("Unsupported file")

    if isinstance(audio, MP3):
        try:
            id3 = ID3(path)
        except ID3NoHeaderError:
            id3 = ID3()
        id3.setall("USLT", [USLT(encoding=3, lang="eng", desc="", text=lyrics)])
        id3.save(path)
    elif isinstance(audio, FLAC):
        audio["LYRICS"] = [lyrics]
        audio.save()
    elif isinstance(audio, MP4):
        if audio.tags is None:
            audio.add_tags()
        audio.tags["\xa9lyr"] = [lyrics]
        audio.save()
    elif isinstance(audio, (OggVorbis, OggOpus)):
        audio["LYRICS"] = [lyrics]
        audio.save()
    else:
        if audio.tags is None:
            raise RuntimeError("No writable tags")
        audio["LYRICS"] = [lyrics]
        audio.save()


def walk_audio_files(root: Path, exts: Set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def load_cache(cache_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    if not cache_path.exists():
        return {"_version": CACHE_VERSION, "items": {}}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "items" in raw:
            return raw
        if isinstance(raw, dict):
            logger.info("Cache upgraded to schema v%s", CACHE_VERSION)
            return {"_version": CACHE_VERSION, "items": raw}
    except Exception as exc:
        logger.warning("Could not load cache, starting fresh: %s", exc)
    return {"_version": CACHE_VERSION, "items": {}}


def save_cache(cache_path: Path, cache: Dict[str, Any], logger: logging.Logger) -> None:
    try:
        atomic_write_json(cache_path, cache)
    except Exception as exc:
        logger.warning("Could not save cache: %s", exc)


def make_cache_key(artist: str, title: str, album: Optional[str], duration_ms: Optional[int]) -> str:
    return f"{artist}||{title}||{album or ''}||{duration_ms or ''}"


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    lyrics: str


class Provider:
    name: str

    def fetch(
        self,
        session: requests.Session,
        artist: str,
        title: str,
        album: Optional[str],
        duration_ms: Optional[int],
        timeout: float,
        retries: int,
        backoff: float,
        logger: logging.Logger,
        keep_synced: bool,
    ) -> Optional[ProviderResult]:
        raise NotImplementedError


class LRCLibProvider(Provider):
    name = "lrclib"

    def fetch(self, session: requests.Session, artist: str, title: str, album: Optional[str], duration_ms: Optional[int], timeout: float, retries: int, backoff: float, logger: logging.Logger, keep_synced: bool) -> Optional[ProviderResult]:
        params: Dict[str, Any] = {"artist_name": artist, "track_name": title}
        if album:
            params["album_name"] = album
        if duration_ms:
            params["duration"] = duration_ms
        for attempt in range(retries):
            try:
                response = session.get(LRCLIB_GET, params=params, timeout=timeout)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                data = response.json() or {}
                plain = (data.get("plainLyrics") or "").strip()
                synced = (data.get("syncedLyrics") or "").strip()
                if plain:
                    return ProviderResult(self.name, plain)
                if synced:
                    return ProviderResult(self.name, synced if keep_synced else clean_synced_lrc_to_plain(synced))
                return None
            except Exception as exc:
                if attempt == retries - 1:
                    logger.debug("Provider %s failed: %s", self.name, exc)
                time.sleep((backoff ** attempt) + 0.2)
        return None


class LyricsOvhProvider(Provider):
    name = "lyrics.ovh"

    def fetch(self, session: requests.Session, artist: str, title: str, album: Optional[str], duration_ms: Optional[int], timeout: float, retries: int, backoff: float, logger: logging.Logger, keep_synced: bool) -> Optional[ProviderResult]:
        url = f"{LYRICS_OVH_HOST}/{requests.utils.quote(artist)}/{requests.utils.quote(title)}"
        for attempt in range(retries):
            try:
                response = session.get(url, timeout=timeout)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                lyrics = (response.json() or {}).get("lyrics", "").strip()
                return ProviderResult(self.name, lyrics) if lyrics else None
            except Exception as exc:
                if attempt == retries - 1:
                    logger.debug("Provider %s failed: %s", self.name, exc)
                time.sleep((backoff ** attempt) + 0.2)
        return None


class ChartLyricsProvider(Provider):
    name = "chartlyrics"

    def fetch(self, session: requests.Session, artist: str, title: str, album: Optional[str], duration_ms: Optional[int], timeout: float, retries: int, backoff: float, logger: logging.Logger, keep_synced: bool) -> Optional[ProviderResult]:
        params = {"artist": artist, "song": title}
        for attempt in range(retries):
            try:
                response = session.get(CHARTLYRICS_DIRECT, params=params, timeout=timeout)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                root = ET.fromstring(response.text)
                lyric_node = root.find(".//Lyric")
                if lyric_node is None or not (lyric_node.text or "").strip():
                    return None
                return ProviderResult(self.name, lyric_node.text.strip())
            except Exception as exc:
                if attempt == retries - 1:
                    logger.debug("Provider %s failed: %s", self.name, exc)
                time.sleep((backoff ** attempt) + 0.2)
        return None


_thread_local = threading.local()


def get_thread_session(user_agent: str) -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": user_agent})
        _thread_local.session = sess
    return sess


class MetadataAIAssistant:
    def __init__(self, enabled: bool, model: str, timeout: float, logger: logging.Logger):
        self.enabled = enabled
        self.model = model
        self.timeout = timeout
        self.logger = logger
        self.client = None
        if not enabled:
            return
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("AI metadata recovery requested, but OPENAI_API_KEY is not set.")
            self.enabled = False
            return
        if OpenAI is None:
            logger.warning("AI metadata recovery requested, but openai package is not installed.")
            self.enabled = False
            return
        self.client = OpenAI(api_key=api_key, timeout=timeout)

    def suggest(self, original: Dict[str, Any]) -> Optional[Dict[str, Optional[str]]]:
        if not self.enabled or self.client is None:
            return None

        schema = {
            "artist": "string|null",
            "title": "string|null",
            "album": "string|null",
            "confidence": "low|medium|high",
            "notes": "string"
        }
        prompt = (
            "You are a music metadata correction assistant."
            " Return ONLY strict JSON and nothing else."
            " If unsure, keep original values and lower confidence."
            " Never invent random songs."
            " JSON schema: " + json.dumps(schema) +
            "\nInput metadata: " + json.dumps(original, ensure_ascii=False)
        )
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": prompt}],
            )
            text = (response.output_text or "").strip()
            if not text:
                return None
            data = json.loads(text)
            if not isinstance(data, dict):
                return None
            confidence = str(data.get("confidence", "")).lower()
            if confidence not in {"high", "medium"}:
                self.logger.info("AI metadata skipped due to low confidence: %s", confidence or "unknown")
                return None
            return {
                "artist": norm(data.get("artist")) or norm(original.get("artist")),
                "title": norm(data.get("title")) or norm(original.get("title")),
                "album": norm(data.get("album")) or norm(original.get("album")),
            }
        except Exception as exc:
            self.logger.warning("AI metadata recovery failed: %s", exc)
            return None


class LineWriter:
    def __init__(self, ok_path: Path, fail_path: Path):
        self._lock = threading.Lock()
        self._ok = ok_path.open("a", encoding="utf-8")
        self._fail = fail_path.open("a", encoding="utf-8")

    def ok(self, msg: str) -> None:
        with self._lock:
            self._ok.write(msg + "\n")
            self._ok.flush()

    def fail(self, msg: str) -> None:
        with self._lock:
            self._fail.write(msg + "\n")
            self._fail.flush()

    def close(self) -> None:
        with self._lock:
            self._ok.close()
            self._fail.close()


@dataclass
class ProcessResult:
    status: str
    rel: str
    reason: Optional[str] = None


def attempt_fetch_and_embed(path: Path, artist: str, title: str, album: Optional[str], duration_ms: Optional[int], keep_synced: bool, timeout: float, retries: int, backoff: float, min_delay: float, providers: Tuple[Provider, ...], user_agent: str, logger: logging.Logger) -> Optional[ProviderResult]:
    session = get_thread_session(user_agent)
    for provider in providers:
        result = provider.fetch(
            session=session,
            artist=artist,
            title=title,
            album=album,
            duration_ms=duration_ms,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            logger=logger,
            keep_synced=keep_synced,
        )
        if min_delay > 0:
            time.sleep(min_delay)
        if result and result.lyrics and result.lyrics.strip():
            embed_lyrics(path, result.lyrics)
            return result
    return None


def process_one(path: Path, root: Path, exts: Set[str], force: bool, min_delay: float, timeout: float, retries: int, backoff: float, keep_synced: bool, providers: Tuple[Provider, ...], cache: Dict[str, Any], cache_lock: threading.Lock, writer: LineWriter, logger: logging.Logger, user_agent: str, ai_assistant: MetadataAIAssistant, ai_rounds: int) -> ProcessResult:
    rel = str(path.relative_to(root)) if path.is_absolute() else str(path)
    try:
        if path.suffix.lower() not in exts:
            return ProcessResult("skipped", rel, reason="extension")
        if not force and has_existing_lyrics(path):
            writer.ok(f"SKIP (has lyrics): {rel}")
            return ProcessResult("skipped", rel, reason="has_lyrics")

        artist, title, album, duration_ms = read_metadata(path)
        if not artist or not title:
            writer.fail(f"NO META: {rel}")
            return ProcessResult("failed", rel, reason="no_meta")

        cache_key = make_cache_key(artist, title, album, duration_ms)
        with cache_lock:
            cached = cache.get("items", {}).get(cache_key)
        if isinstance(cached, dict):
            lyrics = cached.get("lyrics")
            if lyrics:
                embed_lyrics(path, lyrics)
                writer.ok(f"OK (cache): {rel} ({artist} - {title})")
                return ProcessResult("embedded", rel, reason="cache_hit")
            if cached.get("lyrics") is None:
                return ProcessResult("failed", rel, reason="cache_negative")

        result = attempt_fetch_and_embed(path, artist, title, album, duration_ms, keep_synced, timeout, retries, backoff, min_delay, providers, user_agent, logger)
        if result:
            writer.ok(f"OK ({result.provider}): {rel} ({artist} - {title})")
            with cache_lock:
                cache["items"][cache_key] = {"lyrics": result.lyrics, "provider": result.provider, "ts": time.time()}
            return ProcessResult("embedded", rel, reason=result.provider)

        current_artist, current_title, current_album = artist, title, album
        for ai_round in range(ai_rounds):
            suggestion = ai_assistant.suggest({
                "artist": current_artist,
                "title": current_title,
                "album": current_album,
                "duration_ms": duration_ms,
                "filename": path.name,
            })
            if not suggestion or not suggestion.get("artist") or not suggestion.get("title"):
                break
            current_artist = suggestion["artist"]
            current_title = suggestion["title"]
            current_album = suggestion.get("album")
            logger.info("AI retry %d for %s using metadata: %s - %s", ai_round + 1, rel, current_artist, current_title)
            result = attempt_fetch_and_embed(path, current_artist, current_title, current_album, duration_ms, keep_synced, timeout, retries, backoff, min_delay, providers, user_agent, logger)
            if result:
                writer.ok(f"OK ({result.provider}, ai round {ai_round + 1}): {rel} ({current_artist} - {current_title})")
                with cache_lock:
                    cache["items"][cache_key] = {"lyrics": result.lyrics, "provider": f"{result.provider}+ai", "ts": time.time()}
                return ProcessResult("embedded", rel, reason=f"{result.provider}+ai")

        writer.fail(f"NOT FOUND: {rel} ({artist} - {title})")
        with cache_lock:
            cache["items"][cache_key] = {"lyrics": None, "provider": None, "ts": time.time()}
        return ProcessResult("failed", rel, reason="not_found")
    except Exception as exc:
        writer.fail(f"ERROR: {rel} -> {type(exc).__name__}: {exc}")
        return ProcessResult("failed", rel, reason=str(exc))


def setup_logger(level: str, log_file: Optional[Path]) -> logging.Logger:
    logger = logging.getLogger("lyrics_embedder")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and embed lyrics recursively with fallback providers and optional AI metadata recovery.")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-inflight", type=int, default=0)
    parser.add_argument("--min-delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.25)
    parser.add_argument("--keep-synced", action="store_true")
    parser.add_argument("--status-every", type=float, default=5.0)
    parser.add_argument("--cache-flush-every", type=int, default=200)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default="")
    parser.add_argument("--exts", default="")
    parser.add_argument("--ai-if-failed", action="store_true", help="Try AI metadata recovery when all providers fail.")
    parser.add_argument("--ai-rounds", type=int, default=2, help="How many AI correction rounds to run after failures.")
    parser.add_argument("--ai-model", default="gpt-4.1-mini", help="OpenAI model used for metadata correction.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Invalid root directory: {root}")
        return 2

    log_file = Path(args.log_file).expanduser().resolve() if args.log_file else None
    logger = setup_logger(args.log_level, log_file)

    exts = AUDIO_EXTS_DEFAULT if not args.exts.strip() else {ext.strip().lower() for ext in args.exts.split(",") if ext.strip()}
    if not exts:
        logger.error("No extensions configured")
        return 2

    ok_log = root / OK_LOG
    fail_log = root / FAILED_LOG
    ok_log.write_text("", encoding="utf-8")
    fail_log.write_text("", encoding="utf-8")
    writer = LineWriter(ok_log, fail_log)

    cache_path = root / CACHE_FILE
    cache = load_cache(cache_path, logger)
    cache.setdefault("items", {})
    cache_lock = threading.Lock()
    providers: Tuple[Provider, ...] = (LRCLibProvider(), LyricsOvhProvider(), ChartLyricsProvider())
    ai_assistant = MetadataAIAssistant(enabled=args.ai_if_failed, model=args.ai_model, timeout=args.timeout, logger=logger)

    max_inflight = args.max_inflight if args.max_inflight > 0 else max(8, args.workers * 8)
    user_agent = "lyrics-embedder/3.0"

    logger.info("Root: %s", root)
    logger.info("Providers: %s", " -> ".join(p.name for p in providers))
    logger.info("AI enabled: %s | model: %s | rounds: %s", ai_assistant.enabled, args.ai_model, args.ai_rounds)

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    discovered = submitted = completed = embedded = skipped = failed = 0
    start = time.time()
    last_status = start
    futures = set()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for path in walk_audio_files(root, exts):
                discovered += 1
                if args.max_files and submitted >= args.max_files:
                    break
                while len(futures) >= max_inflight:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        completed += 1
                        if result.status == "embedded":
                            embedded += 1
                        elif result.status == "skipped":
                            skipped += 1
                        else:
                            failed += 1
                        if completed % args.cache_flush_every == 0:
                            with cache_lock:
                                save_cache(cache_path, cache, logger)
                futures.add(executor.submit(
                    process_one,
                    path,
                    root,
                    exts,
                    args.force,
                    args.min_delay,
                    args.timeout,
                    args.retries,
                    args.backoff,
                    args.keep_synced,
                    providers,
                    cache,
                    cache_lock,
                    writer,
                    logger,
                    user_agent,
                    ai_assistant,
                    max(0, args.ai_rounds),
                ))
                submitted += 1
                now = time.time()
                if now - last_status >= args.status_every:
                    rate = completed / max(0.001, (now - start))
                    logger.info("Progress: discovered=%d submitted=%d completed=%d embedded=%d skipped=%d failed=%d rate=%.2f/s in_flight=%d", discovered, submitted, completed, embedded, skipped, failed, rate, len(futures))
                    last_status = now

            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    completed += 1
                    if result.status == "embedded":
                        embedded += 1
                    elif result.status == "skipped":
                        skipped += 1
                    else:
                        failed += 1
    finally:
        with cache_lock:
            save_cache(cache_path, cache, logger)
        writer.close()

    elapsed = time.time() - start
    logger.info("Done | completed=%d embedded=%d skipped=%d failed=%d elapsed=%.1fs", completed, embedded, skipped, failed, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
