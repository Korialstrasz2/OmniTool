"""Review lyrics, then tag NEW copies. Originals are never opened for writing."""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlencode

from .content_files import (ContentError, digest, directory, is_link, new_output,
                            portable_relative, publish_new_directory, read_file)

AUDIO_EXTS = {'.mp3', '.flac', '.m4a', '.mp4', '.ogg', '.opus'}
MAX_FILE = 128 * 1024 * 1024
MAX_BATCH = 512 * 1024 * 1024
MAX_TRACKS = 50
MAX_TEXT = 32768
SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.omnitool-local'}


def clean(value, limit=1024) -> str:
    if not isinstance(value, str) or len(value.encode('utf-8')) > limit or '\0' in value:
        raise ContentError('Invalid or oversized metadata/lyrics text')
    return value.replace('\r\n', '\n').replace('\r', '\n').strip()


def plain_lrc(value: str) -> str:
    lines = []
    for line in value.splitlines():
        if re.match(r'^\s*\[(?:ar|al|ti|au|by|offset|re|ve|length):', line, re.I):
            continue
        line = re.sub(r'\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\]', '', line).strip()
        if line:
            lines.append(line)
    return '\n'.join(lines)


def _audio(data: bytes):
    from mutagen import File
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
    from mutagen.flac import FLAC
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    try:
        audio = File(io.BytesIO(data))
    except Exception:
        raise ContentError('Unreadable audio metadata or unsupported audio format') from None
    if not isinstance(audio, (MP3, MP4, FLAC, OggOpus, OggVorbis)):
        raise ContentError('Supported audio: MP3, FLAC, M4A/MP4 audio, Ogg Vorbis, and Opus')
    return audio


def _first(value) -> str:
    if hasattr(value, 'text'):
        value = value.text
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ''
    return clean(str(value or ''))


def metadata(data: bytes, filename: str) -> dict:
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
    audio = _audio(data)
    tags = audio.tags or {}
    if isinstance(audio, MP3):
        keys = ('TPE1', 'TIT2', 'TALB')
        lyrics = '\n'.join(f.text for f in tags.getall('USLT')) if audio.tags else ''
        existing = bool(lyrics.strip()) or bool(tags.getall('SYLT') if audio.tags else False)
    elif isinstance(audio, MP4):
        keys = ('\xa9ART', '\xa9nam', '\xa9alb')
        lyrics = '\n'.join(tags.get('\xa9lyr', []))
        existing = bool(lyrics.strip())
    else:
        keys = ('artist', 'title', 'album')
        lyrics = '\n'.join(tags.get('lyrics', []))
        existing = bool(lyrics.strip())
    artist, title, album = [_first(tags.get(k)) for k in keys]
    inferred = False
    if not title or not artist:
        stem = re.sub(r'^\d{1,3}[\s._-]+', '', Path(filename).stem)
        split = re.split(r'\s[-–—]\s', stem, maxsplit=1)
        title = title or split[-1]
        artist = artist or (split[0] if len(split) == 2 else '')
        inferred = True
    duration = float(getattr(audio.info, 'length', 0))
    if not math.isfinite(duration) or duration < 0:
        raise ContentError('Invalid audio duration')
    return {'artist': clean(artist), 'title': clean(title), 'album': clean(album),
            'duration': round(duration, 3), 'has_lyrics': existing, 'inferred': inferred,
            'format': type(audio).__name__}


def scan(root: Path) -> dict:
    root = directory(root)
    if root == Path(root.anchor) or root == Path.home():
        raise ContentError('Choose a specific music folder, not a drive root or home directory')
    tracks, warnings = [], []
    entries = total = 0
    for parent, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: (_ for _ in ()).throw(e)):
        parent = directory(parent)
        safe_dirs = []
        for name in sorted(dirs):
            entries += 1
            if name in SKIP or is_link((parent / name).lstat()):
                warnings.append({'path': (parent / name).relative_to(root).as_posix(), 'reason': 'Excluded directory or link'})
            else:
                portable_relative((parent / name).relative_to(root).as_posix())
                safe_dirs.append(name)
        dirs[:] = safe_dirs
        for name in sorted(files):
            entries += 1
            if entries > 5000:
                raise ContentError('Scan exceeds 5,000 entries; choose a smaller music folder')
            path = parent / name
            if path.suffix.lower() not in AUDIO_EXTS:
                continue
            rel = path.relative_to(root).as_posix()
            if len(tracks) >= MAX_TRACKS:
                raise ContentError('More than 50 supported tracks; choose a smaller folder')
            try:
                portable_relative(rel)
                data = read_file(path, MAX_FILE)
                total += len(data)
                if total > MAX_BATCH:
                    raise ContentError('Selected audio exceeds the 512 MiB batch limit')
                meta = metadata(data, rel)
                tracks.append(dict(meta, file=rel, sha256=digest(data), bytes=len(data), index=len(tracks)))
            except ContentError as exc:
                warnings.append({'path': rel, 'reason': str(exc)})
            if total > MAX_BATCH:
                raise ContentError('Scanned audio exceeds 512 MiB; choose a smaller folder')
        if entries > 5000:
            raise ContentError('Scan exceeds 5,000 entries')
    return {'root': str(root), 'tracks': tracks, 'warnings': warnings[:100], 'warning_count': len(warnings), 'bytes': total}


def fetch_lrclib(meta: dict) -> dict:
    """Single HTTPS lookup. No fuzzy title stripping, unencrypted fallback, or AI call."""
    import requests
    query = {'artist_name': clean(meta.get('artist', '')), 'track_name': clean(meta.get('title', '')),
             'album_name': clean(meta.get('album', '')), 'duration': meta.get('duration', 0)}
    if not query['artist_name'] or not query['track_name']:
        raise ContentError('Artist and title are required for LRCLIB lookup')
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.get('https://lrclib.net/api/get', params=query, timeout=(5, 20),
                             allow_redirects=False, stream=True,
                             headers={'User-Agent': 'OmniTool/1.0 (local lyrics workbench)', 'Accept': 'application/json'}) as response:
                if response.status_code == 404:
                    raise ContentError('No LRCLIB match; provide a local sidecar instead')
                if response.status_code != 200:
                    raise ContentError(f'LRCLIB returned HTTP {response.status}; no fallback was used')
                chunks, size = [], 0
                deadline = time.monotonic() + 25
                for chunk in response.iter_content(8192):
                    size += len(chunk)
                    if size > 256 * 1024 or time.monotonic() > deadline:
                        raise ContentError('LRCLIB response exceeds time/size limits')
                    chunks.append(chunk)
                result = json.loads(b''.join(chunks))
    except (requests.RequestException, ValueError) as exc:
        if isinstance(exc, ContentError):
            raise
        raise ContentError('LRCLIB is unavailable or returned invalid data') from None
    if not isinstance(result, dict):
        raise ContentError('Unexpected LRCLIB response')
    key = lambda s: unicodedata.normalize('NFKC', clean(s)).casefold()
    if key(result.get('artistName')) != key(query['artist_name']) or key(result.get('trackName')) != key(query['track_name']):
        raise ContentError('Provider artist/title differs from the reviewed metadata')
    duration = result.get('duration')
    if type(duration) not in {int, float} or not math.isfinite(duration) or abs(duration - query['duration']) > 2:
        raise ContentError('Provider duration differs by more than two seconds')
    if result.get('instrumental'):
        raise ContentError('Provider marks this track as instrumental')
    plain = clean(result.get('plainLyrics') or '', MAX_TEXT)
    synced = clean(result.get('syncedLyrics') or '', MAX_TEXT)
    plain = plain or plain_lrc(synced)
    if not plain:
        raise ContentError('Provider returned no lyrics')
    return {'plain': plain, 'synced': synced, 'provider': 'LRCLIB', 'provider_id': str(result.get('id', ''))[:100]}


def prepare(scan_result: dict, selections: list, out: Path, provider: str, replace: bool = False) -> dict:
    root = directory(scan_result['root'])
    output = new_output(out, root)
    if provider not in {'sidecar', 'lrclib'} or type(replace) is not bool:
        raise ContentError('Invalid lyrics preparation options')
    if not isinstance(selections, list) or not 1 <= len(selections) <= MAX_TRACKS:
        raise ContentError('Select between 1 and 50 tracks')
    rows, seen = [], set()
    for selection in selections:
        if not isinstance(selection, dict) or type(selection.get('index')) is not int:
            raise ContentError('Invalid track selection')
        index = selection['index']
        if index in seen or not 0 <= index < len(scan_result['tracks']):
            raise ContentError('Duplicate or invalid track selection')
        seen.add(index)
        source = scan_result['tracks'][index]
        row = {'file': source['file'], 'sha256': source['sha256'], 'status': 'skipped', 'reason': ''}
        rows.append(row)
        try:
            path = root / portable_relative(source['file'])
            data = read_file(path, MAX_FILE)
            if digest(data) != source['sha256']:
                raise ContentError('Source changed after inspection; scan again')
            meta = metadata(data, source['file'])
            if meta['has_lyrics'] and not replace:
                raise ContentError('Existing lyrics protected; replacement is not enabled')
            for name in ('artist', 'title', 'album'):
                if name in selection:
                    meta[name] = clean(selection[name])
            if provider == 'sidecar':
                lrc, txt = path.with_suffix('.lrc'), path.with_suffix('.txt')
                selected = lrc if lrc.exists() else txt
                raw = read_file(selected, MAX_TEXT).decode('utf-8-sig', errors='strict')
                raw = clean(raw, MAX_TEXT)
                lyrics = {'plain': plain_lrc(raw) if selected.suffix == '.lrc' else raw,
                          'synced': raw if selected.suffix == '.lrc' else '', 'provider': 'local-sidecar'}
            else:
                lyrics = fetch_lrclib(meta)
                time.sleep(1)  # Deliberately serial and conservative; no automatic retries.
            if not lyrics['plain']:
                raise ContentError('Lyrics are empty after removing LRC timestamps')
            row.update(status='ready', metadata=meta, bytes=len(data), **lyrics)
        except (ContentError, OSError, UnicodeError) as exc:
            row['reason'] = str(exc) if isinstance(exc, ContentError) else 'Sidecar/source is missing, unreadable, or not UTF-8'
    ready = [r for r in rows if r['status'] == 'ready']
    if sum(r['bytes'] for r in ready) > MAX_BATCH:
        raise ContentError('Selected audio exceeds 512 MiB')
    # Sidecars derived from equal stems must not collide even when audio extensions differ.
    names = []
    for row in ready:
        names.append(row['file'])
        if row.get('synced'):
            names.append(Path(row['file']).with_suffix('.lrc').as_posix())
    normalized = [unicodedata.normalize('NFC', n).casefold() for n in names]
    if len(normalized) != len(set(normalized)):
        raise ContentError('Output audio/sidecar filenames collide; select a non-conflicting subset')
    return {'root': str(root), 'output': str(output), 'rows': rows, 'count': len(ready),
            'provider': provider, 'replace': replace, 'originals_unchanged': True}


def _tag_copy(path: Path, plain: str) -> None:
    from mutagen import File
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
    from mutagen.id3 import USLT
    audio = File(path)
    if audio is None:
        raise ContentError('Cannot tag the staged audio copy')
    if audio.tags is None:
        audio.add_tags()
    if isinstance(audio, MP3):
        audio.tags.delall('USLT')
        audio.tags.delall('SYLT')
        audio.tags.add(USLT(encoding=3, lang='und', desc='', text=plain))
    elif isinstance(audio, MP4):
        audio.tags['\xa9lyr'] = [plain]
    else:
        audio['lyrics'] = [plain]
    audio.save()  # Writes only the new staging copy, never the source.
    check = File(path)
    if isinstance(check, MP3):
        actual = '\n'.join(f.text for f in check.tags.getall('USLT'))
    elif isinstance(check, MP4):
        actual = '\n'.join(check.tags.get('\xa9lyr', []))
    else:
        actual = '\n'.join(check.get('lyrics', []))
    if actual != plain:
        raise ContentError('Tag read-back did not match; output was not published')


def apply(preview: dict) -> dict:
    root = directory(preview['root'])
    output = new_output(preview['output'], root)
    rows = [r for r in preview['rows'] if r['status'] == 'ready']
    if not rows or len(rows) > MAX_TRACKS or sum(r['bytes'] for r in rows) > MAX_BATCH:
        raise ContentError('Invalid or empty reviewed batch')
    # Full preflight before allocating any output; repeat the hash during copying.
    for row in rows:
        if digest(read_file(root / portable_relative(row['file']), MAX_FILE)) != row['sha256']:
            raise ContentError('A source changed since preview; prepare the batch again')
    stage = Path(tempfile.mkdtemp(prefix='.omnitool-lyrics-', suffix='.partial', dir=output.parent))
    published = False
    try:
        for row in rows:
            data = read_file(root / portable_relative(row['file']), MAX_FILE)
            if digest(data) != row['sha256']:
                raise ContentError('A source changed during export; output was not published')
            target = stage / portable_relative(row['file'])
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with target.open('xb') as stream:
                stream.write(data)
            del data
            _tag_copy(target, row['plain'])
            if row.get('synced'):
                with target.with_suffix('.lrc').open('x', encoding='utf-8', newline='\n') as stream:
                    stream.write(row['synced'])
        # No lyrics contents or titles in the on-disk summary, only file paths and source hashes.
        report = {'completed': True, 'output': str(output), 'count': len(rows), 'originals_unchanged': True,
                  'files': [{'file': r['file'], 'source_sha256': r['sha256'], 'provider': r['provider']} for r in rows]}
        with (stage / 'omnitool-lyrics-report.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        for path in stage.rglob('*'):
            if path.is_file():
                with path.open('r+b') as stream:
                    os.fsync(stream.fileno())
        # Revalidate the parent; the no-replace primitive rejects a concurrently created output.
        directory(output.parent)
        publish_new_directory(stage, output)
        published = True
        return report
    finally:
        if not published:
            shutil.rmtree(stage)  # Cleanup errors remain visible; this is not secure erasure.


def main():
    parser = argparse.ArgumentParser(description='Read-only lyrics scan or local-sidecar preview; writes are reviewed in OmniTool')
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, help='Prepare a local-sidecar preview for a new output directory (does not write audio)')
    args = parser.parse_args()
    try:
        result = scan(args.root)
        if args.out:
            result = prepare(result, [{'index': r['index']} for r in result['tracks']], args.out, 'sidecar')
        print(json.dumps(result, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
