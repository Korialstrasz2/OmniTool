"""Read-only duplicate finder. Never deletes, renames, or links input files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import defaultdict
from pathlib import Path


def _linked(info: os.stat_result) -> bool:
    # Junctions and other Windows reparse points are not ordinary directories.
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def _stamp(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)


def _directory(path: Path) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    for parent in (*reversed(path.parents), path):
        info = parent.lstat()
        if _linked(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError('Choose an ordinary directory, not a link or junction')
    return path


def _hash_file(path: Path, enumerated: tuple) -> str:
    _directory(path.parent)
    before = path.lstat()
    if _linked(before) or not stat.S_ISREG(before.st_mode) or _stamp(before) != enumerated:
        raise OSError('File changed since enumeration')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as source:
        opened = os.fstat(source.fileno())
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
            raise OSError('File identity changed while opening')
        digest = hashlib.sha256()
        count = 0
        while chunk := source.read(1024 * 1024):
            count += len(chunk)
            if count > before.st_size:
                raise OSError('File grew during hashing')
            digest.update(chunk)
        after_handle = os.fstat(source.fileno())
    after_path = path.lstat()
    # Windows path-stat and descriptor-stat timestamps/modes can differ.
    # Compare each representation to itself; identity is cross-checked above.
    if (count != before.st_size or count != opened.st_size or
            _stamp(before) != _stamp(after_path) or _stamp(opened) != _stamp(after_handle)):
        raise OSError('File changed during hashing')
    _directory(path.parent)
    return digest.hexdigest()


def find_duplicates(root: Path, min_size: int = 1, max_files: int = 100000) -> dict:
    if type(min_size) is not int or min_size < 0 or type(max_files) is not int or not 1 <= max_files <= 1000000:
        raise ValueError('Invalid size or file-count limit')
    root = _directory(root)
    sizes = defaultdict(list)
    errors, scanned = [], 0

    def on_error(exc):
        errors.append(f"Cannot inspect directory: {getattr(exc, 'filename', '')}")

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=on_error):
        safe_dirs = []
        for name in sorted(dirs):
            try:
                info = (Path(directory) / name).lstat()
                if not _linked(info) and stat.S_ISDIR(info.st_mode):
                    safe_dirs.append(name)
            except OSError:
                errors.append(f'Cannot inspect directory: {Path(directory) / name}')
        dirs[:] = safe_dirs
        for name in sorted(files):
            path = Path(directory) / name
            try:
                info = path.lstat()
                if _linked(info) or not stat.S_ISREG(info.st_mode) or info.st_size < min_size:
                    continue
                scanned += 1
                if scanned > max_files:
                    raise ValueError('File limit exceeded; choose a smaller directory or raise --max-files')
                sizes[info.st_size].append((path, _stamp(info)))
            except OSError:
                errors.append(f'Cannot inspect file: {path}')
    groups = []
    for size, candidates in sorted(sizes.items()):
        if len(candidates) < 2:
            continue
        digests = defaultdict(list)
        for path, snapshot in candidates:
            try:
                digests[_hash_file(path, snapshot)].append(str(path))
            except (OSError, ValueError):
                errors.append(f'Unreadable or changed file: {path}')
        for digest, paths in sorted(digests.items()):
            if len(paths) > 1:
                groups.append({'sha256': digest, 'bytes_each': size, 'files': paths})
    return {'files_scanned': scanned, 'duplicate_groups': groups, 'errors': errors, 'read_only': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--min-size', type=int, default=1)
    parser.add_argument('--max-files', type=int, default=100000)
    args = parser.parse_args()
    try:
        report = find_duplicates(args.root, args.min_size, args.max_files)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 2 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
