"""Read-only duplicate finder. Never deletes, renames, or links input files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import defaultdict
from pathlib import Path


def find_duplicates(root: Path, min_size: int = 1, max_files: int = 100000) -> dict:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Choose an existing directory")
    sizes = defaultdict(list)
    errors, scanned = [], 0
    def on_error(exc):
        errors.append(f"Cannot inspect directory: {getattr(exc, 'filename', '')}")
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=on_error):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size < min_size:
                    continue
                scanned += 1
                if scanned > max_files:
                    raise ValueError("File limit exceeded; choose a smaller directory or raise --max-files")
                sizes[info.st_size].append(path)
            except OSError:
                errors.append(f"Cannot inspect file: {path}")
    groups = []
    for size, candidates in sorted(sizes.items()):
        if len(candidates) < 2:
            continue
        digests = defaultdict(list)
        for path in candidates:
            try:
                before = path.stat(follow_symlinks=False)
                if before.st_size != size:
                    raise OSError("File changed since enumeration")
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
                fd = os.open(path, flags)
                with os.fdopen(fd, "rb") as source:
                    opened = os.fstat(source.fileno())
                    if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                        raise OSError("File changed during scan")
                    digest = hashlib.sha256()
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                    after = os.fstat(source.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise OSError("File changed during hashing")
                digests[digest.hexdigest()].append(str(path))
            except OSError:
                errors.append(f"Unreadable or changed file: {path}")
        for digest, paths in sorted(digests.items()):
            if len(paths) > 1:
                groups.append({"sha256": digest, "bytes_each": size, "files": paths})
    return {"files_scanned": scanned, "duplicate_groups": groups, "errors": errors, "read_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=100000)
    args = parser.parse_args()
    if args.min_size < 0 or not 1 <= args.max_files <= 1000000:
        parser.error("Invalid size or file-count limit")
    try:
        report = find_duplicates(args.root, args.min_size, args.max_files)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
