"""Bounded local reads and new-folder publication for content tools.

No original is opened for writing. This is not a same-user security sandbox.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import stat
import sys
from pathlib import Path


class ContentError(ValueError):
    """An actionable error safe to display in the authenticated workspace."""


def is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def directory(value: str | Path) -> Path:
    path = Path(os.path.abspath(Path(value).expanduser()))
    for part in (*reversed(path.parents), path):
        info = part.lstat()
        if is_link(info) or not stat.S_ISDIR(info.st_mode):
            raise ContentError('Choose an ordinary local directory, not a link or junction')
    return path


def portable_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ContentError('Invalid relative filename')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if any(c in value for c in '\\:"<>|?*') or value.startswith('/'):
        raise ContentError('Nonportable filename')
    parts = value.split('/')
    if any(p in {'', '.', '..'} or p.endswith((' ', '.')) or p.split('.')[0].upper() in reserved or
           any(ord(c) < 32 for c in p) for p in parts):
        raise ContentError('Nonportable filename')
    return Path(*parts)


def stamp(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)


def read_file(path: Path, limit: int) -> bytes:
    """Use path/path and handle/handle checks; Windows metadata representations differ."""
    path = directory(path.parent) / path.name
    before = path.lstat()
    if is_link(before) or not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ContentError('File is linked, unsupported, or exceeds its size limit')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
            raise ContentError('File changed while opening')
        data = stream.read(limit + 1)
        after_handle = os.fstat(stream.fileno())
    after_path = path.lstat()
    if len(data) > limit or len(data) != opened.st_size:
        raise ContentError('File exceeds the size limit or changed during reading')
    if stamp(before) != stamp(after_path) or stamp(opened) != stamp(after_handle):
        raise ContentError('File changed during reading')
    directory(path.parent)
    return data


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_output(value: str | Path, source: Path) -> Path:
    raw = Path(os.path.abspath(Path(value).expanduser()))
    portable_relative(raw.name)
    parent = directory(raw.parent)
    out = parent / raw.name
    if out.exists() or out.is_symlink():
        raise ContentError('Output must be a new folder; existing folders are never reused')
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ContentError('Output must be outside the source tree')
    return out


def publish_new_directory(source: Path, target: Path) -> None:
    """Publish a same-filesystem staging directory with no replacement fallback."""
    if os.name == 'nt':
        os.rename(source, target)  # Windows rejects an existing target.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        result = fn(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    elif sys.platform == 'darwin' and hasattr(libc, 'renamex_np'):
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        result = fn(os.fsencode(source), os.fsencode(target), 4)
    else:
        raise ContentError('Atomic no-overwrite publication is unsupported on this platform')
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))
