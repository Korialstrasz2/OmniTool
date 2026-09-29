"""Previewed, journaled file-only lowercase renaming. No overwrite fallback.

Not a sandbox against a malicious same-user process. Stop other writers first.
Journals record names/identities, not file contents; maintain independent backups.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import time
import unicodedata
from pathlib import Path

SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.omnitool-local'}
LIMIT = 20000
MAX_RENAMES = 500
OP_ID = re.compile(r'[a-f0-9]{32}\Z')


class RenameError(ValueError):
    pass


def state_directory() -> Path:
    if os.name == 'nt':
        home = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData/Local')
    elif sys.platform == 'darwin':
        home = Path.home() / 'Library/Application Support'
    else:
        home = Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local/state')
    return home / 'OmniTool/renames'


def linked(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def identity(path: Path) -> list[int]:
    info = path.lstat()
    if linked(info) or not stat.S_ISREG(info.st_mode):
        raise RenameError('Only ordinary files can be renamed')
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def directory(path: Path) -> Path:
    """Reject symbolic links and Windows reparse points, including ancestors."""
    path = Path(os.path.abspath(path.expanduser()))
    for item in (*reversed(path.parents), path):
        info = item.lstat()
        if linked(info) or not stat.S_ISDIR(info.st_mode):
            raise RenameError('Use a real local directory, not a link or junction')
    return path


def folded(name: str) -> str:
    return unicodedata.normalize('NFC', name).casefold()


def plan(root: Path) -> dict:
    root = directory(root)
    if root == Path(root.anchor) or root == Path.home():
        raise RenameError('Choose a specific working folder, not a drive root or your home directory')
    rows, snapshot, skipped, conflicts = [], [], [], []
    scanned = 0
    def walk(parent: Path):
        nonlocal scanned
        info = parent.stat()
        snapshot.append([parent.relative_to(root).as_posix(), info.st_dev, info.st_ino])
        children = sorted(parent.iterdir(), key=lambda p: p.name)
        names: dict[str, list[str]] = {}
        for child in children:
            names.setdefault(folded(child.name), []).append(child.name)
        for child in children:
            scanned += 1
            if scanned > LIMIT:
                raise RenameError(f'Scan exceeds {LIMIT} entries; choose a smaller folder')
            rel = child.relative_to(root).as_posix()
            info = child.lstat()
            snapshot.append([rel, info.st_mode, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns])
            if ':' in child.name or '\\' in child.name:
                skipped.append({'path': rel, 'reason': 'nonportable path component'})
            elif linked(info):
                skipped.append({'path': rel, 'reason': 'link or reparse point'})
            elif stat.S_ISDIR(info.st_mode):
                if child.name in SKIP:
                    skipped.append({'path': rel, 'reason': 'excluded directory'})
                else:
                    walk(child)
            elif stat.S_ISREG(info.st_mode) and child.name != child.name.lower():
                lower = child.name.lower()
                if len(names.get(folded(lower), [])) > 1:
                    conflicts.append({'path': rel, 'reason': 'case/Unicode collision', 'names': names[folded(lower)]})
                target = child.with_name(lower).relative_to(root).as_posix()
                rows.append({'source': rel, 'target': target, 'identity': identity(child)})
            elif not stat.S_ISREG(info.st_mode):
                skipped.append({'path': rel, 'reason': 'special file'})
    walk(root)
    if len(rows) > MAX_RENAMES:
        raise RenameError('More than 500 renames in one operation; choose a smaller folder')
    payload = {'root': str(root), 'snapshot': snapshot, 'rows': rows, 'conflicts': conflicts}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {'root': str(root), 'fingerprint': fingerprint, 'rows': rows,
            'conflicts': conflicts, 'skipped': skipped, 'scanned': scanned}


def move_noreplace(source: Path, target: Path) -> None:
    """Atomic no-replace rename. Unsupported OS/filesystems fail closed."""
    if os.name == 'nt':
        os.rename(source, target)  # Windows raises FileExistsError if target exists.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        result = fn(-100, os.fsencode(source), -100, os.fsencode(target), 1)  # RENAME_NOREPLACE
    elif sys.platform == 'darwin' and hasattr(libc, 'renamex_np'):
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        result = fn(os.fsencode(source), os.fsencode(target), 4)  # RENAME_EXCL
    else:
        raise RenameError('This OS has no supported atomic no-overwrite rename primitive')
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def _sync_dir(path: Path) -> None:
    if os.name == 'nt':
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Renamer:
    def __init__(self, state: Path | None = None):
        self.state = state or state_directory()

    @contextlib.contextmanager
    def _lock(self):
        self.state.mkdir(parents=True, mode=0o700, exist_ok=True)
        directory(self.state)
        lockpath = self.state / '.lock'
        flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(lockpath, flags, 0o600)
        stream = os.fdopen(fd, 'r+b', buffering=0)
        try:
            if linked(lockpath.lstat()) or not stat.S_ISREG(os.fstat(fd).st_mode):
                raise RenameError('Unsafe operation lock')
            if os.name == 'nt':
                import msvcrt
                if os.fstat(fd).st_size == 0:
                    stream.write(b'0')
                stream.seek(0)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        except BlockingIOError:
            raise RenameError('Another rename or recovery operation is running') from None
        finally:
            stream.close()  # OS releases the lock even after a crashed process.

    def _save(self, document: dict) -> None:
        target = self.state / (document['id'] + '.json')
        temp = self.state / (document['id'] + '.' + secrets.token_hex(8) + '.tmp')
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as out:
                json.dump(document, out, ensure_ascii=True, indent=2)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, target)
            _sync_dir(self.state)
        finally:
            temp.unlink(missing_ok=True)

    def load(self, operation: str) -> dict:
        if not OP_ID.fullmatch(operation):
            raise RenameError('Invalid operation ID')
        path = self.state / (operation + '.json')
        if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
            raise RenameError('Invalid operation journal')
        doc = json.loads(path.read_text(encoding='utf-8'))
        if doc.get('version') != 1 or doc.get('id') != operation or not isinstance(doc.get('rows'), list) or len(doc['rows']) > LIMIT:
            raise RenameError('Invalid operation journal')
        root = directory(Path(doc['root']))
        info = root.stat()
        if [info.st_dev, info.st_ino] != doc['root_identity']:
            raise RenameError('The original root directory has been replaced')
        seen = set()
        for row in doc['rows']:
            for key in ('source', 'target', 'temp'):
                value = row[key]
                parts = value.split('/')
                if not value or '\\' in value or ':' in value or any(p in {'', '.', '..'} for p in parts):
                    raise RenameError('Unsafe journal path')
                directory((root / value).parent)
            if Path(row['source']).parent != Path(row['target']).parent or Path(row['source']).parent != Path(row['temp']).parent:
                raise RenameError('Invalid cross-directory rename in journal')
            kind = doc.get('kind', 'lowercase')
            if kind == 'lowercase':
                if Path(row['source']).name.lower() != Path(row['target']).name:
                    raise RenameError('Invalid lowercase target')
            elif kind == 'dual':
                from .file_workbench import component
                component(row['source']); component(row['target'])
                if Path(row['source']).suffix != Path(row['target']).suffix or row['source'] == row['target']:
                    raise RenameError('Invalid mapped target')
            else:
                raise RenameError('Unknown rename journal kind')
            if row['source'] in seen:
                raise RenameError('Duplicate journal entry')
            seen.add(row['source'])
        return doc

    def recent(self, limit: int | None = 100) -> list[dict]:
        if not self.state.exists():
            return []
        rows = []
        for path in sorted(self.state.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError('Invalid journal')
                doc = json.loads(path.read_text())
                rows.append({k: doc[k] for k in ('id', 'status', 'root', 'created')})
            except (OSError, ValueError, KeyError):
                rows.append({'id': path.stem, 'status': 'invalid-journal', 'root': '(unreadable journal)', 'created': 0})
        return rows

    @staticmethod
    def _location(root: Path, row: dict) -> str:
        # Exact directory spelling matters: exists() aliases source/target on Windows.
        directory((root / row['source']).parent)
        actual = {p.name for p in (root / row['source']).parent.iterdir()}
        matches = []
        for key in ('source', 'temp', 'target'):
            path = root / row[key]
            if path.name in actual and identity(path) == row['identity']:
                matches.append(key)
        if len(matches) != 1:
            raise RenameError('A file was changed, replaced, or has an ambiguous location; recovery stopped')
        return matches[0]

    def _move(self, doc: dict, row: dict, source: str, target: str):
        root = Path(doc['root'])
        directory((root / row[source]).parent)
        if identity(root / row[source]) != row['identity']:
            raise RenameError('A file changed during the operation')
        doc['pending'] = [row['source'], source, target]
        self._save(doc)  # Write-ahead: recovery can locate the file after any interruption.
        move_noreplace(root / row[source], root / row[target])
        _sync_dir((root / row[source]).parent)
        doc['pending'] = None
        self._save(doc)

    def apply(self, root: Path, fingerprint: str) -> dict:
        return self._apply(root, fingerprint, plan, 'lowercase')

    def apply_dual(self, reference: Path, root: Path, pairs: list[dict], fingerprint: str) -> dict:
        from .file_workbench import dual_plan
        return self._apply(root, fingerprint, lambda r: dual_plan(reference, r, pairs), 'dual')

    def _apply(self, root: Path, fingerprint: str, planner, kind: str) -> dict:
        with self._lock():
            if any(x['status'] in {'applying', 'recovering', 'recovery-required', 'invalid-journal'} for x in self.recent(None)):
                raise RenameError('Recover the unfinished operation before starting another')
            fresh = planner(root)
            if fresh['fingerprint'] != fingerprint:
                raise RenameError('Folder contents changed after preview; create a fresh preview')
            if fresh['conflicts']:
                raise RenameError('Resolve all collisions before applying')
            if not fresh['rows']:
                raise RenameError('There are no files to rename')
            root = Path(fresh['root'])
            if self.state.resolve().is_relative_to(root):
                raise RenameError('The journal directory must be outside the selected folder')
            op = secrets.token_hex(16)
            info = root.stat()
            doc = {'version': 1, 'kind': kind, 'id': op, 'created': time.time(), 'root': str(root),
                   'root_identity': [info.st_dev, info.st_ino], 'status': 'applying',
                   'rows': fresh['rows'], 'pending': None}
            for i, row in enumerate(doc['rows']):
                row['temp'] = (Path(row['source']).parent / f'.omnitool-{op}-{i}.tmp').as_posix()
            self._save(doc)
            try:
                # Stage every source before committing destinations (case-only safe).
                for row in doc['rows']:
                    self._move(doc, row, 'source', 'temp')
                for row in doc['rows']:
                    self._move(doc, row, 'temp', 'target')
                doc['status'] = 'applied'
                self._save(doc)
                return doc
            except Exception as exc:
                doc['status'] = 'recovery-required'
                self._save(doc)
                try:
                    self._recover(doc)
                except Exception:
                    raise RenameError(f'Operation {op} needs recovery. No overwrite was attempted; inspect its journal.') from exc
                raise RenameError(f'Operation {op} failed and original names were restored.') from exc

    def _recover(self, doc: dict) -> dict:
        root = directory(Path(doc['root']))
        locations = [self._location(root, row) for row in doc['rows']]
        # Refuse occupied original names before changing any names during recovery.
        for row, where in zip(doc['rows'], locations):
            names = {p.name for p in (root / row['source']).parent.iterdir()}
            if where != 'source' and Path(row['source']).name in names:
                raise RenameError('An original name is occupied; move it aside before recovery')
        # Validate every identity before changing any names during recovery.
        doc['status'] = 'recovering'
        self._save(doc)
        try:
            for row, where in reversed(list(zip(doc['rows'], locations))):
                if where == 'source':
                    continue
                if where == 'target':
                    self._move(doc, row, 'target', 'temp')
                self._move(doc, row, 'temp', 'source')
            doc['status'] = 'restored'
            self._save(doc)
            return doc
        except Exception:
            doc['status'] = 'recovery-required'
            self._save(doc)
            raise

    def recover(self, operation: str) -> dict:
        with self._lock():
            doc = self.load(operation)
            if doc['status'] == 'restored':
                return doc
            return self._recover(doc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preview', 'apply', 'undo'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--fingerprint', default='')
    parser.add_argument('--operation', default='')
    parser.add_argument('--confirm', default='')
    args = parser.parse_args()
    try:
        engine = Renamer()
        if args.action == 'undo':
            if args.confirm != 'UNDO':
                raise RenameError('Use --confirm UNDO after reviewing the journal')
            result = engine.recover(args.operation)
        else:
            if args.root is None:
                raise RenameError('--root is required')
            if args.action == 'preview':
                result = plan(args.root)
            else:
                if args.confirm != 'RENAME':
                    raise RenameError('Use --confirm RENAME only after reviewing the preview')
                result = engine.apply(args.root, args.fingerprint)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
