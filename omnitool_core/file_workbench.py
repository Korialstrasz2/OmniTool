"""Bounded local inspection and explicit dual-folder rename plans.

All roots must be ordinary directories. No mutations take place in this module.
Stop concurrent writers: file identity checks are not an adversarial filesystem sandbox.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import stat
from collections import defaultdict
from pathlib import Path

from .rename import LIMIT, MAX_RENAMES, SKIP, RenameError, directory, folded, identity, linked


class FileToolError(ValueError):
    pass


def integer(value, low: int, high: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise FileToolError(f'{label} must be an integer from {low} to {high}')
    return value


def component(name: str) -> str:
    if (not isinstance(name, str) or not name or name in {'.', '..'} or len(name.encode('utf-8')) > 255
            or any(ord(c) < 32 or c in '/\\:<>"|?*' for c in name) or name.endswith((' ', '.'))):
        raise FileToolError('Use a portable filename, not a path')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if name.split('.')[0].upper() in reserved:
        raise FileToolError('Reserved device filename')
    return name


def stamp(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def inventory(root: Path, recursive: bool = True) -> dict:
    if not isinstance(recursive, bool):
        raise FileToolError('Recursive must be true or false')
    root = directory(root)
    rows, skipped, snapshot = [], [], []
    scanned = 0
    pending = [root]
    while pending:
        parent = pending.pop()
        directory(parent)
        info = parent.stat()
        snapshot.append([parent.relative_to(root).as_posix(), info.st_dev, info.st_ino])
        children = []
        with os.scandir(parent) as entries:
            for entry in entries:
                scanned += 1
                if scanned > LIMIT:
                    raise FileToolError(f'More than {LIMIT} entries; choose a smaller folder')
                children.append(Path(entry.path))
        for child in sorted(children):
            rel = child.relative_to(root).as_posix()
            info = child.lstat()
            snapshot.append([rel, *stamp(info)])
            reason = None
            try:
                component(child.name)
            except FileToolError:
                reason = 'nonportable filename'
            if linked(info):
                reason = 'link or reparse point'
            elif child.name in SKIP and stat.S_ISDIR(info.st_mode):
                reason = 'excluded development directory'
            elif not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                reason = 'special file'
            if reason:
                skipped.append({'path': rel, 'reason': reason})
                continue
            kind = 'directory' if stat.S_ISDIR(info.st_mode) else 'file'
            rows.append({'path': rel, 'kind': kind, 'bytes': info.st_size if kind == 'file' else 0,
                         'modified_ns': info.st_mtime_ns, '_stamp': stamp(info)})
            if kind == 'directory' and recursive:
                pending.append(child)
    rows.sort(key=lambda row: row['path'])
    return {'root': str(root), 'rows': rows, 'skipped': skipped, 'scanned': scanned,
            'fingerprint': hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()}


def dual_plan(reference: Path, root: Path, pairs: list[dict]) -> dict:
    """Each pair is {reference: filename, source: filename}; targets preserve the source suffix.

Top-level files only. Cycles/swaps and occupied targets are rejected, even when
another selected file would move away. This keeps rollback unambiguous.
"""
    left, right = inventory(reference, False), inventory(root, False)
    reference, root = Path(left['root']), Path(right['root'])
    if root.is_relative_to(reference) or reference.is_relative_to(root):
        raise FileToolError('Choose separate, non-overlapping reference and target folders')
    if root == Path(root.anchor) or root == Path.home():
        raise FileToolError('Choose a specific working folder')
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= MAX_RENAMES:
        raise FileToolError('Stage between 1 and 500 mappings')
    left_files = {r['path']: r for r in left['rows'] if r['kind'] == 'file'}
    right_files = {r['path']: r for r in right['rows'] if r['kind'] == 'file'}
    occupied = defaultdict(list)
    # Include skipped paths and directories: they can still collide with a target.
    for p in root.iterdir():
        occupied[folded(p.name)].append(p.name)
    rows, conflicts, targets, sources, references = [], [], set(), set(), set()
    normalized = []
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {'reference', 'source'}:
            raise FileToolError('Invalid mapping')
        ref, src = component(pair['reference']), component(pair['source'])
        if ref not in left_files or src not in right_files:
            raise FileToolError('A selected ordinary file is missing; reload both folders')
        if src in sources or ref in references:
            raise FileToolError('Use each source and reference file only once')
        sources.add(src); references.add(ref)
        target = component(Path(ref).stem + Path(src).suffix)
        normalized.append({'reference': ref, 'source': src})
        if src == target:
            continue
        key = folded(target)
        if key in targets or any(name != src for name in occupied[key]):
            conflicts.append({'path': src, 'target': target, 'reason': 'occupied or duplicate target'})
        targets.add(key)
        rows.append({'reference': ref, 'source': src, 'target': target, 'identity': identity(root / src)})
    payload = {'left': left['fingerprint'], 'right': right['fingerprint'], 'reference': str(reference),
               'root': str(root), 'pairs': normalized}
    return {'reference': str(reference), 'root': str(root), 'pairs': normalized, 'rows': rows,
            'conflicts': conflicts, 'skipped': left['skipped'] + right['skipped'],
            'scanned': left['scanned'] + right['scanned'],
            'fingerprint': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


@contextmanager
def _read_handle(path: Path, before: os.stat_result):
    """Compare path and descriptor snapshots separately, joining by file identity.

    Path-stat and descriptor-stat metadata are not interchangeable on every OS.
    Keep full timestamp/mode checks within each API instead of accepting a
    cross-API mismatch or discarding change checks altogether.
    """
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0))
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (linked(before) or linked(opened) or not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size)
                or stamp(path.lstat()) != stamp(before)):
            raise FileToolError('File changed while opening')
        yield stream
        if stamp(os.fstat(stream.fileno())) != stamp(opened) or stamp(path.lstat()) != stamp(before):
            raise FileToolError('File changed while reading; run the operation again')


def read_regular(path: Path, limit: int) -> bytes:
    """Read a bounded, ordinary file with identity checks before/after reading."""
    directory(path.parent)
    before = path.lstat()
    if linked(before) or not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise FileToolError('Input must be an ordinary file within the size limit')
    with _read_handle(path, before) as stream:
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise FileToolError('Input exceeds the size limit')
    return data


def _digest(root: Path, row: dict, budget: list[int]) -> str:
    path = root / row['path']
    directory(path.parent)
    before = path.lstat()
    if stamp(before) != row['_stamp']:
        raise FileToolError('A file changed during comparison; run it again')
    digest = hashlib.sha256()
    with _read_handle(path, before) as source:
        while chunk := source.read(min(1024 * 1024, budget[0] + 1)):
            budget[0] -= len(chunk)
            if budget[0] < 0:
                raise FileToolError('Content-hash byte budget exceeded; increase it or narrow the folders')
            digest.update(chunk)
    return digest.hexdigest()


def compare(left: Path, right: Path, mode: str = 'content', recursive: bool = True,
            hash_budget_mib: int = 1024) -> dict:
    if mode not in {'content', 'paths', 'stems'}:
        raise FileToolError('Choose content, paths, or stems mode')
    integer(hash_budget_mib, 1, 4096, 'Hash budget (MiB)')
    a, b = inventory(left, recursive), inventory(right, recursive)
    groups = [defaultdict(list), defaultdict(list)]
    for group, scan in zip(groups, (a, b)):
        for row in scan['rows']:
            key = row['path']
            if mode == 'stems':
                if row['kind'] != 'file':
                    continue
                key = str(Path(key).with_suffix(''))
            group[key].append(row)
    budget = [hash_budget_mib * 1024 * 1024]
    rows = []
    for key in sorted(set(groups[0]) | set(groups[1])):
        aa, bb = groups[0].get(key, []), groups[1].get(key, [])
        item = {'path': key, 'left': [r['path'] for r in aa], 'right': [r['path'] for r in bb]}
        if len(aa) > 1 or len(bb) > 1:
            status = 'ambiguous'
        elif not aa:
            status = 'right-only'
        elif not bb:
            status = 'left-only'
        elif aa[0]['kind'] != bb[0]['kind']:
            status = 'type-conflict'
        elif aa[0]['kind'] == 'directory':
            status = 'directory-both'
        elif mode != 'content':
            status = 'same-stem' if mode == 'stems' else 'same-path'
        elif aa[0]['bytes'] != bb[0]['bytes']:
            status = 'different-content'
        else:
            x, y = _digest(Path(a['root']), aa[0], budget), _digest(Path(b['root']), bb[0], budget)
            item.update(left_sha256=x, right_sha256=y)
            status = 'same-content' if x == y else 'different-content'
        item['status'] = status
        rows.append(item)
    # Do not label a changed directory tree as a completed snapshot.
    for scan in (a, b):
        if inventory(Path(scan['root']), recursive)['fingerprint'] != scan['fingerprint']:
            raise FileToolError('Folder contents changed during the scan; compare again after stopping writers')
    counts = {state: sum(r['status'] == state for r in rows) for state in sorted({r['status'] for r in rows})}
    return {'left_root': a['root'], 'right_root': b['root'], 'mode': mode, 'recursive': recursive,
            'rows': rows, 'counts': counts, 'hash_bytes_read': hash_budget_mib * 1024 * 1024 - budget[0],
            'skipped': {'left': a['skipped'], 'right': b['skipped']},
            'scope_complete': not (a['skipped'] or b['skipped']), 'read_only': True}


def compare_main():
    parser = argparse.ArgumentParser(description='Read-only folder comparison; JSON report on stdout.')
    parser.add_argument('left', type=Path); parser.add_argument('right', type=Path)
    parser.add_argument('--mode', choices=['content', 'paths', 'stems'], default='content')
    parser.add_argument('--top-level', action='store_true')
    parser.add_argument('--hash-budget-mib', type=int, default=1024)
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.left, args.right, args.mode, not args.top_level, args.hash_budget_mib), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


def dual_main():
    from .rename import Renamer
    parser = argparse.ArgumentParser(description='Explicit dual renaming. Prefer the workspace for thumbnail matching.')
    parser.add_argument('action', choices=['preview', 'apply', 'undo'])
    parser.add_argument('--reference', type=Path); parser.add_argument('--target', type=Path)
    parser.add_argument('--mapping', type=Path, help='JSON list of {reference: filename, source: filename}')
    parser.add_argument('--fingerprint', default=''); parser.add_argument('--operation', default='')
    parser.add_argument('--confirm', default='')
    args = parser.parse_args()
    try:
        engine = Renamer()
        if args.action == 'undo':
            if args.confirm != 'UNDO':
                raise FileToolError('Use --confirm UNDO after reviewing the journal')
            result = engine.recover(args.operation)
        else:
            if args.reference is None or args.target is None or args.mapping is None:
                raise FileToolError('--reference, --target, and --mapping are required')
            pairs = json.loads(read_regular(args.mapping, 65536))
            preview = dual_plan(args.reference, args.target, pairs)
            if args.action == 'preview':
                result = preview
            else:
                if args.confirm != f"RENAME {len(preview['rows'])}":
                    raise FileToolError(f"Use --confirm 'RENAME {len(preview['rows'])}' after reviewing the plan")
                result = engine.apply_dual(args.reference, args.target, pairs, args.fingerprint)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
