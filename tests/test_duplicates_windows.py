"""Native and simulated Windows metadata regression coverage; fixtures only."""
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

SPEC = importlib.util.spec_from_file_location('duplicate_tool', Path(__file__).parents[1] / 'tools/duplicate-finder/main.py')
duplicates = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(duplicates)


def altered(info, **changes):
    fields = {k: getattr(info, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode')}
    return SimpleNamespace(**(fields | changes))


def test_native_duplicates_and_unicode_paths(tmp_path):
    root = tmp_path / 'Cartella con spazi e accenti à 漢字'; root.mkdir()
    for name in ['uno.txt', 'due.txt']: (root / name).write_bytes(b'\x00same\r\n\x1a')
    (root / 'tre.txt').write_bytes(b'different')
    result = duplicates.find_duplicates(root)
    assert not result['errors']
    assert len(result['duplicate_groups']) == 1
    assert result['duplicate_groups'][0]['sha256'] == hashlib.sha256(b'\x00same\r\n\x1a').hexdigest()
    assert result['files_scanned'] == 3 and result['read_only']
    assert (root / 'uno.txt').read_bytes() == b'\x00same\r\n\x1a'


def test_handle_and_path_timestamps_need_not_match(tmp_path, monkeypatch):
    path = tmp_path / 'file'; path.write_bytes(b'unchanged')
    native = os.fstat
    def descriptor(fd):
        info = native(fd)
        return altered(info, st_ctime_ns=info.st_ctime_ns + 1000, st_mtime_ns=info.st_mtime_ns // 100 * 100)
    monkeypatch.setattr(duplicates.os, 'fstat', descriptor)
    assert duplicates._hash_file(path, duplicates._stamp(path.lstat())) == hashlib.sha256(b'unchanged').hexdigest()


@pytest.mark.parametrize('change', ['identity', 'handle', 'path', 'contents'])
def test_metadata_guard_still_rejects_real_changes(tmp_path, monkeypatch, change):
    path = tmp_path / 'file'; path.write_bytes(b'unchanged')
    snapshot = duplicates._stamp(path.lstat())
    native_fstat, native_lstat = os.fstat, Path.lstat
    calls = 0
    def descriptor(fd):
        nonlocal calls
        calls += 1
        info = native_fstat(fd)
        if change == 'identity': return altered(info, st_ino=info.st_ino + 1)
        if calls == 2 and change == 'handle': return altered(info, st_mtime_ns=info.st_mtime_ns + 1)
        return info
    def pathstat(self, *args, **kwargs):
        info = native_lstat(self, *args, **kwargs)
        if self == path and calls >= 2 and change == 'path': return altered(info, st_mtime_ns=info.st_mtime_ns + 1)
        return info
    monkeypatch.setattr(duplicates.os, 'fstat', descriptor)
    monkeypatch.setattr(Path, 'lstat', pathstat)
    if change == 'contents': path.write_bytes(b'changed length')
    with pytest.raises(OSError): duplicates._hash_file(path, snapshot)


@pytest.mark.parametrize('options', [{'min_size': -1}, {'min_size': True}, {'max_files': 0}, {'max_files': True}])
def test_library_rejects_invalid_limits(tmp_path, options):
    with pytest.raises(ValueError): duplicates.find_duplicates(tmp_path, **options)


def test_reparse_points_are_excluded_without_following(tmp_path, monkeypatch):
    root = tmp_path / 'root'; root.mkdir()
    (root / 'ordinary').write_bytes(b'test')
    reparse = root / 'junction'; reparse.mkdir(); (reparse / 'not-scanned').write_bytes(b'test')
    native = Path.lstat
    def fake(self, *args, **kwargs):
        info = native(self, *args, **kwargs)
        if self == reparse:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info
    monkeypatch.setattr(Path, 'lstat', fake)
    assert duplicates.find_duplicates(root)['files_scanned'] == 1
    with pytest.raises(ValueError): duplicates.find_duplicates(reparse)


def test_errors_are_reported_not_false_matches(tmp_path, monkeypatch):
    for name in ('a', 'b'): (tmp_path / name).write_bytes(b'same')
    def fail(*args): raise PermissionError('fixture')
    monkeypatch.setattr(duplicates, '_hash_file', fail)
    result = duplicates.find_duplicates(tmp_path)
    assert len(result['errors']) == 2 and result['duplicate_groups'] == []
