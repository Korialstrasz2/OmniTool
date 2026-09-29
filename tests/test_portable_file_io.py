"""Regressions for Windows file metadata and flush behavior found by CI."""
import os
from pathlib import Path
import pytest
from PIL import Image
from omnitool_core.file_workbench import FileToolError, compare, read_regular
from omnitool_core.conversion import convert

@pytest.fixture
def picture(tmp_path):
    path = tmp_path/'input.png'
    Image.new('RGBA', (40, 20), (12, 34, 56, 100)).save(path)
    return path


@pytest.mark.parametrize('action', ['read', 'hash'])
def test_path_and_descriptor_metadata_are_checked_independently(tmp_path, monkeypatch, action):
    from types import SimpleNamespace
    import omnitool_core.file_workbench as module
    a, b = tmp_path/'a', tmp_path/'b'; a.mkdir(); b.mkdir()
    for root in (a, b): (root/'file').write_bytes(b'same bytes')
    original = module.os.fstat
    def descriptor_stat(fd):
        info = original(fd)
        fields = {name: getattr(info, name) for name in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}
        fields['st_ctime_ns'] += 100  # Stable API-specific representation, not a changing file.
        fields['st_mode'] ^= 0o111
        return SimpleNamespace(**fields)
    monkeypatch.setattr(module.os, 'fstat', descriptor_stat)
    if action == 'read':
        assert read_regular(a/'file', 100) == b'same bytes'
    else:
        assert compare(a, b)['counts']['same-content'] == 1


@pytest.mark.parametrize('change', ['identity', 'descriptor-time', 'path-time'])
def test_independent_stat_checks_still_reject_changes(tmp_path, monkeypatch, change):
    from types import SimpleNamespace
    import omnitool_core.file_workbench as module
    path = tmp_path/'file'; path.write_bytes(b'stable bytes')
    original = module.os.fstat; calls = 0
    def changed_stat(fd):
        nonlocal calls
        calls += 1
        info = original(fd)
        fields = {name: getattr(info, name) for name in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}
        if change == 'identity':
            fields['st_ino'] += 1
        elif change == 'descriptor-time' and calls > 1:
            fields['st_mtime_ns'] += 1
        elif change == 'path-time' and calls == 1:
            times = path.stat()
            os.utime(path, ns=(times.st_atime_ns, times.st_mtime_ns + 2_000_000_000))
        return SimpleNamespace(**fields)
    monkeypatch.setattr(module.os, 'fstat', changed_stat)
    with pytest.raises(FileToolError, match='changed'):
        read_regular(path, 100)


@pytest.mark.parametrize('suffix', ['.txt', '.png', '.b64', '.pdf', '.exe'])
def test_repeated_native_regular_reads(tmp_path, suffix):
    path = tmp_path/('fresh' + suffix)
    path.write_bytes(b'test bytes')
    for _ in range(10):
        assert read_regular(path, 100) == b'test bytes'



def test_staging_flush_uses_writable_nontruncating_handles(picture, tmp_path, monkeypatch):
    original = Path.open
    modes = []
    def observed_open(self, mode='r', *args, **kwargs):
        if self.parent.name.startswith('.omnitool-convert-') and 'r' in mode:
            modes.append(mode)
            assert mode == 'r+b'
        return original(self, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', observed_open)
    convert(picture, tmp_path/'output')
    assert len(modes) == 2  # PNG and conversion.json are both flushed before publication.
    assert picture.exists()
