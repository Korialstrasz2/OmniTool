"""Actual shared job runner, including redirected Windows Python stdout."""
import sys
from omnitool_core.jobs import Jobs


def test_shared_jobs_use_utf8_despite_legacy_parent_encoding(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONIOENCODING', 'ascii')
    monkeypatch.setenv('PYTHONUTF8', '0')
    cwd = tmp_path / 'Cartella à 漢字'; cwd.mkdir()
    jobs = Jobs(workers=1, timeout=10)
    try:
        token = jobs.submit('owner', 'fixture', [sys.executable, '-c', "print('Caf\\u00e9 \\u6f22\\u5b57')"], cwd)
        item = jobs.items[token]
        assert item.done.wait(15)
        assert item.status == 'succeeded', item.output
        assert item.output.strip() == 'Café 漢字'
    finally:
        jobs.shutdown()
