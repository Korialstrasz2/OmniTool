"""Disposable audio fixtures and loopback HTTP only; never a real provider or library."""
import io
import json
import os
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from omnitool_core import lyrics_workbench as lyrics, prompt_creator as prompts
from omnitool_core.content_files import ContentError, digest, directory, portable_relative, read_file
from omnitool_core.content_tasks import ContentTasks

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def music(tmp_path):
    root = tmp_path / 'music'; root.mkdir()
    with zipfile.ZipFile(ROOT / 'tests/fixtures/generated-silence.zip') as z:
        (root / 'Track.mp3').write_bytes(z.read('silence.mp3'))
    (root / 'Track.lrc').write_text('[ar:Fixture Artist]\n[00:00.00]Original test line\n[00:00.10]Second test line', encoding='utf-8')
    return root

@pytest.fixture
def model(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        code, response, calls = 200, {'result': 'fixture-model'}, []
        def do_GET(self):
            self.calls.append((self.path, None)); self.reply()
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.calls.append((self.path, payload)); self.reply()
        def reply(self):
            raw = self.response if isinstance(self.response, bytes) else json.dumps(self.response).encode()
            self.send_response(self.code); self.send_header('Content-Length', str(len(raw))); self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    monkeypatch.setenv('OMNITOOL_PROMPT_URL', f'http://127.0.0.1:{server.server_port}')
    monkeypatch.setenv('OMNITOOL_PROMPT_KIND', 'kobold')
    yield Handler
    server.shutdown(); server.server_close(); thread.join()

@pytest.mark.parametrize('url', ['https://example.com', 'http://example.com:5001', 'http://0.0.0.0:5001',
    'http://169.254.169.254:8080', 'http://127.0.0.1', 'http://127.0.0.1:5001/path',
    'http://user:secret@127.0.0.1:5001', 'http://127.0.0.1:5001?url=x', 'file:///etc/passwd', 'http://[::]:5001'])
def test_backend_rejects_nonlocal_or_ambiguous_urls(url):
    with pytest.raises(ContentError): prompts.configuration({'OMNITOOL_PROMPT_URL': url})

@pytest.mark.parametrize('url', ['http://127.0.0.1:5001', 'http://localhost:5001', 'http://[::1]:5001'])
def test_backend_accepts_loopback(url):
    assert prompts.configuration({'OMNITOOL_PROMPT_URL': url})['port'] == 5001

def test_real_local_status_and_generation(model, monkeypatch):
    monkeypatch.setenv('HTTP_PROXY', 'http://192.0.2.1:9999')
    assert prompts.status()['model'] == 'fixture-model'
    model.response = {'results': [{'text': 'A generated fixture, not a real prompt.'}]}
    out = prompts.generate({'idea': 'test', 'system': 'instruction', 'max_tokens': 64, 'temperature': .2})
    assert out['content'].startswith('A generated')
    assert model.calls[-1][0] == '/api/v1/generate'
    assert model.calls[-1][1]['max_length'] == 64
    assert len(model.calls) == 2  # No speculative model probing or retry/fallback.

def test_local_chat_contract(model, monkeypatch):
    monkeypatch.setenv('OMNITOOL_PROMPT_KIND', 'local-chat'); monkeypatch.setenv('OMNITOOL_PROMPT_MODEL', 'fixture')
    model.response = {'choices': [{'message': {'content': 'response'}}]}
    assert prompts.generate({'idea': 'test'})['content'] == 'response'
    path, payload = model.calls[-1]
    assert path == '/v1/chat/completions' and payload['model'] == 'fixture' and payload['stream'] is False

@pytest.mark.parametrize('code', [301, 302, 401, 500])
def test_no_redirect_or_retry(model, code):
    model.code = code
    with pytest.raises(ContentError): prompts.status()
    assert len(model.calls) == 1

@pytest.mark.parametrize('response', [b'not-json', [], {'result': None}, b'x' * (prompts.MAX_RESPONSE + 1)])
def test_model_malformed_or_oversized(model, response):
    model.response = response
    with pytest.raises(ContentError): prompts.status()

@pytest.mark.parametrize('extra', [{'max_tokens': True}, {'max_tokens': 0}, {'temperature': float('nan')},
                                   {'temperature': 3}, {'url': 'http://evil'}, {'system': ''}, {'idea': 'x'*12001}])
def test_prompt_validation(extra):
    with pytest.raises(ContentError): prompts.generate({'idea': 'valid', **extra})

@pytest.mark.parametrize('path', ['../x', '/absolute', 'C:/file', 'a//b', 'a\\b', 'CON.txt', 'a:b', 'x"y', 'end.'])
def test_relative_path_guard(path):
    with pytest.raises(ContentError): portable_relative(path)

def test_scan_is_read_only_and_sidecars_remain_local(music, tmp_path, monkeypatch):
    before = {p.name: p.read_bytes() for p in music.iterdir()}
    monkeypatch.setattr(lyrics, 'fetch_lrclib', lambda *_: pytest.fail('Unexpected network request'))
    result = lyrics.scan(music)
    assert len(result['tracks']) == 1 and result['tracks'][0]['artist'] == 'Fixture Artist'
    preview = lyrics.prepare(result, [{'index': 0}], tmp_path / 'output', 'sidecar')
    assert preview['count'] == 1 and preview['rows'][0]['plain'] == 'Original test line\nSecond test line'
    assert not (tmp_path / 'output').exists()
    assert before == {p.name: p.read_bytes() for p in music.iterdir()}

def test_apply_writes_new_copies_and_reads_back(music, tmp_path):
    before = (music / 'Track.mp3').read_bytes()
    preview = lyrics.prepare(lyrics.scan(music), [{'index': 0}], tmp_path / 'output', 'sidecar')
    result = lyrics.apply(preview)
    assert result['count'] == 1 and result['originals_unchanged']
    assert (music / 'Track.mp3').read_bytes() == before
    assert (tmp_path / 'output/Track.lrc').read_text().startswith('[ar:')
    meta = lyrics.metadata((tmp_path / 'output/Track.mp3').read_bytes(), 'Track.mp3')
    assert meta['has_lyrics'] and meta['artist'] == 'Fixture Artist'
    assert not list(tmp_path.glob('.omnitool-lyrics-*.partial'))
    with pytest.raises(ContentError, match='new folder'): lyrics.apply(preview)

@pytest.mark.parametrize('ext', ['mp3', 'flac', 'm4a', 'ogg', 'opus'])
def test_real_tag_roundtrip_formats(tmp_path, ext):
    root = tmp_path / 'music'; root.mkdir()
    with zipfile.ZipFile(ROOT / 'tests/fixtures/generated-silence.zip') as z: data = z.read('silence.' + ext)
    path = root / ('Track.' + ext); path.write_bytes(data)
    path.with_suffix('.txt').write_text('Original fixture lyric only.', encoding='utf-8')
    preview = lyrics.prepare(lyrics.scan(root), [{'index': 0}], tmp_path / 'out', 'sidecar')
    assert preview['count'] == 1, preview
    lyrics.apply(preview)
    tagged = lyrics.metadata((tmp_path / 'out' / path.name).read_bytes(), path.name)
    assert tagged['has_lyrics'] and tagged['title'] == 'Fixture Track'
    assert path.read_bytes() == data

def test_existing_lyrics_protected(music, tmp_path):
    lyrics._tag_copy(music / 'Track.mp3', 'Existing fixture lyric')  # Disposable fixture only.
    scanned = lyrics.scan(music)
    assert lyrics.prepare(scanned, [{'index': 0}], tmp_path/'out', 'sidecar')['count'] == 0
    assert lyrics.prepare(scanned, [{'index': 0}], tmp_path/'out', 'sidecar', replace=True)['count'] == 1

def test_changed_source_stops_apply(music, tmp_path):
    preview = lyrics.prepare(lyrics.scan(music), [{'index': 0}], tmp_path/'out', 'sidecar')
    with (music/'Track.mp3').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ContentError, match='changed'): lyrics.apply(preview)
    assert not (tmp_path/'out').exists()

def test_tag_failure_cleans_staging(music, tmp_path, monkeypatch):
    preview = lyrics.prepare(lyrics.scan(music), [{'index': 0}], tmp_path/'out', 'sidecar')
    before = (music/'Track.mp3').read_bytes()
    def fail(*_): raise ContentError('Simulated tag failure')
    monkeypatch.setattr(lyrics, '_tag_copy', fail)
    with pytest.raises(ContentError): lyrics.apply(preview)
    assert not (tmp_path/'out').exists() and not list(tmp_path.glob('*.partial'))
    assert (music/'Track.mp3').read_bytes() == before

def test_output_created_during_publication_survives(music, tmp_path, monkeypatch):
    preview = lyrics.prepare(lyrics.scan(music), [{'index': 0}], tmp_path/'out', 'sidecar')
    original = lyrics.publish_new_directory
    def race(src, dst):
        dst.mkdir(); (dst/'keep.txt').write_text('keep'); original(src, dst)
    monkeypatch.setattr(lyrics, 'publish_new_directory', race)
    with pytest.raises(OSError): lyrics.apply(preview)
    assert (tmp_path/'out/keep.txt').read_text() == 'keep'

def test_missing_sidecar_is_itemized(music, tmp_path):
    (music/'Track.lrc').unlink()
    result = lyrics.prepare(lyrics.scan(music), [{'index': 0}], tmp_path/'out', 'sidecar')
    assert result['count'] == 0 and result['rows'][0]['reason']

def test_source_and_output_overlap_rejected(music):
    with pytest.raises(ContentError): lyrics.prepare(lyrics.scan(music), [{'index': 0}], music/'out', 'sidecar')

def test_duplicate_selection_rejected(music, tmp_path):
    with pytest.raises(ContentError): lyrics.prepare(lyrics.scan(music), [{'index': 0}, {'index': 0}], tmp_path/'out', 'sidecar')

def test_symlink_not_read(music, tmp_path):
    link = music/'Linked.mp3'
    try: link.symlink_to(music/'Track.mp3')
    except OSError: pytest.skip('No symlink privilege')
    result = lyrics.scan(music)
    assert len(result['tracks']) == 1 and result['warning_count'] == 1
    with pytest.raises(ContentError): read_file(link, lyrics.MAX_FILE)

def test_real_subprocess_scan_and_session_isolation(music):
    tasks = ContentTasks(ROOT)
    try:
        token = tasks.submit('alice', 'lyrics-scan', {'root': str(music)})
        item = tasks.get('alice', token); item['future'].result(timeout=15)
        assert item['state'] == 'succeeded', item['error']
        with pytest.raises(ContentError): tasks.get('bob', token)
        assert tasks.consume('alice', token, 'lyrics-scan')['tracks']
        with pytest.raises(ContentError): tasks.consume('alice', token, 'lyrics-scan')
        item['expires'] = time.monotonic() - 1
        with pytest.raises(ContentError): tasks.get('alice', token)
    finally: tasks.shutdown()

def test_task_clear_and_invalid_actions(music):
    tasks = ContentTasks(ROOT)
    try:
        with pytest.raises(ContentError): tasks.submit('a', 'execute-anything', {})
        token = tasks.submit('a', 'lyrics-scan', {'root': str(music)})
        tasks.get('a', token)['future'].result(timeout=15); tasks.clear('a')
        with pytest.raises(ContentError): tasks.get('a', token)
    finally: tasks.shutdown()

def test_lrclib_fixed_https_contract(monkeypatch):
    import requests
    calls = []
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_content(self, _):
            yield json.dumps({'id': 123, 'artistName': 'Fixture Artist', 'trackName': 'Fixture Track',
                              'duration': 10, 'plainLyrics': 'Original fixture text', 'syncedLyrics': None}).encode()
    class Session:
        trust_env = True
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def get(self, url, **kwargs):
            assert self.trust_env is False
            calls.append((url, kwargs)); return Response()
    monkeypatch.setattr(requests, 'Session', Session)
    result = lyrics.fetch_lrclib({'artist': 'Fixture Artist', 'title': 'Fixture Track', 'album': '', 'duration': 10})
    assert result['plain'] == 'Original fixture text'
    assert len(calls) == 1 and calls[0][0] == 'https://lrclib.net/api/get'
    assert calls[0][1]['allow_redirects'] is False and 'files' not in calls[0][1]
    with pytest.raises(ContentError, match='duration'):
        lyrics.fetch_lrclib({'artist': 'Fixture Artist', 'title': 'Fixture Track', 'duration': 100})
    with pytest.raises(ContentError, match='artist/title'):
        lyrics.fetch_lrclib({'artist': 'Wrong Artist', 'title': 'Fixture Track', 'duration': 10})

def test_private_worker_output_does_not_use_generic_environment(model, monkeypatch):
    model.response = {'results': [{'text': 'Worker fixture text'}]}
    monkeypatch.setenv('OMNITOOL_ACCESS_TOKEN', 'sensitive-test-marker')
    tasks = ContentTasks(ROOT)
    try:
        token = tasks.submit('a', 'prompt-generate', {'idea': 'fixture'})
        item = tasks.get('a', token); item['future'].result(timeout=15)
        assert item['state'] == 'succeeded', item['error']
        assert 'sensitive-test-marker' not in json.dumps(item['result'])
        assert len(model.calls) == 1
    finally: tasks.shutdown()

def test_worker_deadline(music):
    tasks = ContentTasks(ROOT, timeout=.001)
    try:
        token = tasks.submit('a', 'lyrics-scan', {'root': str(music)})
        item = tasks.get('a', token); item['future'].result(timeout=10)
        assert item['state'] == 'failed' and 'timed out' in item['error']
    finally: tasks.shutdown()
