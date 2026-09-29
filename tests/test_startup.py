import errno
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from omnitool_core import bootstrap, startup


@pytest.mark.parametrize('value', ['0', '80', '65536', 'foo', '-1', True, '１２３４'])
def test_port_validation(value):
    with pytest.raises(ValueError): startup.port_number(value)


def test_port_number():
    assert startup.port_number('5000') == 5000


def test_browser_opens_after_real_login_response_only():
    called, requested = [], threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == '/login'
            requested.set(); self.send_response(200); self.send_header('Content-Length', '0'); self.end_headers()
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    def opener(url):
        assert requested.is_set(); called.append(url); return True
    try:
        assert startup.open_when_ready(server.server_port, threading.Event(), opener=opener, timeout=2)
        assert called == [f'http://127.0.0.1:{server.server_port}/']
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_no_browser_after_cancel():
    stopped = threading.Event(); stopped.set()
    assert not startup.open_when_ready(5000, stopped, opener=lambda *_: pytest.fail('opened'))


def test_port_conflict_does_not_open_browser_or_kill_process(monkeypatch):
    app = SimpleNamespace(extensions={'omnitool_access_token': 'test-code'})
    monkeypatch.setattr(startup, 'open_when_ready', lambda *_: pytest.fail('opened'))
    def conflict(*args, **kwargs): raise OSError(errno.EADDRINUSE, 'fixture')
    with pytest.raises(startup.StartupError, match='occupied'):
        startup.serve_local(app, port=0, open_browser=True, server_factory=conflict)


def test_startup_closes_server_and_workers_on_exit(monkeypatch):
    calls = []
    server = SimpleNamespace(effective_port='5000', run=lambda: calls.append('run'),
                             close=lambda: calls.append('close'),
                             task_dispatcher=SimpleNamespace(shutdown=lambda: calls.append('dispatcher')))
    def factory(app, **kwargs):
        assert kwargs['sockets'][0].getsockname()[0] == '127.0.0.1'
        assert not {'host', 'port'} & kwargs.keys()
        assert kwargs['max_request_body_size'] == 65536
        calls.append('bind'); return server
    app = SimpleNamespace(extensions={'omnitool_access_token': 'test-code'})
    startup.serve_local(app, port=0, server_factory=factory)
    assert calls == ['bind', 'run', 'close', 'dispatcher']


def test_main_stops_app_workers_even_on_invalid_port(monkeypatch):
    stopped = []
    app = SimpleNamespace(extensions={'omnitool_shutdown': lambda: stopped.append(True)})
    assert startup.main(app, ['--port', 'invalid']) == 1
    assert stopped == [True]


@pytest.mark.parametrize('installed', ['3.0.0', '4.0.0', '3.1.0rc1', 'unknown'])
def test_bootstrap_rejects_wrong_or_ambiguous_versions(tmp_path, monkeypatch, installed):
    path = tmp_path / 'requirements.txt'; path.write_text('Flask>=3.1,<4\n')
    monkeypatch.setattr(bootstrap.importlib.metadata, 'version', lambda _: installed)
    assert bootstrap.requirement_problems(path)


def test_bootstrap_accepts_matching_release(tmp_path, monkeypatch):
    path = tmp_path / 'requirements.txt'; path.write_text('# core\nFlask>=3.1,<4\n')
    monkeypatch.setattr(bootstrap.importlib.metadata, 'version', lambda _: '3.1.3')
    assert bootstrap.requirement_problems(path) == []


def test_bootstrap_does_not_silently_ignore_new_requirement_syntax(tmp_path):
    path = tmp_path / 'requirements.txt'; path.write_text('-r other.txt\n')
    assert 'Cannot validate' in bootstrap.requirement_problems(path)[0]


def test_listener_is_bound_only_to_loopback():
    listener = startup.bound_listener(0)
    try:
        assert listener.getsockname()[0] == '127.0.0.1'
        assert listener.getsockname()[1] > 0
    finally:
        listener.close()


def test_factory_failure_releases_owned_listener(monkeypatch):
    sockets = []
    app = SimpleNamespace(extensions={'omnitool_access_token': 'test-code'})
    def fail(*args, **kwargs):
        sockets.extend(kwargs['sockets'])
        raise OSError(errno.EADDRINUSE, 'fixture')
    with pytest.raises(startup.StartupError):
        startup.serve_local(app, port=0, server_factory=fail)
    assert sockets[0].fileno() == -1
