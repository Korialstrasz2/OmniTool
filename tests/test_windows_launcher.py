"""Actual cmd.exe launcher acceptance on a disposable checkout and venv."""
import http.client
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Native Windows cmd.exe launcher acceptance')
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def checkout(tmp_path_factory):
    root = tmp_path_factory.mktemp('launcher') / 'OmniTool (test) à 漢字'
    shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__', '.pytest_cache', '.omnitool-local'))
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(root / '.venv')], check=True, timeout=60)
    return root


def environment():
    result = os.environ.copy()
    result.update(OMNITOOL_NO_PAUSE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    return result


def test_native_batch_diagnostics_are_read_only(checkout):
    result = subprocess.run(['cmd.exe', '/d', '/c', 'start.bat', '--diagnose'], cwd=checkout,
                            env=environment(), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout.decode('utf-8', errors='replace')
    assert b'No model/provider was contacted' in result.stdout
    assert b'Installing or repairing' not in result.stdout


def test_native_batch_serves_authenticated_workspace(checkout):
    probe = socket.socket(); probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]; probe.close()
    env = environment() | {'OMNITOOL_PORT': str(port)}
    process = subprocess.Popen(['cmd.exe', '/d', '/c', 'start.bat', '--no-browser'], cwd=checkout, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = queue.Queue()
    reader = threading.Thread(target=lambda: [lines.put(line) for line in iter(process.stdout.readline, b'')], daemon=True)
    reader.start()
    try:
        import time
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            line = lines.get(timeout=max(.1, deadline - time.monotonic()))
            if line.startswith(b'OmniTool: http://'):
                break
        else: pytest.fail('Launcher never reported a bound server')
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            connection.request('GET', '/login'); response = connection.getresponse()
            assert response.status == 200 and b'Local access code' in response.read()
            connection.request('GET', '/api/jobs'); response = connection.getresponse()
            assert response.status == 401; response.read()
        finally: connection.close()
    finally:
        # Dispose of this test-only process tree, never an existing desktop app.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, timeout=10)
        process.wait(timeout=10); reader.join(timeout=2); process.stdout.close()


def test_native_occupied_port_is_not_taken_over(checkout):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0)); listener.listen(1)
    try:
        env = environment() | {'OMNITOOL_PORT': str(listener.getsockname()[1])}
        result = subprocess.run(['cmd.exe', '/d', '/c', 'start.bat', '--no-browser'], cwd=checkout,
                                env=env, capture_output=True, timeout=30)
        assert result.returncode == 1
        assert b'occupied or unavailable' in result.stdout
        assert listener.fileno() != -1
    finally: listener.close()


def test_windows_listener_stays_exclusive_after_waitress_wrap():
    from waitress import create_server
    from omnitool_core.startup import bound_listener
    listener = bound_listener(0)
    server = None
    sockets = {}
    def application(environ, start_response):
        start_response('200 OK', [('Content-Type', 'text/plain')])
        return [b'test']
    try:
        server = create_server(application, sockets=[listener], map=sockets)
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE) == 1
        contender = socket.socket()
        try:
            contender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError):
                contender.bind(listener.getsockname())
        finally:
            contender.close()
    finally:
        if server is not None:
            server.close()
            server.task_dispatcher.shutdown()
        for channel in list(sockets.values()):
            channel.close()
        listener.close()
