"""Bounded asynchronous workbench tasks and session-scoped preview leases."""
from __future__ import annotations
import concurrent.futures
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .file_workbench import FileToolError


class Leases:
    def __init__(self):
        self.lock = threading.RLock()
        self.items = {}

    def put(self, owner: str, kind: str, value: dict) -> str:
        with self.lock:
            self._prune()
            if len(self.items) >= 32 or sum(k[0] == owner for k in self.items) >= 8:
                raise FileToolError('Too many open previews; release one or wait for its ten-minute expiry')
            token = secrets.token_hex(16)
            self.items[(owner, token)] = {'kind': kind, 'value': value, 'expires': time.monotonic() + 600}
            return token

    def _prune(self):
        for key, item in list(self.items.items()):
            if item['expires'] <= time.monotonic():
                del self.items[key]

    def get(self, owner: str, token: str, kind: str, consume: bool = False):
        if not isinstance(token, str):
            raise FileToolError('Invalid preview token')
        with self.lock:
            self._prune()
            item = self.items.get((owner, token))
            if not item or item['kind'] != kind:
                raise FileToolError('Preview expired or belongs to another session; preview again')
            if consume:
                del self.items[(owner, token)]
            return item['value']

    def discard(self, owner: str, token: str):
        with self.lock:
            self.items.pop((owner, token), None)


class FileTasks:
    def __init__(self, base: Path):
        self.base = base
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix='file-workbench')
        self.lock = threading.RLock()
        self.items = {}
        self.closed = False

    def submit(self, owner: str, request: dict) -> str:
        with self.lock:
            if self.closed:
                raise FileToolError('Workspace is shutting down')
            now = time.monotonic()
            for key, task in list(self.items.items()):
                if task['state'] not in {'queued', 'running'} and task['finished'] and now - task['finished'] > 600:
                    del self.items[key]
            if len(self.items) >= 16:
                raise FileToolError('Task limit reached; dismiss completed reports before starting more')
            token = secrets.token_hex(16)
            task = {'owner': owner, 'state': 'queued', 'result': None, 'error': '', 'process': None,
                    'cancel': threading.Event(), 'finished': 0, 'action': request['action'], 'request': dict(request)}
            self.items[token] = task
            task['future'] = self.pool.submit(self._run, task, dict(request))
            return token

    @staticmethod
    def terminate(process):
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], timeout=10,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()

    def _run(self, task, request):
        process = None
        try:
            with self.lock:
                if task['cancel'].is_set():
                    task['state'] = 'canceled'; return
                task['state'] = 'running'
                env = os.environ.copy()
                env.pop('OMNITOOL_ACCESS_TOKEN', None)
                env['PYTHONDONTWRITEBYTECODE'] = '1'
                kwargs = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
                process = subprocess.Popen([sys.executable, '-B', '-m', 'omnitool_core.file_worker'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    cwd=self.base, env=env, **kwargs)
                task['process'] = process
            # Worker input and output are bounded by our own code, not arbitrary tool executables.
            output, _ = process.communicate(json.dumps(request).encode(), timeout=120)
            if len(output) > 16 * 1024 * 1024:
                raise FileToolError('Report too large; narrow the selected folders')
            result = json.loads(output)
            with self.lock:
                if task['cancel'].is_set():
                    task['state'] = 'canceled'
                elif process.returncode or not result.get('ok'):
                    task['state'] = 'failed'; task['error'] = result.get('error', 'Worker failed')
                else:
                    task['state'] = 'succeeded'; task['result'] = result['result']
        except subprocess.TimeoutExpired:
            if process:
                self.terminate(process); process.communicate()
            task['state'] = 'failed'; task['error'] = 'Operation exceeded 120 seconds; narrow the input. A conversion staging folder may remain.'
        except Exception as exc:
            task['state'] = 'canceled' if task['cancel'].is_set() else 'failed'
            task['error'] = 'Worker stopped or failed: ' + str(exc)[:500]
        finally:
            if process and process.poll() is None:
                self.terminate(process); process.wait()
            with self.lock:
                task['process'] = None; task['finished'] = time.monotonic()

    def get(self, owner: str, token: str) -> dict:
        with self.lock:
            if not isinstance(token, str):
                raise FileToolError('Invalid task ID')
            task = self.items.get(token)
            if task and task['state'] not in {'queued', 'running'} and task['finished'] and time.monotonic() - task['finished'] > 600:
                del self.items[token]
                task = None
            if not task or task['owner'] != owner:
                raise FileToolError('Task not found in this session')
            return task

    def dismiss(self, owner: str, token: str):
        with self.lock:
            task = self.get(owner, token)
            if task['state'] in {'queued', 'running'}:
                task['cancel'].set()
                if task['future'].cancel():
                    task['state'] = 'canceled'; task['finished'] = time.monotonic()
                if task['process']:
                    self.terminate(task['process'])
            else:
                del self.items[token]

    def shutdown(self):
        with self.lock:
            self.closed = True
            work = [(t['owner'], key) for key, t in self.items.items()]
        for owner, token in work:
            try:
                self.dismiss(owner, token)
            except FileToolError:
                pass
        self.pool.shutdown(wait=True, cancel_futures=True)
