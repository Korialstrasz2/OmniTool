"""Bounded session-owned subprocess tasks for prompts and lyrics; no disk job logs."""
from __future__ import annotations

import atexit
import concurrent.futures
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

from .content_files import ContentError

MAX_WIRE = 8 * 1024 * 1024
ACTIONS = {'prompt-status', 'prompt-generate', 'lyrics-scan', 'lyrics-prepare', 'lyrics-apply'}


class ContentTasks:
    def __init__(self, base: Path, workers=2, timeout=180, ttl=600):
        self.base, self.timeout, self.ttl = base, timeout, ttl
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix='content')
        self.mutex = threading.RLock()
        self.items: dict[str, dict] = {}
        self.closed = False
        atexit.register(self.shutdown)

    def _purge(self):
        now = time.monotonic()
        for key, item in list(self.items.items()):
            if item['state'] not in {'queued', 'running'} and item['expires'] <= now:
                self.items.pop(key, None)

    def submit(self, owner: str, action: str, payload: dict) -> str:
        if action not in ACTIONS:
            raise ContentError('Unsupported content action')
        wire = json.dumps({'action': action, 'payload': payload}, ensure_ascii=True).encode('utf-8')
        if len(wire) > MAX_WIRE:
            raise ContentError('Batch is too large; select fewer tracks')
        with self.mutex:
            self._purge()
            active = [v for v in self.items.values() if v['state'] in {'queued', 'running'}]
            if self.closed or len(self.items) >= 64 or len(active) >= 8 or sum(v['owner'] == owner for v in active) >= 4:
                raise ContentError('Task capacity reached; finish or clear an existing task')
            # Avoid parallel provider bursts or duplicate expensive generations.
            if action in {'lyrics-prepare', 'prompt-generate'} and any(v['action'] == action for v in active):
                raise ContentError('This operation is already queued or running; wait for its result')
            token = secrets.token_hex(16)
            item = {'owner': owner, 'action': action, 'state': 'queued', 'result': None, 'error': '',
                    'cancel': threading.Event(), 'process': None, 'consumed': False,
                    'expires': time.monotonic() + self.ttl}
            self.items[token] = item
            item['future'] = self.pool.submit(self._run, item, wire)
            return token

    def _run(self, item: dict, wire: bytes):
        process = None
        try:
            with self.mutex:
                if item['cancel'].is_set():
                    item['state'] = 'canceled'
                    return
                allowed = {'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'TMPDIR', 'HOME', 'USERPROFILE',
                           'LOCALAPPDATA', 'APPDATA', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH',
                           'OMNITOOL_PROMPT_URL', 'OMNITOOL_PROMPT_KIND', 'OMNITOOL_PROMPT_MODEL', 'KOBOLD_HOST'}
                env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
                env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
                process = subprocess.Popen([sys.executable, '-B', '-m', 'omnitool_core.content_worker'],
                                           cwd=self.base, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.DEVNULL, shell=False)
                item.update(process=process, state='running')
            deadline = time.monotonic() + self.timeout
            first = True
            while True:
                if item['cancel'].is_set() or time.monotonic() >= deadline:
                    process.kill()
                    process.communicate()
                    raise ContentError('Canceled or timed out. Check for a .partial output folder; originals were not changed.')
                try:
                    out, _ = process.communicate(input=wire if first else None, timeout=min(.2, max(.001, deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    first = False
            if len(out) > MAX_WIRE:
                raise ContentError('Result exceeds the output limit')
            response = json.loads(out)
            if not isinstance(response, dict):
                raise ContentError('Invalid worker result')
            if process.returncode or 'error' in response:
                raise ContentError(response.get('error', 'Content worker failed'))
            with self.mutex:
                if item['cancel'].is_set():
                    item.update(state='canceled', result=None)
                else:
                    item.update(state='succeeded', result=response)
        except Exception as exc:
            with self.mutex:
                item.update(state='canceled' if item['cancel'].is_set() else 'failed',
                            error=str(exc) if isinstance(exc, ContentError) else 'Worker failed; check local dependencies and permissions')
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            with self.mutex:
                item['process'] = None
                item['expires'] = time.monotonic() + self.ttl

    def get(self, owner: str, token: str) -> dict:
        with self.mutex:
            self._purge()
            item = self.items.get(token) if isinstance(token, str) else None
            if item is None or item['owner'] != owner:
                raise ContentError('Task expired or belongs to another session')
            return item

    def consume(self, owner: str, token: str, action: str) -> dict:
        with self.mutex:
            item = self.get(owner, token)
            if item['action'] != action or item['state'] != 'succeeded' or item['consumed']:
                raise ContentError('Preview is not usable or has already been applied')
            item['consumed'] = True
            return item['result']

    def stop(self, owner: str, token: str):
        with self.mutex:
            item = self.get(owner, token)
            if item['state'] not in {'queued', 'running'}:
                return
            item['cancel'].set()
            if item['future'].cancel():
                item.update(state='canceled', expires=time.monotonic() + self.ttl)

    def clear(self, owner: str):
        with self.mutex:
            for token, item in list(self.items.items()):
                if item['owner'] != owner:
                    continue
                if item['state'] in {'queued', 'running'}:
                    self.stop(owner, token)
                else:
                    self.items.pop(token)

    def shutdown(self):
        with self.mutex:
            if self.closed:
                return
            self.closed = True
            for item in self.items.values():
                item['cancel'].set()
        self.pool.shutdown(wait=True, cancel_futures=True)
