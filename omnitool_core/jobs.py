"""Bounded in-memory jobs. Trusted local tools, NOT an execution sandbox."""
from __future__ import annotations

import concurrent.futures
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .vault import materialize


@dataclass
class Job:
    id: str
    owner: str
    tool: str
    vault: str | None
    created: float = field(default_factory=time.time)
    status: str = "queued"
    output: str = ""
    exit_code: int | None = None
    process: subprocess.Popen | None = None
    canceled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    future: concurrent.futures.Future | None = None


class Jobs:
    def __init__(self, workers: int = 3, timeout: int = 1800):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tool")
        self.lock = threading.RLock()
        self.items: dict[str, Job] = {}
        self.timeout = timeout

    def submit(self, owner: str, tool: str, command: list[str], cwd: Path,
               vault: str | None = None, files: dict[str, bytes] | None = None,
               entrypoint: str | None = None) -> str:
        with self.lock:
            if sum(not j.done.is_set() for j in self.items.values()) >= 32:
                raise ValueError("Job queue is full; stop or finish an existing job")
            finished = sorted((j for j in self.items.values() if j.done.is_set()), key=lambda j: j.created)
            for old in finished[:max(0, len(self.items) - 99)]:
                self.items.pop(old.id, None)
            job = Job(secrets.token_hex(16), owner, tool, vault)
            self.items[job.id] = job
            job.future = self.pool.submit(self._run, job, list(command), cwd, files, entrypoint)
            return job.id

    def _run(self, job: Job, command: list[str], cwd: Path,
             files: dict[str, bytes] | None, entrypoint: str | None) -> None:
        temp = None
        try:
            if job.canceled.is_set():
                job.status = "canceled"
                return
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            # The workspace access token must not be inherited by tool processes.
            env.pop("OMNITOOL_ACCESS_TOKEN", None)
            if files is not None:
                temp = tempfile.TemporaryDirectory(prefix="omnitool-private-")
                cwd = Path(temp.name)
                materialize(files, cwd)
                files = None
                command = [sys.executable, "-B", str(cwd / str(entrypoint)), *command]
                child_tmp = cwd / ".runtime-tmp"
                child_tmp.mkdir(mode=0o700)
                for key in ("TMPDIR", "TMP", "TEMP"):
                    env[key] = str(child_tmp)
            if job.canceled.is_set():
                job.status = "canceled"
                return
            kwargs: dict[str, Any] = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            job.process = subprocess.Popen(command, cwd=str(cwd), env=env, shell=False,
                                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, **kwargs)
            job.status = "running"
            started = time.monotonic()

            def reader():
                assert job.process is not None and job.process.stdout is not None
                while chunk := job.process.stdout.read1(4096):
                    with self.lock:
                        job.output = (job.output + chunk.decode("utf-8", errors="replace"))[-65536:]

            read_thread = threading.Thread(target=reader, daemon=True)
            read_thread.start()
            while job.process.poll() is None:
                if job.canceled.is_set() or time.monotonic() - started > self.timeout:
                    job.canceled.set()
                    self._terminate(job.process)
                time.sleep(0.1)
            job.exit_code = job.process.wait()
            # Terminate children still in the POSIX process group before deleting private files.
            if os.name != "nt":
                try:
                    os.killpg(job.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            read_thread.join(timeout=2)
            job.status = "canceled" if job.canceled.is_set() else ("succeeded" if job.exit_code == 0 else "failed")
        except Exception as exc:
            # Do not emit exception strings containing paths or source from locked tools.
            job.output = "Tool execution failed; check local setup." if job.vault else f"Unable to start tool ({type(exc).__name__})."
            job.status = "failed"
        finally:
            if temp is not None:
                try:
                    temp.cleanup()
                except OSError:
                    job.output += "\nPrivate temporary files could not be removed. Check your OS temporary directory."
                    job.status = "cleanup-failed"
            job.done.set()

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
            try:
                process.kill()
            except OSError:
                pass

    def stop(self, owner: str, job_id: str) -> None:
        with self.lock:
            job = self.items.get(job_id)
            if job is None or job.owner != owner:
                raise KeyError("Unknown job")
            job.canceled.set()
            if job.future and job.future.cancel():
                job.status = "canceled"
                job.done.set()

    def forget_vault(self, owner: str, vault: str | None = None) -> bool:
        with self.lock:
            matching = [j for j in self.items.values() if j.owner == owner and j.vault and (vault is None or j.vault == vault)]
        for job in matching:
            self.stop(owner, job.id)
        for job in matching:
            if not job.done.wait(timeout=15) or job.status == "cleanup-failed":
                return False
        with self.lock:
            for job in matching:
                job.output = ""
                self.items.pop(job.id, None)
        return True

    def list(self, owner: str) -> list[dict]:
        with self.lock:
            return [{"id": j.id, "tool": j.tool, "vault": j.vault, "status": j.status,
                     "exit_code": j.exit_code, "output": j.output, "created": j.created}
                    for j in sorted(self.items.values(), key=lambda j: j.created, reverse=True) if j.owner == owner]

    def shutdown(self) -> None:
        with self.lock:
            all_jobs = list(self.items.values())
        for job in all_jobs:
            self.stop(job.owner, job.id)
        self.pool.shutdown(wait=True, cancel_futures=True)
