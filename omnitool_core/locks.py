"""Session-scoped, expiring access to encrypted bundles."""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

from .jobs import Jobs
from .vault import HEADER_SIZE, MAX_ENVELOPE, VaultError, decrypt, unpack

VAULT_ID = re.compile(r"[0-9a-f]{32}\Z")


class Locked(Exception):
    pass


class Vaults:
    def __init__(self, root: Path, jobs: Jobs, ttl: int = 600):
        self.root, self.jobs, self.ttl = root, jobs, ttl
        self.mutex = threading.RLock()
        self.opened: dict[tuple[str, str], dict] = {}
        self.stopped = threading.Event()
        self.worker = threading.Thread(target=self._expire, daemon=True, name="vault-expiry")
        self.worker.start()

    def _path(self, id: str) -> Path:
        if not VAULT_ID.fullmatch(id):
            raise KeyError("Unknown locked bundle")
        path = self.root / f"{id}.otvault"
        if path.is_symlink() or not path.is_file():
            raise KeyError("Unknown locked bundle")
        return path

    def list(self, owner: str) -> list[dict]:
        result = []
        for path in sorted(self.root.glob("*.otvault")):
            if not VAULT_ID.fullmatch(path.stem) or path.is_symlink():
                continue
            with self.mutex:
                item = self.opened.get((owner, path.stem))
                active = item is not None and item["expires"] > time.monotonic() and not item.get("closing")
                row = {"id": path.stem, "locked": not active}
                if active:
                    row.update(name=item["meta"]["name"], kind=item["meta"]["kind"],
                               seconds_left=max(0, int(item["expires"] - time.monotonic())))
                # Locked rows deliberately have NO name, type, child count, or descriptions.
                result.append(row)
        return result

    def unlock(self, owner: str, id: str, passphrase: str) -> None:
        path = self._path(id)
        # Serialize memory-heavy KDFs and cap simultaneous plaintext bundles.
        with self.mutex:
            if len(self.opened) >= 8 and (owner, id) not in self.opened:
                raise VaultError("Lock another bundle before unlocking more")
            if path.stat().st_size > MAX_ENVELOPE + HEADER_SIZE + 16:
                raise VaultError("Bundle too large")
            with path.open("rb") as stream:
                blob = stream.read(MAX_ENVELOPE + HEADER_SIZE + 17)
            meta, files = unpack(decrypt(blob, passphrase))
            self.opened[(owner, id)] = {"meta": meta, "files": files, "expires": time.monotonic() + self.ttl}

    def get(self, owner: str, id: str) -> dict:
        with self.mutex:
            item = self.opened.get((owner, id))
            if item is None or item["expires"] <= time.monotonic() or item.get("closing"):
                raise Locked("Unlock this bundle first")
            return item

    def run(self, owner: str, id: str, tool_index: int, args: list[str]) -> str:
        with self.mutex:
            item = self.get(owner, id)
            tool = item["meta"]["tools"][tool_index]
            # Queue metadata uses an opaque label, never the private tool name.
            return self.jobs.submit(owner, f"Locked tool {id[:8]}", args, Path.cwd(),
                                    vault=id, files=item["files"], entrypoint=tool["entrypoint"])

    def lock(self, owner: str, id: str) -> None:
        with self.mutex:
            item = self.opened.get((owner, id))
            if item is None:
                return
            item["closing"] = True
        if not self.jobs.forget_vault(owner, id):
            raise VaultError("A private process or temporary file could not be cleaned up; close it and retry Lock")
        with self.mutex:
            self.opened.pop((owner, id), None)

    def lock_all(self, owner: str) -> None:
        with self.mutex:
            ids = [id for who, id in self.opened if who == owner]
        for id in ids:
            self.lock(owner, id)

    def _expire(self) -> None:
        while not self.stopped.wait(2):
            with self.mutex:
                expired = [key for key, item in self.opened.items() if item["expires"] <= time.monotonic()]
            for owner, id in expired:
                try:
                    self.lock(owner, id)
                except VaultError:
                    pass  # Remain inaccessible and retry cleanup on the next sweep.

    def shutdown(self) -> None:
        self.stopped.set()
        self.worker.join(timeout=3)
        with self.mutex:
            keys = list(self.opened)
        for owner, id in keys:
            self.lock(owner, id)
