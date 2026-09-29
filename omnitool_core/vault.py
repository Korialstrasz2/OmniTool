"""Versioned encrypted bundles. See docs/VAULT_SECURITY.md before using.

No custom cipher: PyCA AES-256-GCM + scrypt. This *container/integration*
is new and is not independently audited. It is not the age file format.
"""
from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import secrets
import stat
import struct
import tempfile
import zipfile
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .catalog import relative_path, validate

MAGIC = b"OMNIVAULT\x01"
HEADER_SIZE = len(MAGIC) + 16 + 12
MAX_FILES = 4096
MAX_BYTES = 32 * 1024 * 1024
MAX_ENVELOPE = 64 * 1024 * 1024
MIN_PADDING = 64 * 1024
MANIFEST = "__omnitool_bundle__.json"
SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}


class VaultError(ValueError):
    """Safe-to-display vault failure; never include decrypted content."""


def _key(passphrase: str, salt: bytes) -> bytes:
    if not isinstance(passphrase, str) or not 1 <= len(passphrase.encode("utf-8")) <= 1024:
        raise VaultError("Invalid passphrase length")
    # Fixed parameters cannot be raised by a malicious file header.
    return Scrypt(salt=salt, length=32, n=2**17, r=8, p=1).derive(passphrase.encode("utf-8"))


def encrypt(payload: bytes, passphrase: str) -> bytes:
    if len(passphrase) < 16:
        raise VaultError("Use at least 16 characters; a generated high-entropy passphrase is strongly preferred")
    if not 0 < len(payload) <= MAX_BYTES:
        raise VaultError("Bundle exceeds the 32 MiB limit")
    length = len(payload) + 8
    padded_size = max(MIN_PADDING, 1 << (length - 1).bit_length())
    padded = struct.pack(">Q", len(payload)) + payload + secrets.token_bytes(padded_size - length)
    salt, nonce = secrets.token_bytes(16), secrets.token_bytes(12)
    header = MAGIC + salt + nonce
    key = _key(passphrase, salt)
    try:
        return header + AESGCM(key).encrypt(nonce, padded, header)
    finally:
        del key, padded


def decrypt(blob: bytes, passphrase: str) -> bytes:
    if not HEADER_SIZE + 16 + MIN_PADDING <= len(blob) <= MAX_ENVELOPE + HEADER_SIZE + 16:
        raise VaultError("Invalid or oversized encrypted bundle")
    padded_length = len(blob) - HEADER_SIZE - 16
    if blob[:len(MAGIC)] != MAGIC or padded_length & (padded_length - 1):
        raise VaultError("Unsupported encrypted bundle")
    salt = blob[len(MAGIC):len(MAGIC) + 16]
    nonce = blob[len(MAGIC) + 16:HEADER_SIZE]
    key = _key(passphrase, salt)
    try:
        padded = AESGCM(key).decrypt(nonce, blob[HEADER_SIZE:], blob[:HEADER_SIZE])
    except InvalidTag:
        raise VaultError("Incorrect passphrase or damaged bundle") from None
    finally:
        del key
    length = struct.unpack(">Q", padded[:8])[0]
    if not 0 < length <= min(MAX_BYTES, len(padded) - 8):
        raise VaultError("Invalid encrypted payload")
    return padded[8:8 + length]


def _write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    archive.writestr(info, data)


def build(source: Path, name: str, kind: str) -> bytes:
    source = source.expanduser().resolve()
    if not source.is_dir() or kind not in {"tool", "folder"} or not 1 <= len(name) <= 120:
        raise VaultError("Choose a source directory, name, and tool/folder kind")
    files: dict[str, bytes] = {}
    tools = []
    total = 0
    for directory, dirs, names in os.walk(source, followlinks=False):
        for child in dirs:
            if (Path(directory) / child).is_symlink():
                raise VaultError("Symlinks are not supported")
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for filename in sorted(names):
            path = Path(directory) / filename
            if path.is_symlink() or not path.is_file():
                raise VaultError("Only regular files are supported")
            rel = path.relative_to(source).as_posix()
            relative_path(rel)
            if rel == MANIFEST:
                raise VaultError("Reserved bundle metadata filename")
            size = path.stat().st_size
            if size > MAX_BYTES or total + size > MAX_BYTES or len(files) >= MAX_FILES - 1:
                raise VaultError("Bundle too large or too many files")
            with path.open("rb") as stream:
                data = stream.read(MAX_BYTES + 1)
            total += len(data)
            if total > MAX_BYTES:
                raise VaultError("Bundle too large")
            files[rel] = data
            if filename == "tool.json":
                spec = validate(json.loads(data))
                if spec["kind"] != "python":
                    raise VaultError("Locked tools must be Python entrypoints")
                parent = path.parent.relative_to(source)
                spec["entrypoint"] = (parent / spec["entrypoint"]).as_posix()
                tools.append(spec)
    if kind == "tool" and len(tools) != 1:
        raise VaultError("A locked tool requires exactly one tool.json; use folder for a collection")
    if len({t["id"] for t in tools}) != len(tools):
        raise VaultError("Duplicate tool ID in bundle")
    for tool in tools:
        if tool["entrypoint"] not in files:
            raise VaultError("A tool entrypoint is missing")
    metadata = {"version": 1, "name": name, "kind": kind, "tools": tools}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        _write_member(archive, MANIFEST, json.dumps(metadata).encode("utf-8"))
        for filename, data in sorted(files.items()):
            _write_member(archive, filename, data)
    payload = buf.getvalue()
    # Validate the same parser used for untrusted decrypted content.
    unpack(payload)
    if len(payload) > MAX_BYTES:
        raise VaultError("Bundle including archive metadata exceeds 32 MiB")
    return payload


def unpack(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    if len(payload) > MAX_BYTES:
        raise VaultError("Bundle is too large")
    files, folded = {}, set()
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > MAX_FILES or sum(m.file_size for m in members) > MAX_BYTES:
                raise VaultError("Archive limits exceeded")
            for member in members:
                name = str(relative_path(member.filename))
                mode = member.external_attr >> 16
                if member.is_dir() or (stat.S_IFMT(mode) not in {0, stat.S_IFREG}) or member.compress_type != zipfile.ZIP_STORED:
                    raise VaultError("Only uncompressed regular-file entries are accepted")
                if name.casefold() in folded or member.flag_bits & 1:
                    raise VaultError("Duplicate or unsupported archive member")
                folded.add(name.casefold())
                files[name] = archive.read(member)
        # File/directory conflicts must be rejected *before* any materialization.
        for name in files:
            for parent in relative_path(name).parents:
                if str(parent) != "." and str(parent).casefold() in folded:
                    raise VaultError("File/directory conflict")
        meta = json.loads(files.pop(MANIFEST))
        if not isinstance(meta, dict) or meta.get("version") != 1 or meta.get("kind") not in {"tool", "folder"}:
            raise VaultError("Invalid bundle metadata")
        if not isinstance(meta.get("name"), str) or not 1 <= len(meta["name"]) <= 120:
            raise VaultError("Invalid bundle name")
        if not isinstance(meta.get("tools"), list) or len(meta["tools"]) > 1024:
            raise VaultError("Invalid tool collection")
        meta["tools"] = [validate(t) for t in meta["tools"]]
        if len({t["id"] for t in meta["tools"]}) != len(meta["tools"]):
            raise VaultError("Duplicate tool ID")
        if meta["kind"] == "tool" and len(meta["tools"]) != 1:
            raise VaultError("Invalid locked tool")
        if any(t["kind"] != "python" or t["entrypoint"] not in files for t in meta["tools"]):
            raise VaultError("Invalid locked entrypoint")
        return meta, files
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile, RuntimeError):
        raise VaultError("Invalid encrypted bundle contents") from None


def materialize(files: dict[str, bytes], destination: Path) -> None:
    """Extract only into a new empty directory, outside the repository."""
    destination = destination.resolve()
    if any(destination.iterdir()):
        raise VaultError("Extraction directory must be empty")
    for name, data in files.items():
        path = destination / relative_path(name)
        if not path.resolve().is_relative_to(destination):
            raise VaultError("Unsafe archive path")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal private source directories; only ciphertext belongs in Git")
    sub = parser.add_subparsers(dest="action", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("source", type=Path)
    seal.add_argument("--name", required=True)
    seal.add_argument("--kind", choices=["tool", "folder"], required=True)
    seal.add_argument("--output-dir", type=Path, default=Path("vaults"))
    restore = sub.add_parser("restore")
    restore.add_argument("bundle", type=Path)
    restore.add_argument("destination", type=Path, help="New directory OUTSIDE every Git working tree")
    args = parser.parse_args()
    try:
        if args.action == "seal":
            source = args.source.expanduser().resolve()
            if any((p / ".git").exists() for p in (source, *source.parents)):
                raise VaultError("Keep plaintext outside Git working trees before sealing")
            if args.output_dir.resolve().is_relative_to(source):
                raise VaultError("Ciphertext output must be outside the source directory")
            payload = build(source, args.name, args.kind)
            password = getpass.getpass("New passphrase (store it in a password manager): ")
            if password != getpass.getpass("Repeat passphrase: "):
                raise VaultError("Passphrases do not match")
            blob = encrypt(payload, password)
            # Verify round-trip before publishing the new artifact.
            unpack(decrypt(blob, password))
            del password, payload
            args.output_dir.mkdir(parents=True, exist_ok=True)
            target = args.output_dir / f"{secrets.token_hex(16)}.otvault"
            with target.open("xb") as output:
                output.write(blob)
            print(f"Created {target}. Plaintext originals are unchanged; nothing was uploaded.")
        else:
            destination = args.destination.expanduser().resolve()
            if destination.exists() or any((p / ".git").exists() for p in destination.parents):
                raise VaultError("Choose a new destination outside Git working trees")
            if args.bundle.stat().st_size > MAX_ENVELOPE + HEADER_SIZE + 16:
                raise VaultError("Bundle is too large")
            password = getpass.getpass("Passphrase: ")
            _, files = unpack(decrypt(args.bundle.read_bytes(), password))
            del password
            destination.mkdir(mode=0o700, parents=True)
            materialize(files, destination)
            print("Restored plaintext. Protect it with operating-system permissions and full-disk encryption.")
    except (VaultError, OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
