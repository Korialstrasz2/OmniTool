import importlib.util
import io
import json
import os
import stat
import sys
import time
import zipfile
from pathlib import Path

import pytest

from omnitool_core.catalog import Catalog, arguments, relative_path, validate
from omnitool_core.jobs import Jobs
from omnitool_core.locks import Locked, Vaults
from omnitool_core.vault import MANIFEST, VaultError, build, decrypt, encrypt, materialize, unpack

PASS = "test-only-random-phrase-never-use-this"
SPEC = {"schema_version": 1, "id": "private-test", "name": "Hidden Tool Name", "entrypoint": "main.py"}

@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    (path / "tool.json").write_text(json.dumps(SPEC))
    (path / "main.py").write_text("print('test fixture')\n")
    return path

@pytest.fixture(scope="module")
def encrypted(tmp_path_factory):
    root = tmp_path_factory.mktemp("cipher")
    (root / "tool.json").write_text(json.dumps(SPEC))
    (root / "main.py").write_text("print('private source marker')")
    payload = build(root, "Secret Folder Name", "tool")
    return payload, encrypt(payload, PASS)

def test_round_trip_and_opacity(encrypted):
    payload, blob = encrypted
    assert decrypt(blob, PASS) == payload
    meta, files = unpack(payload)
    assert meta["name"] == "Secret Folder Name"
    assert files["main.py"] == b"print('private source marker')"
    for secret in (b"Secret Folder Name", b"Hidden Tool Name", b"private source marker", b"main.py"):
        assert secret not in blob
    assert encrypt(payload, PASS) != blob

@pytest.mark.parametrize("change", ["password", "salt", "nonce", "body", "tag", "truncate", "magic"])
def test_authentication_fails(encrypted, change):
    _, blob = encrypted
    candidate = bytearray(blob)
    password = PASS
    if change == "password": password += "wrong"
    elif change == "truncate": candidate = candidate[:-1]
    else:
        index = {"salt": 11, "nonce": 28, "body": 70, "tag": -1, "magic": 0}[change]
        candidate[index] ^= 1
    with pytest.raises(VaultError): decrypt(bytes(candidate), password)

@pytest.mark.parametrize("path", ["../bad", "/absolute", "a/../b", "a\\b", "C:/bad", "a//b", "a/./b", "CON", "a/LPT1.txt", "trailing."])
def test_unsafe_paths(path):
    with pytest.raises(ValueError): relative_path(path)

def malicious_archive(name, *, kind=stat.S_IFREG, compression=zipfile.ZIP_STORED, duplicate=False):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(MANIFEST, json.dumps({"version": 1, "name": "x", "kind": "folder", "tools": []}))
        info = zipfile.ZipInfo(name)
        info.external_attr = (kind | 0o600) << 16
        info.compress_type = compression
        archive.writestr(info, b"data")
        if duplicate: archive.writestr(name.upper(), b"other")
    return stream.getvalue()

@pytest.mark.parametrize("name,options", [
    ("../escape", {}), ("link", {"kind": stat.S_IFLNK}),
    ("compressed", {"compression": zipfile.ZIP_DEFLATED}), ("same", {"duplicate": True})])
def test_archive_rejects_unsafe_members(name, options):
    with pytest.raises(VaultError): unpack(malicious_archive(name, **options))

def test_materialize_and_tool_discovery(source, tmp_path):
    meta, files = unpack(build(source, "Private", "tool"))
    out = tmp_path / "out"; out.mkdir()
    materialize(files, out)
    assert (out / "main.py").read_bytes() == files["main.py"]
    with pytest.raises(VaultError): materialize(files, out)
    (tmp_path / "tools.json").write_text("[]")
    tool_dir = tmp_path / "tools" / "nested"; tool_dir.mkdir(parents=True)
    (tool_dir / "tool.json").write_text(json.dumps(SPEC))
    (tool_dir / "main.py").write_text("pass")
    catalog = Catalog(tmp_path)
    assert catalog.load()[0]["id"] == "private-test"
    assert not catalog.errors

def test_folder_hides_nested_tools(source):
    child = source / "nested"; child.mkdir()
    (child / "tool.json").write_text(json.dumps({**SPEC, "id": "nested-tool"}))
    (child / "main.py").write_text("pass")
    with pytest.raises(VaultError): build(source, "x", "tool")
    meta, files = unpack(build(source, "x", "folder"))
    assert len(meta["tools"]) == 2
    assert meta["tools"][1]["entrypoint"] == "nested/main.py"

def test_symlinks_rejected(source, tmp_path):
    (source / "link").symlink_to(tmp_path)
    with pytest.raises(VaultError): build(source, "x", "tool")

def test_bad_manifest_does_not_break_catalog(tmp_path):
    (tmp_path / "tools.json").write_text(json.dumps([{"id": "bad"}, {**SPEC, "entrypoint": "../escape.py"}]))
    catalog = Catalog(tmp_path)
    assert catalog.load() == [] and len(catalog.errors) == 2

def test_typed_arguments_are_not_shell_strings():
    spec = validate({**SPEC, "parameters": [
        {"name": "input", "required": True},
        {"name": "workers", "flag": "--workers", "type": "integer", "min": 1, "max": 4},
        {"name": "preview", "flag": "--preview", "type": "boolean", "default": False}]})
    assert arguments(spec, {"input": "a; echo NOT_A_COMMAND", "workers": 2, "preview": True}) == ["a; echo NOT_A_COMMAND", "--workers", "2", "--preview"]
    for values in ({"input": "--other"}, {"input": "x", "workers": 5}, {"input": "x", "preview": "true"}, {"unknown": 1}):
        with pytest.raises(ValueError): arguments(spec, values)

def test_private_job_cleanup_and_owner_isolation(tmp_path):
    jobs = Jobs(workers=1)
    try:
        code = b"import os\nprint(os.getcwd())\nprint('secret-output')\n"
        id = jobs.submit("alice", "Opaque", [], tmp_path, vault="a" * 32, files={"main.py": code}, entrypoint="main.py")
        assert jobs.items[id].done.wait(10)
        report = jobs.list("alice")[0]
        assert report["status"] == "succeeded"
        assert not Path(report["output"].splitlines()[0]).exists()
        assert jobs.list("bob") == []
        with pytest.raises(KeyError): jobs.stop("bob", id)
        assert jobs.forget_vault("alice", "a" * 32)
        assert jobs.list("alice") == []
    finally: jobs.shutdown()

def test_cancel_private_job_cleans_directory(tmp_path):
    jobs = Jobs(workers=1)
    try:
        id = jobs.submit("a", "Opaque", [], tmp_path, vault="b" * 32,
                         files={"main.py": b"import time\ntime.sleep(120)\n"}, entrypoint="main.py")
        for _ in range(100):
            if jobs.items[id].status == "running": break
            time.sleep(.02)
        assert jobs.forget_vault("a", "b" * 32)
        assert jobs.list("a") == []
    finally: jobs.shutdown()

def test_vault_session_scope_and_expiry(tmp_path, encrypted):
    root = tmp_path / "vaults"; root.mkdir()
    id = "a" * 32
    (root / f"{id}.otvault").write_bytes(encrypted[1])
    jobs = Jobs(workers=1); vaults = Vaults(root, jobs, ttl=600)
    try:
        assert vaults.list("a") == [{"id": id, "locked": True}]
        with pytest.raises(Locked): vaults.get("a", id)
        vaults.unlock("a", id, PASS)
        assert vaults.list("a")[0]["name"] == "Secret Folder Name"
        assert vaults.list("b") == [{"id": id, "locked": True}]
        with pytest.raises(Locked): vaults.get("b", id)
        vaults.opened[("a", id)]["expires"] = time.monotonic() - 1
        with pytest.raises(Locked): vaults.get("a", id)
        vaults.lock("a", id)
        assert not vaults.opened
    finally: vaults.shutdown(); jobs.shutdown()

def test_duplicate_finder_read_only(tmp_path):
    spec = importlib.util.spec_from_file_location("duplicates", Path(__file__).parents[1] / "tools/duplicate-finder/main.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for name, text in {"a": "same", "b": "same", "c": "diff"}.items(): (tmp_path / name).write_text(text)
    (tmp_path / "link").symlink_to(tmp_path / "a")
    report = module.find_duplicates(tmp_path)
    assert report["read_only"] and len(report["duplicate_groups"]) == 1
    assert len(report["duplicate_groups"][0]["files"]) == 2
    assert (tmp_path / "c").read_text() == "diff"


def test_output_is_available_before_process_exit(tmp_path):
    jobs = Jobs(workers=1)
    try:
        id = jobs.submit("a", "test", [sys.executable, "-u", "-c", "import time; print('progress'); time.sleep(30)"], tmp_path)
        for _ in range(100):
            if "progress" in jobs.items[id].output: break
            time.sleep(.02)
        assert "progress" in jobs.items[id].output
        assert not jobs.items[id].done.is_set()
    finally: jobs.shutdown()
