from __future__ import annotations

import argparse
import hashlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def safe_venv_name(candidate: Path) -> str:
    digest = hashlib.sha1(str(candidate).lower().encode("utf-8")).hexdigest()[:10]
    stem = candidate.parent.name.replace(" ", "_")
    return f"{stem}_{digest}"


def wait_for_port(host: str, port: int, timeout: int = 20) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def install_and_start(candidate: Path, script_dir: Path, port: int) -> int:
    requirements = script_dir / "requirements.txt"
    app_file = script_dir / "app.py"

    if not app_file.exists():
        print(f"app.py not found in {script_dir}")
        return 1

    venv_root = script_dir / ".venv_candidates"
    venv_root.mkdir(parents=True, exist_ok=True)
    venv_dir = venv_root / safe_venv_name(candidate)
    py_exe = venv_dir / "Scripts" / "python.exe"

    if not py_exe.exists():
        print(f"Creating venv with {candidate} ...")
        result = subprocess.run([str(candidate), "-m", "venv", str(venv_dir)], cwd=script_dir)
        if result.returncode != 0 or not py_exe.exists():
            print("Failed to create venv with this interpreter.")
            return 1

    print(f"Using venv interpreter: {py_exe}")
    pip_upgrade = subprocess.run([str(py_exe), "-m", "pip", "install", "--upgrade", "pip"], cwd=script_dir)
    if pip_upgrade.returncode != 0:
        print("Failed to upgrade pip.")
        return 1

    if requirements.exists():
        install_req = subprocess.run([str(py_exe), "-m", "pip", "install", "-r", str(requirements)], cwd=script_dir)
        if install_req.returncode != 0:
            print("Dependency installation failed.")
            return 1

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE

    process = subprocess.Popen([str(py_exe), str(app_file)], cwd=script_dir, creationflags=creationflags)
    if wait_for_port("127.0.0.1", port):
        return 0

    print("Server did not become healthy in time; stopping process.")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    return 1


def remember_candidate(candidate: Path, bat_file: Path) -> int:
    if not candidate.exists() or candidate.name.lower() != "python.exe":
        return 0

    marker = f"rem AUTO_CANDIDATE_PYTHON={candidate}"
    try:
        content = bat_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 1

    if marker.lower() in content.lower():
        return 0

    newline = "\r\n" if "\r\n" in content else "\n"
    suffix = "" if content.endswith(("\n", "\r")) else newline
    updated = f"{content}{suffix}{marker}{newline}"

    try:
        bat_file.write_text(updated, encoding="utf-8")
    except OSError:
        return 1

    print(f"Saved candidate to batch file: {candidate}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--try-start", action="store_true")
    parser.add_argument("--remember-candidate", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--script-dir")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("bat_file", nargs="?")
    args = parser.parse_args()

    if args.try_start:
        if not args.candidate or not args.script_dir:
            parser.error("--try-start needs --candidate and --script-dir")
        return install_and_start(Path(args.candidate), Path(args.script_dir), args.port)

    if args.remember_candidate:
        if not args.candidate or not args.bat_file:
            parser.error("--remember-candidate needs --candidate and bat_file")
        return remember_candidate(Path(args.candidate), Path(args.bat_file))

    parser.error("No action specified.")


if __name__ == "__main__":
    sys.exit(main())
