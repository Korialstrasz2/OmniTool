"""Read-only startup diagnostics. This module never installs or removes packages."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CORE_MODULES = ('flask', 'cryptography', 'waitress')


def version_tuple(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r'\d+(?:\.\d+)*', value):
        raise ValueError('Only stable release versions are supported at startup')
    return tuple(int(part) for part in value.split('.'))


def requirement_problems(path: Path) -> list[str]:
    """Read this project's deliberately simple stable min/max requirement files."""
    problems = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'([A-Za-z0-9_-]+)>=(\d+(?:\.\d+)*),<(\d+(?:\.\d+)*)', line)
        if not match:
            problems.append(f'Cannot validate requirement: {line}; review {path.name}')
            continue
        name, minimum, maximum = match.groups()
        try:
            installed = importlib.metadata.version(name)
            parsed = version_tuple(installed)
            if not version_tuple(minimum) <= parsed < version_tuple(maximum):
                problems.append(f'{name}: installed {installed}; requires >={minimum},<{maximum}')
        except importlib.metadata.PackageNotFoundError:
            problems.append(f'{name}: not installed; requires >={minimum},<{maximum}')
        except ValueError:
            problems.append(f'{name}: unrecognized/pre-release version; requires >={minimum},<{maximum}')
    return problems


def interpreter_problems() -> list[str]:
    problems = []
    if sys.version_info < (3, 11):
        problems.append('Python 3.11 or newer is required; Python 3.13 x64 is the preferred Windows version.')
    if struct.calcsize('P') != 8:
        problems.append('Use 64-bit Python for this workspace and its conversion/cryptography dependencies.')
    return problems


def core_problems() -> list[str]:
    problems = interpreter_problems() + requirement_problems(BASE / 'requirements-core.txt')
    if not problems:
        try:
            result = subprocess.run([sys.executable, '-c', 'import flask, cryptography, waitress'],
                                    capture_output=True, timeout=20, check=False)
            if result.returncode:
                problems.append('Core package import failed. Repair this virtual environment with requirements-core.txt.')
        except (OSError, subprocess.TimeoutExpired):
            problems.append('The virtual environment could not complete its import check.')
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-core', action='store_true')
    parser.add_argument('--diagnose', action='store_true')
    args = parser.parse_args(argv)
    problems = core_problems()
    if args.diagnose:
        report = {'python': sys.version.split()[0], 'bits': struct.calcsize('P') * 8,
                  'core': problems or ['Ready'],
                  'lyrics_and_prompts': requirement_problems(BASE / 'requirements-content.txt') or ['Ready'],
                  'conversion': requirement_problems(BASE / 'requirements-conversion.txt') or ['Ready'],
                  'ffmpeg': 'Available on PATH' if shutil.which('ffmpeg') else 'Not installed (optional video previews)',
                  'note': 'Checks are local. No model/provider was contacted; no files or packages were changed.'}
        print(json.dumps(report, ensure_ascii=True, indent=2))
    elif problems:
        for problem in problems:
            print(problem)
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
