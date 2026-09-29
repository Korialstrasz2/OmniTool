"""Subprocess boundary for expensive file work. Input/output is bounded JSON.

No shell, no user-supplied executable, and no arbitrary imports from the request.
The caller enforces a wall-clock timeout. Limits are not a native-code sandbox.
"""
from __future__ import annotations
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

from .file_workbench import FileToolError, compare, read_regular, stamp
from .rename import directory


def thumbnail(path: Path, expected: list[int]) -> bytes:
    path = Path(os.path.abspath(path))
    directory(path.parent)
    if stamp(path.lstat()) != expected:
        raise FileToolError('Thumbnail source changed; reload the folders')
    if path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm'}:
        executable = shutil.which('ffmpeg')
        if not executable:
            raise FileToolError('Video thumbnail needs ffmpeg; filename selection remains available')
        # Local files only. Disable network protocols inside media containers/playlists.
        command = [executable, '-nostdin', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-i', str(path),
                   '-frames:v', '1', '-vf', 'scale=192:192:force_original_aspect_ratio=decrease',
                   '-f', 'image2pipe', '-c:v', 'png', 'pipe:1']
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=6, check=True)
        data = process.stdout
    else:
        from PIL import Image, ImageOps
        data = read_regular(path, 16 * 1024 * 1024)
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=('PNG', 'JPEG', 'GIF', 'WEBP', 'BMP')) as image:
                if image.width * image.height > 20_000_000:
                    raise FileToolError('Image too large for a thumbnail')
                image = ImageOps.exif_transpose(image)
                image.thumbnail((192, 192))
                image = image.convert('RGB'); image.info.clear()
                buf = io.BytesIO(); image.save(buf, 'PNG'); data = buf.getvalue()
    if stamp(path.lstat()) != expected or len(data) > 256 * 1024 or not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise FileToolError('Thumbnail unavailable or source changed')
    return data


def dispatch(request: dict) -> dict:
    action = request.get('action')
    if action == 'compare':
        return compare(Path(request['left']), Path(request['right']), request.get('mode', 'content'),
                       request.get('recursive', True), request.get('hash_budget_mib', 1024))
    if action in {'inspect', 'convert'}:
        from .conversion import inspect_input, convert
        args = (Path(request['input']), request.get('dpi', 220), request.get('max_pages', 100))
        if action == 'inspect':
            return inspect_input(*args)
        return convert(args[0], Path(request['out']), args[1], args[2], request['expected_sha256'])
    if action == 'thumbnail':
        return {'png': base64.b64encode(thumbnail(Path(request['path']), request['stamp'])).decode()}
    raise FileToolError('Unknown file operation')


def main():
    if sys.platform.startswith('linux'):
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
        resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    try:
        raw = sys.stdin.buffer.read(65537)
        if len(raw) > 65536:
            raise FileToolError('Request too large')
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise FileToolError('Expected an object')
        response = {'ok': True, 'result': dispatch(request)}
    except Exception as exc:
        response = {'ok': False, 'error': str(exc)[:1000] or type(exc).__name__}
    # ensure_ascii keeps OS paths with unusual characters out of encoding failure paths.
    sys.stdout.write(json.dumps(response, ensure_ascii=True))


if __name__ == '__main__':
    main()
