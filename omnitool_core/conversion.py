"""Strict, resource-bounded Base64/PDF/image to PNG conversion.

All outputs are built in a new sibling staging folder and published with a
no-overwrite directory rename. Input and existing output folders are untouched.
Use the workbench subprocess for time limits; direct library calls are not sandboxed.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path

from .file_workbench import FileToolError, component, integer, read_regular
from .rename import directory, move_noreplace

MAX_INPUT = 32 * 1024 * 1024
MAX_DECODED = 24 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_TOTAL_PIXELS = 100_000_000
MAX_OUTPUT = 512 * 1024 * 1024
BINARY = {'.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
FORMATS = ('PNG', 'JPEG', 'WEBP', 'BMP', 'GIF', 'TIFF')
DATA_URI = re.compile(r'data:(application/pdf|image/(?:png|jpeg|webp|bmp|gif|tiff));base64,', re.I)


def decode_input(path: Path) -> tuple[bytes, str]:
    raw = read_regular(path, MAX_INPUT)
    fingerprint = hashlib.sha256(raw).hexdigest()
    if path.suffix.lower() in BINARY:
        data = raw
    else:
        try:
            text = raw.decode('utf-8-sig').strip()
            if text.lower().startswith('data:'):
                match = DATA_URI.match(text)
                if not match:
                    raise FileToolError('Unsupported data URI; use a supported MIME type with ;base64,')
                text = text[match.end():]
            compact = re.sub(r'[ \t\r\n]', '', text)
            if not compact:
                raise FileToolError('Base64 input is empty')
            data = base64.b64decode(compact, validate=True)
            # Reject noncanonical padding/trailing bits rather than silently accepting corruption.
            if base64.b64encode(data).decode('ascii') != compact:
                raise FileToolError('Noncanonical Base64 encoding')
        except (UnicodeError, binascii.Error) as exc:
            raise FileToolError('Invalid Base64: only strict, padded standard Base64 is accepted') from exc
    if not data or len(data) > MAX_DECODED:
        raise FileToolError('Decoded input must be nonempty and at most 24 MiB')
    return data, fingerprint


def _dimensions(width: int, height: int) -> dict:
    if width < 1 or height < 1 or width * height > MAX_PIXELS:
        raise FileToolError('An output image would exceed 20 million pixels; lower the DPI or input size')
    return {'width': width, 'height': height}


def inspect_input(path: Path, dpi: int = 220, max_pages: int = 100) -> dict:
    integer(dpi, 36, 600, 'DPI'); integer(max_pages, 1, 100, 'Maximum pages')
    path = Path(os.path.abspath(path.expanduser()))
    data, fingerprint = decode_input(path)
    images = []
    if data.startswith(b'%PDF-'):
        import fitz
        with fitz.open(stream=data, filetype='pdf') as doc:
            if doc.needs_pass:
                raise FileToolError('Password-protected PDFs are not supported; unlock a local copy first')
            if doc.is_repaired:
                raise FileToolError('PDF required repair; validate a repaired copy before converting')
            total = len(doc)
            if not total:
                raise FileToolError('PDF contains no pages')
            for i in range(min(total, max_pages)):
                rect = (doc[i].rect * fitz.Matrix(dpi / 72, dpi / 72)).irect
                dim = _dimensions(rect.width, rect.height)
                images.append(dict(dim, filename=f'page-{i+1:03d}.png'))
            kind = 'pdf'
    else:
        from PIL import Image, ImageOps
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=FORMATS) as image:
                if getattr(image, 'n_frames', 1) != 1:
                    raise FileToolError('Multi-frame/animated images are not supported; export a single frame first')
                _dimensions(*image.size)
                image.load()
                oriented = ImageOps.exif_transpose(image)
                images.append(dict(_dimensions(*oriented.size), filename='image.png'))
            kind, total = 'image', 1
    if sum(item['width'] * item['height'] for item in images) > MAX_TOTAL_PIXELS:
        raise FileToolError('This batch exceeds 100 million rendered pixels; reduce pages or DPI')
    return {'input': str(path), 'sha256': fingerprint, 'kind': kind, 'decoded_bytes': len(data),
            'dpi': dpi, 'max_pages': max_pages, 'total_pages': total, 'selected_pages': len(images),
            'omitted_pages': total - len(images), 'images': images}


def output_path(path: Path) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    component(path.name)
    directory(path.parent)
    if os.path.lexists(path):
        raise FileToolError('Output must be a NEW directory; an existing file or folder will not be overwritten')
    return path


def convert(path: Path, out: Path, dpi: int = 220, max_pages: int = 100,
            expected_sha256: str | None = None) -> dict:
    # Decode exactly the same input bytes used by the renderer, not a later file open.
    info = inspect_input(path, dpi, max_pages)
    if expected_sha256 is not None and info['sha256'] != expected_sha256:
        raise FileToolError('Input changed since preview; inspect it again')
    data, digest = decode_input(Path(info['input']))
    if digest != info['sha256']:
        raise FileToolError('Input changed while preparing conversion')
    out = output_path(out)
    stage = Path(tempfile.mkdtemp(prefix='.omnitool-convert-', suffix='.partial', dir=out.parent))
    published = False
    try:
        if info['kind'] == 'pdf':
            import fitz
            with fitz.open(stream=data, filetype='pdf') as doc:
                for i, item in enumerate(info['images']):
                    pix = doc[i].get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
                    _dimensions(pix.width, pix.height)
                    pix.save(stage / item['filename'])
                    del pix
        else:
            from PIL import Image, ImageOps
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data), formats=FORMATS) as image:
                    _dimensions(*image.size)
                    normalized = ImageOps.exif_transpose(image)
                    mode = 'RGBA' if 'A' in normalized.getbands() or 'transparency' in normalized.info else 'RGB'
                    normalized = normalized.convert(mode)
                    normalized.info.clear()  # Drop source metadata; pixel orientation has already been applied.
                    normalized.save(stage / 'image.png', format='PNG')
        if sum(p.stat().st_size for p in stage.iterdir()) > MAX_OUTPUT:
            raise FileToolError('Rendered output exceeds the 512 MiB batch limit')
        report = dict(info, output=str(out), completed=True)
        (stage / 'conversion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        for file in stage.iterdir():
            # Windows flushing requires a writable handle. Only our staged outputs
            # are reopened; r+b neither truncates nor touches the source file.
            with file.open('r+b') as stream:
                os.fsync(stream.fileno())
        directory(out.parent)
        move_noreplace(stage, out)
        published = True
        return report
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Inspect or convert PDF/image/Base64 into a NEW output directory.')
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, default=Path('out_png'))
    parser.add_argument('--dpi', type=int, default=220)
    parser.add_argument('--max-pages', type=int, default=100)
    parser.add_argument('--inspect', action='store_true', help='Read-only metadata preview; create no output')
    parser.add_argument('--expected-sha256', help='Refuse if source bytes changed since inspection')
    args = parser.parse_args()
    try:
        info = inspect_input(args.input, args.dpi, args.max_pages) if args.inspect else convert(
            args.input, args.out, args.dpi, args.max_pages, args.expected_sha256)
        print(json.dumps(info, indent=2))
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(1, f'Conversion failed: {exc}\n')
