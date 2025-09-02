#!/usr/bin/env python3
"""
b64pdf2png.py
Decode a Base64 PDF (or image) and export PNG(s).
- If input is a .b64/.txt containing Base64, it decodes it.
- If input is a .pdf/.png/.jpg/.jpeg/.webp file, it reads bytes directly.
- If decoded bytes start with %PDF, renders each page to PNG using PyMuPDF.
- If bytes are an image, converts/saves to PNG.
Usage:
  python b64pdf2png.py INPUT [--out OUTDIR] [--dpi DPI] [--max-pages N]
Examples:
  python b64pdf2png.py input.b64 --out out_pngs --dpi 220
  python b64pdf2png.py input.pdf --out out_pngs
  python b64pdf2png.py image_base64.txt --out out_img
"""
import argparse
import base64
import binascii
import os
from pathlib import Path
from io import BytesIO

# Third-party
from PIL import Image
import fitz  # PyMuPDF

PDF_MAGIC = b"%PDF"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPG_MAGIC = b"\xff\xd8"
RIFF_MAGIC = b"RIFF"  # could be WEBP if followed by 'WEBP' at offset 8

def read_bytes_from_input(path: Path) -> bytes:
    # If it's a regular file with known binary extension, read raw
    if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
        return path.read_bytes()
    # Otherwise treat as text and try to extract Base64
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    # Strip data URL prefix if present
    if ";base64," in text:
        text = text.split(";base64,", 1)[1]
    # Remove whitespace and newlines that may be present in long Base64
    compact = "".join(text.split())
    try:
        return base64.b64decode(compact, validate=False)
    except binascii.Error as e:
        raise SystemExit(f"[error] Failed to decode Base64 from {path}: {e}")

def ensure_outdir(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    return out

def is_pdf(data: bytes) -> bool:
    return data.startswith(PDF_MAGIC)

def is_image_bytes(data: bytes) -> bool:
    if data.startswith(PNG_MAGIC) or data.startswith(JPG_MAGIC):
        return True
    if data.startswith(RIFF_MAGIC) and len(data) >= 12 and data[8:12] == b"WEBP":
        return True
    # Fallback: try PIL sniff
    try:
        Image.open(BytesIO(data))
        return True
    except Exception:
        return False

def save_image_as_png(data: bytes, outdir: Path, stem: str = "image"):
    outdir = ensure_outdir(outdir)
    # Use PIL to normalize to PNG
    with Image.open(BytesIO(data)) as im:
        im.load()
        out_path = outdir / f"{stem}.png"
        # Convert modes that are not directly PNG-friendly
        if im.mode in ("P", "LA"):
            im = im.convert("RGBA")
        elif im.mode == "CMYK":
            im = im.convert("RGB")
        im.save(out_path, format="PNG", optimize=True)
    print(str(out_path))

def render_pdf_to_pngs(data: bytes, outdir: Path, dpi: int, max_pages: int | None):
    outdir = ensure_outdir(outdir)
    scale = dpi / 72.0  # 72 dpi is PDF point baseline
    mat = fitz.Matrix(scale, scale)
    with fitz.open(stream=data, filetype="pdf") as doc:
        n = min(len(doc), max_pages) if max_pages else len(doc)
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out_path = outdir / f"page-{i+1:03d}.png"
            pix.save(out_path)
            print(str(out_path))

def main():
    ap = argparse.ArgumentParser(description="Decode Base64 PDF/image and export PNG(s).")
    ap.add_argument("input", type=str, help="Path to .b64/.txt Base64 or a .pdf/.png/.jpg file")
    ap.add_argument("--out", type=str, default="out_png", help="Output directory")
    ap.add_argument("--dpi", type=int, default=220, help="Render DPI for PDF pages")
    ap.add_argument("--max-pages", type=int, default=None, help="Limit number of pages rendered")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"[error] Input not found: {in_path}")

    data = read_bytes_from_input(in_path)

    if is_pdf(data):
        render_pdf_to_pngs(data, Path(args.out), dpi=args.dpi, max_pages=args.max_pages)
    elif is_image_bytes(data):
        save_image_as_png(data, Path(args.out), stem=in_path.stem or "image")
    else:
        # Last resort: try to open as PDF via PyMuPDF; if it fails, abort
        try:
            render_pdf_to_pngs(data, Path(args.out), dpi=args.dpi, max_pages=args.max_pages)
        except Exception as e:
            raise SystemExit(f"[error] Unknown data type. Not PDF or image. Details: {e}")

if __name__ == "__main__":
    main()
