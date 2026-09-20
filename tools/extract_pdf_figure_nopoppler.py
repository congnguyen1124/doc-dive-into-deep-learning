#!/usr/bin/env python3
"""Render and crop a figure from didl.pdf, writing the repo's provenance sidecar.

Same CLI and sidecar format as ``extract_pdf_figure.py``. Use this variant on a
machine where Poppler (``pdftoppm``) is unavailable: it rasterizes by extracting
the requested page into a standalone, upscaled one-page PDF with pure-python
helpers (``pdf_page.py``), then converting that page with macOS ``sips``.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402
from pdf_reader import PDF  # noqa: E402
from pdf_page import Writer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT.parent / "didl.pdf"


def parse_crop(value: str) -> tuple[float, float, float, float]:
    parts = tuple(float(p.strip()) for p in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop phải có dạng x,y,width,height")
    x, y, w, h = parts
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 100 or y + h > 100:
        raise argparse.ArgumentTypeError("crop phải nằm trong 0..100 phần trăm trang")
    return x, y, w, h


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_page(doc: PDF, pdf_page: int, dpi: int, workdir: Path) -> Path:
    """Rasterize one page by scaling it into a standalone PDF, then converting with sips."""
    scale = dpi / 72.0
    page_pdf = workdir / "page.pdf"
    page_pdf.write_bytes(Writer(doc, scale).build(doc.pages()[pdf_page - 1]))
    page_png = workdir / "page.png"
    subprocess.run(
        ["sips", "-s", "format", "png", str(page_pdf), "--out", str(page_png)],
        check=True, capture_output=True,
    )
    return page_png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--pdf-page", type=int, required=True)
    parser.add_argument("--book-page", type=int)
    parser.add_argument("--crop", type=parse_crop, default=parse_crop("0,0,100,100"))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--figure-id", default="")
    parser.add_argument("--caption", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    output_path = args.output.resolve()
    if output_path.suffix.lower() != ".png":
        parser.error("Output phải là file .png")

    doc = PDF(str(pdf_path))
    with tempfile.TemporaryDirectory(prefix="d2l-figure-") as tmp:
        page_png = render_page(doc, args.pdf_page, args.dpi, Path(tmp))
        with Image.open(page_png) as rendered:
            # sips emits RGBA with a transparent page ground; flatten onto white
            # so the crop does not come out on a black background.
            rgba = rendered.convert("RGBA")
            canvas = Image.new("RGB", rgba.size, (255, 255, 255))
            canvas.paste(rgba, mask=rgba.split()[3])
            rgb = canvas
            rendered_size = rgb.size
            iw, ih = rendered_size
            x, y, w, h = args.crop
            box = (round(iw * x / 100), round(ih * y / 100),
                   round(iw * (x + w) / 100), round(ih * (y + h) / 100))
            cropped = rgb.crop(box)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(output_path, "PNG", optimize=True)
            output_size = cropped.size

    provenance = {
        "schema_version": 1,
        "kind": "figure-extracted-from-pdf",
        "source_pdf": str(pdf_path),
        "source_pdf_sha256": sha256(pdf_path),
        "pdf_page": args.pdf_page,
        "book_page": args.book_page,
        "figure_id": args.figure_id,
        "caption": args.caption,
        "dpi": args.dpi,
        "crop_percent": {"x": args.crop[0], "y": args.crop[1],
                         "width": args.crop[2], "height": args.crop[3]},
        "rendered_page_pixels": list(rendered_size),
        "output_pixels": list(output_size),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "renderer": "pure-python page extraction + macOS sips (Poppler unavailable)",
    }
    output_path.with_suffix(".source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã tạo: {output_path}  ({output_size[0]}x{output_size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
