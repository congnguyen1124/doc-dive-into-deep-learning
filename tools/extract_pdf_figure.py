#!/usr/bin/env python3
"""Render and crop a figure from didl.pdf while recording its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT.parent / "didl.pdf"


def parse_crop(value: str) -> tuple[float, float, float, float]:
    """Parse x,y,width,height percentages and ensure they stay on the page."""
    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop phải gồm bốn số") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop phải có dạng x,y,width,height")
    x, y, width, height = parts
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("crop phải có vị trí >= 0 và kích thước > 0")
    if x + width > 100 or y + height > 100:
        raise argparse.ArgumentTypeError("crop không được vượt khỏi 100% trang")
    return x, y, width, height


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_box(
    image_size: tuple[int, int], crop: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x, y, width, height = crop
    left = round(image_width * x / 100)
    top = round(image_height * y / 100)
    right = round(image_width * (x + width) / 100)
    bottom = round(image_height * (y + height) / 100)
    return left, top, right, bottom


def extract(args: argparse.Namespace) -> tuple[Path, Path]:
    pdf_path = args.pdf.resolve()
    output_path = args.output.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy PDF: {pdf_path}")
    if output_path.suffix.lower() != ".png":
        raise ValueError("Output phải là file .png")
    renderer = shutil.which("pdftoppm")
    if not renderer:
        raise RuntimeError("Không tìm thấy pdftoppm (Poppler)")

    with tempfile.TemporaryDirectory(prefix="d2l-figure-") as temp_dir:
        prefix = Path(temp_dir) / "page"
        subprocess.run(
            [
                renderer,
                "-f",
                str(args.pdf_page),
                "-l",
                str(args.pdf_page),
                "-r",
                str(args.dpi),
                "-singlefile",
                "-png",
                str(pdf_path),
                str(prefix),
            ],
            check=True,
        )
        rendered_path = prefix.with_suffix(".png")
        with Image.open(rendered_path) as rendered:
            rgb = rendered.convert("RGB")
            rendered_size = rgb.size
            cropped = rgb.crop(crop_box(rgb.size, args.crop))
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
        "crop_percent": {
            "x": args.crop[0],
            "y": args.crop[1],
            "width": args.crop[2],
            "height": args.crop[3],
        },
        "rendered_page_pixels": list(rendered_size),
        "output_pixels": list(output_size),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar_path = output_path.with_suffix(".source.json")
    sidecar_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path, sidecar_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF nguồn")
    parser.add_argument(
        "--pdf-page", type=int, required=True, help="Trang vật lý, bắt đầu từ 1"
    )
    parser.add_argument("--book-page", type=int, help="Số trang in trong sách")
    parser.add_argument(
        "--crop",
        type=parse_crop,
        default=parse_crop("0,0,100,100"),
        help="Vùng x,y,width,height theo phần trăm; mặc định là toàn trang",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Độ phân giải render")
    parser.add_argument("--figure-id", default="", help="Ví dụ: Figure 7.1")
    parser.add_argument("--caption", default="", help="Caption gốc hoặc mô tả")
    parser.add_argument("--output", type=Path, required=True, help="File PNG đầu ra")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.pdf_page < 1:
        parser.error("--pdf-page phải >= 1")
    if args.dpi < 72 or args.dpi > 600:
        parser.error("--dpi phải nằm trong khoảng 72..600")
    try:
        output, sidecar = extract(args)
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Lỗi: {exc}\n")
    print(f"Đã tạo: {output}")
    print(f"Nguồn: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
