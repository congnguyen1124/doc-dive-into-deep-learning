#!/usr/bin/env python3
"""Local notebook reader for the Vietnamese Dive into Deep Learning notes."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse
from xml.etree import ElementTree

try:
    import markdown
    import yaml
    from markdown.extensions import Extension
    from markdown.inlinepatterns import InlineProcessor
except ImportError as exc:  # pragma: no cover - exercised only without dependencies
    print(
        "Thiếu thư viện. Hãy chạy: python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
CONTENT_ROOT = PROJECT_ROOT / "content"
STATIC_ROOT = PROJECT_ROOT / "static"
INDEX_PATH = CONTENT_ROOT / "book-index.json"
GLOSSARY_PATH = CONTENT_ROOT / "glossary.json"
TERM_PATTERN = re.compile(r"\{\{term:([a-z0-9-]+)(?:\|([^}]+))?\}\}")
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^\s)]+)(?:\s+['\"][^'\"]*['\"])?\)")
IMAGE_SRC_PATTERN = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+)(")', re.IGNORECASE)
PAGEBREAK_PATTERN = re.compile(r"<!--\s*pagebreak\s*-->", re.IGNORECASE)


class NotebookError(ValueError):
    """Raised when notebook content violates the reader contract."""


@dataclass(frozen=True)
class Document:
    id: str
    path: Path
    metadata: dict[str, Any]
    body: str

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.metadata["type"],
            "number": str(self.metadata["number"]),
            "order": int(self.metadata["order"]),
            "title": self.metadata["title"],
            "book_pages": self.metadata["book_pages"],
            "pdf_pages": self.metadata["pdf_pages"],
            "status": self.metadata["status"],
        }


class GlossaryInlineProcessor(InlineProcessor):
    """Convert {{term:key|Label}} into an accessible glossary target."""

    def handleMatch(self, match: re.Match[str], data: str):  # noqa: N802
        key = match.group(1)
        label = match.group(2) or key
        element = ElementTree.Element("span")
        element.set("class", "glossary-term")
        element.set("data-term", key)
        element.set("tabindex", "0")
        element.set("role", "button")
        element.set("aria-label", f"Open term note for {label}")
        element.text = label
        return element, match.start(0), match.end(0)


class GlossaryExtension(Extension):
    def extendMarkdown(self, md: markdown.Markdown) -> None:  # noqa: N802
        md.inlinePatterns.register(
            GlossaryInlineProcessor(TERM_PATTERN.pattern, md), "glossary_term", 175
        )


def parse_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise NotebookError(f"{path.name}: thiếu YAML front matter")
    try:
        _, raw_metadata, body = text.split("---\n", 2)
    except ValueError as exc:
        raise NotebookError(f"{path.name}: YAML front matter chưa đóng") from exc
    metadata = yaml.safe_load(raw_metadata)
    if not isinstance(metadata, dict):
        raise NotebookError(f"{path.name}: YAML front matter không hợp lệ")
    return metadata, body.strip()


def document_id(metadata: dict[str, Any]) -> str:
    doc_type = str(metadata.get("type", ""))
    number = str(metadata.get("number", "")).lower()
    if doc_type == "chapter" and number.isdigit():
        return f"chapter-{int(number):02d}"
    if doc_type == "appendix" and re.fullmatch(r"[a-z]", number):
        return f"appendix-{number}"
    raise NotebookError(f"Không thể tạo id từ type={doc_type!r}, number={number!r}")


def discover_documents(content_root: Path = CONTENT_ROOT) -> list[Document]:
    paths = sorted((content_root / "chapters").glob("*.md"))
    paths += sorted((content_root / "appendices").glob("*.md"))
    documents: list[Document] = []
    required = {
        "type",
        "number",
        "order",
        "title",
        "book_pages",
        "pdf_pages",
        "status",
    }
    for path in paths:
        metadata, body = parse_front_matter(path)
        missing = required - metadata.keys()
        if missing:
            raise NotebookError(f"{path.name}: thiếu metadata {sorted(missing)}")
        documents.append(Document(document_id(metadata), path, metadata, body))
    return sorted(documents, key=lambda doc: int(doc.metadata["order"]))


def load_glossary(path: Path = GLOSSARY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("terms"), dict):
        raise NotebookError("content/glossary.json phải chứa object 'terms'")
    return payload


def load_book_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise NotebookError("content/book-index.json phải chứa array 'documents'")
    return payload


def _rewrite_relative_images(rendered: str, document: Document) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_src = html.unescape(match.group(2))
        parsed = urlparse(raw_src)
        if parsed.scheme or raw_src.startswith(("/", "#")):
            return match.group(0)
        resolved = (document.path.parent / unquote(parsed.path)).resolve()
        try:
            relative = resolved.relative_to(CONTENT_ROOT.resolve())
        except ValueError:
            return match.group(0)
        src = "/content/" + quote(relative.as_posix())
        return f"{match.group(1)}{src}{match.group(3)}"

    return IMAGE_SRC_PATTERN.sub(replace, rendered)


def render_document(document: Document) -> str:
    renderer = markdown.Markdown(
        extensions=[
            GlossaryExtension(),
            "attr_list",
            "fenced_code",
            "codehilite",
            "md_in_html",
            "sane_lists",
            "tables",
            "toc",
        ],
        extension_configs={
            "toc": {"permalink": False, "toc_depth": "2-4"},
            "codehilite": {
                "css_class": "codehilite",
                "guess_lang": False,
                "linenums": False,
                "noclasses": False,
                "use_pygments": True,
            },
        },
        output_format="html5",
    )
    return _rewrite_relative_images(renderer.convert(document.body), document)


def chapter_payload(document: Document) -> dict[str, Any]:
    payload = document.summary()
    rendered = render_document(document)
    pages = [page.strip() for page in PAGEBREAK_PATTERN.split(rendered) if page.strip()]
    payload["html"] = rendered
    payload["pages"] = pages or [rendered]
    return payload


def validate_notebook(
    content_root: Path = CONTENT_ROOT,
    index_path: Path = INDEX_PATH,
    glossary_path: Path = GLOSSARY_PATH,
) -> list[str]:
    errors: list[str] = []
    try:
        documents = discover_documents(content_root)
        index = load_book_index(index_path)
        glossary = load_glossary(glossary_path)
    except (NotebookError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return [str(exc)]

    by_id = {document.id: document for document in documents}
    if len(by_id) != len(documents):
        errors.append("Có id chương/phụ lục bị trùng")
    orders = [int(document.metadata["order"]) for document in documents]
    if len(set(orders)) != len(orders):
        errors.append("Có giá trị order bị trùng")

    expected_ids: set[str] = set()
    for expected in index["documents"]:
        expected_id = expected["id"]
        expected_ids.add(expected_id)
        document = by_id.get(expected_id)
        if document is None:
            errors.append(f"Thiếu file cho {expected_id}: {expected['filename']}")
            continue
        comparisons = {
            "type": expected["type"],
            "number": expected["number"],
            "order": expected["order"],
            "title": expected["title"],
            "book_pages": expected["book_pages"],
            "pdf_pages": expected["pdf_pages"],
        }
        for key, value in comparisons.items():
            if str(document.metadata[key]) != str(value):
                errors.append(
                    f"{document.path.name}: {key}={document.metadata[key]!r}, cần {value!r}"
                )
        if document.path.name != expected["filename"]:
            errors.append(
                f"Sai tên file {document.path.name!r}; cần {expected['filename']!r}"
            )
        first_heading = re.search(r"^#\s+(.+)$", document.body, re.MULTILINE)
        if not first_heading or first_heading.group(1).strip() != expected["title"]:
            errors.append(f"{document.path.name}: H1 phải là {expected['title']!r}")

    unexpected = sorted(set(by_id) - expected_ids)
    if unexpected:
        errors.append(f"Có document ngoài book-index: {', '.join(unexpected)}")

    glossary_keys = set(glossary["terms"])
    for document in documents:
        used_terms = {match.group(1) for match in TERM_PATTERN.finditer(document.body)}
        for missing_key in sorted(used_terms - glossary_keys):
            errors.append(f"{document.path.name}: thiếu glossary key {missing_key!r}")
        for image_match in IMAGE_PATTERN.finditer(document.body):
            image_src = unquote(image_match.group(1))
            if urlparse(image_src).scheme or image_src.startswith(("/", "#")):
                continue
            image_path = (document.path.parent / image_src).resolve()
            try:
                image_path.relative_to(content_root.resolve())
            except ValueError:
                errors.append(f"{document.path.name}: ảnh nằm ngoài content: {image_src}")
                continue
            if not image_path.is_file():
                errors.append(f"{document.path.name}: không tìm thấy ảnh {image_src}")
            is_book_asset = "generated" not in image_path.parts
            if document.metadata["status"] != "placeholder" and is_book_asset:
                sidecar = image_path.with_suffix(".source.json")
                if not sidecar.is_file():
                    errors.append(f"{document.path.name}: thiếu provenance {sidecar.name}")
    return errors


def safe_file(root: Path, relative_url: str) -> Path | None:
    candidate = (root / unquote(relative_url)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


class NotebookRequestHandler(BaseHTTPRequestHandler):
    server_version = "D2LNotebook/1.0"

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        cache_control: str = "no-cache",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", status)

    def _send_file(self, path: Path, cache_control: str = "no-cache") -> None:
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if mime_type.startswith("text/") or mime_type in {"application/javascript", "image/svg+xml"}:
            mime_type += "; charset=utf-8"
        self._send_bytes(path.read_bytes(), mime_type, cache_control=cache_control)

    def do_GET(self) -> None:  # noqa: N802
        request_path = unquote(urlparse(self.path).path)
        try:
            if request_path == "/":
                self._send_file(STATIC_ROOT / "index.html")
                return
            if request_path == "/api/notebook":
                documents = discover_documents()
                index = load_book_index()
                self._send_json(
                    {
                        "book": index["book"],
                        "documents": [document.summary() for document in documents],
                    }
                )
                return
            if request_path == "/api/glossary":
                self._send_json(load_glossary())
                return
            if request_path.startswith("/api/chapter/"):
                wanted_id = request_path.removeprefix("/api/chapter/")
                document = next(
                    (doc for doc in discover_documents() if doc.id == wanted_id), None
                )
                if document is None:
                    self._send_json({"error": "Chapter not found"}, HTTPStatus.NOT_FOUND)
                else:
                    self._send_json(chapter_payload(document))
                return
            if request_path.startswith("/static/"):
                path = safe_file(STATIC_ROOT, request_path.removeprefix("/static/"))
                if path:
                    self._send_file(path, "public, max-age=300")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                return
            if request_path.startswith("/content/"):
                path = safe_file(CONTENT_ROOT, request_path.removeprefix("/content/"))
                allowed = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
                if path and path.suffix.lower() in allowed:
                    self._send_file(path, "public, max-age=300")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (NotebookError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Địa chỉ lắng nghe")
    parser.add_argument("--port", default=8000, type=int, help="Cổng HTTP")
    parser.add_argument(
        "--check", action="store_true", help="Kiểm tra nội dung rồi thoát"
    )
    args = parser.parse_args()

    errors = validate_notebook()
    if errors:
        print("Notebook chưa hợp lệ:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if args.check:
        documents = discover_documents()
        print(f"OK: {len(documents)} documents, 0 lỗi.")
        return 0

    server = ThreadingHTTPServer((args.host, args.port), NotebookRequestHandler)
    print(f"Notebook đang chạy tại http://{args.host}:{server.server_port}")
    print("Nhấn Ctrl+C để dừng.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng notebook.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
