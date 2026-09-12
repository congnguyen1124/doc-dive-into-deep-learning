from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402
from tools.extract_pdf_figure import crop_box, parse_crop  # noqa: E402


class NotebookContentTests(unittest.TestCase):
    def test_all_indexed_documents_exist_and_validate(self) -> None:
        documents = app.discover_documents()
        self.assertEqual(len(documents), 23)
        self.assertEqual(len([doc for doc in documents if doc.id.startswith("chapter-")]), 21)
        self.assertEqual(app.validate_notebook(), [])

    def test_chapter_titles_and_filenames_follow_pdf_index(self) -> None:
        expected = {entry["id"]: entry for entry in app.load_book_index()["documents"]}
        for document in app.discover_documents():
            self.assertEqual(document.metadata["title"], expected[document.id]["title"])
            self.assertEqual(document.path.name, expected[document.id]["filename"])

    def test_glossary_markup_renders_accessibly(self) -> None:
        source = app.discover_documents()[0]
        document = app.Document(
            id=source.id,
            path=source.path,
            metadata=source.metadata,
            body="# Demo\n\nMột {{term:tensor|Tensor}} nhỏ.",
        )
        rendered = app.render_document(document)
        self.assertIn('class="glossary-term"', rendered)
        self.assertIn('data-term="tensor"', rendered)
        self.assertIn('tabindex="0"', rendered)
        self.assertIn(">Tensor</span>", rendered)

    def test_safe_file_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "safe"
            root.mkdir()
            outside = Path(temp_dir) / "outside.txt"
            outside.write_text("secret", encoding="utf-8")
            self.assertIsNone(app.safe_file(root, "../outside.txt"))


class FigureExtractionTests(unittest.TestCase):
    def test_crop_parser_and_pixel_box(self) -> None:
        crop = parse_crop("10,20,50,40")
        self.assertEqual(crop, (10.0, 20.0, 50.0, 40.0))
        self.assertEqual(crop_box((1000, 800), crop), (100, 160, 600, 480))

    def test_crop_parser_rejects_page_overflow(self) -> None:
        with self.assertRaises(Exception):
            parse_crop("80,0,30,100")


if __name__ == "__main__":
    unittest.main()
