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

    def test_python_fences_render_with_offline_syntax_highlighting(self) -> None:
        source = app.discover_documents()[0]
        document = app.Document(
            id=source.id,
            path=source.path,
            metadata=source.metadata,
            body="# Demo\n\n```python\nfrom torch import nn\nmodel = nn.Linear(2, 1)\n```",
        )
        rendered = app.render_document(document)
        self.assertIn('class="codehilite"', rendered)
        self.assertIn('class="kn"', rendered)

    def test_authored_chapters_are_split_into_note_pages(self) -> None:
        for document in app.discover_documents()[:3]:
            payload = app.chapter_payload(document)
            self.assertGreaterEqual(len(payload["pages"]), 2)
            self.assertNotIn("<!-- pagebreak -->", "".join(payload["pages"]))
            self.assertIn(payload["status"], {"draft", "reviewed"})

    def test_first_three_chapters_keep_pdf_section_order(self) -> None:
        expected = {
            "chapter-01": [
                "1.1 A Motivating Example", "1.2 Key Components", "1.2.1 Data",
                "1.2.2 Models", "1.2.3 Objective Functions",
                "1.2.4 Optimization Algorithms", "1.3 Kinds of Machine Learning Problems",
                "1.3.1 Supervised Learning", "1.3.2 Unsupervised and Self-Supervised Learning",
                "1.3.3 Interacting with an Environment", "1.3.4 Reinforcement Learning",
                "1.4 Roots", "1.5 The Road to Deep Learning", "1.6 Success Stories",
                "1.7 The Essence of Deep Learning", "1.8 Summary", "1.9 Exercises",
            ],
            "chapter-02": [
                "2.1 Data Manipulation", "2.1.1 Getting Started", "2.1.2 Indexing and Slicing",
                "2.1.3 Operations", "2.1.4 Broadcasting", "2.1.5 Saving Memory",
                "2.1.6 Conversion to Other Python Objects", "2.1.7 Summary", "2.1.8 Exercises",
                "2.2 Data Preprocessing", "2.2.1 Reading the Dataset", "2.2.2 Data Preparation",
                "2.2.3 Conversion to the Tensor Format", "2.2.4 Discussion", "2.2.5 Exercises",
                "2.3 Linear Algebra", "2.3.1 Scalars", "2.3.2 Vectors", "2.3.3 Matrices",
                "2.3.4 Tensors", "2.3.5 Basic Properties of Tensor Arithmetic", "2.3.6 Reduction",
                "2.3.7 Non-Reduction Sum", "2.3.8 Dot Products", "2.3.9 Matrix–Vector Products",
                "2.3.10 Matrix–Matrix Multiplication", "2.3.11 Norms", "2.3.12 Discussion",
                "2.3.13 Exercises", "2.4 Calculus", "2.4.1 Derivatives and Differentiation",
                "2.4.2 Visualization Utilities", "2.4.3 Partial Derivatives and Gradients",
                "2.4.4 Chain Rule", "2.4.5 Discussion", "2.4.6 Exercises",
                "2.5 Automatic Differentiation", "2.5.1 A Simple Function",
                "2.5.2 Backward for Non-Scalar Variables", "2.5.3 Detaching Computation",
                "2.5.4 Gradients and Python Control Flow", "2.5.5 Discussion", "2.5.6 Exercises",
                "2.6 Probability and Statistics", "2.6.1 A Simple Example: Tossing Coins",
                "2.6.2 A More Formal Treatment", "2.6.3 Random Variables",
                "2.6.4 Multiple Random Variables", "2.6.5 An Example", "2.6.6 Expectations",
                "2.6.7 Discussion", "2.6.8 Exercises", "2.7 Documentation",
                "2.7.1 Functions and Classes in a Module", "2.7.2 Specific Functions and Classes",
            ],
            "chapter-03": [
                "3.1 Linear Regression", "3.1.1 Basics", "3.1.2 Vectorization for Speed",
                "3.1.3 The Normal Distribution and Squared Loss",
                "3.1.4 Linear Regression as a Neural Network", "3.1.5 Summary", "3.1.6 Exercises",
                "3.2 Object-Oriented Design for Implementation", "3.2.1 Utilities", "3.2.2 Models",
                "3.2.3 Data", "3.2.4 Training", "3.2.5 Summary", "3.2.6 Exercises",
                "3.3 Synthetic Regression Data", "3.3.1 Generating the Dataset",
                "3.3.2 Reading the Dataset", "3.3.3 Concise Implementation of the Data Loader",
                "3.3.4 Summary", "3.3.5 Exercises",
                "3.4 Linear Regression Implementation from Scratch", "3.4.1 Defining the Model",
                "3.4.2 Defining the Loss Function", "3.4.3 Defining the Optimization Algorithm",
                "3.4.4 Training", "3.4.5 Summary", "3.4.6 Exercises",
                "3.5 Concise Implementation of Linear Regression", "3.5.1 Defining the Model",
                "3.5.2 Defining the Loss Function", "3.5.3 Defining the Optimization Algorithm",
                "3.5.4 Training", "3.5.5 Summary", "3.5.6 Exercises", "3.6 Generalization",
                "3.6.1 Training Error and Generalization Error", "3.6.2 Underfitting or Overfitting?",
                "3.6.3 Model Selection", "3.6.4 Summary", "3.6.5 Exercises", "3.7 Weight Decay",
                "3.7.1 Norms and Weight Decay", "3.7.2 High-Dimensional Linear Regression",
                "3.7.3 Implementation from Scratch", "3.7.4 Concise Implementation",
                "3.7.5 Summary", "3.7.6 Exercises",
            ],
        }
        documents = {doc.id: doc for doc in app.discover_documents()}
        for document_id, headings in expected.items():
            body = documents[document_id].body
            positions = [body.index(f"{heading}") for heading in headings]
            self.assertEqual(positions, sorted(positions), document_id)

    def test_newly_authored_chapters_keep_complete_pdf_section_order(self) -> None:
        expected = {
            "chapter-14": [
                "14.1 Image Augmentation", "14.2 Fine-Tuning",
                "14.3 Object Detection and Bounding Boxes", "14.4 Anchor Boxes",
                "14.5 Multiscale Object Detection", "14.6 The Object Detection Dataset",
                "14.7 Single Shot Multibox Detection", "14.8 Region-based CNNs (R-CNNs)",
                "14.9 Semantic Segmentation and the Dataset", "14.10 Transposed Convolution",
                "14.11 Fully Convolutional Networks", "14.12 Neural Style Transfer",
                "14.13 Image Classification (CIFAR-10) on Kaggle",
                "14.14 Dog Breed Identification (ImageNet Dogs) on Kaggle",
            ],
            "chapter-15": [
                "15.1 Word Embedding (word2vec)", "15.2 Approximate Training",
                "15.3 The Dataset for Pretraining Word Embeddings", "15.4 Pretraining word2vec",
                "15.5 Word Embedding with Global Vectors (GloVe)", "15.6 Subword Embedding",
                "15.7 Word Similarity and Analogy",
                "15.8 Bidirectional Encoder Representations from Transformers (BERT)",
                "15.9 The Dataset for Pretraining BERT", "15.10 Pretraining BERT",
            ],
            "chapter-21": ["21.1 Overview of Recommender Systems"],
        }
        documents = {doc.id: doc for doc in app.discover_documents()}
        for document_id, headings in expected.items():
            document = documents[document_id]
            self.assertEqual(document.metadata["status"], "reviewed")
            positions = [document.body.index(heading) for heading in headings]
            self.assertEqual(positions, sorted(positions), document_id)
            self.assertNotIn("nội dung chưa được biên soạn", document.body)

    def test_new_chapter_figures_have_pdf_provenance(self) -> None:
        expected = [
            "content/assets/chapter-14/figure-14-7-1-ssd-architecture.source.json",
            "content/assets/chapter-15/figure-15-8-1-bert-comparison.source.json",
            "content/assets/chapter-21/figure-21-1-1-recommendation-process.source.json",
        ]
        for relative_path in expected:
            sidecar = PROJECT_ROOT / relative_path
            self.assertTrue(sidecar.is_file(), relative_path)
            self.assertIn('"source_pdf_sha256"', sidecar.read_text(encoding="utf-8"))

    def test_safe_file_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "safe"
            root.mkdir()
            outside = Path(temp_dir) / "outside.txt"
            outside.write_text("secret", encoding="utf-8")
            self.assertIsNone(app.safe_file(root, "../outside.txt"))

    def test_reader_has_real_page_layers_theme_and_no_sheet_scroll(self) -> None:
        index_html = (PROJECT_ROOT / "static" / "index.html").read_text(encoding="utf-8")
        script = (PROJECT_ROOT / "static" / "app.js").read_text(encoding="utf-8")
        styles = (PROJECT_ROOT / "static" / "styles.css").read_text(encoding="utf-8")
        self.assertIn('id="flipFront"', index_html)
        self.assertIn('id="flipBack"', index_html)
        self.assertIn('id="turnUnderlayRight"', index_html)
        self.assertIn('id="themeButton"', index_html)
        self.assertIn('id="bookmarkButton"', index_html)
        self.assertIn('id="resumeButton"', index_html)
        self.assertIn('class="ancient-manuscript"', index_html)
        self.assertIn('aria-hidden="true"><span>學</span>', index_html)
        self.assertIn("paginateAuthoredPages", script)
        self.assertIn("fillRemainingWithList", script)
        self.assertIn('localStorage.setItem("d2l-theme"', script)
        self.assertIn('localStorage.setItem("d2l-bookmarks"', script)
        self.assertIn('classList.toggle("is-bookmarked"', script)
        self.assertIn(':root[data-theme="dark"]', styles)
        self.assertIn("--cinnabar:", styles)
        self.assertIn(".notebook-spread.is-bookmarked::after", styles)
        self.assertIn(".codehilite", styles)
        self.assertRegex(styles, r"\.chapter-page\s*\{[^}]*overflow:hidden")


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
