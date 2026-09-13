from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


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

    def test_math_is_preserved_for_mathjax_without_markdown_emphasis(self) -> None:
        source = app.discover_documents()[0]
        equation = r"$$P(w_o\mid w_c)=\frac{e^{u_o}}{\sum_{i\in V}e^{u_i}}.$$"
        document = app.Document(
            id=source.id,
            path=source.path,
            metadata=source.metadata,
            body=f"# Demo\n\nInline $w_c$ and display:\n\n{equation}",
        )
        rendered = app.render_document(document)
        self.assertIn("Inline $w_c$", rendered)
        self.assertIn(equation, rendered)
        self.assertNotIn("<em", rendered)

    def test_chapter_15_skipgram_algorithm_renders_as_study_block(self) -> None:
        document = next(doc for doc in app.discover_documents() if doc.id == "chapter-15")
        rendered = app.render_document(document)
        self.assertIn('class="algorithm-block"', rendered)
        self.assertIn('class="algorithm-label"', rendered)
        self.assertIn("LUỒNG THUẬT TOÁN · SKIP-GRAM", rendered)
        self.assertNotIn("<em V", rendered)

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

    def test_chapters_7_to_11_keep_complete_pdf_section_order(self) -> None:
        expected = {
            "chapter-07": """
                7.1 From Fully Connected Layers to Convolutions
                7.1.1 Invariance
                7.1.2 Constraining the MLP
                7.1.3 Convolutions
                7.1.4 Channels
                7.1.5 Summary and Discussion
                7.1.6 Exercises
                7.2 Convolutions for Images
                7.2.1 The Cross-Correlation Operation
                7.2.2 Convolutional Layers
                7.2.3 Object Edge Detection in Images
                7.2.4 Learning a Kernel
                7.2.5 Cross-Correlation and Convolution
                7.2.6 Feature Map and Receptive Field
                7.2.7 Summary
                7.2.8 Exercises
                7.3 Padding and Stride
                7.3.1 Padding
                7.3.2 Stride
                7.3.3 Summary and Discussion
                7.3.4 Exercises
                7.4 Multiple Input and Multiple Output Channels
                7.4.1 Multiple Input Channels
                7.4.2 Multiple Output Channels
                7.4.3 1 × 1 Convolutional Layer
                7.4.4 Discussion
                7.4.5 Exercises
                7.5 Pooling
                7.5.1 Maximum Pooling and Average Pooling
                7.5.2 Padding and Stride
                7.5.3 Multiple Channels
                7.5.4 Summary
                7.5.5 Exercises
                7.6 Convolutional Neural Networks (LeNet)
                7.6.1 LeNet
                7.6.2 Training
                7.6.3 Summary
                7.6.4 Exercises
            """,
            "chapter-08": """
                8.1 Deep Convolutional Neural Networks (AlexNet)
                8.1.1 Representation Learning
                8.1.2 AlexNet
                8.1.3 Training
                8.1.4 Discussion
                8.1.5 Exercises
                8.2 Networks Using Blocks (VGG)
                8.2.1 VGG Blocks
                8.2.2 VGG Network
                8.2.3 Training
                8.2.4 Summary
                8.2.5 Exercises
                8.3 Network in Network (NiN)
                8.3.1 NiN Blocks
                8.3.2 NiN Model
                8.3.3 Training
                8.3.4 Summary
                8.3.5 Exercises
                8.4 Multi-Branch Networks (GoogLeNet)
                8.4.1 Inception Blocks
                8.4.2 GoogLeNet Model
                8.4.3 Training
                8.4.4 Discussion
                8.4.5 Exercises
                8.5 Batch Normalization
                8.5.1 Training Deep Networks
                8.5.2 Batch Normalization Layers
                8.5.3 Implementation from Scratch
                8.5.4 LeNet with Batch Normalization
                8.5.5 Concise Implementation
                8.5.6 Discussion
                8.5.7 Exercises
                8.6 Residual Networks (ResNet) and ResNeXt
                8.6.1 Function Classes
                8.6.2 Residual Blocks
                8.6.3 ResNet Model
                8.6.4 Training
                8.6.5 ResNeXt
                8.6.6 Summary and Discussion
                8.6.7 Exercises
                8.7 Densely Connected Networks (DenseNet)
                8.7.1 From ResNet to DenseNet
                8.7.2 Dense Blocks
                8.7.3 Transition Layers
                8.7.4 DenseNet Model
                8.7.5 Training
                8.7.6 Summary and Discussion
                8.7.7 Exercises
                8.8 Designing Convolution Network Architectures
                8.8.1 The AnyNet Design Space
                8.8.2 Distributions and Parameters of Design Spaces
                8.8.3 RegNet
                8.8.4 Training
                8.8.5 Discussion
                8.8.6 Exercises
            """,
            "chapter-09": """
                9.1 Working with Sequences
                9.1.1 Autoregressive Models
                9.1.2 Sequence Models
                9.1.3 Training
                9.1.4 Prediction
                9.1.5 Summary
                9.1.6 Exercises
                9.2 Converting Raw Text into Sequence Data
                9.2.1 Reading the Dataset
                9.2.2 Tokenization
                9.2.3 Vocabulary
                9.2.4 Putting It All Together
                9.2.5 Exploratory Language Statistics
                9.2.6 Summary
                9.2.7 Exercises
                9.3 Language Models
                9.3.1 Learning Language Models
                9.3.2 Perplexity
                9.3.3 Partitioning Sequences
                9.3.4 Summary and Discussion
                9.3.5 Exercises
                9.4 Recurrent Neural Networks
                9.4.1 Neural Networks without Hidden States
                9.4.2 Recurrent Neural Networks with Hidden States
                9.4.3 RNN-Based Character-Level Language Models
                9.4.4 Summary
                9.4.5 Exercises
                9.5 Recurrent Neural Network Implementation
                9.5.1 RNN Model
                9.5.2 RNN-Based Language Model
                9.5.3 Gradient Clipping
                9.5.4 Training
                9.5.5 Decoding
                9.5.6 Summary
                9.5.7 Exercises
                9.6 Concise Implementation of Recurrent Neural Networks
                9.6.1 Defining the Model
                9.6.2 Training and Predicting
                9.6.3 Summary
                9.6.4 Exercises
                9.7 Backpropagation Through Time
                9.7.1 Analysis of Gradients in RNNs
                9.7.2 Backpropagation Through Time in Detail
                9.7.3 Summary
                9.7.4 Exercises
            """,
            "chapter-10": """
                10.1 Long Short-Term Memory (LSTM)
                10.1.1 Gated Memory Cell
                10.1.2 Implementation from Scratch
                10.1.3 Concise Implementation
                10.1.4 Summary
                10.1.5 Exercises
                10.2 Gated Recurrent Units (GRU)
                10.2.1 Reset Gate and Update Gate
                10.2.2 Candidate Hidden State
                10.2.3 Hidden State
                10.2.4 Implementation from Scratch
                10.2.5 Concise Implementation
                10.2.6 Summary
                10.2.7 Exercises
                10.3 Deep Recurrent Neural Networks
                10.3.1 Implementation from Scratch
                10.3.2 Concise Implementation
                10.3.3 Summary
                10.3.4 Exercises
                10.4 Bidirectional Recurrent Neural Networks
                10.4.1 Implementation from Scratch
                10.4.2 Concise Implementation
                10.4.3 Summary
                10.4.4 Exercises
                10.5 Machine Translation and the Dataset
                10.5.1 Downloading and Preprocessing the Dataset
                10.5.2 Tokenization
                10.5.3 Loading Sequences of Fixed Length
                10.5.4 Reading the Dataset
                10.5.5 Summary
                10.5.6 Exercises
                10.6 The Encoder−Decoder Architecture
                10.6.1 Encoder
                10.6.2 Decoder
                10.6.3 Putting the Encoder and Decoder Together
                10.6.4 Summary
                10.6.5 Exercises
                10.7 Sequence-to-Sequence Learning for Machine Translation
                10.7.1 Teacher Forcing
                10.7.2 Encoder
                10.7.3 Decoder
                10.7.4 Encoder–Decoder for Sequence-to-Sequence Learning
                10.7.5 Loss Function with Masking
                10.7.6 Training
                10.7.7 Prediction
                10.7.8 Evaluation of Predicted Sequences
                10.7.9 Summary
                10.7.10 Exercises
                10.8 Beam Search
                10.8.1 Greedy Search
                10.8.2 Exhaustive Search
                10.8.3 Beam Search
                10.8.4 Summary
                10.8.5 Exercises
            """,
            "chapter-11": """
                11.1 Queries, Keys, and Values
                11.1.1 Visualization
                11.1.2 Summary
                11.1.3 Exercises
                11.2 Attention Pooling by Similarity
                11.2.1 Kernels and Data
                11.2.2 Attention Pooling via Nadaraya–Watson Regression
                11.2.3 Adapting Attention Pooling
                11.2.4 Summary
                11.2.5 Exercises
                11.3 Attention Scoring Functions
                11.3.1 Dot Product Attention
                11.3.2 Convenience Functions
                11.3.3 Scaled Dot Product Attention
                11.3.4 Additive Attention
                11.3.5 Summary
                11.3.6 Exercises
                11.4 The Bahdanau Attention Mechanism
                11.4.1 Model
                11.4.2 Defining the Decoder with Attention
                11.4.3 Training
                11.4.4 Summary
                11.4.5 Exercises
                11.5 Multi-Head Attention
                11.5.1 Model
                11.5.2 Implementation
                11.5.3 Summary
                11.5.4 Exercises
                11.6 Self-Attention and Positional Encoding
                11.6.1 Self-Attention
                11.6.2 Comparing CNNs, RNNs, and Self-Attention
                11.6.3 Positional Encoding
                11.6.4 Summary
                11.6.5 Exercises
                11.7 The Transformer Architecture
                11.7.1 Model
                11.7.2 Positionwise Feed-Forward Networks
                11.7.3 Residual Connection and Layer Normalization
                11.7.4 Encoder
                11.7.5 Decoder
                11.7.6 Training
                11.7.7 Summary
                11.7.8 Exercises
                11.8 Transformers for Vision
                11.8.1 Model
                11.8.2 Patch Embedding
                11.8.3 Vision Transformer Encoder
                11.8.4 Putting It All Together
                11.8.5 Training
                11.8.6 Summary and Discussion
                11.8.7 Exercises
                11.9 Large-Scale Pretraining with Transformers
                11.9.1 Encoder-Only
                11.9.2 Encoder–Decoder
                11.9.3 Decoder-Only
                11.9.4 Scalability
                11.9.5 Large Language Models
                11.9.6 Summary and Discussion
                11.9.7 Exercises
            """,
        }
        documents = {doc.id: doc for doc in app.discover_documents()}
        for document_id, heading_text in expected.items():
            document = documents[document_id]
            self.assertEqual(document.metadata["status"], "reviewed")
            headings = [line.strip() for line in heading_text.splitlines() if line.strip()]
            positions = [document.body.index(heading) for heading in headings]
            self.assertEqual(positions, sorted(positions), document_id)
            self.assertNotIn("nội dung chưa được biên soạn", document.body)

    def test_chapters_7_to_11_python_examples_compile(self) -> None:
        documents = {doc.id: doc for doc in app.discover_documents()}
        for number in range(7, 12):
            document = documents[f"chapter-{number:02d}"]
            code_blocks = re.findall(r"```python\n(.*?)```", document.body, re.DOTALL)
            self.assertGreaterEqual(len(code_blocks), 1, document.id)
            for block_number, code in enumerate(code_blocks, start=1):
                compile(code, f"{document.id}:block-{block_number}", "exec")

    def test_new_chapter_figures_have_pdf_provenance(self) -> None:
        expected = [
            "content/assets/chapter-07/figure-7-2-1-cross-correlation.source.json",
            "content/assets/chapter-08/figure-8-6-2-residual-block.source.json",
            "content/assets/chapter-09/figure-9-4-1-rnn-hidden-state.source.json",
            "content/assets/chapter-10/figure-10-1-4-lstm.source.json",
            "content/assets/chapter-11/figure-11-7-1-transformer.source.json",
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
        self.assertIn('id="skinButton"', index_html)
        self.assertIn('id="skinMenu"', index_html)
        self.assertEqual(index_html.count("?v=reader-fixes-5"), 2)
        self.assertIn('id="bookmarkButton"', index_html)
        self.assertIn('id="resumeButton"', index_html)
        self.assertIn('class="ancient-manuscript"', index_html)
        self.assertIn('id="brandSeal">武</span>', index_html)
        self.assertIn("paginateAuthoredPages", script)
        self.assertIn("function paginationBox()", script)
        self.assertIn("filter((box) => box.width > 0 && box.height > 0)", script)
        self.assertIn("fillRemainingWithList", script)
        self.assertIn('localStorage.setItem("d2l-theme"', script)
        self.assertIn('localStorage.setItem("d2l-skin"', script)
        self.assertIn('localStorage.setItem("d2l-bookmarks"', script)
        self.assertIn('classList.toggle("is-bookmarked"', script)
        self.assertIn(':root[data-theme="dark"]', styles)
        self.assertIn("--cinnabar:", styles)
        self.assertIn(".notebook-spread.is-bookmarked::after", styles)
        self.assertIn(".codehilite", styles)
        self.assertIn(".algorithm-block", styles)
        self.assertIn("--algorithm-bg:", styles)
        self.assertIn("clearTextSelection", script)
        self.assertIn("event.preventDefault();", script)
        self.assertRegex(styles, r"\.chapter-page\s*\{[^}]*overflow:hidden")

        skin_ids = ["manuscript", "xuan", "bamboo", "porcelain", "dunhuang", "vermilion"]
        required_skin_tokens = [
            "--paper:", "--ink:", "--accent:", "--body-bg:", "--sidebar-bg:",
            "--sheet-texture:", "--sheet-edge:", "--spine-bg:", "--read-size:",
        ]
        for skin_id in skin_ids:
            self.assertIn(f'id: "{skin_id}"', script)
            light_match = re.search(
                rf':root\[data-skin="{skin_id}"\]\s*\{{(.*?)\n\}}', styles, re.DOTALL
            )
            dark_match = re.search(
                rf':root\[data-skin="{skin_id}"\]\[data-theme="dark"\]\s*\{{(.*?)\n\}}',
                styles,
                re.DOTALL,
            )
            self.assertIsNotNone(light_match, f"missing light tokens for {skin_id}")
            self.assertIsNotNone(dark_match, f"missing dark tokens for {skin_id}")
            for token in required_skin_tokens:
                self.assertIn(token, light_match.group(1), f"{skin_id} light: {token}")
                self.assertIn(token, dark_match.group(1), f"{skin_id} dark: {token}")

    def test_readme_uses_current_chapter_one_screenshots(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        screenshots = [
            "docs/screenshots/chapter-01-opening.jpg",
            "docs/screenshots/chapter-01-learning-loop.jpg",
            "docs/screenshots/chapter-01-dark-glossary.jpg",
        ]
        for relative_path in screenshots:
            self.assertIn(relative_path, readme)
            image_path = PROJECT_ROOT / relative_path
            self.assertTrue(image_path.is_file(), relative_path)
            with Image.open(image_path) as screenshot:
                self.assertEqual(screenshot.format, "JPEG")
                self.assertGreaterEqual(screenshot.width, 1200)
                self.assertGreaterEqual(screenshot.height, 700)


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
