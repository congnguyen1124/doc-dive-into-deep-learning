# Authoring contract for Dive into Deep Learning notes

## Mission and learner profile

This repository is a Vietnamese, beginner-first learning notebook based on the local source book `../didl.pdf` (*Dive into Deep Learning*). The learner is new to machine learning and deep learning and may be discouraged by dense terminology. Teach for understanding, not for literal sentence-by-sentence translation.

Only author whole chapters when the learner explicitly requests them. Multiple explicitly requested chapters may be completed together while preserving the same source-review standard for each one.

## Source of truth

- Treat `../didl.pdf` as the primary source for chapter names, numbering, section order, equations, exercises, figure labels, and original page references.
- Read the complete requested chapter range before writing. Inspect relevant pages visually whenever layout, equations, tables, or figures matter; text extraction alone is not enough.
- Do not silently import content from a different edition or the live D2L website. If another source is useful, label it clearly as supplemental.
- Do not invent quotations, book claims, exercise wording, figure numbers, results, or citations.

## File names and ordering

- Store one chapter in one Markdown file under `content/chapters/`.
- Use the exact English chapter title printed in the PDF and the pattern `NN - Exact Chapter Title.md`. Keep punctuation and capitalization faithful to the PDF.
- Store appendices separately under `content/appendices/` with the pattern `Appendix X - Exact Appendix Title.md`.
- Keep the YAML front matter keys used by the reader: `type`, `number`, `order`, `title`, `book_pages`, `pdf_pages`, and `status`.
- Set `status: placeholder` until meaningful Vietnamese notes exist, `status: draft` while authoring, and `status: reviewed` only after source and rendering checks.
- Never rename an authored chapter merely to translate its filename. The exact English name is the stable identifier.

## Teaching and translation style

- Write the chapter body in natural Vietnamese. Retain an English technical term when a Vietnamese translation would be awkward, misleading, or uncommon.
- On the first important use of a retained English term, use `{{term:key|Displayed English Term}}` and add a Vietnamese explanation to `content/glossary.json`.
- Explain each idea in this order when practical: intuition, concrete example, formal definition or equation, then when and why it is useful.
- Define every mathematical symbol near its first use. Read important equations aloud in Vietnamese and explain what changes when each variable increases or decreases.
- Prefer small numeric examples, shapes of tensors, annotated code, comparisons, and everyday analogies. Mark the limit of an analogy so it does not create a false mental model.
- Distinguish clearly among a book statement, the notebook author's explanation, and an optional modern or practical note.
- Avoid unexplained jargon, translated word salad, false certainty, and motivational filler.

## Required structure for an authored chapter

Keep the original numbered sections and titles from the PDF. Within that structure, include:

1. **Mục tiêu học tập**: observable outcomes for the learner.
2. **Bản đồ chương**: a compact overview showing how the ideas connect.
3. **Bức tranh tổng quan**: why the chapter exists and what prior knowledge it uses.
4. **Bài học theo từng mục**: beginner-first explanations, examples, equations, and runnable Python/PyTorch snippets where helpful.
5. **Điểm hay và ý nghĩa**: what is elegant, surprising, or practically important.
6. **Sau chương này bạn làm được gì?**: concrete capabilities, not vague claims.
7. **Tóm tắt kiến thức**: a compact mental model and a checklist.
8. **Bài tập**: organized from recall to application and extension.
9. **Gợi ý và lời giải**: give a hint before a worked solution; explain reasoning and common mistakes rather than only the final answer. Use `<details>` blocks when hiding a solution improves learning.
10. **Thuật ngữ cần nhớ**: English term, Vietnamese explanation, and a short example.
11. **Nguồn và phạm vi**: printed book pages, physical PDF pages, and any clearly labeled supplemental sources.

## Glossary contract

`content/glossary.json` is the single glossary source for the UI. A term entry uses this shape:

```json
{
  "tensor": {
    "term": "Tensor",
    "vi": "Cách gọi tiếng Việt ngắn gọn hoặc để nguyên tiếng Anh",
    "explanation": "Giải thích thân thiện cho người mới.",
    "example": "Một ví dụ rất ngắn.",
    "chapter": "chapter-02"
  }
}
```

- Keys are lowercase ASCII kebab-case and remain stable.
- Use `{{term:tensor|Tensor}}` in Markdown. The displayed text may differ, but the key must exist in the glossary before a chapter can be marked `reviewed`.
- Keep tooltip explanations concise. Put long explanations in the chapter body.

## Figures and diagrams

- Figures labeled as originating from the book must be extracted from `../didl.pdf`; do not redraw them from memory or substitute a web image.
- Use `tools/extract_pdf_figure.py` to render/crop the physical PDF page. Keep the generated `.source.json` sidecar as provenance.
- Save book figures under `content/assets/chapter-NN/` and use descriptive stable names such as `figure-7-1-correlation-and-convolution.png`.
- Embed an extracted figure with standard Markdown and a title containing its provenance:

  ```markdown
  ![Mô tả dễ hiểu](../assets/chapter-07/figure-7-1.png "Nguồn: didl.pdf, Figure 7.1, trang sách 235, trang PDF 275")
  ```

- Additional diagrams are encouraged when they make a relationship easier to understand. Save them under `content/assets/chapter-NN/generated/` and label them **Sơ đồ bổ sung**, never as an original book figure.
- Preserve labels, axes, aspect ratio, and legibility. Do not crop away information needed to interpret the figure.

## Code, exercises, and correctness

- Prefer Python and PyTorch, matching this PDF edition where applicable. State shapes and dtypes when they matter.
- Keep code examples small and executable. Seed randomness when the result is discussed, and do not claim execution unless it was run.
- Fence Python examples with ` ```python ` so the server can apply offline Pygments syntax highlighting. Follow readable Python conventions: `snake_case` functions/variables, `PascalCase` classes, four-space indentation, useful type hints/docstrings, and shape comments at tensor boundaries. Avoid compressed one-line teaching code.
- Write algorithms and pseudocode as semantic `<div class="algorithm-block" markdown="1">` study blocks with an `.algorithm-label`, explicit inputs/outputs, and ordered steps. Do not use unsupported LaTeX `algorithm` environments. Their jade/ink treatment must remain visually distinct from dark Python code blocks in every skin and theme.
- Preserve the intent and numbering of book exercises, but explain them in original Vietnamese prose. Add notebook exercises only when labeled **Bài tập bổ sung**.
- For each worked problem: identify knowns and unknowns, choose the idea or formula, solve step by step, sanity-check the result, then name the common trap.
- Never expose a full solution before a hint if the exercise is intended for self-practice.

## Reader application contract

- `app.py` discovers Markdown from YAML metadata; do not hardcode chapter content into the UI.
- Keep content assets relative to their Markdown file so the server can rewrite them safely.
- Preserve keyboard navigation, mobile layout, reduced-motion support, completion tracking, image captions, image zoom, and glossary hover/focus behavior when changing the UI.
- Keep all application chrome in English (navigation, buttons, search, status, errors, and accessibility labels). Chapter Markdown and learning explanations remain in Vietnamese. A deliberately featured Vietnamese study motto is the sole chrome exception.
- Present authored Markdown as an open, two-page note-paper spread on desktop. Use `<!-- pagebreak -->` between intentional study sections; the client may subdivide a section into additional physical note pages to fit the current viewport. On narrow screens, show one note page at a time without losing content.
- A physical note page must never have its own vertical scrollbar. Measure and paginate rendered blocks into following pages; do not clip or hide authored material. Horizontal scrolling is allowed only where it is necessary for code or equations.
- Support page turning by pointer drag/swipe as well as buttons and the Left/Right arrow keys. During a turn, stage the bottom page and both faces of the moving leaf so the learner can see the upcoming content before releasing the pointer. Honor `prefers-reduced-motion`, and never make dragging the only navigation method.
- Pointer-driven page turning must never leave browser text selection active. Prevent the drag gesture's default selection behavior, clear any selection on move/end/cancel, and clean up when pointer capture is lost.
- Keep body copy visually aligned through typography rather than ruled lines: note paper has no horizontal rules, while text uses a compact, consistent baseline, restrained heading sizes, modest paragraph spacing, predictable vertical rhythm, and tables/code blocks that do not push text outside the sheet.
- Provide an English-labeled Light/Dark theme control. Use softly off-white paper in Light mode and a dark paper with clearly contrasting text in Dark mode; persist the learner's explicit choice and otherwise respect the operating-system preference.
- Provide persistent page marks. A learner can mark the current physical page, see that it is marked, and jump back to the latest mark. Store a stable text anchor as well as the page index so reflow across viewport sizes does not silently lose the reading position.
- Ship the reader as a **style library**: several complete ancient-Chinese skins selectable from an English-labeled
  picker in the top bar, persisted per learner. A skin is declared once in `SKINS` in `static/app.js` and defined
  purely as custom properties under `:root[data-skin="<id>"]` and `:root[data-skin="<id>"][data-theme="dark"]`
  in `static/styles.css`. Every skin must define the complete token contract in BOTH modes — never hardcode a
  surface colour in a rule, add a token instead — and `layoutSignature()` must keep the skin in its cache key so
  pages are re-measured when type metrics change.
- A picker entry is not complete until selecting it visibly changes computed paper, ink, surrounding surfaces, seal, and turn character in both Light and Dark modes. Keep a regression test that checks every `SKINS` id has both CSS token blocks.
- Bump the `?v=` cache key on `styles.css` and `app.js` whenever a reader release changes either asset; otherwise an already-open browser may keep the old UI and make a valid skin change appear broken.
- A skin may retune the page turn through `data-turn` (duration, easing, lift, bend, sheen width, cast reach), but
  every variant keeps the same staged DOM and must read as real paper: the hinge side darkens, a specular band
  sweeps the leaf, and the lifted leaf casts a shadow on the sheet it uncovers. Never make a turn instant or flat.
- Use an ancient Chinese manuscript visual language throughout the reader: xuan-paper/parchment tones, ink-dark text, cinnabar seals, restrained gold and jade accents, stitched-binding details, squared archival controls, and subtle wood or ink-wash surroundings. Dark mode should feel like a night-ink manuscript rather than a generic blue-black application.
- Keep this theme atmospheric but readable. Chinese characters may appear as decorative seals or archival marks, while operational UI labels remain in English and Vietnamese learning content remains untouched. Do not sacrifice contrast, keyboard access, responsive layout, page density, or the no-scroll physical-page rule for ornament.
- Feature the rhyming motto “Học chậm cho thấm, hiểu sâu nhớ lâu.” prominently in the sidebar rather than as small footer copy.
- The reader must still start with `python app.py` after dependencies are installed.

## Chapter authoring workflow

1. Confirm the requested chapter and its exact printed/PDF page range from the source.
2. Read the complete chapter, including summaries, exercises, captions, and footnotes.
3. Inventory original sections, figures, important terms, prerequisites, and exercises.
4. Change the chapter status to `draft` and author the Vietnamese notes using the required structure.
5. Extract required book figures and create only genuinely helpful supplemental diagrams.
6. Add glossary entries and validate every `{{term:...}}` key.
7. Run `python app.py --check` and `python -m unittest discover -s tests -v`.
8. Open the reader and visually check desktop and narrow layouts, equations, figures, page-turn navigation, and tooltips.
9. Re-read the chapter against the PDF. Only then set `status: reviewed`.

## Definition of done for a translated chapter

- Exact filename, title, section order, and page provenance match the PDF.
- A beginner can state the chapter's purpose, explain its core ideas, read its main equations, and solve representative exercises.
- Technical English is retained only when useful and every retained key term has a clear Vietnamese note.
- Original and supplemental figures are visibly distinguished and correctly sourced.
- Examples and solutions are checked, code is runnable where promised, and no placeholders remain.
- Automated validation passes and the rendered chapter is readable on desktop and mobile.
