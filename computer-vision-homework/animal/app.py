"""Small browser UI for the trained animal classifier."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

from flask import Flask, render_template_string, request
from PIL import Image, ImageOps, UnidentifiedImageError

from inference_cnn import classify_image, load_classifier


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WildScope · Animal Classifier</title>
  <style>
    :root { color-scheme: dark; --ink:#edf7f2; --muted:#9eb4aa; --line:#254136; --leaf:#56e39f; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; font-family:Inter,ui-sans-serif,system-ui,sans-serif; color:var(--ink);
      background:radial-gradient(circle at 10% 10%,#18382b 0,transparent 34%),linear-gradient(140deg,#07110d,#0b1714 55%,#10171f); }
    main { width:min(1120px,calc(100% - 40px)); margin:0 auto; padding:42px 0 54px; }
    header { display:flex; align-items:end; justify-content:space-between; gap:24px; margin-bottom:26px; }
    .eyebrow { color:var(--leaf); text-transform:uppercase; letter-spacing:.18em; font-size:12px; font-weight:800; }
    h1 { margin:7px 0 0; font-size:clamp(34px,6vw,68px); letter-spacing:-.055em; line-height:.95; }
    .status { border:1px solid var(--line); border-radius:999px; padding:9px 13px; color:var(--muted); font-size:13px; white-space:nowrap; }
    .status b { color:var(--leaf); }
    .grid { display:grid; grid-template-columns:minmax(0,1.7fr) minmax(280px,.8fr); gap:20px; }
    .card { background:rgba(10,24,19,.84); border:1px solid var(--line); border-radius:24px; box-shadow:0 24px 70px #0007; overflow:hidden; }
    .stage { min-height:560px; display:grid; place-items:center; padding:28px; position:relative; }
    .stage img { max-width:100%; max-height:500px; border-radius:16px; box-shadow:0 16px 48px #0008; }
    .empty { text-align:center; max-width:430px; }
    .paw { width:76px; height:76px; display:grid; place-items:center; margin:0 auto 20px; border-radius:22px;
      background:#142d24; border:1px solid #315747; font-size:36px; }
    .empty h2 { margin:0 0 9px; font-size:24px; }
    .empty p, .hint { color:var(--muted); line-height:1.6; }
    aside { padding:25px; }
    aside h2 { margin:0 0 8px; font-size:20px; }
    form { margin:24px 0; }
    input[type=file] { width:100%; color:var(--muted); border:1px dashed #3c6755; border-radius:14px; padding:15px; background:#0b1c16; }
    button { width:100%; border:0; border-radius:14px; margin-top:12px; padding:14px 18px; font-weight:800; font-size:15px;
      color:#052016; background:linear-gradient(135deg,#70f2ae,#35c984); cursor:pointer; }
    .result { border-top:1px solid var(--line); padding-top:22px; }
    .label { margin:4px 0; font-size:32px; font-weight:850; text-transform:capitalize; }
    .confidence { color:var(--leaf); font-size:17px; font-weight:750; }
    .bar { height:7px; background:#152b23; border-radius:99px; overflow:hidden; margin:8px 0 18px; }
    .bar span { display:block; height:100%; background:var(--leaf); border-radius:99px; }
    .rank { display:flex; justify-content:space-between; gap:12px; color:var(--muted); padding:10px 0; border-bottom:1px solid #183126; }
    .facts { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:22px; }
    .fact { background:#0b1b16; border:1px solid #1d392e; border-radius:12px; padding:12px; }
    .fact span { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
    .fact b { display:block; margin-top:4px; }
    .error { color:#ffafaf; background:#321b1b; border:1px solid #693131; border-radius:12px; padding:11px; }
    @media (max-width:760px) { header{align-items:start;flex-direction:column}.grid{grid-template-columns:1fr}.stage{min-height:360px} }
  </style>
</head>
<body>
<main>
  <header>
    <div><div class="eyebrow">From-scratch CNN · 10 species</div><h1>WildScope</h1></div>
    <div class="status"><b>●</b> Model ready · {{ device }}</div>
  </header>
  <div class="grid">
    <section class="card stage">
      {% if image_data %}<img src="data:image/jpeg;base64,{{ image_data }}" alt="Uploaded animal">{% else %}
      <div class="empty"><div class="paw">◉</div><h2>Bring an animal into view</h2><p>Upload a JPG, PNG, or WebP image. The local CNN will rank the most likely species without sending the image anywhere.</p></div>{% endif %}
    </section>
    <aside class="card">
      <h2>Classify an image</h2><div class="hint">Choose a clear photo containing one main animal.</div>
      <form method="post" enctype="multipart/form-data"><input type="file" name="image" accept="image/jpeg,image/png,image/webp" required><button type="submit">Run classification →</button></form>
      {% if error %}<div class="error">{{ error }}</div>{% endif %}
      {% if predictions %}<div class="result"><div class="eyebrow">Top prediction</div><div class="label">{{ predictions[0][0] }}</div><div class="confidence">{{ '%.2f'|format(predictions[0][1] * 100) }}% confidence</div>
      <div class="bar"><span style="width:{{ predictions[0][1] * 100 }}%"></span></div>
      {% for label, score in predictions %}<div class="rank"><span>{{ loop.index }} · {{ label|title }}</span><b>{{ '%.2f'|format(score * 100) }}%</b></div>{% endfor %}</div>{% endif %}
      <div class="facts"><div class="fact"><span>Best val accuracy</span><b>{{ '%.2f'|format(best_accuracy * 100) }}%</b></div><div class="fact"><span>Best epoch</span><b>{{ best_epoch }}</b></div><div class="fact"><span>Input</span><b>{{ image_size }} × {{ image_size }}</b></div><div class="fact"><span>Classes</span><b>{{ class_count }}</b></div></div>
    </aside>
  </div>
</main>
</body></html>"""


def create_app(checkpoint_path: str | Path) -> Flask:
    application = Flask(__name__)
    model, categories, image_size, device, checkpoint = load_classifier(checkpoint_path)

    @application.route("/", methods=["GET", "POST"])
    def index():
        predictions = None
        image_data = None
        error = None
        if request.method == "POST":
            upload = request.files.get("image")
            if upload is None or not upload.filename:
                error = "Choose an image before running classification."
            else:
                try:
                    image = Image.open(upload.stream)
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    predictions = classify_image(image, model, categories, image_size, device)
                    preview = image.copy()
                    preview.thumbnail((1400, 1000))
                    buffer = io.BytesIO()
                    preview.save(buffer, format="JPEG", quality=90)
                    image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
                except (UnidentifiedImageError, OSError):
                    error = "The selected file is not a readable image."
        return render_template_string(
            PAGE,
            predictions=predictions,
            image_data=image_data,
            error=error,
            device=str(device).upper(),
            best_accuracy=float(checkpoint.get("best_accuracy", 0)),
            best_epoch=int(checkpoint.get("best_epoch", 0)),
            image_size=image_size,
            class_count=len(categories),
        )

    return application


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the animal-classification web UI")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    app = create_app(args.checkpoint)
    app.run(host=args.host, port=args.port, debug=False)
