"""Run animal classification from a saved AdvancedCNN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms import Compose, Resize, ToTensor

from models import AdvancedCNN


def load_classifier(checkpoint_path: str | Path, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    categories = list(checkpoint["categories"])
    image_size = int(checkpoint.get("image_size", 224))
    model = AdvancedCNN(num_classes=len(categories)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, categories, image_size, device, checkpoint


def classify_image(
    image: Image.Image,
    model: AdvancedCNN,
    categories: list[str],
    image_size: int,
    device: torch.device,
) -> list[tuple[str, float]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    transform = Compose([Resize((image_size, image_size)), ToTensor()])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = probabilities.topk(min(3, len(categories)))
    return [(categories[index], float(value)) for value, index in zip(values.cpu(), indices.cpu())]


def render_prediction(image: Image.Image, predictions: list[tuple[str, float]]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    preview = image.copy()
    preview.thumbnail((920, 620))
    canvas = Image.new("RGB", (preview.width, preview.height + 150), "#07111f")
    canvas.paste(preview, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    small_font = ImageFont.load_default(size=15)
    label, confidence = predictions[0]
    draw.text((24, preview.height + 20), label.upper(), fill="#56e39f", font=font)
    draw.text((24, preview.height + 58), f"Confidence: {confidence:.2%}", fill="white", font=small_font)
    top_three = "  ·  ".join(f"{name}: {score:.1%}" for name, score in predictions)
    draw.text((24, preview.height + 92), top_three, fill="#a7b6c8", font=small_font)
    return canvas


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one animal image")
    parser.add_argument("-p", "--image-path", required=True)
    parser.add_argument("-c", "--checkpoint", default="best.pt")
    parser.add_argument("-o", "--output", default="output-assets/inference-result.png")
    parser.add_argument("--display", action="store_true", help="Open the rendered result in an OpenCV window")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    model, categories, image_size, device, _ = load_classifier(args.checkpoint)
    with Image.open(args.image_path) as source:
        image = source.convert("RGB")
    predictions = classify_image(image, model, categories, image_size, device)
    rendered = render_prediction(image, predictions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)
    print(f"Prediction: {predictions[0][0]} ({predictions[0][1]:.2%})")
    print("Top 3: " + ", ".join(f"{name}={score:.2%}" for name, score in predictions))
    print(f"Rendered result: {output_path.resolve()}")
    if args.display:
        frame = cv2.cvtColor(np.asarray(rendered), cv2.COLOR_RGB2BGR)
        cv2.imshow("Animal classification", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
