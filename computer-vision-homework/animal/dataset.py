"""Dataset utilities for the animal image-classification project."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class AnimalDataset(Dataset):
    """Load ``train/<class>`` or ``test/<class>`` image folders.

    ``max_samples_per_class`` is useful for a quick, balanced demonstration run.
    A value of zero keeps every available image.
    """

    def __init__(
        self,
        data_path: str | Path,
        is_train: bool = True,
        transform=None,
        max_samples_per_class: int = 0,
        seed: int = 42,
    ) -> None:
        split = "train" if is_train else "test"
        self.root = Path(data_path).expanduser().resolve() / split
        self.transform = transform

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Expected the {split!r} split at {self.root}. "
                "Pass --data-path with the directory containing train/ and test/."
            )

        self.categories = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not self.categories:
            raise ValueError(f"No class directories were found in {self.root}")

        rng = random.Random(seed)
        self.samples: list[tuple[Path, int]] = []
        for label, category in enumerate(self.categories):
            images = sorted(
                path
                for path in (self.root / category).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if max_samples_per_class > 0 and len(images) > max_samples_per_class:
                rng.shuffle(images)
                images = sorted(images[:max_samples_per_class])
            self.samples.extend((path, label) for path in images)

        if not self.samples:
            raise ValueError(f"No supported images were found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
