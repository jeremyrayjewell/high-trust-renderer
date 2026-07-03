from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def render_contact_sheet(input_dir: Path, output_path: Path, columns: int = 4, padding: int = 16) -> None:
    image_paths = sorted(
        [path for path in input_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not image_paths:
        raise RuntimeError(f"No debug frames found in {input_dir}")

    images = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append((path, image))
    if not images:
        raise RuntimeError(f"Could not load debug frames from {input_dir}")

    columns = max(1, columns)
    padding = max(0, padding)
    tile_h = max(image.shape[0] for _, image in images)
    tile_w = max(image.shape[1] for _, image in images)
    rows = math.ceil(len(images) / columns)
    label_h = 24

    sheet_h = rows * (tile_h + label_h) + padding * (rows + 1)
    sheet_w = columns * tile_w + padding * (columns + 1)
    sheet = np.full((sheet_h, sheet_w, 3), 18, dtype=np.uint8)

    for index, (path, image) in enumerate(images):
        row = index // columns
        col = index % columns
        x = padding + col * (tile_w + padding)
        y = padding + row * (tile_h + label_h + padding)
        if image.shape[0] != tile_h or image.shape[1] != tile_w:
            image = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        sheet[y:y + tile_h, x:x + tile_w] = image
        label = path.stem[:48]
        cv2.putText(sheet, label, (x, y + tile_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 235), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Failed to write contact sheet to {output_path}")
