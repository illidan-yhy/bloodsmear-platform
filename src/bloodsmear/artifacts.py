from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw

from bloodsmear.domain import InferenceResult
from bloodsmear.errors import InvalidImageError
from bloodsmear.image_ops import decode_image


CSV_FIELDS = [
    "sample_id",
    "model_name",
    "model_version",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]

CLASS_COLORS = {
    "Basophil": (142, 68, 173),
    "Eosinophil": (230, 126, 34),
    "Lymphocyte": (52, 152, 219),
    "Monocyte": (22, 160, 133),
    "Neutrophil": (39, 174, 96),
    "Platelets": (241, 196, 15),
    "RBC": (231, 76, 60),
}


def _atomic_write(destination: Path, writer: Callable[[Path], None]) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def write_result_json(result: InferenceResult, destination: Path) -> Path:
    def writer(path: Path) -> None:
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    return _atomic_write(destination, writer)


def write_detections_csv(result: InferenceResult, destination: Path) -> Path:
    def writer(path: Path) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            for detection in result.detections:
                x1, y1, x2, y2 = detection.bbox_xyxy
                csv_writer.writerow(
                    {
                        "sample_id": result.sample_id,
                        "model_name": result.model.name,
                        "model_version": result.model.version,
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "confidence": f"{detection.confidence:.6f}",
                        "x1": f"{x1:.3f}",
                        "y1": f"{y1:.3f}",
                        "x2": f"{x2:.3f}",
                        "y2": f"{y2:.3f}",
                    }
                )

    return _atomic_write(destination, writer)


def write_annotated_image(
    result: InferenceResult,
    image_data: bytes,
    destination: Path,
) -> Path:
    decoded = decode_image(image_data, result.image.filename)
    if (decoded.width, decoded.height) != (result.image.width, result.image.height):
        raise InvalidImageError("Image dimensions do not match inference result")

    image = Image.fromarray(decoded.rgb, mode="RGB")
    drawing = ImageDraw.Draw(image)
    line_width = max(1, min(decoded.width, decoded.height) // 200)
    for detection in result.detections:
        color = CLASS_COLORS[detection.class_name]
        box = tuple(round(value) for value in detection.bbox_xyxy)
        drawing.rectangle(box, outline=color, width=line_width)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_box = drawing.textbbox((box[0], box[1]), label)
        drawing.rectangle(text_box, fill=color)
        drawing.text((box[0], box[1]), label, fill=(255, 255, 255))

    def writer(path: Path) -> None:
        image.save(path, format="PNG")

    return _atomic_write(destination, writer)
