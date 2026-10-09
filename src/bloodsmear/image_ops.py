from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from bloodsmear.errors import InvalidImageError


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class DecodedImage:
    rgb: NDArray[np.uint8]
    width: int
    height: int
    sha256: str
    filename: str


@dataclass(frozen=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_x: float
    pad_y: float

    def to_model(
        self, bbox_xyxy: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        return (
            x1 * self.scale + self.pad_x,
            y1 * self.scale + self.pad_y,
            x2 * self.scale + self.pad_x,
            y2 * self.scale + self.pad_y,
        )

    def to_original(
        self, bbox_xyxy: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        restored = (
            (x1 - self.pad_x) / self.scale,
            (y1 - self.pad_y) / self.scale,
            (x2 - self.pad_x) / self.scale,
            (y2 - self.pad_y) / self.scale,
        )
        return (
            min(max(restored[0], 0.0), float(self.original_width)),
            min(max(restored[1], 0.0), float(self.original_height)),
            min(max(restored[2], 0.0), float(self.original_width)),
            min(max(restored[3], 0.0), float(self.original_height)),
        )


def decode_image(
    data: bytes,
    filename: str,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> DecodedImage:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise InvalidImageError(f"Unsupported image extension: {extension or '<none>'}")
    if len(data) > max_bytes:
        raise InvalidImageError("Image exceeds the 25 MiB upload limit")

    try:
        with Image.open(BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image).convert("RGB")
            else:
                image = image.convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Unable to decode image data") from exc

    height, width = rgb.shape[:2]
    return DecodedImage(
        rgb=rgb,
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest().upper(),
        filename=Path(filename).name,
    )


def letterbox(
    rgb: NDArray[np.uint8],
    size: int,
) -> tuple[NDArray[np.float32], LetterboxTransform]:
    original_height, original_width = rgb.shape[:2]
    scale = min(size / original_width, size / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    horizontal_padding = size - resized_width
    vertical_padding = size - resized_height
    left = horizontal_padding // 2
    right = horizontal_padding - left
    top = vertical_padding // 2
    bottom = vertical_padding - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0

    return tensor, LetterboxTransform(
        original_width=original_width,
        original_height=original_height,
        input_size=size,
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
    )
