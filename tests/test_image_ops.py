from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from bloodsmear.errors import InvalidImageError
from bloodsmear.image_ops import decode_image, letterbox


def image_bytes(mode: str, file_format: str) -> bytes:
    color = {
        "L": 127,
        "RGB": (255, 0, 0),
        "RGBA": (255, 0, 0, 128),
    }[mode]
    image = Image.new(mode, (20, 10), color=color)
    stream = BytesIO()
    image.save(stream, format=file_format)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("filename", "mode", "file_format"),
    [
        ("cell.jpg", "RGB", "JPEG"),
        ("cell.PNG", "RGBA", "PNG"),
        ("cell.tif", "L", "TIFF"),
    ],
)
def test_decode_image_accepts_supported_formats_and_normalizes_rgb(
    filename: str,
    mode: str,
    file_format: str,
) -> None:
    decoded = decode_image(image_bytes(mode, file_format), filename)

    assert decoded.rgb.shape == (10, 20, 3)
    assert decoded.rgb.dtype == np.uint8
    assert decoded.width == 20
    assert decoded.height == 10
    assert len(decoded.sha256) == 64


def test_decode_image_converts_rgba_without_swapping_red_and_blue() -> None:
    decoded = decode_image(image_bytes("RGBA", "PNG"), "cell.png")

    assert tuple(decoded.rgb[0, 0]) == (255, 127, 127)


def test_decode_image_rejects_unsupported_extension() -> None:
    with pytest.raises(InvalidImageError, match="Unsupported image extension"):
        decode_image(image_bytes("RGB", "PNG"), "cell.bmp")


def test_decode_image_rejects_invalid_bytes() -> None:
    with pytest.raises(InvalidImageError, match="decode"):
        decode_image(b"not-an-image", "cell.png")


def test_decode_image_rejects_payload_larger_than_limit() -> None:
    with pytest.raises(InvalidImageError, match="25 MiB"):
        decode_image(b"x" * (25 * 1024 * 1024 + 1), "cell.png")


def test_letterbox_returns_normalized_fp32_nchw_tensor() -> None:
    rgb = np.full((100, 200, 3), 255, dtype=np.uint8)

    tensor, transform = letterbox(rgb, size=640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert transform.scale == pytest.approx(3.2)
    assert transform.pad_x == pytest.approx(0.0)
    assert transform.pad_y == pytest.approx(160.0)


def test_letterbox_round_trips_box_within_one_pixel() -> None:
    rgb = np.zeros((101, 203, 3), dtype=np.uint8)
    original = (10.0, 20.0, 100.0, 80.0)

    _, transform = letterbox(rgb, size=640)
    restored = transform.to_original(transform.to_model(original))

    assert restored == pytest.approx(original, abs=1.0)
