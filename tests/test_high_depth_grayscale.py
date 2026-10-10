"""High-depth input must fail before RGB conversion, never produce a success/report."""
from io import BytesIO
import json
import logging
import zipfile

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage
from bloodsmear.batch.worker import BatchWorker
from bloodsmear.errors import InvalidImageError
from bloodsmear.image_ops import decode_image
from tiff_helpers import grayscale16_tiff, color16_tiff
from ui_feedback_helpers import make_app, png_bytes, CLASSES


@pytest.mark.parametrize("options", [
    {}, {"byteorder": "big"}, {"signed": True}, {"byteorder": "big", "signed": True},
    {"low_values": True}, {"orientation": 6}, {"photometric": 0},
])
def test_rejects_16bit_grayscale_by_original_format_not_pixel_brightness(options):
    data = grayscale16_tiff(**options)
    with Image.open(BytesIO(data)) as image:
        assert image.tag_v2[258] == (16,) and image.tag_v2[277] == 1
    with pytest.raises(InvalidImageError, match="16.*灰度") as error:
        decode_image(data, "科研相机.tiff")
    assert error.value.code == "UNSUPPORTED_IMAGE_MODE"


@pytest.mark.parametrize("mode,color", [("L", 128), ("RGB", (10, 128, 240))])
def test_8bit_tiff_still_preserves_pixel_values(mode, color):
    stream = BytesIO()
    Image.new(mode, (4, 2), color).save(stream, format="TIFF")
    decoded = decode_image(stream.getvalue(), "supported.tif")
    assert decoded.rgb.dtype == np.uint8
    assert tuple(decoded.rgb[0, 0]) == ((128, 128, 128) if mode == "L" else color)


@pytest.mark.parametrize("endpoint", ["/ui/infer", "/api/v1/infer"])
def test_single_rejection_returns_400_and_creates_no_zero_count_artifacts(tmp_path, endpoint):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(endpoint, files={"image": ("相机16位.tif", grayscale16_tiff(), "image/tiff")})
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "UNSUPPORTED_IMAGE_MODE"
    assert "16" in payload["error"]["message"] and "灰度" in payload["error"]["message"]
    assert "summary" not in payload and "artifacts" not in payload
    assert not list(app.state.settings.output_dir.iterdir())


def test_true_16bit_color_tiff_keeps_existing_8bit_rgb_reading():
    data = color16_tiff()
    with Image.open(BytesIO(data)) as opened:
        assert opened.tag_v2[258] == (16, 16, 16) and opened.tag_v2[277] == 3
    decoded = decode_image(data, "color16.tiff")
    assert tuple(decoded.rgb[0, 0]) == (10, 128, 240)


def test_batch_rejects_camera_file_logs_reason_and_finishes_other_images(tmp_path, caplog):
    app = make_app(tmp_path)
    settings = app.state.settings
    repo = BatchRepository(settings.database_path)
    storage = BatchStorage(settings)
    with TestClient(app) as client:
        submitted = client.post("/api/v1/jobs", data={"mode": "independent"}, files=[
            ("files", ("相机16位.tif", grayscale16_tiff(), "image/tiff")),
            ("files", ("正常.png", png_bytes(), "image/png")),
        ])
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        with caplog.at_level(logging.ERROR):
            assert BatchWorker(repo, storage, app.state.service, CLASSES).run_once()
        status = client.get(f"/api/v1/jobs/{job_id}").json()
        page = client.get(f"/ui/jobs/{job_id}/status").text
    assert status["status"] == "partial_failed" and status["failed_items"] == 1 and status["completed_items"] == 1
    bad, good = status["items"]
    assert bad["error_code"] == "UNSUPPORTED_IMAGE_MODE" and "16" in bad["error_message"] and "灰度" in bad["error_message"]
    assert bad["result_directory"] is None and good["status"] == "completed"
    assert "相机16位.tif" in page and "灰度" in page and "<td>1</td>" in page
    assert "UNSUPPORTED_IMAGE_MODE" not in page
    assert any(record.exc_info for record in caplog.records) and "UNSUPPORTED_IMAGE_MODE" in caplog.text
    with zipfile.ZipFile(storage.job_root(job_id) / "results.zip") as archive:
        exported = json.loads(archive.read("summary.json"))
        bad_index, good_index = exported["items"]
        assert bad_index["result_directory"] is None
        assert good_index["result_directory"] == "items/002_正常.png"
        assert not any(name.startswith("items/001_") for name in archive.namelist())
        result = json.loads(archive.read(f"{good_index['result_directory']}/result.json"))
        assert result["summary"]["total_detected_cells"] == 1
