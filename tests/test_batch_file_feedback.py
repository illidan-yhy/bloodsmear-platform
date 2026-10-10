"""Public upload rejection must identify a file, not blame every selected image."""
from fastapi.testclient import TestClient

from bloodsmear.api import create_app
from bloodsmear.config import AppSettings
from test_batch_storage import build_manager
from test_api import FakeService


def test_upload_rejection_identifies_second_unsupported_file(tmp_path):
    manager, repo, storage = build_manager(tmp_path)
    with TestClient(create_app(FakeService(), AppSettings(output_dir=tmp_path / "outputs"), manager)) as client:
        response = client.post("/ui/jobs", data={"mode": "independent"}, files=[
            ("files", ("normal.png", b"small", "image/png")),
            ("files", ("错误 图片.bmp", b"small", "image/bmp")),
        ])
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "INVALID_IMAGE"
    assert error["file"]["index"] == 2 and error["file"]["filename"] == "错误 图片.bmp"
    assert "格式" in error["file"]["reason"]
    assert not storage.staging_root.exists()
    with repo._connect() as connection:
        assert connection.execute("SELECT count(*) FROM batch_jobs").fetchone()[0] == 0


def test_duplicate_filename_size_rejection_has_exact_index_and_size_reason(tmp_path):
    manager, repo, storage = build_manager(tmp_path, max_file_bytes=5, max_total_bytes=50)
    with TestClient(create_app(FakeService(), AppSettings(output_dir=tmp_path / "outputs"), manager)) as client:
        response = client.post("/ui/jobs", data={"mode": "independent"}, files=[
            ("files", ("同名.png", b"123", "image/png")),
            ("files", ("同名.png", b"123456", "image/png")),
        ])
    assert response.status_code == 400
    file = response.json()["error"]["file"]
    assert file["index"] == 2 and file["filename"] == "同名.png"
    assert "大小" in file["reason"] and "损坏" not in file["reason"]
    assert not storage.staging_root.exists()
    with repo._connect() as connection:
        assert connection.execute("SELECT count(*) FROM batch_jobs").fetchone()[0] == 0
