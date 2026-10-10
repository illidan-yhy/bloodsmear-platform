"""HTTP and real-browser regressions for submission feedback and result identity."""
import base64
import html
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient
from PIL import Image

from ui_feedback_helpers import make_app, png_bytes
from tiff_helpers import grayscale16_tiff


@pytest.mark.parametrize("filename", ["血涂片 样本1.png", "<img src=x onerror=alert(1)>.png"])
def test_single_result_displays_uploaded_filename_as_text(tmp_path, filename):
    with TestClient(make_app(tmp_path)) as client:
        response = client.post(
            "/ui/infer", headers={"HX-Request": "true"},
            files={"image": (filename, png_bytes(), "image/png")},
        )
    assert response.status_code == 200
    assert html.escape(filename) in response.text
    if "<" in filename:
        assert f"<strong>{filename}</strong>" not in response.text


def test_unsafe_pixel_dimensions_return_explicit_400_instead_of_bare_500(tmp_path, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)
    with TestClient(make_app(tmp_path), raise_server_exceptions=False) as client:
        response = client.post(
            "/ui/infer", headers={"HX-Request": "true"},
            files={"image": ("large.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"
    assert "pixel dimensions" in response.json()["error"]["message"]


def test_submission_feedback_in_real_browser(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Browser regression requires Node.js and Playwright")
    probe = subprocess.run(
        [node, "-e", "require('playwright')"], capture_output=True, timeout=10,
    )
    if probe.returncode:
        pytest.skip("Playwright is unavailable; install it or configure NODE_PATH to run browser regressions")
    app = make_app(tmp_path)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started, "Browser test server did not start"
        environment = os.environ.copy()
        environment["BLOODSMEAR_TEST_URL"] = f"http://127.0.0.1:{port}"
        environment["BLOODSMEAR_TEST_PNG_BASE64"] = base64.b64encode(png_bytes()).decode("ascii")
        environment["BLOODSMEAR_TEST_GRAY16_BASE64"] = base64.b64encode(grayscale16_tiff()).decode("ascii")
        environment["BLOODSMEAR_TEST_ARTIFACTS"] = str(tmp_path)
        completed = subprocess.run(
            [node, "--test", str(Path(__file__).with_name("ui_feedback_browser.cjs"))],
            capture_output=True, timeout=150, env=environment,
        )
        assert completed.returncode == 0, (completed.stdout + completed.stderr).decode("utf-8", errors="replace")
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
