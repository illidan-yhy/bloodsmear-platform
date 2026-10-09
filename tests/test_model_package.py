from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from bloodsmear.config import AppSettings
from bloodsmear.errors import ModelHashMismatchError, ModelNotFoundError
from bloodsmear.model_package import load_model_package


CLASSES = [
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelets",
    "RBC",
]


def make_package(
    root: Path,
    *,
    matching_hash: bool = True,
    include_model: bool = True,
) -> Path:
    package_dir = root / "model-package"
    package_dir.mkdir()
    model_bytes = b"test-onnx-model"
    if include_model:
        (package_dir / "model.onnx").write_bytes(model_bytes)
    expected_hash = hashlib.sha256(model_bytes).hexdigest()
    if not matching_hash:
        expected_hash = "0" * 64
    manifest = {
        "name": "blood-cell-yolo11",
        "version": "1.0.0",
        "model_file": "model.onnx",
        "sha256": expected_hash,
        "input_size": 640,
        "input_dtype": "float32",
        "classes": CLASSES,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.70,
        "max_detections": 3000,
        "license_status": "pending_publisher_confirmation",
    }
    (package_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (package_dir / "labels.json").write_text(
        json.dumps(CLASS_MAP, ensure_ascii=False), encoding="utf-8"
    )
    return package_dir


CLASS_MAP = {str(index): name for index, name in enumerate(CLASSES)}


def test_load_model_package_accepts_matching_sha256(tmp_path: Path) -> None:
    package = load_model_package(make_package(tmp_path, matching_hash=True))

    assert package.manifest.input_size == 640
    assert package.manifest.input_dtype == "float32"
    assert package.manifest.classes[-1] == "RBC"
    assert package.labels == CLASS_MAP


def test_load_model_package_rejects_hash_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ModelHashMismatchError):
        load_model_package(make_package(tmp_path, matching_hash=False))


def test_load_model_package_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        load_model_package(make_package(tmp_path, include_model=False))


def test_app_settings_use_mvp_defaults() -> None:
    settings = AppSettings()

    assert settings.model_dir == Path("models/blood-cell-yolo11/1.0.0")
    assert settings.output_dir == Path("outputs")
    assert settings.max_upload_bytes == 25 * 1024 * 1024
    assert settings.require_gpu is False
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.confidence_threshold == 0.25
    assert settings.iou_threshold == 0.70
    assert settings.max_detections == 3000
