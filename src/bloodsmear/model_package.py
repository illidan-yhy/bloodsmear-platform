from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from bloodsmear.errors import ModelHashMismatchError, ModelNotFoundError


class ModelManifest(BaseModel):
    name: str
    version: str
    model_file: str = "model.onnx"
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    input_size: int = Field(gt=0)
    input_dtype: str
    classes: list[str]
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    max_detections: int = Field(gt=0)
    license_status: str


@dataclass(frozen=True)
class ModelPackage:
    root: Path
    model_path: Path
    manifest: ModelManifest
    labels: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_model_package(path: Path) -> ModelPackage:
    root = Path(path)
    manifest_path = root / "manifest.yaml"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = ModelManifest.model_validate(yaml.safe_load(handle))

    model_path = root / manifest.model_file
    if not model_path.is_file():
        raise ModelNotFoundError(f"Model file not found: {model_path}")

    actual_hash = sha256_file(model_path)
    if actual_hash.lower() != manifest.sha256.lower():
        raise ModelHashMismatchError(
            f"Model hash mismatch: expected {manifest.sha256.upper()}, got {actual_hash}"
        )

    labels_path = root / "labels.json"
    with labels_path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)

    return ModelPackage(
        root=root,
        model_path=model_path,
        manifest=manifest,
        labels=labels,
    )

