"""UI integration fixtures: replace only model execution, keep decoding/API/reporting real."""
from io import BytesIO
from pathlib import Path

from PIL import Image

from bloodsmear.adapters.yolo11_onnx import AdapterResult, AdapterTimings
from bloodsmear.api import create_app
from bloodsmear.batch.manager import BatchManager
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage
from bloodsmear.config import AppSettings
from bloodsmear.domain import Detection
from bloodsmear.inference import InferenceService
from bloodsmear.model_package import ModelManifest, ModelPackage


CLASSES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil", "Platelets", "RBC"]


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (20, 10), color="white").save(stream, format="PNG")
    return stream.getvalue()


class FixedAdapter:
    provider = "CUDAExecutionProvider"

    def infer(self, rgb, options):
        return AdapterResult(
            detections=[Detection(class_id=6, class_name="RBC", confidence=0.9, bbox_xyxy=(1, 1, 5, 5))],
            warnings=[], provider=self.provider,
            timings=AdapterTimings(preprocess_ms=1, inference_ms=1, postprocess_ms=1),
        )


def make_app(root: Path):
    settings = AppSettings(
        output_dir=root / "outputs", database_path=root / "data" / "jobs.db",
        jobs_root=root / "data" / "jobs",
    )
    manifest = ModelManifest(
        name="test-model", version="1.0.0", sha256="A" * 64, input_size=640,
        input_dtype="float32", classes=CLASSES, confidence_threshold=0.25,
        iou_threshold=0.7, max_detections=3000, license_status="pending_publisher_confirmation",
    )
    package = ModelPackage(root=root, model_path=root / "unused.onnx", manifest=manifest, labels={})
    service = InferenceService(package=package, adapter=FixedAdapter(), settings=settings)
    repository = BatchRepository(settings.database_path)
    repository.initialize()
    manager = BatchManager(repository, BatchStorage(settings), settings)
    return create_app(service, settings, manager)
