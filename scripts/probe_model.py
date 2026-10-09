from __future__ import annotations

import argparse
import json
from pathlib import Path

from bloodsmear.adapters.yolo11_onnx import YOLO11OnnxAdapter
from bloodsmear.domain import InferenceOptions, build_summary
from bloodsmear.image_ops import decode_image
from bloodsmear.model_package import load_model_package


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the configured ONNX model")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/blood-cell-yolo11/1.0.0"),
    )
    arguments = parser.parse_args()

    package = load_model_package(arguments.model_dir)
    adapter = YOLO11OnnxAdapter(package)
    decoded = decode_image(arguments.image.read_bytes(), arguments.image.name)
    options = InferenceOptions(
        confidence_threshold=package.manifest.confidence_threshold,
        iou_threshold=package.manifest.iou_threshold,
        max_detections=package.manifest.max_detections,
    )
    adapter_result = adapter.infer(decoded.rgb, options)
    summary, summary_warnings = build_summary(
        adapter_result.detections,
        package.manifest.classes,
    )
    payload = {
        "provider": adapter_result.provider,
        "input_shape": adapter.input_shape,
        "output_names": adapter.output_names,
        "timings_ms": {
            "preprocess": adapter_result.timings.preprocess_ms,
            "inference": adapter_result.timings.inference_ms,
            "postprocess": adapter_result.timings.postprocess_ms,
        },
        "warnings": adapter_result.warnings + summary_warnings,
        "counts": summary.cell_counts,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
