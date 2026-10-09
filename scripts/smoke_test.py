from __future__ import annotations

import argparse
import json
from pathlib import Path

from bloodsmear.artifacts import (
    write_annotated_image,
    write_detections_csv,
    write_result_json,
)
from bloodsmear.config import AppSettings
from bloodsmear.inference import InferenceService


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one end-to-end inference")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/blood-cell-yolo11/1.0.0"),
    )
    arguments = parser.parse_args()

    settings = AppSettings(
        model_dir=arguments.model_dir,
        output_dir=arguments.output,
    )
    service = InferenceService.from_settings(settings)
    image_data = arguments.image.read_bytes()
    result = service.infer_bytes(
        image_data,
        arguments.image.name,
        sample_id=arguments.image.stem,
    )

    arguments.output.mkdir(parents=True, exist_ok=True)
    json_path = write_result_json(result, arguments.output / "result.json")
    csv_path = write_detections_csv(result, arguments.output / "detections.csv")
    image_path = write_annotated_image(
        result,
        image_data,
        arguments.output / "annotated.png",
    )
    print(
        json.dumps(
            {
                "provider": result.runtime.provider,
                "total_ms": result.runtime.total_ms,
                "counts": result.summary.cell_counts,
                "warnings": result.qc.warnings,
                "artifacts": [str(json_path), str(csv_path), str(image_path)],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
