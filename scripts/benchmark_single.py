from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from bloodsmear.benchmark import run_benchmark
from bloodsmear.config import AppSettings
from bloodsmear.inference import InferenceService


def gpu_info() -> dict[str, str | None]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        name, driver, memory = [part.strip() for part in output.split(",", 2)]
        return {"name": name, "driver": driver, "memory": memory}
    except Exception:
        return {"name": None, "driver": None, "memory": None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight single-image P50/P95 benchmark")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/single-inference.json"))
    args = parser.parse_args()

    settings = AppSettings(require_gpu=args.require_gpu)
    service = InferenceService.from_settings(settings)
    data = args.image.read_bytes()
    result = run_benchmark(service, data, args.image.name, warmup=args.warmup, iterations=args.iterations, require_gpu=args.require_gpu)
    result.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": {"path": str(args.image), "bytes": len(data)},
        "model": {"name": service.package.manifest.name, "version": service.package.manifest.version, "sha256": service.package.manifest.sha256},
        "settings": {"input_size": settings.input_size, "confidence_threshold": settings.confidence_threshold, "iou_threshold": settings.iou_threshold, "max_detections": settings.max_detections},
        "hardware": {"gpu": gpu_info(), "os": platform.platform(), "python": platform.python_version()},
        "target": {"p95_ms": 3000.0, "passed": result["stages"]["end_to_end"]["p95_ms"] <= 3000.0},
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "provider": result["provider"], "end_to_end": result["stages"]["end_to_end"], "target": result["target"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
