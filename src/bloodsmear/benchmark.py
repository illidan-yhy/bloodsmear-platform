from __future__ import annotations

from time import perf_counter_ns

import numpy as np

from bloodsmear.errors import GPUUnavailableError


def summarize(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one timing sample is required")
    return {
        "p50_ms": float(np.percentile(values, 50, method="linear")),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "mean_ms": float(np.mean(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def run_benchmark(
    service,
    image_data: bytes,
    filename: str,
    *,
    warmup: int,
    iterations: int,
    require_gpu: bool,
) -> dict:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    for _ in range(warmup):
        result = service.infer_bytes(image_data, filename, sample_id="benchmark")
        _verify_provider(result.runtime.provider, require_gpu)

    samples = {name: [] for name in ("preprocess", "inference", "postprocess", "pipeline_total", "end_to_end")}
    provider = None
    for _ in range(iterations):
        started = perf_counter_ns()
        result = service.infer_bytes(image_data, filename, sample_id="benchmark")
        end_to_end_ms = (perf_counter_ns() - started) / 1_000_000.0
        provider = result.runtime.provider
        _verify_provider(provider, require_gpu)
        samples["preprocess"].append(result.runtime.preprocess_ms)
        samples["inference"].append(result.runtime.inference_ms)
        samples["postprocess"].append(result.runtime.postprocess_ms)
        samples["pipeline_total"].append(result.runtime.total_ms)
        samples["end_to_end"].append(end_to_end_ms)

    return {
        "provider": provider,
        "warmup": warmup,
        "iterations": iterations,
        "stages": {name: summarize(values) for name, values in samples.items()},
        "samples_ms": samples,
    }


def _verify_provider(provider: str, require_gpu: bool) -> None:
    if require_gpu and provider != "CUDAExecutionProvider":
        raise GPUUnavailableError("CUDAExecutionProvider is required for this benchmark")

