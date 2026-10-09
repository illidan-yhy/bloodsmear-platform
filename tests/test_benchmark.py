from __future__ import annotations

from types import SimpleNamespace

import pytest

from bloodsmear.benchmark import run_benchmark, summarize


def test_summarize_calculates_linear_p50_and_p95() -> None:
    summary = summarize([1.0, 2.0, 3.0, 4.0])
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 4.0


def test_run_benchmark_excludes_warmup_and_calls_expected_count() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.calls = 0

        def infer_bytes(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                runtime=SimpleNamespace(
                    preprocess_ms=1.0,
                    inference_ms=2.0,
                    postprocess_ms=3.0,
                    total_ms=6.0,
                    provider="CUDAExecutionProvider",
                )
            )

    service = FakeService()
    result = run_benchmark(
        service,
        b"image",
        "sample.png",
        warmup=2,
        iterations=3,
        require_gpu=True,
    )

    assert service.calls == 5
    assert result["warmup"] == 2
    assert result["iterations"] == 3
    assert result["stages"]["inference"]["p50_ms"] == 2.0
    assert len(result["samples_ms"]["end_to_end"]) == 3
