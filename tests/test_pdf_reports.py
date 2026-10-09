from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from bloodsmear.domain import ImageInfo, InferenceResult, ModelInfo, QCInfo, ResultSummary, SampleMetadata, TimingInfo
from bloodsmear.reporting.context import ReportContextFactory
from bloodsmear.reporting.pdf import PdfReportRenderer


def result() -> InferenceResult:
    return InferenceResult(
        sample_id="S001", metadata=SampleMetadata(patient_id="P001", stain_method="瑞氏染色"),
        created_at=datetime(2026, 10, 4, tzinfo=timezone.utc),
        model=ModelInfo(name="model", version="1", sha256="A" * 64),
        image=ImageInfo(filename="sample.png", width=100, height=80, sha256="B" * 64),
        detections=[],
        summary=ResultSummary(total_detected_cells=1, cell_counts={"RBC": 1}, all_cell_ratios={"RBC": 1.0}, wbc_differential_ratios={"Neutrophil": None}),
        qc=QCInfo(status="pass", warnings=[]),
        runtime=TimingInfo(provider="CUDAExecutionProvider", preprocess_ms=1, inference_ms=2, postprocess_ms=1, total_ms=4),
    )


def test_single_pdf_has_extractable_chinese_and_no_images(tmp_path: Path) -> None:
    destination = tmp_path / "report.pdf"
    PdfReportRenderer().render_single(ReportContextFactory.single(result(), "pending"), destination)
    reader = PdfReader(destination)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "外周血涂片" in text
    assert "瑞氏染色" in text
    assert "科研与内部分析，不用于临床诊断" in text
    for page in reader.pages:
        xobjects = page["/Resources"].get("/XObject", {})
        assert all(obj.get_object().get("/Subtype") != "/Image" for obj in xobjects.values())


def test_pdf_replaces_atomically_and_leaves_no_temp_files(tmp_path: Path) -> None:
    destination = tmp_path / "report.pdf"
    destination.write_bytes(b"old")
    PdfReportRenderer().render_single(ReportContextFactory.single(result(), "pending"), destination)
    assert destination.read_bytes().startswith(b"%PDF")
    assert list(tmp_path.glob("*.tmp")) == []
