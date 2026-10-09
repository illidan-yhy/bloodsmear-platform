from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bloodsmear.domain import InferenceResult
from bloodsmear.errors import ReportGenerationError
from bloodsmear.reporting.context import ReportContextFactory
from bloodsmear.reporting.context import BatchReportContext
from bloodsmear.reporting.excel import ExcelReportRenderer
from bloodsmear.reporting.pdf import PdfReportRenderer


@dataclass(frozen=True)
class ReportPaths:
    excel: Path
    pdf: Path


class ReportService:
    def __init__(self, excel=None, pdf=None) -> None:
        self.excel = excel or ExcelReportRenderer()
        self.pdf = pdf or PdfReportRenderer()

    def generate_single(self, result: InferenceResult, license_status: str, output_dir: Path) -> ReportPaths:
        try:
            context = ReportContextFactory.single(result, license_status)
            return ReportPaths(
                excel=self.excel.render_single(context, output_dir / "report.xlsx"),
                pdf=self.pdf.render_single(context, output_dir / "report.pdf"),
            )
        except Exception as exc:
            raise ReportGenerationError("Report generation failed") from exc

    def generate_batch(self, context: BatchReportContext, output_dir: Path) -> ReportPaths:
        try:
            return ReportPaths(
                excel=self.excel.render_batch(context, output_dir / "report.xlsx"),
                pdf=self.pdf.render_batch(context, output_dir / "report.pdf"),
            )
        except Exception as exc:
            raise ReportGenerationError("Report generation failed") from exc
