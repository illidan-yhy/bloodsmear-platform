from __future__ import annotations

import os
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from bloodsmear.batch.domain import BatchMode
from bloodsmear.reporting.context import BatchReportContext, SingleReportContext, shanghai_time


FONT_NAME = "NotoSansSC"
FONT_PATH = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NotoSansSC-Regular.ttf"


class PdfReportRenderer:
    def __init__(self) -> None:
        if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))

    def render_single(self, context: SingleReportContext, destination: Path) -> Path:
        result = context.result
        story = [self._title("外周血涂片单图推理报告")]
        rows = [["Sample ID", result.sample_id], ["图像", result.image.filename]]
        if result.metadata:
            labels = {"patient_id": "Patient ID", "stain_method": "染色方式", "scanner_model": "扫描仪", "magnification": "倍率", "pixel_resolution": "像素分辨率", "operator": "操作员", "notes": "备注"}
            rows.extend([labels[key], value] for key, value in result.metadata.model_dump().items() if value)
        rows.extend([["模型", f"{result.model.name} {result.model.version}"], ["模型 SHA-256", result.model.sha256], ["许可证状态", context.license_status]])
        story.extend([self._table(rows, [42 * mm, 130 * mm]), Spacer(1, 5 * mm)])
        summary_rows = [["细胞类型", "数量", "全细胞比例", "WBC比例"]]
        for name, count in result.summary.cell_counts.items():
            summary_rows.append([name, count, _percent(result.summary.all_cell_ratios.get(name)), _percent(result.summary.wbc_differential_ratios.get(name))])
        story.extend([self._table(summary_rows, [45 * mm] * 4), Spacer(1, 5 * mm)])
        story.append(self._table([["QC", result.qc.status], ["Warnings", ", ".join(result.qc.warnings) or "无"], ["Provider", result.runtime.provider], ["总耗时", f"{result.runtime.total_ms:.2f} ms"], ["生成时间", shanghai_time(context.generated_at).isoformat()]], [42 * mm, 130 * mm]))
        story.extend([Spacer(1, 8 * mm), self._paragraph("科研与内部分析，不用于临床诊断")])
        return self._build(story, Path(destination), A4)

    def render_batch(self, context: BatchReportContext, destination: Path) -> Path:
        title = "独立样本批量报告" if context.job.mode == BatchMode.INDEPENDENT else "多视野样本汇总报告"
        header_rows = [["Job ID", context.job.id], ["模式", context.job.mode.value], ["样本", context.job.sample_id or "—"], ["成功/失败", f"{context.summary.successful_views}/{context.summary.failed_views}"]]
        if context.job.metadata:
            labels = {"patient_id": "Patient ID", "stain_method": "染色方式", "scanner_model": "扫描仪", "magnification": "倍率", "pixel_resolution": "像素分辨率", "operator": "操作员", "notes": "备注"}
            for key, value in context.job.metadata.model_dump().items():
                if value is None or (key == "patient_id" and context.job.mode == BatchMode.INDEPENDENT):
                    continue
                header_rows.append([labels[key], value])
        story = [self._title(title), self._table(header_rows, [42 * mm, 210 * mm]), Spacer(1, 5 * mm)]
        rows = [["Sample ID", "文件", "总细胞", "QC", "耗时(ms)"]]
        for result in context.results:
            rows.append([result.sample_id, result.image.filename, result.summary.total_detected_cells, result.qc.status, f"{result.runtime.total_ms:.2f}"])
        story.append(self._table(rows, [45 * mm, 80 * mm, 35 * mm, 30 * mm, 35 * mm]))
        if context.summary.failures:
            story.extend([Spacer(1, 5 * mm), self._paragraph("失败清单"), self._table([["文件", "错误码"]] + [[item["original_filename"], item["error_code"]] for item in context.summary.failures], [100 * mm, 120 * mm])])
        story.extend([Spacer(1, 8 * mm), self._paragraph("科研与内部分析，不用于临床诊断")])
        return self._build(story, Path(destination), landscape(A4))

    def _build(self, story, destination: Path, pagesize) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".pdf", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            document = SimpleDocTemplate(str(temporary), pagesize=pagesize, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
            document.build(story, onFirstPage=self._page, onLaterPages=self._page)
            if not temporary.read_bytes().startswith(b"%PDF"):
                raise ValueError("Invalid PDF output")
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def _page(self, canvas, document) -> None:
        canvas.saveState(); canvas.setFont(FONT_NAME, 8); canvas.drawRightString(document.pagesize[0] - 15 * mm, 8 * mm, f"第 {document.page} 页"); canvas.restoreState()

    def _title(self, text: str): return Paragraph(escape(text), ParagraphStyle("TitleCN", parent=getSampleStyleSheet()["Title"], fontName=FONT_NAME, fontSize=18, leading=24, spaceAfter=12))
    def _paragraph(self, text: str): return Paragraph(escape(text), ParagraphStyle("BodyCN", fontName=FONT_NAME, fontSize=9, leading=13))
    def _table(self, rows, widths):
        converted = [[self._paragraph("—" if value is None else str(value)) for value in row] for row in rows]
        table = LongTable(converted, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), FONT_NAME), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEDEA")), ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
        return table


def _percent(value): return "—" if value is None else f"{value * 100:.2f}%"
