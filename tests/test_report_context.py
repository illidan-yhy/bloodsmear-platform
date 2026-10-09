from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bloodsmear.domain import SampleMetadata
from bloodsmear.reporting.context import excel_safe_text, shanghai_time


def test_shanghai_time_converts_from_utc() -> None:
    converted = shanghai_time(datetime(2026, 10, 4, 0, 0, tzinfo=timezone.utc))
    assert converted.isoformat() == "2026-10-04T08:00:00+08:00"


@pytest.mark.parametrize("value", ["=1+1", "+SUM(A1:A2)", "-2+3", "@cmd"])
def test_excel_safe_text_neutralizes_formula_prefixes(value: str) -> None:
    assert excel_safe_text(value).startswith("'")


def test_excel_safe_text_preserves_normal_chinese() -> None:
    assert excel_safe_text("瑞氏染色") == "瑞氏染色"


def test_metadata_repr_remains_safe_when_nested() -> None:
    metadata = SampleMetadata(patient_id="P001", notes="private")
    assert "P001" not in repr(metadata)
    assert "private" not in repr(metadata)
