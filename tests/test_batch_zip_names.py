"""Exported folders identify images; internal storage paths must remain unchanged."""
import json
from pathlib import PurePosixPath
import zipfile

import pytest

from bloodsmear.batch.artifacts import write_batch_artifacts
from bloodsmear.batch.domain import BatchMode, ItemStatus, JobStatus
from test_batch_aggregation import item, job, CLASSES


FILES = ("result.json", "detections.csv", "annotated.png", "report.xlsx", "report.pdf")


def package(tmp_path, names, failed=()):
    items = []
    for ordinal, filename in enumerate(names):
        entry = item(ordinal, ItemStatus.FAILED if ordinal in failed else ItemStatus.COMPLETED)
        entry = entry.model_copy(update={
            "id": f"opaque-{ordinal}", "original_filename": filename,
            "result_directory": None if ordinal in failed else f"items/opaque-{ordinal}",
        })
        items.append(entry)
        if entry.result_directory:
            folder = tmp_path / entry.result_directory
            folder.mkdir(parents=True)
            for name in FILES:
                (folder / name).write_bytes(f"{entry.id}/{name}".encode())
    subject = job(BatchMode.INDEPENDENT, JobStatus.PARTIAL_FAILED if failed else JobStatus.COMPLETED, len(names))
    paths = write_batch_artifacts(subject, list(reversed(items)), tmp_path, CLASSES)
    return items, paths


def test_zip_folders_follow_upload_order_and_distinguish_duplicate_chinese_names(tmp_path):
    items, paths = package(tmp_path, ["血涂片 样本1.png", "坏图.png", "血涂片 样本1.png"], failed=(1,))
    with zipfile.ZipFile(paths.zip_path) as archive:
        names = archive.namelist()
        folders = {str(PurePosixPath(name).parent) for name in names if name.startswith("items/")}
        assert folders == {"items/001_血涂片 样本1.png", "items/003_血涂片 样本1.png"}
        assert len(names) == len(set(names))
        for prefix, source in [("items/001_血涂片 样本1.png", "opaque-0"), ("items/003_血涂片 样本1.png", "opaque-2")]:
            for name in FILES:
                assert archive.read(f"{prefix}/{name}") == f"{source}/{name}".encode()
        archive.extractall(tmp_path / "extracted")
    assert (tmp_path / "extracted/items/001_血涂片 样本1.png/report.pdf").is_file()
    assert items[0].result_directory == "items/opaque-0"


def test_zip_summary_points_to_exported_files_but_server_summary_keeps_internal_paths(tmp_path):
    items, paths = package(tmp_path, ["样本.png"])
    with zipfile.ZipFile(paths.zip_path) as archive:
        exported = json.loads(archive.read("summary.json"))
        entry = exported["items"][0]
        assert entry["result_directory"] == "items/001_样本.png"
        assert archive.read(f"{entry['result_directory']}/result.json") == b"opaque-0/result.json"
        assert entry["item_id"] == "opaque-0" and entry["original_filename"] == "样本.png"
    server = json.loads(paths.summary_path.read_text("utf-8"))
    assert server["items"][0]["result_directory"] == "items/opaque-0"
    assert (tmp_path / items[0].result_directory / "report.pdf").is_file()
    assert not (tmp_path / "items/001_样本.png").exists()


@pytest.mark.parametrize("original,expected", [
    ('a<>"b:c/d\\e|f?g*.png', "001_a___b_c_d_e_f_g_.png"),
    ("..\\..\\危险/样本.png", "001_.._.._危险_样本.png"),
    ("\x00报\x1f告?.png", "001__报_告_.png"),
    ("CON.png", "001_CON.png"),
    ("样本.png . ", "001_样本.png"),
])
def test_zip_name_is_safe_to_extract_on_windows(original, expected, tmp_path):
    _, paths = package(tmp_path, [original])
    with zipfile.ZipFile(paths.zip_path) as archive:
        name = f"items/{expected}/report.pdf"
        assert name in archive.namelist()
        assert all(".." not in PurePosixPath(member).parts for member in archive.namelist())
        archive.extractall(tmp_path / "extracted")
    assert (tmp_path / "extracted" / name).read_bytes() == b"opaque-0/report.pdf"


@pytest.mark.parametrize("filename", ["长" * 240 + ".tiff", "长" * 100 + ".png", "🔬" * 100 + ".tif"], ids=["chinese240", "chinese100", "emoji100"])
def test_long_names_are_shortened_with_extension_and_distinct_sequence_prefix(tmp_path, filename):
    _, paths = package(tmp_path, [filename, filename])
    with zipfile.ZipFile(paths.zip_path) as archive:
        directories = {PurePosixPath(name).parent.name for name in archive.namelist() if name.startswith("items/")}
        assert len(directories) == 2
        assert {name[:4] for name in directories} == {"001_", "002_"}
        assert all(len(name) <= 124 and name.endswith(PurePosixPath(filename).suffix) for name in directories)
        assert all(len(name.encode("utf-8")) <= 244 for name in directories)
        exported = json.loads(archive.read("summary.json"))
        assert all(entry["original_filename"] == filename for entry in exported["items"])
        assert all(f"{entry['result_directory']}/report.pdf" in archive.namelist() for entry in exported["items"])
        archive.extractall(tmp_path / "extracted")
