from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from bloodsmear.api import create_app
from bloodsmear.artifacts import (
    write_annotated_image,
    write_detections_csv,
    write_result_json,
)
from bloodsmear.config import AppSettings
from bloodsmear.errors import BloodSmearError
from bloodsmear.inference import InferenceService
from bloodsmear.logging_config import configure_logging
from bloodsmear.batch.cleanup import BatchCleanup
from bloodsmear.batch.manager import BatchManager
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage
from bloodsmear.batch.worker import BatchWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bloodsmear")
    parser.add_argument("--debug", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    infer_parser = commands.add_parser("infer", help="Infer one image")
    infer_parser.add_argument("input", type=Path)
    infer_parser.add_argument("--output", type=Path, default=Path("outputs/cli"))
    infer_parser.add_argument("--sample-id")

    commands.add_parser("model-info", help="Show active model information")
    commands.add_parser("serve", help="Start the local HTTP service")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configure_logging(AppSettings())
    if arguments.command == "infer" and not arguments.input.is_file():
        print(f"Input file does not exist: {arguments.input}", file=sys.stderr)
        return 2

    try:
        if arguments.command == "infer":
            return _infer(arguments)
        if arguments.command == "model-info":
            return _model_info()
        if arguments.command == "serve":
            return _serve()
    except BloodSmearError as exc:
        if arguments.debug:
            raise
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if arguments.debug:
            raise
        print(f"UNEXPECTED_ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


def _infer(arguments: argparse.Namespace) -> int:
    settings = AppSettings(output_dir=arguments.output)
    service = InferenceService.from_settings(settings)
    image_data = arguments.input.read_bytes()
    result = service.infer_bytes(
        image_data,
        arguments.input.name,
        sample_id=arguments.sample_id or arguments.input.stem,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    destinations = [
        write_result_json(result, arguments.output / "result.json"),
        write_detections_csv(result, arguments.output / "detections.csv"),
        write_annotated_image(
            result,
            image_data,
            arguments.output / "annotated.png",
        ),
    ]
    for destination in destinations:
        print(destination)
    return 0


def _model_info() -> int:
    settings = AppSettings()
    service = InferenceService.from_settings(settings)
    manifest = service.package.manifest
    print(
        json.dumps(
            {
                "name": manifest.name,
                "version": manifest.version,
                "sha256": manifest.sha256.upper(),
                "provider": service.adapter.provider,
                "classes": manifest.classes,
                "license_status": manifest.license_status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _serve() -> int:
    settings = AppSettings()
    service = InferenceService.from_settings(settings)
    repository = BatchRepository(settings.database_path)
    repository.initialize()
    storage = BatchStorage(settings)
    manager = BatchManager(repository, storage, settings)
    worker = BatchWorker(
        repository,
        storage,
        service,
        service.package.manifest.classes,
        poll_interval_seconds=settings.batch_poll_interval_seconds,
    )
    cleanup = BatchCleanup(
        repository,
        storage,
        interval_seconds=settings.cleanup_interval_seconds,
    )
    app = create_app(service, settings, manager, worker, cleanup)
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
