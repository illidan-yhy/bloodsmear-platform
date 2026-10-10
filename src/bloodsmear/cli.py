from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
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
from bloodsmear.errors import BloodSmearError, GPUUnavailableError, PortInUseError
from bloodsmear.inference import InferenceService
from bloodsmear.logging_config import configure_logging
from bloodsmear.startup_status import record_startup_status
from bloodsmear.service_instance import ServiceLock, reserve_port, platform_is_running
from bloodsmear.batch.cleanup import BatchCleanup
from bloodsmear.batch.manager import BatchManager
from bloodsmear.batch.repository import BatchRepository
from bloodsmear.batch.storage import BatchStorage
from bloodsmear.batch.worker import BatchWorker


LOGGER = logging.getLogger("bloodsmear.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bloodsmear")
    parser.add_argument("--debug", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    infer_parser = commands.add_parser("infer", help="Infer one image")
    infer_parser.add_argument("input", type=Path)
    infer_parser.add_argument("--output", type=Path, default=Path("outputs/cli"))
    infer_parser.add_argument("--sample-id")

    commands.add_parser("model-info", help="Show active model information")
    serve_parser = commands.add_parser("serve", help="Start the local HTTP service")
    serve_parser.add_argument("--open-browser", action="store_true", help="Open the existing service on repeated startup")
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
            return _serve(open_browser=arguments.open_browser)
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


def _serve(open_browser: bool = False) -> int:
    settings = AppSettings()
    guard = ServiceLock(settings.database_path, port=settings.port)
    if not guard.acquire():
        url = guard.owner_url()
        print(f"服务已打开或正在启动，请使用已有窗口。{url or ''}")
        if open_browser and url:
            webbrowser.open(url)
        return 0
    listener = None
    try:
        try:
            listener = reserve_port(settings.host, settings.port)
        except OSError as exc:
            if platform_is_running(settings.port):
                url = f"http://127.0.0.1:{settings.port}"
                print(f"服务已打开：{url}")
                if open_browser:
                    webbrowser.open(url)
                return 0
            raise PortInUseError(f"启动失败：端口 {settings.port} 已被占用或无法绑定，请检查端口设置。未恢复或处理批量任务。") from exc
        return _serve_reserved(settings, listener, guard)
    finally:
        if listener is not None:
            listener.close()
        guard.release()


def run_http_server(app, settings, listener) -> None:
    config = uvicorn.Config(app, host=settings.host, port=settings.port, log_config=None)
    uvicorn.Server(config).run(sockets=[listener])


def _serve_reserved(settings, listener, guard) -> int:
    provider = None
    worker = None
    record_startup_status(settings, "starting")
    try:
        service = InferenceService.from_settings(settings)
        provider = service.adapter.provider
        if settings.require_gpu and provider != "CUDAExecutionProvider":
            raise GPUUnavailableError(
                f"启动失败：配置要求使用 GPU，但实际使用的是 {provider}。"
                "请检查 NVIDIA 显卡驱动及 GPU 部署依赖；Windows 还需检查 VC++ x64 运行库。服务未启动。"
            )
        record_startup_status(settings, "starting", provider=provider)
        LOGGER.info("startup_provider provider=%s require_gpu=%s port=%d", provider, settings.require_gpu, settings.port)
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
            service_lock=guard,
        )
        cleanup = BatchCleanup(
            repository,
            storage,
            interval_seconds=settings.cleanup_interval_seconds,
        )
        app = create_app(service, settings, manager, worker, cleanup)
        run_http_server(app, settings, listener)
    except Exception as exc:
        code = exc.code if isinstance(exc, BloodSmearError) else "UNEXPECTED_ERROR"
        record_startup_status(settings, "failed", provider=provider, error_code=code)
        LOGGER.error("startup_failed error_code=%s provider=%s port=%d", code, provider, settings.port)
        raise
    except SystemExit as exc:
        if exc.code not in (None, 0):
            record_startup_status(settings, "failed", provider=provider, error_code="SERVER_START_FAILED")
        else:
            record_startup_status(settings, "stopped", provider=provider)
        raise
    except KeyboardInterrupt:
        record_startup_status(settings, "stopped", provider=provider)
        return 0
    finally:
        # Keep the database lock until the active image has finished and the worker exits.
        if worker is not None:
            worker.stop(timeout=None)
    record_startup_status(settings, "stopped", provider=provider)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
