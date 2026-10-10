from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter_ns
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bloodsmear.artifacts import (
    write_annotated_image,
    write_detections_csv,
    write_result_json,
)
from bloodsmear.config import AppSettings
from bloodsmear.batch.domain import BatchMode
from bloodsmear.batch.domain import TERMINAL_JOB_STATUSES
from bloodsmear.domain import InferenceResult, QCInfo, SampleMetadata
from bloodsmear.errors import (
    BloodSmearError,
    BatchUploadFileError,
    GPUUnavailableError,
    InferenceFailedError,
    InvalidImageError,
    ModelHashMismatchError,
    ModelNotFoundError,
    ModelValidationError,
    BatchTooManyFilesError,
    BatchTotalSizeExceededError,
    GroupSampleIdRequiredError,
    JobNotFoundError,
    JobNotReadyError,
)
from bloodsmear.inference import InferenceService
from bloodsmear.reporting.service import ReportService


LOGGER = logging.getLogger("bloodsmear.api")
PACKAGE_DIR = Path(__file__).resolve().parent


def create_app(
    service: InferenceService,
    settings: AppSettings,
    batch_manager=None,
    batch_worker=None,
    batch_cleanup=None,
    report_service=None,
) -> FastAPI:
    reports = report_service or ReportService()
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if batch_worker is not None:
            batch_worker.start()
        if batch_cleanup is not None:
            batch_cleanup.start()
        try:
            yield
        finally:
            if batch_cleanup is not None:
                batch_cleanup.stop()
            if batch_worker is not None:
                batch_worker.stop()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    app = FastAPI(
        title="Blood Smear Inference MVP",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.settings = settings
    app.mount(
        "/artifacts",
        StaticFiles(directory=settings.output_dir),
        name="artifacts",
    )
    app.mount(
        "/static",
        StaticFiles(directory=PACKAGE_DIR / "static"),
        name="static",
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = uuid4().hex
        started = perf_counter_ns()
        response = await call_next(request)
        duration_ms = (perf_counter_ns() - started) / 1_000_000.0
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "request_complete request_id=%s method=%s path=%s status=%d duration_ms=%.3f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.exception_handler(BloodSmearError)
    async def domain_error_handler(_request: Request, exc: BloodSmearError):
        if isinstance(exc, InvalidImageError):
            status_code = 400
        elif isinstance(exc, (BatchTooManyFilesError, GroupSampleIdRequiredError)):
            status_code = 400
        elif isinstance(exc, BatchTotalSizeExceededError):
            status_code = 413
        elif isinstance(exc, JobNotFoundError):
            status_code = 404
        elif isinstance(exc, JobNotReadyError):
            status_code = 409
        elif isinstance(
            exc,
            (
                GPUUnavailableError,
                ModelNotFoundError,
                ModelHashMismatchError,
                ModelValidationError,
            ),
        ):
            status_code = 503
        elif isinstance(exc, InferenceFailedError):
            status_code = 500
        else:
            status_code = 500
        error = {"code": exc.code, "message": str(exc)}
        if isinstance(exc, BatchUploadFileError):
            error["file"] = exc.file_feedback
        return JSONResponse(
            status_code=status_code,
            content={"error": error},
        )

    async def run_upload(
        image: UploadFile,
        sample_id: str | None,
        metadata: SampleMetadata | None = None,
    ) -> tuple[InferenceResult, dict[str, str]]:
        data = await image.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise InvalidImageError("Image exceeds the 25 MiB upload limit")
        filename = image.filename or "upload.png"
        result = service.infer_bytes(data, filename, sample_id, metadata=metadata)
        request_directory = settings.output_dir / uuid4().hex
        json_path = write_result_json(result, request_directory / "result.json")
        csv_path = write_detections_csv(result, request_directory / "detections.csv")
        annotated_path = write_annotated_image(
            result,
            data,
            request_directory / "annotated.png",
        )
        prefix = f"/artifacts/{request_directory.name}"
        excel_url = None
        pdf_url = None
        try:
            report_paths = reports.generate_single(
                result,
                service.package.manifest.license_status,
                request_directory,
            )
            excel_url = f"{prefix}/{report_paths.excel.name}"
            pdf_url = f"{prefix}/{report_paths.pdf.name}"
        except Exception:
            warnings = list(dict.fromkeys([*result.qc.warnings, "REPORT_GENERATION_FAILED"]))
            result = result.model_copy(
                update={"qc": QCInfo(status="warning", warnings=warnings)}
            )
            write_result_json(result, json_path)
        return result, {
            "json": f"{prefix}/{json_path.name}",
            "csv": f"{prefix}/{csv_path.name}",
            "annotated_image": f"{prefix}/{annotated_path.name}",
            "excel": excel_url,
            "pdf": pdf_url,
        }

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def health_ready():
        provider = service.adapter.provider
        if settings.require_gpu and provider != "CUDAExecutionProvider":
            error = GPUUnavailableError(
                "CUDAExecutionProvider is required but unavailable"
            )
            return JSONResponse(
                status_code=503,
                content={"error": {"code": error.code, "message": str(error)}},
            )
        return {"status": "ready", "provider": provider}

    @app.get("/api/v1/models/current")
    async def current_model() -> dict[str, object]:
        manifest = service.package.manifest
        return {
            "name": manifest.name,
            "version": manifest.version,
            "sha256": manifest.sha256.upper(),
            "classes": manifest.classes,
            "license_status": manifest.license_status,
            "provider": service.adapter.provider,
        }

    @app.post("/api/v1/infer")
    async def infer_api(
        image: UploadFile = File(...),
        sample_id: str | None = Form(default=None),
        patient_id: str | None = Form(default=None),
        stain_method: str | None = Form(default=None),
        scanner_model: str | None = Form(default=None),
        magnification: str | None = Form(default=None),
        pixel_resolution: str | None = Form(default=None),
        operator: str | None = Form(default=None),
        notes: str | None = Form(default=None),
    ) -> JSONResponse:
        metadata = SampleMetadata(patient_id=patient_id, stain_method=stain_method, scanner_model=scanner_model, magnification=magnification, pixel_resolution=pixel_resolution, operator=operator, notes=notes)
        if not any(metadata.model_dump().values()): metadata = None
        result, artifact_urls = await run_upload(image, sample_id, metadata)
        payload = result.model_dump(mode="json")
        payload["artifacts"] = artifact_urls
        return JSONResponse(content=payload)

    @app.post("/api/v1/jobs", status_code=202)
    async def create_batch_job(
        mode: BatchMode = Form(...),
        sample_id: str | None = Form(default=None),
        patient_id: str | None = Form(default=None),
        stain_method: str | None = Form(default=None),
        scanner_model: str | None = Form(default=None),
        magnification: str | None = Form(default=None),
        pixel_resolution: str | None = Form(default=None),
        operator: str | None = Form(default=None),
        notes: str | None = Form(default=None),
        files: list[UploadFile] = File(...),
    ) -> dict[str, str]:
        if batch_manager is None:
            raise JobNotReadyError("Batch subsystem is not configured")
        metadata = SampleMetadata(patient_id=patient_id, stain_method=stain_method, scanner_model=scanner_model, magnification=magnification, pixel_resolution=pixel_resolution, operator=operator, notes=notes)
        if not any(metadata.model_dump().values()):
            metadata = None
        if metadata is None:
            job = await batch_manager.create_job(mode, sample_id, files)
        else:
            job = await batch_manager.create_job(mode, sample_id, files, metadata)
        base = f"/api/v1/jobs/{job.id}"
        return {
            "job_id": job.id,
            "status": job.status.value,
            "status_url": base,
            "results_url": f"{base}/results",
            "download_url": f"{base}/download",
        }

    @app.get("/api/v1/jobs/{job_id}")
    async def batch_job_status(job_id: str):
        if batch_manager is None:
            raise JobNotReadyError("Batch subsystem is not configured")
        return batch_manager.get_job(job_id).model_dump(mode="json")

    @app.get("/api/v1/jobs/{job_id}/results")
    async def batch_job_results(job_id: str):
        if batch_manager is None:
            raise JobNotReadyError("Batch subsystem is not configured")
        return batch_manager.get_results(job_id)

    @app.get("/api/v1/jobs/{job_id}/download")
    async def batch_job_download(job_id: str):
        if batch_manager is None:
            raise JobNotReadyError("Batch subsystem is not configured")
        return FileResponse(
            batch_manager.get_download_path(job_id),
            media_type="application/zip",
            filename=f"{job_id}-results.zip",
        )

    @app.delete("/api/v1/jobs/{job_id}", status_code=204)
    async def delete_batch_job(job_id: str) -> Response:
        if batch_manager is None:
            raise JobNotReadyError("Batch subsystem is not configured")
        batch_manager.delete_job(job_id)
        return Response(status_code=204)

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"result": None, "artifacts": None, "fragment": False},
        )

    @app.post("/ui/infer", response_class=HTMLResponse)
    async def infer_ui(
        request: Request,
        image: UploadFile = File(...),
        sample_id: str | None = Form(default=None),
    ):
        result, artifact_urls = await run_upload(image, sample_id)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": result,
                "artifacts": artifact_urls,
                "fragment": request.headers.get("HX-Request") == "true",
            },
        )

    @app.post("/ui/jobs", response_class=HTMLResponse, status_code=202)
    async def create_batch_ui(
        request: Request,
        mode: BatchMode = Form(...),
        sample_id: str | None = Form(default=None),
        patient_id: str | None = Form(default=None),
        stain_method: str | None = Form(default=None),
        scanner_model: str | None = Form(default=None),
        magnification: str | None = Form(default=None),
        pixel_resolution: str | None = Form(default=None),
        operator: str | None = Form(default=None),
        notes: str | None = Form(default=None),
        files: list[UploadFile] = File(...),
    ):
        if batch_manager is None:
            raise JobNotReadyError("Batch subsystem is not configured")
        metadata = SampleMetadata(patient_id=patient_id, stain_method=stain_method, scanner_model=scanner_model, magnification=magnification, pixel_resolution=pixel_resolution, operator=operator, notes=notes)
        if not any(metadata.model_dump().values()): metadata = None
        job = await batch_manager.create_job(mode, sample_id, files, metadata) if metadata else await batch_manager.create_job(mode, sample_id, files)
        return templates.TemplateResponse(
            request=request,
            name="batch_status.html",
            context={"job": job, "terminal": job.status in TERMINAL_JOB_STATUSES},
            status_code=202,
        )

    @app.get("/ui/jobs/{job_id}/status", response_class=HTMLResponse)
    async def batch_status_ui(request: Request, job_id: str):
        if batch_manager is None:
            raise JobNotReadyError("Batch subsystem is not configured")
        job = batch_manager.get_job(job_id)
        return templates.TemplateResponse(
            request=request,
            name="batch_status.html",
            context={"job": job, "terminal": job.status in TERMINAL_JOB_STATUSES},
        )

    return app
