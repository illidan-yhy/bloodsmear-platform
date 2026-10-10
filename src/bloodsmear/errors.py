class BloodSmearError(Exception):
    """Base exception with a stable external error code."""

    code = "BLOODSMEAR_ERROR"


class ModelNotFoundError(BloodSmearError):
    code = "MODEL_NOT_FOUND"


class ModelHashMismatchError(BloodSmearError):
    code = "MODEL_HASH_MISMATCH"


class InvalidImageError(BloodSmearError):
    code = "INVALID_IMAGE"


class UnsupportedImageModeError(InvalidImageError):
    code = "UNSUPPORTED_IMAGE_MODE"

    def __init__(self):
        super().__init__("暂不支持 16 位灰度图像，请提供经确认的 8 位显微图像。")


class BatchUploadFileError(InvalidImageError):
    """A rejected upload with a safe basename and its 1-based selection index."""
    def __init__(self, message: str, *, filename: str, index: int, reason: str):
        super().__init__(message)
        self.file_feedback = {"filename": filename, "index": index, "reason": reason}


class ModelValidationError(BloodSmearError):
    code = "MODEL_VALIDATION_FAILED"


class GPUUnavailableError(BloodSmearError):
    code = "GPU_UNAVAILABLE"


class InferenceFailedError(BloodSmearError):
    code = "INFERENCE_FAILED"


class BatchTooManyFilesError(BloodSmearError):
    code = "BATCH_TOO_MANY_FILES"


class BatchTotalSizeExceededError(BloodSmearError):
    code = "BATCH_TOTAL_SIZE_EXCEEDED"


class GroupSampleIdRequiredError(BloodSmearError):
    code = "GROUP_SAMPLE_ID_REQUIRED"


class JobNotFoundError(BloodSmearError):
    code = "JOB_NOT_FOUND"


class JobNotReadyError(BloodSmearError):
    code = "JOB_NOT_READY"


class ItemInferenceFailedError(BloodSmearError):
    code = "ITEM_INFERENCE_FAILED"


class UnsafeJobPathError(BloodSmearError):
    code = "UNSAFE_JOB_PATH"


class ReportGenerationError(BloodSmearError):
    code = "REPORT_GENERATION_FAILED"


class ServiceAlreadyRunningError(BloodSmearError):
    code = "SERVICE_ALREADY_RUNNING"


class PortInUseError(BloodSmearError):
    code = "PORT_IN_USE"


def batch_item_error(exc: Exception) -> tuple[str, str]:
    """Public batch messages are curated, never raw exception/path/traceback text."""
    if isinstance(exc, UnsupportedImageModeError):
        return exc.code, str(exc)
    if isinstance(exc, InvalidImageError):
        reason = str(exc).lower()
        if "pixel" in reason:
            return exc.code, "图片像素尺寸过大，无法安全读取，请缩小图片后重试。"
        if "exceeds" in reason and "mib" in reason:
            return exc.code, "图片文件大小超过限制，请使用较小的图片。"
        if "unsupported image extension" in reason:
            return exc.code, "文件格式不支持，请使用 JPG、PNG 或 TIFF 图片。"
        return exc.code, "图片损坏或内容无效，无法读取，请重新选择图片。"
    messages = {
        "GPU_UNAVAILABLE": "GPU 不可用，请检查显卡驱动和运行环境。",
        "INFERENCE_FAILED": "模型推理失败，请检查模型和显卡运行环境，并查看日志。",
        "MODEL_NOT_FOUND": "模型文件缺失，请联系维护人员。",
        "MODEL_HASH_MISMATCH": "模型文件校验失败，请联系维护人员。",
        "MODEL_VALIDATION_FAILED": "模型配置或输出不符合要求，请联系维护人员。",
    }
    if isinstance(exc, BloodSmearError):
        return exc.code, messages.get(exc.code, "处理图片失败，请查看日志或联系维护人员。")
    if isinstance(exc, FileNotFoundError):
        return "ITEM_INPUT_MISSING", "图片或所需文件缺失，请重新上传或联系维护人员。"
    return "ITEM_INFERENCE_FAILED", "处理图片失败，请查看日志或联系维护人员。"
