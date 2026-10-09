class BloodSmearError(Exception):
    """Base exception with a stable external error code."""

    code = "BLOODSMEAR_ERROR"


class ModelNotFoundError(BloodSmearError):
    code = "MODEL_NOT_FOUND"


class ModelHashMismatchError(BloodSmearError):
    code = "MODEL_HASH_MISMATCH"


class InvalidImageError(BloodSmearError):
    code = "INVALID_IMAGE"


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
