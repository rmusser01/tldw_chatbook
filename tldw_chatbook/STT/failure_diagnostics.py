"""Finite, content-free diagnostics for local speech failures."""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from enum import Enum


class STTFailureStage(str, Enum):
    """Owned runtime boundaries, rather than untrusted error descriptions."""

    MODEL_LOAD = "model-load"
    MEDIA_PREPARATION = "media-preparation"
    AUDIO_DECODE = "audio-decode"
    VAD = "vad"
    INFERENCE = "inference"
    NORMALIZATION = "normalization"
    PROVENANCE = "provenance"


class STTExceptionCategory(str, Enum):
    """Allowlisted exception identities; messages and arbitrary names are excluded."""

    UNKNOWN = "unknown"
    RUNTIME_ERROR = "RuntimeError"
    VALUE_ERROR = "ValueError"
    TYPE_ERROR = "TypeError"
    OS_ERROR = "OSError"
    FILE_NOT_FOUND_ERROR = "FileNotFoundError"
    PERMISSION_ERROR = "PermissionError"
    MEMORY_ERROR = "MemoryError"
    IMPORT_ERROR = "ImportError"
    MODULE_NOT_FOUND_ERROR = "ModuleNotFoundError"
    TIMEOUT_ERROR = "TimeoutError"
    EOF_ERROR = "EOFError"
    INDEX_ERROR = "IndexError"
    KEY_ERROR = "KeyError"
    STOP_ITERATION = "StopIteration"
    WAVE_ERROR = "wave.Error"
    PROCESS_ERROR = "subprocess.CalledProcessError"
    PROCESS_TIMEOUT = "subprocess.TimeoutExpired"
    ONNX_FAIL = "onnxruntime.Fail"
    ONNX_INVALID_ARGUMENT = "onnxruntime.InvalidArgument"
    ONNX_NO_SUCH_FILE = "onnxruntime.NoSuchFile"
    ONNX_NO_MODEL = "onnxruntime.NoModel"
    ONNX_ENGINE_ERROR = "onnxruntime.EngineError"
    ONNX_RUNTIME_EXCEPTION = "onnxruntime.RuntimeException"
    ONNX_INVALID_PROTOBUF = "onnxruntime.InvalidProtobuf"
    ONNX_MODEL_LOADED = "onnxruntime.ModelLoaded"
    ONNX_NOT_IMPLEMENTED = "onnxruntime.NotImplemented"
    ONNX_INVALID_GRAPH = "onnxruntime.InvalidGraph"
    ONNX_EP_FAIL = "onnxruntime.EPFail"


@dataclass(frozen=True, slots=True)
class STTFailureDiagnostic:
    """Picklable metadata with no free-text or path-bearing fields."""

    stage: STTFailureStage
    exception_category: STTExceptionCategory

    def __post_init__(self) -> None:
        if type(self.stage) is not STTFailureStage:
            raise TypeError("stage must be an STTFailureStage")
        if type(self.exception_category) is not STTExceptionCategory:
            raise TypeError("exception_category must be an STTExceptionCategory")


class STTDiagnosticFailure(RuntimeError):
    """Transient safe wrapper discarded by the existing provider failure boundary."""

    def __init__(self, diagnostic: STTFailureDiagnostic) -> None:
        self.stt_diagnostic = diagnostic
        super().__init__("local_stt_stage_failed")


_EXCEPTION_CATEGORIES = {
    RuntimeError: STTExceptionCategory.RUNTIME_ERROR,
    ValueError: STTExceptionCategory.VALUE_ERROR,
    TypeError: STTExceptionCategory.TYPE_ERROR,
    OSError: STTExceptionCategory.OS_ERROR,
    FileNotFoundError: STTExceptionCategory.FILE_NOT_FOUND_ERROR,
    PermissionError: STTExceptionCategory.PERMISSION_ERROR,
    MemoryError: STTExceptionCategory.MEMORY_ERROR,
    ImportError: STTExceptionCategory.IMPORT_ERROR,
    ModuleNotFoundError: STTExceptionCategory.MODULE_NOT_FOUND_ERROR,
    TimeoutError: STTExceptionCategory.TIMEOUT_ERROR,
    EOFError: STTExceptionCategory.EOF_ERROR,
    IndexError: STTExceptionCategory.INDEX_ERROR,
    KeyError: STTExceptionCategory.KEY_ERROR,
    StopIteration: STTExceptionCategory.STOP_ITERATION,
    wave.Error: STTExceptionCategory.WAVE_ERROR,
    subprocess.CalledProcessError: STTExceptionCategory.PROCESS_ERROR,
    subprocess.TimeoutExpired: STTExceptionCategory.PROCESS_TIMEOUT,
}
_ONNX_CATEGORIES = {
    category.value.removeprefix("onnxruntime."): category
    for category in STTExceptionCategory
    if category.value.startswith("onnxruntime.")
}


def diagnostic_from_exception(
    stage: STTFailureStage, error: BaseException
) -> STTFailureDiagnostic:
    """Retain an owned stage or classify the exact exception without its text."""
    diagnostic = getattr(error, "stt_diagnostic", None)
    if type(diagnostic) is STTFailureDiagnostic:
        return diagnostic
    error_type = type(error)
    category = _EXCEPTION_CATEGORIES.get(error_type, STTExceptionCategory.UNKNOWN)
    if error_type.__module__ == "onnxruntime.capi.onnxruntime_pybind11_state":
        category = _ONNX_CATEGORIES.get(
            error_type.__name__, STTExceptionCategory.UNKNOWN
        )
    return STTFailureDiagnostic(stage, category)
