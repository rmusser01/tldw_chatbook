"""Shared local/server evaluation seam for compat-first evaluation surfaces."""

from .evaluation_normalizers import (
    RESERVED_LOCAL_METADATA_KEY,
    normalize_evaluation_dataset_record,
    normalize_evaluation_record,
    normalize_evaluation_run_record,
    normalize_evaluation_target_record,
)
from .evaluation_scope_service import EvaluationBackend, EvaluationScopeService
from .local_evaluations_service import LocalEvaluationsService

__all__ = [
    "EvaluationBackend",
    "EvaluationScopeService",
    "LocalEvaluationsService",
    "RESERVED_LOCAL_METADATA_KEY",
    "ServerEvaluationsService",
    "normalize_evaluation_dataset_record",
    "normalize_evaluation_record",
    "normalize_evaluation_run_record",
    "normalize_evaluation_target_record",
]


def __getattr__(name: str):
    """Keep server transport imports out of local normalization callers."""
    if name != "ServerEvaluationsService":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .server_evaluations_service import ServerEvaluationsService

    globals()[name] = ServerEvaluationsService
    return ServerEvaluationsService


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
