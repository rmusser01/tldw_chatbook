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
from .server_evaluations_service import ServerEvaluationsService

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
