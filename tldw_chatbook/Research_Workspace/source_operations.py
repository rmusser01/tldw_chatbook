"""Validated durable receipts for Research Workspace source intake."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import re

from .contracts import WorkspaceDataSource


MAX_OPERATION_ID_CHARS = 256
MAX_IDEMPOTENCY_KEY_CHARS = 512
MAX_IDENTITY_CHARS = 256
MAX_ERROR_CODE_CHARS = 64
MAX_ERROR_MESSAGE_CHARS = 512
MAX_TIMESTAMP_CHARS = 64


class SourceOperationValidationError(ValueError):
    """Raised when a source-operation receipt violates its durable contract."""


class CanonicalItemType(StrEnum):
    """Authority-specific owner of the canonical ingested item."""

    LOCAL_LIBRARY = "local_library"
    SERVER_MEDIA = "server_media"


class SourceOperationStage(StrEnum):
    """Ordered source intake stages."""

    CATALOG = "catalog"
    ASSOCIATION = "association"
    READINESS = "readiness"


class SourceOperationStatus(StrEnum):
    """Monotonic state of one source intake stage."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+\S+"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|authorization|password|secret|token)\s*[:=]"),
)
_CREDENTIAL_URL = re.compile(r"(?i)https?://[^\s/:@]+:[^\s/@]+@")
_PRIVATE_PATH = re.compile(
    r"(?:^|[\s\"'])(?:~[/\\]|[A-Za-z]:\\|/(?:Users|home|private|tmp|var|Volumes)/)"
)
_ERROR_CODE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _enum_value(value: object, enum_type: type[StrEnum], field_name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        allowed = ", ".join(item.value for item in enum_type)
        raise SourceOperationValidationError(
            f"{field_name} must be one of: {allowed}"
        ) from None


def _safe_text(
    value: object,
    field_name: str,
    *,
    maximum: int,
    required: bool,
) -> str:
    if not isinstance(value, str):
        raise SourceOperationValidationError(f"{field_name} must be text")
    normalized = value.strip()
    if required and not normalized:
        raise SourceOperationValidationError(f"{field_name} must not be blank")
    if len(normalized) > maximum:
        raise SourceOperationValidationError(
            f"{field_name} exceeds maximum {maximum} characters"
        )
    if any(character in normalized for character in ("\n", "\r", "\x00")):
        raise SourceOperationValidationError(f"{field_name} must be single-line text")
    if any(pattern.search(normalized) for pattern in _SECRET_PATTERNS):
        raise SourceOperationValidationError(
            f"{field_name} must not contain secret material"
        )
    if _CREDENTIAL_URL.search(normalized):
        raise SourceOperationValidationError(
            f"{field_name} must not contain a credential-bearing URL"
        )
    if _PRIVATE_PATH.search(normalized):
        raise SourceOperationValidationError(
            f"{field_name} must not contain a local private path"
        )
    return normalized


def _timestamp(value: object, field_name: str) -> str:
    normalized = _safe_text(
        value,
        field_name,
        maximum=MAX_TIMESTAMP_CHARS,
        required=True,
    )
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        raise SourceOperationValidationError(
            f"{field_name} must be an ISO-8601 timestamp"
        ) from None
    if parsed.tzinfo is None:
        raise SourceOperationValidationError(
            f"{field_name} must include an explicit timezone"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ResearchSourceOperation:
    """Immutable qualified intent and independent stage receipt."""

    operation_id: str
    idempotency_key: str
    data_source: WorkspaceDataSource
    workspace_id: str
    canonical_item_type: CanonicalItemType
    desired_selected: bool
    created_at: str
    updated_at: str
    server_profile_id: str = ""
    principal_id: str = ""
    ingest_job_id: str = ""
    canonical_item_id: str = ""
    workspace_source_id: str = ""
    catalog_status: SourceOperationStatus = SourceOperationStatus.PENDING
    association_status: SourceOperationStatus = SourceOperationStatus.PENDING
    readiness_status: SourceOperationStatus = SourceOperationStatus.PENDING
    error_stage: SourceOperationStage | None = None
    error_code: str = ""
    error_message: str = ""
    revision: int = 1

    def __post_init__(self) -> None:
        data_source = _enum_value(self.data_source, WorkspaceDataSource, "data_source")
        canonical_item_type = _enum_value(
            self.canonical_item_type, CanonicalItemType, "canonical_item_type"
        )
        catalog_status = _enum_value(
            self.catalog_status, SourceOperationStatus, "catalog_status"
        )
        association_status = _enum_value(
            self.association_status, SourceOperationStatus, "association_status"
        )
        readiness_status = _enum_value(
            self.readiness_status, SourceOperationStatus, "readiness_status"
        )
        error_stage = (
            None
            if self.error_stage is None
            else _enum_value(self.error_stage, SourceOperationStage, "error_stage")
        )

        normalized = {
            "operation_id": _safe_text(
                self.operation_id,
                "operation_id",
                maximum=MAX_OPERATION_ID_CHARS,
                required=True,
            ),
            "idempotency_key": _safe_text(
                self.idempotency_key,
                "idempotency_key",
                maximum=MAX_IDEMPOTENCY_KEY_CHARS,
                required=True,
            ),
            "server_profile_id": _safe_text(
                self.server_profile_id,
                "server_profile_id",
                maximum=MAX_IDENTITY_CHARS,
                required=False,
            ),
            "principal_id": _safe_text(
                self.principal_id,
                "principal_id",
                maximum=MAX_IDENTITY_CHARS,
                required=False,
            ),
            "workspace_id": _safe_text(
                self.workspace_id,
                "workspace_id",
                maximum=MAX_IDENTITY_CHARS,
                required=True,
            ),
            "ingest_job_id": _safe_text(
                self.ingest_job_id,
                "ingest_job_id",
                maximum=MAX_IDENTITY_CHARS,
                required=False,
            ),
            "canonical_item_id": _safe_text(
                self.canonical_item_id,
                "canonical_item_id",
                maximum=MAX_IDENTITY_CHARS,
                required=False,
            ),
            "workspace_source_id": _safe_text(
                self.workspace_source_id,
                "workspace_source_id",
                maximum=MAX_IDENTITY_CHARS,
                required=False,
            ),
        }
        error_code = _safe_text(
            self.error_code,
            "error_code",
            maximum=MAX_ERROR_CODE_CHARS,
            required=error_stage is not None,
        )
        error_message = _safe_text(
            self.error_message,
            "error_message",
            maximum=MAX_ERROR_MESSAGE_CHARS,
            required=error_stage is not None,
        )
        created_at = _timestamp(self.created_at, "created_at")
        updated_at = _timestamp(self.updated_at, "updated_at")

        if type(self.desired_selected) is not bool:
            raise SourceOperationValidationError("desired_selected must be bool")
        if type(self.revision) is not int or self.revision < 1:
            raise SourceOperationValidationError("revision must be a positive integer")
        if error_code and not _ERROR_CODE.fullmatch(error_code):
            raise SourceOperationValidationError(
                "error_code must use lowercase letters, digits, dots, dashes, or underscores"
            )

        if data_source is WorkspaceDataSource.LOCAL:
            if normalized["server_profile_id"] or normalized["principal_id"]:
                raise SourceOperationValidationError(
                    "Local operations cannot carry server identity metadata"
                )
            if canonical_item_type is not CanonicalItemType.LOCAL_LIBRARY:
                raise SourceOperationValidationError(
                    "canonical_item_type must be local_library for Local operations"
                )
        else:
            if not normalized["server_profile_id"]:
                raise SourceOperationValidationError(
                    "server_profile_id is required for Server operations"
                )
            if canonical_item_type is not CanonicalItemType.SERVER_MEDIA:
                raise SourceOperationValidationError(
                    "canonical_item_type must be server_media for Server operations"
                )

        if (
            association_status is not SourceOperationStatus.PENDING
            and catalog_status is not SourceOperationStatus.SUCCEEDED
        ):
            raise SourceOperationValidationError(
                "catalog must succeed before association can advance"
            )
        if (
            readiness_status is not SourceOperationStatus.PENDING
            and association_status is not SourceOperationStatus.SUCCEEDED
        ):
            raise SourceOperationValidationError(
                "association must succeed before readiness can advance"
            )
        if (
            catalog_status is SourceOperationStatus.SUCCEEDED
            and not normalized["canonical_item_id"]
        ):
            raise SourceOperationValidationError(
                "canonical_item_id is required when catalog succeeds"
            )
        if (
            association_status is SourceOperationStatus.SUCCEEDED
            and not normalized["workspace_source_id"]
        ):
            raise SourceOperationValidationError(
                "workspace_source_id is required when association succeeds"
            )

        stage_statuses = {
            SourceOperationStage.CATALOG: catalog_status,
            SourceOperationStage.ASSOCIATION: association_status,
            SourceOperationStage.READINESS: readiness_status,
        }
        failed_stages = {
            stage
            for stage, status in stage_statuses.items()
            if status is SourceOperationStatus.FAILED
        }
        if error_stage is None:
            if failed_stages or error_code or error_message:
                raise SourceOperationValidationError(
                    "failed status and diagnostic fields require error_stage"
                )
        elif failed_stages != {error_stage}:
            raise SourceOperationValidationError(
                "error_stage must name the single failed stage"
            )

        object.__setattr__(self, "data_source", data_source)
        object.__setattr__(self, "canonical_item_type", canonical_item_type)
        object.__setattr__(self, "catalog_status", catalog_status)
        object.__setattr__(self, "association_status", association_status)
        object.__setattr__(self, "readiness_status", readiness_status)
        object.__setattr__(self, "error_stage", error_stage)
        object.__setattr__(self, "error_code", error_code)
        object.__setattr__(self, "error_message", error_message)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        for field_name, value in normalized.items():
            object.__setattr__(self, field_name, value)

    @property
    def complete(self) -> bool:
        """Return whether every independently reported stage succeeded."""

        return all(
            status is SourceOperationStatus.SUCCEEDED
            for status in (
                self.catalog_status,
                self.association_status,
                self.readiness_status,
            )
        )
