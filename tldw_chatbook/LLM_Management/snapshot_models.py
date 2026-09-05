"""Strict metadata DTOs and immutable prompt-cache operation projections."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictStr,
    field_validator,
)

from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
)

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

COMPATIBILITY_STATE_KEYS = frozenset(
    {
        "batch-size",
        "cache-type-k",
        "cache-type-v",
        "cont-batching",
        "context-shift",
        "ctx-size",
        "device",
        "effective-slot-contexts",
        "fit",
        "fit-ctx",
        "fit-target",
        "flash-attn",
        "gpu-layers",
        "image-max-tokens",
        "image-min-tokens",
        "keep",
        "kv-offload",
        "main-gpu",
        "mmproj-auto",
        "mmproj-device",
        "mmproj-offload",
        "mtmd-batch-max-tokens",
        "parallel",
        "rope-freq-base",
        "rope-freq-scale",
        "rope-scale",
        "rope-scaling",
        "split-mode",
        "swa-full",
        "tensor-split",
        "ubatch-size",
        "yarn-attn-factor",
        "yarn-beta-fast",
        "yarn-beta-slow",
        "yarn-ext-factor",
        "yarn-orig-ctx",
    }
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("value must be a lowercase SHA-256 digest")
    return value


def _validate_settings(
    value: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    keys = tuple(key for key, _setting in value)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError("settings must have sorted unique keys")
    if any(key not in COMPATIBILITY_STATE_KEYS for key in keys):
        raise ValueError("settings contain an unknown canonical key")
    if any(not setting for _key, setting in value):
        raise ValueError("settings values must not be empty")
    return value


class CompatibilityEvidence(_StrictFrozenModel):
    """Complete model, runtime, build, and effective-state identity."""

    model_sha256: StrictStr
    projector_sha256: StrictStr | None
    runtime_sha256: StrictStr
    build_info: StrictStr
    state_settings: tuple[tuple[StrictStr, StrictStr], ...]

    _digests = field_validator("model_sha256", "projector_sha256", "runtime_sha256")(
        _validate_sha256
    )
    _settings = field_validator("state_settings")(_validate_settings)

    @field_validator("build_info")
    @classmethod
    def _nonempty_build_info(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("build_info must not be empty")
        return value


@dataclass(frozen=True)
class FileIdentity:
    """One verified regular file identity retained only in process memory."""

    path: Path = field(repr=False)
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclass(frozen=True)
class LaunchDescriptor:
    """Immutable pre-readiness or finalized identity for one exact launch."""

    launch_id: str
    claim: ServerLaunchClaim = field(repr=False)
    base_url: str
    bearer_token: str | None = field(repr=False)
    child_env: Mapping[str, str] = field(repr=False)
    files: tuple[FileIdentity, ...] = field(repr=False)
    compatibility: CompatibilityEvidence | None
    disabled_reason: str | None
    _state_settings: tuple[tuple[str, str], ...] = field(
        default=(), repr=False, compare=False
    )
    _required_runtime_keys: frozenset[str] = field(
        default_factory=frozenset, repr=False, compare=False
    )
    _model_paths: tuple[Path, ...] = field(default=(), repr=False, compare=False)
    _projector_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_env", MappingProxyType(dict(self.child_env)))


class SlotObservation(_StrictFrozenModel):
    """Bounded slot status projected from a llama.cpp observation."""

    slot_id: Annotated[int, Field(strict=True, ge=0)]
    busy: StrictBool | None
    tokens: Annotated[int, Field(strict=True, ge=0)] | None
    context_size: Annotated[int, Field(strict=True, gt=0)] | None
    observed_at: StrictFloat


class ReadinessObservation(_StrictFrozenModel):
    """Whitelisted readiness evidence, never a retained raw HTTP response."""

    slots: tuple[SlotObservation, ...]
    build_info: StrictStr
    model_path: StrictStr = Field(repr=False)
    runtime_values: tuple[tuple[StrictStr, StrictStr], ...]

    _runtime_settings = field_validator("runtime_values")(_validate_settings)


class SlotReceipt(_StrictFrozenModel):
    """Validated counters acknowledged by one slot save or restore response."""

    slot_id: Annotated[int, Field(strict=True, ge=0)]
    filename: StrictStr
    tokens: Annotated[int, Field(strict=True, ge=0)]
    bytes: Annotated[int, Field(strict=True, ge=0)]

    @field_validator("filename")
    @classmethod
    def _safe_filename(cls, value: str) -> str:
        if not value or "\0" in value or PurePath(value).name != value:
            raise ValueError("filename must be a basename")
        return value


class SnapshotRecord(_StrictFrozenModel):
    """Versioned path-free metadata for one committed snapshot binary."""

    schema_version: Annotated[int, Field(strict=True, ge=1)] = 1
    snapshot_id: StrictStr
    filename: StrictStr
    created_utc: StrictStr
    publication_sequence: Annotated[int, Field(strict=True, ge=1)]
    source_slot: Annotated[int, Field(strict=True, ge=0)]
    tokens: Annotated[int, Field(strict=True, ge=0)]
    bytes: Annotated[int, Field(strict=True, ge=0)]
    sha256: StrictStr
    model_label: StrictStr
    compatibility: CompatibilityEvidence

    _digest = field_validator("sha256")(_validate_sha256)

    @field_validator("snapshot_id")
    @classmethod
    def _safe_snapshot_id(cls, value: str) -> str:
        if SAFE_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("snapshot_id has invalid syntax")
        return value

    @field_validator("filename")
    @classmethod
    def _record_filename(cls, value: str) -> str:
        if not value or PurePath(value).name != value:
            raise ValueError("filename must be a basename")
        return value

    @field_validator("created_utc", "model_label")
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


@dataclass(frozen=True)
class WorkingFile:
    """App-owned reservation for an active save or staged restore."""

    launch_id: str
    operation_id: str
    path: Path = field(repr=False)
    source_record: SnapshotRecord | None


@dataclass(frozen=True)
class SaveResult:
    """Committed save plus bounded retention cleanup outcomes."""

    record: SnapshotRecord
    removed_ids: tuple[str, ...]
    cleanup_failed_ids: tuple[str, ...]


@dataclass(frozen=True)
class CatalogPage:
    """A bounded retained-snapshot page with honest scan totals."""

    records: tuple[SnapshotRecord, ...]
    next_offset: int | None
    stored_bytes: int | None
    residual_bytes: int | None
    scan_complete: bool


@dataclass(frozen=True)
class ManagerView:
    """Payload-free state safe to project into the Textual widget."""

    launch_id: str | None
    status: str
    operation_id: str | None
    started_at: float | None
    slots: tuple[SlotObservation, ...]
    catalog: CatalogPage
    disabled_reason: str | None
    message: str | None


class SnapshotError(Exception):
    """Typed bounded failure without raw response or path content."""

    def __init__(self, code: str, submission_possible: bool) -> None:
        if SAFE_ID_PATTERN.fullmatch(code) is None:
            raise ValueError("invalid snapshot error code")
        if type(submission_possible) is not bool:
            raise TypeError("submission_possible must be a bool")
        self.code = code
        self.submission_possible = submission_possible
        super().__init__(code)


def replace_descriptor(
    descriptor: LaunchDescriptor, **changes: Any
) -> LaunchDescriptor:
    """Return an immutable descriptor replacement while preserving private state."""

    values = {
        field_name: getattr(descriptor, field_name)
        for field_name in LaunchDescriptor.__dataclass_fields__
    }
    values.update(changes)
    return LaunchDescriptor(**values)
