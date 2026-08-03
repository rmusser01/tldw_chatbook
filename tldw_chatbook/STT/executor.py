"""Generation-fenced protocol and parent controller for local batch STT.

The protocol portion of this module deliberately imports no provider, artifact,
ingestion, or UI implementation.  Spawned workers can import these frozen data
objects without loading a native speech runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .contracts import (
    DeviceFailureOrigin,
    ExecutionDevice,
    TranscriptionFailureCode,
)

_MAX_RECOVERY_ACTIONS = 8
_MAX_RECOVERY_ACTION_LENGTH = 80


def _require_generation_and_attempt(generation: int, attempt_id: str) -> None:
    if type(generation) is not int or generation <= 0:
        raise ValueError("generation must be a positive integer")
    if type(attempt_id) is not str or not attempt_id.strip():
        raise ValueError("attempt_id must be a non-empty string")


def _require_nonempty_text(field_name: str, value: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class WorkerPhase(str, Enum):
    """Stable progress phases owned by the heavy worker."""

    PREPARING = "preparing"
    LOADING = "loading"
    TRANSCRIBING = "transcribing"
    POST_PROCESSING = "post-processing"


@dataclass(frozen=True, slots=True)
class LocalSourceSnapshot:
    """Private transient identity for unmanaged local model files."""

    token: str = field(repr=False)
    paths: tuple[Path, ...] = field(repr=False)
    identities: tuple[tuple[int, int, int, int], ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require_nonempty_text("token", self.token)
        if not self.paths or len(self.paths) != len(self.identities):
            raise ValueError(
                "snapshot paths and identities must be non-empty and aligned"
            )
        if any(not isinstance(path, Path) for path in self.paths):
            raise TypeError("snapshot paths must contain only Path values")
        if any(
            type(identity) is not tuple
            or len(identity) != 4
            or any(type(component) is not int for component in identity)
            for identity in self.identities
        ):
            raise TypeError("snapshot identities must be four-integer tuples")


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Complete identity of the one model allowed to reside in a worker."""

    provider_id: str
    model_id: str
    root_revision: str | None
    closure_fingerprint: str | None
    precision: str
    device: ExecutionDevice
    local_snapshot_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require_nonempty_text("provider_id", self.provider_id)
        _require_nonempty_text("model_id", self.model_id)
        _require_nonempty_text("precision", self.precision)
        if self.root_revision is not None:
            _require_nonempty_text("root_revision", self.root_revision)
        if self.closure_fingerprint is not None:
            _require_nonempty_text("closure_fingerprint", self.closure_fingerprint)
        if type(self.device) is not ExecutionDevice:
            raise TypeError("device must be an ExecutionDevice")
        if self.local_snapshot_token is not None:
            _require_nonempty_text("local_snapshot_token", self.local_snapshot_token)


@dataclass(frozen=True, slots=True)
class ExecutorRequest:
    """One heavy batch request sent to a specific executor generation."""

    generation: int
    attempt_id: str
    job_id: str
    source_path: Path = field(repr=False)
    identity: ModelIdentity
    options: dict[str, Any] = field(repr=False)
    local_source: LocalSourceSnapshot | None = field(default=None, repr=False)
    managed_store_root: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require_generation_and_attempt(self.generation, self.attempt_id)
        _require_nonempty_text("job_id", self.job_id)
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a Path")
        if type(self.identity) is not ModelIdentity:
            raise TypeError("identity must be a ModelIdentity")
        if type(self.options) is not dict:
            raise TypeError("options must be a dict")
        if (
            self.local_source is not None
            and type(self.local_source) is not LocalSourceSnapshot
        ):
            raise TypeError("local_source must be a LocalSourceSnapshot")
        if self.managed_store_root is not None and not isinstance(
            self.managed_store_root, Path
        ):
            raise TypeError("managed_store_root must be a Path")


@dataclass(frozen=True, slots=True)
class ExecutorEvent:
    """One bounded worker-owned phase transition."""

    generation: int
    attempt_id: str
    phase: WorkerPhase

    def __post_init__(self) -> None:
        _require_generation_and_attempt(self.generation, self.attempt_id)
        if type(self.phase) is not WorkerPhase:
            raise TypeError("phase must be a WorkerPhase")


@dataclass(frozen=True, slots=True)
class ExecutorResult:
    """Successful parsed-media payload from one worker attempt."""

    generation: int
    attempt_id: str
    payload: dict[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        _require_generation_and_attempt(self.generation, self.attempt_id)
        if type(self.payload) is not dict:
            raise TypeError("payload must be a dict")


@dataclass(frozen=True, slots=True)
class ExecutorFailure:
    """Bounded path-private failure from one worker attempt."""

    generation: int
    attempt_id: str
    code: TranscriptionFailureCode
    recovery_actions: tuple[str, ...] = ()
    failed_attempt: dict[str, Any] | None = field(default=None, repr=False)
    device_failure_origin: DeviceFailureOrigin | None = None

    def __post_init__(self) -> None:
        _require_generation_and_attempt(self.generation, self.attempt_id)
        if type(self.code) is not TranscriptionFailureCode:
            raise TypeError("code must be a TranscriptionFailureCode")
        if type(self.recovery_actions) is not tuple:
            raise TypeError("recovery_actions must be a tuple")
        if len(self.recovery_actions) > _MAX_RECOVERY_ACTIONS:
            raise ValueError("too many recovery actions")
        if any(
            type(action) is not str
            or not action.strip()
            or len(action) > _MAX_RECOVERY_ACTION_LENGTH
            for action in self.recovery_actions
        ):
            raise ValueError("recovery actions must be bounded non-empty strings")
        if self.failed_attempt is not None and type(self.failed_attempt) is not dict:
            raise TypeError("failed_attempt must be a dict")
        if (
            self.device_failure_origin is not None
            and type(self.device_failure_origin) is not DeviceFailureOrigin
        ):
            raise TypeError("device_failure_origin must be a DeviceFailureOrigin")


class _AttemptTerminalGuard:
    """Accept exactly one matching terminal envelope for one active attempt."""

    __slots__ = ("_attempt_id", "_consumed", "_generation")

    def __init__(self, *, generation: int, attempt_id: str) -> None:
        _require_generation_and_attempt(generation, attempt_id)
        self._generation = generation
        self._attempt_id = attempt_id
        self._consumed = False

    def accept(self, envelope: ExecutorResult | ExecutorFailure) -> bool:
        """Consume and accept a matching terminal envelope once."""

        if type(envelope) not in {ExecutorResult, ExecutorFailure}:
            return False
        if (
            self._consumed
            or envelope.generation != self._generation
            or envelope.attempt_id != self._attempt_id
        ):
            return False
        self._consumed = True
        return True


__all__ = [
    "ExecutorEvent",
    "ExecutorFailure",
    "ExecutorRequest",
    "ExecutorResult",
    "LocalSourceSnapshot",
    "ModelIdentity",
    "WorkerPhase",
]
