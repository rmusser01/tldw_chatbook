"""Strict, bounded stdin/stdout protocol for one-shot workspace workers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tldw_chatbook.Tools.git_tool_impls import GIT_MAX_OUTPUT_BYTES
from tldw_chatbook.Tools.patch_tool_impls import PATCH_MAX_BYTES, PATCH_MAX_FILES
from tldw_chatbook.Utils.filesystem_identity import DirectoryIdentity

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = GIT_MAX_OUTPUT_BYTES + 64 * 1024
MAX_STRING_BYTES = 15 * 1024 * 1024
MAX_PATH_BYTES = 16 * 1024
MAX_COLLECTION_ITEMS = 1_024
MAX_JSON_DEPTH = 16

WorkspaceOperation = Literal[
    "fs_list",
    "fs_read",
    "fs_write",
    "fs_edit",
    "fs_patch",
    "fs_glob",
    "fs_grep",
    "stat_path",
    "git_status",
    "git_diff",
    "git_log",
    "git_blame",
    "git_branches",
]
WorkspaceIntent = Literal["read", "write"]
WorkspaceResponseOutcome = Literal["admitted", "success", "failure"]

_OPERATIONS = frozenset(
    {
        "fs_list",
        "fs_read",
        "fs_write",
        "fs_edit",
        "fs_patch",
        "fs_glob",
        "fs_grep",
        "stat_path",
        "git_status",
        "git_diff",
        "git_log",
        "git_blame",
        "git_branches",
    }
)
_INTENTS = frozenset({"read", "write"})
_OUTCOMES = frozenset({"admitted", "success", "failure"})
_REQUEST_KEYS = frozenset(
    {
        "version",
        "operation_id",
        "operation",
        "intent",
        "root_locator",
        "root_identity",
        "ancestor_identities",
        "arguments",
        "timeout_seconds",
        "output_max_bytes",
    }
)
_RESPONSE_KEYS = frozenset(
    {
        "version",
        "operation_id",
        "outcome",
        "code",
        "result",
        "error",
        "elapsed_ms",
        "truncated",
        "cleanup_proven",
    }
)

_ARGUMENT_SCHEMAS: dict[str, tuple[frozenset[str], dict[str, str]]] = {
    "fs_list": (
        frozenset({"path", "sensitive_exclusions"}),
        {"path": "path", "sensitive_exclusions": "sensitive_exclusions"},
    ),
    "fs_read": (
        frozenset({"path", "sensitive_exclusions"}),
        {"path": "path", "offset": "positive_int", "limit": "nonnegative_int", "sensitive_exclusions": "sensitive_exclusions"},
    ),
    "fs_write": (
        frozenset({"path", "content", "sensitive_exclusions"}),
        {
            "path": "path",
            "content": "text",
            "sensitive_exclusions": "sensitive_exclusions",
        },
    ),
    "fs_edit": (
        frozenset({"path", "old_string", "new_string", "sensitive_exclusions"}),
        {
            "path": "path",
            "old_string": "text",
            "new_string": "text",
            "replace_all": "bool",
            "sensitive_exclusions": "sensitive_exclusions",
        },
    ),
    "fs_patch": (
        frozenset({"diff", "sensitive_exclusions"}),
        {
            "diff": "patch",
            "dry_run": "bool",
            "targets": "patch_targets",
            "sensitive_exclusions": "sensitive_exclusions",
        },
    ),
    "fs_glob": (
        frozenset({"pattern", "sensitive_exclusions"}),
        {"pattern": "glob_pattern", "max_results": "positive_int", "sensitive_exclusions": "sensitive_exclusions"},
    ),
    "fs_grep": (
        frozenset({"pattern", "sensitive_exclusions", "content_exclusions"}),
        {"pattern": "text", "mode": "grep_mode", "max_results": "positive_int", "sensitive_exclusions": "sensitive_exclusions", "content_exclusions": "sensitive_exclusions"},
    ),
    "stat_path": (frozenset({"path"}), {"path": "path"}),
    "git_status": (
        frozenset({"sensitive_exclusions"}),
        {"path": "path", "sensitive_exclusions": "sensitive_exclusions"},
    ),
    "git_diff": (
        frozenset({"sensitive_exclusions"}),
        {
            "staged": "bool",
            "commit_range": "text",
            "path": "path",
            "stat": "bool",
            "sensitive_exclusions": "sensitive_exclusions",
        },
    ),
    "git_log": (
        frozenset({"sensitive_exclusions"}),
        {
            "count": "positive_int",
            "path": "path",
            "sensitive_exclusions": "sensitive_exclusions",
        },
    ),
    "git_blame": (
        frozenset({"path", "sensitive_exclusions"}),
        {
            "path": "path",
            "start_line": "positive_int",
            "end_line": "positive_int",
            "sensitive_exclusions": "sensitive_exclusions",
        },
    ),
    "git_branches": (
        frozenset({"sensitive_exclusions"}),
        {"sensitive_exclusions": "sensitive_exclusions"},
    ),
}
_EXPECTED_INTENTS = {
    operation: ("write" if operation in {"fs_write", "fs_edit", "fs_patch"} else "read")
    for operation in _OPERATIONS
}


class _DirectoryIdentityFrame(BaseModel):
    """Strict typed directory identity at the JSON message boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    device: int
    inode: int
    mode: int
    reparse: bool


class _RequestFrame(BaseModel):
    """Strict typed request decoded before domain construction."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: int
    operation_id: str
    operation: WorkspaceOperation
    intent: WorkspaceIntent
    root_locator: str
    root_identity: _DirectoryIdentityFrame
    ancestor_identities: list[_DirectoryIdentityFrame] = Field(
        min_length=1,
        max_length=MAX_COLLECTION_ITEMS,
    )
    arguments: dict[str, Any]
    timeout_seconds: int
    output_max_bytes: int


class _ResponseFrame(BaseModel):
    """Strict typed response decoded before domain construction."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: int
    operation_id: str
    outcome: WorkspaceResponseOutcome
    code: str
    result: str | None
    error: str | None
    elapsed_ms: int
    truncated: bool
    cleanup_proven: bool


class WorkspaceProtocolError(ValueError):
    """Raised for an invalid frame without reflecting private frame content."""


@dataclass(frozen=True, slots=True)
class WorkspaceToolRequest:
    """One immutable operation admitted to a pinned workspace worker."""

    operation_id: str
    operation: WorkspaceOperation
    intent: WorkspaceIntent
    root_locator: Path = field(repr=False)
    root_identity: DirectoryIdentity
    ancestor_identities: tuple[DirectoryIdentity, ...]
    arguments: dict[str, Any] = field(repr=False)
    timeout_seconds: int
    output_max_bytes: int

    @classmethod
    def from_bytes(cls, raw: bytes) -> WorkspaceToolRequest:
        """Parse one strict bounded request frame."""
        payload = _load_object(raw, cap=MAX_REQUEST_BYTES, frame_name="request")
        _require_exact_keys(payload, _REQUEST_KEYS)
        frame = _validate_request_frame(payload)
        _require_version(frame.version)
        operation_id = _require_string(frame.operation_id, "operation_id")
        operation = _require_closed_string(frame.operation, _OPERATIONS, "operation")
        intent = _require_closed_string(frame.intent, _INTENTS, "intent")
        root_locator = _require_path(frame.root_locator, "root_locator")
        root_identity = _identity_from_frame(frame.root_identity)
        if not frame.ancestor_identities:
            raise WorkspaceProtocolError("ancestor_identities must be a non-empty array")
        if len(frame.ancestor_identities) > MAX_COLLECTION_ITEMS:
            raise WorkspaceProtocolError("ancestor_identities exceeds collection ceiling")
        ancestors = tuple(
            _identity_from_frame(value) for value in frame.ancestor_identities
        )
        if intent != _EXPECTED_INTENTS[operation]:
            raise WorkspaceProtocolError("operation intent mismatch")
        arguments = _require_arguments(frame.arguments, operation=operation)
        timeout_seconds = _require_positive_int(frame.timeout_seconds, "timeout_seconds")
        output_max_bytes = _require_positive_int(
            frame.output_max_bytes, "output_max_bytes"
        )
        return cls(
            operation_id=operation_id,
            operation=operation,  # type: ignore[arg-type]
            intent=intent,  # type: ignore[arg-type]
            root_locator=Path(root_locator),
            root_identity=root_identity,
            ancestor_identities=ancestors,
            arguments=arguments,
            timeout_seconds=timeout_seconds,
            output_max_bytes=output_max_bytes,
        )

    def to_bytes(self) -> bytes:
        """Serialize a request only after applying the same admission checks."""
        payload = {
            "version": PROTOCOL_VERSION,
            "operation_id": self.operation_id,
            "operation": self.operation,
            "intent": self.intent,
            "root_locator": str(self.root_locator),
            "root_identity": _identity_to_payload(self.root_identity),
            "ancestor_identities": [
                _identity_to_payload(identity) for identity in self.ancestor_identities
            ],
            "arguments": self.arguments,
            "timeout_seconds": self.timeout_seconds,
            "output_max_bytes": self.output_max_bytes,
        }
        validated = type(self).from_bytes(_encode_object(payload))
        return _encode_object(
            {
                "version": PROTOCOL_VERSION,
                "operation_id": validated.operation_id,
                "operation": validated.operation,
                "intent": validated.intent,
                "root_locator": str(validated.root_locator),
                "root_identity": _identity_to_payload(validated.root_identity),
                "ancestor_identities": [
                    _identity_to_payload(identity)
                    for identity in validated.ancestor_identities
                ],
                "arguments": validated.arguments,
                "timeout_seconds": validated.timeout_seconds,
                "output_max_bytes": validated.output_max_bytes,
            }
        )


@dataclass(frozen=True, slots=True)
class WorkspaceToolResponse:
    """One content-redacted worker status or terminal result frame."""

    operation_id: str
    outcome: WorkspaceResponseOutcome
    code: str
    result: str | None = field(repr=False)
    error: str | None = field(repr=False)
    elapsed_ms: int
    truncated: bool
    cleanup_proven: bool

    @classmethod
    def from_bytes(
        cls, raw: bytes, *, expected_operation_id: str | None = None
    ) -> WorkspaceToolResponse:
        """Parse one strict bounded response frame."""
        payload = _load_object(raw, cap=MAX_RESPONSE_BYTES, frame_name="response")
        _require_exact_keys(payload, _RESPONSE_KEYS)
        frame = _validate_response_frame(payload)
        _require_version(frame.version)
        operation_id = _require_string(frame.operation_id, "operation_id")
        if expected_operation_id is not None and operation_id != expected_operation_id:
            raise WorkspaceProtocolError("response operation ID mismatch")
        outcome = _require_closed_string(frame.outcome, _OUTCOMES, "outcome")
        code = _require_string(frame.code, "code")
        result = _require_optional_string(frame.result, "result")
        error = _require_optional_string(frame.error, "error")
        elapsed_ms = _require_nonnegative_int(frame.elapsed_ms, "elapsed_ms")
        if outcome == "success" and error is not None:
            raise WorkspaceProtocolError("successful response cannot contain error")
        if outcome == "failure" and result is not None:
            raise WorkspaceProtocolError("failed response cannot contain result")
        return cls(
            operation_id=operation_id,
            outcome=outcome,  # type: ignore[arg-type]
            code=code,
            result=result,
            error=error,
            elapsed_ms=elapsed_ms,
            truncated=frame.truncated,
            cleanup_proven=frame.cleanup_proven,
        )

    def to_bytes(self) -> bytes:
        """Serialize a response only after applying the same frame contract."""
        payload = {
            "version": PROTOCOL_VERSION,
            "operation_id": self.operation_id,
            "outcome": self.outcome,
            "code": self.code,
            "result": self.result,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "truncated": self.truncated,
            "cleanup_proven": self.cleanup_proven,
        }
        type(self).from_bytes(_encode_object(payload))
        return _encode_object(payload)


def _load_object(raw: bytes, *, cap: int, frame_name: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise WorkspaceProtocolError(f"{frame_name} frame must be bytes")
    if len(raw) > cap:
        raise WorkspaceProtocolError(f"{frame_name} frame exceeds byte ceiling")
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except WorkspaceProtocolError:
        raise
    except UnicodeDecodeError as error:
        raise WorkspaceProtocolError(f"{frame_name} frame is not UTF-8") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise WorkspaceProtocolError(f"{frame_name} frame is malformed") from error
    if type(value) is not dict:
        raise WorkspaceProtocolError(f"{frame_name} frame must be an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WorkspaceProtocolError("duplicate key in protocol frame")
        value[key] = item
    return value


def _reject_non_finite(value: str) -> None:
    raise WorkspaceProtocolError("non-finite JSON value")


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str]) -> None:
    if set(value) != expected:
        raise WorkspaceProtocolError("protocol frame has invalid keys")


def _validate_request_frame(payload: Mapping[str, Any]) -> _RequestFrame:
    try:
        return _RequestFrame.model_validate(payload)
    except ValidationError:
        raise WorkspaceProtocolError("request frame validation failed") from None


def _validate_response_frame(payload: Mapping[str, Any]) -> _ResponseFrame:
    try:
        return _ResponseFrame.model_validate(payload)
    except ValidationError:
        raise WorkspaceProtocolError("response frame validation failed") from None


def _require_version(value: int) -> None:
    if value != PROTOCOL_VERSION:
        raise WorkspaceProtocolError("unsupported protocol version")


def _require_string(value: Any, field_name: str, *, cap: int = MAX_STRING_BYTES) -> str:
    if type(value) is not str:
        raise WorkspaceProtocolError(f"{field_name} must be a string")
    if "\x00" in value:
        raise WorkspaceProtocolError(f"{field_name} contains NUL")
    try:
        byte_count = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise WorkspaceProtocolError(f"{field_name} is not UTF-8 encodable") from error
    if byte_count > cap:
        raise WorkspaceProtocolError(f"{field_name} exceeds byte ceiling")
    return value


def _require_optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name)


def _require_closed_string(value: Any, choices: frozenset[str], field_name: str) -> str:
    text = _require_string(value, field_name)
    if text not in choices:
        raise WorkspaceProtocolError(f"unsupported {field_name}")
    return text


def _require_path(value: Any, field_name: str) -> str:
    return _require_string(value, field_name, cap=MAX_PATH_BYTES)


def _require_positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise WorkspaceProtocolError(f"{field_name} must be a positive int")
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise WorkspaceProtocolError(f"{field_name} must be a non-negative int")
    return value


def _identity_to_payload(identity: DirectoryIdentity) -> dict[str, int | bool]:
    return {
        "device": identity.device,
        "inode": identity.inode,
        "mode": identity.mode,
        "reparse": identity.reparse,
    }


def _identity_from_frame(value: _DirectoryIdentityFrame) -> DirectoryIdentity:
    return DirectoryIdentity(
        device=_require_nonnegative_int(value.device, "identity.device"),
        inode=_require_nonnegative_int(value.inode, "identity.inode"),
        mode=_require_nonnegative_int(value.mode, "identity.mode"),
        reparse=value.reparse,
    )


def _require_arguments(value: Any, *, operation: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise WorkspaceProtocolError("arguments must be an object")
    required, accepted = _ARGUMENT_SCHEMAS[operation]
    if not required.issubset(value) or not set(value).issubset(accepted):
        raise WorkspaceProtocolError("invalid operation arguments")
    for key, argument in value.items():
        _require_argument_value(argument, kind=accepted[key])
    return value


def _require_argument_value(value: Any, *, kind: str) -> None:
    if kind == "path":
        _require_string(value, "argument path", cap=MAX_PATH_BYTES)
        return
    if kind == "text":
        _require_string(value, "argument text")
        return
    if kind == "glob_pattern":
        validate_glob_pattern(value)
        return
    if kind == "patch":
        _require_string(value, "patch diff", cap=PATCH_MAX_BYTES)
        return
    if kind == "patch_targets":
        if type(value) is not list or not value or len(value) > PATCH_MAX_FILES:
            raise WorkspaceProtocolError("invalid patch targets")
        for target in value:
            _require_path(target, "patch target")
        return
    if kind == "bool":
        if type(value) is not bool:
            raise WorkspaceProtocolError("argument must be a bool")
        return
    if kind == "positive_int":
        _require_positive_int(value, "argument")
        return
    if kind == "nonnegative_int":
        _require_nonnegative_int(value, "argument")
        return
    if kind == "grep_mode":
        mode = _require_string(value, "grep mode")
        if mode not in {"content", "files", "count"}:
            raise WorkspaceProtocolError("invalid grep mode")
        return
    if kind == "sensitive_exclusions":
        if type(value) is not list or len(value) > MAX_COLLECTION_ITEMS:
            raise WorkspaceProtocolError("invalid sensitive exclusions")
        for exclusion in value:
            if type(exclusion) is not dict or set(exclusion) != {"kind", "value"}:
                raise WorkspaceProtocolError("invalid sensitive exclusions")
            kind_value = _require_closed_string(
                exclusion["kind"],
                frozenset({"subtree", "file", "direct_children", "name"}),
                "sensitive exclusion kind",
            )
            text = _require_path(exclusion["value"], "sensitive exclusion value")
            if "\x00" in text or (kind_value == "name" and ("/" in text or "\\" in text)):
                raise WorkspaceProtocolError("invalid sensitive exclusions")
        return
    raise WorkspaceProtocolError("invalid argument schema")


def validate_glob_pattern(value: Any) -> str:
    """Validate a platform-neutral, root-relative glob grammar."""
    if type(value) is not str:
        raise WorkspaceProtocolError("glob pattern must be a string")
    if "\x00" in value:
        raise WorkspaceProtocolError("invalid glob pattern")
    pattern = _require_string(value, "glob pattern")
    windows = Path(pattern.replace("\\", "/"))
    if (
        pattern.startswith(("/", "\\"))
        or ":" in pattern.split("/")[0]
        or any(part == ".." for part in pattern.replace("\\", "/").split("/"))
        or windows.is_absolute()
    ):
        raise WorkspaceProtocolError("invalid glob pattern")
    return pattern


def _encode_object(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise WorkspaceProtocolError("protocol frame cannot be serialized") from error
