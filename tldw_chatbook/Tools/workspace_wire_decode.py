"""Stdlib-only wire decoder for the pinned workspace worker.

This module is the worker-bundle counterpart of the parent's pydantic
serde in ``Tools/workspace_tool_protocol.py``. The parent keeps pydantic;
the bundle ships this module alone, so it must import nothing outside the
standard library (the bundle also targets Python 3.10).

Accept/reject behaviour must stay identical to
``WorkspaceToolRequest.from_bytes`` / ``WorkspaceToolResponse.from_bytes``
(the Task 3 conformance corpus is the gate). The shared wire constants
defined here are imported by the protocol module so the two sides cannot
drift structurally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

WIRE_VERSION = 1

# Mirrors ``Tools/git_tool_impls.GIT_MAX_OUTPUT_BYTES``. Kept as a literal
# because this module must not import tldw_chatbook code; the literals are
# pinned to the live domain constants by
# ``Tests/Tools/test_workspace_wire_decode.py::test_wire_cap_literals_match_live_domain_constants``.
_GIT_MAX_OUTPUT_BYTES = 1_000_000
# Mirrors ``Tools/patch_tool_impls.PATCH_MAX_BYTES`` / ``PATCH_MAX_FILES``
# (same literal-only rule as above).
_PATCH_MAX_BYTES = 256 * 1024
_PATCH_MAX_FILES = 20

MAX_REQUEST_BYTES = 16 * 1024 * 1024
# JSON may encode one control byte as six ASCII bytes (``\u00xx``). Keep the
# frame bounded while guaranteeing that any result accepted by Git's raw byte
# ceiling plus fixed protocol metadata can cross the response boundary.
MAX_RESPONSE_BYTES = (_GIT_MAX_OUTPUT_BYTES * 6) + (64 * 1024)
MAX_STRING_BYTES = 15 * 1024 * 1024
MAX_PATH_BYTES = 16 * 1024
MAX_COLLECTION_ITEMS = 1_024

REQUEST_FIELD_NAMES = (
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
)
RESPONSE_FIELD_NAMES = (
    "version",
    "operation_id",
    "outcome",
    "code",
    "result",
    "error",
    "elapsed_ms",
    "truncated",
    "cleanup_proven",
)
DIRECTORY_IDENTITY_FIELD_NAMES = (
    "device",
    "inode",
    "mode",
    "reparse",
)

WORKSPACE_OPERATIONS = frozenset(
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
WORKSPACE_WRITE_OPERATIONS = frozenset({"fs_write", "fs_edit", "fs_patch"})
WORKSPACE_INTENTS = frozenset({"read", "write"})
WORKSPACE_OUTCOMES = frozenset({"admitted", "success", "failure"})

# operation -> (required argument keys, accepted key -> value kind)
ARGUMENT_SCHEMAS: dict[str, tuple[frozenset[str], dict[str, str]]] = {
    "fs_list": (
        frozenset({"path", "sensitive_exclusions"}),
        {"path": "path", "sensitive_exclusions": "sensitive_exclusions"},
    ),
    "fs_read": (
        frozenset({"path", "sensitive_exclusions"}),
        {
            "path": "path",
            "offset": "positive_int",
            "limit": "nonnegative_int",
            "sensitive_exclusions": "sensitive_exclusions",
        },
    ),
    "fs_write": (
        frozenset({"path", "content", "sensitive_exclusions"}),
        {
            "path": "path",
            "content": "text",
            "dry_run": "bool",
            "expected_sha256": "sha256",
            "expected_absent": "bool",
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
        {
            "pattern": "glob_pattern",
            "max_results": "positive_int",
            "sensitive_exclusions": "sensitive_exclusions",
        },
    ),
    "fs_grep": (
        frozenset({"pattern", "sensitive_exclusions", "content_exclusions"}),
        {
            "pattern": "text",
            "mode": "grep_mode",
            "max_results": "positive_int",
            "sensitive_exclusions": "sensitive_exclusions",
            "content_exclusions": "sensitive_exclusions",
        },
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

_REQUEST_KEYS = frozenset(REQUEST_FIELD_NAMES)
_RESPONSE_KEYS = frozenset(RESPONSE_FIELD_NAMES)
_IDENTITY_KEYS = frozenset(DIRECTORY_IDENTITY_FIELD_NAMES)
_EXPECTED_INTENTS = {
    operation: ("write" if operation in WORKSPACE_WRITE_OPERATIONS else "read")
    for operation in WORKSPACE_OPERATIONS
}
_SENSITIVE_EXCLUSION_KINDS = frozenset(
    {"subtree", "file", "direct_children", "name"}
)
_GREP_MODES = frozenset({"content", "files", "count"})
_SHA256_ALPHABET = frozenset("0123456789abcdef")


class WireDecodeError(ValueError):
    """Raised for a frame this decoder refuses.

    Messages never reflect frame content, mirroring the hygiene of the
    parent's ``WorkspaceProtocolError``.
    """


def decode_request(raw: bytes) -> dict[str, Any]:
    """Decode one strict bounded request frame.

    Args:
        raw: The frame bytes exactly as sent by the parent.

    Returns:
        The validated payload dict; field names are identical to the
        parent's ``_RequestFrame`` model.

    Raises:
        WireDecodeError: For any frame ``WorkspaceToolRequest.from_bytes``
            would refuse.
    """
    doc = _load_object(raw, cap=MAX_REQUEST_BYTES, frame_name="request")
    if set(doc) != _REQUEST_KEYS:
        raise WireDecodeError("protocol frame has invalid keys")
    version = doc["version"]
    if type(version) is not int or version != WIRE_VERSION:
        raise WireDecodeError("unsupported protocol version")
    _require_string(doc["operation_id"], "operation_id")
    operation = _require_closed_string(
        doc["operation"], WORKSPACE_OPERATIONS, "operation"
    )
    intent = _require_closed_string(doc["intent"], WORKSPACE_INTENTS, "intent")
    _require_path(doc["root_locator"], "root_locator")
    _validate_directory_identity(doc["root_identity"], "root_identity")
    ancestors = doc["ancestor_identities"]
    if type(ancestors) is not list:
        raise WireDecodeError("ancestor_identities must be an array")
    if not ancestors:
        raise WireDecodeError("ancestor_identities must be a non-empty array")
    if len(ancestors) > MAX_COLLECTION_ITEMS:
        raise WireDecodeError("ancestor_identities exceeds collection ceiling")
    for ancestor in ancestors:
        _validate_directory_identity(ancestor, "ancestor_identities")
    if intent != _EXPECTED_INTENTS[operation]:
        raise WireDecodeError("operation intent mismatch")
    _validate_arguments(doc["arguments"], operation=operation)
    _require_positive_int(doc["timeout_seconds"], "timeout_seconds")
    _require_positive_int(doc["output_max_bytes"], "output_max_bytes")
    return doc


def decode_response(raw: bytes) -> dict[str, Any]:
    """Decode one strict bounded response frame.

    Args:
        raw: The frame bytes exactly as emitted by a worker.

    Returns:
        The validated payload dict; field names are identical to the
        parent's ``_ResponseFrame`` model.

    Raises:
        WireDecodeError: For any frame ``WorkspaceToolResponse.from_bytes``
            would refuse (absent the optional ``expected_operation_id``
            caller-side assertion, which is not frame validity).
    """
    doc = _load_object(raw, cap=MAX_RESPONSE_BYTES, frame_name="response")
    if set(doc) != _RESPONSE_KEYS:
        raise WireDecodeError("protocol frame has invalid keys")
    version = doc["version"]
    if type(version) is not int or version != WIRE_VERSION:
        raise WireDecodeError("unsupported protocol version")
    _require_string(doc["operation_id"], "operation_id")
    outcome = _require_closed_string(
        doc["outcome"], WORKSPACE_OUTCOMES, "outcome"
    )
    _require_string(doc["code"], "code")
    result = _require_optional_string(doc["result"], "result")
    error = _require_optional_string(doc["error"], "error")
    _require_nonnegative_int(doc["elapsed_ms"], "elapsed_ms")
    if type(doc["truncated"]) is not bool:
        raise WireDecodeError("truncated must be a bool")
    if type(doc["cleanup_proven"]) is not bool:
        raise WireDecodeError("cleanup_proven must be a bool")
    if outcome == "success" and error is not None:
        raise WireDecodeError("successful response cannot contain error")
    if outcome == "failure" and result is not None:
        raise WireDecodeError("failed response cannot contain result")
    return doc


def _load_object(raw: bytes, *, cap: int, frame_name: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise WireDecodeError(f"{frame_name} frame must be bytes")
    if len(raw) > cap:
        raise WireDecodeError(f"{frame_name} frame exceeds byte ceiling")
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except WireDecodeError:
        raise
    except UnicodeDecodeError as error:
        raise WireDecodeError(f"{frame_name} frame is not UTF-8") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise WireDecodeError(f"{frame_name} frame is malformed") from error
    if type(value) is not dict:
        raise WireDecodeError(f"{frame_name} frame must be an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WireDecodeError("duplicate key in protocol frame")
        value[key] = item
    return value


def _reject_non_finite(value: str) -> None:
    raise WireDecodeError("non-finite JSON value")


def _require_string(
    value: Any, field_name: str, *, cap: int = MAX_STRING_BYTES
) -> str:
    if type(value) is not str:
        raise WireDecodeError(f"{field_name} must be a string")
    if "\x00" in value:
        raise WireDecodeError(f"{field_name} contains NUL")
    try:
        byte_count = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise WireDecodeError(f"{field_name} is not UTF-8 encodable") from error
    if byte_count > cap:
        raise WireDecodeError(f"{field_name} exceeds byte ceiling")
    return value


def _require_optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name)


def _require_closed_string(
    value: Any, choices: frozenset[str], field_name: str
) -> str:
    text = _require_string(value, field_name)
    if text not in choices:
        raise WireDecodeError(f"unsupported {field_name}")
    return text


def _require_path(value: Any, field_name: str) -> str:
    return _require_string(value, field_name, cap=MAX_PATH_BYTES)


def _require_positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise WireDecodeError(f"{field_name} must be a positive int")
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise WireDecodeError(f"{field_name} must be a non-negative int")
    return value


def _validate_directory_identity(value: Any, field_name: str) -> None:
    if type(value) is not dict:
        raise WireDecodeError(f"{field_name} must be an object")
    if set(value) != _IDENTITY_KEYS:
        raise WireDecodeError(f"{field_name} has invalid keys")
    for key in ("device", "inode", "mode"):
        _require_nonnegative_int(value[key], f"{field_name}.{key}")
    if type(value["reparse"]) is not bool:
        raise WireDecodeError(f"{field_name}.reparse must be a bool")


def _validate_arguments(value: Any, *, operation: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise WireDecodeError("arguments must be an object")
    required, accepted = ARGUMENT_SCHEMAS[operation]
    if not required.issubset(value) or not set(value).issubset(accepted):
        raise WireDecodeError("invalid operation arguments")
    for key, argument in value.items():
        _validate_argument_value(argument, kind=accepted[key])
    return value


def _validate_argument_value(value: Any, *, kind: str) -> None:
    if kind == "path":
        _require_string(value, "argument path", cap=MAX_PATH_BYTES)
        return
    if kind == "text":
        _require_string(value, "argument text")
        return
    if kind == "glob_pattern":
        _validate_glob_pattern(value)
        return
    if kind == "patch":
        _require_string(value, "patch diff", cap=_PATCH_MAX_BYTES)
        return
    if kind == "patch_targets":
        if type(value) is not list or not value or len(value) > _PATCH_MAX_FILES:
            raise WireDecodeError("invalid patch targets")
        for target in value:
            _require_path(target, "patch target")
        return
    if kind == "bool":
        if type(value) is not bool:
            raise WireDecodeError("argument must be a bool")
        return
    if kind == "sha256":
        digest = _require_string(value, "SHA-256 digest", cap=64)
        if len(digest) != 64 or any(
            character not in _SHA256_ALPHABET for character in digest
        ):
            raise WireDecodeError("invalid SHA-256 digest")
        return
    if kind == "positive_int":
        _require_positive_int(value, "argument")
        return
    if kind == "nonnegative_int":
        _require_nonnegative_int(value, "argument")
        return
    if kind == "grep_mode":
        mode = _require_string(value, "grep mode")
        if mode not in _GREP_MODES:
            raise WireDecodeError("invalid grep mode")
        return
    if kind == "sensitive_exclusions":
        if type(value) is not list or len(value) > MAX_COLLECTION_ITEMS:
            raise WireDecodeError("invalid sensitive exclusions")
        for exclusion in value:
            if type(exclusion) is not dict or set(exclusion) != {"kind", "value"}:
                raise WireDecodeError("invalid sensitive exclusions")
            kind_value = _require_closed_string(
                exclusion["kind"],
                _SENSITIVE_EXCLUSION_KINDS,
                "sensitive exclusion kind",
            )
            text = _require_path(exclusion["value"], "sensitive exclusion value")
            if "\x00" in text or (
                kind_value == "name" and ("/" in text or "\\" in text)
            ):
                raise WireDecodeError("invalid sensitive exclusions")
        return
    raise WireDecodeError("invalid argument schema")


def _validate_glob_pattern(value: Any) -> str:
    if type(value) is not str:
        raise WireDecodeError("glob pattern must be a string")
    if "\x00" in value:
        raise WireDecodeError("invalid glob pattern")
    pattern = _require_string(value, "glob pattern")
    windows = Path(pattern.replace("\\", "/"))
    if (
        pattern.startswith(("/", "\\"))
        or ":" in pattern.split("/")[0]
        or any(part == ".." for part in pattern.replace("\\", "/").split("/"))
        or windows.is_absolute()
    ):
        raise WireDecodeError("invalid glob pattern")
    return pattern
