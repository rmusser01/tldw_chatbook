"""Conformance-style unit tests for the stdlib wire decoder (Phase 0a).

The worker-bundle decoder must accept/reject exactly what the parent's
pydantic serde (``WorkspaceToolRequest.from_bytes`` /
``WorkspaceToolResponse.from_bytes``) accepts/rejects. The exhaustive
conformance corpus lands with Task 3; these tests pin the decoder's
behaviour and the shared-constants seam now.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from tldw_chatbook.Tools import workspace_wire_decode as wire_module
from tldw_chatbook.Tools.git_tool_impls import GIT_MAX_OUTPUT_BYTES
from tldw_chatbook.Tools.patch_tool_impls import PATCH_MAX_BYTES, PATCH_MAX_FILES
from tldw_chatbook.Tools.workspace_tool_protocol import (
    WorkspaceProtocolError,
    WorkspaceToolRequest,
    WorkspaceToolResponse,
)
from tldw_chatbook.Tools.workspace_wire_decode import (
    MAX_COLLECTION_ITEMS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    REQUEST_FIELD_NAMES,
    RESPONSE_FIELD_NAMES,
    WIRE_VERSION,
    WireDecodeError,
    decode_request,
    decode_response,
)
from tldw_chatbook.Utils.filesystem_identity import DirectoryIdentity

RequestMutation = Callable[[dict[str, Any]], dict[str, Any]]

_ARGUMENTS_BY_OPERATION: dict[str, dict[str, Any]] = {
    "fs_list": {"path": ".", "sensitive_exclusions": []},
    "fs_read": {"path": "read.txt", "sensitive_exclusions": []},
    "fs_write": {
        "path": "write.txt",
        "content": "contents",
        "sensitive_exclusions": [],
    },
    "fs_edit": {
        "path": "edit.txt",
        "old_string": "before",
        "new_string": "after",
        "sensitive_exclusions": [],
    },
    "fs_patch": {
        "diff": "--- a/file\n+++ b/file\n",
        "sensitive_exclusions": [],
    },
    "fs_glob": {"pattern": "**/*.py", "sensitive_exclusions": []},
    "fs_grep": {
        "pattern": "needle",
        "sensitive_exclusions": [],
        "content_exclusions": [],
    },
    "stat_path": {"path": "file.txt"},
    "git_status": {"sensitive_exclusions": []},
    "git_diff": {"sensitive_exclusions": []},
    "git_log": {"sensitive_exclusions": []},
    "git_blame": {"path": "file.txt", "sensitive_exclusions": []},
    "git_branches": {"sensitive_exclusions": []},
}

_WRITE_OPERATIONS = frozenset({"fs_write", "fs_edit", "fs_patch"})


def _valid_request_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "operation_id": "operation-1",
        "operation": "fs_read",
        "intent": "read",
        "root_locator": "/private/workspace",
        "root_identity": {
            "device": 1,
            "inode": 2,
            "mode": 0o040755,
            "reparse": False,
        },
        "ancestor_identities": [
            {"device": 1, "inode": 3, "mode": 0o040755, "reparse": False},
        ],
        "arguments": {"path": "read.txt", "sensitive_exclusions": []},
        "timeout_seconds": 30,
        "output_max_bytes": 1024,
    }


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _valid_response_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "operation_id": "operation-1",
        "outcome": "success",
        "code": "ok",
        "result": "file-body",
        "error": None,
        "elapsed_ms": 12,
        "truncated": False,
        "cleanup_proven": True,
    }


# ---------------------------------------------------------------------------
# Requests: acceptance
# ---------------------------------------------------------------------------


def test_accepts_valid_request() -> None:
    doc = decode_request(_encode(_valid_request_payload()))

    assert set(doc) == set(REQUEST_FIELD_NAMES)
    assert doc["version"] == WIRE_VERSION
    assert doc["operation"] == "fs_read"
    assert doc["intent"] == "read"
    assert doc["root_identity"] == {
        "device": 1,
        "inode": 2,
        "mode": 0o040755,
        "reparse": False,
    }


@pytest.mark.parametrize("operation", sorted(_ARGUMENTS_BY_OPERATION))
def test_accepts_every_parent_serialized_request(operation: str) -> None:
    intent = "write" if operation in _WRITE_OPERATIONS else "read"
    request = WorkspaceToolRequest(
        operation_id="operation-1",
        operation=operation,  # type: ignore[arg-type]
        intent=intent,  # type: ignore[arg-type]
        root_locator=Path("/private/workspace"),
        root_identity=DirectoryIdentity(1, 2, 0o040755, False),
        ancestor_identities=(DirectoryIdentity(1, 3, 0o040755, False),),
        arguments=_ARGUMENTS_BY_OPERATION[operation],
        timeout_seconds=30,
        output_max_bytes=1024,
    )

    doc = decode_request(request.to_bytes())

    assert doc["operation"] == operation
    assert doc["intent"] == intent
    assert doc["arguments"] == _ARGUMENTS_BY_OPERATION[operation]


# ---------------------------------------------------------------------------
# Requests: field-level rejections (mirror of the pydantic strict frame)
# ---------------------------------------------------------------------------


def _drop(key: str) -> RequestMutation:
    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in payload.items() if k != key}

    return mutate


_REQUEST_MUTATIONS: tuple[tuple[str, RequestMutation], ...] = (
    ("unknown-field", lambda d: {**d, "surplus": 0}),
    ("wrong-version", lambda d: {**d, "version": 2}),
    ("version-as-string", lambda d: {**d, "version": "1"}),
    ("version-as-float", lambda d: {**d, "version": 1.0}),
    ("version-as-bool", lambda d: {**d, "version": True}),
    ("version-as-null", lambda d: {**d, "version": None}),
    ("unknown-operation", lambda d: {**d, "operation": "fs_delete"}),
    ("operation-as-int", lambda d: {**d, "operation": 7}),
    ("intent-mismatch", lambda d: {**d, "intent": "write"}),
    ("unknown-intent", lambda d: {**d, "intent": "execute"}),
    ("operation-id-as-int", lambda d: {**d, "operation_id": 9}),
    ("root-locator-as-int", lambda d: {**d, "root_locator": 5}),
    ("root-locator-nul", lambda d: {**d, "root_locator": "/tmp/\x00w"}),
    ("root-identity-nonobject", lambda d: {**d, "root_identity": 5}),
    ("root-identity-extra-key", lambda d: {**d, "root_identity": {**d["root_identity"], "surplus": 1}}),
    ("root-identity-missing-mode", lambda d: {**d, "root_identity": {k: v for k, v in d["root_identity"].items() if k != "mode"}}),
    ("root-identity-negative-device", lambda d: {**d, "root_identity": {**d["root_identity"], "device": -1}}),
    ("root-identity-bool-device", lambda d: {**d, "root_identity": {**d["root_identity"], "device": True}}),
    ("root-identity-int-reparse", lambda d: {**d, "root_identity": {**d["root_identity"], "reparse": 0}}),
    ("ancestors-empty", lambda d: {**d, "ancestor_identities": []}),
    ("ancestors-nonlist", lambda d: {**d, "ancestor_identities": {"device": 1}}),
    (
        "ancestors-oversize",
        lambda d: {
            **d,
            "ancestor_identities": [dict(d["root_identity"])] * (MAX_COLLECTION_ITEMS + 1),
        },
    ),
    (
        "ancestors-bad-item",
        lambda d: {**d, "ancestor_identities": [{"device": 1, "inode": 2, "mode": 3, "reparse": "no"}]},
    ),
    ("arguments-nonobject", lambda d: {**d, "arguments": []}),
    ("arguments-missing-required", lambda d: {**d, "arguments": {"sensitive_exclusions": []}}),
    (
        "arguments-unknown-key",
        lambda d: {**d, "arguments": {"path": "a.txt", "sensitive_exclusions": [], "blob": 1}},
    ),
    (
        "arguments-path-as-int",
        lambda d: {**d, "arguments": {"path": 3, "sensitive_exclusions": []}},
    ),
    (
        "arguments-path-nul",
        lambda d: {**d, "arguments": {"path": "a\x00b", "sensitive_exclusions": []}},
    ),
    ("timeout-zero", lambda d: {**d, "timeout_seconds": 0}),
    ("timeout-negative", lambda d: {**d, "timeout_seconds": -1}),
    ("timeout-float", lambda d: {**d, "timeout_seconds": 30.5}),
    ("timeout-bool", lambda d: {**d, "timeout_seconds": True}),
    ("timeout-string", lambda d: {**d, "timeout_seconds": "30"}),
    ("output-zero", lambda d: {**d, "output_max_bytes": 0}),
    ("output-float", lambda d: {**d, "output_max_bytes": 1024.5}),
    ("output-bool", lambda d: {**d, "output_max_bytes": False}),
)


@pytest.mark.parametrize(
    ("case", "mutate"),
    [pytest.param(case, mutate, id=case) for case, mutate in _REQUEST_MUTATIONS],
)
def test_request_rejects_mutated_fields(case: str, mutate: RequestMutation) -> None:
    del case
    with pytest.raises(WireDecodeError):
        decode_request(_encode(mutate(_valid_request_payload())))


@pytest.mark.parametrize("field", sorted(REQUEST_FIELD_NAMES))
def test_request_rejects_missing_fields(field: str) -> None:
    with pytest.raises(WireDecodeError):
        decode_request(_encode(_drop(field)(_valid_request_payload())))


# ---------------------------------------------------------------------------
# Requests: operation-argument rejections (mirror of the admission checks)
# ---------------------------------------------------------------------------


def _with_operation(
    operation: str, arguments: dict[str, Any]
) -> RequestMutation:
    intent = "write" if operation in _WRITE_OPERATIONS else "read"

    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "operation": operation, "intent": intent, "arguments": arguments}

    return mutate


_ARGUMENT_MUTATIONS: tuple[tuple[str, RequestMutation], ...] = (
    (
        "fs-read-offset-zero",
        _with_operation("fs_read", {"path": "a.txt", "offset": 0, "sensitive_exclusions": []}),
    ),
    (
        "fs-read-limit-negative",
        _with_operation("fs_read", {"path": "a.txt", "limit": -1, "sensitive_exclusions": []}),
    ),
    (
        "fs-write-dry-run-int",
        _with_operation(
            "fs_write",
            {"path": "a.txt", "content": "x", "dry_run": 0, "sensitive_exclusions": []},
        ),
    ),
    (
        "fs-write-sha256-short",
        _with_operation(
            "fs_write",
            {"path": "a.txt", "content": "x", "expected_sha256": "a" * 63, "sensitive_exclusions": []},
        ),
    ),
    (
        "fs-write-sha256-nonhex",
        _with_operation(
            "fs_write",
            {"path": "a.txt", "content": "x", "expected_sha256": "z" * 64, "sensitive_exclusions": []},
        ),
    ),
    (
        "fs-edit-replace-all-int",
        _with_operation(
            "fs_edit",
            {"path": "a.txt", "old_string": "a", "new_string": "b", "replace_all": 1, "sensitive_exclusions": []},
        ),
    ),
    (
        "fs-patch-diff-int",
        _with_operation("fs_patch", {"diff": 5, "sensitive_exclusions": []}),
    ),
    (
        "fs-patch-empty-targets",
        _with_operation(
            "fs_patch",
            {"diff": "--- a/x\n+++ b/x\n", "targets": [], "sensitive_exclusions": []},
        ),
    ),
    (
        "fs-glob-absolute-pattern",
        _with_operation("fs_glob", {"pattern": "/etc/*", "sensitive_exclusions": []}),
    ),
    (
        "fs-glob-parent-pattern",
        _with_operation("fs_glob", {"pattern": "../secret", "sensitive_exclusions": []}),
    ),
    (
        "fs-glob-max-results-zero",
        _with_operation("fs_glob", {"pattern": "*.py", "max_results": 0, "sensitive_exclusions": []}),
    ),
    (
        "fs-grep-unknown-mode",
        _with_operation(
            "fs_grep",
            {"pattern": "n", "mode": "lines", "sensitive_exclusions": [], "content_exclusions": []},
        ),
    ),
    (
        "sensitive-exclusion-unknown-kind",
        _with_operation(
            "fs_read",
            {"path": "a.txt", "sensitive_exclusions": [{"kind": "evil", "value": "x"}]},
        ),
    ),
    (
        "sensitive-exclusion-name-with-slash",
        _with_operation(
            "fs_read",
            {"path": "a.txt", "sensitive_exclusions": [{"kind": "name", "value": "a/b"}]},
        ),
    ),
    (
        "git-log-count-zero",
        _with_operation("git_log", {"count": 0, "sensitive_exclusions": []}),
    ),
    (
        "git-blame-missing-path",
        _with_operation("git_blame", {"sensitive_exclusions": []}),
    ),
)


@pytest.mark.parametrize(
    ("case", "mutate"),
    [pytest.param(case, mutate, id=case) for case, mutate in _ARGUMENT_MUTATIONS],
)
def test_request_rejects_invalid_operation_arguments(
    case: str, mutate: RequestMutation
) -> None:
    del case
    with pytest.raises(WireDecodeError):
        decode_request(_encode(mutate(_valid_request_payload())))


# ---------------------------------------------------------------------------
# Requests: encoding-level rejections (strict JSON rules from TASK-32855)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(
            lambda b: b.replace(b'{"version":1,', b'{"version":1,"surplus":0,', 1),
            id="unknown-field",
        ),
        pytest.param(
            lambda b: b.replace(b'"version":1,', b'"version":2,', 1),
            id="wrong-version",
        ),
        pytest.param(
            lambda b: b.replace(b'"version":1,', b'"version":1,"version":2,', 1),
            id="duplicate-keys",
        ),
        pytest.param(
            lambda b: b.replace(b'"timeout_seconds":30', b'"timeout_seconds":NaN', 1),
            id="nan",
        ),
        pytest.param(
            lambda b: b.replace(b'"output_max_bytes":1024', b'"output_max_bytes":Infinity', 1),
            id="infinity",
        ),
        pytest.param(
            lambda b: b.replace(b'"fs_read"', b'"\xff\xfe"', 1),
            id="invalid-utf8",
        ),
        pytest.param(lambda b: b"[1,2,3]", id="not-an-object"),
        pytest.param(lambda b: b"", id="empty"),
        pytest.param(lambda b: b + b"{}", id="trailing-garbage"),
        pytest.param(
            lambda b: b"x" * (MAX_REQUEST_BYTES + 1),
            id="oversize",
        ),
    ],
)
def test_request_rejects_malformed_bytes(mangle: Callable[[bytes], bytes]) -> None:
    with pytest.raises(WireDecodeError):
        decode_request(mangle(_encode(_valid_request_payload())))


def test_request_rejects_non_bytes_input() -> None:
    with pytest.raises(WireDecodeError):
        decode_request('{"version": 1}')  # type: ignore[arg-type]
    with pytest.raises(WireDecodeError):
        decode_request(bytearray(_encode(_valid_request_payload())))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Parent/decoder agreement (mini conformance smoke; corpus lands in Task 3)
# ---------------------------------------------------------------------------


_AGREEMENT_CASES = frozenset(
    {
        "unknown-field",
        "wrong-version",
        "version-as-bool",
        "timeout-bool",
        "intent-mismatch",
        "root-identity-extra-key",
        "ancestors-empty",
    }
)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(mutate, id=case)
        for case, mutate in _REQUEST_MUTATIONS
        if case in _AGREEMENT_CASES
    ],
)
def test_rejections_agree_with_parent(mutate: RequestMutation) -> None:
    raw = _encode(mutate(_valid_request_payload()))

    with pytest.raises(WorkspaceProtocolError):
        WorkspaceToolRequest.from_bytes(raw)
    with pytest.raises(WireDecodeError):
        decode_request(raw)


def test_valid_request_agrees_with_parent() -> None:
    raw = _encode(_valid_request_payload())

    assert WorkspaceToolRequest.from_bytes(raw).operation == "fs_read"
    assert decode_request(raw)["operation"] == "fs_read"


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def test_accepts_valid_success_response() -> None:
    doc = decode_response(_encode(_valid_response_payload()))

    assert set(doc) == set(RESPONSE_FIELD_NAMES)
    assert doc["outcome"] == "success"
    assert doc["result"] == "file-body"


def test_accepts_failure_response() -> None:
    payload = _valid_response_payload() | {
        "outcome": "failure",
        "result": None,
        "error": "boom",
    }

    assert decode_response(_encode(payload))["error"] == "boom"


def test_accepts_admitted_response() -> None:
    payload = _valid_response_payload() | {
        "outcome": "admitted",
        "code": "pin_ok",
        "result": None,
        "error": None,
    }

    assert decode_response(_encode(payload))["outcome"] == "admitted"


@pytest.mark.parametrize(
    ("outcome", "result", "error"),
    [
        ("admitted", None, None),
        ("success", "body", None),
        ("failure", None, "boom"),
    ],
)
def test_accepts_every_parent_serialized_response(
    outcome: str, result: str | None, error: str | None
) -> None:
    response = WorkspaceToolResponse(
        operation_id="operation-1",
        outcome=outcome,  # type: ignore[arg-type]
        code="ok",
        result=result,
        error=error,
        elapsed_ms=12,
        truncated=False,
        cleanup_proven=True,
    )

    assert decode_response(response.to_bytes())["operation_id"] == "operation-1"


ResponseMutation = Callable[[dict[str, Any]], dict[str, Any]]

_RESPONSE_MUTATIONS: tuple[tuple[str, ResponseMutation], ...] = (
    ("unknown-field", lambda d: {**d, "surplus": 0}),
    ("wrong-version", lambda d: {**d, "version": 2}),
    ("version-as-bool", lambda d: {**d, "version": True}),
    ("unknown-outcome", lambda d: {**d, "outcome": "pending"}),
    ("code-as-null", lambda d: {**d, "code": None}),
    ("result-as-int", lambda d: {**d, "result": 5}),
    ("error-as-int", lambda d: {**d, "error": 5}),
    ("elapsed-negative", lambda d: {**d, "elapsed_ms": -1}),
    ("elapsed-float", lambda d: {**d, "elapsed_ms": 1.5}),
    ("elapsed-bool", lambda d: {**d, "elapsed_ms": True}),
    ("truncated-int", lambda d: {**d, "truncated": 0}),
    ("cleanup-proven-int", lambda d: {**d, "cleanup_proven": 1}),
    ("success-with-error", lambda d: {**d, "error": "boom"}),
    (
        "failure-with-result",
        lambda d: {**d, "outcome": "failure", "error": "boom"},
    ),
)


@pytest.mark.parametrize(
    ("case", "mutate"),
    [pytest.param(case, mutate, id=case) for case, mutate in _RESPONSE_MUTATIONS],
)
def test_response_rejects_mutated_fields(
    case: str, mutate: ResponseMutation
) -> None:
    del case
    with pytest.raises(WireDecodeError):
        decode_response(_encode(mutate(_valid_response_payload())))


@pytest.mark.parametrize("field", sorted(RESPONSE_FIELD_NAMES))
def test_response_rejects_missing_fields(field: str) -> None:
    with pytest.raises(WireDecodeError):
        decode_response(_encode(_drop(field)(_valid_response_payload())))


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(
            lambda b: b.replace(b'"version":1,', b'"version":1,"version":2,', 1),
            id="duplicate-keys",
        ),
        pytest.param(
            lambda b: b.replace(b'"elapsed_ms":12', b'"elapsed_ms":NaN', 1),
            id="nan",
        ),
        pytest.param(
            lambda b: b.replace(b'"operation-1"', b'"\xff\xfe"', 1),
            id="invalid-utf8",
        ),
        pytest.param(lambda b: b"[1,2,3]", id="not-an-object"),
        pytest.param(lambda b: b"x" * (MAX_RESPONSE_BYTES + 1), id="oversize"),
    ],
)
def test_response_rejects_malformed_bytes(mangle: Callable[[bytes], bytes]) -> None:
    payload = _encode(_valid_response_payload())
    with pytest.raises(WireDecodeError):
        decode_response(mangle(payload))


def test_response_rejects_non_bytes_input() -> None:
    with pytest.raises(WireDecodeError):
        decode_response('{"version": 1}')  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Shared-constant parity with the live domain caps
# ---------------------------------------------------------------------------


def test_wire_cap_literals_match_live_domain_constants() -> None:
    """Pin the decoder's mirrored cap literals to the live domain constants.

    The decoder must not import tldw_chatbook modules (worker bundle), so
    the patch/git caps are mirrored as literals. Without this guard a
    domain-cap bump would silently split the parent/decoder accept/reject
    boundary for large patches or oversized responses.
    """
    assert wire_module._PATCH_MAX_BYTES == PATCH_MAX_BYTES
    assert wire_module._PATCH_MAX_FILES == PATCH_MAX_FILES
    assert wire_module._GIT_MAX_OUTPUT_BYTES == GIT_MAX_OUTPUT_BYTES
    assert wire_module.MAX_RESPONSE_BYTES == (GIT_MAX_OUTPUT_BYTES * 6) + (64 * 1024)


# ---------------------------------------------------------------------------
# Worker-bundle import closure
# ---------------------------------------------------------------------------


def test_module_imports_stdlib_only() -> None:
    module_path = Path(wire_module.__file__)
    probe = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('workspace_wire_decode', {str(module_path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "offenders = sorted(\n"
        "    name\n"
        "    for name in sys.modules\n"
        "    if name == 'tldw_chatbook'\n"
        "    or name.startswith('tldw_chatbook.')\n"
        "    or name.split('.')[0] in {'pydantic', 'loguru'}\n"
        ")\n"
        "assert not offenders, offenders\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
