from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.Tools.workspace_tool_protocol import (
    MAX_PATH_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_STRING_BYTES,
    PROTOCOL_VERSION,
    WorkspaceProtocolError,
    WorkspaceToolRequest,
    WorkspaceToolResponse,
)
from tldw_chatbook.Tools.patch_tool_impls import PATCH_MAX_BYTES
from tldw_chatbook.Utils.filesystem_identity import DirectoryIdentity


def _request(**changes: object) -> WorkspaceToolRequest:
    values: dict[str, object] = {
        "operation_id": "operation-1",
        "operation": "stat_path",
        "intent": "read",
        "root_locator": Path("/private/workspace"),
        "root_identity": DirectoryIdentity(1, 2, 0o040755, False),
        "ancestor_identities": (DirectoryIdentity(1, 3, 0o040755, False),),
        "arguments": {"path": "src/app.py"},
        "timeout_seconds": 30,
        "output_max_bytes": 1024,
    }
    values.update(changes)
    return WorkspaceToolRequest(**values)  # type: ignore[arg-type]


def _payload(request: WorkspaceToolRequest) -> dict[str, object]:
    return json.loads(request.to_bytes())


def _arguments_for(operation: str) -> dict[str, object]:
    return {
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
        "fs_grep": {"pattern": "needle", "sensitive_exclusions": [], "content_exclusions": []},
        "stat_path": {"path": "file.txt"},
        "git_status": {},
        "git_diff": {},
        "git_log": {},
        "git_blame": {"path": "file.txt"},
        "git_branches": {},
    }[operation]


def test_request_round_trip_uses_the_closed_exact_schema() -> None:
    request = _request()

    raw = request.to_bytes()
    payload = json.loads(raw)

    assert set(payload) == {
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
    assert payload["version"] == PROTOCOL_VERSION
    assert WorkspaceToolRequest.from_bytes(raw) == request


@pytest.mark.parametrize(
    ("operation", "intent"),
    [
        ("fs_list", "read"),
        ("fs_read", "read"),
        ("fs_write", "write"),
        ("fs_edit", "write"),
        ("fs_patch", "write"),
        ("fs_glob", "read"),
        ("fs_grep", "read"),
        ("stat_path", "read"),
        ("git_status", "read"),
        ("git_diff", "read"),
        ("git_log", "read"),
        ("git_blame", "read"),
        ("git_branches", "read"),
    ],
)
def test_request_accepts_each_closed_operation_and_intent(
    operation: str, intent: str
) -> None:
    request = _request(
        operation=operation, intent=intent, arguments=_arguments_for(operation)
    )

    assert WorkspaceToolRequest.from_bytes(request.to_bytes()) == request


def test_request_encodes_directory_identities_without_paths() -> None:
    payload = _payload(_request())

    assert payload["root_identity"] == {
        "device": 1,
        "inode": 2,
        "mode": 0o040755,
        "reparse": False,
    }
    assert payload["ancestor_identities"] == [
        {"device": 1, "inode": 3, "mode": 0o040755, "reparse": False}
    ]


def test_duplicate_json_keys_fail_before_request_construction() -> None:
    with pytest.raises(WorkspaceProtocolError, match="duplicate key"):
        WorkspaceToolRequest.from_bytes(b'{"version":1,"version":1}')


@pytest.mark.parametrize("raw", [b'{"version":NaN}', b'{"version":Infinity}'])
def test_non_finite_json_values_are_rejected(raw: bytes) -> None:
    with pytest.raises(WorkspaceProtocolError, match="non-finite"):
        WorkspaceToolRequest.from_bytes(raw)


@pytest.mark.parametrize("field", ["operation_id", "root_locator"])
def test_request_rejects_nul_in_private_strings(field: str) -> None:
    payload = _payload(_request())
    payload[field] = "unsafe\u0000value"

    with pytest.raises(WorkspaceProtocolError, match="NUL"):
        WorkspaceToolRequest.from_bytes(json.dumps(payload).encode())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("operation_id", 1),
        ("operation", "shell"),
        ("intent", "execute"),
        ("root_locator", False),
        ("root_identity", []),
        ("ancestor_identities", {}),
        ("arguments", []),
        ("timeout_seconds", 1.5),
        ("output_max_bytes", True),
    ],
)
def test_request_rejects_wrong_types_and_closed_values(field: str, value: object) -> None:
    payload = _payload(_request())
    payload[field] = value

    with pytest.raises(WorkspaceProtocolError):
        WorkspaceToolRequest.from_bytes(json.dumps(payload).encode())


def test_request_rejects_unknown_or_missing_keys() -> None:
    payload = _payload(_request())
    payload["unknown"] = "value"

    with pytest.raises(WorkspaceProtocolError, match="keys"):
        WorkspaceToolRequest.from_bytes(json.dumps(payload).encode())

    payload.pop("unknown")
    payload.pop("intent")
    with pytest.raises(WorkspaceProtocolError, match="keys"):
        WorkspaceToolRequest.from_bytes(json.dumps(payload).encode())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation_id", "a" * (MAX_STRING_BYTES + 1)),
        ("root_locator", "/" + "a" * MAX_PATH_BYTES),
        ("arguments", {"path": "a" * (MAX_PATH_BYTES + 1)}),
    ],
)
def test_request_enforces_string_and_path_ceilings(field: str, value: object) -> None:
    payload = _payload(_request())
    payload[field] = value

    with pytest.raises(WorkspaceProtocolError, match="exceeds"):
        WorkspaceToolRequest.from_bytes(json.dumps(payload).encode())


def test_request_enforces_frame_ceiling_before_json_parse() -> None:
    with pytest.raises(WorkspaceProtocolError, match="request frame exceeds"):
        WorkspaceToolRequest.from_bytes(b" " * (MAX_REQUEST_BYTES + 1))


def test_request_rejects_write_content_for_read_only_operation() -> None:
    payload = _payload(_request())
    payload["operation"] = "fs_read"
    payload["intent"] = "read"
    payload["arguments"] = {"path": "read.txt", "content": "private text"}

    with pytest.raises(WorkspaceProtocolError, match="arguments"):
        WorkspaceToolRequest.from_bytes(json.dumps(payload).encode())


@pytest.mark.parametrize(
    "pattern",
    (
        "../outside/*.txt",
        "/outside/*.txt",
        r"C:\\outside\\*.txt",
        r"\\\\host\\share\\*.txt",
        "safe\x00name/*.txt",
    ),
)
def test_protocol_rejects_unsafe_glob_patterns_before_worker_dispatch(
    pattern: str,
) -> None:
    payload = _payload(
        _request(
            operation="fs_glob", intent="read", arguments=_arguments_for("fs_glob")
        )
    )
    payload["arguments"] = {"pattern": pattern, "sensitive_exclusions": []}

    with pytest.raises(WorkspaceProtocolError, match="glob pattern"):
        WorkspaceToolRequest.from_bytes(json.dumps(payload).encode())


def test_patch_diff_uses_the_utf8_patch_byte_ceiling() -> None:
    payload = _payload(
        _request(
            operation="fs_patch",
            intent="write",
            arguments=_arguments_for("fs_patch"),
        )
    )
    payload["arguments"] = {
        "diff": "é" * (PATCH_MAX_BYTES // 2),
        "sensitive_exclusions": [],
    }

    assert WorkspaceToolRequest.from_bytes(json.dumps(payload).encode())

    payload["arguments"] = {
        "diff": "é" * ((PATCH_MAX_BYTES // 2) + 1),
        "sensitive_exclusions": [],
    }
    with pytest.raises(WorkspaceProtocolError, match="exceeds"):
        WorkspaceToolRequest.from_bytes(json.dumps(payload).encode())


def test_protocol_errors_do_not_echo_model_controlled_argument_keys() -> None:
    secret_key = "/private/secret-root"
    payload = _payload(_request())
    payload["arguments"] = {secret_key: "contains\u0000nul"}

    with pytest.raises(WorkspaceProtocolError) as exc_info:
        WorkspaceToolRequest.from_bytes(json.dumps(payload).encode())

    assert secret_key not in str(exc_info.value)


def test_huge_json_integer_is_normalized_to_protocol_error() -> None:
    raw = b'{"version":' + (b"9" * 5_000) + b"}"

    with pytest.raises(WorkspaceProtocolError, match="malformed"):
        WorkspaceToolRequest.from_bytes(raw)


def test_response_round_trip_requires_matching_operation_id() -> None:
    response = WorkspaceToolResponse(
        operation_id="operation-1",
        outcome="success",
        code="ok",
        result="private result",
        error=None,
        elapsed_ms=12,
        truncated=False,
        cleanup_proven=True,
    )

    assert WorkspaceToolResponse.from_bytes(
        response.to_bytes(), expected_operation_id="operation-1"
    ) == response
    with pytest.raises(WorkspaceProtocolError, match="operation ID"):
        WorkspaceToolResponse.from_bytes(
            response.to_bytes(), expected_operation_id="different"
        )


@pytest.mark.parametrize("raw", [b"", b"[]", b"{", b'{"version": 1}'])
def test_response_rejects_malformed_frames(raw: bytes) -> None:
    with pytest.raises(WorkspaceProtocolError):
        WorkspaceToolResponse.from_bytes(raw, expected_operation_id="operation-1")


def test_response_enforces_frame_ceiling_and_redacts_repr_and_errors() -> None:
    secret_root = "/private/secret-workspace"
    response = WorkspaceToolResponse(
        operation_id="operation-1",
        outcome="failure",
        code="root_mismatch",
        result=None,
        error=f"request failed for {secret_root}",
        elapsed_ms=1,
        truncated=False,
        cleanup_proven=True,
    )

    assert secret_root not in repr(response)
    with pytest.raises(WorkspaceProtocolError) as exc_info:
        WorkspaceToolResponse.from_bytes(
            b" " * (MAX_RESPONSE_BYTES + 1), expected_operation_id="operation-1"
        )
    assert secret_root not in str(exc_info.value)
