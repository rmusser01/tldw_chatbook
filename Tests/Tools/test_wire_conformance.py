"""The one-strict-JSON acceptance contract (TASK-32855 / ADR-175) must hold
on BOTH decoders: the parent's pydantic acceptance
(``WorkspaceToolRequest.from_bytes`` / ``WorkspaceToolResponse.from_bytes``)
and the worker bundle's stdlib decoder (``decode_request`` /
``decode_response``) reach identical accept/reject verdicts on every frame
in ``CORPUS`` / ``RESPONSE_CORPUS``.

Corpus construction rules:

- Valid frames are built from the live pydantic models (``to_bytes()``)
  so the corpus cannot go stale against model semantics; malformed frames
  are derived from those same live frames (parse, mutate one thing,
  re-encode) or hand-written raw bytes for encoding-level rules.
- Naming convention: entries whose id starts with ``valid`` MUST be
  accepted by both sides; every other entry MUST be rejected by both
  sides. Acceptance equality alone would let a shared regression (both
  sides wrongly rejecting a valid frame) pass silently.
- The parent is the contract. A mismatch is always fixed in
  ``Tools/workspace_wire_decode.py``, never in
  ``Tools/workspace_tool_protocol.py``.

``CORPUS`` and ``RESPONSE_CORPUS`` are importable by later drift guards.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tldw_chatbook.Tools.workspace_tool_protocol import (
    WorkspaceProtocolError,
    WorkspaceToolRequest,
    WorkspaceToolResponse,
)
from tldw_chatbook.Tools.workspace_wire_decode import (
    ARGUMENT_SCHEMAS,
    MAX_PATH_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    WIRE_VERSION,
    WORKSPACE_WRITE_OPERATIONS,
    WireDecodeError,
    decode_request,
    decode_response,
    encode_response,
)
from tldw_chatbook.Tools.workspace_tool_worker import run_workspace_worker
from tldw_chatbook.Utils.filesystem_identity import (
    DirectoryIdentity,
    capture_directory_chain,
)


# ---------------------------------------------------------------------------
# Live-frame builders (pydantic models are the source of truth)
# ---------------------------------------------------------------------------


def _model_request(
    operation: str,
    arguments: dict[str, Any],
    **overrides: Any,
) -> WorkspaceToolRequest:
    values: dict[str, Any] = {
        "operation_id": "operation-1",
        "operation": operation,
        "intent": "write" if operation in WORKSPACE_WRITE_OPERATIONS else "read",
        "root_locator": Path("/private/workspace"),
        "root_identity": DirectoryIdentity(1, 2, 0o040755, False),
        "ancestor_identities": (DirectoryIdentity(1, 3, 0o040755, False),),
        "arguments": arguments,
        "timeout_seconds": 30,
        "output_max_bytes": 1024,
    }
    values.update(overrides)
    return WorkspaceToolRequest(**values)


def _base() -> dict[str, Any]:
    """A valid request payload straight from the pydantic model itself."""
    request = _model_request("stat_path", {"path": "src/app.py"})
    return json.loads(request.to_bytes())


def _response_base() -> dict[str, Any]:
    """A valid response payload straight from the pydantic model itself."""
    response = WorkspaceToolResponse(
        operation_id="operation-1",
        outcome="success",
        code="ok",
        result="file-body",
        error=None,
        elapsed_ms=12,
        truncated=False,
        cleanup_proven=True,
    )
    return json.loads(response.to_bytes())


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
    # The bootstrap probe (Task 9): dispatched before the root pin to
    # capture the identity chain every other operation must carry; it
    # takes no operational arguments.
    "ping": {},
}


# ---------------------------------------------------------------------------
# Request corpus
# ---------------------------------------------------------------------------

CORPUS: list[tuple[str, bytes]] = []


def _add(name: str, doc: dict[str, Any]) -> None:
    CORPUS.append((name, json.dumps(doc).encode()))


def _add_valid_request(name: str, **request_overrides: Any) -> None:
    operation = request_overrides.pop("operation", "stat_path")
    arguments = request_overrides.pop("arguments", {"path": "file.txt"})
    CORPUS.append(
        (name, _model_request(operation, arguments, **request_overrides).to_bytes())
    )


# --- valid: the brief's baseline plus every closed operation ---------------

_add("valid", _base())
for _operation in sorted(_ARGUMENTS_BY_OPERATION):
    _add_valid_request(
        f"valid-{_operation}", operation=_operation,
        arguments=_ARGUMENTS_BY_OPERATION[_operation],
    )

# --- valid: optional per-operation argument coverage -----------------------

_add_valid_request(
    "valid-fs-read-window",
    operation="fs_read",
    arguments={"path": "log.txt", "offset": 1, "limit": 0, "sensitive_exclusions": []},
)
_add_valid_request(
    "valid-fs-write-full",
    operation="fs_write",
    arguments={
        "path": "out.bin",
        "content": "data",
        "dry_run": True,
        "expected_sha256": "ab" * 32,
        "expected_absent": False,
        "sensitive_exclusions": [],
    },
)
_add_valid_request(
    "valid-fs-edit-replace-all",
    operation="fs_edit",
    arguments={
        "path": "edit.txt",
        "old_string": "a",
        "new_string": "b",
        "replace_all": True,
        "sensitive_exclusions": [],
    },
)
_add_valid_request(
    "valid-fs-patch-targets",
    operation="fs_patch",
    arguments={
        "diff": "--- a/x\n+++ b/x\n",
        "dry_run": True,
        "targets": ["a.txt", "sub/b.txt"],
        "sensitive_exclusions": [],
    },
)
_add_valid_request(
    "valid-fs-glob-limits",
    operation="fs_glob",
    arguments={"pattern": "**/*.py", "max_results": 25, "sensitive_exclusions": []},
)
for _mode in ("content", "files", "count"):
    _add_valid_request(
        f"valid-fs-grep-mode-{_mode}",
        operation="fs_grep",
        arguments={
            "pattern": "needle",
            "mode": _mode,
            "max_results": 10,
            "sensitive_exclusions": [{"kind": "name", "value": "*.pem"}],
            "content_exclusions": [],
        },
    )
_add_valid_request(
    "valid-git-status-path",
    operation="git_status",
    arguments={"path": ".", "sensitive_exclusions": []},
)
_add_valid_request(
    "valid-git-diff-full",
    operation="git_diff",
    arguments={
        "sensitive_exclusions": [],
        "staged": True,
        "commit_range": "HEAD~3..HEAD",
        "path": "src",
        "stat": False,
    },
)
_add_valid_request(
    "valid-git-log-full",
    operation="git_log",
    arguments={"sensitive_exclusions": [], "count": 5, "path": "README.md"},
)
_add_valid_request(
    "valid-git-blame-full",
    operation="git_blame",
    arguments={
        "path": "mod.py",
        "start_line": 1,
        "end_line": 99,
        "sensitive_exclusions": [],
    },
)
_add_valid_request(
    "valid-sensitive-exclusion-kinds",
    operation="fs_read",
    arguments={
        "path": "read.txt",
        "sensitive_exclusions": [
            {"kind": "subtree", "value": ".git"},
            {"kind": "file", "value": "secret.pem"},
            {"kind": "direct_children", "value": "node_modules"},
            {"kind": "name", "value": "*.env"},
        ],
    },
)
_add_valid_request(
    "valid-unicode-payload",
    operation="fs_write",
    operation_id="операция-🐇",
    root_locator=Path("/tmp/工作区"),
    arguments={
        "path": "文件.txt",
        "content": "héllo 🌍 — 引用",
        "sensitive_exclusions": [],
    },
)
_add_valid_request(
    "valid-many-ancestors",
    ancestor_identities=tuple(
        DirectoryIdentity(1, index, 0o040755, index % 2 == 0)
        for index in range(4, 12)
    ),
)
_add_valid_request(
    "valid-huge-identity-ints",
    root_identity=DirectoryIdentity(2**62, 2**61, 0o100755, False),
    ancestor_identities=(DirectoryIdentity(2**62, 2**60, 0o040755, True),),
)
_add_valid_request("valid-empty-operation-id", operation_id="")

# --- malformed: the brief's verbatim set ------------------------------------

d = _base(); d["version"] = 2; _add("wrong-version", d)
d = _base(); d["timeout_seconds"] = True; _add("bool-for-int", d)
CORPUS.append(("duplicate-key", b'{"version":1,"version":1,"op":"fs_read"}'))
CORPUS.append(("nan", b'{"version":1,"op":"fs_read","args":NaN}'))
CORPUS.append(("invalid-utf8", b'{"version":1,"op":"\xff\xfe"}'))
d = _base(); d["surprise"] = 1; _add("unknown-field", d)
CORPUS.append(("oversize", b'{"version":1,"op":"' + b"a" * (11 * 1024 * 1024) + b'"}'))
CORPUS.append(("truncated", b'{"version":1,"op":'))

# --- malformed: bool-vs-int / strict-number confusion -----------------------

d = _base(); d["version"] = True; _add("version-as-bool", d)
d = _base(); d["version"] = 1.0; _add("version-as-float", d)
d = _base(); d["version"] = "1"; _add("version-as-string", d)
d = _base(); d["version"] = 10**30; _add("wrong-version-huge", d)
d = _base(); d["timeout_seconds"] = 30.5; _add("timeout-as-float", d)
d = _base(); d["timeout_seconds"] = "30"; _add("timeout-as-string", d)
d = _base(); d["output_max_bytes"] = True; _add("output-max-bytes-bool", d)
d = _base(); del d["timeout_seconds"]; _add("missing-field", d)

# --- malformed: closed enums and intent agreement ---------------------------

d = _base(); d["intent"] = "write"; _add("intent-mismatch", d)
d = _base(); d["operation"] = "fs_delete"; _add("unknown-operation", d)
d = _base(); d["operation"] = 7; _add("operation-as-int", d)
d = _base(); d["intent"] = "execute"; _add("unknown-intent", d)

# --- malformed: nested directory-identity frames ----------------------------

d = _base(); d["root_identity"] = 5; _add("root-identity-nonobject", d)
d = _base(); d["root_identity"] = [1, 2, 3]; _add("root-identity-as-list", d)
d = _base()
d["root_identity"] = {**d["root_identity"], "surplus": 1}
_add("identity-extra-key", d)
d = _base()
d["root_identity"] = {
    key: value for key, value in d["root_identity"].items() if key != "mode"
}
_add("identity-missing-key", d)
d = _base(); d["root_identity"]["device"] = True; _add("identity-device-bool", d)
d = _base(); d["root_identity"]["reparse"] = 0; _add("identity-reparse-int", d)
d = _base(); d["root_identity"]["inode"] = -1; _add("identity-inode-negative", d)
d = _base(); d["ancestor_identities"] = []; _add("ancestors-empty", d)
d = _base()
d["ancestor_identities"] = dict(d["root_identity"])
_add("ancestors-nonarray", d)
d = _base()
d["ancestor_identities"] = ["not-an-identity"]
_add("ancestor-item-nonobject", d)
CORPUS.append((
    "duplicate-nested-identity-key",
    json.dumps(_base()).encode().replace(b'"device": 1', b'"device": 1, "device": 2', 1),
))

# --- malformed: arguments shape and per-operation schemas -------------------

d = _base(); d["arguments"] = []; _add("arguments-nonobject", d)
d = _base(); d["arguments"] = {}; _add("arguments-missing-required", d)
d = _base()
d["arguments"] = {"path": "a.txt", "sensitive_exclusions": [], "blob": 1}
_add("arguments-unknown-key", d)

_ARG_CASES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "fs-read-offset-zero",
        "fs_read",
        {"path": "a.txt", "offset": 0, "sensitive_exclusions": []},
    ),
    (
        "fs-read-limit-negative",
        "fs_read",
        {"path": "a.txt", "limit": -1, "sensitive_exclusions": []},
    ),
    (
        "fs-write-dry-run-int",
        "fs_write",
        {"path": "a.txt", "content": "x", "dry_run": 0, "sensitive_exclusions": []},
    ),
    (
        "fs-write-sha-uppercase",
        "fs_write",
        {
            "path": "a.txt",
            "content": "x",
            "expected_sha256": "AB" * 32,
            "sensitive_exclusions": [],
        },
    ),
    (
        "fs-write-sha-short",
        "fs_write",
        {
            "path": "a.txt",
            "content": "x",
            "expected_sha256": "a" * 63,
            "sensitive_exclusions": [],
        },
    ),
    (
        "fs-edit-replace-all-int",
        "fs_edit",
        {
            "path": "a.txt",
            "old_string": "a",
            "new_string": "b",
            "replace_all": 1,
            "sensitive_exclusions": [],
        },
    ),
    ("fs-patch-diff-int", "fs_patch", {"diff": 5, "sensitive_exclusions": []}),
    (
        "fs-patch-targets-empty",
        "fs_patch",
        {"diff": "--- a/x\n+++ b/x\n", "targets": [], "sensitive_exclusions": []},
    ),
    (
        "fs-patch-targets-oversize",
        "fs_patch",
        {
            "diff": "--- a/x\n+++ b/x\n",
            "targets": [f"f{index}.txt" for index in range(21)],
            "sensitive_exclusions": [],
        },
    ),
    ("fs-glob-absolute", "fs_glob", {"pattern": "/etc/*", "sensitive_exclusions": []}),
    ("fs-glob-parent", "fs_glob", {"pattern": "../secret", "sensitive_exclusions": []}),
    ("fs-glob-drive", "fs_glob", {"pattern": "C:/src/*", "sensitive_exclusions": []}),
    ("fs-glob-nul", "fs_glob", {"pattern": "a\x00b", "sensitive_exclusions": []}),
    (
        "fs-grep-bad-mode",
        "fs_grep",
        {
            "pattern": "n",
            "mode": "lines",
            "sensitive_exclusions": [],
            "content_exclusions": [],
        },
    ),
    ("git-log-count-zero", "git_log", {"count": 0, "sensitive_exclusions": []}),
    (
        "git-diff-staged-int",
        "git_diff",
        {"sensitive_exclusions": [], "staged": 1},
    ),
    (
        "git-blame-start-zero",
        "git_blame",
        {"path": "a.py", "start_line": 0, "sensitive_exclusions": []},
    ),
    (
        "sensitive-exclusion-bad-kind",
        "fs_read",
        {"path": "a.txt", "sensitive_exclusions": [{"kind": "evil", "value": "x"}]},
    ),
    (
        "sensitive-exclusion-name-slash",
        "fs_read",
        {"path": "a.txt", "sensitive_exclusions": [{"kind": "name", "value": "a/b"}]},
    ),
    (
        "sensitive-exclusion-extra-key",
        "fs_read",
        {
            "path": "a.txt",
            "sensitive_exclusions": [
                {"kind": "subtree", "value": "x", "mode": 1}
            ],
        },
    ),
    (
        "stat-path-extra-key",
        "stat_path",
        {"path": "a.txt", "sensitive_exclusions": []},
    ),
    (
        "ping-unexpected-argument",
        "ping",
        {"path": "a.txt"},
    ),
    (
        "ping-sensitive-exclusions-argument",
        "ping",
        {"sensitive_exclusions": []},
    ),
    ("path-arg-nul", "fs_read", {"path": "a\x00b", "sensitive_exclusions": []}),
    ("path-arg-int", "fs_read", {"path": 3, "sensitive_exclusions": []}),
)
for _name, _operation, _arguments in _ARG_CASES:
    _intent = "write" if _operation in WORKSPACE_WRITE_OPERATIONS else "read"
    _doc = _base()
    _doc["operation"] = _operation
    _doc["intent"] = _intent
    _doc["arguments"] = _arguments
    _add(f"arg-{_name}", _doc)

# --- malformed: encoding-level strict JSON rules ----------------------------

CORPUS.append(("not-an-object", b"[1, 2, 3]"))
CORPUS.append(("scalar-string-frame", b'"hello"'))
CORPUS.append(("empty-frame", b""))
CORPUS.append(("leading-bom", b"\xef\xbb\xbf" + json.dumps(_base()).encode()))
CORPUS.append(("single-quotes", b"{'version': 1}"))
_encoded_base = json.dumps(_base()).encode()
CORPUS.append(("trailing-comma", _encoded_base[:-1] + b",}"))
CORPUS.append(("trailing-garbage", _encoded_base + b"{}"))
CORPUS.append((
    "leading-zero-int",
    _encoded_base.replace(b'"timeout_seconds": 30', b'"timeout_seconds": 030', 1),
))
CORPUS.append((
    "raw-newline-in-string",
    _encoded_base.replace(b'"operation-1"', b'"oper\nation-1"', 1),
))
d = _base(); d["operation_id"] = "\ud800"; _add("lone-surrogate", d)
d = _base(); d["operation_id"] = "a\x00b"; _add("string-nul", d)
d = _base()
d["arguments"] = {"path": "x", "sensitive_exclusions": [[[[[[[["deep"]]]]]]]]}
_add("nested-junk-arguments", d)
d = _base(); d["root_locator"] = "/" + "a" * MAX_PATH_BYTES
_add("path-over-cap", d)
CORPUS.append(("request-oversize-ceiling", b"x" * (MAX_REQUEST_BYTES + 1)))


# ---------------------------------------------------------------------------
# Response corpus
# ---------------------------------------------------------------------------

RESPONSE_CORPUS: list[tuple[str, bytes]] = []


def _add_response(name: str, doc: dict[str, Any]) -> None:
    RESPONSE_CORPUS.append((name, json.dumps(doc).encode()))


def _add_valid_response(
    name: str,
    outcome: str = "success",
    result: str | None = None,
    error: str | None = None,
    **overrides: Any,
) -> None:
    values: dict[str, Any] = {
        "operation_id": "operation-1",
        "outcome": outcome,
        "code": "ok",
        "result": result,
        "error": error,
        "elapsed_ms": 12,
        "truncated": False,
        "cleanup_proven": True,
    }
    values.update(overrides)
    RESPONSE_CORPUS.append((name, WorkspaceToolResponse(**values).to_bytes()))


_add_valid_response("valid-success", result="file-body")
_add_valid_response("valid-failure", outcome="failure", error="boom")
_add_valid_response("valid-admitted", outcome="admitted", code="pin_ok")
# Task 9: the ping op renders its result as a JSON-encoded object string
# inside the existing string result field (the frame schema is unchanged).
_add_valid_response(
    "valid-ping-json-result",
    result=(
        '{"identity_chain":[["/home/me/proj",1,2,16877],["/home/me",1,3,16877],'
        '["/",1,4,16877]],"canonical_path":"/home/me/proj",'
        '"python_version":"3.12.11","bundle_sha256":"' + "ab" * 32 + '"}'
    ),
)
# Task 9 review: fs_read results end in a worker-reported CAS stamp tail
# (``\nsha256: <64 hex>\nsize: <digits>``). The frame schema keeps
# ``result`` an opaque string — the decoders pin that such results
# round-trip both sides identically; the hex/size SHAPE gate lives in
# ``remote_workspace_executor.parse_fs_read_stamps`` (pinned there).
_add_valid_response(
    "valid-stamp-tail-result",
    result="1\thello wörld\n2\tbody\nsha256: " + "ab" * 32 + "\nsize: 12",
)
# Malformed stamp-tail flavour: a NUL inside the digest field is a frame
# contract violation on BOTH decoders (the string NUL rule).
d = _response_base()
d["result"] = "1\tbody\nsha256: " + "a\x00b"
_add_response("stamp-tail-nul-in-sha256", d)
# Only success forbids ``error`` and only failure forbids ``result``;
# ``admitted`` may carry an error and success may omit its result.
_add_valid_response("valid-admitted-with-error", outcome="admitted", error="late")
_add_valid_response("valid-success-null-result")
_add_valid_response(
    "valid-unicode-result", result="résultat 📄 — 結果", error=None,
)
_add_valid_response("valid-empty-code", result="body", code="")
_add_valid_response("valid-zero-elapsed", result="body", elapsed_ms=0)

d = _response_base(); d["version"] = 2; _add_response("wrong-version", d)
d = _response_base(); d["version"] = True; _add_response("version-as-bool", d)
d = _response_base(); d["elapsed_ms"] = True; _add_response("bool-for-int", d)
d = _response_base(); d["surprise"] = 1; _add_response("unknown-field", d)
d = _response_base(); del d["code"]; _add_response("missing-field", d)
d = _response_base(); d["outcome"] = "pending"; _add_response("unknown-outcome", d)
d = _response_base(); d["outcome"] = 1; _add_response("outcome-as-int", d)
d = _response_base(); d["code"] = None; _add_response("code-as-null", d)
d = _response_base(); d["code"] = 5; _add_response("code-as-int", d)
d = _response_base(); d["result"] = 5; _add_response("result-as-int", d)
d = _response_base()
d["outcome"] = "failure"; d["result"] = None; d["error"] = 5
_add_response("error-as-int", d)
d = _response_base(); d["elapsed_ms"] = -1; _add_response("elapsed-negative", d)
d = _response_base(); d["elapsed_ms"] = 1.5; _add_response("elapsed-float", d)
d = _response_base(); d["truncated"] = 1; _add_response("truncated-as-int", d)
d = _response_base(); d["truncated"] = "false"; _add_response("truncated-as-string", d)
d = _response_base(); d["cleanup_proven"] = 1; _add_response("cleanup-proven-int", d)
d = _response_base(); d["error"] = "boom"; _add_response("success-with-error", d)
d = _response_base()
d["outcome"] = "failure"; d["error"] = "boom"
_add_response("failure-with-result", d)
RESPONSE_CORPUS.append(("duplicate-key", b'{"version":1,"version":1}'))
RESPONSE_CORPUS.append(("nan", b'{"version":1,"elapsed_ms":NaN}'))
RESPONSE_CORPUS.append(("invalid-utf8", b'{"version":1,"code":"\xff\xfe"}'))
RESPONSE_CORPUS.append(("truncated", b'{"version":1,'))
RESPONSE_CORPUS.append(("not-an-object", b"[1, 2, 3]"))
RESPONSE_CORPUS.append(("empty-frame", b""))
RESPONSE_CORPUS.append(
    ("response-oversize-ceiling", b"x" * (MAX_RESPONSE_BYTES + 1))
)


# ---------------------------------------------------------------------------
# The acceptance-contract gate
# ---------------------------------------------------------------------------


def _request_verdicts(raw: bytes) -> tuple[bool, bool]:
    try:
        WorkspaceToolRequest.from_bytes(raw)
        pydantic_ok = True
    except (WorkspaceProtocolError, ValueError):
        pydantic_ok = False
    try:
        decode_request(raw)
        wire_ok = True
    except WireDecodeError:
        wire_ok = False
    return pydantic_ok, wire_ok


def _response_verdicts(raw: bytes) -> tuple[bool, bool]:
    try:
        WorkspaceToolResponse.from_bytes(raw)
        pydantic_ok = True
    except (WorkspaceProtocolError, ValueError):
        pydantic_ok = False
    try:
        decode_response(raw)
        wire_ok = True
    except WireDecodeError:
        wire_ok = False
    return pydantic_ok, wire_ok


@pytest.mark.parametrize("name,raw", CORPUS, ids=[n for n, _ in CORPUS])
def test_request_acceptance_is_identical(name: str, raw: bytes) -> None:
    pydantic_ok, wire_ok = _request_verdicts(raw)
    expected = name.startswith("valid")

    assert pydantic_ok == wire_ok, f"{name}: pydantic={pydantic_ok} wire={wire_ok}"
    assert pydantic_ok == expected, (
        f"{name}: parent verdict {pydantic_ok}, expected {expected}"
    )
    assert wire_ok == expected, (
        f"{name}: decoder verdict {wire_ok}, expected {expected}"
    )


@pytest.mark.parametrize(
    "name,raw", RESPONSE_CORPUS, ids=[n for n, _ in RESPONSE_CORPUS]
)
def test_response_acceptance_is_identical(name: str, raw: bytes) -> None:
    pydantic_ok, wire_ok = _response_verdicts(raw)
    expected = name.startswith("valid")

    assert pydantic_ok == wire_ok, f"{name}: pydantic={pydantic_ok} wire={wire_ok}"
    assert pydantic_ok == expected, (
        f"{name}: parent verdict {pydantic_ok}, expected {expected}"
    )
    assert wire_ok == expected, (
        f"{name}: decoder verdict {wire_ok}, expected {expected}"
    )


def test_corpus_shape_is_wellformed() -> None:
    """The corpus is a shared artifact; pin its structural invariants."""
    for corpus in (CORPUS, RESPONSE_CORPUS):
        names = [name for name, _ in corpus]
        assert len(names) == len(set(names)), "corpus ids must be unique"
        assert all(isinstance(raw, bytes) for _, raw in corpus)
    request_names = {name for name, _ in CORPUS}
    response_names = {name for name, _ in RESPONSE_CORPUS}
    # The brief's canonical entries stay present verbatim.
    for required in (
        "valid",
        "wrong-version",
        "bool-for-int",
        "duplicate-key",
        "nan",
        "invalid-utf8",
        "unknown-field",
        "oversize",
        "truncated",
    ):
        assert required in request_names
    assert sum(name.startswith("valid") for name in request_names) >= 25
    assert sum(not name.startswith("valid") for name in request_names) >= 50
    assert sum(name.startswith("valid") for name in response_names) >= 5
    assert sum(not name.startswith("valid") for name in response_names) >= 20


# ---------------------------------------------------------------------------
# Bonus: deterministic-seeded generative agreement (valid frames only)
# ---------------------------------------------------------------------------

_NON_NUL_TEXT = st.text(
    st.characters(exclude_characters="\x00", exclude_categories=("Cs",)), max_size=24
)
_NAME_SAFE_TEXT = st.text(
    st.characters(exclude_characters="\x00/\\", exclude_categories=("Cs",)),
    max_size=16,
)
_SENSITIVE_EXCLUSION = st.one_of(
    st.builds(lambda value: {"kind": "name", "value": value}, _NAME_SAFE_TEXT),
    st.builds(
        lambda kind, value: {"kind": kind, "value": value},
        st.sampled_from(["subtree", "file", "direct_children"]),
        _NON_NUL_TEXT,
    ),
)
_KIND_STRATEGIES = {
    "path": _NON_NUL_TEXT,
    "text": _NON_NUL_TEXT,
    "glob_pattern": st.sampled_from(
        ["**/*.py", "src/**/?ile", "a/b.txt", "*", "deep/**/x"]
    ),
    "patch": st.sampled_from(["--- a/f\n+++ b/f\n", ""]),
    "patch_targets": st.lists(_NON_NUL_TEXT, min_size=1, max_size=3),
    "bool": st.booleans(),
    "sha256": st.sampled_from(["ab" * 32, "0" * 64]),
    "positive_int": st.integers(min_value=1, max_value=10**9),
    "nonnegative_int": st.integers(min_value=0, max_value=10**9),
    "grep_mode": st.sampled_from(["content", "files", "count"]),
    "sensitive_exclusions": st.lists(_SENSITIVE_EXCLUSION, max_size=3),
}


@settings(
    derandomize=True,
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    operation=st.sampled_from(sorted(_ARGUMENTS_BY_OPERATION)),
    data=st.data(),
)
def test_generated_valid_requests_agree(
    operation: str, data: st.DataObject
) -> None:
    required, accepted = ARGUMENT_SCHEMAS[operation]
    arguments: dict[str, Any] = {}
    for key in sorted(required | frozenset(accepted)):
        if key not in required and not data.draw(st.booleans()):
            continue
        arguments[key] = data.draw(_KIND_STRATEGIES[accepted[key]])
    raw = _model_request(operation, arguments).to_bytes()

    # The parent accepts its own serialization by construction; the decoder
    # must accept it too and hand back exactly the frame it was given.
    assert decode_request(raw) == json.loads(raw)
    assert decode_request(raw)["arguments"] == arguments


@settings(
    derandomize=True,
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(outcome=st.sampled_from(["admitted", "success", "failure"]), data=st.data())
def test_generated_valid_responses_agree(outcome: str, data: st.DataObject) -> None:
    body = data.draw(_NON_NUL_TEXT)
    values: dict[str, Any] = {
        "operation_id": data.draw(_NON_NUL_TEXT),
        "outcome": outcome,
        "code": data.draw(_NON_NUL_TEXT),
        "result": body if outcome == "success" and data.draw(st.booleans()) else None,
        "error": body if outcome != "success" and data.draw(st.booleans()) else None,
        "elapsed_ms": data.draw(st.integers(min_value=0, max_value=10**9)),
        "truncated": data.draw(st.booleans()),
        "cleanup_proven": data.draw(st.booleans()),
    }
    raw = WorkspaceToolResponse(**values).to_bytes()  # type: ignore[arg-type]

    assert decode_response(raw) == json.loads(raw)
    assert decode_response(raw)["outcome"] == outcome


# ---------------------------------------------------------------------------
# Byte-identity of the worker's response encoder (Phase 0c, review fix)
# ---------------------------------------------------------------------------
#
# The parent's ``from_bytes`` is order-insensitive, so a field reorder in
# the worker's serializer would keep every acceptance test green while
# silently changing the wire bytes. These pins hold the worker-emitted
# frame bytes to the parent's ``WorkspaceToolResponse.to_bytes`` layout.


def _scrambled(frame: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a frame dict in reverse key order (order must not matter)."""
    return {name: frame[name] for name in reversed(list(frame))}


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(
            {
                "operation_id": "operation-1",
                "outcome": "admitted",
                "code": "root_pinned",
                "result": None,
                "error": None,
                "elapsed_ms": 0,
                "truncated": False,
                "cleanup_proven": True,
            },
            id="admitted",
        ),
        pytest.param(
            {
                "operation_id": "operation-1",
                "outcome": "success",
                "code": "ok",
                "result": "1\thello wörld — τext\n",
                "error": None,
                "elapsed_ms": 4_567,
                "truncated": True,
                "cleanup_proven": True,
            },
            id="success",
        ),
        pytest.param(
            {
                "operation_id": "unknown",
                "outcome": "failure",
                "code": "invalid_request",
                "result": None,
                "error": "workspace operation failed",
                "elapsed_ms": 12,
                "truncated": False,
                "cleanup_proven": True,
            },
            id="failure",
        ),
        pytest.param(
            {
                "operation_id": "operation-1",
                "outcome": "failure",
                "code": "tool_failure",
                "result": None,
                "error": "file not found: missing.txt",
                "elapsed_ms": 1,
                "truncated": False,
                "cleanup_proven": False,
            },
            id="failure-cleanup-unproven",
        ),
    ],
)
def test_encode_response_bytes_match_parent_to_bytes(values: dict[str, Any]) -> None:
    parent_bytes = WorkspaceToolResponse(**values).to_bytes()  # type: ignore[arg-type]
    frame = {"version": WIRE_VERSION, **values}

    assert encode_response(_scrambled(frame)) == parent_bytes
    # The emitted bytes must also survive the worker's own decode self-check.
    assert decode_response(encode_response(frame))["code"] == values["code"]


def _worker_frames(raw_request: bytes) -> tuple[int, list[bytes]]:
    """Run the pinned worker once and collect its exit code and frame lines."""
    stdin, stdout = io.BytesIO(raw_request), io.BytesIO()
    exit_code = run_workspace_worker(stdin, stdout, io.BytesIO())
    return exit_code, stdout.getvalue().splitlines()


def _pinned_request(
    workspace: Path, operation: str, arguments: dict[str, Any]
) -> bytes:
    chain = capture_directory_chain(workspace)
    request = WorkspaceToolRequest(
        operation_id="byte-identity",
        operation=operation,  # type: ignore[arg-type]
        intent="write" if operation in WORKSPACE_WRITE_OPERATIONS else "read",
        root_locator=chain.canonical_root,
        root_identity=chain.identities[0],
        ancestor_identities=chain.identities,
        arguments=arguments,
        timeout_seconds=30,
        output_max_bytes=MAX_RESPONSE_BYTES,
    )
    return request.to_bytes()


@pytest.mark.parametrize(
    ("operation", "arguments", "expected_codes"),
    [
        pytest.param(
            "fs_read",
            {"path": "note.txt", "sensitive_exclusions": []},
            ["admitted", "success"],
            id="success",
        ),
        pytest.param(
            "fs_read",
            {"path": "missing.txt", "sensitive_exclusions": []},
            ["admitted", "failure"],
            id="tool-failure",
        ),
        pytest.param(
            "fs_read",
            {"path": "../escape.txt", "sensitive_exclusions": []},
            ["admitted", "failure"],
            id="dispatch-refusal",
        ),
    ],
)
def test_worker_emitted_frames_are_byte_identical_to_parent_encoder(
    tmp_path: Path,
    operation: str,
    arguments: dict[str, Any],
    expected_codes: list[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("hello", encoding="utf-8")
    raw = _pinned_request(workspace, operation, arguments)

    exit_code, frames = _worker_frames(raw)

    assert [json.loads(frame)["outcome"] for frame in frames] == expected_codes
    assert exit_code == (0 if expected_codes[-1] == "success" else 2)
    for frame in frames:
        parsed = WorkspaceToolResponse.from_bytes(frame)
        rebuilt = WorkspaceToolResponse(
            operation_id=parsed.operation_id,
            outcome=parsed.outcome,
            code=parsed.code,
            result=parsed.result,
            error=parsed.error,
            elapsed_ms=parsed.elapsed_ms,
            truncated=parsed.truncated,
            cleanup_proven=parsed.cleanup_proven,
        ).to_bytes()
        assert rebuilt == frame


def test_worker_invalid_request_frame_is_byte_identical_to_parent_encoder() -> None:
    exit_code, frames = _worker_frames(b"{not-json")

    assert exit_code == 2
    assert len(frames) == 1
    parsed = WorkspaceToolResponse.from_bytes(frames[0])
    assert parsed.outcome == "failure"
    assert parsed.code == "invalid_request"
    assert parsed.operation_id == "unknown"
    rebuilt = WorkspaceToolResponse(
        operation_id=parsed.operation_id,
        outcome=parsed.outcome,
        code=parsed.code,
        result=parsed.result,
        error=parsed.error,
        elapsed_ms=parsed.elapsed_ms,
        truncated=parsed.truncated,
        cleanup_proven=parsed.cleanup_proven,
    ).to_bytes()
    assert rebuilt == frames[0]
