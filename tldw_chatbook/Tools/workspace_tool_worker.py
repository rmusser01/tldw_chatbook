"""Fixed one-shot stdin/stdout worker for pinned workspace operations.

Import closure (Phase 0c): this module's transitive imports must stay
stdlib-only — Task 8 concatenates the closure into a remote worker bundle,
and ``Tests/Tools/test_worker_import_closure.py`` is the gate. Frames are
decoded by ``Tools/workspace_wire_decode.py`` (the stdlib counterpart of
the parent's pydantic serde in ``Tools/workspace_tool_protocol.py``) and
responses are emitted as JSON with the exact serialization the parent's
``WorkspaceToolResponse.from_bytes`` accepts.
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from tldw_chatbook.Tools.local_tool_impls import LocalToolError
from tldw_chatbook.Tools.workspace_root_pin import (
    WorkspaceRootPinError,
    pin_workspace_root,
)
from tldw_chatbook.Tools.workspace_tool_dispatch import (
    WorkspaceToolDispatchError,
    execute_pinned_operation,
)
from tldw_chatbook.Tools.workspace_wire_decode import (
    MAX_REQUEST_BYTES,
    WIRE_VERSION,
    WireDecodeError,
    decode_request,
    decode_response,
)
from tldw_chatbook.Utils.filesystem_identity import (
    DirectoryChain,
    DirectoryIdentity,
)

_MAX_DOMAIN_ERROR_CHARS = 300


@dataclass(frozen=True, slots=True)
class _DecodedRequest:
    """Attribute view over one decoded frame, for pinning and dispatch.

    Field-for-field the decoded payload of ``decode_request``; the wire
    decoder has already applied every admission check the parent's
    ``WorkspaceToolRequest.from_bytes`` applies.
    """

    operation_id: str
    operation: str
    root_locator: Path = field(repr=False)
    root_identity: DirectoryIdentity
    ancestor_identities: tuple[DirectoryIdentity, ...]
    arguments: dict[str, Any] = field(repr=False)


def run_workspace_worker(
    stdin: BinaryIO,
    stdout: BinaryIO,
    stderr: BinaryIO,
) -> int:
    """Read, pin, dispatch, respond once, and return a process exit code."""
    del stderr  # Reserved for fixed diagnostics; no request-derived text is written.
    started = time.monotonic()
    raw = stdin.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        _emit(stdout, _failure("unknown", "invalid_request", started))
        return 2
    try:
        request = _decode_request(raw)
    except WireDecodeError:
        _emit(stdout, _failure("unknown", "invalid_request", started))
        return 2

    chain = DirectoryChain(
        canonical_root=request.root_locator,
        identities=(request.root_identity, *request.ancestor_identities[1:]),
    )
    try:
        with pin_workspace_root(request.root_locator, chain) as root:
            _emit(
                stdout,
                _frame(
                    request.operation_id,
                    outcome="admitted",
                    code="root_pinned",
                    result=None,
                    error=None,
                    started=started,
                ),
            )
            result = execute_pinned_operation(request, root)
        _emit(
            stdout,
            _frame(
                request.operation_id,
                outcome="success",
                code="ok",
                result=result,
                error=None,
                started=started,
            ),
        )
        return 0
    except WorkspaceToolDispatchError as error:
        _emit(
            stdout,
            _failure(request.operation_id, error.code, started, message=str(error)),
        )
        return 2
    except WorkspaceRootPinError:
        _emit(stdout, _failure(request.operation_id, "root_pin_failed", started))
        return 2
    except LocalToolError as error:
        _emit(
            stdout,
            _failure(
                request.operation_id,
                "tool_failure",
                started,
                message=_sanitized_domain_error(error, request.root_locator),
            ),
        )
        return 2
    except (OSError, ValueError):
        _emit(stdout, _failure(request.operation_id, "tool_failure", started))
        return 2
    except BaseException:
        _emit(stdout, _failure(request.operation_id, "worker_failure", started))
        return 2


def _decode_request(raw: bytes) -> _DecodedRequest:
    """Decode one admitted frame into the worker's pinned-request view."""
    payload = decode_request(raw)
    return _DecodedRequest(
        operation_id=payload["operation_id"],
        operation=payload["operation"],
        root_locator=Path(payload["root_locator"]),
        root_identity=_identity(payload["root_identity"]),
        ancestor_identities=tuple(
            _identity(item) for item in payload["ancestor_identities"]
        ),
        arguments=payload["arguments"],
    )


def _identity(payload: Mapping[str, Any]) -> DirectoryIdentity:
    return DirectoryIdentity(
        device=payload["device"],
        inode=payload["inode"],
        mode=payload["mode"],
        reparse=payload["reparse"],
    )


def _frame(
    operation_id: str,
    *,
    outcome: str,
    code: str,
    result: str | None,
    error: str | None,
    started: float,
) -> dict[str, Any]:
    """Build one response payload in the parent's fixed field order."""
    return {
        "version": WIRE_VERSION,
        "operation_id": operation_id,
        "outcome": outcome,
        "code": code,
        "result": result,
        "error": error,
        "elapsed_ms": _elapsed_ms(started),
        "truncated": False,
        "cleanup_proven": True,
    }


def _failure(
    operation_id: str,
    code: str,
    started: float,
    *,
    message: str = "workspace operation failed",
) -> dict[str, Any]:
    return _frame(
        operation_id,
        outcome="failure",
        code=code,
        result=None,
        error=message,
        started=started,
    )


def _sanitized_domain_error(error: LocalToolError, root_locator: object) -> str:
    """Return bounded model-actionable text from one audited domain type."""
    message = str(error)
    root_text = str(root_locator)
    for separator in ("/", "\\"):
        message = message.replace(root_text + separator, "")
    message = message.replace(root_text, ".")
    message = "".join(
        character
        for character in message
        if unicodedata.category(character) != "Cc"
    )
    return message[:_MAX_DOMAIN_ERROR_CHARS] or "workspace operation failed"


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1_000))


def _encode_frame(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise WireDecodeError("protocol frame cannot be serialized") from error


def _emit(stdout: BinaryIO, payload: Mapping[str, Any]) -> None:
    frame = _encode_frame(payload)
    decode_response(frame)
    stdout.write(frame + b"\n")
    stdout.flush()


def main() -> int:
    """Run one isolated protocol exchange on standard streams."""
    return run_workspace_worker(sys.stdin.buffer, sys.stdout.buffer, sys.stderr.buffer)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_workspace_worker"]
