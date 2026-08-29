"""Fixed one-shot stdin/stdout worker for pinned workspace operations."""

from __future__ import annotations

import sys
import time
from typing import BinaryIO

from tldw_chatbook.Tools.workspace_root_pin import (
    WorkspaceRootPinError,
    pin_workspace_root,
)
from tldw_chatbook.Tools.workspace_tool_dispatch import (
    WorkspaceToolDispatchError,
    execute_pinned_operation,
)
from tldw_chatbook.Tools.workspace_tool_protocol import (
    MAX_REQUEST_BYTES,
    WorkspaceProtocolError,
    WorkspaceToolRequest,
    WorkspaceToolResponse,
)
from tldw_chatbook.Utils.filesystem_identity import DirectoryChain


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
        request = WorkspaceToolRequest.from_bytes(raw)
    except WorkspaceProtocolError:
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
                WorkspaceToolResponse(
                    operation_id=request.operation_id,
                    outcome="admitted",
                    code="root_pinned",
                    result=None,
                    error=None,
                    elapsed_ms=_elapsed_ms(started),
                    truncated=False,
                    cleanup_proven=True,
                ),
            )
            result = execute_pinned_operation(request, root)
        _emit(
            stdout,
            WorkspaceToolResponse(
                operation_id=request.operation_id,
                outcome="success",
                code="ok",
                result=result,
                error=None,
                elapsed_ms=_elapsed_ms(started),
                truncated=False,
                cleanup_proven=True,
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
    except (OSError, ValueError):
        _emit(stdout, _failure(request.operation_id, "tool_failure", started))
        return 2
    except BaseException:
        _emit(stdout, _failure(request.operation_id, "worker_failure", started))
        return 2


def _failure(
    operation_id: str,
    code: str,
    started: float,
    *,
    message: str = "workspace operation failed",
) -> WorkspaceToolResponse:
    return WorkspaceToolResponse(
        operation_id=operation_id,
        outcome="failure",
        code=code,
        result=None,
        error=message,
        elapsed_ms=_elapsed_ms(started),
        truncated=False,
        cleanup_proven=True,
    )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1_000))


def _emit(stdout: BinaryIO, response: WorkspaceToolResponse) -> None:
    stdout.write(response.to_bytes() + b"\n")
    stdout.flush()


def main() -> int:
    """Run one isolated protocol exchange on standard streams."""
    return run_workspace_worker(sys.stdin.buffer, sys.stdout.buffer, sys.stderr.buffer)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_workspace_worker"]
