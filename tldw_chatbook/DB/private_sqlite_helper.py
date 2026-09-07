"""Fixed child-only operations for private SQLite validation (standard library)."""

from __future__ import annotations

import os
import select
import stat
import time
from pathlib import Path

from tldw_chatbook.DB import private_sqlite_files as files
from tldw_chatbook.DB.private_sqlite_protocol import (
    PRIVATE_REASONS,
    FileIdentity,
    PrepareRequest,
    PrepareResult,
    ProtocolError,
    encode_frame,
    read_frame,
)
from tldw_chatbook.Utils import private_paths
from tldw_chatbook.Utils.private_paths import PrivatePathError, PrivatePathStatus

ROUND_TRIP_SECONDS = 5.0
PARENT_POLL_SECONDS = 1.0


class SourcePin:
    """Child-owned original main inode and parent; never transferred to a caller."""

    def __init__(self, request: PrepareRequest, result: PrepareResult) -> None:
        self.selected = Path(request.path)
        self.result = result
        self.enforce_private_mode = not request.preserve_source_mode
        self.parent_fd = self.file_fd = -1
        try:
            self.parent_fd, leaf = private_paths._open_verified_parent(
                self.selected, missing_leaf_allowed=False
            )
            self.parent_identity = FileIdentity.from_stat(os.fstat(self.parent_fd))
            self.file_fd = files._open_artifact_fd(
                self.parent_fd, leaf, writable=False, create=False
            )
            self.recheck()
        except BaseException as exc:
            self.close()
            if isinstance(exc, OSError) and not isinstance(exc, PrivatePathError):
                raise files._path_error_from_oserror(self.selected, exc) from None
            raise

    def recheck(self) -> PrepareResult:
        """Verify the bound source and parent without accepting another path."""
        other_parent_fd = -1
        try:
            other_parent_fd, leaf = private_paths._open_verified_parent(
                self.selected, missing_leaf_allowed=False
            )
            parent = FileIdentity.from_stat(os.fstat(self.parent_fd))
            named_parent = FileIdentity.from_stat(os.fstat(other_parent_fd))
            opened = os.fstat(self.file_fd)
            named = os.stat(leaf, dir_fd=self.parent_fd, follow_symlinks=False)
            valid = (
                parent.same_inode(self.parent_identity)
                and named_parent.same_inode(self.parent_identity)
                and (parent.mode, parent.uid, parent.gid)
                == (
                    self.parent_identity.mode,
                    self.parent_identity.uid,
                    self.parent_identity.gid,
                )
                and FileIdentity.from_stat(opened).same_inode(self.result.main_identity)
                and FileIdentity.from_stat(named).same_inode(self.result.main_identity)
                and private_paths._classify_private_file_stat(
                    opened, expected_uid=os.geteuid()
                )
                is None
                and private_paths._classify_private_file_stat(
                    named, expected_uid=os.geteuid()
                )
                is None
                and (
                    not self.enforce_private_mode
                    or stat.S_IMODE(opened.st_mode) == 0o600
                )
            )
            if valid:
                return PrepareResult(
                    FileIdentity.from_stat(opened), self.result.artifacts
                )
        except OSError:
            pass
        finally:
            if other_parent_fd >= 0:
                os.close(other_parent_fd)
        raise files._failure(
            self.selected,
            PrivatePathStatus.OPERATION_FAILED,
            "private_sqlite_source_identity_changed",
        )

    def close(self) -> None:
        """Release only this child's captured proof descriptors."""
        try:
            if self.file_fd >= 0:
                os.close(self.file_fd)
                self.file_fd = -1
        finally:
            if self.parent_fd >= 0:
                os.close(self.parent_fd)
                self.parent_fd = -1


class _ParentGone(Exception):
    pass


class _PrivatePipe:
    """Idle parent polling and a single deadline across each frame and reply."""

    def __init__(self, parent_pid: int) -> None:
        self.parent_pid = parent_pid
        self.deadline: float | None = None
        os.set_blocking(0, False)
        os.set_blocking(1, False)

    def _wait(self, fd: int, *, writable: bool = False) -> None:
        while True:
            if os.getppid() != self.parent_pid:
                raise _ParentGone()
            remaining = PARENT_POLL_SECONDS
            if self.deadline is not None:
                remaining = min(remaining, self.deadline - time.monotonic())
                if remaining <= 0:
                    raise TimeoutError()
            readable, writeable, _ = select.select(
                [] if writable else [fd], [fd] if writable else [], [], remaining
            )
            if readable or writeable:
                return

    def read(self, count: int) -> bytes:
        while True:
            self._wait(0)
            try:
                value = os.read(0, count)
            except BlockingIOError:
                continue
            if value and self.deadline is None:
                self.deadline = time.monotonic() + ROUND_TRIP_SECONDS
            return value

    def write(self, frame: bytes) -> None:
        offset = 0
        while offset < len(frame):
            self._wait(1, writable=True)
            try:
                offset += os.write(1, frame[offset:])
            except BlockingIOError:
                continue


def run(parent_pid: int) -> int:
    """Serve one operation-owned stream; EOF or close always releases its pin."""
    if os.getppid() != parent_pid:
        return 0
    pipe = _PrivatePipe(parent_pid)
    source: SourcePin | None = None
    initialized = False
    try:
        while True:
            pipe.deadline = None
            try:
                payload = read_frame(pipe)
            except ProtocolError:
                pipe.write(
                    encode_frame(
                        {"version": 1, "operation": "close", "status": "protocol_error"}
                    )
                )
                return 0
            if payload is None:
                return 0
            if os.getppid() != parent_pid:
                return 0
            operation = payload["operation"]
            response = {"version": 1, "operation": operation}
            try:
                if "status" in payload:
                    raise ProtocolError()
                if operation == "close":
                    if source is not None:
                        source.close()
                        source = None
                    pipe.write(encode_frame({**response, "status": "ok"}))
                    return 0
                if operation in {"prepare", "pin_source"}:
                    if initialized:
                        raise ProtocolError()
                    initialized = True
                    request = PrepareRequest(
                        **{
                            key: payload[key]
                            for key in (
                                "path",
                                "writable",
                                "create_if_missing",
                                "preserve_source_mode",
                            )
                        }
                    )
                    result = files.prepare_batch(request)
                    if operation == "pin_source":
                        source = SourcePin(request, result)
                elif operation == "recheck_source" and source is not None:
                    result = source.recheck()
                else:
                    raise ProtocolError()
                response.update(status="ok", result=result.to_payload())
            except PrivatePathError as exc:
                reason = exc.result.reason
                response.update(
                    status="private_path_error",
                    privacy_status=exc.result.status.value,
                    reason=reason if reason in PRIVATE_REASONS else "operation_failed",
                )
            except ProtocolError:
                response.update(status="protocol_error")
            except Exception:  # noqa: BLE001 - child boundary must not emit exception text
                response.update(status="helper_unavailable")
            pipe.write(encode_frame(response))
            if response["status"] != "ok":
                return 0
    except (_ParentGone, BrokenPipeError, TimeoutError):
        return 0
    finally:
        if source is not None:
            source.close()
