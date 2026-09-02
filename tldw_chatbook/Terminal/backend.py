"""Platform-neutral terminal backend protocol."""

from typing import Protocol

from .contracts import (
    AdmissionGate,
    BackendIdentity,
    CleanupAttempt,
    CleanupProof,
    TerminalLaunchRequest,
)


class TerminalBackend(Protocol):
    """Platform-neutral operations required from a terminal backend."""

    def start(
        self, request: TerminalLaunchRequest, admission: AdmissionGate
    ) -> BackendIdentity:
        """Start an admitted interactive terminal.

        Args:
            request: Validated launch values.
            admission: Admission decision that gates interactive startup.

        Returns:
            Opaque identity for the owned backend session.
        """
        ...

    def write(self, data: bytes) -> None:
        """Write bounded bytes to the interactive terminal.

        Args:
            data: Input bytes admitted by the terminal actor.
        """
        ...

    def read(self, maximum: int) -> bytes | None:
        """Read one nonblocking bounded output chunk.

        Args:
            maximum: Positive actor credit available for this read.

        Returns:
            Output bytes, ``None`` when no bytes are ready, or ``b""`` at EOF.
        """
        ...

    def wait_for_shell_exit(self, *, timeout_seconds: float) -> int | None:
        """Observe the backend's sole authoritative shell reaper.

        Args:
            timeout_seconds: Maximum bounded wait duration.

        Returns:
            Exact exit code, or ``None`` when the wait expires.
        """
        ...

    def resize(self, columns: int, rows: int) -> None:
        """Resize the terminal allocation.

        Args:
            columns: Validated terminal width.
            rows: Validated terminal height.
        """
        ...

    def request_priority_close(self) -> None:
        """Request out-of-band idempotent cleanup."""
        ...

    def finalize_shutdown(self) -> None:
        """Fence future startup and close parent-owned handles without waiting."""
        ...

    def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
        """Run cleanup under one absolute attempt deadline.

        Args:
            attempt: Attempt start governing every cleanup-stage offset.

        Returns:
            Platform-neutral process, stream, and output evidence.
        """
        ...
