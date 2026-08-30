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
    def start(
        self, request: TerminalLaunchRequest, admission: AdmissionGate
    ) -> BackendIdentity: ...

    def write(self, data: bytes) -> None: ...

    def resize(self, columns: int, rows: int) -> None: ...

    def request_priority_close(self) -> None: ...

    def cleanup(self, attempt: CleanupAttempt) -> CleanupProof: ...
