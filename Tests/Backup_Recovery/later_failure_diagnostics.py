"""Persist only bounded later-child failure metadata before pytest finalization."""

from pathlib import Path


def record_failure(
    path: Path, *, error: BaseException | None = None, returncode: int | None = None
) -> None:
    """Never replace the native failure with an observation failure."""
    try:
        from Tests.Backup_Recovery.thread_diagnostics import _error_metadata, _write

        record = (
            {"event": "exception", "error": _error_metadata(error)}
            if error is not None
            else {"event": "child_exit", "returncode": returncode}
        )
        _write(path, record)
    except BaseException:  # noqa: BLE001 - preserve the original failure.
        return
