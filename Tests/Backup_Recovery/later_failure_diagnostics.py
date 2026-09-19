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
        if error is not None:
            try:
                from textual.worker import WorkerFailed

                if type(error) is WorkerFailed:
                    record["worker_error"] = None
                    inner = vars(error).get("error")
                    if isinstance(inner, BaseException) and inner is not error:
                        record["worker_error"] = _error_metadata(inner)
            except BaseException:  # noqa: BLE001, S110 - optional metadata cannot discard the root error.  # nosec B110
                pass
        _write(path, record)
    except BaseException:  # noqa: BLE001 - preserve the original failure.
        return
