"""Fold server ingest job statuses back into the local ingest registry.

Split out from the polling worker so the whole decision layer is testable
against a real (Textual-free) ``LibraryIngestJobRegistry`` -- no app, no
worker, no server (task-684.2). The worker's remaining job is only to fetch
statuses and hand them here.

Two properties this layer is responsible for:

- **A settled job is never rewritten.** Polling repeats, so reconciling the
  same terminal status twice must be a no-op rather than re-stamping the
  finish time on every pass.
- **An unrecognised status moves nothing.** See
  ``server_ingest_status.local_state_for_server_status`` for why unknown is
  deliberately not treated as finished.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from loguru import logger

from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Library.server_ingest_status import local_state_for_server_status

#: States a job never leaves, mirroring the registry's own terminal set. Kept
#: local to avoid importing a private name across modules.
_TERMINAL = (IngestJobState.DONE, IngestJobState.FAILED, IngestJobState.CANCELLED)


def _field(status: Any, name: str) -> Any:
    """Read ``name`` from a model or a plain dict.

    The client returns ``MediaIngestJobStatus`` models, but its own tests feed
    dicts through the same paths, so both shapes are accepted.
    """
    if isinstance(status, Mapping):
        return status.get(name)
    return getattr(status, name, None)


def _progress_payload(status: Any) -> dict[str, Any] | None:
    """Build a progress payload from whatever the server reported, or ``None``."""
    percent = _field(status, "progress_percent")
    message = _field(status, "progress_message")
    if percent is None and not message:
        return None
    payload: dict[str, Any] = {}
    if percent is not None:
        payload["percent"] = percent
    if message:
        payload["message"] = message
    return payload


def reconcile_remote_ingest_jobs(
    registry: LibraryIngestJobRegistry,
    statuses: Iterable[Any],
) -> int:
    """Apply server job statuses to their matching local jobs.

    Args:
        registry: The local ingest job registry. Mutated in place, so this must
            be called on the UI thread like every other registry mutation.
        statuses: ``MediaIngestJobStatus``-shaped models or dicts, each carrying
            at least ``id`` and ``status``.

    Returns:
        The number of jobs actually transitioned. A status for an unknown remote
        id, for a job already in that terminal state, or with an unrecognised
        ``status`` string contributes nothing.
    """
    by_remote_id = {
        job.remote_job_id: job
        for job in registry.jobs()
        if job.origin == "server" and job.remote_job_id
    }
    if not by_remote_id:
        return 0

    applied = 0
    for status in statuses:
        remote_id = _field(status, "id")
        if remote_id is None:
            continue
        job = by_remote_id.get(str(remote_id))
        if job is None:
            continue

        raw_status = _field(status, "status")
        target = local_state_for_server_status(raw_status)
        if target is None:
            logger.debug(
                f"Unrecognised server ingest status {raw_status!r} for remote job "
                f"{remote_id}; leaving {job.job_id} as {job.state.value}."
            )
            continue
        if job.state is target and target in _TERMINAL:
            # Already settled on this outcome; polling must not re-stamp it.
            continue

        progress = _progress_payload(status)

        if target is IngestJobState.DONE:
            updated = registry.mark_remote_done(job.job_id)
        elif target is IngestJobState.FAILED:
            updated = registry.mark_failed(
                job.job_id,
                error=str(_field(status, "error_message") or "The server reported a failure."),
            )
        elif target is IngestJobState.CANCELLED:
            updated = registry.mark_cancelled(
                job.job_id,
                reason=str(
                    _field(status, "cancellation_reason") or "Cancelled on the server."
                ),
            )
        elif target is IngestJobState.PARSING:
            updated = (
                registry.mark_parsing(job.job_id)
                if job.state is not IngestJobState.PARSING
                else job
            )
            if progress is not None:
                registry.update_progress(job.job_id, progress=progress)
        else:
            # QUEUED: the server has not started it, which is where a freshly
            # submitted job already sits. Nothing to change.
            updated = None

        if updated is not None:
            applied += 1

    return applied
