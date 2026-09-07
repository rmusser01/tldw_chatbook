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


def pending_remote_batches(registry: LibraryIngestJobRegistry) -> tuple[str, ...]:
    """Return the server batch ids that still have something worth watching.

    The poller's stop condition. A batch whose every job has settled would
    otherwise be re-fetched forever for an answer that cannot change.

    ``registry.jobs()`` already excludes hidden (``superseded``/``dismissed``)
    jobs, so a dismissed row does not keep its batch alive.

    Args:
        registry: The local ingest job registry.

    Returns:
        Batch ids in first-submission order, de-duplicated. Empty when nothing
        is outstanding -- the signal to stop polling.
    """
    batches: list[str] = []
    for job in registry.jobs():
        if job.origin != "server" or not job.batch_id:
            continue
        if job.state in _TERMINAL:
            continue
        if job.batch_id not in batches:
            batches.append(job.batch_id)
    return tuple(batches)


def _field(status: Any, name: str) -> Any:
    """Read ``name`` from a model or a plain dict.

    The client returns ``MediaIngestJobStatus`` models, but its own tests feed
    dicts through the same paths, so both shapes are accepted.
    """
    if isinstance(status, Mapping):
        return status.get(name)
    return getattr(status, name, None)



def _remote_media_id(status: Any) -> str | None:
    """Read the id of the media row the server created, if it reported one.

    A finished job's ``result`` carries it (confirmed live:
    ``{"status": "Success", "media_id": 1125, ...}``). It addresses a row in the
    SERVER's library, so it is kept apart from the local ``media_id`` -- see
    ``LibraryIngestJob.remote_media_id`` (task-700).

    Args:
        status: A ``MediaIngestJobStatus``-shaped model or dict.

    Returns:
        The id as a string, or ``None`` when the result carries none. Returned
        as a string because the id only ever travels back to the server, and a
        stringly id cannot be mistaken for a local integer row id.
    """
    result = _field(status, "result")
    if result is None:
        return None
    media_id = _field(result, "media_id")
    if media_id is None or media_id == "":
        return None
    return str(media_id)


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
        The number of jobs actually transitioned -- state changes only, so a
        progress update on a job that is already running is not counted. A status
        for an unknown remote id, for a job already in the reported state, or
        with an unrecognised ``status`` string contributes nothing.
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
            updated = registry.mark_remote_done(
                job.job_id, remote_media_id=_remote_media_id(status)
            )
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
            # Only a real transition counts. When the job is already PARSING the
            # server is simply re-reporting it, and handing back ``job`` here
            # would inflate the count on every poll -- ``job`` is a ``replace()``
            # copy taken before this call, so it is evidence of nothing.
            updated = (
                registry.mark_parsing(job.job_id)
                if job.state is not IngestJobState.PARSING
                else None
            )
            # Progress is recorded either way: a running job that reports 40%
            # has not changed state, but the queue should still show 40%.
            if progress is not None:
                registry.update_progress(job.job_id, progress=progress)
        else:
            # QUEUED: the server has not started it, which is where a freshly
            # submitted job already sits. Nothing to change.
            updated = None

        if updated is not None:
            applied += 1

    return applied
