"""Map a server ingest job's status onto a local ingest job state.

Kept pure and separate from ``server_ingest_request.py`` (which maps the
*outgoing* submission) so the polling worker's decisions can be unit tested
without a server (task-684.2).

``MediaIngestJobStatus.status`` is an unpinned ``str`` -- the schema declares no
``Literal`` -- so this module cannot rely on an exhaustive set. The vocabulary
below is drawn from the client's own tests
(``Tests/tldw_api/test_media_ingest_jobs_client.py`` exercises ``queued``,
``completed`` and ``cancelled``) and from the sibling job APIs in ``tldw_api/``,
which declare ``queued|running|completed|failed|cancelled`` with
``processing``/``canceled`` spelling variants.

The unknown case is the one that matters most: an unrecognised status resolves
to ``None`` and is *not* terminal, so a job keeps being watched rather than
being silently declared finished. Declaring success on a status we do not
understand is the same class of defect as reporting a successful ingest that
stored nothing (task-677) -- the user gets an outcome they cannot act on.
"""

from __future__ import annotations

from tldw_chatbook.Library.library_ingest_jobs import IngestJobState

#: Server status -> local state. Deliberately never maps to
#: ``IngestJobState.WRITING``: that is the local single-writer persistence
#: stage, which a server-side job never passes through.
_STATE_BY_SERVER_STATUS: dict[str, IngestJobState] = {
    "queued": IngestJobState.QUEUED,
    "pending": IngestJobState.QUEUED,
    "running": IngestJobState.PARSING,
    "processing": IngestJobState.PARSING,
    "in_progress": IngestJobState.PARSING,
    "completed": IngestJobState.DONE,
    "complete": IngestJobState.DONE,
    "success": IngestJobState.DONE,
    "succeeded": IngestJobState.DONE,
    "failed": IngestJobState.FAILED,
    "error": IngestJobState.FAILED,
    "cancelled": IngestJobState.CANCELLED,
    "canceled": IngestJobState.CANCELLED,
}


def _normalise(status: str | None) -> str:
    """Return ``status`` lowercased and stripped, or ``""`` when absent."""
    return (status or "").strip().lower()


def local_state_for_server_status(status: str | None) -> IngestJobState | None:
    """Return the local state for a server job status.

    Args:
        status: The server's ``MediaIngestJobStatus.status`` value.

    Returns:
        The matching :class:`IngestJobState`, or ``None`` when the status is
        absent or unrecognised -- in which case the caller should leave the job
        as it is and keep watching, never assume it finished.
    """
    return _STATE_BY_SERVER_STATUS.get(_normalise(status))


def is_terminal_server_status(status: str | None) -> bool:
    """Return whether a server status means the job will not change again.

    An unrecognised status is deliberately *not* terminal: stopping on a status
    we do not understand would abandon a job that may still be running, whereas
    continuing to watch it is recoverable.

    Args:
        status: The server's ``MediaIngestJobStatus.status`` value.

    Returns:
        True only for a status known to be final.
    """
    state = local_state_for_server_status(status)
    return state in (
        IngestJobState.DONE,
        IngestJobState.FAILED,
        IngestJobState.CANCELLED,
    )
