"""Pure display-state for the Library ingest canvas.

Renders the app-level Library ingest job registry (``library_ingest_jobs.py``)
plus a small local form echo into the immutable state
``LibraryIngestCanvas`` (the widget in ``Widgets/Library/library_ingest_canvas.py``)
renders from. Textual-free (stdlib only) so it is unit-testable without
booting the TUI, mirroring ``library_notes_sync_state.py``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import PurePath
from typing import Sequence

from tldw_chatbook.Library.library_ingest_jobs import (
    DEFAULT_CHUNK_SIZE,
    IngestJobState,
    LibraryIngestJob,
)
from tldw_chatbook.Local_Ingestion.local_file_ingestion import get_supported_extensions

# Exact copy values (binding -- see the L3b plan's Global Constraints).
INGEST_HEADER_COPY = "Import media"
SERVER_QUIET_LINE_COPY = "ingest runs on Local"
MEDIA_DB_UNAVAILABLE_COPY = "Media database is unavailable."
INGEST_UNAVAILABLE_COPY = "Ingest is unavailable in this runtime."
QUEUE_HEADING_COPY = "Queue"
QUEUE_EMPTY_COPY = "No ingest jobs yet."
START_QUIET_LINE_COPY = "Enter a file path to start."

# Re-exported from library_ingest_jobs.py (the lowest-level pure module in
# the Library ingest stack) rather than redefined here -- kept as a module
# attribute of this file too (not just imported-and-discarded) since
# existing consumers/tests import ``DEFAULT_CHUNK_SIZE`` from
# ``library_ingest_state``.
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 5000

# Queue row state glyphs (binding).
_GLYPH_ACTIVE = "●"  # "●" -- queued or running
_GLYPH_DONE = "✓"  # "✓"
_GLYPH_FAILED = "✗"  # "✗"

# L4 (fix batch F1b): the failed-row line's "Supported types: ..." tail
# moves to the form as its own always-visible line, prefixed with this
# exact copy.
SUPPORTED_TYPES_PREFIX = "Supported: "

# L4: the marker `local_file_ingestion.py`'s "Unsupported file type" error
# copy uses to separate the offending extension from its own supported-list
# tail -- shared here so the queue row's ``short_error`` split can never
# drift out of sync with that error string's exact punctuation.
_SUPPORTED_TYPES_ERROR_MARKER = " Supported types:"


def _supported_types_line() -> str:
    """Build the ingest form's supported-extensions line.

    Derived live from ``get_supported_extensions()`` -- the exact same
    function whose values back ``local_file_ingestion.py``'s "Unsupported
    file type" error copy -- rather than a hardcoded duplicate list, so the
    form's line and the runner's error copy can never drift apart (the A2
    lesson: never hand-duplicate a value that already has a canonical
    source).

    Returns:
        ``"Supported: "`` followed by every supported extension (upper-
        cased, dot stripped, comma-joined), in ``get_supported_extensions()``'s
        own media-type -> extension-list order.
    """
    extensions = [
        ext.lstrip(".").upper()
        for exts in get_supported_extensions().values()
        for ext in exts
    ]
    return SUPPORTED_TYPES_PREFIX + ", ".join(extensions)


@dataclass
class LibraryIngestFormState:
    """Mutable form echo for the ingest canvas.

    Owned by the screen as a single bundled field (``self._library_ingest_form``)
    rather than a scatter of scalar attributes, and reset wholesale to
    defaults on rail re-entry into Ingest (see
    ``_reset_library_ingest_transient_state``). Every field here is display
    text only -- validated/coerced values (a resolved path, an int chunk
    size, a keywords tuple) are derived at submit time, never stored back
    into this echo.

    Attributes:
        path: The local file path as typed/picked, unvalidated.
        title: Optional title form field, as typed.
        author: Optional author form field, as typed.
        keywords: Comma-separated keywords, as typed (not yet split).
        analyze: Whether "Analyze after ingest" is toggled on.
        chunk: Whether "Chunk content" is toggled on.
        chunk_size: The chunk-size field's raw text (display-echo only;
            parsed and clamped to ``[MIN_CHUNK_SIZE, MAX_CHUNK_SIZE]`` at
            submit time, never here).
        advanced_open: Whether the "Advanced options" ``Collapsible`` is
            currently expanded. Synced from the live widget's ``collapsed``
            reactive by the screen's ``Collapsible.Toggled`` handler (both
            a manual click and any future programmatic assignment), and
            read back on every render (``collapsed=not advanced_open``) so
            a recompose -- the analyze/chunk toggle handlers' own, or a
            registry-listener-driven one -- never snaps an expanded panel
            shut out from under the user (mirrors
            ``_library_rag_history_collapsed``/
            ``sync_library_rag_history_collapsed`` in ``library_screen.py``).
    """

    path: str = ""
    title: str = ""
    author: str = ""
    keywords: str = ""
    analyze: bool = False
    chunk: bool = False
    chunk_size: str = str(DEFAULT_CHUNK_SIZE)
    advanced_open: bool = False


@dataclass(frozen=True)
class IngestQueueRow:
    """One rendered row in the ingest canvas's job queue.

    Attributes:
        job_id: The registry-assigned job id (``"ingest-job-{n}"``).
        glyph: The row's leading state glyph -- ``"●"`` for queued/running,
            ``"✓"`` for done, ``"✗"`` for failed.
        line: The full rendered row text (binding formats -- see
            ``build_library_ingest_state``). Raw, unescaped: the widget
            layer is responsible for markup-escaping this before it reaches
            a rendered label (a source filename can contain Rich markup
            syntax like ``[/bracket]``).
        can_open: True only for a ``done`` job with a resolved ``media_id``
            -- gates the row's "Open in Library" action.
        can_retry: True only for a ``failed`` job whose ``permanent`` flag
            is ``False`` (M4, fix batch F1b) -- gates the row's "Retry"
            action. A ``permanent`` failure (an unsupported file type or a
            missing source file) will fail the exact same way every time,
            so Retry is withheld entirely rather than offering dead bait;
            ``can_dismiss`` stays available either way.
        can_dismiss: True only for a ``failed`` job -- gates the row's
            "Dismiss" action (L3b AB wave, B2). Currently identical to
            ``can_retry`` (both actions are FAILED-only per the registry's
            ``dismiss``/``requeue`` contracts) but kept as its own field
            rather than reusing ``can_retry`` so the two actions stay
            independently testable if a future change ever lets one apply
            where the other doesn't.
        media_id: The job's resulting media id, when known (``done`` jobs
            only).
    """

    job_id: str
    glyph: str
    line: str
    can_open: bool
    can_retry: bool
    can_dismiss: bool = False
    media_id: int | None = None


@dataclass(frozen=True)
class LibraryIngestCanvasState:
    """Full display state for the Library ingest canvas.

    Attributes:
        header: The canvas header line (always ``"Import media"``).
        server_quiet_line: A muted informational line (``"ingest runs on
            Local"``) shown only when the Library's active runtime source
            is ``"server"`` -- ingest always targets the local media store
            regardless of the browsing scope. Empty when not shown.
        unavailable_line: A blocked-state line explaining why Start is
            disabled, or empty when neither gate is tripped. Precedence:
            a missing registry seam (``registry_available=False``, the
            app-level ingest queue itself is absent) always wins over a
            missing media DB (``media_db_available=False``) -- rendering
            both would be redundant, since without a registry the media-db
            gate can never even be checked in production.
        form: The form echo (see ``LibraryIngestFormState``).
        supported_types_line: (L4, fix batch F1b) A muted, always-visible
            line listing every ingestible extension, built live from
            ``get_supported_extensions()`` (see ``_supported_types_line``)
            rather than hardcoded -- rendered under the Browse… button so
            it stays reachable without being repeated on every failed queue
            row (see ``IngestQueueRow.line``'s ``short_error``).
        start_enabled: Whether the "Start ingest" button is enabled --
            requires a working registry, an available media DB, and a
            non-blank typed path.
        start_quiet_line: (L3b AB wave, A4) A muted line (``"Enter a file
            path to start."``) rendered adjacent to the Start button when
            the path field is blank but both seams are otherwise available.
            Empty once a path is typed, or whenever ``unavailable_line`` is
            already showing -- the db-unavailable/ingest-unavailable lines
            always take precedence so at most one gate line ever renders.
        queue_heading: The queue section heading (always ``"Queue"``).
        queue_counts_line: A per-state job counts summary (L3b AB wave, A2)
            -- empty when the queue itself is empty (``QUEUE_EMPTY_COPY``
            covers that case instead); otherwise only non-zero states,
            queued -> running -> done -> failed order.
        queue_rows: Newest-first queue rows (mirrors the registry's own
            ``jobs()`` snapshot order -- callers pass that tuple straight
            through, unsorted).
        queue_show_clear_finished: (L3b AB wave, B2) Whether the "Clear
            finished" button should render below the queue rows -- true
            whenever at least one ``done`` or ``failed`` job is present in
            ``jobs`` (computed from the raw jobs, not from ``queue_rows``,
            so a defensively-malformed done-without-``media_id`` row --
            which renders with ``can_open=False`` -- still counts).
    """

    header: str
    server_quiet_line: str
    unavailable_line: str
    form: LibraryIngestFormState
    supported_types_line: str
    start_enabled: bool
    start_quiet_line: str
    queue_heading: str
    queue_counts_line: str
    queue_rows: tuple[IngestQueueRow, ...]
    queue_show_clear_finished: bool


def _basename(source_path: str) -> str:
    """Return a path's display basename, falling back to the raw string."""
    return PurePath(source_path).name or source_path


def _format_elapsed(
    started_at: float | None, finished_at: float | None, *, now: float
) -> str:
    """Format a done job's run time as ``"Ns"`` or ``"Nm Ss"``.

    Args:
        started_at: ``time.monotonic()`` timestamp when the job started
            running, or ``None`` (defensive -- should not happen for a
            ``done`` job).
        finished_at: ``time.monotonic()`` timestamp when the job finished,
            or ``None`` (defensive fallback: ``now`` is used instead so a
            malformed job still renders a sane elapsed value rather than
            crashing).
        now: The caller-supplied "current" monotonic time, used only as
            the ``finished_at`` fallback described above.

    Returns:
        ``"0s"`` when ``started_at`` is unknown; otherwise the elapsed
        duration rendered as ``"Ns"`` under a minute, or ``"Nm Ss"`` at or
        above a minute.
    """
    if started_at is None:
        return "0s"
    end = finished_at if finished_at is not None else now
    total_seconds = max(0, int(round(end - started_at)))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}m {seconds}s"


def _build_queue_row(job: LibraryIngestJob, *, now: float) -> IngestQueueRow:
    """Build one ``IngestQueueRow`` from a registry job snapshot.

    Binding row-line formats (see the L3b plan):

    - running: ``"● running · {basename}"``, plus
      ``" · {detected_type}"`` appended when the seam has reported one.
    - queued: ``"● queued · {basename}"``.
    - done: ``"✓ done · {basename} · {elapsed}"``.
    - failed: ``"✗ failed · {basename} · {short_error}"``, where
      ``short_error`` (L4, fix batch F1b) drops a trailing
      ``" Supported types: ..."`` tail from ``job.error`` -- that list now
      lives on the form as ``supported_types_line`` instead, always visible
      rather than repeated on every failed row. An error without that exact
      marker passes through whole.
    """
    basename = _basename(job.source_path)
    if job.state == IngestJobState.RUNNING:
        line = f"{_GLYPH_ACTIVE} running · {basename}"
        if job.detected_type:
            line += f" · {job.detected_type}"
        return IngestQueueRow(
            job_id=job.job_id,
            glyph=_GLYPH_ACTIVE,
            line=line,
            can_open=False,
            can_retry=False,
            media_id=job.media_id,
        )
    if job.state == IngestJobState.QUEUED:
        return IngestQueueRow(
            job_id=job.job_id,
            glyph=_GLYPH_ACTIVE,
            line=f"{_GLYPH_ACTIVE} queued · {basename}",
            can_open=False,
            can_retry=False,
            media_id=job.media_id,
        )
    if job.state == IngestJobState.DONE:
        elapsed = _format_elapsed(job.started_at, job.finished_at, now=now)
        return IngestQueueRow(
            job_id=job.job_id,
            glyph=_GLYPH_DONE,
            line=f"{_GLYPH_DONE} done · {basename} · {elapsed}",
            can_open=job.media_id is not None,
            can_retry=False,
            media_id=job.media_id,
        )
    # FAILED -- the only remaining IngestJobState member.
    short_error = job.error.split(_SUPPORTED_TYPES_ERROR_MARKER)[0].rstrip()
    return IngestQueueRow(
        job_id=job.job_id,
        glyph=_GLYPH_FAILED,
        line=f"{_GLYPH_FAILED} failed · {basename} · {short_error}",
        can_open=False,
        can_retry=not job.permanent,
        can_dismiss=True,
        media_id=job.media_id,
    )


def _queue_counts_line(jobs: Sequence[LibraryIngestJob]) -> str:
    """Build the per-state job counts summary line (L3b AB wave, A2).

    Empty when ``jobs`` is empty (the canvas shows ``QUEUE_EMPTY_COPY``
    instead in that case -- see ``build_library_ingest_state``). Otherwise
    lists only the non-zero ``IngestJobState`` values, always in
    ``queued, running, done, failed`` order so the segment order never
    shifts as jobs move between states -- only which segments are present
    does. Each segment is ``"{n} {state}"`` (no "job"/"jobs" noun, unlike
    ``count_noun`` elsewhere in this module -- e.g. ``"2 done · 1
    failed"``), joined by ``" · "``.
    """
    counts = {state.value: 0 for state in IngestJobState}
    for job in jobs:
        counts[job.state.value] += 1
    return " · ".join(
        f"{counts[state.value]} {state.value}"
        for state in IngestJobState
        if counts[state.value]
    )


def build_library_ingest_state(
    jobs: Sequence[LibraryIngestJob],
    *,
    form: LibraryIngestFormState,
    runtime_source: str = "local",
    media_db_available: bool = True,
    registry_available: bool = True,
    now: float | None = None,
) -> LibraryIngestCanvasState:
    """Build the ingest canvas's full display state.

    Args:
        jobs: The registry's current job snapshot (any order accepted --
            typically the registry's own newest-first ``jobs()`` tuple,
            passed straight through into ``queue_rows``).
        form: The current form echo.
        runtime_source: The Library's active runtime scope (``"local"`` or
            ``"server"``); only affects ``server_quiet_line``, since local
            ingest always targets the local media store regardless of
            browsing scope.
        media_db_available: Whether the app's media database seam is
            present. ``False`` blocks Start with ``MEDIA_DB_UNAVAILABLE_COPY``
            (unless ``registry_available`` is also ``False`` -- see
            ``LibraryIngestCanvasState.unavailable_line``).
        registry_available: Whether the app-level ingest job registry seam
            itself is present at all. ``False`` blocks Start with
            ``INGEST_UNAVAILABLE_COPY``, overriding the media-db line.
        now: The "current" monotonic time used for elapsed-time defensive
            fallbacks; defaults to ``time.monotonic()``.

    Returns:
        The canvas's full display state.
    """
    resolved_now = now if now is not None else time.monotonic()
    server_quiet_line = (
        SERVER_QUIET_LINE_COPY
        if str(runtime_source or "local").strip().lower() == "server"
        else ""
    )
    if not registry_available:
        unavailable_line = INGEST_UNAVAILABLE_COPY
    elif not media_db_available:
        unavailable_line = MEDIA_DB_UNAVAILABLE_COPY
    else:
        unavailable_line = ""
    start_enabled = (
        registry_available and media_db_available and bool(form.path.strip())
    )
    # (L3b AB wave, A4) Only render the blank-path nudge when neither
    # blocking gate line is already showing -- at most one gate line ever
    # renders at once.
    start_quiet_line = (
        START_QUIET_LINE_COPY if not unavailable_line and not form.path.strip() else ""
    )
    queue_rows = tuple(_build_queue_row(job, now=resolved_now) for job in jobs)
    queue_show_clear_finished = any(
        job.state in (IngestJobState.DONE, IngestJobState.FAILED) for job in jobs
    )
    return LibraryIngestCanvasState(
        header=INGEST_HEADER_COPY,
        server_quiet_line=server_quiet_line,
        unavailable_line=unavailable_line,
        form=form,
        supported_types_line=_supported_types_line(),
        start_enabled=start_enabled,
        start_quiet_line=start_quiet_line,
        queue_heading=QUEUE_HEADING_COPY,
        queue_counts_line=_queue_counts_line(jobs),
        queue_rows=queue_rows,
        queue_show_clear_finished=queue_show_clear_finished,
    )


def parse_keywords(raw: str) -> tuple[str, ...]:
    """Comma-split a keywords form field, stripping and dropping empties.

    Args:
        raw: The raw, comma-separated keywords text.

    Returns:
        A tuple of non-empty, stripped keyword strings.
    """
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def clamp_chunk_size(raw: str) -> int:
    """Parse and clamp a chunk-size form field at submit time.

    Args:
        raw: The chunk-size field's raw display text.

    Returns:
        The parsed integer clamped to ``[MIN_CHUNK_SIZE, MAX_CHUNK_SIZE]``,
        or ``DEFAULT_CHUNK_SIZE`` when ``raw`` does not parse as an int.
    """
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_CHUNK_SIZE
    return max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, value))
