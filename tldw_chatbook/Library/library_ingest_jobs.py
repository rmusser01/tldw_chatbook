"""Pure ingest job model and in-memory registry for the Library ingest canvas.

This module is intentionally Textual-free (stdlib + loguru only) so it can be
unit tested without booting the TUI and can be imported by both the app-level
queue-runner and the Library canvas widget without pulling in UI code.

Accepted v1 limits (binding; do not "fix" these without a follow-up task):

- **In-memory only.** The registry has no persistence layer. All job history
  -- queued, running, done, and failed jobs alike -- dies with the app
  process. There is no restart/resume.
- **A running job dies on quit.** If the app exits while a job is
  ``RUNNING``, that job's progress is lost; it is not resumed or marked
  ``FAILED`` on the next launch (matching the legacy ``TAB_INGEST`` behavior
  this replaces).
- **Serial queue.** Exactly one job runs at a time; ``next_queued()`` hands
  jobs out one at a time in FIFO order. Parallel ingestion is a follow-up.

Threading contract: ``LibraryIngestJobRegistry`` itself is deliberately
**mutable** (it is the single source of truth for the ingest queue) but it
does **no internal locking**. Every mutating method (``submit``,
``mark_running``, ``mark_done``, ``mark_failed``, ``requeue``) -- and reads of
``runner_active`` -- MUST only be called from the UI thread. A background
queue-runner thread that needs to touch the registry must marshal every call
through something like Textual's ``App.call_from_thread`` first. This module
does not enforce that contract; it is documented here because callers
(the Task 2 queue-runner) depend on it.

Individual ``LibraryIngestJob`` records are plain (non-frozen) dataclasses
for construction convenience, but the registry never hands out a reference
to its own internal copy: every job returned by a public method (including
inside the ``jobs()`` snapshot) is a fresh ``dataclasses.replace()`` copy.
Every field on ``LibraryIngestJob`` is an immutable value type (``str``,
``tuple[str, ...]``, ``bool``, ``int``, the ``IngestJobState`` enum, or
``float | None``), so a shallow copy is equivalent to a deep copy -- mutating
a returned job (or a job inside a ``jobs()`` snapshot) can never corrupt
registry state. Internally, state transitions replace the stored job with a
new instance (``dataclasses.replace``) rather than mutating it in place, so
a "replaced-on-transition" job is always safe to hand out to callers, too.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from loguru import logger


class IngestJobState(str, Enum):
    """Lifecycle states for a single-file Library ingest job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class LibraryIngestJob:
    """A single-file ingest job tracked by :class:`LibraryIngestJobRegistry`.

    Attributes:
        job_id: Registry-assigned id, formatted ``"ingest-job-{n}"`` with
            ``n`` monotonically increasing (including across ``requeue``).
        source_path: The file path submitted for ingest. Display surfaces
            should render only the basename; this field carries the full
            (validated-by-the-caller) path.
        title: Optional user-supplied title form field.
        author: Optional user-supplied author form field.
        keywords: User-supplied keywords, always stored as a tuple.
        perform_analysis: Whether post-ingest analysis was requested.
        chunk_enabled: Whether chunking was requested.
        chunk_size: Requested chunk size (only meaningful when
            ``chunk_enabled`` is ``True``).
        state: The job's current lifecycle state.
        detected_type: The file type detected by the ingest seam. Empty
            until the job starts running.
        media_id: The resulting media row id. Set only on success.
        error: A sanitized, single-line failure message. Set only on
            failure.
        submitted_at: ``time.monotonic()`` timestamp taken at submission.
        started_at: ``time.monotonic()`` timestamp taken when the job
            transitions to ``RUNNING``; ``None`` until then.
        finished_at: ``time.monotonic()`` timestamp taken when the job
            transitions to ``DONE`` or ``FAILED``; ``None`` until then.
    """

    job_id: str
    source_path: str
    title: str = ""
    author: str = ""
    keywords: tuple[str, ...] = ()
    perform_analysis: bool = False
    chunk_enabled: bool = False
    chunk_size: int = 500
    state: IngestJobState = IngestJobState.QUEUED
    detected_type: str = ""
    media_id: int | None = None
    error: str = ""
    submitted_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None


class LibraryIngestJobRegistry:
    """In-memory, UI-thread-only registry for Library ingest jobs.

    See the module docstring for the accepted v1 limits (in-memory, serial,
    no locking) and the threading contract (UI-thread-only mutation).
    """

    def __init__(self) -> None:
        # Insertion order (submission/requeue order); oldest first. `jobs()`
        # reverses this for its newest-first snapshot contract.
        self._jobs: list[LibraryIngestJob] = []
        self._next_id: int = 1
        self._listeners: list[Callable[[], None]] = []
        # Plain flag flipped by the runner owner (e.g. the app's queue-runner
        # worker). No locking -- see the module docstring's threading
        # contract; this attribute is UI-thread-only like everything else.
        self.runner_active: bool = False

    # -- id allocation -----------------------------------------------------

    def _allocate_job_id(self) -> str:
        job_id = f"ingest-job-{self._next_id}"
        self._next_id += 1
        return job_id

    # -- listeners -----------------------------------------------------

    def add_listener(self, callback: Callable[[], None]) -> None:
        """Register a callback to be fired after every successful mutation.

        Args:
            callback: A zero-argument callable. Invoked synchronously,
                on the UI thread, after ``submit``/``mark_running``/
                ``mark_done``/``mark_failed``/``requeue`` succeed.
        """
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], None]) -> None:
        """Unregister a previously added listener.

        Args:
            callback: The callback passed to a prior ``add_listener`` call.
                Removing a callback that was never added (or already
                removed) is a no-op, not an error.
        """
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def _notify_listeners(self) -> None:
        # Iterate a snapshot so a listener that adds/removes listeners
        # mid-callback can't corrupt this loop.
        for callback in tuple(self._listeners):
            try:
                callback()
            except Exception:
                # A broken listener must never corrupt registry state or
                # block the other listeners from running.
                logger.debug("LibraryIngestJobRegistry listener raised", exc_info=True)

    # -- mutations -----------------------------------------------------

    def submit(
        self,
        *,
        source_path: str,
        title: str = "",
        author: str = "",
        keywords: tuple[str, ...] = (),
        perform_analysis: bool = False,
        chunk_enabled: bool = False,
        chunk_size: int = 500,
    ) -> LibraryIngestJob:
        """Append a new ``QUEUED`` job.

        Args:
            source_path: The file path to ingest.
            title: Optional title form field.
            author: Optional author form field.
            keywords: Keywords form field; coerced to a tuple.
            perform_analysis: Whether to run post-ingest analysis.
            chunk_enabled: Whether to chunk the ingested content.
            chunk_size: Requested chunk size when ``chunk_enabled``.

        Returns:
            The newly created ``QUEUED`` job (a registry-owned copy).
        """
        job = LibraryIngestJob(
            job_id=self._allocate_job_id(),
            source_path=source_path,
            title=title,
            author=author,
            keywords=tuple(keywords),
            perform_analysis=perform_analysis,
            chunk_enabled=chunk_enabled,
            chunk_size=chunk_size,
            state=IngestJobState.QUEUED,
            submitted_at=time.monotonic(),
        )
        self._jobs.append(job)
        self._notify_listeners()
        return replace(job)

    def next_queued(self) -> LibraryIngestJob | None:
        """Return the oldest still-``QUEUED`` job, or ``None`` if none.

        Returns:
            A copy of the oldest queued job in FIFO submission order, or
            ``None`` when no job is queued.
        """
        for job in self._jobs:
            if job.state == IngestJobState.QUEUED:
                return replace(job)
        return None

    def _find_index(self, job_id: str) -> int | None:
        for index, job in enumerate(self._jobs):
            if job.job_id == job_id:
                return index
        return None

    def mark_running(self, job_id: str, *, detected_type: str = "") -> LibraryIngestJob | None:
        """Transition a job to ``RUNNING`` and stamp ``started_at``.

        Args:
            job_id: The job to transition.
            detected_type: The file type detected by the ingest seam.

        Returns:
            The updated job (a copy), or ``None`` when ``job_id`` is
            unknown. Unknown ids never raise.
        """
        index = self._find_index(job_id)
        if index is None:
            return None
        updated = replace(
            self._jobs[index],
            state=IngestJobState.RUNNING,
            detected_type=detected_type,
            started_at=time.monotonic(),
        )
        self._jobs[index] = updated
        self._notify_listeners()
        return replace(updated)

    def mark_done(self, job_id: str, *, media_id: int) -> LibraryIngestJob | None:
        """Transition a job to ``DONE`` and stamp ``finished_at``.

        Args:
            job_id: The job to transition.
            media_id: The resulting media row id.

        Returns:
            The updated job (a copy), or ``None`` when ``job_id`` is
            unknown. Unknown ids never raise.
        """
        index = self._find_index(job_id)
        if index is None:
            return None
        updated = replace(
            self._jobs[index],
            state=IngestJobState.DONE,
            media_id=media_id,
            finished_at=time.monotonic(),
        )
        self._jobs[index] = updated
        self._notify_listeners()
        return replace(updated)

    def mark_failed(self, job_id: str, *, error: str) -> LibraryIngestJob | None:
        """Transition a job to ``FAILED`` and stamp ``finished_at``.

        Args:
            job_id: The job to transition.
            error: A sanitized, single-line failure message.

        Returns:
            The updated job (a copy), or ``None`` when ``job_id`` is
            unknown. Unknown ids never raise.
        """
        index = self._find_index(job_id)
        if index is None:
            return None
        updated = replace(
            self._jobs[index],
            state=IngestJobState.FAILED,
            error=error,
            finished_at=time.monotonic(),
        )
        self._jobs[index] = updated
        self._notify_listeners()
        return replace(updated)

    def requeue(self, job_id: str) -> LibraryIngestJob | None:
        """Append a fresh ``QUEUED`` copy of a ``FAILED`` job.

        Only works on ``FAILED`` jobs -- calling this on a job in any other
        state (or an unknown id) is a no-op that returns ``None``. The
        original failed job is left untouched in the registry (so it still
        shows up in ``jobs()``/history); a brand-new job with a brand-new
        ``job_id`` and fresh timestamps is appended, copying only the form
        fields (``source_path``/``title``/``author``/``keywords``/
        ``perform_analysis``/``chunk_enabled``/``chunk_size``).

        Args:
            job_id: The failed job to requeue.

        Returns:
            The newly appended ``QUEUED`` job (a copy), or ``None`` when
            ``job_id`` is unknown or not currently ``FAILED``.
        """
        index = self._find_index(job_id)
        if index is None:
            return None
        source = self._jobs[index]
        if source.state != IngestJobState.FAILED:
            return None
        new_job = LibraryIngestJob(
            job_id=self._allocate_job_id(),
            source_path=source.source_path,
            title=source.title,
            author=source.author,
            keywords=source.keywords,
            perform_analysis=source.perform_analysis,
            chunk_enabled=source.chunk_enabled,
            chunk_size=source.chunk_size,
            state=IngestJobState.QUEUED,
            submitted_at=time.monotonic(),
        )
        self._jobs.append(new_job)
        self._notify_listeners()
        return replace(new_job)

    # -- reads -----------------------------------------------------

    def jobs(self) -> tuple[LibraryIngestJob, ...]:
        """Return an immutable, newest-first snapshot of all known jobs.

        Returns:
            A tuple of job copies, most-recently-submitted/requeued first.
            Mutating an element of the returned tuple can never affect
            registry state (see the module docstring).
        """
        return tuple(replace(job) for job in reversed(self._jobs))

    def counts(self) -> dict[str, int]:
        """Return per-state job counts.

        Returns:
            A dict keyed by every ``IngestJobState`` value (``"queued"``,
            ``"running"``, ``"done"``, ``"failed"``), always present even
            when zero.
        """
        counts = {state.value: 0 for state in IngestJobState}
        for job in self._jobs:
            counts[job.state.value] += 1
        return counts
