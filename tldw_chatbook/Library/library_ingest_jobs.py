"""Pure ingest job model and in-memory registry for the Library ingest canvas.

This module is intentionally Textual-free (stdlib + loguru only) so it can be
unit tested without booting the TUI and can be imported by both the app-level
queue-runner and the Library canvas widget without pulling in UI code.

Accepted v1 limits (binding; do not "fix" these without a follow-up task):

- **In-memory only.** The registry has no persistence layer. All job history
  -- queued, running, done, and failed jobs alike -- dies with the app
  process. There is no restart/resume.
- **Quitting waits for the in-flight file to finish; queued jobs are
  lost.** The queue-runner worker (``_run_library_ingest_queue`` in
  ``app.py``) runs on a background thread, and the app waits for that
  thread to join before it finishes shutting down -- so a job that is
  already ``RUNNING`` when the user quits completes its ingest and its
  ``mark_done``/``mark_failed`` DB write normally; it is not silently
  dropped mid-file. Any jobs still ``QUEUED`` behind it, however, are never
  claimed -- the runner exits with the process, and nothing resumes them
  or marks them ``FAILED`` on the next launch (matching the legacy
  ``TAB_INGEST`` behavior this replaces).
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

# The default chunk size (in words) used whenever a caller doesn't supply
# one -- the lowest-level pure module in the Library ingest stack, so
# ``library_ingest_state.py`` and ``app.py``'s ``submit_library_ingest_job``
# both import this rather than each hardcoding their own copy of ``500``.
DEFAULT_CHUNK_SIZE: int = 500


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
        superseded: ``True`` once a ``FAILED`` job has been retried via
            ``requeue`` (L3b AB wave, B1) -- hides it from ``jobs()``/
            ``counts()`` and makes ``mark_running``/``mark_done``/
            ``mark_failed``/``requeue``/``dismiss`` safe no-ops against its
            ``job_id``. Never set anywhere except inside ``requeue``.
        dismissed: ``True`` once a ``FAILED`` job has been dismissed via
            ``dismiss`` (L3b AB wave, B2) -- same hiding/no-op effect as
            ``superseded``, but a distinct field so the two "this job is
            gone" reasons (auto-superseded-by-retry vs. user-dismissed)
            never get confused with one another. Never set anywhere except
            inside ``dismiss``.
    """

    job_id: str
    source_path: str
    title: str = ""
    author: str = ""
    keywords: tuple[str, ...] = ()
    perform_analysis: bool = False
    chunk_enabled: bool = False
    chunk_size: int = DEFAULT_CHUNK_SIZE
    state: IngestJobState = IngestJobState.QUEUED
    detected_type: str = ""
    media_id: int | None = None
    error: str = ""
    submitted_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    superseded: bool = False
    dismissed: bool = False


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

        Reentrancy contract: a listener must not mutate the registry (call
        ``submit``/``mark_running``/``mark_done``/``mark_failed``/
        ``requeue``) from inside its own callback. ``_notify_listeners``
        iterates a snapshot of ``self._listeners``, so add/remove-ing a
        *listener* mid-callback is safe -- but a mutating call would
        recursively re-enter ``_notify_listeners`` while the outer call is
        still iterating, and (for ``submit``/``requeue``) would append to
        ``self._jobs`` while an outer mutation's own state may still be in
        the middle of being applied. Nothing in this class enforces the
        contract; it is documented here because callers (e.g.
        ``LibraryScreen._handle_library_ingest_registry_changed``) depend
        on it and must only read (``jobs()``/``counts()``), never mutate.
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
                # block the other listeners from running. loguru's
                # traceback capture is `.opt(exception=True)`, NOT the
                # stdlib `exc_info=True` kwarg -- the latter is a silent
                # no-op under loguru and would otherwise drop the traceback
                # entirely.
                logger.opt(exception=True).debug("LibraryIngestJobRegistry listener raised")

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
        chunk_size: int = DEFAULT_CHUNK_SIZE,
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
            unknown or hidden (``superseded``/``dismissed``). Unknown ids
            never raise.

        No state-machine guard beyond the hidden check above: this
        unconditionally overwrites whatever state a *visible* job was in
        (it does not check that it was ``QUEUED`` first). The queue-runner
        (``_run_library_ingest_queue`` in ``app.py``) is the sole intended
        caller of ``mark_running``/``mark_done``/``mark_failed``, and it
        always drives a job either ``queued`` -> ``running`` ->
        ``done``/``failed``, or straight ``queued`` -> ``failed`` as the
        fast-fail path (an error, e.g. an undetectable file type, raised
        before ``mark_running`` is ever reached). Calling this out of that
        order is not rejected here.
        """
        index = self._find_index(job_id)
        if index is None:
            return None
        current = self._jobs[index]
        if current.superseded or current.dismissed:
            return None
        updated = replace(
            current,
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
            unknown or hidden (``superseded``/``dismissed``). Unknown ids
            never raise.

        No state-machine guard beyond the hidden check above: see
        ``mark_running``'s docstring. The queue-runner is the sole intended
        caller and only ever reaches this after a preceding
        ``mark_running`` on the same job.
        """
        index = self._find_index(job_id)
        if index is None:
            return None
        current = self._jobs[index]
        if current.superseded or current.dismissed:
            return None
        updated = replace(
            current,
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
            unknown or hidden (``superseded``/``dismissed``). Unknown ids
            never raise.

        No state-machine guard beyond the hidden check above: see
        ``mark_running``'s docstring. The queue-runner is the sole intended
        caller, and reaches this from either ``running`` (a mid-ingest
        exception) or, as the fast-fail path, directly from ``queued`` (an
        error before ``mark_running`` was ever called).
        """
        index = self._find_index(job_id)
        if index is None:
            return None
        current = self._jobs[index]
        if current.superseded or current.dismissed:
            return None
        updated = replace(
            current,
            state=IngestJobState.FAILED,
            error=error,
            finished_at=time.monotonic(),
        )
        self._jobs[index] = updated
        self._notify_listeners()
        return replace(updated)

    def requeue(self, job_id: str) -> LibraryIngestJob | None:
        """Append a fresh ``QUEUED`` copy of a ``FAILED`` job, superseding it.

        Only works on a ``FAILED``, not-yet-hidden job -- calling this on a
        job in any other state, an unknown id, or an already
        ``superseded``/``dismissed`` id is a no-op that returns ``None``
        (this rejects retrying the same original twice, which would
        otherwise silently fork duplicate retries off one dead job_id).

        (L3b AB wave, B1) The original failed job is marked ``superseded``
        -- it stays in the registry's internal history (so its data is not
        lost) but is filtered out of ``jobs()``/``counts()`` from this
        point on, and every further ``mark_running``/``mark_done``/
        ``mark_failed``/``requeue``/``dismiss`` call against its ``job_id``
        becomes a safe no-op. A brand-new job with a brand-new ``job_id``
        and fresh timestamps is appended, copying only the form fields
        (``source_path``/``title``/``author``/``keywords``/
        ``perform_analysis``/``chunk_enabled``/``chunk_size``) -- so the
        canvas queue shows exactly ONE row per retried file, not two.

        Args:
            job_id: The failed job to requeue.

        Returns:
            The newly appended ``QUEUED`` job (a copy), or ``None`` when
            ``job_id`` is unknown, not currently ``FAILED``, or already
            hidden.
        """
        index = self._find_index(job_id)
        if index is None:
            return None
        source = self._jobs[index]
        if source.state != IngestJobState.FAILED or source.superseded or source.dismissed:
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
        self._jobs[index] = replace(source, superseded=True)
        self._jobs.append(new_job)
        self._notify_listeners()
        return replace(new_job)

    def dismiss(self, job_id: str) -> LibraryIngestJob | None:
        """Hide a ``FAILED`` job from ``jobs()``/``counts()``.

        (L3b AB wave, B2) Valid ONLY for a ``FAILED``, not-yet-hidden job --
        calling this on a job in any other state, an unknown id, or an
        already ``superseded``/``dismissed`` id is a no-op that returns
        ``None`` and does not fire the listener. On success the job is
        marked ``dismissed`` (kept in the registry's internal history, like
        ``requeue``'s ``superseded`` marking, but filtered out of every
        public read from this point on) and every further
        ``mark_running``/``mark_done``/``mark_failed``/``requeue``/
        ``dismiss`` call against its ``job_id`` becomes a safe no-op.

        Args:
            job_id: The failed job to dismiss.

        Returns:
            The dismissed job (a copy, ``dismissed=True``), or ``None``
            when ``job_id`` is unknown, not currently ``FAILED``, or
            already hidden.
        """
        index = self._find_index(job_id)
        if index is None:
            return None
        current = self._jobs[index]
        if current.state != IngestJobState.FAILED or current.superseded or current.dismissed:
            return None
        updated = replace(current, dismissed=True)
        self._jobs[index] = updated
        self._notify_listeners()
        return replace(updated)

    def clear_finished(self) -> int:
        """Remove every ``DONE``/``FAILED`` job (visible or already hidden).

        (L3b AB wave, B2) Unlike ``dismiss``/``requeue``'s ``superseded``
        marking (a soft hide that keeps history around), this is an actual
        removal from the registry's internal list -- it doubles as the only
        way to garbage-collect the ``superseded``/``dismissed`` jobs that
        accumulate there over a long session (the registry is in-memory-only
        with no other eviction; see the module docstring's accepted v1
        limits). Queued/running jobs are never touched.

        Returns:
            The number of jobs actually removed (0 when there was nothing
            to clear -- a no-op that does not fire the listener).
        """
        before = len(self._jobs)
        self._jobs = [
            job
            for job in self._jobs
            if job.state not in (IngestJobState.DONE, IngestJobState.FAILED)
        ]
        removed = before - len(self._jobs)
        if removed:
            self._notify_listeners()
        return removed

    # -- reads -----------------------------------------------------

    def jobs(self) -> tuple[LibraryIngestJob, ...]:
        """Return an immutable, newest-first snapshot of all visible jobs.

        Superseded (B1, retried) and dismissed (B2) jobs are filtered out --
        see their respective docstrings.

        Returns:
            A tuple of job copies, most-recently-submitted/requeued first.
            Mutating an element of the returned tuple can never affect
            registry state (see the module docstring).
        """
        return tuple(
            replace(job)
            for job in reversed(self._jobs)
            if not (job.superseded or job.dismissed)
        )

    def counts(self) -> dict[str, int]:
        """Return per-state job counts across visible jobs only.

        Superseded (B1, retried) and dismissed (B2) jobs are excluded --
        see their respective docstrings.

        Returns:
            A dict keyed by every ``IngestJobState`` value (``"queued"``,
            ``"running"``, ``"done"``, ``"failed"``), always present even
            when zero.
        """
        counts = {state.value: 0 for state in IngestJobState}
        for job in self._jobs:
            if job.superseded or job.dismissed:
                continue
            counts[job.state.value] += 1
        return counts
