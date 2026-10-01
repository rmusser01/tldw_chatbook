"""The Library ingest queue: ``LibraryIngestQueueMixin`` and its helpers.

Moved verbatim from ``app.py`` (TASK-33011): the job submission seam, the
parallel-parse coordinator, the writer, remote-ingest polling and the local
STT dispatch, plus the module-level helpers only they use. ``TldwCli`` still
mixes the class in at the same MRO position, and ``tldw_chatbook.app``
re-exports the names callers import from it.

Patch names the moved code reads (``get_cli_setting``, ``logger``,
``run_parse_job``, ``multiprocessing``, ``pending_remote_batches`` and so on)
HERE, not on ``tldw_chatbook.app``: the bodies resolve free names through this
module's globals, so a patch on the app module no longer reaches them.
"""

import asyncio
import concurrent.futures
import contextlib
import functools
import hashlib
import inspect
import multiprocessing
import multiprocessing.connection
import os
import queue
import sqlite3
import sys
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional  # noqa: UP035

from loguru import logger
from textual import work

from tldw_chatbook.config import (
    get_cli_setting,
    get_library_ingest_jobs_db_path,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import (
    DatabaseError as MediaDatabaseError,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import (
    InputError as MediaInputError,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import (
    MediaDatabase,
)
from tldw_chatbook.Library.ingest_analysis import resolve_ingest_analysis_provider
from tldw_chatbook.Library.ingest_capabilities import (
    field_gate_open,
    generic_option_default,
    get_type_group,
)
from tldw_chatbook.Library.ingest_preflight import collect_directory_files
from tldw_chatbook.Library.library_ingest_jobs import (
    DEFAULT_CHUNK_SIZE,
    INGEST_DUPLICATE_PROGRESS_PREFIX,
    ActiveIngestConsentScope,
    ActiveIngestJobRef,
    ActiveIngestSubmissionRefused,
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
    build_active_ingest_consent_scope,
    normalize_active_ingest_source,
)
from tldw_chatbook.Library.server_ingest_reconcile import (
    pending_remote_batches,
    reconcile_remote_ingest_jobs,
)
from tldw_chatbook.Library.server_ingest_request import (
    ServerIngestUnsupported,
    build_server_ingest_kwargs,
)
from tldw_chatbook.Library.web_clip_request import (
    NotAWebClipSource,
    build_web_clip_kwargs,
    clip_failure_reason,
    is_web_clip_source,
)
from tldw_chatbook.Local_Ingestion import FileIngestionError
from tldw_chatbook.Local_Ingestion.ingest_parse_progress import (
    INGEST_PARSE_PROGRESS_FLUSH_SECONDS,
    INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE,
    ParseProgressCoalescer,
    ParseProgressEvent,
    make_parse_progress_event,
)
from tldw_chatbook.Local_Ingestion.ingest_parse_worker import (
    classify_parse_failure,
    initialize_ingest_parse_worker,
    run_parse_job,
)
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    classify_ingest_source,
    persist_parsed_media,
)
from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    PARAKEET_V2_MODEL,
    BatchSTTRoutingError,
    resolve_batch_stt_route,
)
from tldw_chatbook.Research_Workspace.source_operation_store import (
    SourceOperationConflictError,
)
from tldw_chatbook.Research_Workspace.source_operations import (
    SourceOperationStage,
    SourceOperationStatus,
)
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)
from tldw_chatbook.STT.contracts import (
    TRANSCRIPTION_FAILURE_CONTRACT,
    ExecutionDevice,
    FileAudioSource,
    TranscriptionFailureCode,
)
from tldw_chatbook.STT.dispatch_coordinator import LocalSTTDispatchCoordinator
from tldw_chatbook.STT.executor import (
    ExecutorBusyError,
    ExecutorEvent,
    ExecutorFailure,
    ExecutorResult,
    ExecutorUnavailableError,
    LocalSTTExecutor,
    ModelIdentity,
    WorkerPhase,
    snapshot_local_source,
)
from tldw_chatbook.Utils.egress import UrlProvenance

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef


def _sanitize_library_ingest_error_text(message: str) -> str:
    """Reduce a raw error message to a single-line, ``<=200``-char string.

    Shared building block for both ingest-pipeline stages (F3): the write
    stage has a real ``Exception`` (see ``_sanitize_library_ingest_error``,
    below); the parse stage only has the already-``str()``-ed message a
    pool worker's structured failure result carries across the process
    boundary (``ingest_parse_worker.run_parse_job``'s ``"error"`` key) --
    both need the exact same single-line/200-cap treatment before landing
    in a job's ``LibraryIngestJob.error`` field.

    Args:
        message: The raw (possibly multi-line, possibly empty) message.

    Returns:
        The first line, stripped and capped at 200 characters. ``""`` when
        ``message`` is empty or all-whitespace.
    """
    message = message.strip()
    first_line = message.splitlines()[0].strip() if message else ""
    return first_line[:200]


def _sanitize_library_ingest_error(exc: Exception) -> str:
    """Reduce an ingest-time exception to a single-line, capped error string.

    Args:
        exc: The exception raised by the ingest seam.

    Returns:
        The first line of ``str(exc)``, stripped and capped at 200
        characters. Falls back to the exception's class name when
        ``str(exc)`` is empty.
    """
    sanitized = _sanitize_library_ingest_error_text(str(exc))
    return sanitized if sanitized else exc.__class__.__name__[:200]


def _library_ingest_write_failure_category(exc: BaseException) -> str:
    """Classify an exception escaping the ingest WRITE stage.

    (task-14821) The stage covers two different things: refusing an empty
    extraction, which happens BEFORE any write, and a genuine database
    write failure. Exceptions that know which they are declare it on
    ``ingest_error_category``.

    (xhigh review round) Everything else used to default to
    ``"write_error"`` -- the ONE category ``ingest_retry_advice`` still
    answers with "a retry can succeed if the write failure was temporary
    — the file itself parsed fine". So the optimistic branch task-14821
    was filed to remove stayed reachable for every unclassified cause,
    through the default. Only a failure of the database write itself
    earns that name now; an unknown cause is unnamed, and an unnamed
    category is silent (task-14821 AC#2) rather than encouraging.

    Args:
        exc: The exception raised while persisting a parsed payload.

    Returns:
        The ``error_detail`` category token, or ``""`` when the cause is
        not known to be a write failure.
    """
    declared = str(getattr(exc, "ingest_error_category", "") or "").strip()
    if declared:
        return declared
    if isinstance(exc, (MediaDatabaseError, MediaInputError, sqlite3.Error)):
        return "write_error"
    return ""


def _resolve_ingest_cookies_file(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Validate the audio/video panel's ``Cookies file for gated URLs`` value.

    (task-3306 xhigh review round) The value used to be forwarded verbatim
    as ``options["cookies"]``. ``download_video`` treats a string that is
    not an existing file as cookie JSON, so a typo'd or moved path became a
    ``json.JSONDecodeError`` caught into a single "Invalid cookie format"
    warning -- the download then ran un-authenticated and failed later for
    a reason that named neither cookies nor the path. Validating here, at
    the option boundary, is the earliest point this module owns.

    NOTE: the canonical home for per-field validation is the shared
    ``validate_ingest_option_value`` seam in ``library_ingest_state``, which
    is where the sibling text fields (``start_time``/``end_time``) are
    format-gated. This check lives here instead because existence is not a
    format question -- a path can be well-formed at typing time and gone by
    the time the job is claimed, which is exactly when this runs.

    Args:
        raw: The stripped field value; ``""`` means the option is unset.

    Returns:
        ``(cookies_path, problem)``. Exactly one is non-``None`` for a
        non-empty input; both are ``None`` when no cookies were requested.
    """
    if not raw:
        return None, None

    from tldw_chatbook.Utils.path_validation import validate_path_simple

    try:
        # Repo security rule: user-supplied file paths go through
        # path_validation before they become a subprocess/library argument.
        validate_path_simple(os.path.expanduser(raw))
    except ValueError as exc:
        return None, f"Unsafe cookies file path: {_sanitize_library_ingest_error(exc)}"

    candidate = Path(os.path.expanduser(raw))
    if not candidate.is_file():
        return None, f"Cookies file not found: {raw}"
    return str(candidate), None


def _library_ingest_done_progress(
    source_path: str, *, was_duplicate: bool, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a done job's ``progress`` dict from its persisted payload.

    (task-3301) Pure and module-level so the analysis-skipped annotation is
    unit-testable without a writer thread. When the parse payload carries
    ``analysis_skipped_reason`` (analysis was requested but no callable
    provider was configured at dispatch time), the done row says so --
    "analysis skipped: ..." on the row's progress sub-line -- instead of
    the analysis being silently absent. Duplicate-match outcomes keep their
    exact ``INGEST_DUPLICATE_PROGRESS_PREFIX`` message untouched: nothing
    new was imported, so there was nothing to analyze.

    Args:
        source_path: The job's source path (basename feeds the message).
        was_duplicate: Whether the write resolved to an existing item.
        payload: The parse payload that was just persisted.

    Returns:
        The ``progress`` dict for ``LibraryIngestJobRegistry.mark_done``.
    """
    if was_duplicate:
        return {
            "message": (
                f"{INGEST_DUPLICATE_PROGRESS_PREFIX} — "
                "matched an existing item; nothing new was "
                "imported."
            )
        }
    # (task-2016) The basename, not the absolute path: the row line already
    # identifies the file and the details surface carries the full path.
    source_name = Path(source_path).name or source_path
    progress: Dict[str, Any] = {"message": f"Imported {source_name}"}
    skip_reason = str(payload.get("analysis_skipped_reason") or "").strip()
    if skip_reason:
        progress["message"] += f" — analysis skipped: {skip_reason}"
        progress["analysis_skipped"] = skip_reason
    # (task-3301 xhigh review round, F4) An analysis that RAN and failed
    # (provider exception or an in-band "Error: ..." result) annotates the
    # done row the same way a skipped one does -- the import succeeded,
    # the analysis did not, and the user must be able to see which.
    failed_reason = str(payload.get("analysis_failed_reason") or "").strip()
    if failed_reason:
        progress["message"] += f" — analysis failed: {failed_reason}"
        progress["analysis_failed"] = failed_reason
    # (task-3306 xhigh review round) A cookies path the option boundary
    # refused to forward. The import itself is fine -- a public URL never
    # needed cookies -- so this is an annotation, not a failure; without it
    # a gated import that silently ran un-authenticated looks identical to
    # one that worked.
    cookies_problem = str(payload.get("cookies_problem") or "").strip()
    if cookies_problem:
        progress["message"] += f" — cookies ignored: {cookies_problem}"
        progress["cookies_problem"] = cookies_problem
    return progress


def _stream_fileno(stream: Any) -> int:
    """Best-effort file descriptor for a possibly-fake stream object.

    Args:
        stream: Anything shaped like a text stream (may be Textual's
            stderr capture object, a pytest capture stream, ``None``, ...).

    Returns:
        The stream's OS-level fd when ``fileno()`` returns a real one;
        ``-1`` when the stream is missing/``None``, ``fileno()`` raises,
        or -- the case that actually bit in production -- ``fileno()``
        returns a non-fd sentinel like ``-1`` without raising (Textual's
        capture object does exactly that).
    """
    try:
        fd = stream.fileno()
    except Exception:
        return -1
    return fd if isinstance(fd, int) and fd >= 0 else -1


# The detect_file_type() values whose parse worker runs transcription
# (see Local_Ingestion/local_file_ingestion.py audio/video branches). The
# heavy-lane cap limits how many of these parse concurrently.
_INGEST_HEAVY_TYPES = frozenset({"audio", "video"})

# ebooklib retains the archive model while extractors build full DOM/text
# representations, so ebook jobs have their own one-at-a-time memory lane.
_INGEST_EBOOK_TYPES = frozenset({"ebook"})
_INGEST_EBOOK_POOL_MODE = "ebook"
_INGEST_GENERAL_POOL_MODE = "general"
_INGEST_PARSE_POOL_RESTART_ERROR = (
    "Library import workers could not shut down cleanly; "
    "restart the app before retrying."
)
_INGEST_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10.0


# (task 10, spec §9.1 AC 37/AC-24b) The named template errors the ingest
# dispatch fails an item on: an unresolvable choice (deleted/renamed) and a
# stored-invalid body refused by the validator.
#
# (task-21102) Resolved at except-time by the sole consumer (the ingest job
# dispatch loop) rather than imported at module scope: these two imports were
# one of the six entry points that executed the full Chunking package
# (~15k LOC shim + vendored engine) during ``import tldw_chatbook.app``.
# The lazily imported classes are the SAME objects the raising code
# (``_ingest_job_options`` -> ``Chunking.template_runtime`` /
# ``chunking_interop_library``) raises, so the except clause catches exactly
# what it always caught.
#
# (task-21102 review round) Because ``except _template_resolution_errors()``
# evaluates this for EVERY exception reaching that clause -- not only
# template errors -- the matcher must be inert for unrelated errors:
# * If ``tldw_chatbook.Chunking`` is not resident, no template error can be
#   in flight (an instance of its exception classes cannot exist without the
#   defining modules having been imported), so return ``()`` -- which
#   matches nothing -- WITHOUT importing ~39 Chunking modules as a side
#   effect of handling an unrelated exception.
# * If the imports themselves fail (broken install), also return ``()`` so
#   the ORIGINAL in-flight exception propagates with its own class instead
#   of being replaced by a ModuleNotFoundError raised from the except
#   clause.
# Guarded by ``Tests/App/test_template_error_lazy_matching.py``.
def _template_resolution_errors() -> tuple[type[Exception], ...]:
    """Return the named template-resolution error types, imported lazily.

    Returns:
        ``(TemplateResolutionError, InvalidTemplateError)`` when the
        Chunking package is resident and importable; ``()`` otherwise, so
        that using this as an ``except`` matcher never masks an unrelated
        in-flight exception and never imports Chunking as a side effect.
    """
    if "tldw_chatbook.Chunking" not in sys.modules:
        return ()
    try:
        from tldw_chatbook.Chunking.chunking_interop_library import (
            InvalidTemplateError,
        )
        from tldw_chatbook.Chunking.template_runtime import TemplateResolutionError
    except Exception:
        return ()

    return (TemplateResolutionError, InvalidTemplateError)


_INGEST_LOCAL_STT_PHASE_MESSAGES: dict[WorkerPhase, str] = {
    WorkerPhase.PREPARING: "Preparing import",
    WorkerPhase.LOADING: "Loading source",
    WorkerPhase.TRANSCRIBING: "Transcribing audio",
    WorkerPhase.POST_PROCESSING: "Post-processing audio",
}

# Cap on how many persisted ingest jobs `_restore_ingest_jobs` carries
# forward on restart (see `Library.library_ingest_jobs.plan_restore`) --
# keeps startup and the in-memory registry bounded for a long-lived store.
_MAX_PERSISTED_INGEST_JOBS = 500


# Keep-alive singleton for `_ingest_pool_real_stderr`'s devnull fallback.
# Module-level on purpose: the multiprocessing resource tracker inherits this
# fd ONCE at its (process-global, once-per-process) launch and keeps writing
# to it for the rest of the process's life -- if the handle were a local that
# got garbage-collected, the OS could reuse the fd number and the tracker's
# error output would silently corrupt an unrelated file.
_INGEST_POOL_STDERR_FALLBACK = None


@dataclass(frozen=True)
class _IngestParsePoolResources:
    """Process-pool resources owned by one ingest parse generation."""

    pool: Any
    progress_queue: Any | None


def _ingest_pool_real_stderr():
    """Return a stream with a REAL file descriptor to stand in for stderr.

    Used by ``LibraryIngestQueueMixin._create_ingest_parse_pool`` when
    ``sys.stderr`` has no usable fd (Textual app mode / textual-serve
    replace it with a capture object whose ``fileno()`` returns ``-1``
    without raising -- see that method's docstring for the crash this
    caused). Preference order:

    1. ``sys.__stderr__`` -- the process's ORIGINAL stderr, still fd-backed
       even after Textual swaps ``sys.stderr`` (Textual redirects the
       high-level name, not the OS-level fd).
    2. A process-lifetime ``os.devnull`` handle (see
       ``_INGEST_POOL_STDERR_FALLBACK``'s comment for why it must stay
       referenced) -- ``sys.__stderr__`` can itself be ``None``/fd-less in
       exotic embed/frozen environments.
    """
    real = sys.__stderr__
    if real is not None and _stream_fileno(real) >= 0:
        return real
    global _INGEST_POOL_STDERR_FALLBACK
    if _INGEST_POOL_STDERR_FALLBACK is None:
        _INGEST_POOL_STDERR_FALLBACK = open(os.devnull, "w")
    return _INGEST_POOL_STDERR_FALLBACK


def _response_field(payload: Any, name: str) -> Any:
    """Read ``name`` from a pydantic model or a plain dict response.

    The tldw client returns models, but its own tests exercise the same paths
    with dicts, so both shapes are accepted.
    """
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


def _accepts_keyword(func: Any, name: str) -> bool:
    """Report whether ``func`` can be called with the ``name`` keyword.

    Asked up front instead of calling and catching ``TypeError``: that pattern
    cannot tell "this callable has no such parameter" from "a ``TypeError`` was
    raised inside it", so a genuine bug downstream reads as a missing feature
    and degrades silently. That is exactly how the remote ingest poller shipped
    asking for an ``offset`` the client did not yet accept, and paginated
    nothing for it (task-684.2).

    A callable whose signature cannot be read (a C builtin, an exotic mock) is
    reported as accepting the keyword, so real services are not downgraded by
    an unreadable signature; ``**kwargs`` counts as accepting it.
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return True
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


class LibraryIngestQueueMixin:
    """Library ingest job submission seam + parallel-parse coordinator + writer.

    Mixed into :class:`TldwCli` (and headless test harnesses -- see
    ``Tests/Library/test_library_ingest_runner.py``) rather than being
    defined directly on the App class, so the coordinator + writer can be
    exercised without booting the full app. A host class is expected to
    provide:

    - ``self.library_ingest_jobs``: a ``LibraryIngestJobRegistry`` instance
      constructed once (e.g. in ``__init__``/app wiring).
    - ``self.media_db``: an ``Optional[MediaDatabase]``.
    - ``self._ingest_parse_pool``, ``self._ingest_parsed_payloads``,
      ``self._ingest_parse_pool_generation``,
      ``self._ingest_parse_jobs_by_generation``, and
      ``self._ingest_parse_pool_mode``,
      ``self._ingest_shutdown``: the coordinator's own state, initialized
      once alongside ``library_ingest_jobs`` -- see ``TldwCli.__init__``.
    - Textual's ``App``/``Widget`` worker machinery (``@work`` and
      ``call_from_thread``), since this mixin is always combined with one
      of those base classes.

    F3 architecture -- two decoupled stages, not one serial loop:

    - **Parse stage (this mixin's coordinator, UI thread).** A lazily
      created spawn-context ``multiprocessing.Pool`` (see
      ``_create_ingest_parse_pool``) fans ordinary file parsing out to N
      workers. Ebook batches instead own one-worker generations, retired
      before ordinary work resumes so parser high-water heaps cannot
      accumulate across the configured pool. ``_top_up_ingest_parse_pool``
      runs after every submission/retry and parse completion. A completion is
      marshaled onto the UI thread (``_on_ingest_parse_complete``); success
      stashes the parsed payload and wakes the writer, failure goes straight
      to ``mark_failed``.
    - **Write stage (the writer, background thread, unchanged shape).**
      Exactly one job is ever being written at a time (SQLite has one
      writer). The writer's claim-or-release loop
      (``_claim_next_ingest_job_or_release`` / ``_run_library_ingest_queue``)
      now claims the OLDEST payload-ready job (by submission order) instead
      of the oldest queued one, persists it via ``persist_parsed_media``,
      and marks it ``DONE``/``FAILED``.

    The coordinator (parse side) and the writer (write side) are the only
    intended callers of the registry's ``mark_parsing``/``mark_writing``/
    ``mark_done``/``mark_failed`` transition methods, respectively. Every
    job is driven either ``queued`` -> ``parsing`` -> ``writing`` ->
    ``done``/``failed``, or ``parsing`` -> ``failed`` directly when the pool
    worker's parse itself fails (e.g. an unsupported/undetectable file type,
    or a missing source file -- classified by ``classify_parse_failure``
    inside the worker, where the real exception type is available). Either
    way, one job's failure is isolated so it never strands a later queued
    job or blocks the writer.

    Shutdown (quit path) order, in ``_shutdown_ingest_parse_pool`` (called
    from ``TldwCli.on_unmount``): (1) ``_ingest_shutdown = True`` + executor
    and pool references detached, synchronously -- callbacks short-circuit
    before ever marshaling; (2) executor close followed by a bounded wait for
    ``pool.terminate()`` + ``pool.join()`` on detached daemon threads, never
    the event-loop thread (deadlock rationale in that method's docstring); (3)
    the writer thread is swept afterward by ``on_unmount``'s
    generic worker cancellation, its in-flight DB write completing as
    before. Steps 2 and 3 run concurrently -- safe because the stages share
    no resources (parse workers never touch ``media_db``; the writer never
    touches either heavy worker).
    """

    _RESEARCH_SOURCE_RETRY_UNAVAILABLE_COPY = (
        "Research source retry is unavailable. Open Research Workspace "
        "and retry from its receipt."
    )

    def _init_library_ingest_runtime_state(self) -> None:
        """Initialize every host attribute the ingest job loop reads.

        The single source of truth for this mixin's host-state contract:
        ``TldwCli``'s wiring calls this, and the headless test harnesses
        (``Tests/UI/test_library_shell.py``'s ``_LibraryIngestCanvasHarness``,
        ``Tests/Library/test_library_ingest_runner.py``'s
        ``_IngestRunnerHarness``) call the same method, so a new
        ``self._ingest_*`` read added to the coordinator/writer is mirrored
        into the fakes automatically instead of hand-listed (task-3315 --
        the hand-listed harness missed ``_ingest_local_stt_jobs`` when the
        local-STT lane landed and ~20 pilots died with AttributeError).
        ``self.media_db`` is deliberately NOT set here: it is a per-host
        input (see the class docstring), not coordinator state.

        F3 parallel-parse coordinator state: the lazily-created parse-pool
        handle, the parse->write handoff (job_id -> parsed payload dict,
        populated by a pool completion and drained by the writer's claim),
        and the shutdown flag pool callbacks check before touching a
        closing app.
        """
        self.library_ingest_jobs = LibraryIngestJobRegistry()
        self._research_source_terminal_jobs_scheduled: set[str] = set()
        self._research_source_parse_dispatch_pending: set[str] = set()
        self._research_source_restore_in_progress = False
        self.library_ingest_jobs.add_listener(
            self._schedule_settled_research_source_operations
        )
        self._ingest_parse_pool = None
        self._ingest_parse_pool_generation: int = 0
        self._ingest_parse_jobs_by_generation: dict[int, set[str]] = {}
        self._ingest_parse_pool_stop_event: Optional[threading.Event] = None
        self._ingest_parse_progress_queue: Any | None = None
        self._ingest_parse_progress_thread: threading.Thread | None = None
        self._ingest_parse_pool_mode: str | None = None
        self._ingest_parse_pool_retiring = False
        self._ingest_parse_pool_retirement_error: str | None = None
        self._ingest_parsed_payloads: dict[str, dict] = {}
        # RLock, not Lock: dev's STT dispatch work re-enters this guard.
        self._local_stt_executor_lock = threading.RLock()
        self._local_stt_executor: Optional[LocalSTTExecutor] = None
        self._local_stt_dispatch_coordinator: Optional[LocalSTTDispatchCoordinator] = (
            None
        )
        self._parakeet_source_service: Any | None = None
        self._parakeet_source_registry_listener: Callable[[], None] | None = None
        self._parakeet_submitting_scope_ids: set[str] = set()
        self._ingest_local_stt_jobs: dict[str, tuple[int, str]] = {}
        self._ingest_shutdown: bool = False
        self._ingest_maintenance_paused = False
        self._ingest_writer_threads: set[threading.Thread] = set()
        self._ingest_writer_threads_lock = threading.Lock()

    def _ingest_writers_pending(self) -> bool:
        """Observe actual thread bodies, including cancelled Textual wrappers."""
        with self._ingest_writer_threads_lock:
            return bool(self._ingest_writer_threads)

    def _ingest_maintenance_close_admission(self) -> None:
        """Stop new submissions/top-ups while admitted results still publish."""
        self._ingest_maintenance_paused = True

    async def _ingest_maintenance_drain(self, deadline: float) -> bool:
        """Wait for owned parses and payload publication without killing work."""
        if not self._ingest_maintenance_paused:
            raise RuntimeError("ingest_maintenance_not_paused")
        while (
            self.library_ingest_jobs.runner_active
            or self._ingest_writers_pending()
            or self._ingest_parsed_payloads
            or self._ingest_local_stt_jobs
            or any(self._ingest_parse_jobs_by_generation.values())
            or any(
                worker.group.startswith("library_ingest_") and not worker.is_finished
                for worker in getattr(self, "workers", ())
            )
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(remaining, 0.02))
        return True

    def _ingest_maintenance_resume(self) -> None:
        """Keep queued jobs and resume the existing writer/parser dispatch."""
        self._ingest_maintenance_paused = False
        if self._ingest_shutdown:
            return
        if self._ingest_parsed_payloads:
            self._start_library_ingest_queue_if_idle()
        self._top_up_ingest_parse_pool()
        self.poll_remote_ingest_jobs()

    def _schedule_settled_research_source_operations(self) -> None:
        """Schedule durable association work after a linked job has settled.

        Registry listeners run after the in-memory transition has completed.
        This listener only queues an async worker; it never calls the
        coordinator (and therefore never touches SQLite) synchronously inside
        the registry mutation.
        """
        # TASK-32804.5: this listener fires on every registry mutation. Run
        # the cheap early-returns BEFORE touching the queue, and iterate the
        # jobs WITHOUT the per-job deep copy jobs() makes (read-only here) --
        # a 1,000-file folder import fired this per file, each deep-copying
        # the whole queue (O(n^2), 1.65 s).
        if self._research_source_restore_in_progress:
            return
        scheduler = getattr(self, "research_source_association_scheduler", None)
        if scheduler is None:
            return
        jobs = list(self.library_ingest_jobs.iter_jobs_for_listeners())
        if self._research_source_terminal_jobs_scheduled:
            self._research_source_terminal_jobs_scheduled.intersection_update(
                job.job_id for job in jobs
            )
        terminal_states = {
            IngestJobState.DONE,
            IngestJobState.FAILED,
            IngestJobState.CANCELLED,
            IngestJobState.SKIPPED,
        }
        for job in jobs:
            operation_id = str(job.research_source_operation_id or "").strip()
            if (
                not operation_id
                or job.state not in terminal_states
                or job.job_id in self._research_source_terminal_jobs_scheduled
            ):
                continue
            self._research_source_terminal_jobs_scheduled.add(job.job_id)
            self.run_worker(
                self._resume_settled_research_source_operation(
                    job.job_id,
                    operation_id,
                ),
                group="research_source_association",
            )

    async def _resume_settled_research_source_operation(
        self, job_id: str, operation_id: str
    ) -> None:
        """Run one scheduled resume and release suppression after exceptions."""

        scheduler = getattr(self, "research_source_association_scheduler", None)
        if scheduler is None:
            self._research_source_terminal_jobs_scheduled.discard(job_id)
            return
        try:
            operation = await scheduler.resume(operation_id)
            staging_store = getattr(self, "research_paste_staging_store", None)
            job = self.library_ingest_jobs.get_job(job_id)
            state = str(getattr(getattr(job, "state", None), "value", ""))
            if staging_store is not None and (
                state in {"cancelled", "skipped"}
                or (
                    operation is not None
                    and operation.catalog_status is SourceOperationStatus.SUCCEEDED
                )
            ):
                await asyncio.to_thread(staging_store.delete, operation_id)
        except Exception:
            self._research_source_terminal_jobs_scheduled.discard(job_id)
            logger.opt(exception=True).warning(
                "Research source association worker failed; operation remains resumable"
            )

    def _restore_ingest_jobs_and_schedule_research_sources(self) -> None:
        """Restore ingest history before queuing bounded source-operation work."""

        restore_was_in_progress = self._research_source_restore_in_progress
        self._research_source_restore_in_progress = True
        try:
            self._restore_ingest_jobs()
        finally:
            self._research_source_restore_in_progress = restore_was_in_progress
        ingest_store = getattr(self, "_library_ingest_jobs_store", None)
        operation_store = getattr(self, "research_source_operation_store", None)
        if ingest_store is not None and operation_store is not None:
            self.run_worker(
                self._reconcile_research_source_held_jobs(),
                group="research_source_held_startup",
            )
        scheduler = getattr(self, "research_source_association_scheduler", None)
        if scheduler is not None:
            self.run_worker(
                scheduler.resume_startup(),
                group="research_source_association_startup",
            )
        staging_store = getattr(self, "research_paste_staging_store", None)
        if staging_store is not None and operation_store is not None:
            self.run_worker(
                self._sweep_research_paste_staging(),
                group="research_paste_staging_startup",
            )

    async def _sweep_research_paste_staging(self) -> None:
        """Run one bounded fail-safe startup sweep away from the UI loop."""

        staging_store = getattr(self, "research_paste_staging_store", None)
        operation_store = getattr(self, "research_source_operation_store", None)
        if staging_store is None or operation_store is None:
            return
        try:
            await asyncio.to_thread(
                staging_store.sweep,
                operation_store,
                job_registry=self.library_ingest_jobs,
                limit=100,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Research paste staging sweep failed; artifacts were retained"
            )

    async def _reconcile_research_source_held_jobs(self, *, limit: int = 50) -> None:
        """Boundedly link or cancel durable Research jobs left held at restart."""

        ingest_store = getattr(self, "_library_ingest_jobs_store", None)
        operation_store = getattr(self, "research_source_operation_store", None)
        if ingest_store is None or operation_store is None:
            return
        try:
            rows = await asyncio.to_thread(ingest_store.list_dispatch_held, limit=limit)
        except Exception:
            logger.opt(exception=True).warning(
                "Research source held-job startup scan failed; jobs remain held"
            )
            return
        for row in rows:
            job_id = str(row.get("job_id") or "")
            operation_id = str(row.get("research_source_operation_id") or "")
            job = self.library_ingest_jobs.get_job(job_id)
            if (
                job is None
                or job.state is not IngestJobState.QUEUED
                or not job.dispatch_held
                or job.research_source_operation_id != operation_id
            ):
                continue
            try:
                operation = await asyncio.to_thread(operation_store.get, operation_id)
            except Exception:
                logger.opt(exception=True).warning(
                    "Research source held-job receipt read failed "
                    "(job_id={}, operation_id={}); retained for recovery",
                    job_id,
                    operation_id,
                )
                continue

            expected_origin = str(
                getattr(getattr(operation, "data_source", None), "value", "")
            )
            compatible = operation is not None and expected_origin == job.origin
            if (
                compatible
                and operation.catalog_status is SourceOperationStatus.PENDING
                and not operation.ingest_job_id
            ):
                try:
                    operation = await asyncio.to_thread(
                        operation_store.advance_stage,
                        operation_id,
                        stage=SourceOperationStage.CATALOG,
                        status=SourceOperationStatus.IN_PROGRESS,
                        expected_revision=operation.revision,
                        ingest_job_id=job_id,
                    )
                except Exception:
                    try:
                        operation = await asyncio.to_thread(
                            operation_store.get, operation_id
                        )
                    except Exception:
                        logger.opt(exception=True).warning(
                            "Research source held-job link remains pending "
                            "(job_id={}, operation_id={})",
                            job_id,
                            operation_id,
                        )
                        continue
                    expected_origin = str(
                        getattr(getattr(operation, "data_source", None), "value", "")
                    )
                    compatible = operation is not None and expected_origin == job.origin

            linked = (
                compatible
                and operation.catalog_status is SourceOperationStatus.IN_PROGRESS
                and operation.ingest_job_id == job_id
            )
            if linked:
                try:
                    released = self.library_ingest_jobs.release_dispatch_hold(
                        job_id, require_persisted=True
                    )
                    if released is None:
                        continue
                    self._dispatch_research_source_catalog_job(job_id)
                except Exception:
                    logger.opt(exception=True).warning(
                        "Research source held-job dispatch could not start "
                        "(job_id={}, operation_id={})",
                        job_id,
                        operation_id,
                    )
                    try:
                        self._fail_research_source_prepared_job(job_id)
                    except Exception:
                        logger.opt(exception=True).warning(
                            "Research source held-job dispatch failure could not be persisted "
                            "(job_id={}, operation_id={})",
                            job_id,
                            operation_id,
                        )
                continue

            still_pending = (
                compatible
                and operation.catalog_status is SourceOperationStatus.PENDING
                and not operation.ingest_job_id
            )
            if still_pending:
                continue
            try:
                cancelled = self._cancel_research_source_prepared_job(job_id)
            except Exception:
                logger.opt(exception=True).warning(
                    "Research source incompatible held-job cancellation failed "
                    "(job_id={}, operation_id={}); staging retained",
                    job_id,
                    operation_id,
                )
                continue
            if cancelled.state not in {
                IngestJobState.CANCELLED,
                IngestJobState.FAILED,
                IngestJobState.DONE,
                IngestJobState.SKIPPED,
            }:
                continue
            staging_store = getattr(self, "research_paste_staging_store", None)
            if staging_store is not None:
                try:
                    await asyncio.to_thread(staging_store.delete, operation_id)
                except Exception:
                    logger.opt(exception=True).warning(
                        "Research source terminal held-job staging cleanup failed "
                        "(job_id={}, operation_id={})",
                        job_id,
                        operation_id,
                    )

    def _restore_ingest_jobs(self) -> None:
        """Start the one-time restore of persisted ingest job history.

        Returns immediately: the store open (schema create/migrate on first
        run), the read, the plan and the reconcile writes all run on a worker
        thread, and only the in-memory registry seeding comes back to the UI
        thread (TASK-21111(c) -- measured 1.7-11.6 ms of synchronous
        ``on_mount`` work depending on history size, x3-5 on constrained
        hardware).

        Never raises, on either thread: a corrupt or unreadable store leaves
        the registry empty and store-less, exactly as before.
        """
        if getattr(self, "_ingest_shutdown", False):
            return
        self.run_worker(
            self._restore_ingest_jobs_off_thread,
            name="restore_ingest_jobs",
            group="ingest_restore",
            thread=True,
            exclusive=True,
            # The body already catches and logs everything; `exit_on_error`
            # is off so no future edit to it can turn a history-restore
            # failure into an app exit. Restoring history must never be able
            # to prevent boot -- that was true of the synchronous version and
            # stays true here.
            exit_on_error=False,
        )

    def _restore_ingest_jobs_off_thread(self) -> None:
        """Worker body for :meth:`_restore_ingest_jobs`. Runs on a thread.

        Catches everything. A worker that raised would surface as an uncaught
        ``WorkerFailed`` and take the app down -- the failure mode this
        function's synchronous predecessor could not have.
        """
        from datetime import datetime, timezone
        from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
        from tldw_chatbook.Library.library_ingest_jobs import plan_restore

        # Bound before the `try` so the failure path can tell "never opened"
        # from "opened, then a later step failed" -- the second case owns a
        # live SQLite connection (the store opens one in its constructor, via
        # `_initialize_schema`) that nothing else will ever close, because the
        # registry is left store-less.
        store = None
        try:
            # Construct and reconcile on this worker, then retire its native
            # connection before publishing the reopenable store to the UI.
            store = LibraryIngestJobsDB(get_library_ingest_jobs_db_path())
            # Do ALL fallible work -- corrupt read, plan, and the store
            # reconcile writes -- BEFORE touching the in-memory registry, so any
            # failure leaves the registry empty + store unattached: a clean
            # in-memory fallback that matches the "starting empty" warning below
            # (rather than a half-restored registry contradicting the log).
            plan = plan_restore(
                store.all_jobs(),
                max_persisted=_MAX_PERSISTED_INGEST_JOBS,
                now_iso=datetime.now(timezone.utc).isoformat(),
            )
            for job in plan.upsert:
                store.upsert_job(job)
            for job_id in plan.delete_ids:
                store.delete_job(job_id)
            # Successful handoff transfers the store, not this worker's cache.
            store.close()
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to restore persisted ingest job history; starting empty."
            )
            if store is not None:
                try:
                    store.close()
                except Exception:
                    logger.opt(exception=True).debug(
                        "Ingest job store close after failed restore failed."
                    )
            return

        try:
            self.call_from_thread(self._apply_ingest_job_restore, store, plan)
        except Exception:
            # The app stopped (quit during startup) or the callback itself
            # failed. Either way the registry stays store-less; close the
            # connection this thread opened rather than leaking it.
            logger.opt(exception=True).debug(
                "Ingest job history restore could not be applied; discarding."
            )
            try:
                store.close()
            except Exception:
                logger.opt(exception=True).debug(
                    "Ingest job store close after failed restore failed."
                )

    def _apply_ingest_job_restore(self, store: Any, plan: Any) -> None:
        """Seed the registry from a completed restore plan. UI thread only.

        Args:
            store: The opened ``LibraryIngestJobsDB`` to attach as the
                registry's write-through sink.
            plan: The ``RestorePlan`` produced off-thread.

        The registry is documented UI-thread-only, so the seeding and the
        store attach stay here even though the I/O moved. Uses
        ``merge_restored`` rather than ``restore`` so a job submitted in the
        few milliseconds between ``on_mount`` and this callback survives.

        The store is attached BEFORE the merge, not after: a job submitted in
        that window was submitted while the registry was store-less, so its
        own ``_persist`` was a no-op, and nothing later re-offers it. With
        the store attached first, ``merge_restored`` writes those live jobs
        through -- which also replaces any persisted row that happens to
        share their id (both sessions allocate from ``ingest-job-1`` upward,
        so a collision in this window is the likely case, and the stale row
        would otherwise be restored in the live job's place next launch).
        """
        if getattr(self, "_ingest_shutdown", False):
            store.close()
            return
        self._library_ingest_jobs_store = store
        self.library_ingest_jobs.attach_store(store)
        self.library_ingest_jobs.merge_restored(plan.jobs, plan.next_id)

    def _expand_library_ingest_source(self, source_path: str) -> list[str] | None:
        """Expand a directory source into the files it contains.

        Args:
            source_path: The submitted source: a file path, a URL, or a
                directory.

        Returns:
            ``None`` when ``source_path`` is not a directory (URLs and files
            are submitted as-is), otherwise the list of contained file paths
            -- which is empty when the directory holds nothing ingestible.
        """
        try:
            candidate = Path(source_path).expanduser()
            if not candidate.is_dir():
                return None
        except (OSError, ValueError):
            # Unreadable, over-length, or malformed for this platform (Windows
            # raises ValueError where POSIX raises OSError). Not a directory we
            # can expand -- let the single-source path report the failure.
            return None

        raw_limit = get_cli_setting("library.ingest_directory_scan_limit", 1000)
        try:
            scan_limit = int(raw_limit)
        except (TypeError, ValueError):
            scan_limit = 1000

        files, truncated = collect_directory_files(candidate, scan_limit)
        if truncated:
            logger.warning(
                f"Library ingest directory {source_path!r} exceeded the scan "
                f"limit of {scan_limit}; only the first {len(files)} files "
                "were queued."
            )
        return [str(path) for path in files]

    def submit_library_ingest_job(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any] | None = None,
        title: str = "",
        author: str = "",
        keywords: tuple[str, ...] = (),
        perform_analysis: bool = False,
        chunk_enabled: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        batch_id: str | None = None,
        active_duplicate_consent: ActiveIngestConsentScope | None = None,
        research_source_operation_id: str | None = None,
        required_origin: str | None = None,
        _prepare_only: bool = False,
    ) -> LibraryIngestJob:
        """Submit a new Library ingest job and top up the parse pool.

        UI-thread only. Appends a ``QUEUED`` job to ``self.library_ingest_jobs``.
        When ``self.media_db`` is unavailable, the job is failed immediately
        (with the exact copy ``"Media database is unavailable."``) and it
        never reaches the parse pool.
        ``batch_id`` carries the folder-expansion batch id (task-2221) so
        the queue can group one submission's jobs; ``None`` for single
        files.

        Args:
            source_path: The file path to ingest.
            ingest_options: Per-type ingestion options snapshot captured at
                submit time. This is the canonical source of ingestion
                settings; the older ``perform_analysis``/``chunk_enabled``/
                ``chunk_size`` arguments are deprecated fallbacks.
            title: Optional title form field.
            author: Optional author form field.
            keywords: Keywords form field.
            perform_analysis: Whether to run post-ingest analysis.
            chunk_enabled: Whether to chunk the ingested content.
            chunk_size: Requested chunk size when ``chunk_enabled``.
            active_duplicate_consent: Exact candidate and active-membership scope
                captured by an explicitly confirmed submission.
            required_origin: Optional fail-closed owner precondition for a captured
                Research workspace authority. General Library submissions omit it.

        Returns:
            The newly created job: ``QUEUED`` normally, or immediately
            ``FAILED`` when ``media_db`` is unavailable. A directory source
            queues one job per contained file and returns the first of them,
            so each file gets its own queue row, its own outcome and its own
            retry -- one unsupported file no longer fails its siblings.
        """
        if getattr(self, "_ingest_maintenance_paused", False):
            raise RuntimeError("ingest_maintenance_paused")
        normalized_required_origin = (
            str(required_origin).strip().lower()
            if required_origin is not None
            else None
        )
        if normalized_required_origin not in {None, "local", "server"}:
            raise ValueError("required_origin must be local or server")
        backend = self._resolve_ingest_backend()
        if (
            normalized_required_origin is not None
            and backend != normalized_required_origin
        ):
            selected = normalized_required_origin.title()
            raise ValueError(
                f"Ingestion is unavailable for the selected {selected} authority. "
                f"The active Library ingest owner is {backend.title()}."
            )
        if research_source_operation_id and normalized_required_origin is not None:
            self._validate_research_source_operation_authority(
                research_source_operation_id,
                expected_origin=normalized_required_origin,
            )
        expanded = self._expand_library_ingest_source(source_path)
        if _prepare_only and expanded is not None:
            raise ValueError(
                "Research source preparation accepts one file or URL, not a folder."
            )
        if research_source_operation_id and expanded is not None and len(expanded) > 1:
            raise ValueError(
                "Folder imports require one Research source operation per catalog item."
            )
        sources = tuple(expanded) if expanded is not None else (source_path,)
        matches = self.library_ingest_jobs.find_active_source_matches(
            sources, origin=backend
        )
        matched_source_keys = set()
        for job in matches:
            try:
                matched_source_keys.add(
                    normalize_active_ingest_source(job.source_path, origin=backend)
                )
            except (TypeError, ValueError, OSError):
                continue
        current_consent = build_active_ingest_consent_scope(
            sources,
            origin=backend,
            active_job_ids=(job.job_id for job in matches),
            active_source_count=len(matched_source_keys),
        )
        candidates_changed = active_duplicate_consent is not None and (
            active_duplicate_consent.origin != current_consent.origin
            or active_duplicate_consent.candidate_digest
            != current_consent.candidate_digest
            or active_duplicate_consent.candidate_count
            != current_consent.candidate_count
        )
        matches_covered = (
            active_duplicate_consent is not None
            and active_duplicate_consent.covers(current_consent)
        )
        if candidates_changed or (matches and not matches_covered):
            raise ActiveIngestSubmissionRefused(
                (ActiveIngestJobRef(job.job_id, job.state) for job in matches),
                consent_scope=current_consent,
                candidate_changed=candidates_changed,
            )

        normalized_options = ingest_options or {}
        if expanded is not None:
            if not expanded:
                empty_job = self.library_ingest_jobs.submit(
                    source_path=source_path,
                    title=title,
                    author=author,
                    keywords=keywords,
                    perform_analysis=perform_analysis,
                    chunk_enabled=chunk_enabled,
                    chunk_size=chunk_size,
                    detected_type="",
                    ingest_options=normalized_options,
                    research_source_operation_id=research_source_operation_id,
                )
                failed = self.library_ingest_jobs.mark_failed(
                    empty_job.job_id,
                    error="No files to import were found in this folder.",
                )
                return failed if failed is not None else empty_job
            first_job: LibraryIngestJob | None = None
            # (task-2221 owner ruling) One batch id per folder submission,
            # so the queue can group this run's rows under one header and
            # the tally can answer "what did THIS run just do".
            folder_batch_id = f"local-{uuid.uuid4().hex[:12]}"
            audio_options = normalized_options.get("audio_video", {})
            scope_id = (
                str(audio_options.get("transcription_external_scope_id") or "").strip()
                if isinstance(audio_options, dict)
                else ""
            )
            submitting_scopes = getattr(self, "_parakeet_submitting_scope_ids", None)
            if submitting_scopes is None:
                submitting_scopes = self._parakeet_submitting_scope_ids = set()
            if scope_id:
                submitting_scopes.add(scope_id)
            try:
                for expanded_path in expanded:
                    job = self._submit_library_ingest_job_admitted(
                        source_path=expanded_path,
                        ingest_options=normalized_options,
                        batch_id=folder_batch_id,
                        # Title is per-file (the ingest form clears it on submit
                        # for exactly this reason), so a folder's files each take
                        # their own filename-derived title rather than all
                        # sharing one. Author and keywords are batch metadata and
                        # do carry across.
                        title="",
                        author=author,
                        keywords=keywords,
                        perform_analysis=perform_analysis,
                        chunk_enabled=chunk_enabled,
                        chunk_size=chunk_size,
                        backend=backend,
                        research_source_operation_id=research_source_operation_id,
                    )
                    if first_job is None:
                        first_job = job
            finally:
                if scope_id:
                    submitting_scopes.discard(scope_id)
                    self._sync_parakeet_source_scopes()
            # ``expanded`` is non-empty here, so the loop always assigns.
            assert first_job is not None
            return first_job

        admitted_kwargs = dict(
            source_path=source_path,
            ingest_options=normalized_options,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            chunk_enabled=chunk_enabled,
            chunk_size=chunk_size,
            batch_id=batch_id,
            backend=backend,
            research_source_operation_id=research_source_operation_id,
        )
        if _prepare_only:
            return self._prepare_library_ingest_job_admitted(
                **admitted_kwargs,
                dispatch_held=True,
                require_persisted=True,
            )
        return self._submit_library_ingest_job_admitted(**admitted_kwargs)

    def prepare_research_source_ingest_job(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any] | None = None,
        title: str = "",
        author: str = "",
        keywords: tuple[str, ...] = (),
        perform_analysis: bool = False,
        chunk_enabled: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        research_source_operation_id: str,
        required_origin: str,
    ) -> LibraryIngestJob:
        """Durably queue one qualified Research source without dispatching it."""

        return self.submit_library_ingest_job(
            source_path=source_path,
            ingest_options=ingest_options,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            chunk_enabled=chunk_enabled,
            chunk_size=chunk_size,
            research_source_operation_id=research_source_operation_id,
            required_origin=required_origin,
            _prepare_only=True,
        )

    def _prepare_library_ingest_job_admitted(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any],
        title: str,
        author: str,
        keywords: tuple[str, ...],
        perform_analysis: bool,
        chunk_enabled: bool,
        chunk_size: int,
        batch_id: str | None,
        backend: str,
        research_source_operation_id: str | None,
        require_persisted: bool,
        dispatch_held: bool = False,
    ) -> LibraryIngestJob:
        """Create one queued row without starting its Local or Server owner."""

        detected_type = ""
        if backend == "server":
            if is_web_clip_source(source_path):
                build_web_clip_kwargs(
                    source_path,
                    options=ingest_options,
                    title=title,
                    author=author,
                    keywords=keywords,
                )
                detected_type = "web"
            else:
                kwargs = build_server_ingest_kwargs(
                    source_path,
                    options=ingest_options,
                    title=title,
                    author=author,
                    keywords=keywords,
                    perform_analysis=perform_analysis,
                )
                detected_type = str(kwargs.get("media_type") or "")
        else:
            try:
                detected_type = classify_ingest_source(source_path) or ""
            except FileIngestionError:
                detected_type = ""
            except Exception:
                logger.warning(
                    "classify_ingest_source failed unexpectedly "
                    "(operation_id={}, origin={}); treating as light work "
                    "(heavy-lane cap may not apply).",
                    research_source_operation_id or "none",
                    backend,
                )
        return self.library_ingest_jobs.submit(
            source_path=source_path,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            chunk_enabled=chunk_enabled,
            chunk_size=chunk_size,
            detected_type=detected_type,
            ingest_options=ingest_options,
            origin=backend,
            batch_id=batch_id,
            research_source_operation_id=research_source_operation_id,
            dispatch_held=dispatch_held,
            require_persisted=require_persisted,
        )

    def _submit_library_ingest_job_admitted(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any],
        title: str,
        author: str,
        keywords: tuple[str, ...],
        perform_analysis: bool,
        chunk_enabled: bool,
        chunk_size: int,
        batch_id: str | None,
        backend: str,
        research_source_operation_id: str | None,
    ) -> LibraryIngestJob:
        """Route a source already admitted by ``submit_library_ingest_job``."""
        if backend == "server":
            # A web page goes to the clipper, not the ingest-jobs API: that API
            # has no media type for one. A local ingest needs no such branch --
            # classify_ingest_source already routes an article through the
            # pipeline's own extractor.
            submit_remote = (
                self._submit_web_clip_job
                if is_web_clip_source(source_path)
                else self._submit_server_ingest_job
            )
            return submit_remote(
                source_path=source_path,
                ingest_options=ingest_options,
                title=title,
                author=author,
                keywords=keywords,
                perform_analysis=perform_analysis,
                research_source_operation_id=research_source_operation_id,
            )

        return self._submit_local_library_ingest_job(
            source_path=source_path,
            ingest_options=ingest_options,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            chunk_enabled=chunk_enabled,
            chunk_size=chunk_size,
            batch_id=batch_id,
            research_source_operation_id=research_source_operation_id,
        )

    def _submit_local_library_ingest_job(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any],
        title: str,
        author: str,
        keywords: tuple[str, ...],
        perform_analysis: bool,
        chunk_enabled: bool,
        chunk_size: int,
        batch_id: str | None,
        research_source_operation_id: str | None,
    ) -> LibraryIngestJob:
        """Append one admitted local source and top up the parse pool."""
        job = self._prepare_library_ingest_job_admitted(
            source_path=source_path,
            ingest_options=ingest_options,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            chunk_enabled=chunk_enabled,
            chunk_size=chunk_size,
            batch_id=batch_id,
            backend="local",
            research_source_operation_id=research_source_operation_id,
            require_persisted=False,
        )
        self._dispatch_research_source_catalog_job(job.job_id)
        if self.media_db is None:
            return self.library_ingest_jobs.get_job(job.job_id) or job
        return job

    def retry_library_ingest_job(
        self,
        job_id: str,
        *,
        transcription_provider: str | None = None,
    ) -> Optional[LibraryIngestJob]:
        """Retry a previously failed Library or Research-owned ingest job.

        UI-thread only. Ordinary Library jobs use the legacy synchronous
        ``LibraryIngestJobRegistry.requeue`` path. Research-owned jobs hand
        catalog retry ownership to their durable source-operation scheduler.

        Args:
            job_id: The failed job to requeue.

        Returns:
            The newly appended ``QUEUED`` job (or immediately ``FAILED``
            when ``media_db`` is unavailable), or ``None`` when nothing was
            requeued. Research-owned jobs schedule their durable catalog-stage
            retry and return ``None``; the async owner returns the exact
            replacement only after its operation lineage is reconciled.
        """
        if getattr(self, "_ingest_maintenance_paused", False):
            raise RuntimeError("ingest_maintenance_paused")
        replacement_options = None
        if transcription_provider not in {None, "faster-whisper"}:
            return None
        source = self.library_ingest_jobs.get_job(job_id)
        if source is None:
            return None
        operation_id = str(source.research_source_operation_id or "").strip()
        if operation_id:
            self._schedule_research_source_catalog_retry(
                source,
                operation_id=operation_id,
            )
            return None
        if transcription_provider is not None:
            replacement_options = deepcopy(source.ingest_options)
            replacement_options.setdefault("audio_video", {})[
                "transcription_provider"
            ] = transcription_provider
        requeued = self.library_ingest_jobs.requeue(
            job_id,
            ingest_options=replacement_options,
        )
        if requeued is None:
            return None
        if self.media_db is None:
            failed = self.library_ingest_jobs.mark_failed(
                requeued.job_id, error="Media database is unavailable."
            )
            return failed if failed is not None else requeued
        self._top_up_ingest_parse_pool()
        return requeued

    def _schedule_research_source_catalog_retry(
        self,
        source: LibraryIngestJob,
        *,
        operation_id: str,
        notify_unavailable: bool = True,
    ) -> bool:
        """Queue the durable Research retry owner without generic requeueing."""

        scheduler = getattr(self, "research_source_association_scheduler", None)
        operation_store = getattr(self, "research_source_operation_store", None)
        run_worker = getattr(self, "run_worker", None)
        if (
            source.state is not IngestJobState.FAILED
            or source.superseded
            or source.dismissed
            or source.permanent
            or scheduler is None
            or operation_store is None
            or not callable(run_worker)
        ):
            if notify_unavailable:
                self._notify_research_source_retry_unavailable()
            return False
        awaitable = self._retry_research_source_catalog_job(
            source,
            operation_id=operation_id,
        )
        try:
            run_worker(awaitable, group="research_source_catalog_retry")
        except Exception:
            awaitable.close()
            if notify_unavailable:
                self._notify_research_source_retry_unavailable()
            return False
        return True

    async def _retry_research_source_catalog_job(
        self,
        source: LibraryIngestJob,
        *,
        operation_id: str,
    ) -> LibraryIngestJob | None:
        """Retry one exact Research catalog receipt and reload its replacement."""

        operation_store = getattr(self, "research_source_operation_store", None)
        scheduler = getattr(self, "research_source_association_scheduler", None)
        if operation_store is None or scheduler is None:
            self._notify_research_source_retry_unavailable()
            return None
        try:
            # Keep this indexed preflight on the event-loop turn so concurrent
            # clicks reach the scheduler fence in order instead of racing the
            # same SQLite connection from two executor threads.
            operation = operation_store.get(operation_id)
        except Exception:
            operation = None
        operation_source = getattr(getattr(operation, "data_source", None), "value", "")
        expected_origin = (
            operation_source if operation_source in {"local", "server"} else ""
        )
        if (
            operation is None
            or operation.operation_id != operation_id
            or operation.ingest_job_id != source.job_id
            or source.research_source_operation_id != operation_id
            or source.origin != expected_origin
        ):
            self._notify_research_source_retry_unavailable()
            return None
        try:
            receipt = await scheduler.retry(
                operation_id,
                stage=SourceOperationStage.CATALOG,
            )
        except SourceOperationConflictError:
            receipt = None
        except Exception:
            self._notify_research_source_retry_unavailable()
            return None
        replacement = self._research_source_retry_replacement(
            source,
            operation_id=operation_id,
            receipt=receipt,
        )
        if replacement is None:
            # A second click may have waited behind the scheduler fence. Re-read
            # the durable winner so every caller converges on the same job.
            try:
                receipt = await asyncio.to_thread(operation_store.get, operation_id)
            except Exception:
                receipt = None
            replacement = self._research_source_retry_replacement(
                source,
                operation_id=operation_id,
                receipt=receipt,
            )
        if replacement is None:
            self._notify_research_source_retry_unavailable()
        return replacement

    def _research_source_retry_replacement(
        self,
        source: LibraryIngestJob,
        *,
        operation_id: str,
        receipt: Any,
    ) -> LibraryIngestJob | None:
        """Return only the released replacement named by the exact receipt."""

        if (
            receipt is None
            or getattr(receipt, "operation_id", "") != operation_id
            or getattr(receipt, "catalog_status", None)
            not in {
                SourceOperationStatus.IN_PROGRESS,
                SourceOperationStatus.SUCCEEDED,
            }
        ):
            return None
        replacement_id = str(getattr(receipt, "ingest_job_id", "") or "")
        if not replacement_id or replacement_id == source.job_id:
            return None
        replacement = self.library_ingest_jobs.get_job(replacement_id)
        if (
            replacement is None
            or replacement.retry_of_job_id != source.job_id
            or replacement.research_source_operation_id != operation_id
            or replacement.origin != source.origin
            or replacement.dispatch_held
        ):
            return None
        return replacement

    def _notify_research_source_retry_unavailable(self) -> None:
        """Report a fixed path-free recovery without exposing owner failures."""

        notify = getattr(self, "notify", None)
        if callable(notify):
            notify(
                self._RESEARCH_SOURCE_RETRY_UNAVAILABLE_COPY,
                severity="warning",
            )

    def _requeue_research_source_catalog_job(
        self, job_id: str
    ) -> Optional[LibraryIngestJob]:
        """Persist a replacement Research ingest without dispatching it."""

        source = self.library_ingest_jobs.get_job(job_id)
        if source is None or source.origin not in {"local", "server"}:
            return None
        return self.library_ingest_jobs.requeue(job_id, dispatch_held=True)

    def _cancel_research_source_prepared_job(self, job_id: str) -> LibraryIngestJob:
        """Durably cancel an undispatched row whose operation link failed."""

        current = self.library_ingest_jobs.get_job(job_id)
        if current is None:
            raise ValueError("Prepared Research ingest job does not exist.")
        if current.state in {
            IngestJobState.DONE,
            IngestJobState.FAILED,
            IngestJobState.CANCELLED,
            IngestJobState.SKIPPED,
        }:
            return current
        cancelled = self.library_ingest_jobs.mark_cancelled(
            job_id,
            reason="Research source operation could not be linked.",
            require_persisted=True,
        )
        if cancelled is None:
            raise ValueError("Prepared Research ingest job cannot be cancelled.")
        return cancelled

    def _fail_research_source_prepared_job(self, job_id: str) -> LibraryIngestJob:
        """Durably fail a linked row whose owner dispatch did not start."""

        current = self.library_ingest_jobs.get_job(job_id)
        if current is None:
            raise ValueError("Prepared Research ingest job does not exist.")
        if current.state in {
            IngestJobState.DONE,
            IngestJobState.FAILED,
            IngestJobState.CANCELLED,
            IngestJobState.SKIPPED,
        }:
            return current
        failed = self.library_ingest_jobs.mark_failed(
            job_id,
            error="Research catalog dispatch could not be started.",
            require_persisted=True,
        )
        if failed is None:
            raise ValueError("Prepared Research ingest job cannot be failed.")
        return failed

    def _dispatch_research_source_catalog_job(self, job_id: str) -> None:
        """Dispatch an already-persisted ingest through its bound adapter."""

        requeued = self.library_ingest_jobs.get_job(job_id)
        if requeued is None:
            raise ValueError("Replacement ingest job does not exist.")
        if requeued.dispatch_held:
            requeued = self.library_ingest_jobs.release_dispatch_hold(
                job_id, require_persisted=True
            )
            if requeued is None:
                raise ValueError("Prepared Research ingest job cannot be released.")
        if requeued.origin == "local":
            if self.media_db is None:
                self.library_ingest_jobs.mark_failed(
                    requeued.job_id,
                    error="Media database is unavailable.",
                )
                return
            if (
                requeued.state is IngestJobState.PARSING
                and requeued.retry_of_job_id
                and requeued.research_source_operation_id
            ):
                pending = getattr(
                    self,
                    "_research_source_parse_dispatch_pending",
                    None,
                )
                if pending is None:
                    pending = set()
                    self._research_source_parse_dispatch_pending = pending
                pending.add(requeued.job_id)
            self._top_up_ingest_parse_pool()
            return
        if requeued.origin != "server":
            raise ValueError("Replacement ingest authority is unsupported.")

        try:
            if is_web_clip_source(requeued.source_path):
                kwargs = build_web_clip_kwargs(
                    requeued.source_path,
                    options=requeued.ingest_options,
                    title=requeued.title,
                    author=requeued.author,
                    keywords=requeued.keywords,
                )
                self._send_web_clip_job(requeued.job_id, kwargs)
            else:
                kwargs = build_server_ingest_kwargs(
                    requeued.source_path,
                    options=requeued.ingest_options,
                    title=requeued.title,
                    author=requeued.author,
                    keywords=requeued.keywords,
                    perform_analysis=requeued.perform_analysis,
                )
                self._send_server_ingest_job(requeued.job_id, kwargs)
        except (NotAWebClipSource, ServerIngestUnsupported) as exc:
            self.library_ingest_jobs.mark_failed(
                requeued.job_id,
                error=str(exc),
                permanent=True,
            )
            return None

    def retry_library_ingest_job_with_provider(
        self,
        job_id: str,
        provider: str,
    ) -> Optional[LibraryIngestJob]:
        """Run the supported provider recovery for ordinary Library jobs.

        Research-owned jobs preserve the operation's captured options and
        route through the durable catalog retry owner instead.
        """

        if provider != "faster-whisper":
            return None
        return self.retry_library_ingest_job(
            job_id,
            transcription_provider=provider,
        )

    # -- Parse-pool sizing + lifecycle (coordinator) -----------------------

    def _ingest_parse_worker_count(self) -> int:
        """Resolve the parse-pool size from config, with a safe default.

        UI-thread only. Reads ``library.ingest_parse_workers`` via the
        dotted 1-arg ``get_cli_setting`` form (``load_settings()`` doesn't
        carry CLI ``[library.*]`` tables -- same bug-class guard as the
        rail-state read). An invalid, missing, or non-positive value falls
        back to the spec's default formula.

        Returns:
            The configured worker count when it int-coerces to a positive
            value; otherwise ``min(3, max(1, cpu_count - 1))``, where
            ``cpu_count`` is ``os.cpu_count()`` (guarded to ``2`` when that
            returns ``None``, e.g. on some containerized/sandboxed hosts).
        """
        try:
            configured = int(get_cli_setting("library.ingest_parse_workers"))
        except (TypeError, ValueError):
            configured = 0
        if configured > 0:
            return configured
        cpu_count = os.cpu_count() or 2
        return min(3, max(1, cpu_count - 1))

    def _ingest_heavy_lane_max_workers(self) -> int:
        """Resolve the heavy-lane (audio/video transcription) cap from config.

        UI-thread only. Reads ``library.ingest_heavy_lane_max_workers`` via the
        dotted 1-arg ``get_cli_setting`` form (same reason as
        ``_ingest_parse_worker_count``). Defaults to 1; a missing, invalid, or
        non-positive value clamps to 1 so heavy work is never permanently
        starved.
        """
        try:
            configured = int(get_cli_setting("library.ingest_heavy_lane_max_workers"))
        except (TypeError, ValueError):
            configured = 0
        return configured if configured > 0 else 1

    def _create_ingest_parse_pool(self, *, processes: int | None = None):
        """Create the Library ingest parse pool.

        UI-thread only. Test seam: monkeypatched to an inline-synchronous
        fake resource bundle (see
        ``Tests/Library/test_library_ingest_runner.py``) so pilots stay
        deterministic without spawning real OS processes. Real callers get a
        spawn-context ``multiprocessing.Pool`` and bounded progress queue.

        Not a ``concurrent.futures.ProcessPoolExecutor`` -- see the F3
        design spec's Architecture section: the executor's ``atexit`` hook
        joins running tasks, so an in-flight long transcription would block
        app exit for its full duration. ``Pool`` has a public
        ``terminate()`` the quit path relies on instead.

        Textual stderr workaround (live-QA crash fix): under Textual (app
        mode / textual-serve), ``sys.stderr`` is replaced by a capture
        object whose ``fileno()`` returns ``-1`` WITHOUT raising. CPython
        3.12's ``multiprocessing.resource_tracker._launch`` appends
        ``sys.stderr.fileno()`` to the fds it hands
        ``util.spawnv_passfds`` (its ``except Exception`` guard never
        fires, since ``-1`` is returned rather than raised), and
        ``spawnv_passfds`` rejects the list with ``ValueError: bad
        value(s) in fds_to_keep`` -- so the very first Pool construction
        (which ensure-runs the process-global resource tracker) crashed
        the app on its first ingest submission. When ``sys.stderr`` has no
        usable fd, both the queue and Pool are constructed under
        ``contextlib.redirect_stderr`` pointing at a genuinely fd-backed
        stream (``_ingest_pool_real_stderr``: ``sys.__stderr__``, else a
        kept-alive devnull handle). The tracker launches at most once per
        process, so covering construction is sufficient -- and applying
        the redirect on every (re)construction is harmless. Queue and Pool
        creation are one atomic owner operation: if Pool creation fails, the
        already-created queue is closed before the exception escapes.

        Args:
            processes: Physical worker count for this generation. ``None``
                uses the configured ordinary parse-pool size.
        """
        ctx = multiprocessing.get_context("spawn")
        if processes is None:
            processes = self._ingest_parse_worker_count()

        def _construct_resources() -> _IngestParsePoolResources:
            progress_queue = None
            try:
                progress_queue = ctx.Queue(maxsize=INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE)
                pool = ctx.Pool(
                    processes=processes,
                    initializer=initialize_ingest_parse_worker,
                    initargs=(progress_queue,),
                )
            except Exception:
                if progress_queue is not None:
                    for method_name in ("close", "cancel_join_thread"):
                        method = getattr(progress_queue, method_name, None)
                        if method is None:
                            continue
                        try:
                            method()
                        except Exception:
                            logger.error(
                                "Error cleaning up a partially constructed "
                                "Library ingest progress queue "
                                "(operation={}, queue_type={}).",
                                method_name,
                                type(progress_queue).__name__,
                            )
                raise
            return _IngestParsePoolResources(pool, progress_queue)

        # The combined initializer keeps worker import noise off the TUI and
        # installs this generation's progress sink.
        if _stream_fileno(sys.stderr) >= 0:
            return _construct_resources()
        with contextlib.redirect_stderr(_ingest_pool_real_stderr()):
            return _construct_resources()

    def _ensure_ingest_parse_pool(self, mode: str = _INGEST_GENERAL_POOL_MODE):
        """Return the current parse pool, lazily creating one if needed.

        UI-thread only.

        Args:
            mode: Resource class owned by a newly created generation.
        """
        if self._ingest_parse_pool is None:
            processes = (
                1
                if mode == _INGEST_EBOOK_POOL_MODE
                else self._ingest_parse_worker_count()
            )
            resources = self._create_ingest_parse_pool(processes=processes)
            pool = resources.pool
            progress_queue = resources.progress_queue
            try:
                sentinels = self._ingest_parse_pool_worker_sentinels(pool)
            except Exception:
                self._terminate_ingest_parse_pool_off_thread(
                    pool,
                    progress_queue,
                    None,
                )
                raise

            generation = getattr(self, "_ingest_parse_pool_generation", 0) + 1
            stop_event = threading.Event()
            self._ingest_parse_pool_generation = generation
            self._ingest_parse_jobs_by_generation = getattr(
                self, "_ingest_parse_jobs_by_generation", {}
            )
            self._ingest_parse_jobs_by_generation[generation] = set()
            self._ingest_parse_pool_stop_event = stop_event
            self._ingest_parse_pool = pool
            self._ingest_parse_pool_mode = mode
            self._ingest_parse_progress_queue = progress_queue
            self._ingest_parse_progress_thread = None
            if progress_queue is not None:
                self._ingest_parse_progress_thread = (
                    self._start_ingest_parse_progress_drain(
                        generation,
                        progress_queue,
                        stop_event,
                    )
                )
            if sentinels:
                self._start_ingest_parse_pool_monitor(generation, sentinels, stop_event)
        return self._ingest_parse_pool

    def _start_ingest_parse_progress_drain(
        self,
        generation: int,
        progress_queue: Any,
        stop_event: threading.Event,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> threading.Thread:
        """Start the bounded, latest-per-job drain for one pool generation."""

        def _drain() -> None:
            coalescer = ParseProgressCoalescer(
                interval=INGEST_PARSE_PROGRESS_FLUSH_SECONDS,
                started_at=clock(),
            )
            while not stop_event.is_set() and not self._ingest_shutdown:
                try:
                    raw_event = progress_queue.get(timeout=0.05)
                except queue.Empty:
                    raw_event = None
                except (EOFError, OSError, ValueError):
                    return
                if stop_event.is_set() or self._ingest_shutdown:
                    return
                if raw_event is not None:
                    try:
                        event = make_parse_progress_event(
                            raw_event.generation,
                            raw_event.job_id,
                            raw_event.phase,
                            raw_event.message,
                            raw_event.percent,
                        )
                    except Exception:
                        event = None
                    if event is not None:
                        coalescer.accept(event)
                batch = coalescer.take_due(clock())
                if batch:
                    if stop_event.is_set() or self._ingest_shutdown:
                        return
                    self._marshal_ingest_pool_call(
                        self._on_ingest_parse_progress_batch,
                        generation,
                        batch,
                    )

        thread = threading.Thread(
            target=_drain,
            name=f"library-ingest-progress-drain-{generation}",
            daemon=True,
        )
        thread.start()
        return thread

    @staticmethod
    def _ingest_parse_pool_worker_sentinels(pool: Any) -> Optional[tuple[Any, ...]]:
        """Snapshot real Pool worker sentinels; injected fakes may opt out."""
        workers = getattr(pool, "_pool", None)
        if workers is None:
            return None
        try:
            sentinels = tuple(worker.sentinel for worker in workers)
        except Exception as exc:
            raise RuntimeError(
                "Could not inspect parse-pool worker sentinels."
            ) from exc
        if not sentinels:
            raise RuntimeError("Parse pool started without worker sentinels.")
        return sentinels

    def _start_ingest_parse_pool_monitor(
        self,
        generation: int,
        sentinels: tuple[Any, ...],
        stop_event: threading.Event,
    ) -> threading.Thread:
        """Watch one real Pool generation for an unexpected worker exit."""

        def _monitor() -> None:
            try:
                ready = multiprocessing.connection.wait(sentinels)
            except Exception as exc:
                if stop_event.is_set() or self._ingest_shutdown:
                    return
                failure = RuntimeError(f"Parse-pool sentinel monitor failed: {exc}")
            else:
                if not ready or stop_event.is_set() or self._ingest_shutdown:
                    return
                failure = RuntimeError(
                    f"Library ingest parse-pool worker exited unexpectedly "
                    f"(generation {generation})."
                )
            if stop_event.is_set() or self._ingest_shutdown:
                return
            self.call_from_thread(
                self._handle_broken_ingest_parse_pool,
                generation,
                None,
                failure,
            )

        thread = threading.Thread(
            target=_monitor,
            name=f"library-ingest-pool-monitor-{generation}",
            daemon=True,
        )
        thread.start()
        return thread

    def _ingest_job_options(self, job: LibraryIngestJob) -> Dict[str, Any]:
        """Build ``run_parse_job``'s ``options`` dict from a job's fields.

        ``job.ingest_options`` is the canonical source of ingestion settings.
        It is expected to be a group-keyed snapshot (e.g. ``{"generic": {...},
        "pdf": {...}}``); values from the detected type group override values
        from the ``generic`` group. The older scalar fields
        (``perform_analysis``, ``chunk_enabled``, ``chunk_size``) are used only
        as deprecated fallbacks when ``ingest_options`` is empty or does not
        contain a value.

        (task-3301) The three previously dead controls resolve here:

        * ``encoding`` (generic group) travels to the plaintext/html readers.
        * ``chunk_options`` carries the form's size/overlap as ints (the
          snapshot boundary coerces, but restored/persisted jobs may still
          hold display strings), in both the ``size`` spelling (audio/video
          option maps) and the ``max_size`` spelling
          (``improved_chunking_process``); the overlap fallback is the
          generic schema default -- the value the UI displays -- not a
          hardcoded constant. (task-3301/3303 xhigh review round 2,
          F11+F12) An explicit ``method`` ALWAYS travels: pdf and
          audio/video get ``words`` so the generic size/overlap hint
          ("words · 100-5000") is true everywhere the processors would
          otherwise setdefault sentences (a ~10-30x unit lie); the ebook
          group maps its panel choice ("chapters" -> the chunker's
          ``ebook_chapters``, other names verbatim) and falls back to the
          pre-branch ``sentences`` when the snapshot predates the field
          (fresh snapshots always carry the schema default -- absence IS
          the legacy marker, so a requeued old job keeps its original
          scheme). The text tail's own default is already words.
        * When analysis is requested, the configured analysis provider
          (``[analysis_defaults] provider``) is resolved through the shared
          readiness seam, then constrained to a chat-dispatchable name
          (task-3301 xhigh review round): ready adds ``api_name`` (the
          normalized ``API_CALL_HANDLERS`` key), ``api_key`` (``None`` for
          keyless local providers, paired with the explicit
          ``analysis_keyless_ok`` opt-in the processors' credential gates
          require), and ``analysis_call`` (model/temperature/top_p/min_p/
          max_tokens from ``[analysis_defaults]``, viewer-parity defaults)
          plus ``system_prompt`` when the section configures one; not ready
          (including readiness-ready providers with no chat handler) adds
          ``analysis_skipped_reason`` so the job records WHY analysis is
          absent instead of silently dropping it.

        ``custom_prompt`` and ``system_prompt`` are copied from the generic
        snapshot only when analysis is requested. They remain in the job
        snapshot while analysis is off, but parser options omit them so stale
        instructions cannot reach a backend that will not execute analysis.
        ``metadata`` remains absent (``None`` inside the worker's
        ``options.get(...)`` reads).
        """
        opts = job.ingest_options or {}
        group = get_type_group(job.source_path)

        # Resolve a flat option map from the generic group and the detected
        # type-specific group, with type-specific values taking precedence.
        generic_opts: dict[str, Any] = dict(opts.get("generic", {}))
        flat_opts: dict[str, Any] = dict(generic_opts)
        flat_opts.update(opts.get(group, {}) or {})

        def _as_int(value: Any, fallback: int) -> int:
            """Coerce a possibly-display-string number, falling back."""
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                return fallback

        perform_analysis = bool(flat_opts.get("analyze", job.perform_analysis))
        overlap_default = _as_int(generic_option_default("chunk_overlap", 100), 100)
        chunk_size = _as_int(
            flat_opts.get("chunk_size", job.chunk_size), job.chunk_size
        )
        chunk_enabled = bool(flat_opts.get("chunk", job.chunk_enabled))

        # (task 10, spec §9.1 AC 34) Template resolution -- ingest order:
        # picker/batch choice -> config [chunking] default_template -> plain
        # options. Resolution happens HERE (the app process owns the media
        # DB) and the resolved DICT travels inside chunk_options: the parse
        # worker must stay DB-free. An unresolvable or stored-invalid choice
        # raises a NAMED error (TemplateResolutionError /
        # InvalidTemplateError, AC 37 / AC-24b) -- the ingest dispatch
        # catches both and fails THIS item; there is never a silent
        # fallback to plain chunking.
        # (task 4, auto-selection spec §4.3) A picker choice of the Auto
        # sentinel ("auto") resolves to an AutoDecision instead: the job's
        # ALREADY-KNOWN metadata (detected type / title / filename / URL)
        # feeds resolve_auto -- nothing re-reads file contents at selection
        # time. A template-tier win is consumed exactly like a manual pick;
        # a plan-tier win materializes the planner's options as this
        # parse's defaults; a plain-tier win changes nothing below.
        ingest_template: dict[str, Any] | None = None
        auto_decision: Any = None
        plan_options: dict[str, Any] | None = None
        if chunk_enabled:
            from .Chunking.template_runtime import resolve_ingest_template

            source_is_url = (
                str(job.source_path or "").lower().startswith(("http://", "https://"))
            )
            resolved = resolve_ingest_template(
                getattr(self, "media_db", None),
                str(flat_opts.get("chunk_template") or "").strip() or None,
                media_type=str(job.detected_type or "").strip() or None,
                title=str(job.title or "").strip() or None,
                filename=(None if source_is_url else PurePath(job.source_path).name),
                url=str(job.source_path) if source_is_url else None,
            )
            from .Chunking.auto_selection import AutoDecision

            if isinstance(resolved, AutoDecision):
                auto_decision = resolved
                if resolved.tier == "template" and isinstance(resolved.template, dict):
                    ingest_template = resolved.template
                elif resolved.tier == "plan" and isinstance(
                    resolved.chunk_options, dict
                ):
                    plan_options = dict(resolved.chunk_options)
                # plain tier: fall through to today's default options
            else:
                ingest_template = resolved

        if ingest_template is not None or plan_options is not None:
            # (task 10, spec §9.1 AC 35 -- the precedence ruling) A resolved
            # template's chunk-stage options beat the ingest builder's
            # DEFAULTS; only a value the user explicitly CHANGED in the
            # ingest form beats the template. Left as-is, the builder's
            # always-on size/overlap (+ per-group method injection) would
            # arrive at the Chunker as explicit options that override the
            # template on every path -- the picker would be inert.
            # (task 4, auto-selection §4.3) The plan tier rides the SAME
            # ruling: the planner's options are the defaults, a
            # user-changed form value still wins.
            #
            # Mechanism: the form snapshot ALWAYS carries explicit values
            # (``_build_ingest_options_snapshot`` seeds every schema
            # default), so "differs from the schema default" is the only
            # user-changed signal available at this seam. Values equal to
            # the schema default are dropped here and re-derived from the
            # template by the parse seam's materialization
            # (``materialize_template_chunk_options``); values that differ
            # ride along and win at the Chunker's explicit-beats-template
            # merge. A snapshot WITHOUT the key (pre-field legacy jobs) has
            # no user signal at all and defaults to the template winning.
            size_schema_default = _as_int(
                generic_option_default("chunk_size", DEFAULT_CHUNK_SIZE),
                DEFAULT_CHUNK_SIZE,
            )
            overlap_schema_default = overlap_default
            if ingest_template is not None:
                chunk_options: dict[str, Any] = {"template": ingest_template}
            else:
                # Plan tier: the planner's options travel as this parse's
                # chunk-stage defaults; ``size`` mirrors ``max_size`` for
                # the audio/video key-by-key re-projection (the same alias
                # ``materialize_template_chunk_options`` fills).
                chunk_options = dict(plan_options)
                if "max_size" in chunk_options:
                    chunk_options.setdefault("size", chunk_options["max_size"])
            if "chunk_size" in flat_opts and chunk_size != size_schema_default:
                chunk_options["size"] = chunk_size
                chunk_options["max_size"] = chunk_size
            if (
                "chunk_overlap" in flat_opts
                and _as_int(flat_opts.get("chunk_overlap"), overlap_default)
                != overlap_schema_default
            ):
                chunk_options["overlap"] = _as_int(
                    flat_opts.get("chunk_overlap"), overlap_default
                )
        else:
            chunk_options = (
                {
                    "size": chunk_size,
                    "max_size": chunk_size,
                    "overlap": _as_int(
                        flat_opts.get("chunk_overlap", overlap_default),
                        overlap_default,
                    ),
                }
                if chunk_enabled
                else None
            )
        if auto_decision is not None and chunk_options is not None:
            # (task 4, auto-selection spec §4.4) The decision's travel
            # ticket to the persist seam (mode/auto_tier/auto_rationale).
            # The parse seam POPS this key before any branch dispatch, so
            # no processor or the Chunker ever sees it.
            chunk_options["auto"] = {
                "tier": str(auto_decision.tier),
                "rationale": [str(line) for line in (auto_decision.rationale or [])],
            }

        options: dict[str, Any] = {
            "title": job.title or None,
            "author": job.author or None,
            "keywords": list(job.keywords) or None,
            "perform_analysis": perform_analysis,
            # (TASK-20973) Mint the URL-provenance fact HERE, where the
            # submission's lineage is known, so the video arm's egress
            # check trusts a private host only for a URL the user actually
            # entered. A general Library-import submission IS user entry
            # (the source string came from the import form); a
            # research-source job's URL is agent-discovered catalog
            # content, NOT user entry, and must fail closed. Deriving
            # this from the job's own field -- rather than from "who
            # happens to call" -- is what makes adding a caller unable to
            # silently change the trust decision.
            "url_provenance": (
                UrlProvenance.UNKNOWN
                if job.research_source_operation_id
                else UrlProvenance.USER_ENTERED
            ),
            # These generic fields intentionally travel independently of the
            # detected type-group branch. The downstream local overwrite/RAG
            # behavior is owned by later work; this seam only makes the form
            # snapshot honest for consumers that already read these options.
            "overwrite_existing": bool(
                generic_opts.get(
                    "overwrite_existing",
                    generic_option_default("overwrite_existing", False),
                )
            ),
            "generate_embeddings": bool(
                generic_opts.get(
                    "generate_embeddings",
                    generic_option_default("generate_embeddings", True),
                )
            ),
            "encoding": flat_opts.get("encoding"),
            "chunk_options": chunk_options,
        }
        # ``template_active`` gates the per-group METHOD injection below:
        # the pdf/audio-video/image "words" and the ebook group mapping are
        # builder DEFAULTS (the user cannot type a method in those panels),
        # so under a resolved template they must not be injected -- the
        # template's method wins via materialization. A user-changed ebook
        # chunk_method (differs from the select's schema default) still
        # travels. (task 4, auto-selection §4.3) An auto PLAN-tier win
        # governs identically: its method is a derived default, not a user
        # choice, so the injection is skipped for it too; the auto PLAIN
        # tier keeps today's injections (it changes nothing).
        template_active = ingest_template is not None or plan_options is not None

        if perform_analysis:
            # Prompts remain in the persisted generic snapshot while analysis
            # is off, but parser options must not carry instructions no
            # backend will execute.
            options["custom_prompt"] = generic_opts.get(
                "custom_prompt", generic_option_default("custom_prompt", "")
            )
            options["system_prompt"] = generic_opts.get(
                "system_prompt", generic_option_default("system_prompt", "")
            )
            resolution = resolve_ingest_analysis_provider(
                getattr(self, "app_config", None)
            )
            if resolution.ready:
                # (task-3301 xhigh review round) The NORMALIZED dispatch
                # name (an `API_CALL_HANDLERS` key) travels, not the
                # display spelling -- it is what `chat_api_call` and the
                # summarizer's alias map accept (F5).
                options["api_name"] = resolution.dispatch_name
                options["api_key"] = resolution.api_key
                if resolution.keyless:
                    # (F8) Explicit keyless opt-in: the processors' analysis
                    # gates only dispatch without a credential when the
                    # readiness seam vouched for keyless operation.
                    options["analysis_keyless_ok"] = True
                # (F10) The full [analysis_defaults] call shape, so an
                # ingest analysis runs with the same model/sampling the
                # Media viewer's analysis panel would use.
                options["analysis_call"] = {
                    "model": resolution.model,
                    "temperature": resolution.temperature,
                    "top_p": resolution.top_p,
                    "min_p": resolution.min_p,
                    "max_tokens": resolution.max_tokens,
                }
                if resolution.system_prompt and not options.get("system_prompt"):
                    options["system_prompt"] = resolution.system_prompt
            else:
                options["analysis_skipped_reason"] = resolution.short_reason

        if group == "pdf":
            if options["chunk_options"] is not None and not template_active:
                # (F12) ``process_pdf`` setdefaults method='sentences',
                # under which the form's "words · 100-5000" size hint is a
                # ~10-30x unit lie (500 SENTENCES ~= one chunk per
                # document). Words is what the hint promises. (task 10)
                # Under a resolved template this injection is a builder
                # DEFAULT and is skipped -- the template's method wins.
                options["chunk_options"]["method"] = "words"
            options["pdf_engine"] = flat_opts.get("engine") or flat_opts.get(
                "pdf_engine"
            )
            options["page_range"] = flat_opts.get("pages")
            options["ocr"] = flat_opts.get("ocr", flat_opts.get("enable_ocr", False))
            options["extract_images"] = flat_opts.get("extract_images", False)
            # (task-3303) OCR detail: language + backend, with the
            # processor's own defaults as the fallbacks. The panel gates
            # the OCR toggle to the docling/docext engines, so a silent
            # OCR-under-pymupdf no-op can no longer be *asked for*; the
            # values themselves always travel (process_pdf ignores them
            # when the parser cannot OCR).
            options["ocr_language"] = flat_opts.get("ocr_language") or "en"
            options["ocr_backend"] = flat_opts.get("ocr_backend") or "auto"
        elif group == "document":
            # (task-3303) The document group layers ON TOP of generic:
            # ``flat_opts`` already merged generic (analyze/chunk/encoding)
            # under these, so document files keep task-3301's chunking and
            # analysis while gaining ``process_document``'s own knobs.
            options["processing_method"] = flat_opts.get("processing_method") or "auto"
            options["enable_ocr"] = flat_opts.get(
                "ocr", flat_opts.get("enable_ocr", False)
            )
            options["ocr_language"] = flat_opts.get("ocr_language") or "en"
        elif group == "audio_video":
            if options["chunk_options"] is not None and not template_active:
                # (F12) The audio/video branch defaults chunk_method to
                # sentences too -- same unit-lie fix as the pdf branch.
                # (task 10) Skipped under a resolved template (a builder
                # default, not a user choice).
                options["chunk_options"]["method"] = "words"
            provider = flat_opts.get("transcription_provider")
            if provider is None:
                provider = "default"
            target_language = flat_opts.get("translation_target_language")
            if target_language is None:
                target_language = flat_opts.get("target_language")
            if (
                target_language is None
                and flat_opts.get("translate_to_english")
                # (task-3303 xhigh review round 2, F9) Honor the checkbox's
                # own schema gate: its value survives in the snapshot after
                # the provider select moves to one that rejects translation
                # (transcribe-cpp/parakeet raise BatchSTTRoutingError, which
                # failed the WHOLE batch at dispatch). The normalized
                # provider is passed so the gate sees the same value the
                # route resolution below will use.
                and field_gate_open(
                    "audio_video",
                    "translate_to_english",
                    {**flat_opts, "transcription_provider": provider},
                )
            ):
                # (task-3303) The panel's translate toggle. An explicit
                # target (retry overrides, older snapshots) stays
                # authoritative; the checkbox only fills the gap.
                target_language = "en"
            route = resolve_batch_stt_route(
                provider=provider,
                language=flat_opts.get("language"),
                target_language=target_language,
                precision=flat_opts.get("transcription_precision"),
            )
            options["transcription_provider"] = route.provider
            selected_model_dir = (
                str(flat_opts.get("transcription_model_dir") or "").strip()
                if route.provider == "parakeet-onnx"
                else ""
            )
            options["transcription_model_dir"] = selected_model_dir or None
            selected_model = route.model
            if selected_model is None and route.requested_provider not in {
                "default",
                "transcribe-cpp",
            }:
                selected_model = flat_opts.get("model") or flat_opts.get(
                    "transcription_model"
                )
                if route.requested_provider == "faster-whisper" and not selected_model:
                    selected_model = "base"
            options["transcription_model"] = selected_model
            options["language"] = route.requested_language
            options["translation_target_language"] = route.target_language
            options["transcription_precision"] = route.precision
            options["transcription_local_files_only"] = route.local_files_only
            options["transcription_batch_route_resolved"] = True
            options["timestamps"] = flat_opts.get("timestamps", True)
            options["diarization"] = flat_opts.get("diarization", False)
            # (task-3303) VAD filter -- travels as its own option; the
            # parse worker hands it to the processors' ``vad_use``.
            options["vad_filter"] = bool(flat_opts.get("vad_filter", False))
            # (task-3306) Time-range trim: format-gated at the option layer
            # (HH:MM:SS or seconds); blank means unbounded on that side.
            start_trim = str(flat_opts.get("start_time") or "").strip()
            end_trim = str(flat_opts.get("end_time") or "").strip()
            options["start_time"] = start_trim or None
            options["end_time"] = end_trim or None
            # (task-3306) Gated URL downloads: a cookies FILE PATH only
            # (yt-dlp cookiefile) -- raw cookie text is a credential, and
            # this options dict persists with the job and echoes into
            # config.toml. Its presence IS the use_cookies flag, so there
            # is no separate toggle to go stale. Only the video (yt-dlp)
            # branch of ``parse_local_file_for_ingest`` consumes it; the
            # audio downloader's cookies parameter has JSON-dict semantics
            # a path would crash.
            # (xhigh review round) Validated here, not forwarded verbatim:
            # an unusable path used to degrade into a silent "Invalid
            # cookie format" debug line inside the downloader.
            cookies_file = str(flat_opts.get("cookies_file") or "").strip()
            cookies_path, cookies_problem = _resolve_ingest_cookies_file(cookies_file)
            options["use_cookies"] = bool(cookies_path)
            options["cookies"] = cookies_path
            if cookies_problem:
                options["cookies_problem"] = cookies_problem
            # (task-3306) Recursive map-reduce summary; the processors'
            # analysis tail consumes it only when analysis actually runs,
            # so an idle True is inert rather than a stale hazard.
            options["summarize_recursively"] = bool(
                flat_opts.get("summarize_recursively", False)
            )
            failed_attempt = job.retry_source_failure_provenance
            options["transcription_context"] = {
                "attempt_id": f"{job.job_id}-attempt-{job.retry_count + 1}",
                "batch_id": job.batch_id,
                "job_id": job.job_id,
                "retry_of_attempt_id": failed_attempt.get("attempt_id")
                if failed_attempt
                else None,
                "retry_of_job_id": job.retry_of_job_id,
                "retry_source_failure_provenance": failed_attempt,
            }
            external_scope_id = str(
                flat_opts.get("transcription_external_scope_id") or ""
            ).strip()
            if external_scope_id:
                options["transcription_context"]["external_scope_id"] = (
                    external_scope_id
                )
            if route.provider == "transcribe-cpp":
                configured_path = get_cli_setting(
                    "transcription.transcribe_cpp.model_path"
                )
                options["transcription_context"]["model_path"] = (
                    configured_path
                    if isinstance(configured_path, str) and configured_path
                    else None
                )
        elif group == "image":
            # (task-3307) The image panel's OCR knobs travel under the
            # names the parse branch reads; fallbacks mirror
            # ``process_image``'s own declared defaults. OCR defaults ON:
            # the extracted text IS the imported content, and a no-text
            # parse fails honestly at the persist seam.
            if options["chunk_options"] is not None and not template_active:
                # (F12 parity) ``process_image`` chunks the OCR text via
                # ``improved_chunking_process``; an explicit words method
                # keeps the generic "words · 100-5000" size hint true here
                # too. (task 10) Skipped under a resolved template.
                options["chunk_options"]["method"] = "words"
            options["ocr"] = flat_opts.get("ocr", flat_opts.get("enable_ocr", True))
            options["ocr_language"] = flat_opts.get("ocr_language") or "en"
            options["ocr_backend"] = flat_opts.get("ocr_backend") or "auto"
        elif group == "ebook":
            options["extraction_method"] = (
                flat_opts.get("extraction_method")
                or flat_opts.get("method")
                or flat_opts.get("html_converter")
            )
            options["split_chapters"] = flat_opts.get("split_chapters", True)
            options["include_toc"] = flat_opts.get(
                "include_toc", flat_opts.get("extract_toc", True)
            )
            # (task-3303) The panel's chunk-method choice: the human
            # "chapters" maps to the chunker's real ``ebook_chapters``
            # method; the other names travel verbatim. Only meaningful when
            # chunking is on.
            ebook_chunk_method = str(flat_opts.get("chunk_method") or "").strip()
            if options["chunk_options"] is not None:
                if template_active and ebook_chunk_method in (
                    "",
                    "chapters",  # the ebook select's schema default
                ):
                    # (task 10, AC 35) The select's schema default
                    # ("chapters") is a builder default: under a resolved
                    # template the template's method wins and nothing is
                    # injected. An ABSENT value under a template also lets
                    # the template win (the template IS the scheme the
                    # user picked; there is no legacy scheme to preserve).
                    pass
                elif ebook_chunk_method:
                    options["chunk_options"]["method"] = (
                        "ebook_chapters"
                        if ebook_chunk_method == "chapters"
                        else ebook_chunk_method
                    )
                else:
                    # (task-3303 xhigh review round 2, F11) No chunk_method
                    # in the snapshot means the job PREDATES the field --
                    # fresh submissions always seed the schema default (see
                    # ``_build_ingest_options_snapshot``). The old builder
                    # forced sentences for every group, so a requeued
                    # legacy job must keep that scheme rather than silently
                    # switching to the processor's chapters default.
                    options["chunk_options"]["method"] = "sentences"

        return options

    def _create_local_stt_executor(self) -> LocalSTTExecutor:
        """Construct the one app-owned heavy STT executor lazily."""

        return LocalSTTExecutor()

    def _ensure_local_stt_executor(self) -> LocalSTTExecutor:
        with self._local_stt_executor_lock:
            if self._ingest_shutdown:
                raise ExecutorUnavailableError("Library ingest is shutting down")
            executor = getattr(self, "_local_stt_executor", None)
            if executor is None:
                executor = self._create_local_stt_executor()
                self._local_stt_executor = executor
            return executor

    def _recycle_idle_local_stt_reference(self, reference: "ArtifactRef") -> bool:
        """Recycle an existing idle STT resident that leases ``reference``."""

        with self._local_stt_executor_lock:
            if self._ingest_shutdown:
                return False
            executor = getattr(self, "_local_stt_executor", None)
        if executor is None:
            return False
        return executor.recycle_idle_managed_reference(
            (reference.artifact_id, reference.revision, reference.variant)
        )

    def _create_parakeet_source_service(self) -> Any:
        """Construct the shared download-free Parakeet source service lazily."""

        from tldw_chatbook.STT.parakeet_sources import ParakeetSourceService

        return ParakeetSourceService()

    def _ensure_parakeet_source_service(self) -> Any:
        """Return the one app-owned Parakeet source service."""

        with self._local_stt_executor_lock:
            if self._ingest_shutdown:
                raise ExecutorUnavailableError("Local STT is shutting down")
            service = getattr(self, "_parakeet_source_service", None)
            if service is None:
                service = self._create_parakeet_source_service()
                listener = self._sync_parakeet_source_scopes
                self._parakeet_source_service = service
                self._parakeet_source_registry_listener = listener
                self.library_ingest_jobs.add_listener(listener)
                listener()
            return service

    @staticmethod
    def _parakeet_scope_id_for_job(job: LibraryIngestJob) -> str:
        """Return the path-free verifier owner captured for one Library job."""

        audio_options = (job.ingest_options or {}).get("audio_video", {})
        if isinstance(audio_options, dict):
            scope_id = audio_options.get("transcription_external_scope_id")
            if isinstance(scope_id, str) and scope_id.strip():
                return scope_id.strip()
        return job.batch_id or job.job_id

    def _sync_parakeet_source_scopes(self) -> None:
        """Release only source scopes the registry observed and then settled."""

        service = getattr(self, "_parakeet_source_service", None)
        if service is None:
            return
        active_states = {
            IngestJobState.QUEUED,
            IngestJobState.PARSING,
            IngestJobState.WRITING,
        }
        active = {
            self._parakeet_scope_id_for_job(job)
            for job in self.library_ingest_jobs.jobs()
            if job.state in active_states
        }
        active.update(getattr(self, "_parakeet_submitting_scope_ids", ()))
        service.release_scopes_except(active)

    def _ensure_local_stt_dispatch_coordinator(
        self,
    ) -> LocalSTTDispatchCoordinator:
        """Return the one app-owned admission coordinator lazily."""

        with self._local_stt_executor_lock:
            if self._ingest_shutdown:
                raise ExecutorUnavailableError("Local STT is shutting down")
            executor = self._ensure_local_stt_executor()
            coordinator = getattr(self, "_local_stt_dispatch_coordinator", None)
            if coordinator is None:
                coordinator = LocalSTTDispatchCoordinator(
                    executor,
                    on_dictation_idle=lambda: self._marshal_local_stt_call(
                        self._top_up_ingest_parse_pool
                    ),
                )
                self._local_stt_dispatch_coordinator = coordinator
            return coordinator

    def _create_console_dictation_service(self, **kwargs: Any) -> Any:
        """Build Console dictation without importing its native stack eagerly."""

        from tldw_chatbook.Audio.dictation_service_lazy import (
            LazyLiveDictationService,
        )
        from tldw_chatbook.Local_Ingestion.transcription_service import (
            TranscriptionService,
        )

        app_loop_running = getattr(self, "_loop", None) is not None
        on_app_thread = threading.get_ident() == getattr(self, "_thread_id", None)
        if app_loop_running and not on_app_thread:
            source_service = self.call_from_thread(self._ensure_parakeet_source_service)
        else:
            source_service = self._ensure_parakeet_source_service()
        return LazyLiveDictationService(
            **kwargs,
            transcription_service_factory=lambda: TranscriptionService(
                local_stt_dispatcher=self._ensure_local_stt_dispatch_coordinator(),
                parakeet_source_service=source_service,
            ),
        )

    def _build_local_stt_dispatch(
        self,
        job: LibraryIngestJob,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the exact private model identity for one eligible job."""

        provider = options["transcription_provider"]
        attempt_id = f"{job.job_id}-attempt-{job.retry_count + 1}"
        local_source = None
        managed_store_root = None
        managed_artifact_ref = None
        managed_dependency_refs: tuple[tuple[str, str, str], ...] = ()
        root_revision = None
        closure_fingerprint = None
        device = ExecutionDevice.CPU

        if provider == "transcribe-cpp":
            from tldw_chatbook.Model_Artifacts.gguf_admission import (
                validate_local_gguf,
            )

            context = options.get("transcription_context") or {}
            configured_path = (
                context.get("model_path") if isinstance(context, dict) else None
            )
            model_id = "local-gguf:unavailable"
            if isinstance(configured_path, str) and configured_path:
                admission = validate_local_gguf(Path(configured_path))
                local_source = snapshot_local_source((admission.path,))
                model_id = f"local-gguf:{admission.metadata.architecture}"
            device = ExecutionDevice.AUTO
        else:
            model_id = options.get("transcription_model") or PARAKEET_V2_MODEL
            precision = options.get("transcription_precision") or "int8"
            selected_dir = options.get("transcription_model_dir")
            from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

            resolved = self._ensure_parakeet_source_service().resolve(
                ParakeetSourceKey.from_values(model_id, precision),
                override=selected_dir,
                scope_id=self._parakeet_scope_id_for_job(job),
            )
            options.update(resolved.option_updates)
            return {
                "attempt_id": attempt_id,
                "identity": resolved.identity,
                "local_source": resolved.local_source,
                "managed_store_root": resolved.managed_store_root,
                "managed_artifact_ref": resolved.managed_artifact_ref,
                "managed_dependency_refs": resolved.managed_dependency_refs,
            }

        identity = ModelIdentity(
            provider_id=provider,
            model_id=model_id,
            root_revision=root_revision,
            closure_fingerprint=closure_fingerprint,
            precision=options.get("transcription_precision") or "int8",
            device=device,
            local_snapshot_token=(
                local_source.token if local_source is not None else None
            ),
        )
        return {
            "attempt_id": attempt_id,
            "identity": identity,
            "local_source": local_source,
            "managed_store_root": managed_store_root,
            "managed_artifact_ref": managed_artifact_ref,
            "managed_dependency_refs": managed_dependency_refs,
        }

    def _submit_local_stt_job(
        self,
        job: LibraryIngestJob,
        options: dict[str, Any],
    ) -> None:
        if options.get("transcription_provider") == "parakeet-onnx":
            self._ensure_parakeet_source_service()
        attempt_id = f"{job.job_id}-attempt-{job.retry_count + 1}"
        self._ingest_local_stt_jobs[job.job_id] = (0, attempt_id)
        thread = threading.Thread(
            target=self._dispatch_local_stt_job,
            args=(job, options, attempt_id),
            name=f"library-local-stt-dispatch-{job.job_id}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._ingest_local_stt_jobs.pop(job.job_id, None)
            raise

    def _dispatch_local_stt_job(
        self,
        job: LibraryIngestJob,
        options: dict[str, Any],
        attempt_id: str,
    ) -> None:
        """Build identity and perform the bounded spawn handshake off-loop."""

        try:
            dispatch = self._build_local_stt_dispatch(job, options)
            if dispatch["attempt_id"] != attempt_id:
                raise RuntimeError("Local STT attempt identity changed")
            coordinator = self._ensure_local_stt_dispatch_coordinator()
            generation = coordinator.submit_library(
                attempt_id=attempt_id,
                job_id=job.job_id,
                source=FileAudioSource(Path(job.source_path)),
                identity=dispatch["identity"],
                options=options,
                local_source=dispatch["local_source"],
                managed_store_root=dispatch["managed_store_root"],
                managed_artifact_ref=dispatch["managed_artifact_ref"],
                managed_dependency_refs=dispatch["managed_dependency_refs"],
                on_event=functools.partial(self._ingest_local_stt_event, job.job_id),
                on_result=functools.partial(self._ingest_local_stt_result, job.job_id),
                on_failure=functools.partial(
                    self._ingest_local_stt_failure, job.job_id
                ),
                explicit_retry=job.retry_count > 0,
            )
        except ExecutorBusyError:
            self._marshal_local_stt_call(
                self._on_ingest_local_stt_deferred,
                job.job_id,
                attempt_id,
            )
            return
        except Exception as exc:
            provider = str(options.get("transcription_provider") or "")
            code, actions = self._classify_local_stt_dispatch_error(provider, exc)
            self._marshal_local_stt_call(
                self._on_ingest_local_stt_dispatch_failure,
                job.job_id,
                attempt_id,
                code,
                actions,
                type(exc).__name__,
            )
            return
        self._marshal_local_stt_call(
            self._on_ingest_local_stt_submitted,
            job.job_id,
            generation,
            attempt_id,
        )

    @staticmethod
    def _classify_local_stt_dispatch_error(
        provider: str,
        error: BaseException,
    ) -> tuple[TranscriptionFailureCode, tuple[str, ...]]:
        from tldw_chatbook.STT.parakeet_sources import (
            ParakeetSourceError,
            ParakeetSourceErrorCode,
        )

        missing_model = isinstance(error, ParakeetSourceError) and error.code in {
            ParakeetSourceErrorCode.VAD_UNAVAILABLE,
            ParakeetSourceErrorCode.MANAGED_UNAVAILABLE,
        }
        unavailable = isinstance(error, (ExecutorBusyError, ExecutorUnavailableError))
        if missing_model:
            code = TranscriptionFailureCode.MODEL_NOT_INSTALLED
        elif unavailable:
            code = TranscriptionFailureCode.PROVIDER_UNAVAILABLE
        else:
            code = TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE
        actions = ["retry_faster_whisper"]
        if provider == "transcribe-cpp":
            actions.insert(0, "choose_another_gguf")
        return code, tuple(actions)

    def _marshal_local_stt_call(
        self,
        callback: Callable[..., Any],
        *args: Any,
    ) -> None:
        if self._ingest_shutdown:
            return
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:
            if not self._ingest_shutdown:
                callback_name = getattr(
                    callback,
                    "__name__",
                    type(callback).__name__,
                )
                logger.error(
                    "Library local STT callback could not be marshaled (callback={}).",
                    callback_name,
                )

    def _on_ingest_local_stt_submitted(
        self,
        job_id: str,
        generation: int,
        attempt_id: str,
    ) -> None:
        binding = self._ingest_local_stt_jobs.get(job_id)
        if (
            self._ingest_shutdown
            or binding is None
            or binding[1] != attempt_id
            or generation <= binding[0]
        ):
            return
        if self._claim_ingest_local_stt_job(job_id) is None:
            self._ingest_local_stt_jobs.pop(job_id, None)
            return
        self._ingest_local_stt_jobs[job_id] = (generation, attempt_id)

    def cancel_local_ingest_job(self, job_id: str) -> bool:
        """Request cooperative cancellation for one bound local STT attempt."""

        binding = self._ingest_local_stt_jobs.get(job_id)
        executor = getattr(self, "_local_stt_executor", None)
        job = self.library_ingest_jobs.get_job(job_id)
        if (
            binding is None
            or binding[0] <= 0
            or executor is None
            or job is None
            or job.state is not IngestJobState.PARSING
        ):
            return False
        if not executor.cancel(binding[1]):
            return False
        progress = dict(job.progress or {})
        progress["cancel_requested"] = True
        self.library_ingest_jobs.update_progress(job_id, progress=progress)
        return True

    def force_stop_local_ingest_job(self, job_id: str) -> bool:
        """Force-stop one cancel-requested local STT attempt off the UI thread."""

        binding = self._ingest_local_stt_jobs.get(job_id)
        executor = getattr(self, "_local_stt_executor", None)
        job = self.library_ingest_jobs.get_job(job_id)
        if (
            binding is None
            or binding[0] <= 0
            or executor is None
            or job is None
            or job.state is not IngestJobState.PARSING
            or not bool((job.progress or {}).get("cancel_requested"))
        ):
            return False
        thread = threading.Thread(
            target=self._force_stop_local_stt_attempt,
            args=(executor, binding[1]),
            name=f"library-local-stt-force-stop-{job_id}",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            return False
        return True

    def _force_stop_local_stt_attempt(
        self,
        executor: LocalSTTExecutor,
        attempt_id: str,
    ) -> None:
        if not executor.force_stop(attempt_id):
            return
        if executor.wait_for_retirement(10.0):
            self._marshal_local_stt_call(self._top_up_ingest_parse_pool)

    def _on_ingest_local_stt_deferred(
        self,
        job_id: str,
        attempt_id: str,
    ) -> None:
        """Release a provisional Library claim blocked by dictation admission."""

        if self._ingest_shutdown or self._ingest_local_stt_jobs.get(job_id) != (
            0,
            attempt_id,
        ):
            return
        self._ingest_local_stt_jobs.pop(job_id, None)
        self._top_up_ingest_parse_pool()

    def _claim_ingest_local_stt_job(
        self,
        job_id: str,
    ) -> LibraryIngestJob | None:
        """Publish a provisionally dispatched local-STT job as parsing."""

        current = self.library_ingest_jobs.get_job(job_id)
        if current is None:
            return None
        if current.state is IngestJobState.PARSING:
            return current
        if current.state is not IngestJobState.QUEUED:
            return None
        return self.library_ingest_jobs.mark_parsing(
            job_id,
            detected_type=current.detected_type,
        )

    def _on_ingest_local_stt_dispatch_failure(
        self,
        job_id: str,
        attempt_id: str,
        code: TranscriptionFailureCode,
        actions: tuple[str, ...],
        error_type: str,
    ) -> None:
        if self._ingest_shutdown or self._ingest_local_stt_jobs.get(job_id) != (
            0,
            attempt_id,
        ):
            return
        self._ingest_local_stt_jobs.pop(job_id, None)
        message = TRANSCRIPTION_FAILURE_CONTRACT[code][0]
        logger.error(
            "Library local STT dispatch failed "
            f"(job_id={job_id}, error_type={error_type})."
        )
        self.library_ingest_jobs.mark_failed(
            job_id,
            error=message,
            permanent=False,
            error_detail={
                "category": "stt_failure",
                "code": code.value,
                "message": message,
                "actions": list(actions),
            },
        )
        self._top_up_ingest_parse_pool()

    def _marshal_local_stt_callback(
        self,
        callback: Callable[..., Any],
        job_id: str,
        envelope: ExecutorEvent | ExecutorResult | ExecutorFailure,
    ) -> None:
        self._marshal_local_stt_call(callback, job_id, envelope)

    def _ingest_local_stt_event(self, job_id: str, event: ExecutorEvent) -> None:
        self._marshal_local_stt_callback(self._on_ingest_local_stt_event, job_id, event)

    def _ingest_local_stt_result(self, job_id: str, result: ExecutorResult) -> None:
        self._marshal_local_stt_callback(
            self._on_ingest_local_stt_result, job_id, result
        )

    def _ingest_local_stt_failure(
        self,
        job_id: str,
        failure: ExecutorFailure,
    ) -> None:
        self._marshal_local_stt_callback(
            self._on_ingest_local_stt_failure, job_id, failure
        )

    def _local_stt_callback_matches(
        self,
        job_id: str,
        envelope: ExecutorEvent | ExecutorResult | ExecutorFailure,
    ) -> bool:
        return self._ingest_local_stt_jobs.get(job_id) == (
            envelope.generation,
            envelope.attempt_id,
        )

    def _local_stt_terminal_matches(
        self,
        job_id: str,
        envelope: ExecutorResult | ExecutorFailure,
    ) -> bool:
        """Adopt the first controller-fenced generation before submit returns."""

        if self._local_stt_callback_matches(job_id, envelope):
            return True
        binding = self._ingest_local_stt_jobs.get(job_id)
        if binding == (0, envelope.attempt_id) and envelope.generation > 0:
            self._ingest_local_stt_jobs[job_id] = (
                envelope.generation,
                envelope.attempt_id,
            )
            return True
        return False

    def _on_ingest_local_stt_event(
        self,
        job_id: str,
        event: ExecutorEvent,
    ) -> None:
        if self._ingest_shutdown:
            return
        if not self._local_stt_callback_matches(job_id, event):
            binding = self._ingest_local_stt_jobs.get(job_id)
            if (
                binding is None
                or binding[1] != event.attempt_id
                or event.generation <= binding[0]
                or event.phase is not WorkerPhase.PREPARING
            ):
                return
            self._ingest_local_stt_jobs[job_id] = (
                event.generation,
                event.attempt_id,
            )
        if self._claim_ingest_local_stt_job(job_id) is None:
            return
        existing = self.library_ingest_jobs.get_job(job_id)
        progress: dict[str, Any] = {
            "phase": event.phase.value,
            "message": _INGEST_LOCAL_STT_PHASE_MESSAGES[event.phase],
        }
        if (
            existing is not None
            and (existing.progress or {}).get("cancel_requested") is True
        ):
            progress["cancel_requested"] = True
        self.library_ingest_jobs.update_progress(
            job_id,
            progress=progress,
            persist=False,
        )

    def _on_ingest_local_stt_result(
        self,
        job_id: str,
        result: ExecutorResult,
    ) -> None:
        if self._ingest_shutdown or not self._local_stt_terminal_matches(
            job_id, result
        ):
            return
        if self._claim_ingest_local_stt_job(job_id) is None:
            self._ingest_local_stt_jobs.pop(job_id, None)
            return
        self._ingest_local_stt_jobs.pop(job_id, None)
        self._ingest_parsed_payloads[job_id] = result.payload
        self._start_library_ingest_queue_if_idle()
        self._top_up_ingest_parse_pool()

    def _on_ingest_local_stt_failure(
        self,
        job_id: str,
        failure: ExecutorFailure,
    ) -> None:
        """Record an accepted worker failure and its safe diagnostic context.

        Args:
            job_id: Owning Library job identifier.
            failure: Generation-fenced, path-private worker failure.
        """
        if self._ingest_shutdown or not self._local_stt_terminal_matches(
            job_id, failure
        ):
            return
        current = self._claim_ingest_local_stt_job(job_id)
        if current is None:
            self._ingest_local_stt_jobs.pop(job_id, None)
            return
        self._ingest_local_stt_jobs.pop(job_id, None)
        message = TRANSCRIPTION_FAILURE_CONTRACT[failure.code][0]
        if failure.code is TranscriptionFailureCode.CANCELLED:
            self.library_ingest_jobs.mark_cancelled(job_id, reason=message)
        else:
            # Worker log sinks are silenced; this parent owns the application
            # log. Preserve correlation and stage without native exception text
            # or caller-controlled paths/progress messages.
            phase_value = (
                current.progress.get("phase")
                if isinstance(current.progress, dict)
                else None
            )
            try:
                phase = WorkerPhase(phase_value).value
            except (TypeError, ValueError):
                phase = "unknown"
            logger.error(
                "Library local STT failed "
                "(job_id={}, attempt_id={}, generation={}, phase={}, code={}).",
                job_id,
                failure.attempt_id,
                failure.generation,
                phase,
                failure.code.value,
            )
            self.library_ingest_jobs.mark_failed(
                job_id,
                error=message,
                permanent=False,
                error_detail={
                    "category": "stt_failure",
                    "code": failure.code.value,
                    "message": message,
                    "actions": list(failure.recovery_actions),
                },
                stt_failure_provenance=failure.failed_attempt,
            )
        executor = getattr(self, "_local_stt_executor", None)
        if executor is None or not executor.retiring:
            self._top_up_ingest_parse_pool()

    def _top_up_ingest_parse_pool(self) -> None:
        """Submit ``QUEUED`` jobs to the parse pool up to the worker cap.

        UI-thread only. Called after every submission/retry and after every
        parse completion (ok or not) so the pool stays saturated at up to N
        concurrent ``PARSING`` jobs -- this cap IS the backpressure: at most
        N parsed payloads (plus the one currently being written) are ever
        held in memory at once.

        A no-op once ``self._ingest_shutdown`` is set (the app is closing;
        no new parse work should be handed to a pool that's about to be
        terminated).

        ``classify_ingest_source`` is called once, at enqueue time (in
        ``submit_library_ingest_job``), and its result is stamped onto the
        job's ``detected_type`` -- not recomputed here. Dispatch reuses that
        stored value both to claim the job (``mark_parsing``) and to decide
        eligibility under the heavy-lane gate below; an unsupported
        extension at enqueue time is silently left ``""`` rather than
        fast-failing the job: real classification (permanent vs. retryable)
        happens inside the pool worker, where the authoritative exception
        is available (see ``classify_parse_failure``), matching the F3
        design spec's "permanent-vs-retryable classification happens inside
        the worker" decision.

        Heavy-lane gate: at most ``_ingest_heavy_lane_max_workers()`` jobs
        whose ``detected_type`` is in ``_INGEST_HEAVY_TYPES`` (audio/video
        transcription) may be ``PARSING`` at once, independent of the
        overall pool cap -- when that lane is full, ``next_queued`` is asked
        to skip those types so a queued document can fill the slot instead,
        letting document parses fan out wide while transcriptions stay
        capped. Ebook jobs use a separate one-process pool generation. A pool
        generation never mixes ebook and ordinary jobs, because sequential
        ebooks scheduled through a wider persistent pool can still rotate
        across workers and retain one high-water heap per process.
        """
        if self._ingest_shutdown or getattr(self, "_ingest_maintenance_paused", False):
            return
        if self._ingest_parse_pool_retirement_error:
            self._fail_queued_ingest_after_parse_pool_retirement()
            return
        if self._ingest_parse_pool_retiring:
            return
        heavy_cap = self._ingest_heavy_lane_max_workers()
        pending_research = getattr(
            self,
            "_research_source_parse_dispatch_pending",
            set(),
        )
        pending_research_jobs: dict[str, LibraryIngestJob] = {}
        for pending_job_id in tuple(pending_research):
            pending_job = self.library_ingest_jobs.get_job(pending_job_id)
            if (
                pending_job is None
                or pending_job.state is not IngestJobState.PARSING
                or pending_job.origin != "local"
            ):
                pending_research.discard(pending_job_id)
                continue
            pending_research_jobs[pending_job_id] = pending_job
        # Read the total + heavy in-flight counts ONCE, then include local-STT
        # jobs provisionally owned by an off-loop dispatch thread. Those rows
        # remain QUEUED until coordinator admission succeeds, but still consume
        # capacity; otherwise a later top-up could overfill the pool with light
        # work while identity resolution is in flight.
        parsing_count = max(
            0,
            self.library_ingest_jobs.counts().get("parsing", 0)
            - len(pending_research_jobs),
        )
        heavy_parsing_count = max(
            0,
            self.library_ingest_jobs.parsing_count_for_types(_INGEST_HEAVY_TYPES)
            - sum(
                job.detected_type in _INGEST_HEAVY_TYPES
                for job in pending_research_jobs.values()
            ),
        )
        ebook_parsing_count = max(
            0,
            self.library_ingest_jobs.parsing_count_for_types(_INGEST_EBOOK_TYPES)
            - sum(
                job.detected_type in _INGEST_EBOOK_TYPES
                for job in pending_research_jobs.values()
            ),
        )
        provisional_local_jobs = []
        for provisional_job_id in self._ingest_local_stt_jobs:
            provisional = self.library_ingest_jobs.get_job(provisional_job_id)
            if provisional is not None and provisional.state is IngestJobState.QUEUED:
                provisional_local_jobs.append(provisional)
        parsing_count += len(provisional_local_jobs)
        heavy_parsing_count += sum(
            job.detected_type in _INGEST_HEAVY_TYPES for job in provisional_local_jobs
        )
        while True:
            pool_mode = self._ingest_parse_pool_mode
            worker_count = (
                1
                if pool_mode == _INGEST_EBOOK_POOL_MODE
                else self._ingest_parse_worker_count()
            )
            # Local STT owns a separate executor. It still participates in the
            # ordinary global cap, but it must not consume the sole slot in an
            # ebook pool generation or keep that worker resident after its
            # ebook batch drains.
            capacity_count = (
                ebook_parsing_count
                if pool_mode == _INGEST_EBOOK_POOL_MODE
                else parsing_count
            )
            if capacity_count >= worker_count:
                return
            # LocalSTTExecutor intentionally accepts one request at a time.
            # A legacy heavy-lane override above one must not turn the next
            # queued audio/video job into a spurious ExecutorBusyError.
            local_stt_busy = bool(self._ingest_local_stt_jobs)
            coordinator = getattr(self, "_local_stt_dispatch_coordinator", None)
            dictation_reserved = bool(
                coordinator is not None and coordinator.dictation_reserved
            )
            heavy_full = (
                heavy_parsing_count >= heavy_cap or local_stt_busy or dictation_reserved
            )
            ebook_full = ebook_parsing_count >= 1
            skipped_types = (_INGEST_HEAVY_TYPES if heavy_full else frozenset()) | (
                _INGEST_EBOOK_TYPES if ebook_full else frozenset()
            )
            only_types = None
            if pool_mode == _INGEST_EBOOK_POOL_MODE:
                only_types = _INGEST_EBOOK_TYPES
            elif pool_mode == _INGEST_GENERAL_POOL_MODE:
                skipped_types |= _INGEST_EBOOK_TYPES
            preclaimed = False
            eligible_pending = (
                job
                for job in pending_research_jobs.values()
                if job.detected_type not in skipped_types
                and (only_types is None or job.detected_type in only_types)
            )
            job = min(
                eligible_pending,
                key=lambda item: item.submitted_at,
                default=None,
            )
            if job is not None:
                preclaimed = True
                pending_research_jobs.pop(job.job_id, None)
            else:
                job = self.library_ingest_jobs.next_queued(
                    skip_types=skipped_types,
                    only_types=only_types,
                )
            if job is None:
                if pool_mode is not None:
                    generation_jobs = self._ingest_parse_jobs_by_generation.get(
                        self._ingest_parse_pool_generation,
                        set(),
                    )
                    queued_ebook = self.library_ingest_jobs.next_queued(
                        only_types=_INGEST_EBOOK_TYPES
                    )
                    should_retire = (
                        pool_mode == _INGEST_EBOOK_POOL_MODE or queued_ebook is not None
                    )
                    if not generation_jobs and should_retire:
                        self._retire_idle_ingest_parse_pool()
                return
            try:
                options = self._ingest_job_options(job)
            except BatchSTTRoutingError as exc:
                error_text = _sanitize_library_ingest_error_text(str(exc))
                failure_text = error_text or "Batch transcription routing failed."
                logger.warning(
                    "Library ingest batch STT routing failed "
                    f"(job_id={job.job_id}, "
                    f"detected_type={job.detected_type}, "
                    f"error={failure_text})."
                )
                self.library_ingest_jobs.mark_failed(
                    job.job_id,
                    error=failure_text,
                    permanent=False,
                )
                if preclaimed:
                    pending_research.discard(job.job_id)
                continue
            except _template_resolution_errors() as exc:
                # (task 10, AC 37/AC-24b) A template choice that no longer
                # resolves (or a stored-invalid body) FAILS THIS ITEM with
                # the named error -- never a silent fallback to plain
                # chunking, which is how a library gets chunked two ways
                # without the user knowing. Not permanent: re-creating or
                # re-naming the template makes a retry succeed.
                failure_text = _sanitize_library_ingest_error_text(str(exc)) or (
                    "Chunking template resolution failed."
                )
                logger.warning(
                    "Library ingest template resolution failed "
                    f"(job_id={job.job_id}, "
                    f"detected_type={job.detected_type}, "
                    f"error={failure_text})."
                )
                self.library_ingest_jobs.mark_failed(
                    job.job_id,
                    error=failure_text,
                    permanent=False,
                )
                continue
            job_id = job.job_id
            source_path = job.source_path
            if options.get("transcription_provider") in {
                "parakeet-onnx",
                "transcribe-cpp",
            }:
                try:
                    self._submit_local_stt_job(job, options)
                except Exception as exc:
                    if preclaimed:
                        pending_research.discard(job_id)
                    code, recovery_actions = self._classify_local_stt_dispatch_error(
                        str(options.get("transcription_provider") or ""), exc
                    )
                    message = TRANSCRIPTION_FAILURE_CONTRACT[code][0]
                    logger.error(
                        "Library local STT dispatch failed "
                        f"(job_id={job_id}, provider="
                        f"{options.get('transcription_provider')}, "
                        f"error_type={type(exc).__name__})."
                    )
                    self.library_ingest_jobs.mark_failed(
                        job_id,
                        error=message,
                        permanent=False,
                        error_detail={
                            "category": "stt_failure",
                            "code": code.value,
                            "message": message,
                            "actions": list(recovery_actions),
                        },
                    )
                    continue
                if preclaimed:
                    pending_research.discard(job_id)
                parsing_count += 1
                if job.detected_type in _INGEST_HEAVY_TYPES:
                    heavy_parsing_count += 1
                continue
            claimed = (
                job
                if preclaimed
                else self.library_ingest_jobs.mark_parsing(
                    job.job_id, detected_type=job.detected_type
                )
            )
            if claimed is None:
                logger.error(
                    f"Library ingest coordinator: mark_parsing rejected "
                    f"job {job.job_id} (expected QUEUED) -- abandoning "
                    f"this top-up pass."
                )
                break
            parsing_count += 1
            if job.detected_type in _INGEST_HEAVY_TYPES:
                heavy_parsing_count += 1
            if job.detected_type in _INGEST_EBOOK_TYPES:
                ebook_parsing_count += 1
            try:
                mode = (
                    _INGEST_EBOOK_POOL_MODE
                    if job.detected_type in _INGEST_EBOOK_TYPES
                    else _INGEST_GENERAL_POOL_MODE
                )
                pool = self._ensure_ingest_parse_pool(mode)
            except Exception as exc:
                if preclaimed:
                    pending_research.discard(job_id)
                # CONTAINMENT (live-QA crash fix): pool CREATION itself
                # failed -- e.g. the spawn machinery raising at
                # construction time (the fileno-less-stderr resource-
                # tracker crash `_create_ingest_parse_pool` now works
                # around, or any environment-specific successor). This is
                # a UI-thread call reached synchronously from
                # submit/retry, so letting it propagate would crash the
                # app on the user's submission. Same containment
                # philosophy as `_handle_broken_ingest_parse_pool`, but
                # scoped to just the triggering job: no pool ever existed
                # here, so no OTHER job's parse was riding on it -- fail
                # this one retryable, keep the pool slot empty (the next
                # submit/retry attempts creation from scratch), and
                # return cleanly.
                logger.opt(exception=True).error(
                    f"Library ingest parse pool could not be created "
                    f"(job_id={job_id}, source={source_path})."
                )
                self._ingest_parse_pool = None
                # (task-32054) An unsupported file was never going to be
                # parsed: the worker would have recorded it SKIPPED. Losing
                # the pool must not convert the pre-flight's own "will
                # skip" forecast into a retryable failure carrying an
                # unrelated pool error.
                unsupported = self._unsupported_ingest_source_error(source_path)
                if unsupported is not None:
                    self.library_ingest_jobs.mark_skipped(
                        job_id,
                        reason=unsupported,
                        error_detail={
                            "category": "unsupported_file_type",
                            "message": unsupported,
                        },
                    )
                    return
                self.library_ingest_jobs.mark_failed(
                    job_id,
                    error=_sanitize_library_ingest_error_text(
                        f"Parse pool could not start: {exc}"
                    )
                    or "Parse pool could not start.",
                    permanent=False,
                )
                return
            generation = self._ingest_parse_pool_generation
            generation_jobs = self._ingest_parse_jobs_by_generation[generation]
            generation_jobs.add(job_id)
            try:
                pool.apply_async(
                    run_parse_job,
                    (source_path, options, (generation, job_id)),
                    callback=functools.partial(
                        self._ingest_pool_callback, generation, job_id
                    ),
                    error_callback=functools.partial(
                        self._ingest_pool_error_callback, generation, job_id
                    ),
                )
                if preclaimed:
                    pending_research.discard(job_id)
            except Exception as exc:
                if preclaimed:
                    pending_research.discard(job_id)
                # The pool itself rejected the submission synchronously
                # (e.g. it was already terminated/closed) -- every job
                # currently PARSING was submitted to this same broken pool
                # and can't be trusted to ever complete either.
                self._handle_broken_ingest_parse_pool(generation, job_id, exc)
                return

    @staticmethod
    def _unsupported_ingest_source_error(source_path: str) -> Optional[str]:
        """Return the unsupported-type reason for ``source_path``, if any.

        The parse worker is what normally classifies a source, so a pool
        that never starts leaves that verdict unmade. This asks the same
        classifier the worker would (task-32054) so a file the pre-flight
        forecast as "will skip" still records as a skip.

        Args:
            source_path: The queued source.

        Returns:
            The classifier's own message when the type is unsupported, or
            ``None`` for anything the pipeline would have attempted.
        """
        try:
            classify_ingest_source(source_path)
        except FileIngestionError as exc:
            message = str(exc).strip()
            if message.startswith("Unsupported file type"):
                return _sanitize_library_ingest_error_text(message) or message
        except Exception:
            return None
        return None

    def _retire_idle_ingest_parse_pool(self) -> None:
        """Release an empty pool generation, then resume queued work.

        Pool termination and joining stay off the UI thread. New submissions
        pause behind ``_ingest_parse_pool_retiring`` until teardown completes,
        preventing an ebook worker's retained heap from overlapping the next
        ordinary pool generation.
        """
        if self._ingest_parse_pool_retiring or self._ingest_parse_pool is None:
            return
        generation = self._ingest_parse_pool_generation
        generation_jobs = self._ingest_parse_jobs_by_generation.get(generation)
        if generation_jobs:
            return

        self._ingest_parse_jobs_by_generation.pop(generation, None)
        pool = self._ingest_parse_pool
        stop_event = self._ingest_parse_pool_stop_event
        progress_queue = self._ingest_parse_progress_queue
        progress_thread = self._ingest_parse_progress_thread
        if stop_event is not None:
            stop_event.set()
        self._ingest_parse_pool = None
        self._ingest_parse_pool_mode = None
        self._ingest_parse_pool_stop_event = None
        self._ingest_parse_progress_queue = None
        self._ingest_parse_progress_thread = None
        self._ingest_parse_pool_retiring = True

        self._terminate_ingest_parse_pool_off_thread(
            pool,
            progress_queue,
            progress_thread,
            on_complete=self._resume_ingest_after_parse_pool_retirement,
            on_failure=self._fail_ingest_after_parse_pool_retirement,
        )

    def _resume_ingest_after_parse_pool_retirement(self) -> None:
        """Resume on the UI loop, or just release the gate after loop exit."""
        if self._ingest_shutdown:
            return
        loop = getattr(self, "_loop", None)
        if loop is None or not loop.is_running():
            self._ingest_parse_pool_retiring = False
            return
        self._marshal_ingest_pool_call(self._on_ingest_parse_pool_retired)

    def _fail_ingest_after_parse_pool_retirement(
        self,
        _exc: BaseException,
    ) -> None:
        """Surface teardown failure without releasing the no-overlap gate."""
        if self._ingest_shutdown:
            return
        loop = getattr(self, "_loop", None)
        if loop is None or not loop.is_running():
            return
        self._marshal_ingest_pool_call(self._on_ingest_parse_pool_retirement_failed)

    def _on_ingest_parse_pool_retirement_failed(self) -> None:
        """Fail queued local work when old workers cannot be proven stopped."""
        if self._ingest_shutdown:
            return
        self._ingest_parse_pool_retirement_error = _INGEST_PARSE_POOL_RESTART_ERROR
        self._fail_queued_ingest_after_parse_pool_retirement()

    def _fail_queued_ingest_after_parse_pool_retirement(self) -> None:
        """Fail local jobs submitted after an unrecoverable pool teardown."""
        error = self._ingest_parse_pool_retirement_error
        if not error:
            return
        pending_research = getattr(
            self,
            "_research_source_parse_dispatch_pending",
            set(),
        )
        pending_job_ids = tuple(pending_research)
        pending_research.difference_update(pending_job_ids)
        for job_id in pending_job_ids:
            job = self.library_ingest_jobs.get_job(job_id)
            if (
                job is None
                or job.origin != "local"
                or job.state is not IngestJobState.PARSING
            ):
                continue
            self.library_ingest_jobs.mark_failed(
                job.job_id,
                error=error,
                permanent=False,
            )
        for job in self.library_ingest_jobs.jobs():
            if job.origin != "local" or job.state is not IngestJobState.QUEUED:
                continue
            self.library_ingest_jobs.mark_failed(
                job.job_id,
                error=error,
                permanent=False,
            )

    def _on_ingest_parse_pool_retired(self) -> None:
        """Finish one pool-mode transition on the UI thread."""
        if self._ingest_shutdown:
            return
        self._ingest_parse_pool_retirement_error = None
        self._ingest_parse_pool_retiring = False
        self._top_up_ingest_parse_pool()

    def _marshal_ingest_pool_call(
        self,
        callback: Callable[..., Any],
        *args: Any,
    ) -> None:
        """Marshal a pool callback, tolerating only shutdown cancellation."""

        if self._ingest_shutdown:
            return
        try:
            self.call_from_thread(callback, *args)
        except concurrent.futures.CancelledError:
            if not self._ingest_shutdown:
                raise

    def _ingest_pool_callback(
        self, generation: int, job_id: str, result: Dict[str, Any]
    ) -> None:
        """``apply_async`` ``callback``: runs on the pool's result-handler thread.

        Checks ``_ingest_shutdown`` BEFORE marshaling (quit-deadlock
        guard, Task 4 review): Textual's ``call_from_thread`` blocks the
        calling thread on the marshaled call's result and only guards
        against the loop being ``None``, not against it shutting down --
        and CPython's ``Pool._terminate_pool`` does an unbounded
        ``result_handler.join()``, with ``_handle_results`` able to run
        callbacks before it observes TERMINATE. So if a parse completed
        right as the user quit, this thread could park inside
        ``call_from_thread`` while the quit path parked waiting on THIS
        thread inside ``pool.terminate()`` -- mutual deadlock, app hangs
        on quit. Checking the flag here (on this thread, before any
        marshaling) narrows that window; running terminate/join off the
        loop thread entirely (``_shutdown_ingest_parse_pool``) closes it
        -- with both layers, a callback that slips past this check parks
        only until the still-free loop drains it (and the marshaled body
        then no-ops via the same flag inside
        ``_on_ingest_parse_complete``).

        Args:
            job_id: Bound at submission time via ``functools.partial`` in
                ``_top_up_ingest_parse_pool``.
            result: ``run_parse_job``'s structured return value.
        """
        self._marshal_ingest_pool_call(
            self._on_ingest_parse_complete, generation, job_id, result
        )

    def _ingest_pool_error_callback(
        self, generation: int, job_id: str, exc: BaseException
    ) -> None:
        """``apply_async`` ``error_callback``: same thread + shutdown
        contract as ``_ingest_pool_callback`` (see its docstring)."""
        self._marshal_ingest_pool_call(
            self._handle_broken_ingest_parse_pool, generation, job_id, exc
        )

    def _on_ingest_parse_progress_batch(
        self,
        generation: int,
        events: tuple[ParseProgressEvent, ...],
    ) -> None:
        """Apply one validated progress batch for the current parse generation.

        Progress and terminal results travel on separate channels, so this
        UI-thread boundary rechecks every piece of coordinator authority after
        IPC. Unknown or malformed queue data is ignored; local live telemetry
        is projected in memory only.
        """
        if self._ingest_shutdown or generation != self._ingest_parse_pool_generation:
            return
        generation_jobs = self._ingest_parse_jobs_by_generation.get(generation)
        if generation_jobs is None:
            return

        for raw_event in events:
            try:
                event = make_parse_progress_event(
                    raw_event.generation,
                    raw_event.job_id,
                    raw_event.phase,
                    raw_event.message,
                    raw_event.percent,
                )
            except Exception:
                continue
            if event is None:
                continue
            job = self.library_ingest_jobs.get_job(event.job_id)
            if (
                event.generation != generation
                or event.job_id not in generation_jobs
                or event.job_id in self._ingest_parsed_payloads
                or job is None
                or job.state is not IngestJobState.PARSING
            ):
                continue
            progress: dict[str, Any] = {
                "phase": event.phase,
                "message": event.message,
            }
            if event.percent is not None:
                progress["percent"] = event.percent
            self.library_ingest_jobs.update_progress(
                event.job_id,
                progress=progress,
                persist=False,
            )

    def _on_ingest_parse_complete(
        self, generation: int, job_id: str, result: Dict[str, Any]
    ) -> None:
        """Handle one pool completion (success or structured parse failure).

        UI-thread only; invoked via ``call_from_thread`` from the pool's
        result-handler thread (the ``apply_async`` ``callback``). No-ops
        immediately once ``self._ingest_shutdown`` is set -- a completion
        can still be marshaled onto the UI thread for a brief window after
        the app starts closing (it may have already been in flight when
        ``pool.terminate()`` was called), and this guard is what keeps that
        race from touching a closing app's registry/pool state.

        Args:
            job_id: The job this result belongs to (bound at submission
                time in ``_top_up_ingest_parse_pool``, not re-derived here).
            result: ``run_parse_job``'s structured return value -- either
                ``{"ok": True, "payload": {...}}`` or
                ``{"ok": False, "error": str, "permanent": bool}``.
        """
        if self._ingest_shutdown:
            return
        generation_jobs = self._ingest_parse_jobs_by_generation.get(generation)
        if (
            generation != self._ingest_parse_pool_generation
            or generation_jobs is None
            or job_id not in generation_jobs
        ):
            return
        generation_jobs.remove(job_id)
        if result.get("ok"):
            self._ingest_parsed_payloads[job_id] = result["payload"]
            self._start_library_ingest_queue_if_idle()
        else:
            error_text = _sanitize_library_ingest_error_text(
                str(result.get("error") or "Library import parsing failed.")
            )
            error_detail = result.get("error_detail")
            # (task-2220 owner ruling) An unsupported file was never
            # attempted -- it records as SKIPPED, a neutral terminal
            # outcome; "failed" is reserved for files the pipeline tried.
            if (
                isinstance(error_detail, dict)
                and error_detail.get("category") == "unsupported_file_type"
            ):
                self.library_ingest_jobs.mark_skipped(
                    job_id,
                    reason=error_text or "Unsupported file type.",
                    error_detail=error_detail,
                )
            else:
                self.library_ingest_jobs.mark_failed(
                    job_id,
                    error=error_text or "Library import parsing failed.",
                    permanent=bool(result.get("permanent", False)),
                    error_detail=error_detail,
                    stt_failure_provenance=result.get("stt_failure_provenance"),
                )
        self._top_up_ingest_parse_pool()

    def _handle_broken_ingest_parse_pool(
        self,
        generation: int,
        job_id: Optional[str],
        exc: BaseException,
    ) -> None:
        """Fail every still-mid-parse ``PARSING`` job and drop the broken pool.

        UI-thread only. Shared by the pool's ``error_callback`` (an async,
        pool-level failure marshaled via ``call_from_thread`` -- e.g. a
        worker process died) and a synchronous ``apply_async`` submission
        failure in ``_top_up_ingest_parse_pool`` (the pool was already
        broken when we tried to use it). Either way, a job whose parse is
        still genuinely in flight on the SAME pool object may never see
        its callback fire, so it can't be trusted to complete -- failing
        those (retryable) and dropping the pool reference is the only
        sound recovery (see the F3 design spec's "Worker-process death"
        section). The pool is rebuilt lazily by
        ``_create_ingest_parse_pool`` after the broken generation has fully
        terminated. Queued work resumes automatically from the retirement
        callback, and submissions remain gated until then so generations
        cannot overlap in memory.

        Payload-ready jobs are SPARED (Task 4 review fix): a job whose
        parse already completed sits ``PARSING`` with its payload in
        ``_ingest_parsed_payloads`` until the writer claims it -- it needs
        nothing further from the pool, so failing it here would throw a
        finished parse away just because an unrelated worker died. Such
        jobs are skipped (left ``PARSING`` for the writer), and the writer
        is woken at the end so they drain even if it had already released.

        No-ops once ``self._ingest_shutdown`` is set, same as
        ``_on_ingest_parse_complete``.
        """
        if self._ingest_shutdown:
            return
        generation_jobs = self._ingest_parse_jobs_by_generation.get(generation)
        if (
            generation != self._ingest_parse_pool_generation
            or generation_jobs is None
            or (job_id is not None and job_id not in generation_jobs)
        ):
            return

        affected_jobs = set(generation_jobs)
        self._ingest_parse_jobs_by_generation.pop(generation, None)
        pool = self._ingest_parse_pool
        stop_event = self._ingest_parse_pool_stop_event
        progress_queue = self._ingest_parse_progress_queue
        progress_thread = self._ingest_parse_progress_thread
        if stop_event is not None:
            stop_event.set()
        self._ingest_parse_pool_stop_event = None
        self._ingest_parse_pool = None
        self._ingest_parse_pool_mode = None
        self._ingest_parse_progress_queue = None
        self._ingest_parse_progress_thread = None
        self._ingest_parse_pool_retiring = True

        logger.opt(exception=exc).error(f"Library ingest parse pool failed: {exc}")
        for job in self.library_ingest_jobs.jobs():
            if job.job_id not in affected_jobs or job.state != IngestJobState.PARSING:
                continue
            if job.job_id in self._ingest_parsed_payloads:
                # Parse already finished -- the payload is waiting for the
                # writer; the broken pool can't hurt this job anymore.
                continue
            self.library_ingest_jobs.mark_failed(
                job.job_id,
                error="Library import parse pool failed unexpectedly; retry to resume.",
                permanent=False,
            )
        if self._ingest_parsed_payloads:
            self._start_library_ingest_queue_if_idle()

        self._terminate_ingest_parse_pool_off_thread(
            pool,
            progress_queue,
            progress_thread,
            on_complete=self._resume_ingest_after_parse_pool_retirement,
            on_failure=self._fail_ingest_after_parse_pool_retirement,
        )

    @staticmethod
    def _terminate_ingest_parse_pool_off_thread(
        pool: Any | None,
        progress_queue: Any | None = None,
        progress_thread: threading.Thread | None = None,
        *,
        on_complete: Callable[[], None] | None = None,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> threading.Thread | None:
        """Clean up one detached parse generation away from the UI thread."""
        return LibraryIngestQueueMixin._shutdown_ingest_workers_off_thread(
            None,
            None,
            None,
            pool,
            progress_queue,
            progress_thread,
            on_complete=on_complete,
            on_failure=on_failure,
        )

    def _shutdown_ingest_parse_pool(self) -> Optional[threading.Thread]:
        """Quit-path teardown: flag up, pool detached, terminate off-loop.

        Called from ``TldwCli.on_unmount`` (i.e. on the app's event-loop
        thread). Synchronously: sets ``_ingest_shutdown = True`` FIRST (so
        pool callbacks -- ``_ingest_pool_callback``/
        ``_ingest_pool_error_callback``, running on the pool's
        result-handler thread -- short-circuit before marshaling from this
        point on) and drops every worker reference (nothing can submit to
        them anymore). Source/coordinator/executor close, parse-pool
        terminate/join, queue cleanup, and bounded drain-thread join then run
        on detached daemon threads with a bounded pool-shutdown wait,
        NEVER on the caller's (loop) thread: verifier close may wait and
        CPython's ``Pool._terminate_pool`` does an unbounded
        ``result_handler.join()``, and if that result-handler thread is at
        that moment parked inside a ``call_from_thread`` it entered just
        before the flag went up, joining it from the loop thread would
        deadlock (the loop can't drain the marshaled call it is itself waiting
        behind). Off-loop, the loop stays free: the in-flight marshaled call
        runs, no-ops via the flag, the result-handler thread unblocks, and the
        join completes. The daemon thread is deliberately not joined by the
        caller -- worst case it outlives the app briefly and dies with the
        process.

        Returns:
            The one teardown thread that owns every detached ingest resource,
            or ``None`` when no ingest resource was ever created. The
            shutdown flag is still set in that case.
        """
        self._ingest_shutdown = True
        with self._local_stt_executor_lock:
            source_service = getattr(self, "_parakeet_source_service", None)
            source_listener = getattr(self, "_parakeet_source_registry_listener", None)
            coordinator = getattr(self, "_local_stt_dispatch_coordinator", None)
            executor = getattr(self, "_local_stt_executor", None)
            self._parakeet_source_service = None
            self._parakeet_source_registry_listener = None
            self._local_stt_dispatch_coordinator = None
            self._local_stt_executor = None
            if source_listener is not None:
                self.library_ingest_jobs.remove_listener(source_listener)
        local_jobs = getattr(self, "_ingest_local_stt_jobs", None)
        if local_jobs is None:
            self._ingest_local_stt_jobs = {}
        else:
            local_jobs.clear()
        pool = getattr(self, "_ingest_parse_pool", None)
        stop_event = getattr(self, "_ingest_parse_pool_stop_event", None)
        progress_queue = getattr(self, "_ingest_parse_progress_queue", None)
        progress_thread = getattr(self, "_ingest_parse_progress_thread", None)
        if stop_event is not None:
            stop_event.set()
        self._ingest_parse_pool_stop_event = None
        self._ingest_parse_pool = None
        self._ingest_parse_pool_mode = None
        self._ingest_parse_progress_queue = None
        self._ingest_parse_progress_thread = None
        if all(
            resource is None
            for resource in (
                source_service,
                coordinator,
                executor,
                pool,
                progress_queue,
                progress_thread,
            )
        ):
            return None
        return self._shutdown_ingest_workers_off_thread(
            source_service,
            coordinator,
            executor,
            pool,
            progress_queue,
            progress_thread,
        )

    @staticmethod
    def _shutdown_ingest_workers_off_thread(
        source_service: Any | None,
        coordinator: Any | None,
        executor: Any | None,
        pool: Any | None,
        progress_queue: Any | None,
        progress_thread: threading.Thread | None,
        *,
        on_complete: Callable[[], None] | None = None,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> threading.Thread | None:
        """Close detached ingest workers without blocking the UI thread.

        Executor shutdown remains ahead of parse-pool teardown. The parse pool
        gets a bounded terminate/join window before its queue is
        closed/cancelled, then the already-stopped daemon drain receives only a
        bounded join. A timeout reports failure once and never calls the later
        completion callback, so callers keep their no-overlap gate asserted.
        """

        def _shutdown_workers() -> None:
            pool_failure: BaseException | None = None
            if source_service is not None:
                try:
                    source_service.close()
                except Exception:
                    logger.error("Error closing the Parakeet source service.")
            if coordinator is not None:
                try:
                    coordinator.close()
                except Exception:
                    logger.error("Error closing the local STT dispatch coordinator.")
            if executor is not None:
                try:
                    executor.close()
                except Exception:
                    logger.opt(exception=True).error(
                        "Error closing the Library local STT executor."
                    )
            if pool is not None:
                pool_shutdown_done = threading.Event()
                pool_failures: list[BaseException] = []

                def _terminate_and_join_pool() -> None:
                    try:
                        pool.terminate()
                        pool.join()
                    except Exception as exc:
                        pool_failures.append(exc)
                    finally:
                        pool_shutdown_done.set()

                try:
                    pool_shutdown_thread = threading.Thread(
                        target=_terminate_and_join_pool,
                        name="library-ingest-parse-pool-shutdown",
                        daemon=True,
                    )
                    pool_shutdown_thread.start()
                except Exception as exc:
                    pool_failure = exc
                else:
                    if not pool_shutdown_done.wait(
                        timeout=_INGEST_WORKER_SHUTDOWN_TIMEOUT_SECONDS
                    ):
                        pool_failure = TimeoutError(
                            "Library ingest parse pool shutdown timed out."
                        )
                    elif pool_failures:
                        pool_failure = pool_failures[0]
                if pool_failure is not None:
                    logger.opt(exception=pool_failure).error(
                        "Error terminating the Library ingest parse pool."
                    )
            if progress_queue is not None:
                close = getattr(progress_queue, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        logger.error(
                            "Error cleaning up the Library ingest progress queue "
                            "(operation={}, queue_type={}).",
                            "close",
                            type(progress_queue).__name__,
                        )
                cancel_join = getattr(progress_queue, "cancel_join_thread", None)
                if cancel_join is not None:
                    try:
                        cancel_join()
                    except Exception:
                        logger.error(
                            "Error cleaning up the Library ingest progress queue "
                            "(operation={}, queue_type={}).",
                            "cancel_join_thread",
                            type(progress_queue).__name__,
                        )
            if progress_thread is not None:
                try:
                    progress_thread.join(timeout=1.0)
                except Exception:
                    logger.error(
                        "Error joining the Library ingest progress drain thread."
                    )
            if pool_failure is not None:
                if on_failure is not None:
                    try:
                        on_failure(pool_failure)
                    except Exception:
                        logger.opt(exception=True).error(
                            "Error reporting Library ingest pool retirement failure."
                        )
            elif on_complete is not None:
                try:
                    on_complete()
                except Exception:
                    logger.opt(exception=True).error(
                        "Error resuming Library ingest after pool retirement."
                    )

        try:
            thread = threading.Thread(
                target=_shutdown_workers,
                name="library-ingest-workers-shutdown",
                daemon=True,
            )
            thread.start()
        except Exception as exc:
            logger.opt(exception=True).error(
                "Could not start the Library ingest worker shutdown thread."
            )
            if on_failure is not None:
                try:
                    on_failure(exc)
                except Exception:
                    logger.opt(exception=True).error(
                        "Error reporting Library ingest pool retirement failure."
                    )
            return None
        return thread

    # -- Remote poller (server-origin jobs) --------------------------------

    #: Seconds between remote status polls. Server ingests are minutes-long
    #: (transcription, OCR), so a slow cadence is plenty and keeps this off the
    #: server's back.
    REMOTE_INGEST_POLL_SECONDS: float = 5.0

    #: Cap on status pages fetched per batch per pass, so a server
    #: reporting has_more forever cannot pin the loop.
    REMOTE_INGEST_MAX_PAGES: int = 20

    def _resolve_ingest_backend(self) -> str:
        """Return the backend a new ingest should run on: ``local`` or ``server``.

        Deliberately its **own** preference rather than the Media destination's
        browse scope. Reusing the browse scope looked tidier -- one notion of
        "which backend am I on" -- but it
        would mean a user who switched scope to look at server-side media and
        then imported a file had that file leave their machine without ever
        asking for it. ``build_library_ingest_state``'s own contract is explicit
        that ingest "always targets the local media store regardless of browsing
        scope", and quietly inverting that is not a change to make on the user's
        behalf.

        So sending an ingest to a server is an explicit opt-in, and anything
        unrecognised or unset means local -- the backend that always works and
        keeps the file where it already is.
        """
        raw = get_cli_setting("library.ingest", "backend", "local")
        if str(raw or "local").strip().lower() != "server":
            return "local"
        # The opt-in is necessary but not sufficient. Runtime policy declares
        # ``media.ingestion_jobs.launch.server`` as ``required_source="server"``,
        # so the service refuses the launch while the Library runtime is local.
        # Honouring that here means an opted-in user whose runtime is local gets
        # a local ingest -- the file stays put and the canvas explains how to
        # enable server imports -- rather than a job that fails with "requires
        # server mode" (seen live against a real server).
        runtime_state = getattr(getattr(self, "runtime_policy", None), "state", None)
        active_source = (
            str(getattr(runtime_state, "active_source", "local") or "local")
            .strip()
            .lower()
        )
        return "server" if active_source == "server" else "local"

    def _validate_research_source_operation_authority(
        self,
        operation_id: str,
        *,
        expected_origin: str,
    ) -> Any:
        """Recover and validate the durable qualified intake authority.

        This is called before queue admission and again by delayed Server
        dispatch workers.  The visible Research screen and the current Library
        origin are never accepted as substitutes for the persisted operation.
        """

        store = getattr(self, "research_source_operation_store", None)
        get_operation = getattr(store, "get", None)
        if not callable(get_operation):
            raise ValueError(
                "Durable Research source authority is unavailable; reopen Add Sources."
            )
        operation = get_operation(operation_id)
        operation_origin = str(
            getattr(getattr(operation, "data_source", None), "value", "") or ""
        )
        if operation is None or operation_origin != expected_origin:
            raise ValueError(
                "The intake no longer matches its captured Research workspace authority."
            )
        if expected_origin != "server":
            return operation

        context_provider = getattr(self, "server_context_provider", None)
        get_context = getattr(context_provider, "get_active_context", None)
        if not callable(get_context):
            raise ValueError(
                "The captured Server workspace authority is unavailable; restore it and retry."
            )

        context = get_context()
        profile_id = str(getattr(context, "active_server_id", "") or "").strip()
        principal_id = event_principal_id_from_active_context(context) or ""
        if profile_id != getattr(
            operation, "server_profile_id", ""
        ) or principal_id != getattr(operation, "principal_id", ""):
            raise ValueError(
                "The captured Server workspace authority changed; restore it and retry."
            )
        return operation

    def _submit_server_ingest_job(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any],
        title: str,
        author: str,
        keywords: tuple[str, ...],
        perform_analysis: bool,
        research_source_operation_id: str | None = None,
    ) -> LibraryIngestJob:
        """Queue a ``server``-origin job and send it to the server.

        The registry row is created synchronously so the queue shows the job the
        moment the user starts it, then an async worker performs the submission
        and records the ids the server issues. A source the server has no
        handler for fails immediately, with the reason, rather than being sent
        and rejected later.

        Returns:
            The queued job, or an already-``FAILED`` one when the source cannot
            be sent at all.
        """
        try:
            job = self._prepare_library_ingest_job_admitted(
                source_path=source_path,
                ingest_options=ingest_options,
                title=title,
                author=author,
                keywords=keywords,
                perform_analysis=perform_analysis,
                chunk_enabled=False,
                chunk_size=DEFAULT_CHUNK_SIZE,
                batch_id=None,
                backend="server",
                research_source_operation_id=research_source_operation_id,
                require_persisted=False,
            )
        except ServerIngestUnsupported as exc:
            job = self.library_ingest_jobs.submit(
                source_path=source_path,
                title=title,
                author=author,
                keywords=keywords,
                perform_analysis=perform_analysis,
                origin="server",
                ingest_options=ingest_options,
                research_source_operation_id=research_source_operation_id,
            )
            return (
                self.library_ingest_jobs.mark_failed(
                    job.job_id, error=str(exc), permanent=True
                )
                or job
            )
        self._dispatch_research_source_catalog_job(job.job_id)
        return job

    def _submit_web_clip_job(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any],
        title: str,
        author: str,
        keywords: tuple[str, ...],
        perform_analysis: bool,
        research_source_operation_id: str | None = None,
    ) -> LibraryIngestJob:
        """Queue a ``server``-origin job that clips a web page.

        A page cannot go through the ingest-jobs API -- it has no media type for
        one -- so this uses the clipper endpoint instead. That endpoint is
        synchronous and issues no job or batch id, so unlike a server file
        ingest there is nothing to attach or poll: the job settles when the call
        returns (task-684.3).

        Returns:
            The queued job, or an already-``FAILED`` one when the source cannot
            be clipped at all.
        """
        try:
            job = self._prepare_library_ingest_job_admitted(
                source_path=source_path,
                ingest_options=ingest_options,
                title=title,
                author=author,
                keywords=keywords,
                perform_analysis=perform_analysis,
                chunk_enabled=False,
                chunk_size=DEFAULT_CHUNK_SIZE,
                batch_id=None,
                backend="server",
                research_source_operation_id=research_source_operation_id,
                require_persisted=False,
            )
        except NotAWebClipSource as exc:
            job = self.library_ingest_jobs.submit(
                source_path=source_path,
                title=title,
                author=author,
                keywords=keywords,
                perform_analysis=perform_analysis,
                origin="server",
                ingest_options=ingest_options,
                research_source_operation_id=research_source_operation_id,
            )
            return (
                self.library_ingest_jobs.mark_failed(
                    job.job_id, error=str(exc), permanent=True
                )
                or job
            )
        self._dispatch_research_source_catalog_job(job.job_id)
        return job

    @work(group="library_ingest_remote_submit")
    async def _send_web_clip_job(self, job_id: str, kwargs: dict[str, Any]) -> None:
        """Clip a page on the server and settle the job on the answer.

        Shares the submit worker group with ``_send_server_ingest_job``: both are
        one-shot submissions on the user's behalf, and neither should be able to
        pile up.
        """
        job = self.library_ingest_jobs.get_job(job_id)
        if job is not None and job.research_source_operation_id:
            try:
                self._validate_research_source_operation_authority(
                    job.research_source_operation_id,
                    expected_origin="server",
                )
            except ValueError:
                self.library_ingest_jobs.mark_failed(
                    job_id,
                    error=(
                        "The captured Server workspace authority changed before "
                        "submission. Restore it and retry this intake."
                    ),
                )
                return
        service = getattr(self, "server_media_reading_service", None)
        clip = getattr(service, "ingest_web_content", None)
        if not callable(clip):
            self.library_ingest_jobs.mark_failed(
                job_id,
                error=(
                    "No server backend is configured, so this page cannot be "
                    "clipped on the server. Configure one in Settings, or switch "
                    "this Library to Local."
                ),
                permanent=True,
            )
            return

        self.library_ingest_jobs.mark_parsing(job_id, detected_type="web")
        try:
            response = await clip(**kwargs)
        except Exception as exc:
            logger.opt(exception=True).warning(f"Web clip failed for job {job_id}.")
            self.library_ingest_jobs.mark_failed(
                job_id, error=f"The server could not clip the page: {exc}"
            )
            return

        # A 200 is not a captured page: the endpoint reports its outcome in the
        # body, so an extraction that found nothing arrives as success.
        reason = clip_failure_reason(response)
        if reason is not None:
            self.library_ingest_jobs.mark_failed(job_id, error=reason)
            return

        # No media id comes back, so this finishes like a remote job: done, with
        # "Open in Library" withheld because the content is in the server's.
        self.library_ingest_jobs.mark_remote_done(job_id)

    @work(group="library_ingest_remote_submit")
    async def _send_server_ingest_job(
        self, job_id: str, kwargs: dict[str, Any]
    ) -> None:
        """Submit to the server, then attach the ids it issued.

        Async for the same reason the poller is: the service call is a
        coroutine, so staying on the event loop keeps every registry mutation
        on the UI thread without marshalling.
        """
        job = self.library_ingest_jobs.get_job(job_id)
        if job is not None and job.research_source_operation_id:
            try:
                self._validate_research_source_operation_authority(
                    job.research_source_operation_id,
                    expected_origin="server",
                )
            except ValueError:
                self.library_ingest_jobs.mark_failed(
                    job_id,
                    error=(
                        "The captured Server workspace authority changed before "
                        "submission. Restore it and retry this intake."
                    ),
                )
                return
        service = getattr(self, "server_media_reading_service", None)
        submit = getattr(service, "submit_ingest_jobs", None) or getattr(
            service, "submit_media_ingest_jobs", None
        )
        if not callable(submit):
            self.library_ingest_jobs.mark_failed(
                job_id,
                error=(
                    "No server backend is configured, so this import cannot run "
                    "on the server. Configure one in Settings, or switch this "
                    "Library to Local."
                ),
                permanent=True,
            )
            return

        try:
            response = await submit(**kwargs)
        except Exception as exc:
            logger.opt(exception=True).warning(
                f"Server ingest submission failed for job {job_id}."
            )
            self.library_ingest_jobs.mark_failed(
                job_id, error=f"The server refused the import: {exc}"
            )
            return

        batch_id = _response_field(response, "batch_id")
        jobs = _response_field(response, "jobs") or []
        remote_job_id = None
        if jobs:
            remote_job_id = _response_field(jobs[0], "id")
        self.library_ingest_jobs.attach_remote(
            job_id,
            remote_job_id=None if remote_job_id is None else str(remote_job_id),
            batch_id=None if batch_id is None else str(batch_id),
        )
        errors = _response_field(response, "errors") or []
        if errors and not jobs:
            self.library_ingest_jobs.mark_failed(
                job_id, error=f"The server rejected the import: {errors[0]}"
            )
            return

        # Following a remote job needs BOTH ids: ``pending_remote_batches``
        # decides what to poll from ``batch_id``, and the reconciler matches
        # statuses to jobs by ``remote_job_id``. Without the first, the job is
        # never polled; without the second, the batch is polled forever while no
        # status can ever be matched to it. Either way the row sits at "queued"
        # indefinitely -- the same never-resolves failure the mistyped ``result``
        # field caused, and not something a queue may do quietly.
        if not batch_id or remote_job_id is None:
            self.library_ingest_jobs.mark_failed(
                job_id,
                error=(
                    "The server accepted this import but did not return the ids "
                    "needed to track it, so its progress cannot be followed. It "
                    "may still be running on the server; check there before "
                    "importing again."
                ),
                permanent=True,
            )
            return

        self.poll_remote_ingest_jobs()

    async def _reconcile_remote_batch(self, service: Any, batch_id: str) -> None:
        """Fetch every page of ``batch_id``'s statuses and reconcile them.

        The server's list response is paginated (``has_more``/``next_offset``,
        per its OpenAPI schema). Reading only the first page would leave later
        jobs unreconciled -- and since they stay unsettled, the poller would
        keep re-fetching that batch forever, which is what the stop condition
        exists to prevent.

        A transient failure is logged and left for the next pass rather than
        killing the poller; the jobs stay visibly unfinished meanwhile, which is
        the recoverable direction.
        """
        lister = service.list_media_ingest_jobs
        supports_offset = _accepts_keyword(lister, "offset")

        offset = 0
        for _ in range(self.REMOTE_INGEST_MAX_PAGES):
            if self._ingest_shutdown:
                return
            try:
                response = (
                    await lister(batch_id, offset=offset)
                    if supports_offset
                    else await lister(batch_id)
                )
            except Exception:
                logger.opt(exception=True).debug(
                    f"Remote ingest poll failed for batch {batch_id!r}; "
                    "will retry on the next pass."
                )
                return

            self._reconcile_page(response)
            if not _response_field(response, "has_more"):
                return
            if not supports_offset:
                # Without an offset there is no way to ask for page two, and
                # re-asking would just re-read page one until the cap.
                logger.debug(
                    f"Batch {batch_id!r} has more statuses but "
                    f"{type(service).__name__}.list_media_ingest_jobs takes no "
                    "offset; reconciled the first page only."
                )
                return
            next_offset = _response_field(response, "next_offset")
            if next_offset is None or next_offset == offset:
                # Server says there is more but gives no way forward; stop
                # rather than spin on the same page.
                logger.debug(
                    f"Batch {batch_id!r} reports has_more with no usable "
                    "next_offset; stopping pagination."
                )
                return
            offset = int(next_offset)
        else:
            logger.warning(
                f"Batch {batch_id!r} exceeded {self.REMOTE_INGEST_MAX_PAGES} "
                "pages of statuses; the rest will be picked up next pass."
            )

    def _reconcile_page(self, response: Any) -> None:
        """Hand one page of statuses to the reconciler, if it has any."""
        statuses = _response_field(response, "jobs")
        if statuses:
            reconcile_remote_ingest_jobs(self.library_ingest_jobs, statuses)

    def cancel_remote_ingest_batch(self, batch_id: str) -> None:
        """Ask the server to cancel every job in ``batch_id``.

        UI-thread entry point. Deliberately does *not* mark the local jobs
        cancelled: the request is asynchronous and may be refused, so the queue
        must not claim an outcome the server has not confirmed. The poller
        records the real state when the server reports it -- which is also why
        polling is (re)started here, so a batch cancelled while nothing was
        being watched still gets its outcome.

        Args:
            batch_id: The server batch to cancel. An empty value is ignored, so
                a queue row that never received a batch id cannot send a cancel
                for every job on the server.
        """
        if not batch_id:
            return
        self._request_remote_ingest_cancel(batch_id)
        self.poll_remote_ingest_jobs()

    @work(group="library_ingest_remote_cancel")
    async def _request_remote_ingest_cancel(self, batch_id: str) -> None:
        """Send the cancel request. Async for the same reason the poller is."""
        service = getattr(self, "server_media_reading_service", None)
        cancel = getattr(service, "cancel_media_ingest_jobs_batch", None)
        if not callable(cancel):
            logger.debug(
                "Remote ingest cancel requested but no server seam is available."
            )
            return
        try:
            # Keyword-only on both the client and the service wrapper; a
            # positional call raises TypeError at runtime.
            await cancel(batch_id=batch_id)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to cancel remote ingest batch {batch_id!r}."
            )
            self.notify(
                "Could not reach the server to cancel that import.",
                severity="warning",
            )

    def poll_remote_ingest_jobs(self) -> None:
        """Start watching server-origin ingest jobs, if any are outstanding.

        Idempotent: the worker is ``exclusive`` within its own group, so calling
        this again while a poll loop is already running is a no-op rather than a
        second poller.
        """
        if getattr(self, "_ingest_maintenance_paused", False):
            return
        if not pending_remote_batches(self.library_ingest_jobs):
            return
        self._run_remote_ingest_poll()

    @work(exclusive=True, group="library_ingest_remote_poll")
    async def _run_remote_ingest_poll(self) -> None:
        """Poll server ingest batches until none are outstanding.

        Deliberately an **async** worker rather than a thread worker. The
        service calls are already coroutines, and running on the event loop
        means every registry mutation here is already on the UI thread -- so
        this needs no ``call_from_thread`` at all, and therefore cannot hit the
        quit-path deadlock documented on ``_ingest_pool_callback`` (that
        marshal blocks the calling thread and does not observe loop shutdown).
        The ``await`` points are network I/O and a sleep, neither of which
        blocks the loop.

        Exits when every server batch has settled, on shutdown, or when the
        server seam is unavailable -- never spins on an answer that cannot
        change.
        """
        service = getattr(self, "server_media_reading_service", None)
        if service is None:
            logger.debug("Remote ingest poll: no server media service; not polling.")
            return

        while not self._ingest_shutdown and not getattr(self, "_ingest_maintenance_paused", False):
            batches = pending_remote_batches(self.library_ingest_jobs)
            if not batches:
                return

            for batch_id in batches:
                if self._ingest_shutdown or getattr(self, "_ingest_maintenance_paused", False):
                    return
                await self._reconcile_remote_batch(service, batch_id)

            if self._ingest_shutdown or getattr(self, "_ingest_maintenance_paused", False):
                return
            await asyncio.sleep(self.REMOTE_INGEST_POLL_SECONDS)

    # -- Writer (claim-or-release loop, narrowed to the write stage) -------

    def _start_library_ingest_queue_if_idle(self) -> None:
        """Start the writer worker, unless one is already active.

        UI-thread only. Sets ``runner_active = True`` synchronously, before
        scheduling the worker, so a rapid double-wake can never
        double-start the writer.

        If scheduling the ``@work`` worker itself raises synchronously
        (e.g. the app isn't in a state that accepts new workers), the
        ``runner_active`` flag is rolled back to ``False`` before
        re-raising -- otherwise a scheduling failure here would leave the
        registry permanently believing a runner is active when none was
        ever started, silently stranding every future payload.
        """
        if self.library_ingest_jobs.runner_active:
            return
        self.library_ingest_jobs.runner_active = True
        try:
            self._run_library_ingest_queue()
        except Exception:
            self.library_ingest_jobs.runner_active = False
            raise

    def _claim_next_ingest_job_or_release(
        self,
    ) -> Optional[tuple[LibraryIngestJob, Dict[str, Any]]]:
        """Atomically claim the oldest payload-ready job, or release the writer.

        UI-thread only; must only ever be invoked via ``call_from_thread``
        from the writer worker thread (see ``_run_library_ingest_queue``),
        never called directly from that thread.

        "Payload-ready" means the job's parsed payload is sitting in
        ``self._ingest_parsed_payloads`` (stashed by
        ``_on_ingest_parse_complete`` on a successful parse) -- claiming
        means popping that payload out of the dict AND transitioning the
        job ``PARSING`` -> ``WRITING`` via ``mark_writing``, both inside
        this single call. Jobs are visited oldest-submission-first
        (``self.library_ingest_jobs.jobs()`` is newest-first; this walks it
        reversed) so writes happen in submission order among ready
        payloads, even though a small file may finish parsing before an
        older large one.

        A successful claim also tops up the parse pool
        (``_top_up_ingest_parse_pool``): a payload-ready job still counts
        against the ``PARSING`` cap until this call's ``mark_writing``
        transitions it out (there is no separate registry state for
        "parsed but not yet claimed" -- see ``IngestJobState``), so a
        completion's own top-up call (in ``_on_ingest_parse_complete``,
        which always runs *before* the writer gets around to claiming) can
        still see the cap as full. Topping up again here is what actually
        frees that slot for a still-``QUEUED`` job once the claim lands.

        Atomicity contract: this is a single, plain synchronous UI-thread
        call, so the "is there a payload-ready job?" check and the "clear
        ``runner_active``" decision happen in the same turn of the UI event
        loop with no ``await``/yield between them -- exactly the discipline
        the pre-F3 claim-or-release fix established (see the git history:
        the previous two-step implementation had a submission land in the
        gap between "check" and "clear ``runner_active``", stranding a job
        behind a stale ``runner_active`` flag). Do not reintroduce a
        two-``call_from_thread`` exit path.

        Returns:
            ``(job, payload)`` for the oldest payload-ready job, if one
            exists -- ``runner_active`` is left untouched (still ``True``)
            and the writer must keep looping. ``None`` when no job is
            payload-ready -- ``runner_active`` is cleared before returning,
            and the writer must exit.
        """
        if self._ingest_shutdown:
            self.library_ingest_jobs.runner_active = False
            return None
        for job in reversed(self.library_ingest_jobs.jobs()):
            payload = self._ingest_parsed_payloads.get(job.job_id)
            if payload is None:
                continue
            del self._ingest_parsed_payloads[job.job_id]
            claimed = self.library_ingest_jobs.mark_writing(job.job_id)
            if claimed is None:
                # Invariant violation (Task-3 reviewer's guard note): a
                # payload existed for a job that wasn't PARSING when we
                # tried to claim it -- should be impossible (a payload only
                # ever enters the dict from a PARSING-state parse
                # completion, and this is the only caller of
                # `mark_writing`), but if it ever happens, the orphaned
                # payload is discarded (already popped above) and we keep
                # looking rather than crashing the writer loop.
                logger.error(
                    f"Library ingest writer: mark_writing rejected job "
                    f"{job.job_id} despite a ready payload -- discarding "
                    f"the orphaned payload and skipping."
                )
                continue
            self._top_up_ingest_parse_pool()
            return claimed, payload
        self.library_ingest_jobs.runner_active = False
        return None

    def _release_ingest_runner_after_crash(self) -> None:
        """Safety-net cleanup for the writer's ``finally`` block.

        UI-thread only; invoked via ``call_from_thread`` from the writer
        worker's ``finally``, on every exit path (clean or not).

        On the normal, clean-exit path this is a no-op: the writer already
        exited because ``_claim_next_ingest_job_or_release`` returned
        ``None``, which already cleared ``runner_active``. It only does
        real work when the worker thread is unwinding from something that
        bypassed that atomic exit -- i.e. an exception escaped a job's own
        isolation (see ``_run_library_ingest_queue``) or the marshaled call
        itself raised. In that case: clear ``runner_active`` if it is still
        set, and, since the crash may have left one or more parsed payloads
        sitting unclaimed with nothing left to drain them, restart the
        writer when a payload is still waiting at that moment. Restarting
        here is safe: this method runs on the UI thread, and the dying
        worker thread is already unwinding and will not touch the registry
        again.
        """
        if self.library_ingest_jobs.runner_active:
            self.library_ingest_jobs.runner_active = False
        if self._ingest_parsed_payloads:
            self._start_library_ingest_queue_if_idle()

    @work(exclusive=True, thread=True, group="library_ingest_queue")
    def _run_library_ingest_queue(self) -> None:
        """Drain payload-ready Library ingest jobs on a background thread.

        This is the write stage only (F3): parsing already happened in the
        pool, and this worker's whole job is persisting an already-parsed
        payload via ``persist_parsed_media`` -- one ``add_media_with_keywords``
        call at a time, since SQLite has exactly one writer.

        Runs until no job is payload-ready, then clears ``runner_active``
        (via ``_claim_next_ingest_job_or_release``, atomically -- see that
        method's docstring) and exits -- a later parse completion wakes a
        fresh worker (``_on_ingest_parse_complete`` ->
        ``_start_library_ingest_queue_if_idle``). Every registry touch is
        marshaled onto the UI thread via ``call_from_thread`` because
        ``LibraryIngestJobRegistry`` does no internal locking (see its
        module docstring). A single job's write failure (DB error, ...) is
        caught locally and turned into a ``mark_failed`` transition; it
        never aborts the loop.

        The outer ``try/finally`` is a separate safety net for failures
        *outside* that per-job isolation -- e.g. the marshaled claim call
        itself raising (a genuinely unexpected/"catastrophic" failure, not
        a per-job write error). See ``_release_ingest_runner_after_crash``
        for why the crash-recovery callable is skipped on a clean exit.
        """
        clean_exit = False
        media_db = self.media_db
        writer_thread = threading.current_thread()
        with self._ingest_writer_threads_lock:
            self._ingest_writer_threads.add(writer_thread)
        try:
            while True:
                claim = self.call_from_thread(self._claim_next_ingest_job_or_release)
                if claim is None:
                    clean_exit = True
                    return
                job, payload = claim
                try:
                    generic_options = (job.ingest_options or {}).get("generic", {})
                    overwrite_existing = bool(
                        generic_options.get(
                            "overwrite_existing",
                            generic_option_default("overwrite_existing", False),
                        )
                        if isinstance(generic_options, dict)
                        else generic_option_default("overwrite_existing", False)
                    )
                    generate_embeddings = bool(
                        generic_options.get(
                            "generate_embeddings",
                            generic_option_default("generate_embeddings", True),
                        )
                        if isinstance(generic_options, dict)
                        else generic_option_default("generate_embeddings", True)
                    )
                    media_id, _media_uuid, _message = persist_parsed_media(
                        payload,
                        media_db,
                        overwrite_existing=overwrite_existing,
                        generate_embeddings=generate_embeddings,
                    )
                    # ``add_media_with_keywords`` returns ``media_id=None`` on
                    # exactly one success path: the duplicate skip ("already
                    # exists. Overwrite not enabled."). A same-path re-ingest
                    # resolves by canonical URL; a byte-identical file at a
                    # DIFFERENT path has a different URL, so fall back to the
                    # content hash -- otherwise the row is a done-without-
                    # media_id husk with no "Open in Library" and nothing
                    # telling the user the file was already there (task-2013).
                    # ``self.media_db`` is unreachable-``None`` here in
                    # practice (submit already fails the job before this point
                    # when it's absent), but the guard is cheap insurance
                    # against an ``AttributeError`` on a stale/racy reference.
                    was_duplicate = media_id is None
                    content_hash = payload.get("content_hash")
                    if media_id is None and media_db is not None:
                        existing = media_db.get_media_by_url(payload["url"])
                        if existing is None:
                            if content_hash is None and isinstance(
                                payload.get("content"), str
                            ):
                                # The parse payload carries no hash; the DB
                                # computes sha256(content) itself inside
                                # ``add_media_with_keywords``. Mirror that
                                # exact computation, but only on this
                                # duplicate-with-URL-miss path -- never on
                                # the plain success path, which runs on the
                                # single-SQLite-writer critical path and
                                # would pay a second O(n) pass per file.
                                content_hash = hashlib.sha256(
                                    payload["content"].encode()
                                ).hexdigest()
                            if content_hash:
                                try:
                                    existing = media_db.get_media_by_hash(
                                        content_hash
                                    )
                                except (
                                    MediaDatabaseError,
                                    MediaInputError,
                                ) as exc:
                                    # The media row exists (the DB deduped
                                    # against it), so a failed lookup must
                                    # not fail the job -- but a silent miss
                                    # leaves a DONE row with no media_id and
                                    # no diagnostic trail.
                                    logger.warning(
                                        "Library ingest duplicate-resolution "
                                        "hash lookup failed "
                                        f"(job_id={job.job_id}, "
                                        f"source={job.source_path}, "
                                        f"hash={content_hash[:12]}…): {exc}"
                                    )
                                    existing = None
                        if existing is not None:
                            media_id = existing.get("id")
                            if content_hash is None:
                                content_hash = existing.get("content_hash")
                    # (task-3301) Includes the "analysis skipped: ..."
                    # annotation when the payload carries a skip reason.
                    progress = _library_ingest_done_progress(
                        job.source_path,
                        was_duplicate=was_duplicate,
                        payload=payload,
                    )
                    self.call_from_thread(
                        self.library_ingest_jobs.mark_done,
                        job.job_id,
                        media_id=media_id,
                        progress=progress,
                        content_hash=content_hash,
                    )
                except Exception as exc:
                    # loguru's traceback capture is `.opt(exception=True)`,
                    # NOT the stdlib `exc_info=True` kwarg (a silent no-op
                    # under loguru) -- log the full traceback here before
                    # mark_failed so a debugging session isn't left with only
                    # the registry's sanitized, single-line error string.
                    logger.opt(exception=True).error(
                        f"Library ingest job failed during write "
                        f"(job_id={job.job_id}, source={job.source_path})."
                    )
                    self.call_from_thread(
                        self.library_ingest_jobs.mark_failed,
                        job.job_id,
                        error=_sanitize_library_ingest_error(exc),
                        permanent=classify_parse_failure(exc),
                        error_detail={
                            # (task-14821 / xhigh review round) The stage
                            # covers two different things: refusing an
                            # empty extraction (BEFORE any write) and a
                            # genuine database write failure. Stamping
                            # both "write_error" told users nothing was
                            # saved because of a write problem when there
                            # had been nothing to save -- and, since
                            # "write_error" is the one category that still
                            # earns the optimistic retry advisory, the
                            # blanket DEFAULT smuggled that advisory back
                            # in for every unclassified cause.
                            "category": _library_ingest_write_failure_category(exc),
                            "message": str(exc),
                            "exception_type": exc.__class__.__name__,
                        },
                    )
        finally:
            try:
                if type(media_db) is MediaDatabase and not media_db.is_memory_db:
                    media_db.close_connection()
            finally:
                try:
                    if not clean_exit:
                        self.call_from_thread(self._release_ingest_runner_after_crash)
                finally:
                    with self._ingest_writer_threads_lock:
                        self._ingest_writer_threads.discard(writer_thread)
