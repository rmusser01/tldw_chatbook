# tldw_chatbook/Local_Ingestion/ingest_parse_worker.py
"""Spawn-safe process-pool entry point for the Library ingest parse stage (F3).

This module is the pool's target module: ``multiprocessing.get_context("spawn")``
re-imports it fresh in every worker process, so **module scope here must
import stdlib only, plus the stdlib-only progress contract**. Every actually-heavy import (``local_file_ingestion``
and everything it pulls in for a given file type -- PDF/docling, ebook,
audio/video transcription, LLM analysis, ...) is deferred into
``run_parse_job``'s function body. Importing this module must never pull in
``local_file_ingestion`` as a side effect (regression-guarded by
``Tests/Local_Ingestion/test_ingest_parse_worker.py`` via a subprocess
``sys.modules`` check) -- that keeps pool worker startup light (see
``Tests/Local_Ingestion/test_ingest_import_weight.py`` for the underlying
import-chain budget this relies on).

``options`` dict schema
------------------------
``run_parse_job`` forwards ``options`` unchanged to
``parse_local_file_for_ingest``. It mirrors ``ingest_local_file``'s
keyword arguments (minus ``file_path``/``media_db``, which never cross the
process boundary -- workers never touch the media DB):

    {
        "title": str | None,
        "author": str | None,
        "keywords": list[str] | None,
        "custom_prompt": str | None,
        "system_prompt": str | None,
        "perform_analysis": bool,
        "api_name": str | None,
        "api_key": str | None,
        "analysis_keyless_ok": bool,
        "analysis_call": dict | None,
        "chunk_options": dict | None,
        "metadata": dict | None,
        "encoding": str | None,
        "analysis_skipped_reason": str | None,
    }

The Library ingest queue coordinator (F3 Task 4) builds this dict from a
``LibraryIngestJob``'s fields via ``app.py``'s ``_ingest_job_options``.
(task-3301) Since the ingest-controls wiring, that builder also resolves
the live option values the schema above carries:

* ``chunk_options`` is ``None`` when Chunk content is OFF (the parse
  pipeline treats ``None`` as "do not chunk" for every type) and
  ``{"size": N, "max_size": N, "overlap": M}`` (ints) when ON;
* ``encoding`` carries the generic Encoding selection for the
  plaintext/html readers;
* ``api_name``/``api_key`` are present when analysis was requested AND
  the configured ``[analysis_defaults]`` provider resolves as ready
  (``Library/ingest_analysis.py``); otherwise
  ``analysis_skipped_reason`` says why analysis will not run, and the
  writer surfaces it on the done row. (task-3301 xhigh review round)
  ``api_name`` is the NORMALIZED chat dispatch name (an
  ``API_CALL_HANDLERS`` key); ``analysis_keyless_ok`` is the explicit
  opt-in that lets a keyless-READY provider analyze without a
  credential (direct callers that never set it keep the historical
  no-key => silent-skip contract); ``analysis_call`` carries the full
  ``[analysis_defaults]`` call shape
  (model/temperature/top_p/min_p/max_tokens) for viewer-parity
  analysis. A failed analysis travels back as the payload's
  ``analysis_failed_reason`` (plus a warning) and annotates the done
  row as "analysis failed: ...".

(``custom_prompt``/``system_prompt``/``metadata`` have no
``LibraryIngestJob`` counterpart -- the Library queue never sets them, so
they're simply absent/``None``; they exist in the schema only because
``ingest_local_file``'s direct programmatic callers --
``batch_ingest_files``, ``quick_ingest``, the server ingest path -- still
use them.)

Every value in ``options`` (and everything in the payload
``parse_local_file_for_ingest`` returns) must be plain, picklable data --
this dict, and the structured result below, cross the process boundary as
``apply_async`` arguments/return values.
"""

from __future__ import annotations

from typing import Any, Dict

from .ingest_parse_progress import emit_parse_progress, install_parse_progress_sink


def silence_ingest_worker_import_noise() -> None:
    """Pool ``initializer``: keep worker import noise off the parent's TTY.

    (task-2016) Spawn workers inherit a REAL-TTY stderr: the parent
    constructs the pool under ``redirect_stderr(sys.__stderr__)`` when
    Textual's fd-less stderr wrapper is active (see
    ``_create_ingest_parse_pool``), so anything a worker's lazy imports
    emit -- loguru's default stderr sink ("python-frontmatter not
    installed…"), ``RequestsDependencyWarning`` and friends -- paints raw
    text over the running TUI on the first submit. Runs INSIDE each worker
    process before any job. Deliberately does NOT touch ``sys.stderr``
    itself: a hard worker crash should still be able to reach a terminal
    for diagnosis; only the known import-noise channels are silenced.
    """
    import logging
    import warnings

    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
    except Exception:
        pass
    # Route ``warnings.warn`` through logging (which has no configured
    # handlers in a worker) instead of straight to stderr.
    logging.captureWarnings(True)
    warnings.simplefilter("ignore")
    # (task-2041) A handler-less root logger auto-basicConfigs a stderr
    # StreamHandler on the first bare ``logging.warning()`` -- the
    # "WARNING:root:…" flood channel. A NullHandler keeps root non-empty
    # so neither auto-basicConfig nor lastResort fires.
    logging.getLogger().addHandler(logging.NullHandler())


def initialize_ingest_parse_worker(progress_queue: Any | None = None) -> None:
    """Initialize import-noise handling and the worker-local progress sink."""
    silence_ingest_worker_import_noise()
    install_parse_progress_sink(progress_queue)


#: Max underlying exception-chain messages captured for the UI's
#: expanded failure details (task-2130).
_ERROR_CHAIN_CAP = 3


def classify_parse_failure(exc: Exception) -> bool:
    """Return whether an ingest-time exception is a *permanent* failure.

    (F1b M4, relocated for F3) A permanent (validation-class) failure -- a
    missing source file or an unsupported file type -- fails the exact same
    way on every retry, since the file at that path never changes shape on
    its own; offering Retry for one is dead bait. Every other exception (a
    transient I/O hiccup, a DB error, a corrupt/unparseable file, ...) stays
    retryable, since the same job genuinely might succeed on a later
    attempt.

    This is the single source of truth for that classification -- used
    both by ``run_parse_job`` (for parse-stage failures, inside the worker
    process) and by ``app.py``'s queue-runner (for write-stage failures,
    on the writer thread).

    Args:
        exc: The exception raised by the per-job ingest attempt.

    Returns:
        ``True`` for a ``PermanentIngestError`` (the explicit permanent
        marker raised by the URL/web extractor for a bad URL, a 4xx, a
        non-HTML page, empty extraction, or a missing extractor dependency),
        a missing-file failure (``FileNotFoundError``, raised by
        ``parse_local_file_for_ingest``/``ingest_local_file`` when the
        source path doesn't exist), or an unsupported-file-type failure
        (``detect_file_type`` raises ``FileIngestionError`` with a message
        starting "Unsupported file type" -- matched by message prefix
        rather than exception type, so a differently-raised validation
        error carrying the same copy still classifies consistently).
        ``False`` for everything else.
    """
    try:
        from .local_file_ingestion import PermanentIngestError

        if isinstance(exc, PermanentIngestError):
            return True
    except Exception:
        pass
    if isinstance(exc, FileNotFoundError):
        return True
    return str(exc).strip().startswith("Unsupported file type")


def run_parse_job(
    file_path: str,
    options: Dict[str, Any],
    progress_context: tuple[int, str] | None = None,
) -> Dict[str, Any]:
    """Pool entry point: parse one file into a picklable, structured result.

    Top-level and spawn-safe -- this is the exact callable submitted to a
    ``multiprocessing.get_context("spawn").Pool`` via ``apply_async``. It
    never raises across the process boundary: every exception raised while
    parsing (including a missing file or an unsupported extension) is
    caught here and turned into a structured failure result instead, since
    an exception raised inside a pool worker would otherwise need to survive
    unpickling on the parent side -- a surprise-prone path this avoids
    entirely.

    Args:
        file_path: Path to the file to parse.
        options: See the module docstring for the schema.
        progress_context: Optional ``(generation, job_id)`` identity bound by
            the parent process for best-effort progress telemetry.

    Returns:
        ``{"ok": True, "payload": <dict>}`` on success, where ``payload``
        is exactly what ``parse_local_file_for_ingest`` returned (consumed
        by ``persist_parsed_media`` on the writer thread). On failure,
        ``{"ok": False, "error": <str>, "permanent": <bool>}``, where
        ``error`` is ``str(exc)`` (or the exception's class name if that's
        empty) and ``permanent`` is ``classify_parse_failure(exc)``.
    """
    try:
        if progress_context is not None:
            generation, job_id = progress_context

            def progress_callback(phase, message, percent=None):
                emit_parse_progress(
                    generation,
                    job_id,
                    phase,
                    message,
                    percent,
                )

        else:
            progress_callback = None

        # Deferred import: keeps this module's own import stdlib-only (see
        # module docstring) so a freshly spawned worker process doesn't pay
        # for local_file_ingestion's parse-chain imports just to register
        # this function as the pool's target.
        from .local_file_ingestion import parse_local_file_for_ingest

        if progress_callback is None:
            payload = parse_local_file_for_ingest(file_path, options)
        else:
            payload = parse_local_file_for_ingest(
                file_path,
                options,
                progress_callback=progress_callback,
            )
    except Exception as exc:  # noqa: BLE001 - must never raise across the process boundary
        message = str(exc).strip() or exc.__class__.__name__
        permanent = classify_parse_failure(exc)
        stt_error_detail = getattr(exc, "error_detail", None)
        stt_failure_provenance = getattr(exc, "stt_failure_provenance", None)
        category = (
            "unsupported_file_type"
            if str(exc).strip().startswith("Unsupported file type")
            else "missing_source"
            if isinstance(exc, FileNotFoundError)
            else "parse_error"
        )
        # (task-2130) The generic wrapper message ("PDF Extraction Error.")
        # often hides the actual failure in the exception chain -- capture
        # up to three distinct underlying messages so the UI's expanded
        # details can say more than the one-line summary.
        chain: list[str] = []
        seen = {message}
        visited_ids: set[int] = set()
        cause = exc.__cause__ or exc.__context__
        # Visited-identity guard (Qodo round): __cause__/__context__ can
        # form a cycle, and repeated messages keep len(chain) flat -- a
        # message-only loop condition could spin forever.
        while (
            cause is not None
            and len(chain) < _ERROR_CHAIN_CAP
            and id(cause) not in visited_ids
        ):
            visited_ids.add(id(cause))
            text = str(cause).strip() or cause.__class__.__name__
            if text not in seen:
                chain.append(f"{cause.__class__.__name__}: {text}")
                seen.add(text)
            cause = cause.__cause__ or cause.__context__
        failure = {
            "ok": False,
            "error": message,
            "permanent": permanent,
            "error_detail": stt_error_detail
            if isinstance(stt_error_detail, dict)
            else {
                "category": category,
                "message": message,
                "exception_type": exc.__class__.__name__,
                "chain": chain,
            },
        }
        if isinstance(stt_failure_provenance, dict):
            failure["stt_failure_provenance"] = stt_failure_provenance
        return failure
    return {"ok": True, "payload": payload}
