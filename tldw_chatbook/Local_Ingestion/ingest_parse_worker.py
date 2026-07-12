# tldw_chatbook/Local_Ingestion/ingest_parse_worker.py
"""Spawn-safe process-pool entry point for the Library ingest parse stage (F3).

This module is the pool's target module: ``multiprocessing.get_context("spawn")``
re-imports it fresh in every worker process, so **module scope here must
import stdlib only**. Every actually-heavy import (``local_file_ingestion``
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
        "chunk_options": dict | None,
        "metadata": dict | None,
    }

The Library ingest queue coordinator (F3 Task 4) builds this dict
mechanically from a ``LibraryIngestJob``'s fields -- ``title``, ``author``,
``keywords``, ``perform_analysis``, ``chunk_enabled``, ``chunk_size`` -- the
same one-step translation ``app.py``'s (pre-F3-pool) queue-runner already
performs for ``ingest_local_file``:

    options = {
        "title": job.title or None,
        "author": job.author or None,
        "keywords": list(job.keywords) or None,
        "perform_analysis": job.perform_analysis,
        "chunk_options": (
            {"method": "sentences", "size": job.chunk_size, "overlap": 100}
            if job.chunk_enabled
            else None
        ),
    }

(``custom_prompt``/``system_prompt``/``api_name``/``api_key``/``metadata``
have no ``LibraryIngestJob`` counterpart -- the Library queue never sets
them, so they're simply absent/``None``; they exist in the schema only
because ``ingest_local_file``'s direct programmatic callers --
``batch_ingest_files``, ``quick_ingest``, the server ingest path -- still
use them.)

Every value in ``options`` (and everything in the payload
``parse_local_file_for_ingest`` returns) must be plain, picklable data --
this dict, and the structured result below, cross the process boundary as
``apply_async`` arguments/return values.
"""

from __future__ import annotations

from typing import Any, Dict


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
        ``True`` for a missing-file failure (``FileNotFoundError``, raised
        by ``parse_local_file_for_ingest``/``ingest_local_file`` when the
        source path doesn't exist) or an unsupported-file-type failure
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


def run_parse_job(file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
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

    Returns:
        ``{"ok": True, "payload": <dict>}`` on success, where ``payload``
        is exactly what ``parse_local_file_for_ingest`` returned (consumed
        by ``persist_parsed_media`` on the writer thread). On failure,
        ``{"ok": False, "error": <str>, "permanent": <bool>}``, where
        ``error`` is ``str(exc)`` (or the exception's class name if that's
        empty) and ``permanent`` is ``classify_parse_failure(exc)``.
    """
    try:
        # Deferred import: keeps this module's own import stdlib-only (see
        # module docstring) so a freshly spawned worker process doesn't pay
        # for local_file_ingestion's parse-chain imports just to register
        # this function as the pool's target.
        from .local_file_ingestion import parse_local_file_for_ingest

        payload = parse_local_file_for_ingest(file_path, options)
    except Exception as exc:  # noqa: BLE001 - must never raise across the process boundary
        message = str(exc).strip() or exc.__class__.__name__
        return {
            "ok": False,
            "error": message,
            "permanent": classify_parse_failure(exc),
        }
    return {"ok": True, "payload": payload}
