"""Session-local owner for Library Re-chunk work and feedback (ADR-164)."""

from __future__ import annotations

import asyncio

from loguru import logger
from textual.app import App
from textual.signal import Signal

from ...Library.library_rechunk_service import (
    RECHUNK_SLOT,
    RECHUNK_WORKER_GROUP,
    acquire_bulk_rag_slot,
    format_rechunk_summary,
    release_bulk_rag_slot,
)


class LibraryRechunkRun:
    """Own one app session's current Re-chunk operation and latest receipt."""

    def __init__(self, app: App) -> None:
        self.app = app
        self.running = False
        self.summary = ""
        self.changed: Signal[LibraryRechunkRun] = Signal(app, "library-rechunk")

    def start(self) -> None:
        """Admit a local run using the existing shared backfill exclusion."""
        refusal = acquire_bulk_rag_slot(RECHUNK_SLOT)
        if refusal is not None:
            self.app.notify(refusal, severity="warning")
            return
        self.running = True
        self.summary = "Re-chunking…"
        self.changed.publish(self)
        try:
            # App ownership survives panel and screen replacement. Refuse
            # overlaps via slots; exclusive=True would cancel existing work.
            self.app.run_worker(
                self._run,
                thread=True,
                group=RECHUNK_WORKER_GROUP,
                exclusive=False,
            )
        except Exception as exc:  # noqa: BLE001 - release admission on scheduling failure
            logger.error(f"Legacy re-chunk worker could not start: {exc}")
            self._finish("", f"Re-chunk could not start: {exc}", "error")

    def _run(self) -> None:
        from ...RAG_Search.ingestion_indexing import (
            get_shared_rag_service,
            semantic_indexing_available,
        )
        from ...runtime_policy.types import PolicyDeniedError

        line = ""
        severity = "error"
        try:
            scope = getattr(self.app, "rag_admin_scope_service", None)
            launch = getattr(scope, "rechunk_legacy_media", None)
            if not callable(launch):
                notice = (
                    "Re-chunk could not start: the RAG admin service is "
                    "unavailable right now."
                )
            else:
                # Resolve the shared service before entering a transient loop.
                rag_service = (
                    get_shared_rag_service() if semantic_indexing_available() else None
                )
                outcome = asyncio.run(launch(mode="local", rag_service=rag_service))
                line = format_rechunk_summary(outcome)
                notice = f"Re-chunk finished: {line}"
                severity = "information"
        except PolicyDeniedError as denied:
            notice = f"Re-chunk was blocked by policy: {denied.user_message}"
        except Exception as exc:  # noqa: BLE001 - surface backend failure and allow retry
            logger.error(f"Legacy re-chunk worker crashed: {exc}")
            notice = f"Re-chunk failed: {exc}"
        try:
            self.app.call_from_thread(self._finish, line, notice, severity)
        except RuntimeError:
            # The app has stopped; no panel may be updated, but exclusion
            # must not remain held after the thread actually finishes.
            release_bulk_rag_slot(RECHUNK_SLOT)

    def _finish(self, line: str, notice: str, severity: str) -> None:
        """Publish completion and release admission together on the UI thread."""
        self.running = False
        self.summary = line
        release_bulk_rag_slot(RECHUNK_SLOT)
        self.changed.publish(self)
        self.app.notify(notice, severity=severity)


def get_library_rechunk_run(app: App) -> LibraryRechunkRun:
    """Return the app's ephemeral run owner without constructing data services."""
    run = getattr(app, "_library_rechunk_run", None)
    if run is None:
        run = LibraryRechunkRun(app)
        app._library_rechunk_run = run
    return run
