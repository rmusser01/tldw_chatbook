"""TASK-21238: a corrupt notifications store must not fail media ingest jobs.

The client notifications DB went lazy in TASK-21105, so a corrupt or
unopenable store surfaces at DISPATCH time -- after the ingest job's own DB
write already succeeded. These tests pin that the dispatch guard skips the
notification instead of failing the flow (same shape as the study/quiz/evals
trio, ``Study_Interop/local_quiz_service.py``).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any

from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService


class ExplodingDispatcher:
    """Stands in for a dispatch service backed by a corrupt store."""

    def dispatch(self, **kwargs: Any) -> Any:
        raise sqlite3.DatabaseError("file is not a database")


class StubIngestDb:
    """Minimal media-DB double: records the terminal UPDATE, returns the job."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.job: dict[str, Any] = {
            "id": 7,
            "status": "queued",
            "progress_message": None,
        }

    @staticmethod
    def _get_current_utc_timestamp_str() -> str:
        return "2026-10-11T00:00:00Z"

    @contextmanager
    def transaction(self):
        yield self

    def execute(self, query: str, params: tuple = ()) -> None:
        self.executed.append((query, params))
        if "UPDATE local_ingestion_jobs" in query:
            self.job["status"] = params[0]


def _service_with_corrupt_notification_store(db: Any) -> LocalMediaReadingService:
    service = LocalMediaReadingService(
        db,
        notification_dispatcher=ExplodingDispatcher(),
        notification_app=object(),
    )
    # The read-back is not under test (it would need a real media DB); pin it
    # to the stub's copy so the flow reaches the dispatch call.
    service.get_ingest_job = lambda job_id: dict(db.job)  # type: ignore[method-assign]
    return service


def test_corrupt_notification_store_does_not_fail_terminal_dispatch():
    service = LocalMediaReadingService(
        None, notification_dispatcher=ExplodingDispatcher(), notification_app=object()
    )
    # Must not raise: the guard skips the notification.
    service._dispatch_terminal_ingest_job_notification(
        {"id": 7, "status": "completed", "progress_message": "Ingested 3 items"}
    )


def test_corrupt_notification_store_does_not_fail_job_completion():
    db = StubIngestDb()
    service = _service_with_corrupt_notification_store(db)

    job = service._complete_ingest_job(
        7,
        status="completed",
        progress_percent=100,
        progress_message="Ingested 3 items",
        result={"items": 3},
    )

    assert job["status"] == "completed"
    assert db.job["status"] == "completed"
