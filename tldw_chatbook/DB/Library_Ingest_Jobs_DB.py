"""SQLite persistence for the Library ingest job registry.

Single-user, UI-thread-only: keeps ONE persistent WAL connection reused across
all reads/writes (safe because every registry mutation runs on the UI thread),
rather than opening/closing per operation.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Union

from loguru import logger

from .base_db import BaseDB


class LibraryIngestJobsDB(BaseDB):
    _CURRENT_SCHEMA_VERSION = 1

    def __init__(self, db_path: Union[str, Path], client_id: str = "default") -> None:
        self._conn: sqlite3.Connection | None = None
        super().__init__(db_path, client_id)  # calls _initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path_str, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                logger.opt(exception=True).debug("LibraryIngestJobsDB: close failed")
            finally:
                self._conn = None

    def _initialize_schema(self) -> None:
        conn = self._get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY NOT NULL);
            INSERT OR IGNORE INTO schema_version (version) VALUES (1);

            CREATE TABLE IF NOT EXISTS ingest_jobs (
                seq INTEGER PRIMARY KEY,
                job_id TEXT UNIQUE NOT NULL,
                source_path TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT '',
                keywords TEXT NOT NULL DEFAULT '[]',
                perform_analysis INTEGER NOT NULL DEFAULT 0,
                chunk_enabled INTEGER NOT NULL DEFAULT 0,
                chunk_size INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL CHECK (state IN ('queued','parsing','writing','done','failed')),
                retry_count INTEGER NOT NULL DEFAULT 0,
                detected_type TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                finished_at_wall TEXT NOT NULL DEFAULT '',
                media_id INTEGER,
                superseded INTEGER NOT NULL DEFAULT 0,
                dismissed INTEGER NOT NULL DEFAULT 0,
                permanent INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.commit()

    @staticmethod
    def _seq_of(job_id: str) -> int:
        # "ingest-job-{n}" -> n
        return int(job_id.rsplit("-", 1)[-1])

    def upsert_job(self, job) -> None:
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO ingest_jobs
              (seq, job_id, source_path, title, author, keywords, perform_analysis,
               chunk_enabled, chunk_size, state, retry_count, detected_type, error,
               finished_at_wall, media_id, superseded, dismissed, permanent)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(job_id) DO UPDATE SET
              source_path=excluded.source_path, title=excluded.title, author=excluded.author,
              keywords=excluded.keywords, perform_analysis=excluded.perform_analysis,
              chunk_enabled=excluded.chunk_enabled, chunk_size=excluded.chunk_size,
              state=excluded.state, retry_count=excluded.retry_count,
              detected_type=excluded.detected_type, error=excluded.error,
              finished_at_wall=excluded.finished_at_wall, media_id=excluded.media_id,
              superseded=excluded.superseded, dismissed=excluded.dismissed, permanent=excluded.permanent
            """,
            (
                self._seq_of(job.job_id), job.job_id, job.source_path, job.title, job.author,
                json.dumps(list(job.keywords)), int(job.perform_analysis), int(job.chunk_enabled),
                job.chunk_size, job.state.value, job.retry_count, job.detected_type, job.error,
                job.finished_at_wall, job.media_id, int(job.superseded), int(job.dismissed),
                int(job.permanent),
            ),
        )
        conn.commit()

    def delete_job(self, job_id: str) -> None:
        conn = self._get_connection()
        conn.execute("DELETE FROM ingest_jobs WHERE job_id = ?", (job_id,))
        conn.commit()

    def all_jobs(self) -> list[dict]:
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM ingest_jobs ORDER BY seq ASC").fetchall()
        return [dict(r) for r in rows]
