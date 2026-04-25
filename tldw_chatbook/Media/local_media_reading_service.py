"""Thin local media-reading service around the client media DB."""

from __future__ import annotations

import json
import uuid
from typing import Any, Mapping, Optional


class LocalMediaReadingService:
    """Thin wrapper around the local media DB methods used by the media seam."""

    _SUPPORTED_METADATA_FIELDS = {"title", "media_type", "author", "url", "keywords"}
    _SUPPORTED_INGESTION_SOURCE_TYPES = {"local_directory", "archive_snapshot", "git_repository"}
    _SUPPORTED_INGESTION_SINK_TYPES = {"media", "notes"}
    _SUPPORTED_INGESTION_POLICIES = {"canonical", "import_only"}

    def __init__(self, media_db: Any):
        self.media_db = media_db

    def _require_db(self) -> Any:
        if self.media_db is None:
            raise ValueError("Local media DB is required for local media operations.")
        return self.media_db

    def _coerce_media_id(self, media_id: Any) -> int:
        return int(media_id)

    def _unsupported_ingestion_sources(self) -> ValueError:
        return ValueError("Local ingestion sources are not available yet.")

    def _unsupported_ingestion_jobs(self) -> ValueError:
        return ValueError("Local ingestion jobs are not available through this scope yet.")

    def _normalize_media_id_filter(self, media_ids: Any) -> list[int]:
        normalized: list[int] = []
        for media_id in media_ids or []:
            normalized.append(self._coerce_media_id(media_id))
        return normalized

    def _enrich_with_read_it_later_state(self, row: Mapping[str, Any]) -> dict[str, Any]:
        enriched = dict(row)
        state = self._require_db().get_media_read_it_later_state(self._coerce_media_id(row["id"]))
        if state is None:
            return enriched
        enriched["is_read_it_later"] = state.get("is_read_it_later")
        enriched["saved_at"] = state.get("saved_at")
        enriched["read_it_later_saved_at"] = state.get("saved_at")
        return enriched

    def _enrich_rows_with_read_it_later_state(self, rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self._enrich_with_read_it_later_state(row) for row in rows]

    def search_media(
        self,
        *,
        query: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        **filters: Any,
    ) -> dict[str, Any]:
        db = self._require_db()

        caller_media_ids_filter = filters.get("media_ids_filter")
        media_ids_filter = self._normalize_media_id_filter(caller_media_ids_filter)
        if filters.get("read_it_later_only", False):
            saved_media_ids = self._normalize_media_id_filter(
                db.list_read_it_later_media_ids(
                    include_deleted=bool(filters.get("include_deleted", False)),
                    include_trash=bool(filters.get("include_trash", False)),
                )
            )
            if caller_media_ids_filter is not None:
                saved_id_set = set(saved_media_ids)
                media_ids_filter = [media_id for media_id in media_ids_filter if media_id in saved_id_set]
            else:
                media_ids_filter = saved_media_ids
            if not media_ids_filter:
                return {
                    "items": [],
                    "total": 0,
                    "offset": offset,
                    "limit": limit,
                }

        results_per_page = max(limit + offset, limit)
        rows, total = db.search_media_db(
            search_query=query,
            search_fields=filters.get("fields"),
            media_types=filters.get("media_types"),
            date_range=filters.get("date_range"),
            must_have_keywords=filters.get("must_have") or filters.get("must_have_keywords"),
            must_not_have_keywords=filters.get("must_not") or filters.get("must_not_have_keywords"),
            sort_by=filters.get("sort_by", "last_modified_desc"),
            media_ids_filter=media_ids_filter,
            page=1,
            results_per_page=results_per_page,
            include_trash=bool(filters.get("include_trash", False)),
            include_deleted=bool(filters.get("include_deleted", False)),
        )
        items = self._enrich_rows_with_read_it_later_state(list(rows)[offset:offset + limit])
        return {
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def get_media_detail(self, media_id: Any, *, include_deleted: bool = False, include_trash: bool = False) -> Any:
        db = self._require_db()
        detail = db.get_media_by_id(
            self._coerce_media_id(media_id),
            include_deleted=include_deleted,
            include_trash=include_trash,
        )
        return self._enrich_with_read_it_later_state(detail)

    def update_media_metadata(self, media_id: Any, **metadata: Any) -> Any:
        db = self._require_db()
        unsupported = sorted(
            key for key, value in metadata.items()
            if value is not None and key not in self._SUPPORTED_METADATA_FIELDS
        )
        if unsupported:
            unsupported_text = ", ".join(unsupported)
            raise ValueError(f"Unsupported local media metadata fields: {unsupported_text}")
        return db.update_media_metadata(self._coerce_media_id(media_id), **metadata)

    def delete_media(self, media_id: Any) -> Any:
        return self._require_db().soft_delete_media(self._coerce_media_id(media_id))

    def undelete_media(self, media_id: Any) -> Any:
        return self._require_db().undelete_media(self._coerce_media_id(media_id))

    def get_reading_progress(self, media_id: Any) -> Any:
        return self._require_db().get_reading_progress(self._coerce_media_id(media_id))

    def update_reading_progress(self, media_id: Any, progress_data: Mapping[str, Any]) -> Any:
        return self._require_db().upsert_reading_progress(self._coerce_media_id(media_id), dict(progress_data))

    def delete_reading_progress(self, media_id: Any) -> Any:
        return self._require_db().delete_reading_progress(self._coerce_media_id(media_id))

    def save_to_read_it_later(self, media_id: Any) -> Any:
        return self._require_db().save_media_to_read_it_later(self._coerce_media_id(media_id))

    def remove_from_read_it_later(self, media_id: Any) -> Any:
        db = self._require_db()
        normalized_media_id = self._coerce_media_id(media_id)
        current_time = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            conn.execute(
                """
                DELETE FROM MediaReadItLaterState
                WHERE media_id = ?
                """,
                (normalized_media_id,),
            )
        return {
            "media_id": normalized_media_id,
            "is_read_it_later": False,
            "saved_at": None,
            "updated_at": current_time,
        }

    def list_ingestion_sources(self) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        cursor = db.get_connection().execute(
            """
            SELECT * FROM local_ingestion_sources
            ORDER BY id DESC
            """
        )
        return [self._ingestion_source_row_to_dict(row) for row in cursor.fetchall()]

    def create_ingestion_source(
        self,
        *,
        source_type: str,
        sink_type: str,
        policy: str = "canonical",
        enabled: bool = True,
        schedule_enabled: bool = False,
        schedule: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        normalized_source_type = self._validate_ingestion_value(
            source_type,
            allowed=self._SUPPORTED_INGESTION_SOURCE_TYPES,
            label="source_type",
        )
        normalized_sink_type = self._validate_ingestion_value(
            sink_type,
            allowed=self._SUPPORTED_INGESTION_SINK_TYPES,
            label="sink_type",
        )
        normalized_policy = self._validate_ingestion_value(
            policy,
            allowed=self._SUPPORTED_INGESTION_POLICIES,
            label="policy",
        )
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO local_ingestion_sources (
                    source_type, sink_type, policy, enabled, schedule_enabled,
                    schedule_json, config_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_source_type,
                    normalized_sink_type,
                    normalized_policy,
                    1 if enabled else 0,
                    1 if schedule_enabled else 0,
                    self._json_dumps(schedule or {}),
                    self._json_dumps(config or {}),
                    now,
                    now,
                ),
            )
            source_id = cursor.lastrowid
        return self.get_ingestion_source(source_id)

    def get_ingestion_source(self, source_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        row = db.get_connection().execute(
            "SELECT * FROM local_ingestion_sources WHERE id = ?",
            (int(source_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Local ingestion source not found: {source_id}")
        return self._ingestion_source_row_to_dict(row)

    def patch_ingestion_source(self, source_id: Any, **changes: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        self.get_ingestion_source(source_id)
        field_map = {
            "source_type": "source_type",
            "sink_type": "sink_type",
            "policy": "policy",
            "enabled": "enabled",
            "schedule_enabled": "schedule_enabled",
            "schedule": "schedule_json",
            "config": "config_json",
        }
        updates: dict[str, Any] = {}
        for key, column in field_map.items():
            if key not in changes:
                continue
            value = changes[key]
            if key == "source_type":
                value = self._validate_ingestion_value(
                    value,
                    allowed=self._SUPPORTED_INGESTION_SOURCE_TYPES,
                    label="source_type",
                )
            elif key == "sink_type":
                value = self._validate_ingestion_value(
                    value,
                    allowed=self._SUPPORTED_INGESTION_SINK_TYPES,
                    label="sink_type",
                )
            elif key == "policy":
                value = self._validate_ingestion_value(
                    value,
                    allowed=self._SUPPORTED_INGESTION_POLICIES,
                    label="policy",
                )
            elif key in {"enabled", "schedule_enabled"}:
                value = 1 if bool(value) else 0
            elif key in {"schedule", "config"}:
                value = self._json_dumps(value or {})
            updates[column] = value
        if not updates:
            return self.get_ingestion_source(source_id)
        updates["updated_at"] = db._get_current_utc_timestamp_str()
        assignments = ", ".join(f"{column} = ?" for column in updates)
        values = list(updates.values()) + [int(source_id)]
        with db.transaction() as conn:
            conn.execute(f"UPDATE local_ingestion_sources SET {assignments} WHERE id = ?", values)
        return self.get_ingestion_source(source_id)

    def delete_ingestion_source(self, source_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        self.get_ingestion_source(source_id)
        with db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM local_ingestion_sources WHERE id = ?",
                (int(source_id),),
            )
        return {"deleted": cursor.rowcount > 0, "source_id": int(source_id)}

    def list_ingestion_source_items(self, source_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        self.get_ingestion_source(source_id)
        cursor = db.get_connection().execute(
            """
            SELECT * FROM local_ingestion_source_items
            WHERE source_id = ?
            ORDER BY id DESC
            """,
            (int(source_id),),
        )
        return [self._ingestion_source_item_row_to_dict(row) for row in cursor.fetchall()]

    def trigger_ingestion_source_sync(self, source_id: Any) -> Any:
        source = self.get_ingestion_source(source_id)
        job = self._create_ingest_job(
            job_type="ingestion_source_sync",
            media_type=str(source.get("sink_type") or "media"),
            source=str(source.get("config", {}).get("path") or source.get("config", {}).get("repo_url") or source_id),
            source_kind=str(source.get("source_type") or "ingestion_source"),
            source_id=int(source_id),
        )
        self._set_ingestion_source_active_job(source_id, job["id"], status="queued")
        return {"status": "queued", "source_id": int(source_id), "job_id": job["id"]}

    def upload_ingestion_source_archive(self, source_id: Any, archive_path: str) -> Any:
        source = self.get_ingestion_source(source_id)
        if str(source.get("source_type") or "") != "archive_snapshot":
            raise ValueError("Archive upload is only supported for archive_snapshot sources.")
        job = self._create_ingest_job(
            job_type="ingestion_source_archive",
            media_type=str(source.get("sink_type") or "media"),
            source=str(archive_path),
            source_kind="archive_snapshot",
            source_id=int(source_id),
        )
        self._set_ingestion_source_active_job(source_id, job["id"], status="queued")
        return {
            "status": "queued",
            "source_id": int(source_id),
            "job_id": job["id"],
            "snapshot_status": "staged",
        }

    def submit_ingest_jobs(self, **kwargs: Any) -> Any:
        media_type = str(kwargs.get("media_type") or "").strip()
        if not media_type:
            raise ValueError("media_type is required for local ingest jobs.")
        batch_id = self._new_batch_id()
        jobs: list[dict[str, Any]] = []
        for url in kwargs.get("urls") or []:
            jobs.append(
                self._create_ingest_job(
                    batch_id=batch_id,
                    job_type="media_ingest",
                    media_type=media_type,
                    source=str(url),
                    source_kind="url",
                    options=kwargs,
                )
            )
        for file_path in kwargs.get("file_paths") or []:
            jobs.append(
                self._create_ingest_job(
                    batch_id=batch_id,
                    job_type="media_ingest",
                    media_type=media_type,
                    source=str(file_path),
                    source_kind="file",
                    options=kwargs,
                )
            )
        if not jobs:
            raise ValueError("At least one URL or file path is required for local ingest jobs.")
        return {"batch_id": batch_id, "jobs": jobs, "errors": []}

    def get_ingest_job(self, job_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        row = db.get_connection().execute(
            "SELECT * FROM local_ingestion_jobs WHERE id = ?",
            (int(job_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Local ingestion job not found: {job_id}")
        return self._ingest_job_row_to_dict(row)

    def list_ingest_jobs(self, batch_id: str, *, limit: int = 100) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        rows = db.get_connection().execute(
            """
            SELECT * FROM local_ingestion_jobs
            WHERE batch_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (batch_id, int(limit)),
        ).fetchall()
        return {"batch_id": batch_id, "jobs": [self._ingest_job_row_to_dict(row) for row in rows]}

    def stream_ingest_job_events(self, *, batch_id: str | None = None, after_id: int = 0) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        params: list[Any] = [int(after_id)]
        where = "id > ?"
        if batch_id is not None:
            where += " AND batch_id = ?"
            params.append(batch_id)
        rows = db.get_connection().execute(
            f"""
            SELECT * FROM local_ingestion_jobs
            WHERE {where}
            ORDER BY id ASC
            """,
            params,
        ).fetchall()
        return [
            {
                "event": "status",
                "data": self._ingest_job_row_to_dict(row),
            }
            for row in rows
        ]

    def cancel_ingest_job(self, job_id: Any, *, reason: str | None = None) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE local_ingestion_jobs
                SET status = ?, cancelled_at = ?, completed_at = ?, cancellation_reason = ?,
                    progress_percent = ?, progress_message = ?
                WHERE id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                """,
                ("cancelled", now, now, reason, 0, "Cancelled", int(job_id)),
            )
            if cursor.rowcount == 0:
                existing = self.get_ingest_job(job_id)
                return {
                    "success": existing.get("status") == "cancelled",
                    "job_id": int(job_id),
                    "status": existing.get("status"),
                    "message": "Job is already terminal.",
                }
        return {"success": True, "job_id": int(job_id), "status": "cancelled", "message": reason}

    def cancel_ingest_batch(
        self,
        *,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> Any:
        resolved_batch_id = batch_id or session_id
        if not resolved_batch_id:
            raise ValueError("batch_id or session_id is required.")
        current = self.list_ingest_jobs(str(resolved_batch_id), limit=1000)["jobs"]
        requested = len(current)
        cancelled = 0
        already_terminal = 0
        for job in current:
            if job.get("status") in {"completed", "failed", "cancelled"}:
                already_terminal += 1
                continue
            result = self.cancel_ingest_job(job["id"], reason=reason)
            if result.get("success"):
                cancelled += 1
        return {
            "success": True,
            "batch_id": str(resolved_batch_id),
            "requested": requested,
            "cancelled": cancelled,
            "already_terminal": already_terminal,
            "failed": 0,
        }

    def reprocess_media(self, media_id: Any, **options: Any) -> Any:
        job = self._create_ingest_job(
            job_type="media_reprocess",
            media_type="media",
            source=str(media_id),
            source_kind="media",
            options={"media_id": self._coerce_media_id(media_id), **options},
        )
        return {"status": "queued", "media_id": self._coerce_media_id(media_id), "job_id": job["id"]}

    def list_document_versions(self, media_id: Any, *, include_deleted: bool = False) -> Any:
        return self._require_db().get_all_document_versions(
            self._coerce_media_id(media_id),
            include_content=True,
            include_deleted=include_deleted,
        )

    def save_analysis_version(
        self,
        media_id: Any,
        *,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
    ) -> Any:
        return self._require_db().create_document_version(
            self._coerce_media_id(media_id),
            content=content,
            prompt=prompt,
            analysis_content=analysis_content,
        )

    def overwrite_analysis_version(
        self,
        media_id: Any,
        *,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
    ) -> Any:
        return self.save_analysis_version(
            media_id,
            content=content,
            analysis_content=analysis_content,
            prompt=prompt,
        )

    def delete_analysis_version(self, version_uuid: str) -> Any:
        return self._require_db().soft_delete_document_version(version_uuid)

    @staticmethod
    def _json_dumps(value: Mapping[str, Any] | list[Any] | None) -> str:
        return json.dumps(value or {})

    @staticmethod
    def _json_loads(value: Any) -> Any:
        if value in (None, ""):
            return {}
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _new_batch_id() -> str:
        return f"local-batch-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _validate_ingestion_value(value: Any, *, allowed: set[str], label: str) -> str:
        normalized = str(value or "").strip()
        if normalized not in allowed:
            raise ValueError(f"Unsupported local ingestion {label}: {normalized}")
        return normalized

    @staticmethod
    def _ensure_local_ingestion_schema(db: Any) -> None:
        with db.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS local_ingestion_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    sink_type TEXT NOT NULL,
                    policy TEXT NOT NULL DEFAULT 'canonical',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    schedule_enabled INTEGER NOT NULL DEFAULT 0,
                    schedule_json TEXT NOT NULL DEFAULT '{}',
                    config_json TEXT NOT NULL DEFAULT '{}',
                    active_job_id TEXT,
                    last_successful_snapshot_id INTEGER,
                    last_sync_started_at TEXT,
                    last_sync_completed_at TEXT,
                    last_sync_status TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_ingestion_source_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    normalized_relative_path TEXT NOT NULL,
                    content_hash TEXT,
                    sync_status TEXT NOT NULL DEFAULT 'pending',
                    binding_json TEXT NOT NULL DEFAULT '{}',
                    present_in_source INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES local_ingestion_sources(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS local_ingestion_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT,
                    batch_id TEXT NOT NULL,
                    source_id INTEGER,
                    job_type TEXT NOT NULL,
                    media_type TEXT,
                    source TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    keywords_json TEXT NOT NULL DEFAULT '[]',
                    options_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    cancelled_at TEXT,
                    cancellation_reason TEXT,
                    progress_percent REAL,
                    progress_message TEXT,
                    result_json TEXT,
                    error_message TEXT,
                    FOREIGN KEY (source_id) REFERENCES local_ingestion_sources(id) ON DELETE SET NULL
                );
                """
            )

    def _ingestion_source_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "user_id": 0,
            "source_type": payload.get("source_type"),
            "sink_type": payload.get("sink_type"),
            "policy": payload.get("policy"),
            "enabled": bool(payload.get("enabled", True)),
            "schedule_enabled": bool(payload.get("schedule_enabled", False)),
            "schedule_config": self._json_loads(payload.get("schedule_json")),
            "config": self._json_loads(payload.get("config_json")),
            "active_job_id": payload.get("active_job_id"),
            "last_successful_snapshot_id": payload.get("last_successful_snapshot_id"),
            "last_sync_started_at": payload.get("last_sync_started_at"),
            "last_sync_completed_at": payload.get("last_sync_completed_at"),
            "last_sync_status": payload.get("last_sync_status"),
            "last_error": payload.get("last_error"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    def _ingestion_source_item_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "source_id": payload.get("source_id"),
            "normalized_relative_path": payload.get("normalized_relative_path"),
            "content_hash": payload.get("content_hash"),
            "sync_status": payload.get("sync_status"),
            "binding": self._json_loads(payload.get("binding_json")),
            "present_in_source": bool(payload.get("present_in_source", True)),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    def _ingest_job_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "uuid": payload.get("uuid"),
            "batch_id": payload.get("batch_id"),
            "source_id": payload.get("source_id"),
            "job_type": payload.get("job_type"),
            "media_type": payload.get("media_type"),
            "source": payload.get("source"),
            "source_kind": payload.get("source_kind"),
            "status": payload.get("status"),
            "created_at": payload.get("created_at"),
            "started_at": payload.get("started_at"),
            "completed_at": payload.get("completed_at"),
            "cancelled_at": payload.get("cancelled_at"),
            "cancellation_reason": payload.get("cancellation_reason"),
            "progress_percent": payload.get("progress_percent"),
            "progress_message": payload.get("progress_message"),
            "result": self._json_loads(payload.get("result_json")),
            "error_message": payload.get("error_message"),
        }

    def _create_ingest_job(
        self,
        *,
        job_type: str,
        media_type: str,
        source: str,
        source_kind: str,
        batch_id: str | None = None,
        source_id: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        now = db._get_current_utc_timestamp_str()
        resolved_batch_id = batch_id or self._new_batch_id()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO local_ingestion_jobs (
                    uuid, batch_id, source_id, job_type, media_type, source, source_kind,
                    status, keywords_json, options_json, created_at, progress_percent, progress_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    resolved_batch_id,
                    source_id,
                    job_type,
                    media_type,
                    source,
                    source_kind,
                    "queued",
                    self._json_dumps(list((options or {}).get("keywords") or [])),
                    self._json_dumps(dict(options or {})),
                    now,
                    0,
                    "Queued",
                ),
            )
            job_id = cursor.lastrowid
        return self.get_ingest_job(job_id)

    def _set_ingestion_source_active_job(self, source_id: Any, job_id: Any, *, status: str) -> None:
        db = self._require_db()
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            conn.execute(
                """
                UPDATE local_ingestion_sources
                SET active_job_id = ?, last_sync_started_at = ?, last_sync_status = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(job_id), now, status, now, int(source_id)),
            )
