"""Local watchlists adapter backed by the existing subscriptions database."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from ..DB.Subscriptions_DB import SubscriptionsDB
from .watchlist_normalizers import (
    normalize_local_subscription_row,
    normalize_watchlist_alert_rule,
    normalize_watchlist_run,
)


_ALERT_CONDITION_TYPES = frozenset(
    {
        "no_items",
        "error_rate_above",
        "items_below",
        "items_above",
        "run_failed",
    }
)


class LocalWatchlistsService:
    """Thin adapter over `SubscriptionsDB` for the shared watchlists seam."""

    def __init__(self, *, db_factory: Callable[[], SubscriptionsDB]):
        self.db_factory = db_factory

    def _db(self) -> SubscriptionsDB:
        return self.db_factory()

    async def list_sources(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self._db().get_all_subscriptions(include_inactive=True, limit=limit, offset=offset)
        return [normalize_local_subscription_row(row) for row in rows]

    async def get_source(self, source_id: Any) -> dict[str, Any]:
        row = self._db().get_subscription(int(source_id))
        if row is None:
            raise KeyError(f"Subscription not found: {source_id}")
        return normalize_local_subscription_row(row)

    async def create_source(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        db = self._db()
        local_type = self._local_type_for_source_type(payload.get("source_type"))
        source_id = db.add_subscription(
            name=str(payload.get("name") or "Untitled subscription"),
            type=local_type,
            source=str(payload.get("url") or payload.get("source") or ""),
            tags=list(payload.get("tags") or []),
            description=payload.get("description"),
        )
        return normalize_local_subscription_row(db.get_subscription(source_id))

    async def update_source(self, source_id: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
        db = self._db()
        changes: dict[str, Any] = {}
        if "name" in payload:
            changes["name"] = payload["name"]
        if "url" in payload:
            changes["source"] = payload["url"]
        if "tags" in payload:
            changes["tags"] = payload["tags"]
        if "active" in payload:
            changes["is_active"] = bool(payload["active"])
        if "description" in payload:
            changes["description"] = payload["description"]
        if "source_type" in payload:
            changes["type"] = self._local_type_for_source_type(payload["source_type"])
        if changes:
            db.update_subscription(int(source_id), **changes)
        return normalize_local_subscription_row(db.get_subscription(int(source_id)))

    async def delete_source(self, source_id: Any) -> dict[str, Any]:
        success = self._db().delete_subscription(int(source_id))
        return {
            "success": success,
            "id": f"local:subscription:{source_id}",
            "backend": "local",
            "entity_kind": "subscription",
            "source_id": int(source_id),
        }

    async def launch_run(self, *, source_id: Any = None, job_id: Any = None) -> dict[str, Any]:
        resolved_source_id = int(source_id if source_id is not None else job_id)
        db = self._db()
        if db.get_subscription(resolved_source_id) is None:
            raise KeyError(f"Subscription not found: {resolved_source_id}")
        self._ensure_run_schema(db)
        now = self._utc_now()
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO local_watchlist_runs (
                    source_id, job_id, status, stats_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_source_id,
                    resolved_source_id,
                    "queued",
                    json.dumps({"source_id": resolved_source_id}),
                    now,
                    now,
                ),
            )
            run_id = cursor.lastrowid
        return await self.get_run(run_id)

    async def list_runs(
        self,
        *,
        source_id: Any = None,
        job_id: Any = None,
        limit: int = 100,
        offset: int = 0,
        **_: Any,
    ) -> list[dict[str, Any]]:
        db = self._db()
        self._ensure_run_schema(db)
        filters: list[str] = []
        values: list[Any] = []
        resolved_source_id = source_id if source_id is not None else job_id
        if resolved_source_id is not None:
            filters.append("source_id = ?")
            values.append(int(resolved_source_id))
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        values.extend([int(limit), int(offset)])
        cursor = db.conn.cursor()
        cursor.execute(
            f"""
            SELECT * FROM local_watchlist_runs
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            values,
        )
        return [normalize_watchlist_run("local", self._run_row_to_dict(row)) for row in cursor.fetchall()]

    async def get_run(self, run_id: Any) -> dict[str, Any]:
        db = self._db()
        self._ensure_run_schema(db)
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM local_watchlist_runs WHERE id = ?", (int(run_id),))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Watchlist run not found: {run_id}")
        return normalize_watchlist_run("local", self._run_row_to_dict(row))

    async def get_run_detail(self, run_id: Any, **_: Any) -> dict[str, Any]:
        return await self.get_run(run_id)

    async def cancel_run(self, run_id: Any) -> dict[str, Any]:
        db = self._db()
        self._ensure_run_schema(db)
        now = self._utc_now()
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE local_watchlist_runs
                SET status = ?, finished_at = ?, updated_at = ?
                WHERE id = ?
                """,
                ("cancelled", now, now, int(run_id)),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Watchlist run not found: {run_id}")
        return await self.get_run(run_id)

    async def list_alert_rules(self, *, job_id: Any = None, source_id: Any = None) -> list[dict[str, Any]]:
        db = self._db()
        self._ensure_alert_rule_schema(db)
        resolved_job_id = job_id if job_id is not None else source_id
        cursor = db.conn.cursor()
        if resolved_job_id is None:
            cursor.execute("SELECT * FROM local_watchlist_alert_rules ORDER BY created_at DESC")
        else:
            cursor.execute(
                """
                SELECT * FROM local_watchlist_alert_rules
                WHERE job_id = ? OR job_id IS NULL
                ORDER BY created_at DESC
                """,
                (int(resolved_job_id),),
            )
        return [normalize_watchlist_alert_rule("local", self._alert_rule_row_to_dict(row)) for row in cursor.fetchall()]

    async def get_alert_rule(self, rule_id: Any) -> dict[str, Any]:
        db = self._db()
        self._ensure_alert_rule_schema(db)
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM local_watchlist_alert_rules WHERE id = ?", (int(rule_id),))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Watchlist alert rule not found: {rule_id}")
        return normalize_watchlist_alert_rule("local", self._alert_rule_row_to_dict(row))

    async def create_alert_rule(
        self,
        *,
        name: str,
        condition_type: str,
        condition_value: Mapping[str, Any] | None = None,
        job_id: Any = None,
        source_id: Any = None,
        severity: str = "warning",
    ) -> dict[str, Any]:
        normalized_condition_type = self._validate_condition_type(condition_type)
        resolved_job_id = job_id if job_id is not None else source_id
        if resolved_job_id is not None and self._db().get_subscription(int(resolved_job_id)) is None:
            raise KeyError(f"Subscription not found: {resolved_job_id}")
        db = self._db()
        self._ensure_alert_rule_schema(db)
        now = self._utc_now()
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO local_watchlist_alert_rules (
                    job_id, name, enabled, condition_type, condition_value_json, severity, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(resolved_job_id) if resolved_job_id is not None else None,
                    name,
                    1,
                    normalized_condition_type,
                    self._serialize_condition_value(condition_value),
                    severity,
                    now,
                    now,
                ),
            )
            rule_id = cursor.lastrowid
        return await self.get_alert_rule(rule_id)

    async def update_alert_rule(self, rule_id: Any, **fields: Any) -> dict[str, Any]:
        db = self._db()
        self._ensure_alert_rule_schema(db)
        current = await self.get_alert_rule(rule_id)
        updates: dict[str, Any] = {}
        if "name" in fields:
            updates["name"] = fields["name"]
        if "enabled" in fields:
            updates["enabled"] = 1 if bool(fields["enabled"]) else 0
        if "condition_type" in fields:
            updates["condition_type"] = self._validate_condition_type(fields["condition_type"])
        if "condition_value" in fields:
            updates["condition_value_json"] = self._serialize_condition_value(fields["condition_value"])
        if "severity" in fields:
            updates["severity"] = fields["severity"]
        if "job_id" in fields:
            job_id = fields["job_id"]
            if job_id is not None and db.get_subscription(int(job_id)) is None:
                raise KeyError(f"Subscription not found: {job_id}")
            updates["job_id"] = int(job_id) if job_id is not None else None
        if "source_id" in fields:
            source_id = fields["source_id"]
            if source_id is not None and db.get_subscription(int(source_id)) is None:
                raise KeyError(f"Subscription not found: {source_id}")
            updates["job_id"] = int(source_id) if source_id is not None else None
        if not updates:
            return current

        updates["updated_at"] = self._utc_now()
        assignments = ", ".join(f"{field} = ?" for field in updates)
        values = list(updates.values()) + [int(rule_id)]
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE local_watchlist_alert_rules SET {assignments} WHERE id = ?",
                values,
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Watchlist alert rule not found: {rule_id}")
        return await self.get_alert_rule(rule_id)

    async def delete_alert_rule(self, rule_id: Any) -> dict[str, Any]:
        db = self._db()
        self._ensure_alert_rule_schema(db)
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM local_watchlist_alert_rules WHERE id = ?", (int(rule_id),))
            deleted = cursor.rowcount > 0
        if not deleted:
            raise KeyError(f"Watchlist alert rule not found: {rule_id}")
        return {
            "deleted": True,
            "id": f"local:watchlist_alert_rule:{rule_id}",
            "backend": "local",
            "entity_kind": "watchlist_alert_rule",
            "rule_id": int(rule_id),
        }

    @staticmethod
    def _local_type_for_source_type(source_type: Any) -> str:
        normalized = str(source_type or "rss").strip()
        if normalized == "site":
            return "url"
        if normalized in {"rss", "atom", "json_feed", "url", "url_list", "podcast", "sitemap", "api"}:
            return normalized
        raise ValueError(f"Unsupported local watchlist source type: {normalized}")

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _ensure_run_schema(db: SubscriptionsDB) -> None:
        with db.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS local_watchlist_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    job_id INTEGER,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    stats_json TEXT,
                    error_msg TEXT,
                    log_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES subscriptions(id) ON DELETE CASCADE
                )
                """
            )

    @staticmethod
    def _ensure_alert_rule_schema(db: SubscriptionsDB) -> None:
        with db.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS local_watchlist_alert_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    name TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    condition_type TEXT NOT NULL,
                    condition_value_json TEXT NOT NULL DEFAULT '{}',
                    severity TEXT NOT NULL DEFAULT 'warning',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES subscriptions(id) ON DELETE CASCADE
                )
                """
            )

    @staticmethod
    def _run_row_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        stats: dict[str, Any] = {}
        if payload.get("stats_json"):
            try:
                parsed = json.loads(payload["stats_json"])
                if isinstance(parsed, dict):
                    stats = parsed
            except json.JSONDecodeError:
                stats = {}
        return {
            "id": payload["id"],
            "source_id": payload.get("source_id"),
            "job_id": payload.get("job_id") or payload.get("source_id"),
            "status": payload.get("status"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "stats": stats,
            "error_msg": payload.get("error_msg"),
            "log_text": payload.get("log_text"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    @staticmethod
    def _alert_rule_row_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "user_id": "local",
            "job_id": payload.get("job_id"),
            "source_id": payload.get("job_id"),
            "name": payload.get("name"),
            "enabled": bool(payload.get("enabled", True)),
            "condition_type": payload.get("condition_type"),
            "condition_value": payload.get("condition_value_json") or "{}",
            "severity": payload.get("severity"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    @staticmethod
    def _serialize_condition_value(value: Mapping[str, Any] | str | None) -> str:
        if value is None:
            return "{}"
        if isinstance(value, str):
            return value
        return json.dumps(dict(value))

    @staticmethod
    def _validate_condition_type(condition_type: Any) -> str:
        normalized = str(condition_type or "").strip()
        if normalized not in _ALERT_CONDITION_TYPES:
            raise ValueError(
                "Invalid watchlist alert condition_type. "
                f"Expected one of: {', '.join(sorted(_ALERT_CONDITION_TYPES))}"
            )
        return normalized
