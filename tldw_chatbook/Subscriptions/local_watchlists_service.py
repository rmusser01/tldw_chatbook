"""Local watchlists adapter backed by the existing subscriptions database."""

from __future__ import annotations

import json
import inspect
import time
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

    def __init__(
        self,
        *,
        db_factory: Callable[[], SubscriptionsDB],
        notification_dispatcher: Any | None = None,
        notification_app: Any | None = None,
        run_executor: Callable[[Mapping[str, Any]], Any] | None = None,
    ):
        self.db_factory = db_factory
        self.notification_dispatcher = notification_dispatcher
        self.notification_app = notification_app
        self.run_executor = run_executor

    def _db(self) -> SubscriptionsDB:
        return self.db_factory()

    async def list_sources(self, *, limit: int = 100, offset: int = 0, q: str | None = None) -> list[dict[str, Any]]:
        normalized_limit = int(limit)
        normalized_offset = int(offset)
        fetch_limit = normalized_limit if not q else max(normalized_limit + normalized_offset, 1000)
        rows = self._db().get_all_subscriptions(include_inactive=True, limit=fetch_limit, offset=0 if q else normalized_offset)
        items = [normalize_local_subscription_row(row) for row in rows]
        if not q:
            return items
        needle = str(q).strip().lower()
        filtered = [
            item
            for item in items
            if needle in str(item.get("title") or "").lower()
            or needle in str(item.get("url") or "").lower()
        ]
        return filtered[normalized_offset : normalized_offset + normalized_limit]

    async def get_source(self, source_id: Any) -> dict[str, Any]:
        row = self._db().get_subscription(int(source_id))
        if row is None:
            raise KeyError(f"Subscription not found: {source_id}")
        return normalize_local_subscription_row(row)

    async def create_source(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        db = self._db()
        local_type = self._local_type_for_source_type(payload.get("source_type"))
        source = str(
            payload.get("url")
            or payload.get("source")
            or self._first_configured_url(payload)
            or ""
        )
        source_id = db.add_subscription(
            name=str(payload.get("name") or "Untitled subscription"),
            type=local_type,
            source=source,
            tags=list(payload.get("tags") or []),
            description=payload.get("description"),
            **self._subscription_config_fields(payload),
        )
        return normalize_local_subscription_row(db.get_subscription(source_id))

    async def update_source(self, source_id: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
        db = self._db()
        changes: dict[str, Any] = {}
        if "name" in payload:
            changes["name"] = payload["name"]
        if "url" in payload:
            changes["source"] = payload["url"]
        elif "source" in payload:
            changes["source"] = payload["source"]
        elif "extraction_rules" in payload:
            configured_url = self._first_configured_url(payload)
            if configured_url:
                changes["source"] = configured_url
        if "tags" in payload:
            changes["tags"] = payload["tags"]
        if "active" in payload:
            changes["is_active"] = bool(payload["active"])
        if "description" in payload:
            changes["description"] = payload["description"]
        if "source_type" in payload:
            changes["type"] = self._local_type_for_source_type(payload["source_type"])
        changes.update(self._subscription_config_fields(payload))
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

    async def execute_run(self, run_id: Any) -> dict[str, Any]:
        """Execute a queued local watchlist run and persist its observed result."""
        db = self._db()
        self._ensure_run_schema(db)
        current = await self.get_run(run_id)
        source_id = int(current.get("source_id") or current.get("job_id"))
        subscription = db.get_subscription(source_id)
        if subscription is None:
            raise KeyError(f"Subscription not found: {source_id}")

        self._mark_run_started(db, int(run_id))
        start_time = time.time()
        try:
            result = await self._execute_subscription(subscription, db)
            items = list(result.get("items") or [])
            stats = dict(result.get("stats") or {})
            stats.setdefault("items_found", len(items))
            stats.setdefault("items_ingested", len(items))
            stats.setdefault("new_items_found", len(items))
            stats.setdefault("response_time_ms", int((time.time() - start_time) * 1000))
            db.record_check_result(source_id, items=items or None, stats=stats)
            return await self.record_run_result(
                run_id,
                status=str(result.get("status") or "completed"),
                stats=stats,
                error_msg=result.get("error_msg"),
                log_text=result.get("log_text"),
            )
        except Exception as exc:
            error_msg = str(exc)
            db.record_check_error(source_id, error_msg)
            return await self.record_run_result(
                run_id,
                status="failed",
                stats={
                    "items_found": 0,
                    "items_ingested": 0,
                    "error_msg": error_msg,
                    "response_time_ms": int((time.time() - start_time) * 1000),
                },
                error_msg=error_msg,
                log_text=f"Local watchlist execution failed: {error_msg}",
            )

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

    async def record_run_result(
        self,
        run_id: Any,
        *,
        status: str,
        stats: Mapping[str, Any] | None = None,
        error_msg: str | None = None,
        log_text: str | None = None,
        dispatch_notifications: bool = True,
    ) -> dict[str, Any]:
        """Persist a completed local run and emit notifications for matching alert rules."""
        db = self._db()
        self._ensure_run_schema(db)
        current = await self.get_run(run_id)
        now = self._utc_now()
        stats_payload = dict(stats or {})
        if error_msg and "error_msg" not in stats_payload:
            stats_payload["error_msg"] = error_msg
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE local_watchlist_runs
                SET status = ?, finished_at = ?, stats_json = ?, error_msg = ?, log_text = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    str(status),
                    now,
                    json.dumps(stats_payload, sort_keys=True),
                    error_msg,
                    log_text,
                    now,
                    int(run_id),
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Watchlist run not found: {run_id}")

        triggered_alerts = self._evaluate_alert_rules_for_run(
            run_id=int(run_id),
            job_id=int(current.get("job_id") or current.get("source_id")),
            stats=stats_payload,
            status=str(status),
        )
        if dispatch_notifications:
            for alert in triggered_alerts:
                notification = self._dispatch_alert_notification(alert)
                if notification is not None:
                    alert["notification_id"] = notification.get("id")

        updated = await self.get_run(run_id)
        updated["triggered_alerts"] = triggered_alerts
        return updated

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

    @classmethod
    def _first_configured_url(cls, payload: Mapping[str, Any]) -> str | None:
        extraction_rules = cls._parse_json_value(payload.get("extraction_rules"))
        urls = cls._coerce_url_list(
            extraction_rules.get("urls") if isinstance(extraction_rules, Mapping) else None
        )
        if urls:
            return urls[0]
        return None

    @staticmethod
    def _subscription_config_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed_fields = (
            "check_frequency",
            "extraction_method",
            "extraction_rules",
            "processing_options",
            "notification_config",
            "change_threshold",
            "ignore_selectors",
            "custom_headers",
            "rate_limit_config",
            "auto_pause_threshold",
        )
        return {
            field: payload[field]
            for field in allowed_fields
            if field in payload and payload[field] is not None
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _mark_run_started(self, db: SubscriptionsDB, run_id: int) -> None:
        now = self._utc_now()
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE local_watchlist_runs
                SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ?
                """,
                ("running", now, now, run_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Watchlist run not found: {run_id}")

    async def _execute_subscription(
        self,
        subscription: Mapping[str, Any],
        db: SubscriptionsDB,
    ) -> dict[str, Any]:
        executor = self.run_executor
        if executor is None:
            result = await self._default_run_executor(subscription, db)
        else:
            result = await self._maybe_await(executor(subscription))
        if result is None:
            return {"items": []}
        if isinstance(result, list):
            return {"items": result}
        if not isinstance(result, Mapping):
            raise ValueError("Local watchlist run executor must return a mapping or list of items.")
        return dict(result)

    async def _default_run_executor(
        self,
        subscription: Mapping[str, Any],
        db: SubscriptionsDB,
    ) -> dict[str, Any]:
        from .monitoring_engine import FeedMonitor, URLMonitor

        subscription_config = self._subscription_execution_config(subscription)
        source_type = str(subscription_config.get("type") or "").strip()
        if source_type in {"rss", "atom", "json_feed", "podcast"}:
            items = await FeedMonitor().check_feed(subscription_config)
        elif source_type == "url":
            result = await URLMonitor(db).check_url(subscription_config)
            items = [result] if result else []
        elif source_type == "url_list":
            monitor = URLMonitor(db)
            items = []
            for url in self._urls_for_url_list(subscription_config):
                result = await monitor.check_url(
                    {
                        **subscription_config,
                        "source": url,
                        "type": "url",
                    }
                )
                if result:
                    items.append(result)
        elif source_type == "sitemap":
            monitor = URLMonitor(db)
            items = []
            for url in await self._urls_for_sitemap(subscription_config):
                result = await monitor.check_url(
                    {
                        **subscription_config,
                        "source": url,
                        "type": "url",
                    }
                )
                if result:
                    items.append(result)
        else:
            raise ValueError(f"Unsupported local watchlist source type for execution: {source_type}")
        return {
            "items": items,
            "log_text": f"Local watchlist execution completed with {len(items)} item(s).",
        }

    @classmethod
    def _subscription_execution_config(cls, subscription: Mapping[str, Any]) -> dict[str, Any]:
        config = dict(subscription)
        for field in (
            "extraction_rules",
            "processing_options",
            "notification_config",
            "rate_limit_config",
        ):
            if field in config:
                config[field] = cls._parse_json_value(config[field])
        return config

    @classmethod
    def _urls_for_url_list(cls, subscription: Mapping[str, Any]) -> list[str]:
        extraction_rules = subscription.get("extraction_rules")
        urls = []
        if isinstance(extraction_rules, Mapping):
            urls = cls._coerce_url_list(extraction_rules.get("urls"))
        if not urls:
            urls = cls._coerce_url_list(subscription.get("source"))

        return cls._apply_max_urls(urls, subscription)

    @classmethod
    async def _urls_for_sitemap(cls, subscription: Mapping[str, Any]) -> list[str]:
        import httpx

        try:
            import defusedxml.ElementTree as ET
        except ImportError:
            import xml.etree.ElementTree as ET

        source = str(subscription.get("source") or "").strip()
        if not source:
            return []

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(source)
            response.raise_for_status()

        root = ET.fromstring(response.text)
        urls: list[str] = []
        for url_node in root.iter():
            if cls._xml_local_name(url_node.tag) != "url":
                continue
            for child in list(url_node):
                if cls._xml_local_name(child.tag) == "loc" and child.text:
                    normalized_url = child.text.strip()
                    if normalized_url:
                        urls.append(normalized_url)
                    break
        return cls._apply_max_urls(urls, subscription)

    @staticmethod
    def _xml_local_name(tag: Any) -> str:
        text = str(tag)
        if "}" in text:
            return text.rsplit("}", 1)[-1]
        return text

    @staticmethod
    def _apply_max_urls(urls: list[str], subscription: Mapping[str, Any]) -> list[str]:
        processing_options = subscription.get("processing_options")
        max_urls = None
        if isinstance(processing_options, Mapping) and processing_options.get("max_urls") is not None:
            try:
                max_urls = max(int(processing_options["max_urls"]), 0)
            except (TypeError, ValueError):
                max_urls = None
        return urls[:max_urls] if max_urls is not None else urls

    @staticmethod
    def _parse_json_value(value: Any) -> Any:
        if value in (None, ""):
            return {}
        if isinstance(value, (Mapping, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value

    @staticmethod
    def _coerce_url_list(value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            parts = value.replace(",", "\n").splitlines()
            return [part.strip() for part in parts if part.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

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

    def _evaluate_alert_rules_for_run(
        self,
        *,
        run_id: int,
        job_id: int,
        stats: Mapping[str, Any],
        status: str,
    ) -> list[dict[str, Any]]:
        db = self._db()
        self._ensure_alert_rule_schema(db)
        rules = db.conn.execute(
            """
            SELECT * FROM local_watchlist_alert_rules
            WHERE enabled = 1 AND (job_id = ? OR job_id IS NULL)
            ORDER BY created_at DESC
            """,
            (job_id,),
        ).fetchall()
        triggered: list[dict[str, Any]] = []
        for row in rules:
            rule = normalize_watchlist_alert_rule("local", self._alert_rule_row_to_dict(row))
            message = self._alert_message_for_rule(rule, stats=stats, status=status)
            if message is None:
                continue
            rule_id = int(rule["rule_id"])
            triggered.append(
                {
                    "rule_id": rule_id,
                    "rule_name": rule["name"],
                    "condition_type": rule["condition_type"],
                    "severity": rule["severity"],
                    "message": message,
                    "notification_payload": {
                        "kind": "watchlist_alert",
                        "source_job_id": str(job_id),
                        "source_domain": "watchlists",
                        "source_job_type": "watchlist_run",
                        "link_type": "watchlist_run",
                        "link_id": str(run_id),
                        "dedupe_key": f"watchlist-alert:{rule_id}:{run_id}",
                    },
                }
            )
        return triggered

    def _dispatch_alert_notification(self, alert: Mapping[str, Any]) -> dict[str, Any] | None:
        dispatcher = self.notification_dispatcher
        if dispatcher is None:
            return None
        return dispatcher.dispatch(
            app=self.notification_app,
            category="watchlists",
            title=f"Alert: {alert['rule_name']}",
            message=str(alert["message"]),
            severity=str(alert["severity"]),
            source_backend="local",
            source_entity_kind="watchlist_run",
            source_entity_id=str(alert["notification_payload"]["link_id"]),
            payload=dict(alert["notification_payload"]),
        )

    def _alert_message_for_rule(
        self,
        rule: Mapping[str, Any],
        *,
        stats: Mapping[str, Any],
        status: str,
    ) -> str | None:
        condition_type = rule.get("condition_type")
        items_found = self._coerce_int(stats.get("items_found"), default=0)
        items_ingested = self._coerce_int(stats.get("items_ingested"), default=0)
        error_rate = 1.0 - (items_ingested / items_found) if items_found > 0 else 0.0
        condition_value = dict(rule.get("condition_value") or {})

        if condition_type == "no_items":
            if items_ingested == 0:
                return f"Run produced 0 items (found {items_found})"
            return None
        if condition_type == "error_rate_above":
            threshold = self._coerce_float(condition_value.get("threshold"), default=0.5)
            if threshold is None:
                return None
            if error_rate > threshold:
                return f"Error rate {error_rate:.0%} exceeds {threshold:.0%} threshold"
            return None
        if condition_type == "items_below":
            threshold = self._coerce_optional_int(condition_value.get("threshold"), default=1)
            if threshold is None:
                return None
            if items_ingested < threshold:
                return f"Only {items_ingested} items ingested (threshold: {threshold})"
            return None
        if condition_type == "items_above":
            threshold = self._coerce_optional_int(condition_value.get("threshold"), default=1000)
            if threshold is None:
                return None
            if items_ingested > threshold:
                return f"{items_ingested} items ingested exceeds {threshold} threshold"
            return None
        if condition_type == "run_failed":
            if status == "failed":
                return f"Run failed: {stats.get('error_msg') or 'unknown error'}"
            return None
        return None

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_optional_int(value: Any, *, default: int) -> int | None:
        value = default if value is None else value
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float | None:
        value = default if value is None else value
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
