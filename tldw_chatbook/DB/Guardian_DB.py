"""SQLite persistence for the local Guardian self-monitoring subsystem (ADR-204).

Digest-only storage (ADR-204 contract 2): ``guardian_alerts`` carries
sha256 digests, topic labels, ids and timestamps -- never message text and
never matched spans. Crisis write-boundary caps (contract 7) live at
``upsert_rule`` and ``bump_escalation``: ``is_crisis=1`` rules can never
hold or escalate to ``redact``/``block`` and can never set
``feeds_discovery``.

The store follows the ``Dreams_DB`` template (per-subsystem ``BaseDB``,
additive schema-versioned DDL, thread-local held connections) because the
Guardian, like Dreams, is reached both from the UI thread and from
``asyncio.to_thread`` hops in the check pipeline.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

from loguru import logger

from ..Utils.timestamps import utc_now_iso
from .base_db import BaseDB

#: Action ladder rungs, lowest to highest (spec §Rules engine: thresholds
#: replace the base action notify -> redact -> block).
ACTION_LADDER: tuple[str, ...] = ("notify", "redact", "block")

_RULE_FIELDS: tuple[str, ...] = (
    "name",
    "topic",
    "pattern",
    "except_patterns",
    "action",
    "severity",
    "is_crisis",
    "notification_frequency",
    "display_mode",
    "escalate_session_threshold",
    "escalate_window_threshold",
    "escalate_window_days",
    "cooldown_minutes",
    "feeds_discovery",
    "enabled",
)

_RULE_DEFAULTS: dict[str, object] = {
    "except_patterns": "[]",
    "action": "notify",
    "severity": "info",
    "is_crisis": 0,
    "notification_frequency": "every_message",
    "display_mode": "inline_banner",
    "escalate_session_threshold": None,
    "escalate_window_threshold": None,
    "escalate_window_days": None,
    "cooldown_minutes": None,
    "feeds_discovery": 0,
    "enabled": 1,
}


class GuardianRuleConflict(RuntimeError):
    """Typed rejection of an invalid rule write (crisis caps, ADR-204 #7)."""

    reason_code: str = "guardian_rule_conflict"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class GuardianSchemaError(RuntimeError):
    """Typed failure for an unavailable or unsupported Guardian schema."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _utc_now_iso() -> str:
    """Return the current UTC time in the canonical stored shape (ADR-173)."""
    return utc_now_iso()


class GuardianDB(BaseDB):
    """Database wrapper for the Guardian subsystem (rules, alerts, escalation).

    Schema v1 (spec §Data model): ``guardian_rules``,
    ``guardian_alerts`` (``rule_id NULL`` rows are trend-analyzer output,
    Task 2), ``guardian_escalation_state``, ``guardian_visit_summaries``
    (written from Task 2), plus a one-row ``guardian_meta`` table holding
    the ``rules_version`` counter every rule write bumps (the compilation
    cache's invalidation key). Connections are per thread via the
    ``Dreams_DB`` idiom; writes go through ``transaction()``
    (``BEGIN IMMEDIATE``), single-statement reads use ``connection()``.

    Seed rules (spec §Seed rules) are inserted exactly once, at the
    version 0 -> 1 transition of a fresh build -- never at import, and
    never again on reopen.
    """

    _CURRENT_SCHEMA_VERSION = 1
    _WAL_SETUP_TIMEOUT_SECONDS = 5.0
    _LIVENESS_PING_IDLE_SECONDS = 30.0

    _SCHEMA_DDL = (
        """
        CREATE TABLE IF NOT EXISTS guardian_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            topic TEXT NOT NULL,
            pattern TEXT NOT NULL,
            except_patterns TEXT NOT NULL DEFAULT '[]',
            action TEXT NOT NULL DEFAULT 'notify'
                CHECK(action IN ('notify', 'redact', 'block')),
            severity TEXT NOT NULL DEFAULT 'info'
                CHECK(severity IN ('info', 'warning', 'critical')),
            is_crisis INTEGER NOT NULL DEFAULT 0,
            notification_frequency TEXT NOT NULL DEFAULT 'every_message'
                CHECK(notification_frequency IN (
                    'every_message', 'once_per_conversation',
                    'once_per_session', 'once_per_day')),
            display_mode TEXT NOT NULL DEFAULT 'inline_banner'
                CHECK(display_mode IN (
                    'inline_banner', 'post_visit_summary', 'silent_log')),
            escalate_session_threshold INTEGER,
            escalate_window_threshold INTEGER,
            escalate_window_days INTEGER,
            cooldown_minutes INTEGER,
            feeds_discovery INTEGER NOT NULL DEFAULT 0,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS guardian_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id INTEGER,
            session_id TEXT NOT NULL,
            visit_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            message_digest TEXT NOT NULL,
            ts TEXT NOT NULL
        )
        """,
        # EXPLAIN-pinned indexes (spec §Data model): the dedup counter reads
        # (rule, ts) with session/visit equality filters, the trend analyzer
        # (Task 2) reads (topic, ts), and per-session summaries read
        # (session_id, ts).
        """
        CREATE INDEX IF NOT EXISTS idx_guardian_alerts_rule_ts
            ON guardian_alerts(rule_id, ts DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_guardian_alerts_topic_ts
            ON guardian_alerts(topic, ts DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_guardian_alerts_session_ts
            ON guardian_alerts(session_id, ts)
        """,
        """
        CREATE TABLE IF NOT EXISTS guardian_escalation_state (
            rule_id INTEGER PRIMARY KEY,
            session_count INTEGER NOT NULL DEFAULT 0,
            window_count INTEGER NOT NULL DEFAULT 0,
            window_start TEXT,
            current_action TEXT,
            cooldown_until TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS guardian_visit_summaries (
            visit_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS guardian_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
    )

    def __init__(
        self, database_path: Union[str, Path], client_id: str | None = None
    ) -> None:
        # Per-thread connections (DreamsDB trap): every ":memory:" connection
        # is its own empty database, so to_thread hops would see no schema.
        if str(database_path) == ":memory:":
            raise ValueError(
                "GuardianDB needs a file path; ':memory:' is not shared "
                "across its per-thread connections"
            )
        self._thread_local = threading.local()
        super().__init__(
            database_path, client_id if client_id is not None else "default"
        )

    # ------------------------------------------------------------------
    # Connection handling (Dreams_DB idiom)
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        if not self.is_memory_db:
            self._enable_wal(conn)
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.isolation_level = None
        return conn

    def _enable_wal(self, conn: sqlite3.Connection) -> None:
        deadline = time.monotonic() + self._WAL_SETUP_TIMEOUT_SECONDS
        while True:
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def _held_connection(self) -> sqlite3.Connection:
        conn = getattr(self._thread_local, "conn", None)
        if conn is not None:
            last_used = getattr(self._thread_local, "conn_last_used", None)
            if (
                last_used is None
                or (time.monotonic() - last_used)
                >= self._LIVENESS_PING_IDLE_SECONDS
            ):
                try:
                    conn.execute("SELECT 1")
                except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001 - already unusable
                        pass
                    conn = None
        if conn is None:
            conn = self._get_connection()
            self._thread_local.conn = conn
        self._thread_local.conn_last_used = time.monotonic()
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        yield self._held_connection()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._held_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    def close(self) -> None:
        conn = getattr(self._thread_local, "conn", None)
        self._thread_local.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    # ------------------------------------------------------------------
    # Schema + seeding
    # ------------------------------------------------------------------

    def _initialize_schema(self) -> None:
        """Atomically initialize the Guardian schema (additive, idempotent).

        Seed rules ride the version 0 -> 1 transition only: a second build
        finds version 1 already recorded and never re-seeds (pinned by
        ``test_second_build_does_not_reseed``).
        """
        with self.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version"
                " (version INTEGER PRIMARY KEY NOT NULL)"
            )
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current_version = int(row[0] or 0) if row is not None else 0
            if current_version > self._CURRENT_SCHEMA_VERSION:
                raise GuardianSchemaError("schema_too_new")
            fresh_build = current_version == 0
            for statement in self._SCHEMA_DDL:
                conn.execute(statement)
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (self._CURRENT_SCHEMA_VERSION,),
            )
            if fresh_build:
                self._seed_defaults(conn)

    @staticmethod
    def _seed_defaults(conn: sqlite3.Connection) -> None:
        """Insert the three v1 seed rules into a FRESH store (spec §Seed rules).

        All visible and deletable, none with ``feeds_discovery``; the
        crisis rule is capped to ``notify`` and excepts research/treatment
        vocabulary so clinical discussion does not fire it.
        """
        existing = conn.execute("SELECT COUNT(*) FROM guardian_rules").fetchone()
        if existing is not None and int(existing[0]) > 0:
            return
        now = _utc_now_iso()
        seeds = (
            {
                "name": "Crisis awareness (self-harm)",
                "topic": "crisis_awareness",
                "pattern": r"\b(kill myself|end my life|want to die|"
                r"hurt myself|self[- ]harm|suicid\w*)\b",
                "except_patterns": json.dumps(
                    [
                        "prevention",
                        "hotline",
                        "awareness",
                        "research",
                        "study",
                        "clinical",
                        "treatment",
                        "therapy",
                    ]
                ),
                "action": "notify",
                "severity": "critical",
                "is_crisis": 1,
                "notification_frequency": "once_per_conversation",
                "display_mode": "inline_banner",
                "escalate_session_threshold": None,
                "escalate_window_threshold": None,
                "escalate_window_days": None,
                "cooldown_minutes": None,
                "feeds_discovery": 0,
                "enabled": 1,
            },
            {
                "name": "Doomscrolling awareness (demo)",
                "topic": "doomscrolling",
                "pattern": r"doomscroll\w*|can't stop scrolling|"
                r"endless scroll\w*",
                "except_patterns": "[]",
                "action": "notify",
                "severity": "info",
                "is_crisis": 0,
                "notification_frequency": "once_per_day",
                "display_mode": "silent_log",
                "escalate_session_threshold": None,
                "escalate_window_threshold": None,
                "escalate_window_days": None,
                "cooldown_minutes": None,
                "feeds_discovery": 0,
                "enabled": 1,
            },
            {
                # Editable example demonstrating every editor field.
                "name": "Late-night work (example)",
                "topic": "late_night_work",
                "pattern": r"one more thing|just finishing up|"
                r"quick fix before bed|working late again",
                "except_patterns": "[]",
                "action": "notify",
                "severity": "info",
                "is_crisis": 0,
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
                "escalate_session_threshold": 5,
                "escalate_window_threshold": 20,
                "escalate_window_days": 7,
                "cooldown_minutes": 5,
                "feeds_discovery": 0,
                "enabled": 1,
            },
        )
        columns = ", ".join(_RULE_FIELDS)
        placeholders = ", ".join("?" for _ in _RULE_FIELDS)
        for seed in seeds:
            conn.execute(
                f"INSERT INTO guardian_rules ({columns}, created_at, updated_at)"
                f" VALUES ({placeholders}, ?, ?)",
                (*(seed[field] for field in _RULE_FIELDS), now, now),
            )
        conn.execute(
            "INSERT OR REPLACE INTO guardian_meta (key, value)"
            " VALUES ('rules_version', '1')"
        )

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def list_rules(self, enabled_only: bool = False) -> list[dict]:
        """Return rule rows (parsed ``except_patterns``), ordered by id."""
        query = "SELECT * FROM guardian_rules"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY id"
        with self.connection() as conn:
            rows = conn.execute(query).fetchall()
        return [self._rule_row(dict(row)) for row in rows]

    def get_rule(self, rule_id: int) -> dict | None:
        """Return one rule row, or None."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM guardian_rules WHERE id = ?", (rule_id,)
            ).fetchone()
        return self._rule_row(dict(row)) if row is not None else None

    @staticmethod
    def _rule_row(row: dict) -> dict:
        excepts = row.get("except_patterns", "[]")
        if isinstance(excepts, str):
            try:
                row["except_patterns"] = json.loads(excepts)
            except (TypeError, ValueError):
                row["except_patterns"] = []
        return row

    def upsert_rule(self, **fields: object) -> int:
        """Insert or update one rule; return its row id.

        Crisis write-boundary caps (ADR-204 contract 7, enforced here
        because SQLite CHECKs cannot express cross-column constraints):
        a row whose MERGED state has ``is_crisis=1`` combined with an
        action other than ``notify`` or with ``feeds_discovery=1`` is
        rejected with :class:`GuardianRuleConflict`. Every write bumps the
        ``rules_version`` counter, the compilation cache's invalidation
        key.

        Args:
            **fields: Any subset of rule columns; ``id`` selects update,
                its absence insert. Unpassed fields keep their value on
                update / their default on insert.

        Returns:
            The rule's row id.

        Raises:
            GuardianRuleConflict: The merged row violates a crisis cap.
            ValueError: An unknown field name was passed.
        """
        unknown = set(fields) - set(_RULE_FIELDS) - {"id"}
        if unknown:
            raise ValueError(f"Unknown rule fields: {sorted(unknown)}")

        with self.transaction() as conn:
            rule_id = fields.pop("id", None)
            merged: dict[str, object] = dict(_RULE_DEFAULTS)
            if rule_id is not None:
                row = conn.execute(
                    "SELECT * FROM guardian_rules WHERE id = ?", (rule_id,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"No guardian rule with id {rule_id}")
                merged.update(dict(row))
            merged.update(fields)
            merged["is_crisis"] = 1 if merged.get("is_crisis") else 0
            merged["feeds_discovery"] = 1 if merged.get("feeds_discovery") else 0

            if merged["is_crisis"]:
                if merged.get("action") != "notify":
                    raise GuardianRuleConflict(
                        "Crisis-flagged rules may only surface notifications; "
                        f"action {merged.get('action')!r} is not allowed."
                    )
                if merged["feeds_discovery"]:
                    raise GuardianRuleConflict(
                        "Crisis-flagged rules may never feed Dreams "
                        "discovery (crisis-adjacent text must not become "
                        "an outbound search query)."
                    )

            now = _utc_now_iso()
            if rule_id is None:
                columns = [field for field in _RULE_FIELDS]
                values = [merged[field] for field in columns]
                cursor = conn.execute(
                    "INSERT INTO guardian_rules"
                    f" ({', '.join(columns)}, created_at, updated_at)"
                    f" VALUES ({', '.join('?' for _ in columns)}, ?, ?)",
                    (*values, now, now),
                )
                rule_id = int(cursor.lastrowid)
            else:
                columns = [field for field in _RULE_FIELDS if field in fields]
                if columns:
                    assignments = ", ".join(f"{name} = ?" for name in columns)
                    conn.execute(
                        f"UPDATE guardian_rules SET {assignments},"
                        " updated_at = ? WHERE id = ?",
                        (*(merged[name] for name in columns), now, rule_id),
                    )
            self._bump_rules_version(conn)
            return int(rule_id)

    def delete_rule(self, rule_id: int) -> None:
        """Delete one rule and its escalation state (alerts are history)."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM guardian_rules WHERE id = ?", (rule_id,))
            conn.execute(
                "DELETE FROM guardian_escalation_state WHERE rule_id = ?",
                (rule_id,),
            )
            self._bump_rules_version(conn)

    def rules_version(self) -> int:
        """Return the rules-version counter (bumped by every rule write)."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value FROM guardian_meta WHERE key = 'rules_version'"
            ).fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def _bump_rules_version(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO guardian_meta (key, value)"
            " VALUES ('rules_version', '1')"
            " ON CONFLICT(key) DO UPDATE SET"
            " value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
        )

    # ------------------------------------------------------------------
    # Alerts (digest-only storage)
    # ------------------------------------------------------------------

    def insert_alert(
        self,
        *,
        rule_id: int | None = None,
        session_id: str,
        visit_id: str,
        topic: str,
        message_digest: str,
        ts: str | None = None,
    ) -> int:
        """Insert one alert row; returns the new row id.

        ``message_digest`` is the sha256 of the whitespace-stripped prompt
        (dedup counting only) -- never the prompt itself (ADR-204
        contract 2). ``rule_id NULL`` rows are trend-analyzer output.
        ``ts`` defaults to the wall clock; the checker passes its own
        injected clock so alert timestamps, dedup windows, and escalation
        windows share one time base.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO guardian_alerts"
                " (rule_id, session_id, visit_id, topic, message_digest, ts)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (rule_id, session_id, visit_id, topic, message_digest,
                 ts or _utc_now_iso()),
            )
            return int(cursor.lastrowid)

    def count_recent_alerts(
        self,
        rule_id: int,
        *,
        session_id: str | None = None,
        visit_id: str | None = None,
        since_iso: str,
    ) -> int:
        """Count a rule's alerts in a scope since a cutoff.

        Args:
            rule_id: Rule whose alerts are counted.
            session_id: Optional conversation scope (once_per_conversation
                dedup -- the persisted chat session).
            visit_id: Optional visit scope (once_per_session dedup in the
                server's naming: one sitting).
            since_iso: ISO cutoff (24h window for once_per_day dedup).

        Returns:
            The matching alert count.
        """
        query = "SELECT COUNT(*) FROM guardian_alerts WHERE rule_id = ?"
        params: list[object] = [rule_id]
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        if visit_id is not None:
            query += " AND visit_id = ?"
            params.append(visit_id)
        query += " AND ts >= ?"
        params.append(since_iso)
        with self.connection() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row[0]) if row is not None else 0

    def prune_alerts(self, cutoff_iso: str) -> int:
        """Delete alerts older than the cutoff; returns the deleted count.

        Retention sweep (ADR-204 contract 9): best-effort, logged; visit
        summaries are deliberately not pruned (small, kept indefinitely).
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM guardian_alerts WHERE ts < ?", (cutoff_iso,)
            )
            count = int(cursor.rowcount)
        if count:
            logger.debug("Guardian retention pruned {} alerts", count)
        return count

    def count_topic_alerts(self, topic: str, *, visit_id: str) -> int:
        """Count a topic's alert rows inside one visit (any rule_id).

        The trend analyzer's frequency-cap read: one trend notice per topic
        per visit (spec §Trend analyzer). Topic-scoped, not rule-scoped, so
        it counts exactly the rows the analyzer itself inserts
        (``rule_id IS NULL``) plus any rule rows sharing the topic.

        Args:
            topic: The alert topic (e.g. ``fixation:<t>``).
            visit_id: The visit whose rows are counted.

        Returns:
            The matching alert count.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM guardian_alerts"
                " WHERE topic = ? AND visit_id = ?",
                (topic, visit_id),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def visit_alert_rows(self, visit_id: str) -> list[dict]:
        """Return per-rule aggregates for one visit, joined with rule state.

        The post-visit summary's read (spec §Post-visit summary): one row
        per rule that fired in the visit, with its hit count, display mode,
        configured action, and the escalation state's current action -- the
        inputs for ``per_topic_counts`` (non-silent rules only) and
        ``escalated_rules`` (current action above the rule's base action).

        Args:
            visit_id: The visit being summarized.

        Returns:
            ``[{rule_id, topic, name, display_mode, rule_action,
            current_action, hits}]`` ordered by rule id.
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT a.rule_id AS rule_id, r.topic AS topic,"
                " r.name AS name, r.display_mode AS display_mode,"
                " r.action AS rule_action, s.current_action AS"
                " current_action, COUNT(*) AS hits"
                " FROM guardian_alerts a"
                " JOIN guardian_rules r ON r.id = a.rule_id"
                " LEFT JOIN guardian_escalation_state s"
                " ON s.rule_id = a.rule_id"
                " WHERE a.visit_id = ?"
                " GROUP BY a.rule_id ORDER BY a.rule_id",
                (visit_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Visit summaries
    # ------------------------------------------------------------------

    def insert_visit_summary(
        self, *, visit_id: str, session_id: str, payload: dict
    ) -> int:
        """Insert (or replace) one visit's summary; returns the row id.

        ``guardian_visit_summaries`` is keyed by ``visit_id`` (PRIMARY
        KEY), so a visit finalized twice replaces its row rather than
        duplicating it.

        Args:
            visit_id: The visit this summary describes.
            session_id: The visit's persisted chat session.
            payload: JSON-serializable summary dict (per-topic counts,
                escalated rules, trend notices).

        Returns:
            The summary row id.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO guardian_visit_summaries"
                " (visit_id, session_id, created_at, payload)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(visit_id) DO UPDATE SET"
                " session_id = excluded.session_id,"
                " created_at = excluded.created_at,"
                " payload = excluded.payload",
                (visit_id, session_id, _utc_now_iso(), json.dumps(payload)),
            )
            return int(cursor.lastrowid)

    def get_visit_summary(self, visit_id: str) -> dict | None:
        """Return one stored visit summary row, or None.

        Args:
            visit_id: The visit whose summary is read.

        Returns:
            ``{visit_id, session_id, created_at, payload}`` with ``payload``
            still the stored JSON string, or None when no row exists.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM guardian_visit_summaries WHERE visit_id = ?",
                (visit_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # Meta (rules_version + the daily trend watermark)
    # ------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        """Return one ``guardian_meta`` value, or None when unset."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value FROM guardian_meta WHERE key = ?", (key,)
            ).fetchone()
        return str(row[0]) if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        """Set one ``guardian_meta`` value (upsert)."""
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO guardian_meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ------------------------------------------------------------------
    # Escalation + cooldown
    # ------------------------------------------------------------------

    def bump_escalation(self, rule_id: int, *, now: str) -> dict:
        """Increment a rule's counters and return its (capped) action.

        Applies the server's escalation semantics: the session counter
        never resets; the window counter resets when ``now`` falls beyond
        ``escalate_window_days`` of ``window_start``. Thresholds replace
        the base action up the ladder (notify -> redact -> block): the
        session threshold escalates one rung, the window threshold two
        (capped at block). Crisis cap (ADR-204 contract 7): a rule with
        ``is_crisis=1`` always resolves to ``notify`` -- crisis rules
        never silence or hide distress text, no matter the thresholds.

        Args:
            rule_id: The matched rule.
            now: Current UTC ISO timestamp.

        Returns:
            ``{"session_count", "window_count", "current_action"}``.
        """
        with self.transaction() as conn:
            rule = conn.execute(
                "SELECT action, is_crisis, escalate_session_threshold,"
                " escalate_window_threshold, escalate_window_days"
                " FROM guardian_rules WHERE id = ?",
                (rule_id,),
            ).fetchone()
            if rule is None:
                raise ValueError(f"No guardian rule with id {rule_id}")
            state = conn.execute(
                "SELECT * FROM guardian_escalation_state WHERE rule_id = ?",
                (rule_id,),
            ).fetchone()

            session_count = 1 if state is None else int(state["session_count"]) + 1
            window_count = 1
            window_start: str = now
            if state is not None:
                window_days = rule["escalate_window_days"]
                expired = (
                    window_days is not None
                    and state["window_start"] is not None
                    and str(now) > _shift_iso(str(state["window_start"]),
                                             float(window_days))
                )
                window_count = (
                    1 if expired else int(state["window_count"]) + 1
                )
                window_start = now if expired else str(state["window_start"])

            base = ACTION_LADDER.index(str(rule["action"]))
            rung = base
            session_threshold = rule["escalate_session_threshold"]
            window_threshold = rule["escalate_window_threshold"]
            if (
                session_threshold is not None
                and session_count >= int(session_threshold)
            ):
                rung = max(rung, base + 1)
            if (
                window_threshold is not None
                and window_count >= int(window_threshold)
            ):
                rung = max(rung, base + 2)
            rung = min(rung, len(ACTION_LADDER) - 1)
            current_action = (
                "notify" if rule["is_crisis"] else ACTION_LADDER[rung]
            )

            conn.execute(
                "INSERT INTO guardian_escalation_state"
                " (rule_id, session_count, window_count, window_start,"
                "  current_action, cooldown_until)"
                " VALUES (?, ?, ?, ?, ?, NULL)"
                " ON CONFLICT(rule_id) DO UPDATE SET"
                " session_count = excluded.session_count,"
                " window_count = excluded.window_count,"
                " window_start = excluded.window_start,"
                " current_action = excluded.current_action",
                (rule_id, session_count, window_count, window_start,
                 current_action),
            )
            return {
                "session_count": session_count,
                "window_count": window_count,
                "current_action": current_action,
            }

    def set_cooldown(self, rule_id: int, until_iso: str) -> None:
        """Arm (or clear, with a past timestamp) a rule's cooldown."""
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO guardian_escalation_state"
                " (rule_id, session_count, window_count, window_start,"
                "  current_action, cooldown_until)"
                " VALUES (?, 0, 0, NULL, NULL, ?)"
                " ON CONFLICT(rule_id) DO UPDATE SET"
                " cooldown_until = excluded.cooldown_until",
                (rule_id, until_iso),
            )

    def cooldown_active(self, rule_id: int, *, now: str) -> bool:
        """Return whether the rule's cooldown is still in the future."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT cooldown_until FROM guardian_escalation_state"
                " WHERE rule_id = ?",
                (rule_id,),
            ).fetchone()
        if row is None or row["cooldown_until"] is None:
            return False
        return str(row["cooldown_until"]) > str(now)


def _shift_iso(iso: str, days: float) -> str:
    """Return ``iso + days`` preserving the input's suffix style.

    Window expiry compares ISO strings; a shifted bound must keep the
    stored ``Z``/``+00:00`` shape rather than ``isoformat()``'s
    ``+00:00`` default.
    """
    from datetime import datetime, timedelta

    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    shifted = (moment + timedelta(days=days)).isoformat()
    if iso.endswith("Z") and shifted.endswith("+00:00"):
        return shifted[:-6] + "Z"
    return shifted
