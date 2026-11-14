"""Parameterized storage-side recovery qualification and bounded projections.

The page limit bounds materialization, not scan cost. A deadline interrupts all
SQL work, including qualification and provenance; no embedding index is used.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .library_search_engine import escape_library_like

if TYPE_CHECKING:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def normalized_archive_time(value: str | None, assume_utc: bool = False) -> str | None:
    """Normalize UTC; only legacy last-modified values may assume a UTC offset."""
    if not value:
        return None
    try:
        date = datetime.fromisoformat(value)
        if date.tzinfo is None or date.utcoffset() is None:
            if not assume_utc:
                return None
            date = date.replace(tzinfo=UTC)
        return date.astimezone(UTC).isoformat(timespec="microseconds")
    except (ValueError, TypeError, AttributeError):
        return None


# Canonical role/content only: sidecars never enter predicates or projections.
# Ambiguous legacy JSON/control encodings retain title/keyword discovery.
# Unicode White_Space plus Python's ASCII separators and the BOM, which
# commonly precedes serialized legacy imports. SQLite ltrim's default is only
# ASCII space; the same bound set is used for qualification and projection.
_LEADING_WHITESPACE = (
    " \t\n\r\v\f\x1c\x1d\x1e\x1f\x85\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_VISIBLE = """m.deleted = 0 AND m.role IN ('user', 'assistant')
    AND typeof(m.content) = 'text'
    AND substr(ltrim(m.content, :leading_whitespace), 1, 1) NOT IN ('{', '[')
    AND lower(m.content) NOT LIKE '%<think%'
    AND lower(m.content) NOT LIKE '%<analysis%'
    AND lower(m.content) NOT LIKE '%<|%'
"""

_BASE = f"""
WITH workspace AS (
    SELECT json_extract(value, '$.workspace_id') AS workspace_id,
           json_extract(value, '$.archived') AS archived,
           json_extract(value, '$.archived_at') AS archived_at
    FROM json_each(:workspaces)
), eligible AS (
    SELECT c.id, c.version, substr(c.title,1,1025) AS title,
           c.archived, c.scope_type, c.workspace_id,
           archive_recovery_utc(c.last_modified, 1) AS modified_at,
           CASE WHEN c.archived = 1 THEN archive_recovery_utc(c.archived_at) END AS chat_at,
           CASE WHEN w.archived = 1 THEN archive_recovery_utc(w.archived_at) END AS workspace_at,
           w.archived AS workspace_archived,
           c.active_leaf_message_id,
           (c.title LIKE :pattern ESCAPE '\\') AS title_match,
           EXISTS (SELECT 1 FROM conversation_keywords ck JOIN keywords k ON k.id = ck.keyword_id
                   WHERE ck.conversation_id = c.id AND k.deleted = 0
                     AND k.keyword LIKE :pattern ESCAPE '\\') AS keyword_match
    FROM conversations c LEFT JOIN workspace w ON w.workspace_id = c.workspace_id AND c.scope_type = 'workspace'
    WHERE c.deleted = 0 AND (c.archived = 1 OR w.archived = 1)
      AND (:workspace_filter IS NULL OR
           (:workspace_filter = :default_id AND (c.scope_type = 'global' OR c.workspace_id = :default_id)) OR
           (c.scope_type = 'workspace' AND c.workspace_id = :workspace_filter))
      AND (:query = '' OR c.title LIKE :pattern ESCAPE '\\'
           OR EXISTS (SELECT 1 FROM conversation_keywords ck JOIN keywords k ON k.id = ck.keyword_id
                      WHERE ck.conversation_id = c.id AND k.deleted = 0 AND k.keyword LIKE :pattern ESCAPE '\\')
           OR EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id AND {_VISIBLE}
                      AND m.content LIKE :pattern ESCAPE '\\'))
), events AS (
    SELECT *, (chat_at IS NOT NULL AND (:lower IS NULL OR chat_at >= :lower)
                  AND (:upper IS NULL OR chat_at < :upper)) AS chat_match,
              (workspace_at IS NOT NULL AND (:lower IS NULL OR workspace_at >= :lower)
                  AND (:upper IS NULL OR workspace_at < :upper)) AS workspace_match,
              ((archived = 1 AND chat_at IS NULL) OR
               (workspace_archived = 1 AND workspace_at IS NULL)) AS unknown_time
    FROM eligible
), ordered AS (
    SELECT *, CASE WHEN chat_match OR workspace_match THEN 0 ELSE 1 END AS sort_group,
           CASE WHEN chat_match OR workspace_match THEN
               max(CASE WHEN chat_match THEN chat_at ELSE '' END,
                   CASE WHEN workspace_match THEN workspace_at ELSE '' END)
               ELSE coalesce(modified_at, '') END AS sort_time
    FROM events
    WHERE :dated = 0 OR chat_match OR workspace_match OR (:include_unknown = 1 AND unknown_time)
)
"""

_PAGE = (
    _BASE
    + """
SELECT * FROM ordered
WHERE :anchor_group IS NULL OR sort_group > :anchor_group OR
      (sort_group = :anchor_group AND sort_time < :anchor_time) OR
      (sort_group = :anchor_group AND sort_time = :anchor_time AND id > :anchor_id)
ORDER BY sort_group ASC, sort_time DESC, id ASC LIMIT :fetch_limit
"""
)


def query_archive_recovery_page(
    db: CharactersRAGDB,
    request_data: Mapping[str, object],
    workspace_rows: tuple[Mapping[str, object], ...],
    *,
    after: tuple[int, str, str] | None,
    deadline: float,
    monotonic: Callable[[], float],
) -> dict[str, object]:
    """Qualify before limiting, project at most limit+1 rows within a deadline.

    Args:
        db: Owning worker's conversation store.
        request_data: Validated request serialized by the search service.
        workspace_rows: Captured lifecycle inputs, bound through JSON.
        after: Complete previous keyset tuple, never SQL text.
        deadline: Shared absolute monotonic storage deadline.
        monotonic: Deadline clock.

    Returns:
        Raw candidate fields, continuation anchor and undated inventory indicator.

    Raises:
        TimeoutError: The query budget or bounded lock wait is exhausted.
        sqlite3.Error: A storage failure, translated by the service boundary.
    """
    from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

    def check() -> None:
        if monotonic() >= deadline:
            raise TimeoutError("Archive search deadline exceeded")

    check()
    conn = db.get_connection(
        operation_deadline=time.monotonic() + max(0.0, deadline - monotonic())
    )
    check()
    query = str(request_data.get("query", "")).strip()
    pattern = "%" + escape_library_like(query) + "%"
    lower = normalized_archive_time(request_data.get("archived_after"))
    upper = normalized_archive_time(request_data.get("archived_before"))
    limit = int(request_data.get("limit", 10))
    params = {
        "workspaces": json.dumps(workspace_rows, ensure_ascii=False),
        "leading_whitespace": _LEADING_WHITESPACE,
        "workspace_filter": request_data.get("workspace_id"),
        "default_id": DEFAULT_WORKSPACE_ID,
        "query": query,
        "pattern": pattern,
        "lower": lower,
        "upper": upper,
        "dated": bool(lower or upper),
        "include_unknown": request_data.get("include_unknown_archive_times", False),
        "anchor_group": after[0] if after else None,
        "anchor_time": after[1] if after else None,
        "anchor_id": after[2] if after else None,
        "fetch_limit": limit + 1,
    }
    old_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]

    def execute(sql: str, bindings: Mapping[str, object]) -> sqlite3.Cursor:
        check()
        remaining_ms = max(0, int((deadline - monotonic()) * 1000))
        conn.execute(f"PRAGMA busy_timeout = {remaining_ms}")
        return conn.execute(sql, bindings)

    try:
        conn.create_function(
            "archive_recovery_utc", 1, normalized_archive_time, deterministic=True
        )
        conn.create_function(
            "archive_recovery_utc", 2, normalized_archive_time, deterministic=True
        )
        conn.set_progress_handler(lambda: int(monotonic() >= deadline), 100)
        check()
        unknown = bool(
            execute(
                _BASE + "SELECT EXISTS(SELECT 1 FROM events WHERE unknown_time)", params
            ).fetchone()[0]
        )
        rows = execute(_PAGE, params).fetchall()
        items = []
        for row in rows[:limit]:
            check()
            message = execute(
                f"""
                SELECT m.id, substr(m.content,
                    CASE WHEN :query = '' THEN 1 ELSE max(1, instr(lower(m.content), lower(:query)) - 80) END,
                    321) AS excerpt
                FROM messages m WHERE m.conversation_id = :cid AND {_VISIBLE}
                  AND (:query = '' OR m.content LIKE :pattern ESCAPE '\\')
                ORDER BY m.timestamp DESC, m.id ASC LIMIT 1
            """,
                {
                    "cid": row["id"],
                    "query": query,
                    "pattern": pattern,
                    "leading_whitespace": _LEADING_WHITESPACE,
                },
            ).fetchone()
            basis = [kind for kind in ("chat", "workspace") if row[kind + "_match"]]
            if query:
                if row["title_match"]:
                    basis.append("title")
                if row["keyword_match"]:
                    basis.append("keywords")
                if message:
                    basis.append("message")
            if message:
                # The effective pointer falls back to the newest live message,
                # matching the Console's existing active-path convention.
                active = execute(
                    """
                    WITH RECURSIVE path(id, parent_message_id) AS (
                        SELECT id, parent_message_id FROM messages
                        WHERE conversation_id = :cid AND deleted = 0 AND id = coalesce(
                            (SELECT id FROM messages WHERE id = :leaf AND conversation_id = :cid AND deleted = 0),
                            (SELECT id FROM messages WHERE conversation_id = :cid AND deleted = 0 ORDER BY timestamp DESC, id ASC LIMIT 1))
                        UNION
                        SELECT m.id, m.parent_message_id FROM messages m JOIN path p ON m.id = p.parent_message_id
                        WHERE m.conversation_id = :cid AND m.deleted = 0
                    ) SELECT EXISTS(SELECT 1 FROM path WHERE id = :mid)
                """,
                    {
                        "cid": row["id"],
                        "leaf": row["active_leaf_message_id"],
                        "mid": message["id"],
                    },
                ).fetchone()[0]
                if not active:
                    basis.append("inactive_branch")
            excerpt = message["excerpt"] if message else ""
            if len(excerpt) > 320:
                excerpt = excerpt[:319] + "…"
            title = row["title"] or "Untitled chat"
            if len(title) > 1024:
                title = title[:1023] + "…"
            items.append(
                {
                    "conversation_id": row["id"],
                    "conversation_version": row["version"],
                    "title": title,
                    "conversation_archived": bool(row["archived"]),
                    "workspace_id": row["workspace_id"]
                    if row["scope_type"] == "workspace"
                    else None,
                    "conversation_archived_at": row["chat_at"],
                    "excerpt": excerpt,
                    "match_basis": tuple(basis),
                    "sort_key": (row["sort_group"], row["sort_time"], row["id"]),
                    "chronology_incomplete": bool(row["unknown_time"]),
                    "match_message_id": message["id"] if message else None,
                }
            )
        check()
        return {
            "items": items,
            "next_anchor": items[-1]["sort_key"] if len(rows) > limit else None,
            "unknown_times_present": unknown,
        }
    except sqlite3.OperationalError as exc:
        if monotonic() >= deadline or any(
            word in str(exc).lower() for word in ("interrupt", "locked", "busy")
        ):
            raise TimeoutError("Archive search incomplete") from exc
        raise
    finally:
        conn.set_progress_handler(None, 0)
        conn.create_function("archive_recovery_utc", 1, None)
        conn.create_function("archive_recovery_utc", 2, None)
        conn.execute(f"PRAGMA busy_timeout = {old_timeout}")
