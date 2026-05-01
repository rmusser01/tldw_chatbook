"""Local/offline Kanban service foundation."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..runtime_policy.types import PolicyDeniedError
from .local_kanban_db import initialize_schema, open_connection, transaction
from .server_kanban_service import KANBAN_OPERATION_SPECS


class LocalKanbanService:
    """SQLite-backed local Kanban backend.

    Operation methods are added in behavior slices. This foundation owns the
    durable local schema, transactions, identity helpers, and policy seam.
    """

    operations = KANBAN_OPERATION_SPECS

    def __init__(self, *, db_path: str | Path, policy_enforcer: Any | None = None) -> None:
        self.db_path = Path(db_path) if str(db_path) != ":memory:" else db_path
        self.policy_enforcer = policy_enforcer
        conn = self.connect()
        try:
            initialize_schema(conn)
        finally:
            conn.close()

    def connect(self):
        return open_connection(self.db_path)

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        conn = self.connect()
        try:
            with transaction(conn) as tx:
                yield tx
        finally:
            conn.close()

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Local Kanban action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "local",
                    authority_owner=getattr(decision, "authority_owner", None) or "local",
                )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_uuid() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _pagination(*, limit: int, offset: int, total: int) -> dict[str, Any]:
        next_offset = offset + limit
        has_more = next_offset < total
        return {
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": has_more,
            "next_offset": next_offset if has_more else None,
        }

    def get_storage_status(self) -> dict[str, Any]:
        conn = self.connect()
        try:
            rows = {
                row["key"]: row["value"]
                for row in conn.execute("SELECT key, value FROM local_kanban_schema_meta").fetchall()
            }
            return {
                "schema_version": int(rows.get("schema_version", "0")),
                "fts_available": rows.get("fts_available") == "1",
                "db_path": str(self.db_path),
            }
        finally:
            conn.close()
