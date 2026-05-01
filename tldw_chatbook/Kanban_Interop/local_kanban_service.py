"""Local/offline Kanban service foundation."""

from __future__ import annotations

import uuid
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    KanbanBoardCreate,
    KanbanBoardUpdate,
    KanbanCardCopyRequest,
    KanbanCardCreate,
    KanbanCardMoveRequest,
    KanbanCardUpdate,
    KanbanListCreate,
    KanbanListUpdate,
    KanbanReorderRequest,
)
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

    @classmethod
    def _local_action_id(cls, operation_name: str) -> str:
        action_id = cls.operations[operation_name].action_id
        if action_id.endswith(".server"):
            return f"{action_id[:-len('.server')]}.local"
        return action_id

    @staticmethod
    def _json_dump(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _json_load(value: str | None) -> Any:
        if value is None:
            return None
        return json.loads(value)

    @staticmethod
    def _iso(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _bool(value: Any) -> bool:
        return bool(int(value or 0))

    @staticmethod
    def _row(conn: Any, query: str, params: tuple[Any, ...], *, entity: str, entity_id: int) -> Any:
        row = conn.execute(query, params).fetchone()
        if row is None:
            raise ValueError(f"local_kanban_not_found:{entity}:{entity_id}")
        return row

    @staticmethod
    def _check_version(entity: str, row: Any, expected_version: int | None) -> None:
        if expected_version is not None and int(row["version"]) != expected_version:
            raise ValueError(f"local_kanban_version_conflict:{entity}:{row['id']}")

    def _board_row_to_dict(self, conn: Any, row: Any) -> dict[str, Any]:
        list_count = conn.execute(
            "SELECT COUNT(*) FROM kanban_lists WHERE board_id = ? AND is_deleted = 0",
            (row["id"],),
        ).fetchone()[0]
        card_count = conn.execute(
            "SELECT COUNT(*) FROM kanban_cards WHERE board_id = ? AND is_deleted = 0",
            (row["id"],),
        ).fetchone()[0]
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "user_id": row["user_id"],
            "client_id": row["client_id"],
            "name": row["name"],
            "description": row["description"],
            "archived": self._bool(row["is_archived"]),
            "archived_at": row["archived_at"],
            "activity_retention_days": row["activity_retention_days"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted": self._bool(row["is_deleted"]),
            "deleted_at": row["deleted_at"],
            "version": row["version"],
            "metadata": self._json_load(row["metadata_json"]),
            "list_count": list_count,
            "card_count": card_count,
        }

    def _list_row_to_dict(self, conn: Any, row: Any) -> dict[str, Any]:
        card_count = conn.execute(
            "SELECT COUNT(*) FROM kanban_cards WHERE list_id = ? AND is_deleted = 0",
            (row["id"],),
        ).fetchone()[0]
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "board_id": row["board_id"],
            "client_id": row["client_id"],
            "name": row["name"],
            "position": int(row["position"]),
            "archived": self._bool(row["is_archived"]),
            "archived_at": row["archived_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted": self._bool(row["is_deleted"]),
            "deleted_at": row["deleted_at"],
            "version": row["version"],
            "card_count": card_count,
        }

    def _card_row_to_dict(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "board_id": row["board_id"],
            "list_id": row["list_id"],
            "client_id": row["client_id"],
            "title": row["title"],
            "description": row["description"],
            "position": int(row["position"]),
            "due_date": row["due_date"],
            "due_complete": self._bool(row["due_complete"]),
            "start_date": row["start_date"],
            "priority": row["priority"],
            "archived": self._bool(row["is_archived"]),
            "archived_at": row["archived_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted": self._bool(row["is_deleted"]),
            "deleted_at": row["deleted_at"],
            "version": row["version"],
            "metadata": self._json_load(row["metadata_json"]),
        }

    def _activity_row_to_dict(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "board_id": row["board_id"],
            "list_id": row["list_id"],
            "card_id": row["card_id"],
            "user_id": "local",
            "action_type": row["action_type"],
            "entity_type": row["entity_type"],
            "entity_id": row["entity_id"],
            "details": self._json_load(row["details_json"]),
            "created_at": row["created_at"],
        }

    def _record_activity(
        self,
        conn: Any,
        *,
        board_id: int,
        action_type: str,
        entity_type: str,
        entity_id: int | None,
        list_id: int | None = None,
        card_id: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO kanban_activities
                (uuid, board_id, list_id, card_id, entity_type, entity_id, action_type, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._new_uuid(),
                board_id,
                list_id,
                card_id,
                entity_type,
                entity_id,
                action_type,
                self._json_dump(details),
                self._now(),
            ),
        )

    def _next_position(self, conn: Any, table: str, scope_column: str, scope_id: int) -> int:
        value = conn.execute(
            f"SELECT COALESCE(MAX(position), -1) + 1 FROM {table} WHERE {scope_column} = ? AND is_deleted = 0",
            (scope_id,),
        ).fetchone()[0]
        return int(value)

    async def create_board(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("create_board"))
        request = KanbanBoardCreate(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO kanban_boards
                    (uuid, client_id, name, description, activity_retention_days, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._new_uuid(),
                    request.client_id,
                    request.name,
                    request.description,
                    request.activity_retention_days,
                    self._json_dump(request.metadata),
                    now,
                    now,
                ),
            )
            board_id = int(cursor.lastrowid)
            self._record_activity(conn, board_id=board_id, action_type="create", entity_type="board", entity_id=board_id)
            row = self._row(
                conn,
                "SELECT * FROM kanban_boards WHERE id = ?",
                (board_id,),
                entity="board",
                entity_id=board_id,
            )
            return self._board_row_to_dict(conn, row)

    async def list_boards(
        self,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_boards"))
        where = ["1 = 1"]
        if not include_archived:
            where.append("is_archived = 0")
        if not include_deleted:
            where.append("is_deleted = 0")
        where_sql = " AND ".join(where)
        conn = self.connect()
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM kanban_boards WHERE {where_sql}").fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM kanban_boards WHERE {where_sql} ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return {
                "boards": [self._board_row_to_dict(conn, row) for row in rows],
                "pagination": self._pagination(limit=limit, offset=offset, total=total),
            }
        finally:
            conn.close()

    async def get_board(self, board_id: int, **_: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("get_board"))
        conn = self.connect()
        try:
            row = self._row(
                conn,
                "SELECT * FROM kanban_boards WHERE id = ?",
                (board_id,),
                entity="board",
                entity_id=board_id,
            )
            return self._board_row_to_dict(conn, row)
        finally:
            conn.close()

    async def update_board(
        self,
        board_id: int,
        request_data: Any,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        self._enforce(self._local_action_id("update_board"))
        request = KanbanBoardUpdate(**dict(request_data or {}))
        with self.transaction() as conn:
            row = self._row(
                conn,
                "SELECT * FROM kanban_boards WHERE id = ?",
                (board_id,),
                entity="board",
                entity_id=board_id,
            )
            self._check_version("board", row, expected_version)
            conn.execute(
                """
                UPDATE kanban_boards
                SET name = COALESCE(?, name),
                    description = COALESCE(?, description),
                    activity_retention_days = COALESCE(?, activity_retention_days),
                    metadata_json = COALESCE(?, metadata_json),
                    updated_at = ?,
                    version = version + 1
                WHERE id = ?
                """,
                (
                    request.name,
                    request.description,
                    request.activity_retention_days,
                    self._json_dump(request.metadata),
                    self._now(),
                    board_id,
                ),
            )
            self._record_activity(conn, board_id=board_id, action_type="update", entity_type="board", entity_id=board_id)
            return self._board_row_to_dict(
                conn,
                self._row(conn, "SELECT * FROM kanban_boards WHERE id = ?", (board_id,), entity="board", entity_id=board_id),
            )

    async def archive_board(self, board_id: int) -> dict[str, Any]:
        return await self._set_board_flags(board_id, archived=True, deleted=None, action_type="archive")

    async def unarchive_board(self, board_id: int) -> dict[str, Any]:
        return await self._set_board_flags(board_id, archived=False, deleted=None, action_type="restore")

    async def delete_board(self, board_id: int) -> dict[str, Any]:
        return await self._set_board_flags(board_id, archived=None, deleted=True, action_type="delete")

    async def restore_board(self, board_id: int) -> dict[str, Any]:
        return await self._set_board_flags(board_id, archived=None, deleted=False, action_type="restore")

    async def _set_board_flags(
        self,
        board_id: int,
        *,
        archived: bool | None,
        deleted: bool | None,
        action_type: str,
    ) -> dict[str, Any]:
        operation_name = {
            ("archive", True, None): "archive_board",
            ("restore", False, None): "unarchive_board",
            ("delete", None, True): "delete_board",
            ("restore", None, False): "restore_board",
        }[(action_type, archived, deleted)]
        self._enforce(self._local_action_id(operation_name))
        now = self._now()
        with self.transaction() as conn:
            self._row(conn, "SELECT * FROM kanban_boards WHERE id = ?", (board_id,), entity="board", entity_id=board_id)
            updates = ["updated_at = ?", "version = version + 1"]
            params: list[Any] = [now]
            if archived is not None:
                updates.extend(["is_archived = ?", "archived_at = ?"])
                params.extend([1 if archived else 0, now if archived else None])
            if deleted is not None:
                updates.extend(["is_deleted = ?", "deleted_at = ?"])
                params.extend([1 if deleted else 0, now if deleted else None])
            params.append(board_id)
            conn.execute(f"UPDATE kanban_boards SET {', '.join(updates)} WHERE id = ?", tuple(params))
            self._record_activity(conn, board_id=board_id, action_type=action_type, entity_type="board", entity_id=board_id)
            return self._board_row_to_dict(
                conn,
                self._row(conn, "SELECT * FROM kanban_boards WHERE id = ?", (board_id,), entity="board", entity_id=board_id),
            )

    async def create_list(self, board_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("create_list"))
        request = KanbanListCreate(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            self._row(conn, "SELECT * FROM kanban_boards WHERE id = ?", (board_id,), entity="board", entity_id=board_id)
            position = request.position if request.position is not None else self._next_position(conn, "kanban_lists", "board_id", board_id)
            cursor = conn.execute(
                """
                INSERT INTO kanban_lists (uuid, board_id, client_id, name, position, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (self._new_uuid(), board_id, request.client_id, request.name, position, now, now),
            )
            list_id = int(cursor.lastrowid)
            self._record_activity(conn, board_id=board_id, list_id=list_id, action_type="create", entity_type="list", entity_id=list_id)
            return self._list_row_to_dict(
                conn,
                self._row(conn, "SELECT * FROM kanban_lists WHERE id = ?", (list_id,), entity="list", entity_id=list_id),
            )

    async def list_lists(self, board_id: int, *, include_archived: bool = False, include_deleted: bool = False) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_lists"))
        where = ["board_id = ?"]
        if not include_archived:
            where.append("is_archived = 0")
        if not include_deleted:
            where.append("is_deleted = 0")
        conn = self.connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM kanban_lists WHERE {' AND '.join(where)} ORDER BY position ASC, id ASC",
                (board_id,),
            ).fetchall()
            return {"lists": [self._list_row_to_dict(conn, row) for row in rows]}
        finally:
            conn.close()

    async def get_list(self, list_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("get_list"))
        conn = self.connect()
        try:
            return self._list_row_to_dict(
                conn,
                self._row(conn, "SELECT * FROM kanban_lists WHERE id = ?", (list_id,), entity="list", entity_id=list_id),
            )
        finally:
            conn.close()

    async def update_list(
        self,
        list_id: int,
        request_data: Any,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        self._enforce(self._local_action_id("update_list"))
        request = KanbanListUpdate(**dict(request_data or {}))
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_lists WHERE id = ?", (list_id,), entity="list", entity_id=list_id)
            self._check_version("list", row, expected_version)
            conn.execute(
                """
                UPDATE kanban_lists
                SET name = COALESCE(?, name),
                    position = COALESCE(?, position),
                    updated_at = ?,
                    version = version + 1
                WHERE id = ?
                """,
                (request.name, request.position, self._now(), list_id),
            )
            self._record_activity(conn, board_id=row["board_id"], list_id=list_id, action_type="update", entity_type="list", entity_id=list_id)
            return self._list_row_to_dict(
                conn,
                self._row(conn, "SELECT * FROM kanban_lists WHERE id = ?", (list_id,), entity="list", entity_id=list_id),
            )

    async def archive_list(self, list_id: int) -> dict[str, Any]:
        return await self._set_list_flags(list_id, archived=True, deleted=None, action_type="archive")

    async def unarchive_list(self, list_id: int) -> dict[str, Any]:
        return await self._set_list_flags(list_id, archived=False, deleted=None, action_type="restore")

    async def delete_list(self, list_id: int) -> dict[str, Any]:
        return await self._set_list_flags(list_id, archived=None, deleted=True, action_type="delete")

    async def restore_list(self, list_id: int) -> dict[str, Any]:
        return await self._set_list_flags(list_id, archived=None, deleted=False, action_type="restore")

    async def _set_list_flags(
        self,
        list_id: int,
        *,
        archived: bool | None,
        deleted: bool | None,
        action_type: str,
    ) -> dict[str, Any]:
        operation_name = {
            ("archive", True, None): "archive_list",
            ("restore", False, None): "unarchive_list",
            ("delete", None, True): "delete_list",
            ("restore", None, False): "restore_list",
        }[(action_type, archived, deleted)]
        self._enforce(self._local_action_id(operation_name))
        now = self._now()
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_lists WHERE id = ?", (list_id,), entity="list", entity_id=list_id)
            updates = ["updated_at = ?", "version = version + 1"]
            params: list[Any] = [now]
            if archived is not None:
                updates.extend(["is_archived = ?", "archived_at = ?"])
                params.extend([1 if archived else 0, now if archived else None])
            if deleted is not None:
                updates.extend(["is_deleted = ?", "deleted_at = ?"])
                params.extend([1 if deleted else 0, now if deleted else None])
            params.append(list_id)
            conn.execute(f"UPDATE kanban_lists SET {', '.join(updates)} WHERE id = ?", tuple(params))
            self._record_activity(conn, board_id=row["board_id"], list_id=list_id, action_type=action_type, entity_type="list", entity_id=list_id)
            return self._list_row_to_dict(
                conn,
                self._row(conn, "SELECT * FROM kanban_lists WHERE id = ?", (list_id,), entity="list", entity_id=list_id),
            )

    async def reorder_lists(self, board_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("reorder_lists"))
        request = KanbanReorderRequest(**dict(request_data or {}))
        with self.transaction() as conn:
            self._row(conn, "SELECT * FROM kanban_boards WHERE id = ?", (board_id,), entity="board", entity_id=board_id)
            for position, list_id in enumerate(request.ids or []):
                row = self._row(conn, "SELECT * FROM kanban_lists WHERE id = ?", (list_id,), entity="list", entity_id=list_id)
                if row["board_id"] != board_id:
                    raise ValueError(f"local_kanban_invalid_scope:list:{list_id}")
                conn.execute(
                    "UPDATE kanban_lists SET position = ?, updated_at = ?, version = version + 1 WHERE id = ?",
                    (position, self._now(), list_id),
                )
            self._record_activity(conn, board_id=board_id, action_type="reorder", entity_type="list", entity_id=None, details={"ids": request.ids})
            return {"success": True, "message": None}

    async def create_card(self, list_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("create_card"))
        request = KanbanCardCreate(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            list_row = self._row(conn, "SELECT * FROM kanban_lists WHERE id = ?", (list_id,), entity="list", entity_id=list_id)
            position = request.position if request.position is not None else self._next_position(conn, "kanban_cards", "list_id", list_id)
            cursor = conn.execute(
                """
                INSERT INTO kanban_cards
                    (uuid, board_id, list_id, client_id, title, description, position, due_date, start_date, priority, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._new_uuid(),
                    list_row["board_id"],
                    list_id,
                    request.client_id,
                    request.title,
                    request.description,
                    position,
                    self._iso(request.due_date),
                    self._iso(request.start_date),
                    request.priority,
                    self._json_dump(request.metadata),
                    now,
                    now,
                ),
            )
            card_id = int(cursor.lastrowid)
            self._record_activity(
                conn,
                board_id=list_row["board_id"],
                list_id=list_id,
                card_id=card_id,
                action_type="create",
                entity_type="card",
                entity_id=card_id,
            )
            return self._card_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            )

    async def list_cards(self, list_id: int, *, include_archived: bool = False, include_deleted: bool = False) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_cards"))
        where = ["list_id = ?"]
        if not include_archived:
            where.append("is_archived = 0")
        if not include_deleted:
            where.append("is_deleted = 0")
        conn = self.connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM kanban_cards WHERE {' AND '.join(where)} ORDER BY position ASC, id ASC",
                (list_id,),
            ).fetchall()
            return {"cards": [self._card_row_to_dict(row) for row in rows]}
        finally:
            conn.close()

    async def get_card(self, card_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("get_card"))
        conn = self.connect()
        try:
            return self._card_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            )
        finally:
            conn.close()

    async def update_card(
        self,
        card_id: int,
        request_data: Any,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        self._enforce(self._local_action_id("update_card"))
        request = KanbanCardUpdate(**dict(request_data or {}))
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            self._check_version("card", row, expected_version)
            conn.execute(
                """
                UPDATE kanban_cards
                SET title = COALESCE(?, title),
                    description = COALESCE(?, description),
                    due_date = COALESCE(?, due_date),
                    due_complete = COALESCE(?, due_complete),
                    start_date = COALESCE(?, start_date),
                    priority = COALESCE(?, priority),
                    metadata_json = COALESCE(?, metadata_json),
                    updated_at = ?,
                    version = version + 1
                WHERE id = ?
                """,
                (
                    request.title,
                    request.description,
                    self._iso(request.due_date),
                    None if request.due_complete is None else int(request.due_complete),
                    self._iso(request.start_date),
                    request.priority,
                    self._json_dump(request.metadata),
                    self._now(),
                    card_id,
                ),
            )
            self._record_activity(
                conn,
                board_id=row["board_id"],
                list_id=row["list_id"],
                card_id=card_id,
                action_type="update",
                entity_type="card",
                entity_id=card_id,
            )
            return self._card_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            )

    async def move_card(self, card_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("move_card"))
        request = KanbanCardMoveRequest(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            target_list = self._row(
                conn,
                "SELECT * FROM kanban_lists WHERE id = ?",
                (request.target_list_id,),
                entity="list",
                entity_id=request.target_list_id,
            )
            position = request.position if request.position is not None else self._next_position(conn, "kanban_cards", "list_id", target_list["id"])
            conn.execute(
                """
                UPDATE kanban_cards
                SET board_id = ?, list_id = ?, position = ?, updated_at = ?, version = version + 1
                WHERE id = ?
                """,
                (target_list["board_id"], target_list["id"], position, now, card_id),
            )
            self._record_activity(
                conn,
                board_id=target_list["board_id"],
                list_id=target_list["id"],
                card_id=card_id,
                action_type="move",
                entity_type="card",
                entity_id=card_id,
                details={"from_list_id": card["list_id"], "to_list_id": target_list["id"]},
            )
            return self._card_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            )

    async def copy_card(self, card_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("copy_card"))
        request = KanbanCardCopyRequest(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            target_list = self._row(
                conn,
                "SELECT * FROM kanban_lists WHERE id = ?",
                (request.target_list_id,),
                entity="list",
                entity_id=request.target_list_id,
            )
            position = request.position if request.position is not None else self._next_position(conn, "kanban_cards", "list_id", target_list["id"])
            cursor = conn.execute(
                """
                INSERT INTO kanban_cards
                    (uuid, board_id, list_id, client_id, title, description, position, due_date, due_complete, start_date, priority, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._new_uuid(),
                    target_list["board_id"],
                    target_list["id"],
                    request.new_client_id,
                    request.new_title or card["title"],
                    card["description"],
                    position,
                    card["due_date"],
                    card["due_complete"],
                    card["start_date"],
                    card["priority"],
                    card["metadata_json"],
                    now,
                    now,
                ),
            )
            copied_id = int(cursor.lastrowid)
            self._record_activity(
                conn,
                board_id=target_list["board_id"],
                list_id=target_list["id"],
                card_id=copied_id,
                action_type="copy",
                entity_type="card",
                entity_id=copied_id,
                details={"source_card_id": card_id},
            )
            return self._card_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (copied_id,), entity="card", entity_id=copied_id)
            )

    async def archive_card(self, card_id: int) -> dict[str, Any]:
        return await self._set_card_flags(card_id, archived=True, deleted=None, action_type="archive")

    async def unarchive_card(self, card_id: int) -> dict[str, Any]:
        return await self._set_card_flags(card_id, archived=False, deleted=None, action_type="restore")

    async def delete_card(self, card_id: int) -> dict[str, Any]:
        return await self._set_card_flags(card_id, archived=None, deleted=True, action_type="delete")

    async def restore_card(self, card_id: int) -> dict[str, Any]:
        return await self._set_card_flags(card_id, archived=None, deleted=False, action_type="restore")

    async def _set_card_flags(
        self,
        card_id: int,
        *,
        archived: bool | None,
        deleted: bool | None,
        action_type: str,
    ) -> dict[str, Any]:
        operation_name = {
            ("archive", True, None): "archive_card",
            ("restore", False, None): "unarchive_card",
            ("delete", None, True): "delete_card",
            ("restore", None, False): "restore_card",
        }[(action_type, archived, deleted)]
        self._enforce(self._local_action_id(operation_name))
        now = self._now()
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            updates = ["updated_at = ?", "version = version + 1"]
            params: list[Any] = [now]
            if archived is not None:
                updates.extend(["is_archived = ?", "archived_at = ?"])
                params.extend([1 if archived else 0, now if archived else None])
            if deleted is not None:
                updates.extend(["is_deleted = ?", "deleted_at = ?"])
                params.extend([1 if deleted else 0, now if deleted else None])
            params.append(card_id)
            conn.execute(f"UPDATE kanban_cards SET {', '.join(updates)} WHERE id = ?", tuple(params))
            self._record_activity(
                conn,
                board_id=row["board_id"],
                list_id=row["list_id"],
                card_id=card_id,
                action_type=action_type,
                entity_type="card",
                entity_id=card_id,
            )
            return self._card_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            )

    async def reorder_cards(self, list_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("reorder_cards"))
        request = KanbanReorderRequest(**dict(request_data or {}))
        with self.transaction() as conn:
            list_row = self._row(conn, "SELECT * FROM kanban_lists WHERE id = ?", (list_id,), entity="list", entity_id=list_id)
            for position, card_id in enumerate(request.ids or []):
                row = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
                if row["list_id"] != list_id:
                    raise ValueError(f"local_kanban_invalid_scope:card:{card_id}")
                conn.execute(
                    "UPDATE kanban_cards SET position = ?, updated_at = ?, version = version + 1 WHERE id = ?",
                    (position, self._now(), card_id),
                )
            self._record_activity(conn, board_id=list_row["board_id"], list_id=list_id, action_type="reorder", entity_type="card", entity_id=None, details={"ids": request.ids})
            return {"success": True, "message": None}

    async def list_board_activities(self, board_id: int, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_board_activities"))
        conn = self.connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM kanban_activities WHERE board_id = ?", (board_id,)).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM kanban_activities WHERE board_id = ? ORDER BY id ASC LIMIT ? OFFSET ?",
                (board_id, limit, offset),
            ).fetchall()
            return {
                "activities": [self._activity_row_to_dict(row) for row in rows],
                "pagination": self._pagination(limit=limit, offset=offset, total=total),
            }
        finally:
            conn.close()

    async def list_card_activities(self, card_id: int, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_card_activities"))
        conn = self.connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM kanban_activities WHERE card_id = ?", (card_id,)).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM kanban_activities WHERE card_id = ? ORDER BY id ASC LIMIT ? OFFSET ?",
                (card_id, limit, offset),
            ).fetchall()
            return {
                "activities": [self._activity_row_to_dict(row) for row in rows],
                "pagination": self._pagination(limit=limit, offset=offset, total=total),
            }
        finally:
            conn.close()
