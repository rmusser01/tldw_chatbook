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
    KanbanBoardExportRequest,
    KanbanBoardImportRequest,
    KanbanBoardUpdate,
    KanbanBulkArchiveCardsRequest,
    KanbanBulkCardLinksRequest,
    KanbanBulkDeleteCardsRequest,
    KanbanBulkLabelCardsRequest,
    KanbanBulkMoveCardsRequest,
    KanbanCardCopyRequest,
    KanbanCardCopyWithChecklistsRequest,
    KanbanCardCreate,
    KanbanCardLinkCreate,
    KanbanCardSearchRequest,
    KanbanCardMoveRequest,
    KanbanCardUpdate,
    KanbanChecklistCreate,
    KanbanChecklistItemCreate,
    KanbanChecklistItemReorderRequest,
    KanbanChecklistItemUpdate,
    KanbanChecklistReorderRequest,
    KanbanChecklistUpdate,
    KanbanCommentCreate,
    KanbanCommentUpdate,
    KanbanLabelCreate,
    KanbanLabelUpdate,
    KanbanListCreate,
    KanbanListUpdate,
    KanbanReorderRequest,
    KanbanSearchRequest,
    KanbanToggleAllChecklistItemsRequest,
)
from .local_kanban_db import initialize_schema, open_connection, transaction
from .server_kanban_service import KANBAN_OPERATION_SPECS


_SERVER_ACTION_SUFFIX = ".server"
_LOCAL_ACTION_SUFFIX = ".local"


def _derive_local_action_id(action_id: str) -> str:
    if not action_id.endswith(_SERVER_ACTION_SUFFIX):
        raise ValueError(f"Kanban operation action_id must end with {_SERVER_ACTION_SUFFIX!r}: {action_id}")
    return f"{action_id[:-len(_SERVER_ACTION_SUFFIX)]}{_LOCAL_ACTION_SUFFIX}"


LOCAL_KANBAN_OPERATION_ACTION_IDS = {
    name: _derive_local_action_id(spec.action_id) for name, spec in KANBAN_OPERATION_SPECS.items()
}


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
        try:
            return LOCAL_KANBAN_OPERATION_ACTION_IDS[operation_name]
        except KeyError as exc:
            raise ValueError(f"Unknown Kanban operation: {operation_name}") from exc

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

    def _label_row_to_dict(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "board_id": row["board_id"],
            "name": row["name"],
            "color": row["color"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _checklist_row_to_dict(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "card_id": row["card_id"],
            "name": row["name"],
            "position": int(row["position"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _checklist_item_row_to_dict(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "checklist_id": row["checklist_id"],
            "name": row["name"],
            "position": int(row["position"]),
            "checked": self._bool(row["checked"]),
            "checked_at": row["checked_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _comment_row_to_dict(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "card_id": row["card_id"],
            "user_id": row["user_id"],
            "content": row["content"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted": self._bool(row["is_deleted"]),
        }

    def _link_row_to_dict(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "card_id": row["card_id"],
            "linked_type": row["linked_type"],
            "linked_id": row["linked_id"],
            "created_at": row["created_at"],
        }

    async def get_board_export(self, board_id: int, **kwargs: Any) -> dict[str, Any]:
        return await self.export_board(board_id, kwargs)

    async def export_board(self, board_id: int, request_data: Any | None = None) -> dict[str, Any]:
        self._enforce(self._local_action_id("export_board"))
        request = KanbanBoardExportRequest(**dict(request_data or {}))
        conn = self.connect()
        try:
            board = self._board_row_to_dict(
                conn,
                self._row(conn, "SELECT * FROM kanban_boards WHERE id = ?", (board_id,), entity="board", entity_id=board_id),
            )
            labels = [
                self._label_row_to_dict(row)
                for row in conn.execute("SELECT * FROM kanban_labels WHERE board_id = ? ORDER BY id ASC", (board_id,)).fetchall()
            ]
            list_filters = ["board_id = ?"]
            if not request.include_archived:
                list_filters.append("is_archived = 0")
            if not request.include_deleted:
                list_filters.append("is_deleted = 0")
            lists = []
            for list_row in conn.execute(
                f"SELECT * FROM kanban_lists WHERE {' AND '.join(list_filters)} ORDER BY position ASC, id ASC",
                (board_id,),
            ).fetchall():
                list_payload = self._list_row_to_dict(conn, list_row)
                card_filters = ["list_id = ?"]
                if not request.include_archived:
                    card_filters.append("is_archived = 0")
                if not request.include_deleted:
                    card_filters.append("is_deleted = 0")
                cards = []
                for card_row in conn.execute(
                    f"SELECT * FROM kanban_cards WHERE {' AND '.join(card_filters)} ORDER BY position ASC, id ASC",
                    (list_row["id"],),
                ).fetchall():
                    card_payload = self._card_row_to_dict(card_row)
                    card_payload["labels"] = (await self.list_card_labels(card_row["id"]))["labels"]
                    card_payload["checklists"] = self._checklists_for_card(conn, card_row["id"])
                    card_payload["comments"] = (await self.list_comments(card_row["id"], include_deleted=request.include_deleted))["comments"]
                    card_payload["links"] = (await self.list_card_links(card_row["id"]))["links"]
                    cards.append(card_payload)
                list_payload["cards"] = cards
                lists.append(list_payload)
            return {
                "format": "json",
                "exported_at": self._now(),
                "board": board,
                "labels": labels,
                "lists": lists,
            }
        finally:
            conn.close()

    async def import_board(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("import_board"))
        request = KanbanBoardImportRequest(**dict(request_data or {}))
        data = request.data
        source_board = data.get("board") or {}
        board = await self.create_board(
            {
                "name": request.board_name or source_board.get("name") or "Imported Board",
                "client_id": source_board.get("client_id") or f"import-{self._new_uuid()}",
                "description": source_board.get("description"),
                "activity_retention_days": source_board.get("activity_retention_days"),
                "metadata": source_board.get("metadata"),
            }
        )
        label_map: dict[int, int] = {}
        for label_payload in data.get("labels") or []:
            created = await self.create_label(
                board["id"],
                {"name": label_payload["name"], "color": label_payload.get("color") or "gray"},
            )
            label_map[int(label_payload["id"])] = created["id"]
        lists_imported = cards_imported = checklists_imported = checklist_items_imported = comments_imported = 0
        for list_payload in data.get("lists") or []:
            created_list = await self.create_list(
                board["id"],
                {
                    "name": list_payload["name"],
                    "client_id": list_payload.get("client_id") or f"import-list-{self._new_uuid()}",
                    "position": list_payload.get("position"),
                },
            )
            lists_imported += 1
            for card_payload in list_payload.get("cards") or []:
                created_card = await self.create_card(
                    created_list["id"],
                    {
                        "title": card_payload["title"],
                        "client_id": card_payload.get("client_id") or f"import-card-{self._new_uuid()}",
                        "description": card_payload.get("description"),
                        "position": card_payload.get("position"),
                        "priority": card_payload.get("priority"),
                        "metadata": card_payload.get("metadata"),
                    },
                )
                cards_imported += 1
                for label_payload in card_payload.get("labels") or []:
                    mapped_label_id = label_map.get(int(label_payload["id"]))
                    if mapped_label_id is not None:
                        await self.assign_label_to_card(created_card["id"], mapped_label_id)
                for checklist_payload in card_payload.get("checklists") or []:
                    created_checklist = await self.create_checklist(
                        created_card["id"],
                        {"name": checklist_payload["name"], "position": checklist_payload.get("position")},
                    )
                    checklists_imported += 1
                    for item_payload in checklist_payload.get("items") or []:
                        await self.create_checklist_item(
                            created_checklist["id"],
                            {
                                "name": item_payload["name"],
                                "position": item_payload.get("position"),
                                "checked": item_payload.get("checked", False),
                            },
                        )
                        checklist_items_imported += 1
                for comment_payload in card_payload.get("comments") or []:
                    if not comment_payload.get("deleted"):
                        await self.create_comment(created_card["id"], {"content": comment_payload["content"]})
                        comments_imported += 1
        return {
            "board": board,
            "import_stats": {
                "board_id": board["id"],
                "lists_imported": lists_imported,
                "cards_imported": cards_imported,
                "labels_imported": len(label_map),
                "checklists_imported": checklists_imported,
                "checklist_items_imported": checklist_items_imported,
                "comments_imported": comments_imported,
            },
        }

    async def create_label(self, board_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("create_label"))
        request = KanbanLabelCreate(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            self._row(conn, "SELECT * FROM kanban_boards WHERE id = ?", (board_id,), entity="board", entity_id=board_id)
            cursor = conn.execute(
                "INSERT INTO kanban_labels (uuid, board_id, name, color, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (self._new_uuid(), board_id, request.name, request.color, now, now),
            )
            label_id = int(cursor.lastrowid)
            self._record_activity(conn, board_id=board_id, action_type="label", entity_type="label", entity_id=label_id)
            return self._label_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_labels WHERE id = ?", (label_id,), entity="label", entity_id=label_id)
            )

    async def list_labels(self, board_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_labels"))
        conn = self.connect()
        try:
            rows = conn.execute("SELECT * FROM kanban_labels WHERE board_id = ? ORDER BY id ASC", (board_id,)).fetchall()
            return {"labels": [self._label_row_to_dict(row) for row in rows]}
        finally:
            conn.close()

    async def get_label(self, label_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("get_label"))
        conn = self.connect()
        try:
            return self._label_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_labels WHERE id = ?", (label_id,), entity="label", entity_id=label_id)
            )
        finally:
            conn.close()

    async def update_label(self, label_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("update_label"))
        request = KanbanLabelUpdate(**dict(request_data or {}))
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_labels WHERE id = ?", (label_id,), entity="label", entity_id=label_id)
            conn.execute(
                "UPDATE kanban_labels SET name = COALESCE(?, name), color = COALESCE(?, color), updated_at = ?, version = version + 1 WHERE id = ?",
                (request.name, request.color, self._now(), label_id),
            )
            self._record_activity(conn, board_id=row["board_id"], action_type="label", entity_type="label", entity_id=label_id)
            return self._label_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_labels WHERE id = ?", (label_id,), entity="label", entity_id=label_id)
            )

    async def delete_label(self, label_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("delete_label"))
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_labels WHERE id = ?", (label_id,), entity="label", entity_id=label_id)
            conn.execute("DELETE FROM kanban_labels WHERE id = ?", (label_id,))
            self._record_activity(conn, board_id=row["board_id"], action_type="label", entity_type="label", entity_id=label_id, details={"deleted": True})
            return {"detail": "deleted"}

    async def assign_label_to_card(self, card_id: int, label_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("assign_label_to_card"))
        with self.transaction() as conn:
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            self._row(conn, "SELECT * FROM kanban_labels WHERE id = ?", (label_id,), entity="label", entity_id=label_id)
            conn.execute(
                "INSERT OR IGNORE INTO kanban_card_labels (card_id, label_id, created_at) VALUES (?, ?, ?)",
                (card_id, label_id, self._now()),
            )
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card_id, action_type="label", entity_type="card_label", entity_id=label_id)
            return {"detail": "assigned"}

    async def remove_label_from_card(self, card_id: int, label_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("remove_label_from_card"))
        with self.transaction() as conn:
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            conn.execute("DELETE FROM kanban_card_labels WHERE card_id = ? AND label_id = ?", (card_id, label_id))
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card_id, action_type="label", entity_type="card_label", entity_id=label_id, details={"removed": True})
            return {"detail": "removed"}

    async def list_card_labels(self, card_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_card_labels"))
        conn = self.connect()
        try:
            rows = conn.execute(
                """
                SELECT labels.*
                FROM kanban_labels labels
                JOIN kanban_card_labels card_labels ON card_labels.label_id = labels.id
                WHERE card_labels.card_id = ?
                ORDER BY labels.id ASC
                """,
                (card_id,),
            ).fetchall()
            return {"labels": [self._label_row_to_dict(row) for row in rows]}
        finally:
            conn.close()

    async def create_checklist(self, card_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("create_checklist"))
        request = KanbanChecklistCreate(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            position = request.position if request.position is not None else int(conn.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM kanban_checklists WHERE card_id = ?", (card_id,)).fetchone()[0])
            cursor = conn.execute(
                "INSERT INTO kanban_checklists (uuid, card_id, name, position, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (self._new_uuid(), card_id, request.name, position, now, now),
            )
            checklist_id = int(cursor.lastrowid)
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card_id, action_type="create", entity_type="checklist", entity_id=checklist_id)
            return self._checklist_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_checklists WHERE id = ?", (checklist_id,), entity="checklist", entity_id=checklist_id)
            )

    async def list_checklists(self, card_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_checklists"))
        conn = self.connect()
        try:
            rows = conn.execute("SELECT * FROM kanban_checklists WHERE card_id = ? ORDER BY position ASC, id ASC", (card_id,)).fetchall()
            return {"checklists": [self._checklist_row_to_dict(row) for row in rows]}
        finally:
            conn.close()

    async def get_checklist(self, checklist_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("get_checklist"))
        conn = self.connect()
        try:
            return self._checklist_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_checklists WHERE id = ?", (checklist_id,), entity="checklist", entity_id=checklist_id)
            )
        finally:
            conn.close()

    async def update_checklist(self, checklist_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("update_checklist"))
        request = KanbanChecklistUpdate(**dict(request_data or {}))
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_checklists WHERE id = ?", (checklist_id,), entity="checklist", entity_id=checklist_id)
            conn.execute(
                "UPDATE kanban_checklists SET name = COALESCE(?, name), updated_at = ?, version = version + 1 WHERE id = ?",
                (request.name, self._now(), checklist_id),
            )
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (row["card_id"],), entity="card", entity_id=row["card_id"])
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card["id"], action_type="update", entity_type="checklist", entity_id=checklist_id)
            return self._checklist_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_checklists WHERE id = ?", (checklist_id,), entity="checklist", entity_id=checklist_id)
            )

    async def delete_checklist(self, checklist_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("delete_checklist"))
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_checklists WHERE id = ?", (checklist_id,), entity="checklist", entity_id=checklist_id)
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (row["card_id"],), entity="card", entity_id=row["card_id"])
            conn.execute("DELETE FROM kanban_checklists WHERE id = ?", (checklist_id,))
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card["id"], action_type="delete", entity_type="checklist", entity_id=checklist_id)
            return {"detail": "deleted"}

    async def reorder_checklists(self, card_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("reorder_checklists"))
        request = KanbanChecklistReorderRequest(**dict(request_data or {}))
        with self.transaction() as conn:
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            for position, checklist_id in enumerate(request.checklist_ids):
                conn.execute("UPDATE kanban_checklists SET position = ?, updated_at = ? WHERE id = ? AND card_id = ?", (position, self._now(), checklist_id, card_id))
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card_id, action_type="reorder", entity_type="checklist", entity_id=None, details={"ids": request.checklist_ids})
            return {"success": True, "message": None}

    async def create_checklist_item(self, checklist_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("create_checklist_item"))
        request = KanbanChecklistItemCreate(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            checklist = self._row(conn, "SELECT * FROM kanban_checklists WHERE id = ?", (checklist_id,), entity="checklist", entity_id=checklist_id)
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (checklist["card_id"],), entity="card", entity_id=checklist["card_id"])
            position = request.position if request.position is not None else int(conn.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM kanban_checklist_items WHERE checklist_id = ?", (checklist_id,)).fetchone()[0])
            cursor = conn.execute(
                """
                INSERT INTO kanban_checklist_items
                    (uuid, checklist_id, name, checked, checked_at, position, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (self._new_uuid(), checklist_id, request.name, int(request.checked), now if request.checked else None, position, now, now),
            )
            item_id = int(cursor.lastrowid)
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card["id"], action_type="create", entity_type="checklist_item", entity_id=item_id)
            return self._checklist_item_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_checklist_items WHERE id = ?", (item_id,), entity="checklist_item", entity_id=item_id)
            )

    async def list_checklist_items(self, checklist_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_checklist_items"))
        conn = self.connect()
        try:
            rows = conn.execute("SELECT * FROM kanban_checklist_items WHERE checklist_id = ? ORDER BY position ASC, id ASC", (checklist_id,)).fetchall()
            return {"items": [self._checklist_item_row_to_dict(row) for row in rows]}
        finally:
            conn.close()

    async def get_checklist_item(self, item_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("get_checklist_item"))
        conn = self.connect()
        try:
            return self._checklist_item_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_checklist_items WHERE id = ?", (item_id,), entity="checklist_item", entity_id=item_id)
            )
        finally:
            conn.close()

    async def update_checklist_item(self, item_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("update_checklist_item"))
        request = KanbanChecklistItemUpdate(**dict(request_data or {}))
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_checklist_items WHERE id = ?", (item_id,), entity="checklist_item", entity_id=item_id)
            checked_at = self._now() if request.checked else None
            conn.execute(
                """
                UPDATE kanban_checklist_items
                SET name = COALESCE(?, name),
                    checked = COALESCE(?, checked),
                    checked_at = CASE WHEN ? IS NULL THEN checked_at ELSE ? END,
                    updated_at = ?,
                    version = version + 1
                WHERE id = ?
                """,
                (request.name, None if request.checked is None else int(request.checked), request.checked, checked_at, self._now(), item_id),
            )
            checklist = self._row(conn, "SELECT * FROM kanban_checklists WHERE id = ?", (row["checklist_id"],), entity="checklist", entity_id=row["checklist_id"])
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (checklist["card_id"],), entity="card", entity_id=checklist["card_id"])
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card["id"], action_type="update", entity_type="checklist_item", entity_id=item_id)
            return self._checklist_item_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_checklist_items WHERE id = ?", (item_id,), entity="checklist_item", entity_id=item_id)
            )

    async def delete_checklist_item(self, item_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("delete_checklist_item"))
        with self.transaction() as conn:
            conn.execute("DELETE FROM kanban_checklist_items WHERE id = ?", (item_id,))
            return {"detail": "deleted"}

    async def reorder_checklist_items(self, checklist_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("reorder_checklist_items"))
        request = KanbanChecklistItemReorderRequest(**dict(request_data or {}))
        with self.transaction() as conn:
            for position, item_id in enumerate(request.item_ids):
                conn.execute("UPDATE kanban_checklist_items SET position = ?, updated_at = ? WHERE id = ? AND checklist_id = ?", (position, self._now(), item_id, checklist_id))
            return {"success": True, "message": None}

    async def check_checklist_item(self, item_id: int) -> dict[str, Any]:
        return await self.update_checklist_item(item_id, {"checked": True})

    async def uncheck_checklist_item(self, item_id: int) -> dict[str, Any]:
        return await self.update_checklist_item(item_id, {"checked": False})

    async def toggle_all_checklist_items(self, checklist_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("toggle_all_checklist_items"))
        request = KanbanToggleAllChecklistItemsRequest(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            conn.execute(
                "UPDATE kanban_checklist_items SET checked = ?, checked_at = ?, updated_at = ?, version = version + 1 WHERE checklist_id = ?",
                (int(request.checked), now if request.checked else None, now, checklist_id),
            )
            return {"success": True, "message": None}

    def _checklists_for_card(self, conn: Any, card_id: int) -> list[dict[str, Any]]:
        checklists = []
        for checklist_row in conn.execute("SELECT * FROM kanban_checklists WHERE card_id = ? ORDER BY position ASC, id ASC", (card_id,)).fetchall():
            payload = self._checklist_row_to_dict(checklist_row)
            item_rows = conn.execute("SELECT * FROM kanban_checklist_items WHERE checklist_id = ? ORDER BY position ASC, id ASC", (checklist_row["id"],)).fetchall()
            payload["items"] = [self._checklist_item_row_to_dict(item_row) for item_row in item_rows]
            payload["total_items"] = len(payload["items"])
            payload["checked_items"] = sum(1 for item in payload["items"] if item["checked"])
            payload["progress_percent"] = int((payload["checked_items"] / payload["total_items"]) * 100) if payload["total_items"] else 0
            checklists.append(payload)
        return checklists

    async def create_comment(self, card_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("create_comment"))
        request = KanbanCommentCreate(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            cursor = conn.execute(
                "INSERT INTO kanban_comments (uuid, card_id, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (self._new_uuid(), card_id, request.content, now, now),
            )
            comment_id = int(cursor.lastrowid)
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card_id, action_type="comment", entity_type="comment", entity_id=comment_id)
            return self._comment_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_comments WHERE id = ?", (comment_id,), entity="comment", entity_id=comment_id)
            )

    async def list_comments(self, card_id: int, *, include_deleted: bool = False, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_comments"))
        where = ["card_id = ?"]
        if not include_deleted:
            where.append("is_deleted = 0")
        conn = self.connect()
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM kanban_comments WHERE {' AND '.join(where)}", (card_id,)).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM kanban_comments WHERE {' AND '.join(where)} ORDER BY created_at ASC, id ASC LIMIT ? OFFSET ?",
                (card_id, limit, offset),
            ).fetchall()
            return {"comments": [self._comment_row_to_dict(row) for row in rows], "pagination": self._pagination(limit=limit, offset=offset, total=total)}
        finally:
            conn.close()

    async def get_comment(self, comment_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("get_comment"))
        conn = self.connect()
        try:
            return self._comment_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_comments WHERE id = ?", (comment_id,), entity="comment", entity_id=comment_id)
            )
        finally:
            conn.close()

    async def update_comment(self, comment_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("update_comment"))
        request = KanbanCommentUpdate(**dict(request_data or {}))
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_comments WHERE id = ?", (comment_id,), entity="comment", entity_id=comment_id)
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (row["card_id"],), entity="card", entity_id=row["card_id"])
            conn.execute("UPDATE kanban_comments SET content = ?, updated_at = ?, version = version + 1 WHERE id = ?", (request.content, self._now(), comment_id))
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card["id"], action_type="comment", entity_type="comment", entity_id=comment_id)
            return self._comment_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_comments WHERE id = ?", (comment_id,), entity="comment", entity_id=comment_id)
            )

    async def delete_comment(self, comment_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("delete_comment"))
        with self.transaction() as conn:
            row = self._row(conn, "SELECT * FROM kanban_comments WHERE id = ?", (comment_id,), entity="comment", entity_id=comment_id)
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (row["card_id"],), entity="card", entity_id=row["card_id"])
            conn.execute("UPDATE kanban_comments SET is_deleted = 1, deleted_at = ?, updated_at = ?, version = version + 1 WHERE id = ?", (self._now(), self._now(), comment_id))
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card["id"], action_type="comment", entity_type="comment", entity_id=comment_id, details={"deleted": True})
            return self._comment_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_comments WHERE id = ?", (comment_id,), entity="comment", entity_id=comment_id)
            )

    async def bulk_move_cards(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("bulk_move_cards"))
        request = KanbanBulkMoveCardsRequest(**dict(request_data or {}))
        cards = []
        for card_id in request.card_ids:
            cards.append(await self.move_card(card_id, {"target_list_id": request.target_list_id, "position": request.position}))
        return {"success": True, "moved_count": len(cards), "cards": cards}

    async def bulk_archive_cards(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("bulk_archive_cards"))
        request = KanbanBulkArchiveCardsRequest(**dict(request_data or {}))
        for card_id in request.card_ids:
            await self.archive_card(card_id)
        return {"success": True, "archived_count": len(request.card_ids)}

    async def bulk_unarchive_cards(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("bulk_unarchive_cards"))
        request = KanbanBulkArchiveCardsRequest(**dict(request_data or {}))
        for card_id in request.card_ids:
            await self.unarchive_card(card_id)
        return {"success": True, "unarchived_count": len(request.card_ids)}

    async def bulk_delete_cards(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("bulk_delete_cards"))
        request = KanbanBulkDeleteCardsRequest(**dict(request_data or {}))
        for card_id in request.card_ids:
            await self.delete_card(card_id)
        return {"success": True, "deleted_count": len(request.card_ids)}

    async def bulk_label_cards(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("bulk_label_cards"))
        request = KanbanBulkLabelCardsRequest(**dict(request_data or {}))
        for card_id in request.card_ids:
            for label_id in request.add_label_ids or []:
                await self.assign_label_to_card(card_id, label_id)
            for label_id in request.remove_label_ids or []:
                await self.remove_label_from_card(card_id, label_id)
        return {"success": True, "updated_count": len(request.card_ids)}

    async def filter_board_cards(
        self,
        board_id: int,
        *,
        query: str | None = None,
        priority: str | None = None,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        self._enforce(self._local_action_id("filter_board_cards"))
        cards, total = self._search_cards_raw(
            query=query,
            board_id=board_id,
            priority=priority,
            include_archived=include_archived,
            limit=limit,
            offset=offset,
        )
        return {"cards": cards, "pagination": self._pagination(limit=limit, offset=offset, total=total)}

    async def copy_card_with_checklists(self, card_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("copy_card_with_checklists"))
        request = KanbanCardCopyWithChecklistsRequest(**dict(request_data or {}))
        copied = await self.copy_card(
            card_id,
            {
                "target_list_id": request.target_list_id,
                "new_client_id": request.new_client_id,
                "position": request.position,
                "new_title": request.new_title,
            },
        )
        conn = self.connect()
        try:
            checklist_rows = conn.execute("SELECT * FROM kanban_checklists WHERE card_id = ? ORDER BY position ASC", (card_id,)).fetchall()
            label_rows = conn.execute("SELECT label_id FROM kanban_card_labels WHERE card_id = ?", (card_id,)).fetchall()
        finally:
            conn.close()
        if request.copy_labels:
            for row in label_rows:
                await self.assign_label_to_card(copied["id"], row["label_id"])
        if request.copy_checklists:
            for checklist_row in checklist_rows:
                created = await self.create_checklist(copied["id"], {"name": checklist_row["name"], "position": checklist_row["position"]})
                conn = self.connect()
                try:
                    item_rows = conn.execute("SELECT * FROM kanban_checklist_items WHERE checklist_id = ? ORDER BY position ASC", (checklist_row["id"],)).fetchall()
                finally:
                    conn.close()
                for item_row in item_rows:
                    await self.create_checklist_item(created["id"], {"name": item_row["name"], "position": item_row["position"], "checked": bool(item_row["checked"])})
        return copied

    def _search_cards_raw(
        self,
        *,
        query: str | None,
        board_id: int | None,
        priority: str | None = None,
        include_archived: bool = False,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        where = ["cards.is_deleted = 0"]
        params: list[Any] = []
        if not include_archived:
            where.append("cards.is_archived = 0")
        if board_id is not None:
            where.append("cards.board_id = ?")
            params.append(board_id)
        if priority is not None:
            where.append("cards.priority = ?")
            params.append(priority)
        if query:
            where.append("(cards.title LIKE ? OR COALESCE(cards.description, '') LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        where_sql = " AND ".join(where)
        conn = self.connect()
        try:
            total = conn.execute(
                f"SELECT COUNT(*) FROM kanban_cards cards WHERE {where_sql}",
                tuple(params),
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT cards.* FROM kanban_cards cards WHERE {where_sql} ORDER BY cards.updated_at DESC, cards.id DESC LIMIT ? OFFSET ?",
                tuple(params + [limit, offset]),
            ).fetchall()
            return [self._card_row_to_dict(row) for row in rows], total
        finally:
            conn.close()

    def _search_result_for_card(self, card: dict[str, Any]) -> dict[str, Any]:
        conn = self.connect()
        try:
            board = self._row(conn, "SELECT * FROM kanban_boards WHERE id = ?", (card["board_id"],), entity="board", entity_id=card["board_id"])
            list_row = self._row(conn, "SELECT * FROM kanban_lists WHERE id = ?", (card["list_id"],), entity="list", entity_id=card["list_id"])
        finally:
            conn.close()
        labels = []
        return {
            "id": card["id"],
            "uuid": card["uuid"],
            "board_id": card["board_id"],
            "board_name": board["name"],
            "list_id": card["list_id"],
            "list_name": list_row["name"],
            "title": card["title"],
            "description": card["description"],
            "priority": card["priority"],
            "due_date": card["due_date"],
            "labels": labels,
            "created_at": card["created_at"],
            "updated_at": card["updated_at"],
            "relevance_score": 1.0,
        }

    async def search_cards_basic(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("search_cards_basic"))
        request = KanbanCardSearchRequest(**dict(request_data or {}))
        cards, total = self._search_cards_raw(query=request.query, board_id=request.board_id, limit=request.limit, offset=request.offset)
        return {"cards": cards, "pagination": self._pagination(limit=request.limit, offset=request.offset, total=total)}

    async def search_cards_basic_get(self, query: str = "", **kwargs: Any) -> dict[str, Any]:
        return await self.search_cards_basic({"query": query or kwargs.get("q") or kwargs.get("query", ""), **kwargs})

    async def search_cards(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("search_cards"))
        request = KanbanSearchRequest(**dict(request_data or {}))
        effective_mode = "fts" if self.get_storage_status()["fts_available"] else "like"
        cards, total = self._search_cards_raw(
            query=request.query,
            board_id=request.board_id,
            priority=request.priority,
            include_archived=request.include_archived,
            limit=request.limit,
            offset=request.offset,
        )
        results = [self._search_result_for_card(card) for card in cards]
        return {
            "query": request.query,
            "search_mode": request.search_mode,
            "effective_search_mode": effective_mode,
            "local_search_degraded": request.search_mode in {"vector", "hybrid"} or effective_mode == "like",
            "results": results,
            "pagination": self._pagination(limit=request.limit, offset=request.offset, total=total),
        }

    async def search_cards_get(self, query: str = "", **kwargs: Any) -> dict[str, Any]:
        return await self.search_cards({"query": query or kwargs.get("q") or kwargs.get("query", ""), **kwargs})

    async def get_search_status(self) -> dict[str, Any]:
        self._enforce(self._local_action_id("get_search_status"))
        status = self.get_storage_status()
        return {
            "index_ready": True,
            "fts_available": status["fts_available"],
            "effective_search_mode": "fts" if status["fts_available"] else "like",
            "local_search_degraded_modes": ["vector", "hybrid"],
        }

    async def add_card_link(self, card_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("add_card_link"))
        request = KanbanCardLinkCreate(**dict(request_data or {}))
        now = self._now()
        with self.transaction() as conn:
            card = self._row(conn, "SELECT * FROM kanban_cards WHERE id = ?", (card_id,), entity="card", entity_id=card_id)
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO kanban_card_links (uuid, card_id, linked_type, linked_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self._new_uuid(), card_id, request.linked_type, request.linked_id, now),
            )
            if cursor.lastrowid:
                link_id = int(cursor.lastrowid)
            else:
                link_id = conn.execute(
                    "SELECT id FROM kanban_card_links WHERE card_id = ? AND linked_type = ? AND linked_id = ?",
                    (card_id, request.linked_type, request.linked_id),
                ).fetchone()["id"]
            self._record_activity(conn, board_id=card["board_id"], list_id=card["list_id"], card_id=card_id, action_type="link", entity_type="card_link", entity_id=link_id)
            return self._link_row_to_dict(
                self._row(conn, "SELECT * FROM kanban_card_links WHERE id = ?", (link_id,), entity="card_link", entity_id=link_id)
            )

    async def list_card_links(self, card_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_card_links"))
        conn = self.connect()
        try:
            rows = conn.execute("SELECT * FROM kanban_card_links WHERE card_id = ? ORDER BY id ASC", (card_id,)).fetchall()
            return {"links": [self._link_row_to_dict(row) for row in rows]}
        finally:
            conn.close()

    async def get_card_link_counts(self, card_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("get_card_link_counts"))
        conn = self.connect()
        try:
            rows = conn.execute("SELECT linked_type, COUNT(*) AS count FROM kanban_card_links WHERE card_id = ? GROUP BY linked_type", (card_id,)).fetchall()
            counts = {"media": 0, "note": 0}
            counts.update({row["linked_type"]: row["count"] for row in rows})
            return counts
        finally:
            conn.close()

    async def remove_card_link(self, card_id: int, linked_type: str, linked_id: str) -> dict[str, Any]:
        self._enforce(self._local_action_id("remove_card_link"))
        with self.transaction() as conn:
            conn.execute("DELETE FROM kanban_card_links WHERE card_id = ? AND linked_type = ? AND linked_id = ?", (card_id, linked_type, str(linked_id)))
            return {"detail": "removed"}

    async def remove_card_link_by_id_for_card(self, card_id: int, link_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("remove_card_link_by_id_for_card"))
        with self.transaction() as conn:
            conn.execute("DELETE FROM kanban_card_links WHERE card_id = ? AND id = ?", (card_id, link_id))
            return {"detail": "removed"}

    async def remove_card_link_by_id(self, link_id: int) -> dict[str, Any]:
        self._enforce(self._local_action_id("remove_card_link_by_id"))
        with self.transaction() as conn:
            conn.execute("DELETE FROM kanban_card_links WHERE id = ?", (link_id,))
            return {"detail": "removed"}

    async def bulk_add_card_links(self, card_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("bulk_add_card_links"))
        request = KanbanBulkCardLinksRequest(**dict(request_data or {}))
        links = []
        skipped = 0
        for link_request in request.links:
            before = (await self.get_card_link_counts(card_id))[link_request.linked_type]
            link = await self.add_card_link(card_id, link_request.model_dump())
            after = (await self.get_card_link_counts(card_id))[link_request.linked_type]
            skipped += 1 if after == before else 0
            links.append(link)
        return {"added_count": len(links) - skipped, "skipped_count": skipped, "links": links}

    async def bulk_remove_card_links(self, card_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("bulk_remove_card_links"))
        request = KanbanBulkCardLinksRequest(**dict(request_data or {}))
        removed = 0
        for link_request in request.links:
            await self.remove_card_link(card_id, link_request.linked_type, link_request.linked_id)
            removed += 1
        return {"removed_count": removed}

    async def list_cards_by_linked_content(self, linked_type: str, linked_id: str, **_: Any) -> dict[str, Any]:
        self._enforce(self._local_action_id("list_cards_by_linked_content"))
        conn = self.connect()
        try:
            rows = conn.execute(
                """
                SELECT cards.*, links.id AS link_id, links.created_at AS linked_at, boards.name AS board_name, lists.name AS list_name
                FROM kanban_card_links links
                JOIN kanban_cards cards ON cards.id = links.card_id
                JOIN kanban_boards boards ON boards.id = cards.board_id
                JOIN kanban_lists lists ON lists.id = cards.list_id
                WHERE links.linked_type = ? AND links.linked_id = ?
                ORDER BY links.id ASC
                """,
                (linked_type, str(linked_id)),
            ).fetchall()
            cards = [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "description": row["description"],
                    "board_id": row["board_id"],
                    "board_name": row["board_name"],
                    "list_id": row["list_id"],
                    "list_name": row["list_name"],
                    "position": int(row["position"]),
                    "is_archived": self._bool(row["is_archived"]),
                    "is_deleted": self._bool(row["is_deleted"]),
                    "link_id": row["link_id"],
                    "linked_at": row["linked_at"],
                }
                for row in rows
            ]
            return {"linked_type": linked_type, "linked_id": str(linked_id), "cards": cards}
        finally:
            conn.close()
