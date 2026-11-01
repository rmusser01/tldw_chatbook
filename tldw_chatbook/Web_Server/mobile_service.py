"""Authenticated adapter to the retained canonical Console owner (ADR-244)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_remote_models import ConsoleHostIdentity, ConsoleTarget
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase, get_media_transcripts
from tldw_chatbook.DB.console_command_receipts import ConsoleCommandReceipt

_NOTES_PREVIEW_CHARS = 200


def _wire_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _wire_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_safe(item) for item in value]
    return value


class MobileSendRefused(RuntimeError):
    """An obsolete mobile mutation needs the current client protocol."""


class MobileOwnerUnavailable(RuntimeError):
    """Canonical execution is unavailable; no alternative engine is started."""


@dataclass(frozen=True)
class MobileEvent:
    type: str
    session_id: str
    payload: dict
    seq: int

    def to_wire(self):
        return {
            "type": self.type,
            "session_id": self.session_id,
            "payload": self.payload,
            "seq": self.seq,
        }


class CanonicalMobileReads:
    """Canonical saved-history and Library readers; never execution owners."""

    def __init__(self, chat_path, media_path, run_path):
        self.chat_path, self.media_path, self.run_path = chat_path, media_path, run_path
        self._notes_db = self._media_db = None
        self._lock = threading.Lock()

    def _core(self):
        with self._lock:
            if self._notes_db is None:
                self._notes_db = CharactersRAGDB.open_read_only(self.chat_path)
        return self._notes_db

    def _media(self):
        with self._lock:
            if self._media_db is None:
                self._media_db = MediaDatabase.open_read_only(self.media_path)
        return self._media_db

    def sessions_page(self, *, page=1, per_page=50):
        result = ChatConversationService(self._core()).list_conversations(
            limit=per_page, offset=(page - 1) * per_page, scope_type="all"
        )
        return _wire_safe(
            {
                "items": result["items"],
                "page": page,
                "has_more": result["pagination"]["has_more"],
            }
        )

    def session_messages(self, conversation_id, **coordinates):
        result = ChatConversationService(
            self._core()
        ).get_library_conversation_messages(conversation_id, **coordinates)
        if result is None:
            raise KeyError(conversation_id)
        result["messages"] = [
            row
            for row in result["messages"]
            if row["sender"] in {"user", "assistant", "tool"}
        ]
        return _wire_safe(result)

    def query_command(self, command_id):
        database = AgentRunsDB.open_read_only(self.run_path)
        try:
            return database.command_receipts.get(command_id)
        finally:
            database.close()

    def close(self):
        for database in (self._notes_db, self._media_db):
            if database is not None:
                database.close_connection()

    def notes_page(self, *, page: int = 1, per_page: int = 50) -> dict[str, Any]:
        """List non-deleted notes newest-first with truncated previews.

        Args:
            page: 1-based page number.
            per_page: Page size (the route layer caps this).

        Returns:
            ``{"items": [{id, title, preview, created_at, last_modified}],
            "page": n, "has_more": bool}`` — ``preview`` is the note's
            first ``_NOTES_PREVIEW_CHARS`` characters (never the full
            body); the timestamps pass through from ``list_notes`` as-is.
            ``has_more`` comes from the one-over fetch row.
        """
        notes_db = self._core()
        rows = notes_db.list_notes(
            limit=per_page + 1,
            offset=(page - 1) * per_page,
            preview_chars=_NOTES_PREVIEW_CHARS,
        )
        has_more = len(rows) > per_page
        items = [
            _wire_safe(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "preview": row["preview"],
                    "created_at": row.get("created_at"),
                    "last_modified": row.get("last_modified"),
                }
            )
            for row in rows[:per_page]
        ]
        return {"items": items, "page": page, "has_more": has_more}

    def note_detail(self, note_id: str) -> dict[str, Any] | None:
        """Return one note's full row, or ``None`` when it does not exist.

        Args:
            note_id: Note id.

        Returns:
            The note row (``id`` / ``title`` / ``content`` plus the other
            scalar row fields ``get_note_by_id`` selects; DATETIME columns
            as ISO 8601 strings), or ``None`` for an unknown or
            soft-deleted note.
        """
        notes_db = self._core()
        row = notes_db.get_note_by_id(note_id)
        return _wire_safe(row) if row is not None else None

    def library_page(self, *, page: int = 1, per_page: int = 25) -> dict[str, Any]:
        """List active (non-deleted, non-trashed) media newest-first.

        Args:
            page: 1-based page number.
            per_page: Page size (the route layer caps this).

        Returns:
            ``{"items": [{media_id, title, media_type, ingestion_date,
            last_modified}], "page": n, "has_more": bool}`` — ``has_more``
            derives from ``search_media_db``'s total-matches count.
        """
        media_db = self._media()
        # search_media_db returns ``(results_list, total_matches)`` — a
        # tuple, not a dict; extraction adapts to that shape while the
        # wire shape above stays fixed.
        results, total = media_db.search_media_db(
            search_query=None,
            page=page,
            results_per_page=per_page,
            include_trash=False,
            include_deleted=False,
        )
        items = [_wire_safe(self._library_item(row)) for row in results]
        has_more = page * per_page < int(total)
        return {"items": items, "page": page, "has_more": has_more}

    def library_detail(self, media_id: int) -> dict[str, Any] | None:
        """Return one media row joined with its active transcripts.

        Args:
            media_id: Media id.

        Returns:
            ``{media_id, title, media_type, ...remaining row fields,
            transcripts: [...]}`` — the base wire projection plus every
            other scalar row field passed through, plus the
            ``get_media_transcripts`` list; ``None`` when no active row
            matches.
        """
        media_db = self._media()
        results, _total = media_db.search_media_db(
            search_query=None,
            media_ids_filter=[int(media_id)],
            page=1,
            results_per_page=1,
            include_trash=False,
            include_deleted=False,
        )
        if not results:
            return None
        row = results[0]
        item = self._library_item(row)
        for key, value in row.items():
            if key not in ("id", "title", "type", "ingestion_date", "last_modified"):
                item[key] = value
        item["transcripts"] = _wire_safe(get_media_transcripts(media_db, int(media_id)))
        return _wire_safe(item)

    @staticmethod
    def _library_item(row: dict[str, Any]) -> dict[str, Any]:
        """Project one ``search_media_db`` row onto the mobile wire keys."""
        return {
            "media_id": row["id"],
            "title": row["title"],
            "media_type": row["type"],
            "ingestion_date": row.get("ingestion_date"),
            "last_modified": row.get("last_modified"),
        }


class MobileChatService:
    """Route exact authenticated targets; the server owns child lifetime."""

    def __init__(self, *, hosts, storage_identity, broker, readers, require_cutover):
        self.hosts = hosts
        self.storage_identity = storage_identity
        self.broker = broker
        self.readers = readers
        self.require_cutover = require_cutover
        self._host = None
        self._closed = False
        self._snapshot_boundaries = {}

    def sessions_page(self, **kwargs):
        return self.readers.sessions_page(**kwargs)

    async def session_messages(self, conversation_id, **kwargs):
        from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker

        return await asyncio.to_thread(
            run_finite_local_worker,
            self.readers.session_messages,
            conversation_id,
            **kwargs,
        )

    def notes_page(self, **kwargs):
        return self.readers.notes_page(**kwargs)

    def note_detail(self, note_id):
        return self.readers.note_detail(note_id)

    def library_page(self, **kwargs):
        return self.readers.library_page(**kwargs)

    def library_detail(self, media_id):
        return self.readers.library_detail(media_id)

    async def _owner(self, *, start=False):
        if self._closed or self.broker is None:
            raise MobileOwnerUnavailable("console_unavailable")
        if start:
            await asyncio.to_thread(self.require_cutover)
            self._host = await self.hosts.get_or_start(self.storage_identity)
        if self._host is None:
            getter = getattr(self.hosts, "get_existing", None)
            self._host = getter(self.storage_identity) if getter else None
        if self._host is None or self._host._closed:
            raise MobileOwnerUnavailable("console_unavailable")
        child_id = self._host.service.app_service_id
        await self.broker.wait_connected(child_id, timeout=5)
        return child_id

    async def _request(self, message_type, request, *, start=False):
        child_id = await self._owner(start=start)
        response = await self.broker.request(
            child_id, message_type, {"request": request}, timeout=5
        )
        return response.payload

    async def bootstrap_host(self) -> ConsoleHostIdentity:
        """Explicit first use, with positive native execution admission."""
        await self._owner(start=True)
        snapshot = await self.snapshot()
        if snapshot.get("execution_admitted") is not True:
            raise MobileOwnerUnavailable("execution_owner_unavailable")
        return ConsoleHostIdentity.model_validate(
            {key: snapshot[key] for key in ("server_epoch", "console_id")}
        )

    async def snapshot(
        self, *, target=None, cursor=None, limit=50, receipt_cursor=None
    ):
        child_id = await self._owner()
        buffer = self.broker.console_events(child_id)
        # Transport receive cursor and native capture sequence are different domains.
        before = buffer.read(0).sequence
        scope = target.model_dump_json() if target else None
        prior = (
            self._snapshot_boundaries.pop(
                (child_id, scope, cursor, receipt_cursor), None
            )
            if cursor or receipt_cursor
            else None
        )
        boundary = before if prior is None else prior
        data = {
            "target": target.model_dump() if target else None,
            "cursor": cursor,
            "limit": limit,
            "receipt_cursor": receipt_cursor,
        }
        response = await self.broker.request(
            child_id, "console.snapshot.request", {"request": data}, timeout=5
        )
        snapshot = dict(response.payload["snapshot"])
        after = buffer.read(boundary)
        continuation = snapshot.get("next_cursor") or snapshot.get(
            "next_receipt_cursor"
        )
        raced = after.sequence != boundary or (
            (cursor or receipt_cursor) and prior is None
        )
        snapshot["transport_sequence"] = boundary
        snapshot["reset_required"] = bool(
            snapshot.get("reset_required") or after.reset_required or raced
        )
        if continuation:
            if len(self._snapshot_boundaries) >= 64:
                self._snapshot_boundaries.pop(next(iter(self._snapshot_boundaries)))
            key = (
                child_id,
                scope,
                snapshot.get("next_cursor"),
                snapshot.get("next_receipt_cursor"),
            )
            self._snapshot_boundaries[key] = min(
                boundary, self._snapshot_boundaries.get(key, boundary)
            )
        elif not raced:
            await self.broker.request(
                child_id,
                "console.events.request",
                {
                    "request": {
                        "operation": "acknowledge",
                        "sequence": snapshot["sequence"],
                    }
                },
                timeout=5,
            )
            buffer.acknowledge_snapshot(boundary)
        return snapshot

    async def issue_command_ticket(self, action, target, payload):
        await asyncio.to_thread(self.require_cutover)
        result = await self._request(
            "console.ticket.request",
            {"action": action, "target": target.model_dump(), "payload": payload},
            start=False,
        )
        ticket = result["ticket"]
        return {"ticket": ticket["token"], "command_id": ticket["command_id"]}

    @staticmethod
    def _receipt(value):
        data = dict(value)
        target = data["target"]
        data["target"] = (
            ConsoleTarget if "session_id" in target else ConsoleHostIdentity
        ).model_validate(target)
        return ConsoleCommandReceipt(**data)

    async def submit_command(self, ticket, payload) -> ConsoleCommandReceipt:
        await asyncio.to_thread(self.require_cutover)
        # A lost response is unknown: query the original ID; never mint/replay here.
        result = await self._request(
            "console.command.request", {"ticket": ticket, "payload": payload}
        )
        return self._receipt(result["receipt"])

    async def query_command(self, command_id):
        from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker

        return await asyncio.to_thread(
            run_finite_local_worker, self.readers.query_command, command_id
        )

    async def set_answerable(self, client_id, target, answerable):
        await self._request(
            "console.events.request",
            {
                "operation": "answerable",
                "client_id": client_id,
                "target": target.model_dump(),
                "answerable": answerable,
            },
        )

    async def events(self, *, client_id=None) -> AsyncIterator[MobileEvent]:
        child_id = await self._owner()
        client_id = client_id or uuid4().hex
        buffer = self.broker.console_events(child_id)
        sequence = 0
        try:
            await self._request(
                "console.events.request",
                {"operation": "register", "client_id": client_id},
            )
            while not self._closed:
                batch = buffer.read(sequence)
                if batch.reset_required:
                    yield MobileEvent(
                        "reset", "", {"reset_required": True}, batch.sequence
                    )
                for event in batch.events:
                    if event["kind"] == "console.changed":
                        yield MobileEvent(
                            event["kind"], "", event["payload"], event["sequence"]
                        )
                    else:
                        yield MobileEvent(
                            "reset", "", {"reset_required": True}, event["sequence"]
                        )
                sequence = batch.sequence
                await buffer.wait_changed(sequence)
        finally:
            try:
                await self._request(
                    "console.events.request",
                    {"operation": "disconnect", "client_id": client_id},
                )
            except Exception as error:  # noqa: BLE001 - cleanup must preserve the original stream outcome.
                logger.debug(
                    "Console subscription disconnect unavailable ({})",
                    type(error).__name__,
                )

    async def shutdown(self):
        self._closed = True
        close = getattr(self.readers, "close", None)
        if close:
            close()
