"""In-memory event cursor and dedupe tracking for realtime foundations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

from tldw_chatbook.runtime_policy.server_parity_models import (
    EventCursor,
    EventDedupeKey,
    NormalizedEventRecord,
    SourceAuthority,
)


class CursorAdvanceStatus(str, Enum):
    ADVANCED = "advanced"
    IGNORED_NO_CURSOR = "ignored_no_cursor"
    STALE_RESET = "stale_reset"


@dataclass(frozen=True, slots=True)
class CursorAdvanceResult:
    status: CursorAdvanceStatus
    cursor: EventCursor
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class DedupeResult:
    key: EventDedupeKey
    is_duplicate: bool


class EventCursorStore:
    """Small process-local cursor store.

    This intentionally does not persist or subscribe to server contracts. It
    gives observers a shared cursor/dedupe primitive that can later be backed by
    durable storage without changing the event model.
    """

    def __init__(self, *, dedupe_retention: int = 1_000) -> None:
        if dedupe_retention < 1:
            raise ValueError("dedupe_retention must be at least 1")
        self._dedupe_retention = dedupe_retention
        self._cursors: dict[str, EventCursor] = {}
        self._dedupe: OrderedDict[EventDedupeKey, None] = OrderedDict()

    @property
    def dedupe_size(self) -> int:
        return len(self._dedupe)

    def get_cursor(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
    ) -> EventCursor:
        cursor = EventCursor(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
        )
        return self._cursors.get(cursor.storage_key(), cursor)

    def remember_event(self, event: NormalizedEventRecord) -> DedupeResult:
        key = EventDedupeKey.from_event(event)
        if key in self._dedupe:
            self._dedupe.move_to_end(key)
            return DedupeResult(key=key, is_duplicate=True)

        self._dedupe[key] = None
        while len(self._dedupe) > self._dedupe_retention:
            self._dedupe.popitem(last=False)
        return DedupeResult(key=key, is_duplicate=False)

    def acknowledge_event(
        self,
        event: NormalizedEventRecord,
        *,
        expected_cursor: str | None = None,
    ) -> CursorAdvanceResult:
        current = self.get_cursor(
            source_authority=event.source_authority,
            server_profile_id=event.server_profile_id,
            stream_name=event.stream_name,
            stream_instance_id=event.stream_instance_id,
        )
        if expected_cursor is not None and current.cursor != expected_cursor:
            reset = EventCursor(
                source_authority=current.source_authority,
                server_profile_id=current.server_profile_id,
                stream_name=current.stream_name,
                stream_instance_id=current.stream_instance_id,
                cursor=None,
            )
            self._cursors[reset.storage_key()] = reset
            return CursorAdvanceResult(
                status=CursorAdvanceStatus.STALE_RESET,
                cursor=reset,
                reason="cursor_mismatch",
            )

        if event.server_cursor is None:
            return CursorAdvanceResult(
                status=CursorAdvanceStatus.IGNORED_NO_CURSOR,
                cursor=current,
                reason="missing_server_cursor",
            )

        advanced = EventCursor(
            source_authority=event.source_authority,
            server_profile_id=event.server_profile_id,
            stream_name=event.stream_name,
            stream_instance_id=event.stream_instance_id,
            cursor=event.server_cursor,
        )
        self._cursors[advanced.storage_key()] = advanced
        return CursorAdvanceResult(status=CursorAdvanceStatus.ADVANCED, cursor=advanced)

    def reset_cursor(self, cursor: EventCursor, *, reason: str = "stale_cursor") -> CursorAdvanceResult:
        reset = EventCursor(
            source_authority=cursor.source_authority,
            server_profile_id=cursor.server_profile_id,
            stream_name=cursor.stream_name,
            stream_instance_id=cursor.stream_instance_id,
            cursor=None,
        )
        self._cursors[reset.storage_key()] = reset
        return CursorAdvanceResult(
            status=CursorAdvanceStatus.STALE_RESET,
            cursor=reset,
            reason=reason,
        )
