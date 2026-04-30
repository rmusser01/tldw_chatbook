"""In-memory event cursor and dedupe tracking for realtime foundations."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum

from tldw_chatbook.runtime_policy.server_parity_models import (
    EventCursor,
    EventDedupeKey,
    NormalizedEventRecord,
    SourceAuthority,
)

EventDedupeIdentity = tuple[Hashable, ...]


class _NoExpectedCursor:
    pass


_NO_EXPECTED_CURSOR = _NoExpectedCursor()


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
    key: EventDedupeIdentity
    is_duplicate: bool


class EventCursorStore:
    """Small process-local test/compatibility cursor store.

    Production server event observation should use EventStateRepository. This
    store intentionally does not persist or subscribe to server contracts and
    must not be treated as a second active event-state authority.
    """

    def __init__(self, *, dedupe_retention: int = 1_000) -> None:
        if dedupe_retention < 1:
            raise ValueError("dedupe_retention must be at least 1")
        self._dedupe_retention = dedupe_retention
        self._cursors: dict[str, EventCursor] = {}
        self._dedupe: OrderedDict[EventDedupeIdentity, None] = OrderedDict()

    @property
    def dedupe_size(self) -> int:
        return len(self._dedupe)

    def is_duplicate_event(self, event: NormalizedEventRecord) -> bool:
        return self._dedupe_identity(event) in self._dedupe

    @staticmethod
    def _dedupe_identity(event: NormalizedEventRecord) -> EventDedupeIdentity:
        scope = (
            event.source_authority,
            event.server_profile_id,
            event.authenticated_principal_id,
            event.stream_name,
            event.stream_instance_id,
        )
        if event.event_id:
            return (*scope, "event_id", event.event_id)
        if event.server_cursor:
            return (*scope, "server_cursor", event.server_cursor)
        return (*scope, "fallback", EventDedupeKey.from_event(event))

    def get_cursor(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
    ) -> EventCursor:
        cursor = EventCursor(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
            authenticated_principal_id=authenticated_principal_id,
        )
        return self._cursors.get(cursor.storage_key(), cursor)

    def remember_event(self, event: NormalizedEventRecord) -> DedupeResult:
        key = self._dedupe_identity(event)
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
        expected_cursor: str | None | _NoExpectedCursor = _NO_EXPECTED_CURSOR,
    ) -> CursorAdvanceResult:
        current = self.get_cursor(
            source_authority=event.source_authority,
            server_profile_id=event.server_profile_id,
            authenticated_principal_id=event.authenticated_principal_id,
            stream_name=event.stream_name,
            stream_instance_id=event.stream_instance_id,
        )
        if not isinstance(expected_cursor, _NoExpectedCursor) and current.cursor != expected_cursor:
            reset = EventCursor(
                source_authority=current.source_authority,
                server_profile_id=current.server_profile_id,
                stream_name=current.stream_name,
                stream_instance_id=current.stream_instance_id,
                cursor=None,
                authenticated_principal_id=current.authenticated_principal_id,
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
            authenticated_principal_id=event.authenticated_principal_id,
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
            authenticated_principal_id=cursor.authenticated_principal_id,
        )
        self._cursors[reset.storage_key()] = reset
        return CursorAdvanceResult(
            status=CursorAdvanceStatus.STALE_RESET,
            cursor=reset,
            reason=reason,
        )
