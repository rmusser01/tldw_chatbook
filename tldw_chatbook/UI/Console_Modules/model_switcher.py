"""Switch model: recent and previous provider·model pairs (TASK-33004.3).

Spec §4 rule 6 forbids a new store, so recents come only from data that
already exists: the open Console sessions (temporary chats included) and the
ADR-095 snapshots of the 50 most recently modified global-scope chats. A
workspace chat is never listed, so it appears only while it is open. PREVIOUS
per chat lives in process memory.

ADR-097: import this module lazily from the switcher's openers only; it is not
on the boot path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from ...Chat.console_environment_state import relative_age
from ...Chat.console_generation_settings_metadata import (
    parse_console_generation_settings,
)
from ...DB.base_db import run_owned_db_call

if TYPE_CHECKING:
    from ...Chat.console_chat_store import ConsoleChatSession
    from ...DB.ChaChaNotes_DB import CharactersRAGDB

RECENT_CONVERSATION_LIMIT = 50

Pair = tuple[str, str | None]


@dataclass(frozen=True, slots=True)
class ModelPairUse:
    """One provider·model pair and when it was last used."""

    provider: str
    model: str
    last_used: datetime
    in_this_chat: bool = False

    @property
    def pair(self) -> tuple[str, str]:
        return (self.provider, self.model)

    def used_label(self, now: datetime) -> str:
        """Return the row's last-use text, e.g. ``used 2h ago in this chat``."""
        age = relative_age(self.last_used, now)
        text = "used just now" if age == "0m ago" else f"used {age}"
        return f"{text} in this chat" if self.in_this_chat else text


def _utc(value: object) -> datetime | None:
    """Read a datetime or ISO timestamp as UTC (naive means UTC).

    SQLite's declared-type parsing already returns ``last_modified`` as a
    datetime; live sessions carry an ISO string.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def live_model_pair_uses(
    sessions: Iterable[ConsoleChatSession],
    stored: Mapping[str, ModelPairUse] | None = None,
) -> list[ModelPairUse]:
    """Return the pairs of open Console sessions. UI thread only.

    A chat reopened from Sessions carries the time it was opened as
    ``updated_at``, which is not a use. While its pair still matches its stored
    row (``stored``, keyed by conversation id) the row's ``last_modified``
    stands, so the session adds nothing; its own time counts once the pair
    changed.
    """
    stored = stored or {}
    uses = []
    for session in sessions:
        settings = session.settings
        if settings is None or not settings.model:
            continue
        pair = (settings.provider, settings.model)
        conversation_id = session.persisted_conversation_id
        kept = stored.get(str(conversation_id)) if conversation_id else None
        if kept is not None and kept.pair == pair:
            continue
        # ponytail: a reopened chat whose row was not read (workspace scope,
        # older than the 50th, no valid snapshot) still shows its open time;
        # stamp updated_at from the row in restore_persisted_session if that
        # matters in testing.
        when = _utc(session.updated_at)
        if when is not None:
            uses.append(ModelPairUse(*pair, when))
    return uses


def persisted_model_pair_uses(db: CharactersRAGDB) -> dict[str, ModelPairUse]:
    """Return the newest global-scope chats' pairs by conversation id. Blocking: run off the loop.

    One listing capped at 50 and one batched metadata read. Snapshots
    that are malformed, missing, from a newer version or carry no model are
    skipped (fail closed).
    """
    rows = [
        row
        for row in db.list_all_active_conversations(limit=RECENT_CONVERSATION_LIMIT)
        if row.get("id")
    ]
    metadata = db.get_conversations_metadata_by_ids([str(row["id"]) for row in rows])
    uses = {}
    for row in rows:
        conversation_id = str(row["id"])
        if conversation_id not in metadata:
            continue
        snapshot = parse_console_generation_settings(metadata[conversation_id]).snapshot
        when = _utc(row.get("last_modified"))
        if snapshot is None or not snapshot.model or when is None:
            continue
        uses[conversation_id] = ModelPairUse(snapshot.provider, snapshot.model, when)
    return uses


def newest_distinct(uses: Iterable[ModelPairUse]) -> list[ModelPairUse]:
    """Keep each pair's newest use and order the pairs newest first."""
    newest: dict[tuple[str, str], ModelPairUse] = {}
    for use in uses:
        kept = newest.get(use.pair)
        if kept is None or use.last_used > kept.last_used:
            newest[use.pair] = use
    return sorted(newest.values(), key=lambda use: use.last_used, reverse=True)


async def read_recent_model_pairs(
    db: CharactersRAGDB | None,
    sessions: Iterable[ConsoleChatSession],
) -> list[ModelPairUse]:
    """Return RECENT: open sessions plus the newest global chats, off the UI thread.

    The database read runs in a worker thread; the live sessions are read on
    the caller's (UI) thread once it returns. A failed read keeps the open
    sessions' pairs.
    """
    sessions = list(sessions)
    stored: dict[str, ModelPairUse] = {}
    if db is not None:
        try:
            stored = await run_owned_db_call(db, persisted_model_pair_uses, db)
        except Exception as exc:  # noqa: BLE001 - DB adapters have no shared error base.
            logger.debug("Recent model pairs read skipped: {}", type(exc).__name__)
    return newest_distinct([*live_model_pair_uses(sessions, stored), *stored.values()])


class PreviousPairMemory:
    """PREVIOUS per chat, kept in process memory only (spec §4 rule 6).

    A switcher commit records the pair it replaced. A pair change made
    elsewhere (Chat settings) is not recorded; PREVIOUS then falls back to
    RECENT's newest other pair.
    """

    def __init__(self) -> None:
        # ponytail: one entry per chat that switched, never evicted; bound it
        # if process lifetimes reach thousands of chats.
        self._by_session: dict[str, ModelPairUse] = {}

    def record_switch(
        self, session_id: str, before: Pair, after: Pair, *, now: datetime | None = None
    ) -> None:
        """Remember ``before`` as this chat's PREVIOUS when the pair changed."""
        provider, model = before
        if not model or before == after:
            return
        self._by_session[session_id] = ModelPairUse(
            provider, model, now or datetime.now(UTC), in_this_chat=True
        )

    def previous(
        self, session_id: str, current: Pair, recent: Sequence[ModelPairUse]
    ) -> ModelPairUse | None:
        """Return this chat's previous pair, else RECENT's newest other pair."""
        remembered = self._by_session.get(session_id)
        if remembered is not None and remembered.pair != current:
            return remembered
        return next((use for use in recent if use.pair != current), None)


#: The process-lifetime PREVIOUS memory the switcher reads and records into.
PREVIOUS_PAIRS = PreviousPairMemory()
