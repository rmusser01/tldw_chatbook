"""Switch model: recent and previous provider·model pairs (TASK-33004.3).

Spec §4 rule 6 forbids a new store, so recents come only from data that
already exists: the open Console sessions (temporary chats included) and the
ADR-095 snapshots of the 50 most recently modified global-scope chats. An open
chat contributes only through its live session, so a workspace chat appears
only while it is open. PREVIOUS per chat lives in process memory.

It also opens the switcher (``open_model_switcher``) and routes NEEDS SETUP
rows to Settings (``open_provider_setup``), so ``chat_screen.py`` keeps one
line per opener.

ADR-097: import this module lazily from the switcher's openers only; it is not
on the boot path.
"""

from __future__ import annotations

import asyncio
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
from ...Utils.timestamps import as_utc

if TYPE_CHECKING:
    from ...Chat.console_chat_store import ConsoleChatSession
    from ...DB.ChaChaNotes_DB import CharactersRAGDB
    from ..Screens.chat_screen import ChatScreen

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
        """Return the row's last-use text, e.g. ``used 2h ago in this chat``.

        Args:
            now: The current time the age is measured against (UTC).

        Returns:
            ``used just now`` under a minute, else ``used <age>``, with
            `` in this chat`` appended for this chat's own PREVIOUS pair.
        """
        age = relative_age(self.last_used, now)
        text = "used just now" if age == "0m ago" else f"used {age}"
        return f"{text} in this chat" if self.in_this_chat else text


def live_model_pair_uses(sessions: Iterable[ConsoleChatSession]) -> list[ModelPairUse]:
    """Return the pairs of open Console sessions. UI thread only.

    A session's ``updated_at`` is its chat's last use: sending moves it, while
    ``add_message`` never touches the row's ``last_modified``, and applying a
    new pair (``commit_console_settings_live``) moves it too. A reopened chat
    starts from its row's ``last_modified`` (``hydrate_console_session``), and
    a chat restored at startup from its saved ``updated_at``.

    Args:
        sessions: The open Console sessions, temporary chats included.

    Returns:
        One use per session with a model and a readable ``updated_at``, in
        the sessions' order; sessions without either are skipped.
    """
    uses = []
    for session in sessions:
        settings = session.settings
        when = as_utc(session.updated_at)
        if settings is None or not settings.model or when is None:
            continue
        uses.append(ModelPairUse(settings.provider, settings.model, when))
    return uses


def persisted_model_pair_uses(
    db: CharactersRAGDB,
    *,
    skip_conversation_ids: frozenset[str] = frozenset(),
) -> list[ModelPairUse]:
    """Return the pairs of the newest global-scope chats. Blocking: run off the loop.

    One listing capped at 50 and one batched metadata read. Snapshots
    that are malformed, missing, from a newer version or carry no model are
    skipped (fail closed).
    """
    rows = [
        row
        for row in db.list_all_active_conversations(limit=RECENT_CONVERSATION_LIMIT)
        if row.get("id") and str(row["id"]) not in skip_conversation_ids
    ]
    metadata = db.get_conversations_metadata_by_ids([str(row["id"]) for row in rows])
    uses = []
    for row in rows:
        conversation_id = str(row["id"])
        if conversation_id not in metadata:
            continue
        snapshot = parse_console_generation_settings(metadata[conversation_id]).snapshot
        when = as_utc(row.get("last_modified"))
        if snapshot is None or not snapshot.model or when is None:
            continue
        uses.append(ModelPairUse(snapshot.provider, snapshot.model, when))
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

    The live sessions are read before the first await, on the caller's (UI)
    thread; the database read runs in a worker thread. A failed read keeps
    the open sessions' pairs.
    """
    sessions = list(sessions)
    uses = live_model_pair_uses(sessions)
    if db is None:
        return newest_distinct(uses)
    open_ids = frozenset(
        str(session.persisted_conversation_id)
        for session in sessions
        if session.persisted_conversation_id
    )
    try:
        uses += await run_owned_db_call(
            db,
            persisted_model_pair_uses,
            db,
            skip_conversation_ids=open_ids,
        )
    except Exception as exc:  # noqa: BLE001 - DB adapters have no shared error base.
        logger.debug("Recent model pairs read skipped: {}", type(exc).__name__)
    return newest_distinct(uses)


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


_SETUP_FIELDS = {
    "configure_credential": "api_key",
    "configure_endpoint": "endpoint",
    "save_endpoint": "endpoint",
}


def open_provider_setup(
    screen: ChatScreen, provider: str, model: str | None = None
) -> None:
    """Open Settings ▸ Providers & Models at one provider's fix (D4, ADR-012).

    Uses the screen-context keys ``_open_console_provider_recovery`` uses; the
    way back to Console waits for P8's handoff store. Settings keeps its own
    model field when no model is routed, so the row's pair travels with it.
    """
    from ...Constants import TAB_SETTINGS
    from ..Navigation.main_navigation import NavigateToScreen
    from ..Screens.settings_config_models import SettingsCategoryId

    readiness = screen._console_default_readiness(provider, model)
    context: dict[str, object] = {
        "category": SettingsCategoryId.PROVIDERS_MODELS.value,
        "provider": provider,
    }
    if model:
        context["model"] = model
    field = _SETUP_FIELDS.get(str(readiness.recovery_action or ""))
    if field:
        context["field"] = field
    screen.post_message(NavigateToScreen(TAB_SETTINGS, screen_context=context))


async def load_provider_catalog(
    screen: ChatScreen,
    providers_models: Mapping[str, Sequence[str]],
    provider: str,
) -> list[str]:
    """Return one provider's cached catalog for Switch model, off the UI thread.

    The local catalog merge is synchronous under its async wrapper: about
    6 ms a provider, so 13 ready providers held the event loop for 110 ms
    in one block on every open. It runs in a worker thread; the options'
    warnings are remembered back on the UI thread.
    """
    from ..Screens.provider_model_resolution import resolve_provider_model_options

    options = await asyncio.to_thread(
        asyncio.run,
        resolve_provider_model_options(
            providers_models,
            getattr(screen.app_instance, "llm_provider_catalog_scope_service", None),
            provider=provider,
            merge_cap=None,
        ),
    )
    screen._remember_console_model_options(provider, options)
    return [option.model_id for option in options]


async def open_model_switcher(screen: ChatScreen, query: str = "") -> None:
    """Open Switch model for the active chat (Alt+M, chips, palette, /model).

    ``/model <query>`` passes its text here: it opens in Find, so the best
    match is highlighted and nothing applies until Enter (TASK-33004.7).
    """
    from ...Chat.console_settings_apply import QUICK_MODEL_DEFAULT_FIELDS
    from ...Widgets.Console.console_model_popover import ConsoleModelPopover

    if screen._console_setup_modal_blocking():
        return
    store = screen._ensure_console_chat_store()
    session_id = store.active_session_id
    if session_id is None:
        return
    origin = store.capture_console_settings_origin(session_id)
    settings = store.session_settings(session_id)
    if settings is None:
        return
    session = store.switch_session(session_id)
    before = (settings.provider, settings.model)
    providers_models = screen._providers_models()

    def commit(submission):  # type: ignore[no-untyped-def]
        live_commit = screen._commit_console_settings_submission_live(submission)
        after = (submission.draft.settings.provider, submission.draft.settings.model)
        PREVIOUS_PAIRS.record_switch(session.id, before, after)
        return live_commit

    screen.app.push_screen(
        ConsoleModelPopover(
            origin=origin,
            app_config=screen._provider_readiness_app_config(),
            initial_draft=screen._console_settings_initial_draft(
                settings,
                store.session_context_policy_overrides(session_id),
                exposed_fields=QUICK_MODEL_DEFAULT_FIELDS,
            ),
            providers_models=providers_models,
            scope_copy="Applies to: this chat only",
            durability_copy=(
                "Temporary until this chat is promoted"
                if session.ephemeral
                else "Saved with the conversation after its first message"
                if session.persisted_conversation_id is None
                else "Saved with this conversation"
            ),
            draft_rebaser=(
                screen._ensure_console_chat_controller().rebase_console_settings_draft
            ),
            live_committer=commit,
            default_readiness_resolver=screen._console_default_readiness,
            recent_pairs_loader=lambda: read_recent_model_pairs(
                getattr(screen.app_instance, "chachanotes_db", None), store.sessions()
            ),
            previous_pair=lambda recent: PREVIOUS_PAIRS.previous(
                session.id, before, recent
            ),
            catalog_loader=lambda provider: load_provider_catalog(
                screen, providers_models, provider
            ),
            setup_opener=lambda provider, model: open_provider_setup(
                screen, provider, model
            ),
            query=query.strip(),
        ),
        callback=screen._apply_console_model_popover_result,
    )
