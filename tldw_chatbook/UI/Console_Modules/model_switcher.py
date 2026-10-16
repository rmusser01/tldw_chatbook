"""Switch model: recent and previous provider·model pairs (TASK-33004.3).

Spec §4 rule 6 forbids a new store, so recents come only from data that
already exists: the open Console sessions (temporary chats included) and the
ADR-095 snapshots of the 50 most recently modified global-scope chats. An open
chat contributes only through its live session, so a workspace chat appears
only while it is open. PREVIOUS per chat lives in process memory.

It also opens the switcher (``open_model_switcher``), with its local-server
probe (``connection_probe``, TASK-33005.5), opens it in pick-only mode for
Chat settings' Change (``open_model_picker``, TASK-33006.4), and routes NEEDS
SETUP rows to Settings (``open_provider_setup``), so ``chat_screen.py`` keeps
one line per opener.

ADR-097: import this module lazily from the switcher's openers only; it is not
on the boot path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NoReturn

from loguru import logger

from ...Chat.console_environment_state import relative_age
from ...Chat.console_generation_settings_metadata import (
    parse_console_generation_settings,
)
from ...DB.base_db import run_owned_db_call
from ...Utils.timestamps import as_utc

if TYPE_CHECKING:
    from ...Chat.console_chat_store import ConsoleChatSession
    from ...Chat.console_settings_apply import (
        ConsoleSettingsDraftState,
        ConsoleSettingsOrigin,
    )
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
        """Return the row's pair.

        Returns:
            ``(provider, model)``.
        """
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

    Args:
        db: The ChaChaNotes database to read.
        skip_conversation_ids: Conversations already open as live sessions.

    Returns:
        One use per readable conversation, in the listing's order.
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
    """Keep each pair's newest use and order the pairs newest first.

    Args:
        uses: Pair uses, possibly repeating a pair.

    Returns:
        One use per pair, newest first.
    """
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

    Args:
        db: The ChaChaNotes database, or None when none is attached.
        sessions: The open Console sessions.

    Returns:
        One use per pair, newest first.
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
        """Remember ``before`` as this chat's PREVIOUS when the pair changed.

        Args:
            session_id: The chat that switched.
            before: The pair it switched from.
            after: The pair it switched to.
            now: The switch time (UTC); defaults to now.
        """
        provider, model = before
        if not model or before == after:
            return
        self._by_session[session_id] = ModelPairUse(
            provider, model, now or datetime.now(UTC), in_this_chat=True
        )

    def previous(
        self, session_id: str, current: Pair, recent: Sequence[ModelPairUse]
    ) -> ModelPairUse | None:
        """Return this chat's previous pair, else RECENT's newest other pair.

        Args:
            session_id: The chat Switch model opened for.
            current: The chat's current pair.
            recent: RECENT, newest first.

        Returns:
            The pair to offer as PREVIOUS, or None when there is none.
        """
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

    Args:
        screen: The Console screen that opens Settings.
        provider: The provider whose fix Settings opens at.
        model: The row's model, when it has one.
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

    Args:
        screen: The Console screen whose catalog scope service is read.
        providers_models: The configured models per provider.
        provider: The provider whose catalog is loaded.

    Returns:
        The provider's model ids.
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


def _switcher_sources(
    screen: ChatScreen, session_id: str, before: Pair
) -> dict[str, object]:
    """Return the pair sources both Switch model modes list from.

    Args:
        screen: The Console screen that owns the chat.
        session_id: The chat whose PREVIOUS pair the list offers.
        before: The pair the chat (or the Chat settings draft) holds now.

    Returns:
        ``ConsoleModelPopover`` keyword arguments: the configuration, the
        saved and cached catalogs, readiness, RECENT and PREVIOUS, the local
        probe and the controller's rebaser.
    """
    from .connection_probe import switcher_connection_prober

    store = screen._ensure_console_chat_store()
    providers_models = screen._providers_models()
    app_config = screen._provider_readiness_app_config()
    probe = switcher_connection_prober(screen.app, app_config)

    async def connection_prober(targets, settled):  # type: ignore[no-untyped-def]
        def settled_here_and_under(provider: str) -> None:
            settled(provider)
            if not screen.is_attached:  # A probe outlives a closed switcher.
                return
            # TASK-30011 AC#6: the Console under the switcher reads the same
            # word now; its idle poll skips a covered screen.
            with screen._console_derivation_scope():
                screen._sync_console_settings_summary()
                screen._sync_console_control_bar()

        await probe(targets, settled_here_and_under)

    return {
        "app_config": app_config,
        "providers_models": providers_models,
        "draft_rebaser": (
            screen._ensure_console_chat_controller().rebase_console_settings_draft
        ),
        "default_readiness_resolver": screen._console_default_readiness,
        "recent_pairs_loader": lambda: read_recent_model_pairs(
            getattr(screen.app_instance, "chachanotes_db", None), store.sessions()
        ),
        "previous_pair": lambda recent: PREVIOUS_PAIRS.previous(
            session_id, before, recent
        ),
        "catalog_loader": lambda provider: load_provider_catalog(
            screen, providers_models, provider
        ),
        "connection_prober": connection_prober,
    }


async def open_model_switcher(screen: ChatScreen, query: str = "") -> None:
    """Open Switch model for the active chat (Alt+M, chips, palette, /model).

    ``/model <query>`` passes its text here: it opens in Find, so the best
    match is highlighted and nothing applies until Enter (TASK-33004.7).

    Args:
        screen: The Console screen whose active chat Switch model edits.
        query: Text Find opens with ("" lists everything).
    """
    from ...Chat.console_settings_apply import QUICK_MODEL_DEFAULT_FIELDS
    from ...Widgets.Console.console_model_popover import ConsoleModelPopover

    # TASK-34720: one dialog however many queued requests reach here.
    if screen._console_setup_modal_blocking() or not (
        screen._owns_console_screen_stack()
    ):
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

    def commit(submission):  # type: ignore[no-untyped-def]
        live_commit = screen._commit_console_settings_submission_live(submission)
        after = (submission.draft.settings.provider, submission.draft.settings.model)
        PREVIOUS_PAIRS.record_switch(session.id, before, after)
        return live_commit

    screen.app.push_screen(
        ConsoleModelPopover(
            origin=origin,
            initial_draft=screen._console_settings_initial_draft(
                settings,
                store.session_context_policy_overrides(session_id),
                exposed_fields=QUICK_MODEL_DEFAULT_FIELDS,
            ),
            scope_copy="Applies to: this chat only",
            durability_copy=(
                "Temporary until this chat is promoted"
                if session.ephemeral
                else "Saved with the conversation after its first message"
                if session.persisted_conversation_id is None
                else "Saved with this conversation"
            ),
            live_committer=commit,
            setup_opener=lambda provider, model: open_provider_setup(
                screen, provider, model
            ),
            query=query.strip(),
            **_switcher_sources(screen, session.id, before),  # type: ignore[arg-type]
        ),
        callback=screen._apply_console_model_popover_result,
    )


def refuse_live_commit(submission: object) -> NoReturn:
    """Pick mode's committer: it applies nothing, ever (TASK-33006.4).

    Args:
        submission: The submission pick mode must never commit.

    Raises:
        ValueError: Always; the switcher shows it and keeps the chat as is.
    """
    del submission
    raise ValueError("Pick mode applies nothing; Chat settings' Apply does.")


def open_model_picker(
    screen: ChatScreen,
    origin: ConsoleSettingsOrigin,
    draft: ConsoleSettingsDraftState,
    query: str,
    on_pick: Callable[[tuple[str, str] | None], None],
    served: Mapping[str, Sequence[str]],
) -> None:
    """Open Switch model in pick-only mode over Chat settings (TASK-33006.4).

    It lists the pairs Alt+M lists, plus the models Chat settings' listings
    found, and hands the chosen one to ``on_pick`` (``None`` on Esc); Chat
    settings rebases its own draft, and nothing is applied until its Apply.
    Pick mode never calls the rebaser, the committer or Settings, and its
    committer refuses, so a regression there cannot reach the chat.

    Args:
        screen: The Console screen under Chat settings.
        origin: The chat Chat settings edits.
        draft: Chat settings' draft; its pair is marked current.
        query: Text Find opens with ("" lists everything).
        on_pick: Receives ``(provider, model)``, or ``None`` on Esc.
        served: Models Chat settings' listings found, per provider.
    """
    from ...Widgets.Console.console_model_popover import ConsoleModelPopover

    settings = draft.settings
    screen.app.push_screen(
        ConsoleModelPopover(
            origin=origin,
            initial_draft=draft,
            scope_copy="",
            durability_copy="",
            live_committer=refuse_live_commit,
            pick_only=True,
            query=query,  # "<entry name> " keeps its space for the model id
            served_models=served,
            **_switcher_sources(  # type: ignore[arg-type]
                screen, origin.session_id, (settings.provider, settings.model)
            ),
        ),
        callback=on_pick,
    )
