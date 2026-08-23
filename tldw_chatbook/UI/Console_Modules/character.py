"""Controller-owned Console character policy and avatar orchestration.

The controller owns picker projection, session handoff, active identity, card
retrieval, request snapshots, resolution/cache policy, invalidation, and
stale-result arbitration.  It intentionally has no screen or DOM handle; the
framework worker/modal edges and final Textual mutation remain named callbacks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup

from ...Character_Chat.visual_identity import normalize_expression_key
from ...Chat.console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID
from ...Chat.console_expression_state import (
    CharacterEmoteHistoryIdentity,
    resolve_console_expression_selection,
    resolve_console_expression_state,
)
from ...Chat.console_image_view import (
    resolve_react_character_expressions,
    resolve_show_character_avatar,
)
from ...Widgets.Console.console_character_picker_modal import (
    ConsoleCharacterChoice,
    ConsoleCharacterOption,
)
from ..character_display_text import sanitize_character_display_label
from .session import (
    _canonical_card_character_id,
    _character_session_prompt_seed,
    _console_global_user_display_name,
)

_EXPRESSION_SPEC_CACHE_MAX = 16

ActorScope = tuple[str, str, str]
AvatarRequest = tuple[
    ActorScope | None,
    str,
    str | None,
    int,
    str | None,
    bool,
    bool,
    str | None,
    str,
    str | None,
    CharacterEmoteHistoryIdentity | None,
]


class ConsoleCharacterController:
    """Own non-DOM character policy, identity, and avatar decisions."""

    def __init__(
        self,
        *,
        app_config_accessor: Callable[[], Mapping[str, Any]],
        chat_store_accessor: Callable[[], Any | None],
        active_native_session_accessor: Callable[[], Any | None],
        current_conversation_id_accessor: Callable[[], str | None],
        character_db_accessor: Callable[[], Any | None],
        ensure_chat_store: Callable[[], Any],
        provider_readiness_config_accessor: Callable[[], Mapping[str, Any]],
        default_session_settings: Callable[[], Any],
        swap_session_character: Callable[..., bool],
        sync_temporary_chip: Callable[[], None],
        sync_native_chat_ui: Callable[[], Awaitable[None]],
        notify: Callable[..., None],
        actor_scope_accessor: Callable[[], ActorScope | None],
        manual_reaction_key: Callable[[ActorScope], str | None],
        resolve_visual_identity: Callable[[ActorScope, str, str | None], Any | None],
        resolve_historical_visual_identity: Callable[
            [ActorScope, CharacterEmoteHistoryIdentity], Any | None
        ]
        | None = None,
        ensure_console_image_view: Callable[[], tuple[Any, Any]],
        console_image_default_mode: Callable[[], str | None],
        is_mounted: Callable[[], bool],
        render_character_avatar: Callable[..., Awaitable[None]],
    ) -> None:
        self._app_config_accessor = app_config_accessor
        self._chat_store_accessor = chat_store_accessor
        self._active_native_session_accessor = active_native_session_accessor
        self._current_conversation_id_accessor = current_conversation_id_accessor
        self._character_db_accessor = character_db_accessor
        self._ensure_chat_store = ensure_chat_store
        self._provider_readiness_config_accessor = provider_readiness_config_accessor
        self._default_session_settings = default_session_settings
        self._swap_session_character = swap_session_character
        self._sync_temporary_chip = sync_temporary_chip
        self._sync_native_chat_ui = sync_native_chat_ui
        self._notify = notify
        self._actor_scope_accessor = actor_scope_accessor
        self._manual_reaction_key = manual_reaction_key
        self._resolve_visual_identity = resolve_visual_identity
        self._resolve_historical_visual_identity = resolve_historical_visual_identity
        self._ensure_console_image_view = ensure_console_image_view
        self._console_image_default_mode = console_image_default_mode
        self._is_mounted = is_mounted
        self._render_character_avatar = render_character_avatar

        self._active_character_avatar: dict | None = None
        self._active_character_avatar_name: str | None = None
        self._last_console_avatar_scope: Any | None = None
        self._last_console_avatar_request_key: Any | None = None
        self._console_expression_spec_cache: dict[tuple[str, ...], dict] = {}
        self._character_emote_cursor_by_session: dict[str, int] = {}
        self._character_emote_explicit_by_session: dict[str, tuple[str, str]] = {}

    def _live_config(self) -> Mapping[str, Any]:
        return self._app_config_accessor() or {}

    def _console_character_picker_options(
        self,
    ) -> tuple[ConsoleCharacterOption, ...]:
        """Read selectable character cards (worker thread; never raises)."""
        db = self._character_db_accessor()
        if db is None:
            return ()
        try:
            cards = db.list_character_cards(limit=500)
        except Exception:
            logger.opt(exception=True).warning("Character picker: list failed.")
            return ()
        options: list[ConsoleCharacterOption] = []
        for card in cards or ():
            card_id = _canonical_card_character_id(card.get("id"))
            name = str(card.get("name") or "").strip()
            if card_id is None or not name:
                continue
            options.append(
                ConsoleCharacterOption(
                    character_id=card_id,
                    name=name,
                    description=str(card.get("description") or "")[:200],
                )
            )
        return tuple(options)

    def _current_console_rail_conversation_id(self) -> str | None:
        """Return the conversation scope used only for rail persistence."""
        native_session = self._active_native_session_accessor()
        if native_session is not None:
            conversation_id = getattr(
                native_session,
                "persisted_conversation_id",
                None,
            )
            return str(conversation_id) if conversation_id else None
        return self._current_conversation_id_accessor()

    def _current_console_rail_character_id(self) -> int | None:
        """Return the active native session's local character id, if any."""
        native_session = self._active_native_session_accessor()
        if native_session is None:
            return None
        return native_session.local_character_id()

    def _current_console_rail_character_name(self) -> str | None:
        """Return the active native session's character name, if any."""
        native_session = self._active_native_session_accessor()
        if native_session is None:
            return None
        name = getattr(native_session, "character_name", None)
        return str(name) if name else None

    def _fetch_character_card_for_avatar(self, character_id: int) -> dict | None:
        """Fetch one character card off-thread, failing soft on DB errors."""
        db = self._character_db_accessor()
        if db is None:
            return None
        try:
            return db.get_character_card_by_id(int(character_id))
        except Exception:
            logger.opt(exception=True).debug("avatar: character fetch failed")
            return None

    async def _apply_console_character_choice_async(
        self,
        choice: ConsoleCharacterChoice,
    ) -> None:
        """Apply a picked character to this session or a fresh one."""
        card = await asyncio.to_thread(
            self._fetch_character_card_for_avatar,
            choice.character_id,
        )
        if card is None:
            display_name = (
                sanitize_character_display_label(
                    choice.name,
                    max_characters=180,
                )
                or "that character"
            )
            self._notify(
                f"Could not load {escape_markup(display_name)}.",
                severity="error",
            )
            return

        store = self._ensure_chat_store()
        global_name = _console_global_user_display_name(
            self._provider_readiness_config_accessor()
        )
        if choice.placement == "new" or store.active_session_id is None:
            effective_name = global_name
        else:
            effective_name = store.presentation_context(
                store.active_session_id,
                global_name,
            ).user_name
        seed = _character_session_prompt_seed(
            card,
            choice.name,
            user_name=effective_name,
        )
        display_name = (
            sanitize_character_display_label(seed.name, max_characters=180)
            or "that character"
        )
        notification_name = escape_markup(display_name)
        if choice.placement == "new":
            settings = replace(
                self._default_session_settings(),
                system_prompt=seed.system_prompt,
            )
            session = store.create_session(
                title=f"Chat with {seed.name}",
                workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
                settings=settings,
                runtime_backend="local",
                assistant_kind="character",
                assistant_id=str(choice.character_id),
                assistant_authority_id=None,
                character_id=choice.character_id,
                character_name=seed.name,
            )
            try:
                store.seed_character_roleplay(
                    session.id,
                    system_template=seed.system_template,
                    greeting_template=seed.greeting_template,
                    global_default=global_name,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Character picker: roleplay template seed failed; continuing."
                )
            store.switch_session(session.id)
            self._sync_temporary_chip()
            self._notify(f"Started a new chat with {notification_name}.")
        else:
            if not self._swap_session_character(
                store,
                choice.character_id,
                seed,
                global_default=global_name,
            ):
                return
            self._notify(f"This chat now uses {notification_name}.")
        await self._sync_native_chat_ui()
        await self._refresh_active_character_avatar_if_scope_changed()

    def _current_request(self) -> AvatarRequest:
        config = self._live_config()
        store = self._chat_store_accessor()
        actor = self._actor_scope_accessor()
        session_id = getattr(store, "active_session_id", None) if store else None
        react = resolve_react_character_expressions(config)
        explicit_message_id, explicit_state = (
            self._character_emote_explicit_by_session.get(session_id, (None, None))
        )
        selection = resolve_console_expression_selection(
            store,
            session_id,
            react_enabled=react,
            explicit_message_id=explicit_message_id,
            explicit_state=explicit_state,
        )
        state = selection.state
        if selection.source in {"idle", "operational"}:
            state = resolve_console_expression_state(
                store, session_id, react_enabled=react
            )
        manual = self._manual_reaction_key(actor) if actor else None
        source = selection.source
        history_identity = selection.history_identity
        message_id = selection.message_id
        if manual is not None:
            state = "idle"
            source = "manual"
            history_identity = None
            message_id = None
        return (
            actor,
            state,
            manual,
            id(store),
            session_id,
            react,
            resolve_show_character_avatar(config),
            self._current_console_rail_character_name(),
            source,
            message_id,
            history_identity,
        )

    async def _consume_character_emote_events(self) -> bool:
        """Advance the active feed and paint each eligible live beat in order."""

        store = self._chat_store_accessor()
        session_id = getattr(store, "active_session_id", None) if store else None
        if store is None or not isinstance(session_id, str) or not session_id:
            return False
        cursor = self._character_emote_cursor_by_session.get(session_id, 0)
        try:
            events = store.character_emote_events_after(session_id, cursor)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        painted = False
        for event in events:
            self._character_emote_cursor_by_session[session_id] = event.sequence
            selection = resolve_console_expression_selection(
                store,
                session_id,
                react_enabled=resolve_react_character_expressions(self._live_config()),
                explicit_message_id=event.message_id,
                explicit_state=event.state,
            )
            if selection.source != "explicit":
                continue
            self._character_emote_explicit_by_session[session_id] = (
                event.message_id,
                event.state,
            )
            actor = self._actor_scope_accessor()
            if actor is not None and self._manual_reaction_key(actor) is None:
                await self._refresh_avatar_request(force=True)
                painted = True
        return painted

    def _request_is_current(self, request: AvatarRequest) -> bool:
        return self._current_request() == request and self._is_mounted()

    def _invalidate_actor(self, actor_kind: str, actor_id: str) -> None:
        actor_tokens = (f"actor_kind={actor_kind}", f"actor_id={actor_id}")

        def belongs_to_actor(identity: tuple[str, ...]) -> bool:
            return all(token in identity for token in actor_tokens)

        for identity in tuple(self._console_expression_spec_cache):
            if belongs_to_actor(identity):
                self._console_expression_spec_cache.pop(identity, None)
        active_identity = (
            self._active_character_avatar.get("resolution_cache_identity", ())
            if self._active_character_avatar is not None
            else ()
        )
        if belongs_to_actor(tuple(active_identity)):
            self._active_character_avatar = None
        previous_actor = (
            self._last_console_avatar_scope[0]
            if isinstance(self._last_console_avatar_scope, tuple)
            and self._last_console_avatar_scope
            else None
        )
        if (
            isinstance(previous_actor, tuple)
            and len(previous_actor) == 3
            and previous_actor[1:] == (actor_kind, actor_id)
        ):
            self._last_console_avatar_scope = None
            self._last_console_avatar_request_key = None

    def invalidate_refresh_scope(self) -> None:
        """Force the next tick to repaint after the rail becomes visible."""

        self._last_console_avatar_scope = None
        self._last_console_avatar_request_key = None

    async def _paint(
        self,
        request: AvatarRequest,
        spec: dict | None,
        *,
        name: str | None,
        manual_label: str | None,
    ) -> None:
        if not self._request_is_current(request):
            return
        self._active_character_avatar = spec
        self._active_character_avatar_name = name

        def is_current() -> bool:
            return (
                self._request_is_current(request)
                and self._active_character_avatar is spec
                and self._active_character_avatar_name == name
            )

        await self._render_character_avatar(
            spec=spec,
            name=name,
            manual_label=manual_label,
            is_current=is_current,
        )

    async def _refresh_active_character_avatar_if_scope_changed(
        self,
        *,
        force: bool = False,
        invalidate_actor: tuple[str, str] | None = None,
    ) -> None:
        """Consume live beats, then resolve one race-fenced avatar request."""
        if invalidate_actor is not None:
            self._invalidate_actor(*invalidate_actor)
            force = True

        painted_live_event = await self._consume_character_emote_events()
        if painted_live_event:
            return
        await self._refresh_avatar_request(force=force)

    async def _refresh_avatar_request(self, *, force: bool = False) -> None:
        """Resolve and apply the current immutable avatar request."""

        if not resolve_show_character_avatar(self._live_config()):
            self._active_character_avatar = None
            self._active_character_avatar_name = None
            self._last_console_avatar_scope = None
            self._last_console_avatar_request_key = None
            return

        request = self._current_request()
        actor_scope, state, manual_key = request[:3]
        source = request[8]
        history_identity = request[10]
        scope = (actor_scope, state, manual_key)
        request_key = (*scope, source, history_identity)
        if not force and request_key == self._last_console_avatar_request_key:
            return
        self._last_console_avatar_scope = scope
        self._last_console_avatar_request_key = request_key
        name = request[7]
        manual_label = (
            manual_key.rsplit(":", 1)[-1].replace("_", " ").replace("-", " ").title()
            if manual_key
            else None
        )

        if actor_scope is None:
            await self._paint(request, None, name=name, manual_label=manual_label)
            return
        _session_id, actor_kind, actor_id = actor_scope
        resolution_state = state
        resolver: Callable[[], Any | None] | None
        if source == "historical" and history_identity is not None:
            if self._resolve_historical_visual_identity is None:
                resolver = None
            else:
                resolver = partial(
                    self._resolve_historical_visual_identity,
                    actor_scope,
                    history_identity,
                )
        else:
            if source == "explicit":
                resolution_state = normalize_expression_key(state) or state
            resolver = partial(
                self._resolve_visual_identity,
                actor_scope,
                resolution_state,
                manual_key,
            )
        try:
            resolution = (
                await asyncio.to_thread(resolver) if resolver is not None else None
            )
        except Exception:  # noqa: BLE001 -- the 0.2s sync tick must fail soft.
            logger.opt(exception=True).debug("avatar: expression resolution failed")
            resolution = None
        if not self._request_is_current(request):
            return
        exact_required = source in {"explicit", "historical"}
        exact_available = (
            resolution is not None
            and resolution.image_bytes is not None
            and (
                resolution.resolution_source == "history_immutable"
                if source == "historical"
                else resolution.asset_id is not None
                and resolution.resolved_expression_key == resolution_state
            )
        )
        if exact_required and not exact_available:
            if self._active_character_avatar is not None:
                return
            resolution_state = "idle"
            resolver = partial(
                self._resolve_visual_identity, actor_scope, resolution_state, None
            )
            try:
                resolution = await asyncio.to_thread(resolver)
            except Exception:  # noqa: BLE001 -- base portrait fallback is fail-soft.
                logger.opt(exception=True).debug("avatar: base resolution failed")
                return
        if resolution is None:
            return
        identity = resolution.cache_identity
        cached = self._console_expression_spec_cache.get(identity)
        if cached is not None:
            await self._paint(request, cached, name=name, manual_label=manual_label)
            return

        _, cache = self._ensure_console_image_view()
        mode = self._console_image_default_mode() or "pixels"
        key = "visual-identity:" + "|".join(identity)
        spec = {
            "actor_kind": actor_kind,
            "actor_id": actor_id,
            "state": state,
            "name": name,
            "mode": mode,
            "pil": None,
            "pixels": None,
            "manual_expression_key": manual_key,
            "resolution_cache_identity": identity,
        }
        if actor_kind == "character":
            spec["character_id"] = int(actor_id)
        try:
            if resolution.image_bytes:
                ok = await asyncio.to_thread(cache.prepare, key, resolution.image_bytes)
                if not self._request_is_current(request):
                    return
                if ok:
                    spec["pil"] = cache.get_pil(key)
                current = (
                    await asyncio.to_thread(resolver) if resolver is not None else None
                )
                if (
                    not self._request_is_current(request)
                    or current is None
                    or current.cache_identity != identity
                ):
                    return
        except Exception:  # noqa: BLE001 -- sync-tick avatar decode is fail-soft.
            logger.opt(exception=True).debug("avatar: expression decode failed")
        if not self._request_is_current(request):
            return
        self._console_expression_spec_cache[identity] = spec
        while len(self._console_expression_spec_cache) > _EXPRESSION_SPEC_CACHE_MAX:
            del self._console_expression_spec_cache[
                next(iter(self._console_expression_spec_cache))
            ]
        await self._paint(request, spec, name=name, manual_label=manual_label)
