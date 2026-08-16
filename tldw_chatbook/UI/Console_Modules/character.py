"""Controller-owned Console character avatar refresh orchestration.

The controller owns request snapshots, resolution/cache policy, invalidation,
and stale-result arbitration.  It intentionally has no screen or DOM handle;
the final Textual mutation is supplied as one named, fenced callback.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from loguru import logger

from ...Chat.console_expression_state import resolve_console_expression_state
from ...Chat.console_image_view import (
    resolve_react_character_expressions,
    resolve_show_character_avatar,
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
]


class ConsoleCharacterController:
    """Own non-DOM character-avatar state and refresh decisions."""

    def __init__(
        self,
        *,
        app_config_accessor: Callable[[], Mapping[str, Any]],
        chat_store_accessor: Callable[[], Any | None],
        actor_scope_accessor: Callable[[], ActorScope | None],
        character_name_accessor: Callable[[], str | None],
        manual_reaction_key: Callable[[ActorScope], str | None],
        resolve_visual_identity: Callable[[ActorScope, str, str | None], Any | None],
        ensure_console_image_view: Callable[[], tuple[Any, Any]],
        console_image_default_mode: Callable[[], str | None],
        is_mounted: Callable[[], bool],
        render_character_avatar: Callable[..., Awaitable[None]],
    ) -> None:
        self._app_config_accessor = app_config_accessor
        self._chat_store_accessor = chat_store_accessor
        self._actor_scope_accessor = actor_scope_accessor
        self._character_name_accessor = character_name_accessor
        self._manual_reaction_key = manual_reaction_key
        self._resolve_visual_identity = resolve_visual_identity
        self._ensure_console_image_view = ensure_console_image_view
        self._console_image_default_mode = console_image_default_mode
        self._is_mounted = is_mounted
        self._render_character_avatar = render_character_avatar

        self._active_character_avatar: dict | None = None
        self._active_character_avatar_name: str | None = None
        self._last_console_avatar_scope: Any | None = None
        self._console_expression_spec_cache: dict[tuple[str, ...], dict] = {}

    def _live_config(self) -> Mapping[str, Any]:
        return self._app_config_accessor() or {}

    def _current_request(self) -> AvatarRequest:
        config = self._live_config()
        store = self._chat_store_accessor()
        actor = self._actor_scope_accessor()
        session_id = getattr(store, "active_session_id", None) if store else None
        react = resolve_react_character_expressions(config)
        state = resolve_console_expression_state(store, session_id, react_enabled=react)
        manual = self._manual_reaction_key(actor) if actor else None
        return (
            actor,
            state,
            manual,
            id(store),
            session_id,
            react,
            resolve_show_character_avatar(config),
            self._character_name_accessor(),
        )

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

    def invalidate_refresh_scope(self) -> None:
        """Force the next tick to repaint after the rail becomes visible."""

        self._last_console_avatar_scope = None

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
        """Resolve and apply one race-fenced Visual Identity avatar."""
        if invalidate_actor is not None:
            self._invalidate_actor(*invalidate_actor)
            force = True

        if not resolve_show_character_avatar(self._live_config()):
            self._active_character_avatar = None
            self._active_character_avatar_name = None
            self._last_console_avatar_scope = None
            return

        request = self._current_request()
        actor_scope, state, manual_key = request[:3]
        scope = (actor_scope, state, manual_key)
        if not force and scope == self._last_console_avatar_scope:
            return
        self._last_console_avatar_scope = scope
        name = request[-1]
        manual_label = (
            manual_key.rsplit(":", 1)[-1].replace("_", " ").replace("-", " ").title()
            if manual_key
            else None
        )

        if actor_scope is None:
            await self._paint(request, None, name=name, manual_label=manual_label)
            return
        character_id = int(actor_scope[2])
        resolution = await asyncio.to_thread(
            self._resolve_visual_identity,
            actor_scope,
            state,
            manual_key,
        )
        if not self._request_is_current(request):
            return
        if resolution is None:
            await self._paint(request, None, name=name, manual_label=manual_label)
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
            "character_id": character_id,
            "state": state,
            "name": name,
            "mode": mode,
            "pil": None,
            "pixels": None,
            "manual_expression_key": manual_key,
            "resolution_cache_identity": identity,
        }
        try:
            if resolution.image_bytes:
                ok = await asyncio.to_thread(cache.prepare, key, resolution.image_bytes)
                if not self._request_is_current(request):
                    return
                if ok:
                    spec["pil"] = cache.get_pil(key)
                current = await asyncio.to_thread(
                    self._resolve_visual_identity,
                    actor_scope,
                    state,
                    manual_key,
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
