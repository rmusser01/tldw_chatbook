"""TASK-22204: transcript-copy budget for the Console avatar tick.

PR #2020 (streaming emotes) made `_current_request` resolve the expression
state twice and made `_request_is_current` re-enter `_current_request`, so a
repainting 0.2 s tick paid 4-12 whole-transcript `dataclasses.replace` copy
passes through `store.messages_for_session`. These probes pin the budget: a
full paint tick pays exactly ONE `messages_for_session` copy, and the
race-fence checks pay ZERO. Reintroducing an unshared second resolution (the
22204 mutation) turns the first probe red.

The harness drives the real ``ConsoleCharacterController`` and the real
``ConsoleChatStore`` (streaming assistant message included, so the
idle/operational double-resolution gate is exercised) with plain late-bound
edges, mirroring ``Tests/UI/test_console_character_controller.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Console_Modules.character import ConsoleCharacterController


async def _noop_async(*_args: object, **_kwargs: object) -> None:
    return None


class _FakeImageCache:
    """Minimal stand-in for the console image cache used by the paint path."""

    def prepare(self, _key: str, _image_bytes: bytes) -> bool:
        return True

    def get_pil(self, _key: str) -> object:
        return object()


def _operational_resolution() -> SimpleNamespace:
    """One resolvable expression identity for the operational paint path."""

    return SimpleNamespace(
        image_bytes=b"fake-png-bytes",
        asset_id=11,
        resolved_expression_key="speaking",
        resolution_source="pack_operational",
        cache_identity=(
            "actor_kind=character",
            "actor_id=7",
            "expression=speaking",
        ),
    )


def _count_transcript_copies(store: ConsoleChatStore) -> dict[str, int]:
    """Shadow ``messages_for_session`` on the instance with a call counter."""

    counter = {"copies": 0}
    original = store.messages_for_session

    def counting(session_id: str):
        counter["copies"] += 1
        return original(session_id)

    store.messages_for_session = counting  # type: ignore[method-assign]
    return counter


def _streaming_store() -> tuple[ConsoleChatStore, str]:
    """Real store with a user turn and a still-streaming assistant message."""

    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hello"
    )
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(message.id, "streamed prose")
    return store, session.id


def _avatar_controller(
    store: ConsoleChatStore | None,
    session_id: str | None,
    *,
    painted: list[dict | None],
    fence_results: list[bool] | None = None,
    is_mounted=lambda: True,
    actor_scope_accessor=None,
    resolve_visual_identity=None,
) -> ConsoleCharacterController:
    """Real controller wired for the avatar paint path with recording edges."""

    if actor_scope_accessor is None:
        scope = (session_id, "character", "7") if session_id else None
        actor_scope_accessor = lambda: scope  # noqa: E731
    if resolve_visual_identity is None:
        resolution = _operational_resolution()
        resolve_visual_identity = (  # noqa: E731
            lambda _scope, _state, _manual: resolution
        )

    async def render(
        *,
        spec: dict | None,
        name: str | None,
        manual_label: str | None,
        is_current,
    ) -> None:
        # Mirror the real screen edge: it consults the fence before mutating
        # the DOM. Under pre-22204 code every one of these re-entered
        # `_current_request` and paid 2 more transcript copies.
        alive = is_current()
        if fence_results is not None:
            fence_results.append(alive)
        painted.append(spec)

    dependencies: dict[str, Any] = {
        "app_config_accessor": lambda: {},
        "chat_store_accessor": lambda: store,
        "active_native_session_accessor": lambda: None,
        "current_conversation_id_accessor": lambda: None,
        "character_db_accessor": lambda: None,
        "ensure_chat_store": ConsoleChatStore,
        "provider_readiness_config_accessor": lambda: {},
        "default_session_settings": lambda: ConsoleSessionSettings(),
        "swap_session_character": lambda *_args, **_kwargs: False,
        "sync_temporary_chip": lambda: None,
        "sync_native_chat_ui": _noop_async,
        "notify": lambda *_args, **_kwargs: None,
        "actor_scope_accessor": actor_scope_accessor,
        "manual_reaction_key": lambda _scope: None,
        "resolve_visual_identity": resolve_visual_identity,
        "resolve_historical_visual_identity": lambda *_args: None,
        "ensure_console_image_view": lambda: (None, _FakeImageCache()),
        "console_image_default_mode": lambda: "pixels",
        "is_mounted": is_mounted,
        "render_character_avatar": render,
    }
    return ConsoleCharacterController(**dependencies)


@pytest.mark.asyncio
async def test_streaming_repaint_tick_pays_exactly_one_transcript_copy() -> None:
    """A full paint tick (build + every fence check) copies the transcript once."""

    store, session_id = _streaming_store()
    painted: list[dict | None] = []
    fence_results: list[bool] = []
    controller = _avatar_controller(
        store, session_id, painted=painted, fence_results=fence_results
    )
    counter = _count_transcript_copies(store)

    await controller._refresh_active_character_avatar_if_scope_changed()

    # The paint genuinely happened (operational "speaking" for a streaming
    # assistant message) and the fence agreed it was current -- so every
    # fence site on the paint path really ran before the count is read.
    assert fence_results == [True]
    assert len(painted) == 1
    assert painted[0] is not None
    assert painted[0]["state"] == "speaking"
    assert counter["copies"] == 1


@pytest.mark.asyncio
async def test_steady_state_tick_pays_at_most_one_transcript_copy() -> None:
    """An unchanged follow-up tick (request-key dedupe) stays within budget."""

    store, session_id = _streaming_store()
    painted: list[dict | None] = []
    controller = _avatar_controller(store, session_id, painted=painted)
    await controller._refresh_active_character_avatar_if_scope_changed()
    counter = _count_transcript_copies(store)

    await controller._refresh_active_character_avatar_if_scope_changed()

    assert len(painted) == 1  # deduped: no second paint
    assert counter["copies"] <= 1


@pytest.mark.asyncio
async def test_zero_message_session_tick_is_idle_and_within_budget() -> None:
    """A session with no messages resolves idle in at most one copy."""

    store = ConsoleChatStore()
    session = store.ensure_session(title="Empty")
    painted: list[dict | None] = []
    controller = _avatar_controller(store, session.id, painted=painted)
    counter = _count_transcript_copies(store)

    await controller._refresh_active_character_avatar_if_scope_changed()

    assert len(painted) == 1
    assert painted[0] is not None
    assert painted[0]["state"] == "idle"
    assert counter["copies"] <= 1


@pytest.mark.asyncio
async def test_mid_teardown_tick_never_paints_and_never_raises() -> None:
    """Unmounting during off-thread resolution drops the paint, fail-soft."""

    store, session_id = _streaming_store()
    painted: list[dict | None] = []
    mounted = {"value": True}
    resolution = _operational_resolution()

    def resolve_and_unmount(_scope, _state, _manual):
        # Runs inside asyncio.to_thread during the refresh: the screen goes
        # away while the expression is being resolved.
        mounted["value"] = False
        return resolution

    controller = _avatar_controller(
        store,
        session_id,
        painted=painted,
        is_mounted=lambda: mounted["value"],
        resolve_visual_identity=resolve_and_unmount,
    )

    await controller._refresh_active_character_avatar_if_scope_changed()

    assert painted == []


@pytest.mark.asyncio
async def test_storeless_teardown_tick_is_a_safe_noop() -> None:
    """A tick after the store handle is gone neither copies nor raises."""

    painted: list[dict | None] = []
    controller = _avatar_controller(
        None,
        None,
        painted=painted,
        is_mounted=lambda: False,
        actor_scope_accessor=lambda: None,
    )

    await controller._refresh_active_character_avatar_if_scope_changed()

    assert painted == []
