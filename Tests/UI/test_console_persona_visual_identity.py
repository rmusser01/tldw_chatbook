"""Console contracts for local Persona Shared Visual Identity reactions."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Character_Chat.visual_identity import VisualIdentityResolution
from tldw_chatbook.UI.Console_Modules import session as session_module
from tldw_chatbook.UI.Console_Modules.character import ConsoleCharacterController
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.Widgets.Console.console_reaction_picker_modal import ReactionOption


def _persona_record(*, revision: int = 2, active: bool = True) -> dict:
    return {
        "backend": "local",
        "id": "persona-alpha",
        "version": revision,
        "is_active": active,
        "deleted": False,
    }


def _session(*, backend: str = "local") -> ConsoleChatSession:
    return ConsoleChatSession(
        id="session-a",
        runtime_backend=backend,
        assistant_kind="persona",
        assistant_id="persona-alpha",
    )


def _session_controller(session: ConsoleChatSession) -> ConsoleSessionController:
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._active_native_console_session = lambda: session
    controller._manual_reaction_overrides = {}
    local = Mock()
    local.get_persona_profile.return_value = _persona_record()
    controller.app_instance = SimpleNamespace(
        character_persona_scope_service=SimpleNamespace(local_service=local),
    )
    return controller


def _resolution() -> VisualIdentityResolution:
    return VisualIdentityResolution(
        actor_kind="persona",
        actor_id="persona-alpha",
        requested_expression_key="neutral",
        manual_expression_key=None,
        resolved_expression_key="neutral",
        pack_id=7,
        pack_version_id=8,
        asset_id=9,
        expression_id=None,
        storage_source="manual",
        storage_relpath=None,
        content_type="image/png",
        is_animated=False,
        resolution_source="pack_default",
        fallback_reason="none",
        cache_identity=(
            "visual-identity-v1",
            "actor_kind=persona",
            "actor_id=persona-alpha",
            "persona_revision=2",
            "binding_version=3",
            "pack_version_id=8",
            "asset_id=9",
            "sha256=abc",
        ),
        image_bytes=b"image",
    )


def test_current_visual_identity_scope_returns_local_persona_id_without_integer_coercion() -> (
    None
):
    controller = _session_controller(_session())

    assert controller._current_visual_identity_actor_scope() == (
        "session-a",
        "persona",
        "persona-alpha",
    )


def test_server_persona_never_claims_local_visual_identity() -> None:
    controller = _session_controller(_session(backend="server"))

    assert controller._current_visual_identity_actor_scope() is None


def test_persona_resolution_uses_exact_local_service_and_opaque_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = object()
    service = object()
    expected = _resolution()
    resolver = Mock(return_value=expected)
    monkeypatch.setattr(session_module, "resolve_persona_visual_identity", resolver)

    actual = session_module._resolve_visual_identity_for_db(
        db,
        ("session-a", "persona", "persona-alpha"),
        "thinking",
        "custom:relief",
        service,
    )

    assert actual is expected
    resolver.assert_called_once_with(
        db,
        service,
        persona_id="persona-alpha",
        requested_state="thinking",
        manual_expression_key="custom:relief",
    )


def test_inactive_persona_has_no_reaction_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = Mock()
    local.get_persona_profile.return_value = _persona_record(active=False)
    repository = Mock(side_effect=AssertionError("graph read must not start"))
    monkeypatch.setattr(session_module, "VisualIdentityRepository", repository)

    assert (
        session_module._visual_identity_options_for_db(
            object(),
            ("session-a", "persona", "persona-alpha"),
            local,
        )
        == ()
    )
    repository.assert_not_called()


@pytest.mark.asyncio
async def test_persona_picker_discards_inventory_after_local_service_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _session_controller(_session())
    controller._screen = SimpleNamespace(app=SimpleNamespace(push_screen=Mock()))
    store = SimpleNamespace(active_session_id="session-a")
    controller._current_chat_store_accessor = lambda: store
    controller._visual_identity_db_accessor = lambda: db
    controller.app_instance.app_config = {
        "console": {"react_character_expressions": True}
    }
    db = object()
    first_service = controller._local_persona_visual_identity_service()
    started = threading.Event()
    release = threading.Event()
    option = ReactionOption("custom:relief", "Relief", "image/png", False)

    def blocked_options(_db, _scope, service):
        assert service is first_service
        started.set()
        assert release.wait(timeout=5)
        return (option,)

    monkeypatch.setattr(
        session_module, "_visual_identity_options_for_db", blocked_options
    )
    pending = asyncio.create_task(controller._open_console_reaction_picker())
    assert await asyncio.to_thread(started.wait, 5)
    controller.app_instance.character_persona_scope_service.local_service = object()
    release.set()
    await pending

    controller._screen.app.push_screen.assert_not_called()


def test_persona_replacement_clears_only_prior_session_manual_reaction() -> None:
    controller = _session_controller(_session())
    controller._manual_reaction_overrides = {
        ("session-a", "persona", "persona-old"): "custom:alarm",
        ("session-a", "persona", "persona-alpha"): "custom:relief",
        ("session-b", "persona", "persona-old"): "custom:love",
    }

    controller._clear_replaced_actor_reactions(
        "session-a", actor_kind="persona", actor_id="persona-alpha"
    )

    assert controller._manual_reaction_overrides == {
        ("session-a", "persona", "persona-alpha"): "custom:relief",
        ("session-b", "persona", "persona-old"): "custom:love",
    }


def test_console_persona_manual_reaction_is_session_and_actor_scoped() -> None:
    controller = _session_controller(_session())
    first = ("session-a", "persona", "persona-alpha")
    second_session = ("session-b", "persona", "persona-alpha")
    second_actor = ("session-a", "persona", "persona-beta")

    controller._set_manual_reaction(first, "custom:relief")
    controller._set_manual_reaction(second_session, "custom:alarm")
    controller._set_manual_reaction(second_actor, "custom:love")
    controller._clear_manual_reaction(first)

    assert controller._manual_reaction_overrides == {
        second_session: "custom:alarm",
        second_actor: "custom:love",
    }


@pytest.mark.asyncio
async def test_console_persona_expression_resolves_and_paints_active_asset() -> None:
    scope = ("session-a", "persona", "persona-alpha")
    render = AsyncMock()
    cache = SimpleNamespace(
        prepare=lambda _key, _data: True,
        get_pil=lambda _key: "decoded-persona-frame",
    )
    store = SimpleNamespace(active_session_id="session-a")
    controller = ConsoleCharacterController(
        app_config_accessor=lambda: {
            "console": {"show_character_avatar": True},
            "chat": {"images": {"show_character_avatar": True}},
        },
        chat_store_accessor=lambda: store,
        active_native_session_accessor=lambda: _session(),
        current_conversation_id_accessor=lambda: None,
        character_db_accessor=lambda: None,
        ensure_chat_store=lambda: None,
        provider_readiness_config_accessor=lambda: {},
        default_session_settings=lambda: None,
        swap_session_character=lambda *_args, **_kwargs: False,
        sync_temporary_chip=lambda: None,
        sync_native_chat_ui=AsyncMock(),
        notify=lambda *_args, **_kwargs: None,
        actor_scope_accessor=lambda: scope,
        manual_reaction_key=lambda _scope: None,
        resolve_visual_identity=lambda *_args: _resolution(),
        ensure_console_image_view=lambda: (None, cache),
        console_image_default_mode=lambda: "pixels",
        is_mounted=lambda: True,
        render_character_avatar=render,
    )

    await controller._refresh_active_character_avatar_if_scope_changed(force=True)

    spec = controller._active_character_avatar
    assert spec["actor_kind"] == "persona"
    assert spec["actor_id"] == "persona-alpha"
    assert spec["pil"] == "decoded-persona-frame"
    assert "character_id" not in spec
    render.assert_awaited_once()


@pytest.mark.asyncio
async def test_persona_source_revision_binding_or_asset_change_drops_stale_decode() -> (
    None
):
    scope = ("session-a", "persona", "persona-alpha")
    store = SimpleNamespace(active_session_id="session-a")
    first = _resolution()
    changed = replace(
        first,
        cache_identity=(*first.cache_identity[:-1], "sha256=changed"),
    )
    resolutions = iter((first, changed))
    render = AsyncMock()
    controller = ConsoleCharacterController(
        app_config_accessor=lambda: {
            "console": {"show_character_avatar": True},
            "chat": {"images": {"show_character_avatar": True}},
        },
        chat_store_accessor=lambda: store,
        active_native_session_accessor=lambda: _session(),
        current_conversation_id_accessor=lambda: None,
        character_db_accessor=lambda: None,
        ensure_chat_store=lambda: None,
        provider_readiness_config_accessor=lambda: {},
        default_session_settings=lambda: None,
        swap_session_character=lambda *_args, **_kwargs: False,
        sync_temporary_chip=lambda: None,
        sync_native_chat_ui=AsyncMock(),
        notify=lambda *_args, **_kwargs: None,
        actor_scope_accessor=lambda: scope,
        manual_reaction_key=lambda _scope: None,
        resolve_visual_identity=lambda *_args: next(resolutions),
        ensure_console_image_view=lambda: (
            None,
            SimpleNamespace(
                prepare=lambda _key, _data: True,
                get_pil=lambda _key: "stale-frame",
            ),
        ),
        console_image_default_mode=lambda: "pixels",
        is_mounted=lambda: True,
        render_character_avatar=render,
    )

    await controller._refresh_active_character_avatar_if_scope_changed(force=True)

    assert controller._active_character_avatar is None
    render.assert_not_awaited()


@pytest.mark.asyncio
async def test_persona_publication_invalidates_only_matching_actor_cache() -> None:
    controller = ConsoleCharacterController.__new__(ConsoleCharacterController)
    persona = _resolution().cache_identity
    character = tuple(
        "actor_kind=character" if item == "actor_kind=persona" else item
        for item in persona
    )
    controller._console_expression_spec_cache = {
        persona: {"actor_kind": "persona"},
        character: {"actor_kind": "character"},
    }
    controller._active_character_avatar = None
    controller._last_console_avatar_scope = None

    controller._invalidate_actor("persona", "persona-alpha")

    assert persona not in controller._console_expression_spec_cache
    assert character in controller._console_expression_spec_cache


def test_persona_expression_does_not_touch_buddy_state() -> None:
    controller = _session_controller(_session())
    buddy = SimpleNamespace(generation=11, leases={"voice": "listening"})
    controller.app_instance.persona_buddy_controller = buddy
    scope = controller._current_visual_identity_actor_scope()

    controller._set_manual_reaction(scope, "custom:relief")

    assert buddy.generation == 11
    assert buddy.leases == {"voice": "listening"}
