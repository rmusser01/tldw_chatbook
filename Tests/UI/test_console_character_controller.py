"""No-mount behavior contract for the Console character controller."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Console_Modules.character import ConsoleCharacterController
from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
    ConsoleCharacterChoice,
    ConsoleCharacterOption,
)


async def _noop_async(*_args: object, **_kwargs: object) -> None:
    return None


def _controller(**overrides: Any) -> ConsoleCharacterController:
    """Build the real controller with plain late-bound test edges."""

    dependencies: dict[str, Any] = {
        "app_config_accessor": lambda: {},
        "chat_store_accessor": lambda: None,
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
        "actor_scope_accessor": lambda: None,
        "manual_reaction_key": lambda _scope: None,
        "resolve_visual_identity": lambda *_args: None,
        "ensure_console_image_view": lambda: (None, None),
        "console_image_default_mode": lambda: "pixels",
        "is_mounted": lambda: False,
        "render_character_avatar": _noop_async,
    }
    dependencies.update(overrides)
    return ConsoleCharacterController(**dependencies)


def test_character_picker_projection_is_bounded_and_fail_closed() -> None:
    class CharacterDb:
        def list_character_cards(self, *, limit: int) -> list[dict[str, object]]:
            assert limit == 500
            return [
                {"id": "7", "name": "  Alraune  ", "description": "forest"},
                {"id": None, "name": "Missing identity"},
                {"id": 8, "name": ""},
                {"id": 9, "name": "Brynn", "description": None},
            ]

        def get_character_card_by_id(self, character_id: int) -> dict[str, object]:
            assert character_id == 7
            return {"id": character_id, "name": "Alraune"}

    controller = _controller(character_db_accessor=CharacterDb)

    assert controller._console_character_picker_options() == (
        ConsoleCharacterOption(character_id=7, name="Alraune", description="forest"),
        ConsoleCharacterOption(
            character_id=9,
            name="Brynn",
            description="",
        ),
    )
    assert controller._fetch_character_card_for_avatar(7) == {
        "id": 7,
        "name": "Alraune",
    }

    class FailingDb:
        def list_character_cards(self, *, limit: int) -> list[dict[str, object]]:
            raise RuntimeError(limit)

        def get_character_card_by_id(self, character_id: int) -> dict[str, object]:
            raise RuntimeError(character_id)

    controller._character_db_accessor = FailingDb
    assert controller._console_character_picker_options() == ()
    assert controller._fetch_character_card_for_avatar(7) is None


def test_character_identity_reads_active_session_and_conversation_fallback() -> None:
    active = SimpleNamespace(
        persisted_conversation_id="conversation-A",
        character_name="Alraune",
        local_character_id=lambda: 7,
    )
    current = active
    controller = _controller(
        active_native_session_accessor=lambda: current,
        current_conversation_id_accessor=lambda: "fallback-conversation",
    )

    assert controller._current_console_rail_conversation_id() == "conversation-A"
    assert controller._current_console_rail_character_id() == 7
    assert controller._current_console_rail_character_name() == "Alraune"

    current = None
    assert controller._current_console_rail_conversation_id() == (
        "fallback-conversation"
    )
    assert controller._current_console_rail_character_id() is None
    assert controller._current_console_rail_character_name() is None


@pytest.mark.asyncio
async def test_new_character_choice_preserves_prompt_seed_and_sync_order() -> None:
    store = ConsoleChatStore()
    notifications: list[tuple[str, str | None]] = []
    syncs: list[str] = []
    card = {
        "id": 7,
        "name": "Alraune",
        "system_prompt": "Protect {{user}} as {{character}}.",
        "first_message": "Hello, {{user}}.",
    }

    async def sync_ui() -> None:
        syncs.append("ui")

    controller = _controller(
        chat_store_accessor=lambda: store,
        character_db_accessor=lambda: SimpleNamespace(
            get_character_card_by_id=lambda character_id: (
                card if character_id == 7 else None
            )
        ),
        ensure_chat_store=lambda: store,
        provider_readiness_config_accessor=lambda: {
            "chat_defaults": {"user_display_name": "Captain Rowan"}
        },
        default_session_settings=lambda: ConsoleSessionSettings(
            provider="openai",
            model="gpt-4.1",
        ),
        sync_temporary_chip=lambda: syncs.append("temporary"),
        sync_native_chat_ui=sync_ui,
        notify=lambda message, severity=None: notifications.append((message, severity)),
    )

    await controller._apply_console_character_choice_async(
        ConsoleCharacterChoice(
            character_id=7,
            name="Alraune",
            placement="new",
        )
    )

    session = store.switch_session(store.active_session_id)
    assert session.character_id == 7
    assert session.character_name == "Alraune"
    assert session.settings.system_prompt == "Protect Captain Rowan as Alraune."
    messages = store.messages_for_session(session.id)
    assert [message.content for message in messages] == ["Hello, Captain Rowan."]
    assert syncs == ["temporary", "ui"]
    assert notifications == [("Started a new chat with Alraune.", None)]


@pytest.mark.asyncio
async def test_current_character_choice_uses_named_swap_edge() -> None:
    store = ConsoleChatStore()
    session = store.ensure_session(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-4.1")
    )
    store.set_session_user_display_name_override(
        session.id,
        "Per Chat",
        global_default="Global User",
    )
    card = {
        "id": 8,
        "name": "Brynn",
        "system_prompt": "Guard {{user}} beside {{character}}.",
        "first_message": "Hello, {{user}}.",
    }
    swaps: list[tuple[object, int, str, str]] = []
    syncs: list[str] = []
    notifications: list[tuple[str, str | None]] = []

    def swap(
        actual_store: ConsoleChatStore,
        character_id: int,
        seed: Any,
        *,
        global_default: str,
    ) -> bool:
        swaps.append((actual_store, character_id, seed.system_prompt, global_default))
        return True

    controller = _controller(
        chat_store_accessor=lambda: store,
        character_db_accessor=lambda: SimpleNamespace(
            get_character_card_by_id=lambda _character_id: card
        ),
        ensure_chat_store=lambda: store,
        provider_readiness_config_accessor=lambda: {
            "chat_defaults": {"user_display_name": "Global User"}
        },
        swap_session_character=swap,
        sync_native_chat_ui=lambda: _record_async(syncs, "ui"),
        notify=lambda message, severity=None: notifications.append((message, severity)),
    )

    await controller._apply_console_character_choice_async(
        ConsoleCharacterChoice(
            character_id=8,
            name="Brynn",
            placement="current",
        )
    )

    assert swaps == [(store, 8, "Guard Per Chat beside Brynn.", "Global User")]
    assert syncs == ["ui"]
    assert notifications == [("This chat now uses Brynn.", None)]


async def _record_async(records: list[str], value: str) -> None:
    records.append(value)
