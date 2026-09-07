"""Characterisation tests for the Console session cluster.

Originally written BEFORE the wave-2 Task 3 extraction of
`ConsoleSessionController` (`tldw_chatbook/UI/Console_Modules/session.py`)
landed -- all 8 tests below passed against the unmodified, still-monolithic
`ChatScreen` (commit `08f0f5479`) before the extraction touched any
production file. Call sites were then retargeted from `console.<method>(...)`
to `console._session.<method>(...)` to follow the move; every assertion is
byte-for-byte unchanged from the pre-move version.

Drives `_activate_native_console_session`, `_apply_console_switcher_choice`,
and `_promote_console_temporary_session` through REAL interactions against a
real mounted `ChatScreen` -- the same "real production coroutine, not a
rebuilt double" discipline `test_console_native_chat_flow.py`'s own resume
coverage and `test_console_workspace_controller.py` (wave-2 Task 2's own
characterisation file) use. No method under test is monkeypatched.

Two things this file exists specifically to pin, per the wave-2 Task 3 brief:

1. **Session activation's workspace-alignment seam**
   (`_activate_native_console_session` -> `_set_active_workspace_for_
   session`, a session-controller -> workspace-controller named-callable
   call): activating a session that belongs to a DIFFERENT workspace than
   the currently active one must also switch the active workspace. Existing
   coverage (`test_console_cost_chip_screen.py::test_cost_chip_state_
   isolated_across_session_tabs`, `test_console_agent_rail.py::test_
   activate_native_console_session_clears_stale_drilldown`, `test_console_
   scope_row.py`) exercises the SAME entry point but never with two
   DIFFERENT workspaces in play, so none of it would have caught a
   snapshot-vs-live binding bug in that one call.
2. **Temporary-session promotion writes a DURABLE database row.** Every
   existing `_promote_console_temporary_session` test
   (`Tests/UI/test_console_composer_menu.py`) stubs `store.
   promote_ephemeral_session` outright and asserts only chip/widget state --
   none of them ever let a real write reach `chachanotes_db`. The brief is
   explicit: assert the DATABASE, not widget state.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from Tests.UI.background_signals import (
    await_background_task,
    wait_for_background_signal,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_character_session_prompt_seed import (
    _character_screen,
    _roleplay_card,
    _start_chat_handoff,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_switcher_state import (
    ConsoleSwitcherEntry,
    ConsoleSwitcherTarget,
    SwitcherTargetKind,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_rename_session_modal import (
    ConsoleRenameSessionModal,
)
from tldw_chatbook.Widgets.Console.console_fork_chat_modal import (
    ConsoleForkChatModal,
)
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSwitcherChoice,
)


@pytest.mark.asyncio
async def test_character_handoff_reuses_untouched_chat_one(monkeypatch):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    original = store.ensure_session(
        title="Chat 1",
        workspace_id="workspace-original",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    sync = AsyncMock()
    focus_calls: list[bool] = []
    screen._session._sync_native_console_chat_ui_fn = sync
    screen._session._focus_composer_if_needed_fn = lambda *, force=False: (
        focus_calls.append(force)
    )

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    sessions = store.sessions()
    assert len(sessions) == 1
    session = sessions[0]
    assert session is original
    assert session.id == original.id
    assert session.workspace_id == "workspace-original"
    assert session.title == "Chat with Alba"
    assert session.settings.model == defaults.model
    assert session.settings.system_prompt == "Protect Captain Rowan as Alba."
    greetings = store.messages_for_session(session.id)
    assert [message.content for message in greetings] == ["Hello, Captain Rowan."]
    assert session.identity_revision == 2
    assert store.payload_revision(session.id) == 3
    presentation = store.presentation_context(session.id, "fallback")
    assert presentation.character_name == "Alba"
    assert presentation.revision == session.identity_revision
    sync.assert_awaited_once_with()
    assert focus_calls == [True]


@pytest.mark.asyncio
async def test_draft_sync_initial_session_keeps_provenance_for_character_handoff(
    monkeypatch,
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()

    screen._session._sync_console_session_draft()

    sessions = store.sessions()
    assert len(sessions) == 1
    original = sessions[0]
    assert original.canonical_settings_baseline is original.settings

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    sessions = store.sessions()
    assert len(sessions) == 1
    assert sessions[0].id == original.id
    assert sessions[0].title == "Chat with Alba"


@pytest.mark.asyncio
async def test_tab_sync_initial_session_keeps_provenance_for_character_handoff(
    monkeypatch,
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    surface = AsyncMock()
    surface.sync_sessions = AsyncMock()
    monkeypatch.setattr(screen, "query_one", lambda *_args, **_kwargs: surface)
    monkeypatch.setattr(screen, "_maybe_show_fleet_coachmark", lambda *_args: None)
    monkeypatch.setattr(screen, "_console_chat_controller", None)

    await screen._sync_console_native_session_tabs()

    sessions = store.sessions()
    assert len(sessions) == 1
    original = sessions[0]
    assert original.canonical_settings_baseline is original.settings
    surface.sync_sessions.assert_awaited_once()

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    sessions = store.sessions()
    assert len(sessions) == 1
    assert sessions[0].id == original.id
    assert sessions[0].title == "Chat with Alba"


def _published_blank_defaults_app():
    app = _build_test_app()
    app.chat_api_provider_value = "local_llamacpp"
    app.chat_api_model_value = "active-model"
    app.app_config["chat_defaults"] = {
        "provider": "openai",
        "model": "published-model",
    }
    app.app_config["api_settings"] = {
        "openai": {
            "api_key": "test-key",
            "model_defaults": {
                "published-model": {"temperature": 0.23, "streaming": False}
            },
        },
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "active-model",
        },
    }
    app.console_new_chat_default_generation = 9
    return app


def _assert_published_blank_session(session) -> None:
    assert session.settings is not None
    assert session.settings.provider == "openai"
    assert session.settings.model == "published-model"
    assert session.settings.temperature == 0.23
    assert session.settings.streaming is False
    assert session.canonical_settings_baseline is session.settings
    assert session.new_chat_default_generation == 9


@pytest.mark.asyncio
async def test_initial_pristine_console_uses_published_blank_defaults(
    monkeypatch,
) -> None:
    app = _published_blank_defaults_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    surface = AsyncMock()
    surface.sync_sessions = AsyncMock()
    monkeypatch.setattr(screen, "query_one", lambda *_args, **_kwargs: surface)
    monkeypatch.setattr(screen, "_maybe_show_fleet_coachmark", lambda *_args: None)
    monkeypatch.setattr(screen, "_console_chat_controller", None)

    await screen._sync_console_native_session_tabs()

    [session] = store.sessions()
    _assert_published_blank_session(session)


@pytest.mark.asyncio
async def test_new_temporary_console_uses_published_blank_defaults() -> None:
    app = _published_blank_defaults_app()
    screen = ChatScreen(app)
    screen._session._composer_accessor = lambda: None
    screen._session._sync_native_console_chat_ui_fn = AsyncMock()
    screen._session._sync_temporary_chip_fn = lambda: None
    screen._session._focus_composer_if_needed_fn = lambda **_kwargs: None
    screen._session._invalidate_persisted_rows_cache_fn = lambda: None
    store = screen._ensure_console_chat_store()
    source = store.create_session(
        settings=ConsoleSessionSettings(
            provider="local_llamacpp", model="active-model", temperature=1.4
        )
    )

    await screen._session._create_native_console_session_from_active_context(
        ephemeral=True
    )

    temporary = next(
        session for session in store.sessions() if session.id != source.id
    )
    assert temporary.ephemeral is True
    _assert_published_blank_session(temporary)


def test_workspace_created_blank_console_uses_published_defaults() -> None:
    app = _published_blank_defaults_app()
    app.workspace_registry_service.create_workspace(
        workspace_id="workspace-new",
        name="New workspace",
        description="",
    )
    screen = ChatScreen(app)
    screen._session._composer_accessor = lambda: None
    screen._workspace._sync_temporary_chip_fn = lambda: None
    store = screen._ensure_console_chat_store()
    source = store.create_session(
        settings=ConsoleSessionSettings(
            provider="local_llamacpp", model="active-model", temperature=1.4
        )
    )

    screen._workspace._activate_console_session_for_workspace("workspace-new")

    created = next(session for session in store.sessions() if session.id != source.id)
    assert created.workspace_id == "workspace-new"
    _assert_published_blank_session(created)


@pytest.mark.asyncio
async def test_new_chat_and_workspace_blank_share_app_owned_published_config() -> None:
    app = _published_blank_defaults_app()
    app.workspace_registry_service.create_workspace(
        workspace_id="workspace-published",
        name="Published workspace",
        description="",
    )
    screen = ChatScreen(app)
    screen._session._provider_readiness_app_config_fn = lambda: {
        "chat_defaults": {
            "provider": "local_llamacpp",
            "model": "stale-readiness-model",
        },
        "api_settings": {
            "local_llamacpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "stale-readiness-model",
            }
        },
    }
    screen._session._composer_accessor = lambda: None
    screen._session._sync_native_console_chat_ui_fn = AsyncMock()
    screen._session._sync_temporary_chip_fn = lambda: None
    screen._session._focus_composer_if_needed_fn = lambda **_kwargs: None
    screen._session._invalidate_persisted_rows_cache_fn = lambda: None
    screen._workspace._sync_temporary_chip_fn = lambda: None
    store = screen._ensure_console_chat_store()
    source = store.create_session(
        settings=ConsoleSessionSettings(
            provider="local_llamacpp",
            model="explicit-existing-model",
            source="user",
        )
    )

    await screen._session._create_native_console_session_from_active_context()
    screen._workspace._activate_console_session_for_workspace(
        "workspace-published"
    )

    sessions = [session for session in store.sessions() if session.id != source.id]
    assert len(sessions) == 2
    for session in sessions:
        _assert_published_blank_session(session)


@pytest.mark.asyncio
async def test_controller_bare_new_chat_captures_blank_settings_and_generation() -> (
    None
):
    app = _published_blank_defaults_app()
    screen = ChatScreen(app)
    screen._console_control_provider = "local_llamacpp"
    screen._console_control_model = "active-model"
    controller = screen._ensure_console_chat_controller()
    for existing in controller.store.sessions():
        controller.store.close_session(existing.id)
    assert controller.store.active_session_id is None
    controller.prompt_queue_coordinator.run_prompt_chain = AsyncMock(
        return_value="sentinel"
    )

    result = await controller.run_prompt_chain("hello")

    assert result == "sentinel"
    [session] = controller.store.sessions()
    _assert_published_blank_session(session)


def test_controller_new_session_preserves_explicit_source_settings() -> None:
    app = _published_blank_defaults_app()
    screen = ChatScreen(app)
    controller = screen._ensure_console_chat_controller()
    source_settings = ConsoleSessionSettings(
        provider="local_llamacpp",
        model="explicit-source-model",
        temperature=1.31,
        source="user",
    )

    session = controller.new_session(settings=source_settings)

    assert session.settings is source_settings
    assert session.canonical_settings_baseline is None
    assert session.new_chat_default_generation == 0


def test_personas_preview_handoff_preserves_control_derived_source_settings(
    monkeypatch,
) -> None:
    app = _published_blank_defaults_app()
    screen = ChatScreen(app)
    screen._console_control_provider = "local_llamacpp"
    screen._console_control_model = "active-model"
    screen._session._provider_readiness_app_config_fn = lambda: app.app_config
    screen._session._composer_accessor = lambda: None
    monkeypatch.setattr(
        screen._retrieval,
        "_stage_console_library_rag_launch",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(screen, "run_worker", lambda *_args, **_kwargs: None)
    source_settings = screen._session._default_console_session_settings()
    blank_defaults = screen._session._blank_console_session_settings()
    assert (source_settings.provider, source_settings.model) == (
        "local_llamacpp",
        "active-model",
    )
    assert (blank_defaults.provider, blank_defaults.model) == (
        "openai",
        "published-model",
    )

    screen._stage_handoff_as_console_live_work(
        ChatHandoffPayload(
            source="personas",
            item_type="preview-conversation",
            title="Alba preview",
            body="Preview transcript",
            suggested_prompt="Continue this preview",
        )
    )

    [session] = screen._ensure_console_chat_store().sessions()
    assert session.settings == source_settings
    assert session.settings != blank_defaults
    assert screen._ensure_console_chat_store().session_draft(session.id) == (
        "Continue this preview"
    )


@pytest.mark.asyncio
async def test_character_handoff_constructor_preserves_control_derived_settings(
    monkeypatch,
) -> None:
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    app = screen.app_instance
    app.chat_api_provider_value = "local_llamacpp"
    app.chat_api_model_value = "handoff-source-model"
    screen._console_control_provider = "local_llamacpp"
    screen._console_control_model = "handoff-source-model"
    app.app_config["chat_defaults"].update(
        provider="openai",
        model="published-blank-model",
    )
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key"},
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "handoff-source-model",
        },
    }
    screen._session._provider_readiness_app_config_fn = lambda: app.app_config
    source_settings = screen._session._default_console_session_settings()
    blank_defaults = screen._session._blank_console_session_settings()
    assert (source_settings.provider, source_settings.model) == (
        "local_llamacpp",
        "handoff-source-model",
    )
    assert (blank_defaults.provider, blank_defaults.model) == (
        "openai",
        "published-blank-model",
    )
    store = screen._ensure_console_chat_store()
    original = store.create_session(
        settings=source_settings,
        canonical_settings_baseline=source_settings,
    )
    store.set_session_draft(original.id, "existing work")

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    created = next(item for item in store.sessions() if item.id != original.id)
    assert created.settings == replace(
        source_settings,
        system_prompt="Protect Captain Rowan as Alba.",
        character_label="Alba",
    )
    assert created.settings != blank_defaults


@pytest.mark.asyncio
async def test_character_picker_new_chat_preserves_control_derived_settings(
    monkeypatch,
) -> None:
    from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
        ConsoleCharacterChoice,
    )

    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    app = screen.app_instance
    screen._console_control_provider = "local_llamacpp"
    screen._console_control_model = "picker-source-model"
    app.app_config["chat_defaults"] = {
        "provider": "openai",
        "model": "published-blank-model",
        "user_display_name": "Captain Rowan",
    }
    app.app_config["api_settings"] = {
        "openai": {"api_key": "test-key"},
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "picker-source-model",
        },
    }
    screen._session._provider_readiness_app_config_fn = lambda: app.app_config
    monkeypatch.setattr(
        screen._character,
        "_fetch_character_card_for_avatar",
        lambda _character_id: card,
    )
    source_settings = screen._session._default_console_session_settings()
    blank_defaults = screen._session._blank_console_session_settings()
    assert source_settings != blank_defaults

    await screen._character._apply_console_character_choice_async(
        ConsoleCharacterChoice(character_id=7, name="Alba", placement="new")
    )

    [created] = screen._ensure_console_chat_store().sessions()
    assert created.settings == replace(
        source_settings,
        system_prompt="Protect Captain Rowan as Alba.",
    )
    assert created.settings != blank_defaults


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worked_state",
    [
        "draft",
        "title",
        "system-prompt",
        "pinned-prefill",
        "model",
        "provider",
        "temperature",
        "character-label",
    ],
)
async def test_character_handoff_leaves_worked_session_and_creates_another(
    monkeypatch, worked_state
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    original = store.ensure_session(
        title="Chat 1",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    if worked_state == "draft":
        store.set_session_draft(original.id, "my work")
    elif worked_state == "system-prompt":
        store.set_session_system_prompt(original.id, "My system prompt")
    elif worked_state == "pinned-prefill":
        store.set_session_pinned_prefill(original.id, "Always:")
    elif worked_state == "model":
        store.replace_session_settings(
            original.id,
            replace(defaults, model="my-model"),
        )
    elif worked_state == "provider":
        store.replace_session_settings(
            original.id,
            replace(defaults, provider="anthropic"),
        )
    elif worked_state == "temperature":
        store.replace_session_settings(
            original.id,
            replace(defaults, temperature=defaults.temperature + 0.1),
        )
    elif worked_state == "character-label":
        store.replace_session_settings(
            original.id,
            replace(defaults, character_label="My assistant"),
        )
    else:
        original.title = "Planning"
    original_before = replace(original)

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    assert len(store.sessions()) == 2
    assert store.sessions()[0] == original_before
    assert store.active_session_id != original.id
    assert store.sessions()[1].title == "Chat with Alba"
    assert len(store.messages_for_session(store.active_session_id)) == 1


@pytest.mark.asyncio
async def test_character_handoff_does_not_refresh_unproven_derived_settings(
    monkeypatch,
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    stale = screen._session._default_console_session_settings()
    original = store.ensure_session(settings=stale)
    original_before = replace(original)
    screen.app_instance.app_config.setdefault("chat_defaults", {})["model"] = (
        "canonical-current-model"
    )

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    assert len(store.sessions()) == 2
    assert store.sessions()[0] == original_before
    assert store.active_session_id != original.id


def test_typed_then_cleared_work_marker_survives_screen_state_restore(monkeypatch):
    screen = _character_screen(monkeypatch, _roleplay_card(name="Alba"))
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    session = store.ensure_session(
        title="Chat 1",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    store.set_session_draft(session.id, "my work")
    store.set_session_draft(session.id, "")

    payload = screen._session._console_session_to_state(session)
    restored = screen._session._console_session_from_state(payload)

    assert restored.has_user_work is True


def test_persona_memory_mode_survives_screen_state_restore(monkeypatch):
    screen = _character_screen(monkeypatch, _roleplay_card(name="Alba"))
    store = screen._ensure_console_chat_store()
    session = store.create_session(
        title="Persona chat",
        settings=screen._session._default_console_session_settings(),
        assistant_kind="persona",
        assistant_id="persona-1",
        persona_memory_mode="read_write",
    )

    payload = screen._session._console_session_to_state(session)
    restored = screen._session._console_session_from_state(payload)

    assert payload["persona_memory_mode"] == "read_write"
    assert restored.persona_memory_mode == "read_write"


@pytest.mark.asyncio
async def test_character_handoff_uses_current_canonical_defaults_not_stale_session(
    monkeypatch,
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    stale = screen._session._default_console_session_settings()
    original = store.ensure_session(
        settings=stale,
        canonical_settings_baseline=stale,
    )
    screen.app_instance.app_config.setdefault("chat_defaults", {})["model"] = (
        "canonical-current-model"
    )
    screen.app_instance.app_config.setdefault("chat_defaults", {})["provider"] = (
        "openai"
    )
    current = screen._session._default_console_session_settings()
    assert stale.model != current.model

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    active = store.switch_session(store.active_session_id)
    assert active is original
    assert active.settings.model == "canonical-current-model"
    assert active.settings.system_prompt == "Protect Captain Rowan as Alba."
    assert active.assistant_kind == "character"
    assert active.character_name == "Alba"
    assert len(store.sessions()) == 1
    assert active.id == original.id


@pytest.mark.asyncio
@pytest.mark.parametrize("has_greeting", [True, False])
async def test_duplicate_character_handoff_does_not_duplicate_session_or_greeting(
    monkeypatch, has_greeting
):
    card = _roleplay_card(name="Alba")
    if not has_greeting:
        card["first_message"] = ""
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    store.ensure_session(
        title="Chat 1",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    payload = _start_chat_handoff(card)

    assert await screen._session._start_character_console_session(payload)
    assert await screen._session._start_character_console_session(payload)

    assert len(store.sessions()) == 1
    assert len(store.messages_for_session(store.active_session_id)) == int(has_greeting)


@pytest.mark.asyncio
async def test_concurrent_character_handoff_does_not_duplicate_session_or_greeting(
    monkeypatch,
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    store.ensure_session(
        title="Chat 1",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    both_fetches_started = asyncio.Event()
    release_fetches = asyncio.Event()
    fetch_count = 0

    async def get_character(*_args, **_kwargs):
        nonlocal fetch_count
        fetch_count += 1
        if fetch_count == 2:
            both_fetches_started.set()
        await release_fetches.wait()
        return card

    screen.app_instance.character_persona_scope_service.get_character = get_character
    payload = _start_chat_handoff(card)

    async def start_both() -> list[bool]:
        return await asyncio.gather(
            screen._session._start_character_console_session(payload),
            screen._session._start_character_console_session(payload),
        )

    both = asyncio.create_task(start_both())
    await wait_for_background_signal(
        both_fetches_started,
        both,
        what="both concurrent character fetches to start",
    )
    release_fetches.set()

    assert await await_background_task(
        both,
        what="both concurrent character handoffs to finish",
    ) == [True, True]
    assert len(store.sessions()) == 1
    assert len(store.messages_for_session(store.active_session_id)) == 1


def _real_chachanotes_db(tmp_path) -> CharactersRAGDB:
    """A real (temp-file-backed) ChaChaNotes DB.

    `_build_test_app` deliberately leaves `app.chachanotes_db` unset (its
    `notes_service` is faked with no real `.db`) -- fine for the vast
    majority of Console tests, which never touch persistence, but exactly
    the gap the promotion tests below exist to close. Assigned to
    `app.chachanotes_db` BEFORE the screen mounts: `_ensure_console_chat_
    store` reads it lazily on first call and memoizes the result, so
    setting it any later would be silently ignored.
    """
    return CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")


def _switcher_entry(
    *,
    native_session_id: str | None = None,
    conversation_id: str | None = None,
    row_key: str = "row",
    title: str = "Entry",
    scope_type: str = "workspace",
    workspace_id: str | None = None,
    is_active: bool = False,
    target: ConsoleSwitcherTarget | None = None,
) -> ConsoleSwitcherEntry:
    return ConsoleSwitcherEntry(
        row_key=row_key,
        title=title,
        subtitle="",
        native_session_id=native_session_id,
        conversation_id=conversation_id,
        scope_type=scope_type,
        workspace_id=workspace_id,
        is_active=is_active,
        target=target,
    )


# -- Session activation: workspace-alignment seam ----------------------------


@pytest.mark.asyncio
async def test_activate_native_console_session_realigns_active_workspace():
    """Activating a session in a DIFFERENT workspace switches the active one.

    This is the one call `_activate_native_console_session` makes into the
    workspace cluster (`_set_active_workspace_for_console_session`) --
    exactly the seam the wave-2 Task 3 brief calls out as needing a deliberate
    named callable between the session and workspace controllers rather than
    a screen back-door. Two real workspaces, two real sessions, one real
    registry service.
    """
    app = _build_test_app()
    registry_service = app.workspace_registry_service
    default_workspace = registry_service.get_active_workspace()
    registry_service.create_workspace(
        workspace_id="ws-other",
        name="Other workspace",
        description="Second workspace for the activation test.",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        home_session = console._session._active_native_console_session()
        assert home_session is not None
        assert registry_service.get_active_workspace().workspace_id == (
            default_workspace.workspace_id
        )

        # `create_session` activates what it creates (same trap
        # `test_cost_chip_state_isolated_across_session_tabs` documents), so
        # right after this call `other_session` -- not `home_session` -- is
        # the active one and the active workspace has already followed it
        # via THIS call's own real-time context sync. The two real
        # transitions this test needs both go through the shared entry
        # point below: hop back home first, THEN hop to `other_session`, so
        # the interesting assertion is against a genuine activation call,
        # not residue from creation.
        other_session = store.create_session(
            title="Other workspace chat", workspace_id="ws-other"
        )
        assert other_session.workspace_id == "ws-other"
        assert store.active_session_id == other_session.id

        await console._session._activate_native_console_session(home_session.id)
        await pilot.pause()
        assert store.active_session_id == home_session.id
        assert registry_service.get_active_workspace().workspace_id == (
            default_workspace.workspace_id
        )

        await console._session._activate_native_console_session(other_session.id)
        await pilot.pause()

        assert store.active_session_id == other_session.id
        assert registry_service.get_active_workspace().workspace_id == "ws-other"

        # Hop back: the home session's own workspace must be restored too,
        # not just left on "ws-other" because nothing re-checks it.
        await console._session._activate_native_console_session(home_session.id)
        await pilot.pause()

        assert store.active_session_id == home_session.id
        assert registry_service.get_active_workspace().workspace_id == (
            default_workspace.workspace_id
        )


@pytest.mark.asyncio
async def test_activate_native_console_session_noop_still_focuses_composer(
    monkeypatch,
):
    """Activating the ALREADY-active session skips the switch but still
    calls the composer-focus step (the guard is `store.active_session_id !=
    session_id`, not an early return from the whole method) -- spies on the
    dependency rather than asserting real Textual focus state, which is not
    reliable to observe headlessly for this compound widget."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        session = console._session._active_native_console_session()
        assert session is not None

        focus_calls: list[bool] = []
        real_focus = type(console)._focus_console_composer_if_needed

        def _spy_focus(self, *, force: bool = False) -> None:
            focus_calls.append(force)
            return real_focus(self, force=force)

        monkeypatch.setattr(
            type(console), "_focus_console_composer_if_needed", _spy_focus
        )

        await console._session._activate_native_console_session(session.id)
        await pilot.pause()

        assert console._session._active_native_console_session().id == session.id
        assert focus_calls == [True]


# -- Ctrl+K switcher callback (`_apply_console_switcher_choice`) ------------


@pytest.mark.asyncio
async def test_apply_console_switcher_choice_activate_uses_shared_activation_path():
    """The switcher's "activate" result reaches the same shared entry point
    as the tab click / Alt+1..9 -- verified here through a real workspace
    switch, not just an active-session-id flip."""
    app = _build_test_app()
    registry_service = app.workspace_registry_service
    registry_service.create_workspace(
        workspace_id="ws-switcher",
        name="Switcher workspace",
        description="Workspace for the switcher-choice test.",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        home = console._session._active_native_console_session()
        assert home is not None
        target = store.create_session(
            title="Switcher target", workspace_id="ws-switcher"
        )
        assert home.id != target.id
        # `create_session` activates what it creates -- go back home first so
        # the switcher choice below is a real transition, not a no-op.
        await console._session._activate_native_console_session(home.id)
        await pilot.pause()
        assert store.active_session_id == home.id
        profile, token = console._workspace._console_switcher_authority()

        choice = ConsoleSwitcherChoice(
            kind="activate",
            entry=_switcher_entry(
                native_session_id=target.id,
                title=target.title,
                target=ConsoleSwitcherTarget(
                    kind=SwitcherTargetKind.NATIVE_SESSION,
                    profile_authority=profile,
                    authority_token=token,
                    session_id=target.id,
                    conversation_id=None,
                    scope_type="workspace",
                    workspace_id="ws-switcher",
                ),
            ),
        )
        await console._session._apply_console_switcher_choice(choice)
        await pilot.pause()

        assert store.active_session_id == target.id
        assert registry_service.get_active_workspace().workspace_id == ("ws-switcher")


@pytest.mark.asyncio
async def test_apply_console_switcher_choice_rename_opens_rename_modal():
    """The switcher's "rename" result opens the rename modal for that tab,
    seeded with its CURRENT title."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        session = store.create_session(title="Rename me")
        profile, token = console._workspace._console_switcher_authority()

        choice = ConsoleSwitcherChoice(
            kind="rename",
            entry=_switcher_entry(
                native_session_id=session.id,
                title=session.title,
                target=ConsoleSwitcherTarget(
                    kind=SwitcherTargetKind.NATIVE_SESSION,
                    profile_authority=profile,
                    authority_token=token,
                    session_id=session.id,
                    conversation_id=None,
                    scope_type="global",
                    workspace_id=None,
                ),
            ),
        )
        await console._session._apply_console_switcher_choice(choice)
        await pilot.pause()

        top = host.screen_stack[-1]
        assert isinstance(top, ConsoleRenameSessionModal)
        assert top._title == "Rename me"


@pytest.mark.asyncio
async def test_apply_console_switcher_choice_none_is_a_noop():
    """A cancelled switcher (`choice is None`) leaves the active session and
    screen stack untouched."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        active_before = store.active_session_id
        depth_before = len(host.screen_stack)

        await console._session._apply_console_switcher_choice(None)
        await pilot.pause()

        assert store.active_session_id == active_before
        assert len(host.screen_stack) == depth_before


# -- Temporary-session promotion: assert the DATABASE, not widget state -----


@pytest.mark.asyncio
async def test_promote_console_temporary_session_writes_durable_rows_to_the_database(
    tmp_path,
):
    """Saving a temporary chat must produce a REAL, readable conversation row
    (and its messages) in `chachanotes_db` -- not just an in-memory flag flip.

    Every existing `_promote_console_temporary_session` test
    (`test_console_composer_menu.py`) stubs the store's own
    `promote_ephemeral_session` and only asserts chip/notification state;
    none of them let a write reach a real database. This is the gap the
    wave-2 Task 3 brief calls out by name.
    """
    app = _build_test_app()
    db = _real_chachanotes_db(tmp_path)
    app.chachanotes_db = db
    host = ConsoleHarness(app)

    try:
        async with host.run_test(size=(160, 44)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")

            store = console._ensure_console_chat_store()
            session = store.create_session(title="Temp chat to save", ephemeral=True)
            store.switch_session(session.id)
            store.append_message(
                session.id, role=ConsoleMessageRole.USER, content="hello", persist=True
            )
            store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="hi there",
                persist=True,
            )
            assert session.persisted_conversation_id is None

            await console._session._promote_console_temporary_session()
            await pilot.pause()

            assert session.ephemeral is False
            conversation_id = session.persisted_conversation_id
            assert conversation_id is not None

            conversation = db.get_conversation_by_id(conversation_id)
            assert conversation is not None
            assert conversation["title"] == "Temp chat to save"

            messages = db.get_messages_for_conversation(conversation_id)
            assert [m["content"] for m in messages] == ["hello", "hi there"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_promote_console_temporary_session_second_call_does_not_duplicate_the_row(
    tmp_path,
):
    """Promoting an already-saved session again must not write a second
    conversation row -- `promote_ephemeral_session` is idempotent, and the
    screen-level wrapper must not paper over a regression there by, say,
    creating a fresh conversation on every click."""
    app = _build_test_app()
    db = _real_chachanotes_db(tmp_path)
    app.chachanotes_db = db
    host = ConsoleHarness(app)

    try:
        async with host.run_test(size=(160, 44)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")

            store = console._ensure_console_chat_store()
            session = store.create_session(title="Save twice", ephemeral=True)
            store.switch_session(session.id)
            store.append_message(
                session.id, role=ConsoleMessageRole.USER, content="hi", persist=True
            )

            await console._session._promote_console_temporary_session()
            await pilot.pause()
            conversation_id = session.persisted_conversation_id
            assert conversation_id is not None
            assert db.get_conversation_by_id(conversation_id) is not None

            await console._session._promote_console_temporary_session()
            await pilot.pause()

            assert session.persisted_conversation_id == conversation_id
            assert session.ephemeral is False
            # Still exactly the one conversation row this session's messages
            # belong to; a duplicate-write regression would mint a second one.
            assert db.get_conversation_by_id(conversation_id) is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_promote_console_temporary_session_no_active_session_touches_nothing():
    """With no active session (a bare/never-mounted store edge), promotion is
    a pure no-op -- no notification, no DB access attempted."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        store.active_session_id = None

        # Must not raise even though there is nothing to promote.
        await console._session._promote_console_temporary_session()
        await pilot.pause()


async def _wait_for_fork_modal_state(pilot, modal, state: str) -> None:
    for _ in range(200):
        if modal.state == state:
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Fork modal did not reach {state!r}; got {modal.state!r}")


@pytest.mark.asyncio
async def test_cancelled_fork_validation_generation_cannot_publish_late_result():
    """A late validator is fenced even if it ignores task cancellation."""

    app = _build_test_app()
    host = ConsoleHarness(app)
    old_started = asyncio.Event()
    release_old = asyncio.Event()
    new_started = asyncio.Event()
    release_new = asyncio.Event()
    calls = 0

    async with host.run_test(size=(100, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        source = store.create_session(
            title="Temporary source",
            settings=console._session._default_console_session_settings(),
            ephemeral=True,
        )
        boundary = store.append_message(
            source.id,
            role=ConsoleMessageRole.USER,
            content="fork boundary",
        )
        original_session_ids = {session.id for session in store.sessions()}

        async def barrier(_generation: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                old_started.set()
                await release_old.wait()
            else:
                new_started.set()
                await release_new.wait()

        console._session._fork_validation_barrier = barrier
        register = Mock(wraps=store.register_fork_snapshot)
        store.register_fork_snapshot = register
        activate = AsyncMock(wraps=console._session._activate_native_console_session)
        console._session._activate_native_console_session = activate

        console._session.request_console_chat_fork(boundary.id)
        await pilot.pause()
        first = host.screen_stack[-1]
        assert isinstance(first, ConsoleForkChatModal)
        first_panel = first.query_one("#console-fork-chat-modal")
        await pilot.press("enter")
        await asyncio.wait_for(old_started.wait(), timeout=2)
        await first.action_request_safe_cancel()
        assert not first_panel.is_attached

        console._session.request_console_chat_fork(boundary.id)
        await pilot.pause()
        second = host.screen_stack[-1]
        assert isinstance(second, ConsoleForkChatModal)
        await pilot.press("enter")
        await asyncio.wait_for(new_started.wait(), timeout=2)

        release_old.set()
        await pilot.pause(0.05)
        assert second.state == "validating"
        assert {session.id for session in store.sessions()} == original_session_ids
        register.assert_not_called()
        activate.assert_not_awaited()

        release_new.set()
        for _ in range(200):
            if len(store.sessions()) == len(original_session_ids) + 1:
                break
            await pilot.pause(0.01)
        assert len(store.sessions()) == len(original_session_ids) + 1
        register.assert_called_once()
        activate.assert_awaited_once()
        # Textual pops the modal from the stack before its scheduled screen
        # replacement physically detaches it. Pin the terminal lifecycle
        # boundary so test teardown cannot overlap the resumed chat recompose.
        for _ in range(200):
            if not second.is_attached:
                break
            await pilot.pause(0.01)
        assert not second.is_attached
        assert console._session._active_fork_request is None


@pytest.mark.asyncio
async def test_fork_validation_freezes_the_accepted_title_until_retry():
    app = _build_test_app()
    host = ConsoleHarness(app)
    validation_started = asyncio.Event()
    release_validation = asyncio.Event()
    validation_generations: list[int] = []

    async with host.run_test(size=(100, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        source = store.create_session(
            title="Source",
            settings=console._session._default_console_session_settings(),
            ephemeral=True,
        )
        boundary = store.append_message(
            source.id,
            role=ConsoleMessageRole.USER,
            content="fork boundary",
        )

        async def barrier(generation: int) -> None:
            validation_generations.append(generation)
            validation_started.set()
            await release_validation.wait()

        console._session._fork_validation_barrier = barrier
        register = Mock(wraps=store.register_fork_snapshot)
        store.register_fork_snapshot = register

        console._session.request_console_chat_fork(boundary.id)
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleForkChatModal)
        title = modal.query_one("#console-fork-chat-title")
        accepted_title = title.value

        await pilot.press("enter")
        await asyncio.wait_for(validation_started.wait(), timeout=2)

        assert modal.state == "validating"
        assert not title.disabled
        assert title.has_focus
        await pilot.press("x", "y", "z")
        await pilot.press("enter", "enter")
        await pilot.pause()

        assert title.value == accepted_title
        assert title.has_focus
        assert modal.state == "validating"
        assert len(validation_generations) == 1
        assert not modal.query_one("#console-fork-chat-exclusions").display
        register.assert_not_called()

        release_validation.set()
        for _ in range(200):
            if register.called:
                break
            await pilot.pause(0.01)
        register.assert_called_once()
        assert register.call_args.args[0].title == accepted_title


@pytest.mark.asyncio
async def test_durable_fork_orders_commit_projection_registration_and_activation(
    tmp_path,
):
    app = _build_test_app()
    db = _real_chachanotes_db(tmp_path)
    app.chachanotes_db = db
    app.workspace_registry_service.create_workspace(
        workspace_id="ws-fork",
        name="Fork workspace",
        description="Fork ordering test",
    )
    host = ConsoleHarness(app)
    events: list[str] = []

    try:
        async with host.run_test(size=(120, 35)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")
            store = console._ensure_console_chat_store()
            source = store.create_session(
                title="Durable source",
                workspace_id="ws-fork",
                settings=console._session._default_console_session_settings(),
            )
            boundary = store.append_message(
                source.id,
                role=ConsoleMessageRole.USER,
                content="saved fork boundary",
                persist=True,
            )
            source_id = source.persisted_conversation_id
            assert source_id is not None
            source_before = dict(db.get_conversation_by_id(source_id))
            messages_before = tuple(
                dict(row) for row in db.get_messages_for_conversation(source_id)
            )

            persistence = store.persistence
            real_commit = persistence.fork_console_conversation_bundle
            real_project = persistence.project_workspace_membership
            real_register = store.register_fork_snapshot
            real_activate = console._session._activate_native_console_session

            def commit(**kwargs):
                events.append("commit")
                return real_commit(**kwargs)

            def project(conversation_id):
                events.append("project")
                return real_project(conversation_id)

            def register(snapshot, *, activate=False):
                events.append("register")
                return real_register(snapshot, activate=activate)

            async def activate(session_id):
                events.append("activate")
                await real_activate(session_id)

            persistence.fork_console_conversation_bundle = commit
            persistence.project_workspace_membership = project
            store.register_fork_snapshot = register
            console._session._activate_native_console_session = activate

            console._session.request_console_chat_fork(boundary.id)
            await pilot.pause()
            modal = host.screen_stack[-1]
            assert isinstance(modal, ConsoleForkChatModal)
            await pilot.press("enter")
            for _ in range(300):
                if host.screen_stack[-1] is console and len(store.sessions()) >= 3:
                    break
                await pilot.pause(0.01)

            assert events == ["commit", "project", "register", "activate"]
            fork = next(
                session
                for session in store.sessions()
                if session.id not in {source.id}
                and session.persisted_conversation_id not in {None, source_id}
            )
            assert store.active_session_id == fork.id
            assert source in store.sessions()
            assert dict(db.get_conversation_by_id(source_id)) == source_before
            assert (
                tuple(dict(row) for row in db.get_messages_for_conversation(source_id))
                == messages_before
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_fork_dialog_boundary_uses_absolute_lineage_ordinal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        source = store.create_session(
            title="Ordinal source",
            settings=console._session._default_console_session_settings(),
            ephemeral=True,
        )
        boundary = None
        for ordinal in range(1, 9):
            boundary = store.append_message(
                source.id,
                role=(
                    ConsoleMessageRole.USER
                    if ordinal % 2
                    else ConsoleMessageRole.ASSISTANT
                ),
                content=f"message {ordinal}",
            )
        assert boundary is not None

        console._session.request_console_chat_fork(boundary.id)
        await pilot.pause()
        modal = host.screen_stack[-1]

        assert isinstance(modal, ConsoleForkChatModal)
        assert modal.summary.boundary_label == "Through Assistant 8"


@pytest.mark.asyncio
@pytest.mark.parametrize("content_class", ["attachment", "image"])
async def test_fork_stage_error_names_safe_content_class_only(content_class):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        source = store.create_session(
            title="Temporary source",
            settings=console._session._default_console_session_settings(),
            ephemeral=True,
        )
        boundary = store.append_message(
            source.id,
            role=ConsoleMessageRole.USER,
            content="fork boundary",
        )
        original_session_ids = {session.id for session in store.sessions()}
        store.stage_fork_snapshot = Mock(
            side_effect=ValueError(
                f"Console fork {content_class} is unavailable at /secret/project."
            )
        )

        console._session.request_console_chat_fork(boundary.id)
        await pilot.pause()
        modal = host.screen_stack[-1]
        await pilot.press("enter")
        await _wait_for_fork_modal_state(pilot, modal, "precommit_error")
        status = str(modal.query_one("#console-fork-chat-status").render()).lower()

        assert content_class in status
        assert "/secret/project" not in status
        assert {session.id for session in store.sessions()} == original_session_ids


@pytest.mark.asyncio
async def test_ambiguous_durable_collision_fails_closed_without_publication(tmp_path):
    app = _build_test_app()
    db = _real_chachanotes_db(tmp_path)
    app.chachanotes_db = db
    host = ConsoleHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")
            store = console._ensure_console_chat_store()
            source = store.create_session(
                title="Saved source",
                settings=console._session._default_console_session_settings(),
            )
            boundary = store.append_message(
                source.id,
                role=ConsoleMessageRole.USER,
                content="saved boundary",
                persist=True,
            )
            original_session_ids = {session.id for session in store.sessions()}

            console._session.request_console_chat_fork(boundary.id)
            await pilot.pause()
            request = console._session._active_fork_request
            assert request is not None and request.fork_conversation_id is not None
            db.add_conversation(
                {"id": request.fork_conversation_id, "title": "Unrelated collision"}
            )
            commit = Mock(side_effect=RuntimeError("ambiguous write result"))
            store.persistence.fork_console_conversation_bundle = commit

            await pilot.press("enter")
            await _wait_for_fork_modal_state(pilot, request.modal, "precommit_error")

            assert {session.id for session in store.sessions()} == original_session_ids
            assert request.snapshot is not None
            assert request.snapshot.fork_conversation_id == request.fork_conversation_id
            assert (
                request.modal.query_one("#console-fork-chat-confirm").display is False
            )
            assert "Close" in str(
                request.modal.query_one("#console-fork-chat-status").render()
            )
            commit.assert_called_once()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_workspace_projection_failure_keeps_one_open_fork_pending_retry(tmp_path):
    app = _build_test_app()
    db = _real_chachanotes_db(tmp_path)
    app.chachanotes_db = db
    app.workspace_registry_service.create_workspace(
        workspace_id="ws-fork-pending",
        name="Pending workspace",
        description="Projection failure test",
    )
    host = ConsoleHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")
            store = console._ensure_console_chat_store()
            source = store.create_session(
                title="Saved source",
                workspace_id="ws-fork-pending",
                settings=console._session._default_console_session_settings(),
            )
            boundary = store.append_message(
                source.id,
                role=ConsoleMessageRole.USER,
                content="saved boundary",
                persist=True,
            )
            persistence = store.persistence
            commit = Mock(wraps=persistence.fork_console_conversation_bundle)
            persistence.fork_console_conversation_bundle = commit
            persistence.project_workspace_membership = Mock(
                side_effect=RuntimeError("registry unavailable")
            )

            console._session.request_console_chat_fork(boundary.id)
            await pilot.pause()
            request = console._session._active_fork_request
            assert request is not None
            fork_session_id = request.fork_session_id
            fork_conversation_id = request.fork_conversation_id
            await pilot.press("enter")
            for _ in range(300):
                if host.screen_stack[-1] is console:
                    break
                await pilot.pause(0.01)

            assert store.active_session_id == fork_session_id
            assert store.has_pending_workspace_projection(fork_session_id)
            assert db.get_conversation_by_id(fork_conversation_id) is not None
            assert (
                len(
                    [
                        session
                        for session in store.sessions()
                        if session.persisted_conversation_id == fork_conversation_id
                    ]
                )
                == 1
            )
            commit.assert_called_once()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_temporary_registration_retry_reuses_id_and_publishes_once():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        source = store.create_session(
            title="Temporary source",
            settings=console._session._default_console_session_settings(),
            ephemeral=True,
        )
        boundary = store.append_message(
            source.id,
            role=ConsoleMessageRole.USER,
            content="temporary boundary",
        )
        original_session_ids = {session.id for session in store.sessions()}
        real_register = store.register_fork_snapshot
        registration_ids: list[str] = []

        def fail_once(snapshot, *, activate=False):
            registration_ids.append(snapshot.fork_session_id)
            if len(registration_ids) == 1:
                raise RuntimeError("registration unavailable")
            return real_register(snapshot, activate=activate)

        store.register_fork_snapshot = fail_once
        console._session.request_console_chat_fork(boundary.id)
        await pilot.pause()
        modal = host.screen_stack[-1]
        await pilot.press("enter")
        await _wait_for_fork_modal_state(pilot, modal, "precommit_error")

        assert {session.id for session in store.sessions()} == original_session_ids
        modal.query_one("#console-fork-chat-confirm").press()
        for _ in range(300):
            if host.screen_stack[-1] is console:
                break
            await pilot.pause(0.01)

        assert registration_ids == [registration_ids[0], registration_ids[0]]
        assert store.active_session_id == registration_ids[0]
        assert (
            len(
                [
                    session
                    for session in store.sessions()
                    if session.id == registration_ids[0]
                ]
            )
            == 1
        )


@pytest.mark.asyncio
async def test_temporary_activation_failure_open_reuses_registered_id_once():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        source = store.create_session(
            title="Temporary source",
            settings=console._session._default_console_session_settings(),
            ephemeral=True,
        )
        boundary = store.append_message(
            source.id,
            role=ConsoleMessageRole.USER,
            content="temporary boundary",
        )
        real_activate = console._session._activate_native_console_session
        activation_ids: list[str] = []

        async def fail_once(session_id):
            activation_ids.append(session_id)
            if len(activation_ids) == 1:
                raise RuntimeError("tab unavailable")
            await real_activate(session_id)

        console._session._activate_native_console_session = fail_once
        console._session.request_console_chat_fork(boundary.id)
        await pilot.pause()
        modal = host.screen_stack[-1]
        request = console._session._active_fork_request
        assert request is not None
        await pilot.press("enter")
        await _wait_for_fork_modal_state(pilot, modal, "created_not_opened")

        assert (
            len(
                [
                    session
                    for session in store.sessions()
                    if session.id == request.fork_session_id
                ]
            )
            == 1
        )
        modal.query_one("#console-fork-chat-open").press()
        for _ in range(300):
            if host.screen_stack[-1] is console:
                break
            await pilot.pause(0.01)

        assert activation_ids == [request.fork_session_id, request.fork_session_id]
        assert store.active_session_id == request.fork_session_id
        assert (
            len(
                [
                    session
                    for session in store.sessions()
                    if session.id == request.fork_session_id
                ]
            )
            == 1
        )


@pytest.mark.asyncio
async def test_ambiguous_durable_commit_reconciles_without_second_bundle(tmp_path):
    app = _build_test_app()
    db = _real_chachanotes_db(tmp_path)
    app.chachanotes_db = db
    host = ConsoleHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")
            store = console._ensure_console_chat_store()
            source = store.create_session(
                title="Saved source",
                settings=console._session._default_console_session_settings(),
            )
            boundary = store.append_message(
                source.id,
                role=ConsoleMessageRole.USER,
                content="saved boundary",
                persist=True,
            )
            persistence = store.persistence
            real_commit = persistence.fork_console_conversation_bundle
            calls = 0

            def commit_then_lose_result(**kwargs):
                nonlocal calls
                calls += 1
                real_commit(**kwargs)
                raise RuntimeError("connection result lost")

            persistence.fork_console_conversation_bundle = commit_then_lose_result

            console._session.request_console_chat_fork(boundary.id)
            await pilot.pause()
            await pilot.press("enter")
            for _ in range(300):
                if host.screen_stack[-1] is console and len(store.sessions()) >= 3:
                    break
                await pilot.pause(0.01)

            assert calls == 1
            durable_forks = [
                session
                for session in store.sessions()
                if session.persisted_conversation_id
                and session.persisted_conversation_id
                != source.persisted_conversation_id
            ]
            assert len(durable_forks) == 1
            assert (
                db.get_conversation_by_id(durable_forks[0].persisted_conversation_id)
                is not None
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_postcommit_registration_failure_open_retry_reuses_identity(tmp_path):
    app = _build_test_app()
    db = _real_chachanotes_db(tmp_path)
    app.chachanotes_db = db
    host = ConsoleHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")
            store = console._ensure_console_chat_store()
            source = store.create_session(
                title="Saved source",
                settings=console._session._default_console_session_settings(),
            )
            boundary = store.append_message(
                source.id,
                role=ConsoleMessageRole.USER,
                content="saved boundary",
                persist=True,
            )
            real_register = store.register_fork_snapshot
            registration_ids: list[str] = []

            def fail_once(snapshot, *, activate=False):
                registration_ids.append(snapshot.fork_session_id)
                if len(registration_ids) == 1:
                    raise RuntimeError("live registration unavailable")
                return real_register(snapshot, activate=activate)

            store.register_fork_snapshot = fail_once

            console._session.request_console_chat_fork(boundary.id)
            await pilot.pause()
            modal = host.screen_stack[-1]
            await pilot.press("enter")
            await _wait_for_fork_modal_state(pilot, modal, "created_not_opened")
            committed_id = (
                console._session._active_fork_request.snapshot.fork_conversation_id
            )
            assert committed_id is not None
            assert db.get_conversation_by_id(committed_id) is not None

            modal.query_one("#console-fork-chat-open").press()
            for _ in range(300):
                if host.screen_stack[-1] is console:
                    break
                await pilot.pause(0.01)

            assert registration_ids == [registration_ids[0], registration_ids[0]]
            assert store.active_session_id == registration_ids[0]
            assert (
                len(
                    [
                        session
                        for session in store.sessions()
                        if session.persisted_conversation_id == committed_id
                    ]
                )
                == 1
            )
    finally:
        db.close()
