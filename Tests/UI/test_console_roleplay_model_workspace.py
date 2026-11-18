"""Roleplay activation must survive returning from Switch model (TASK-37620)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.console_fixture_ownership import (
    owned_console_apps as owned_console_apps,  # noqa: PLC0414
)
from Tests.UI.test_character_session_prompt_seed import (
    _character_screen,
    _roleplay_card,
    _start_chat_handoff,
)
from Tests.UI.test_console_model_apply_chips import _wait_for
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsReadiness
from tldw_chatbook.UI.Console_Modules import model_switcher
from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
    ConsoleCharacterChoice,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusText
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

pytestmark = pytest.mark.bootstrap_profile


def _persona_handoff() -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source="personas",
        item_type="persona-card",
        title="Archivist",
        body="Persona summary",
        runtime_backend="local",
        source_owner="local",
        source_selector_state="local",
        metadata={
            "intent": "start_chat",
            "selected_kind": "persona",
            "selected_record_id": "persona-7",
            "selected_name": "Archivist",
            "selected_target_id": "local:persona:persona-7",
            "backend": "local",
        },
    )


async def _start_roleplay(screen, app, entry, monkeypatch):
    card = _roleplay_card(name="pixel-migu")
    card["first_message"] = ""

    async def get_character(character_id, *, mode):
        assert (character_id, mode) == (7, "local")
        return card

    async def get_persona_profile(persona_id, *, mode):
        assert (persona_id, mode) == ("persona-7", "local")
        return {"id": "persona-7", "name": "Archivist", "system_prompt": "Be precise."}

    monkeypatch.setattr(
        app, "chachanotes_db", SimpleNamespace(get_local_authority_id=lambda: "local")
    )
    monkeypatch.setattr(
        app,
        "character_persona_scope_service",
        SimpleNamespace(
            get_character=get_character, get_persona_profile=get_persona_profile
        ),
    )
    if entry == "picker":
        monkeypatch.setattr(
            screen._character, "_fetch_character_card_for_avatar", lambda _id: card
        )
        await screen._character._apply_console_character_choice_async(
            ConsoleCharacterChoice(7, "pixel-migu", "new")
        )
    elif entry == "character":
        assert await screen._session._start_character_console_session(
            _start_chat_handoff(card)
        )
    else:
        assert await screen._session._start_persona_console_session(_persona_handoff())


@pytest.mark.parametrize("entry", ["character", "persona"])
async def test_pristine_roleplay_handoff_keeps_its_named_workspace(entry, monkeypatch):
    screen = _character_screen(monkeypatch, _roleplay_card(name="pixel-migu"))
    app = screen.app_instance
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="roleplay-project", name="Roleplay project")
    registry.set_active_workspace("roleplay-project")
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    original = store.create_session(
        workspace_id="roleplay-project",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )

    await _start_roleplay(screen, app, entry, monkeypatch)

    assert store.active_session_id == original.id
    assert original.workspace_id == "roleplay-project"
    assert registry.get_active_workspace().workspace_id == "roleplay-project"
    assert len(store.sessions()) == 1
    assert original.assistant_kind == entry


@pytest.mark.parametrize("entry", ["picker", "character", "persona"])
@pytest.mark.parametrize("finish", ["apply", "cancel"])
@private_profile_test
async def test_roleplay_model_picker_preserves_originating_chat(
    entry, finish, monkeypatch, request
):
    """Creating roleplay from a worked named chat must not bounce back on resume."""
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
        "api_settings": {
            "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "model-a"},
            "vllm": {"api_url": "http://127.0.0.1:9098", "model": "model-b"},
        },
        "providers": {"llama_cpp": ["model-a"], "vLLM": ["model-b"]},
    }
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="roleplay-project", name="Roleplay project")
    original_sources = model_switcher._switcher_sources

    async def no_probe(_targets, _settled):
        return None

    async def catalog(provider):
        return {"llama_cpp": ["model-a"], "vllm": ["model-b"]}.get(provider, [])

    def offline_sources(*args, **kwargs):
        sources = original_sources(*args, **kwargs)
        sources.update(
            catalog_loader=catalog,
            connection_prober=no_probe,
            default_readiness_resolver=lambda *_a, **_kw: ConsoleSettingsReadiness(
                "Ready", "Ready.", True
            ),
        )
        return sources

    monkeypatch.setattr(model_switcher, "_switcher_sources", offline_sources)
    host = ConsoleHarness(app)
    async with host.run_test(size=(211, 44)) as pilot:
        screen = host.screen
        await _wait_for(
            pilot,
            lambda: screen._console_attach_reconciled,
            "Console attach",
            timeout=30,
        )
        assert screen._workspace._switch_console_workspace("roleplay-project")
        await pilot.pause()
        store = screen._ensure_console_chat_store()
        previous = store.ensure_session()
        assert previous.workspace_id == registry.get_active_workspace().workspace_id
        store.append_message(
            previous.id,
            role=ConsoleMessageRole.USER,
            content="Previous work",
            persist=False,
        )
        previous_settings = previous.settings
        await _start_roleplay(screen, app, entry, monkeypatch)
        origin = store.ensure_session()
        assert origin.id != previous.id and origin.workspace_id == "global"
        store.append_message(
            origin.id,
            role=ConsoleMessageRole.USER,
            content="Keep this chat",
            persist=False,
        )
        await screen._sync_native_console_chat_ui()
        await pilot.pause()
        session_ids = [session.id for session in store.sessions()]
        messages = [
            (message.id, message.content)
            for message in store.messages_for_session(origin.id)
        ]
        identity = (
            origin.assistant_kind,
            origin.assistant_id,
            origin.settings.system_prompt,
        )

        line = screen.query_one("#console-status-text", ConsoleStatusText)
        segment = "model" if finish == "apply" else "provider"
        start = next(
            start for name, start, _end in line.segment_spans if name == segment
        )
        assert await pilot.click("#console-status-text", offset=(start + 2, 0))
        await _wait_for(
            pilot, lambda: isinstance(host.screen, ConsoleModelPopover), "model picker"
        )
        picker = host.screen
        assert picker._origin.session_id == origin.id
        await _wait_for(pilot, lambda: picker._first_readiness_done, "picker readiness")
        if finish == "apply":
            await pilot.press(*"model-b")
            await pilot.pause()
            row = picker.highlighted_row()
            assert (row.provider, row.model) == ("vllm", "model-b")
        await pilot.press("enter" if finish == "apply" else "escape")
        await _wait_for(
            pilot,
            lambda: host.screen is screen and screen._console_attach_reconciled,
            "Console resume",
        )

        assert store.active_session_id == origin.id
        assert [session.id for session in store.sessions()] == session_ids
        assert [
            (message.id, message.content)
            for message in store.messages_for_session(origin.id)
        ] == messages
        assert (
            origin.assistant_kind,
            origin.assistant_id,
            origin.settings.system_prompt,
        ) == identity
        assert registry.get_active_workspace().workspace_id == DEFAULT_WORKSPACE_ID
        assert previous.settings == previous_settings
        assert (origin.settings.provider, origin.settings.model) == (
            ("vllm", "model-b") if finish == "apply" else ("llama_cpp", "model-a")
        )
