"""Settings ▸ Providers & Models says who a save reaches (TASK-33007.4).

ADR-095 as amended 2026-09-26 (D1): a saved default reaches new chats and an
open chat nobody has used yet; a chat with messages or edits keeps its own
pair and switches in Console with Alt+M. The card's Applies-to row names the
open Console chat and the pair it will use; the Inspector says the same in
blocks (Applies to, Next new chat will use, the focused field, Key), and keeps
config keys in one closed "config key" disclosure.

The last test runs the whole app on a private profile: a live Console store,
the real Settings save writer, and Ctrl+T in Console.
"""

from __future__ import annotations

import pytest
from textual.widgets import Collapsible, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen, _static_text
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness
from tldw_chatbook.Chat.console_session_settings import blank_console_session_settings
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

_SIZE = (211, 44)
_FAKE_KEY = "sk-proj-abcdefghijklmnop1234"
_NEW_CHATS = "new chats (Ctrl+T, temporary, workspace)."
PROVIDERS_MODELS = SettingsCategoryId.PROVIDERS_MODELS


def _app():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.4,
        "max_tokens": 2048,
    }
    app.app_config["api_settings"] = {"openai": {"api_key": _FAKE_KEY}}
    app.providers_models = {"OpenAI": ["gpt-4o", "gpt-4.1"]}
    return app


async def _open_providers(host, pilot):
    await _settle_settings(pilot)
    await _click_settings_category(pilot, "providers-models")
    await host.workers.wait_for_complete()
    await pilot.pause()
    return _active_destination_screen(host)


def _text(screen, selector: str) -> str:
    return _static_text(screen.query_one(selector, Static))


async def _wait_until(pilot, predicate, what: str, *, attempts: int = 250) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def _inspector_sections(screen) -> list[str]:
    body = screen.query_one("#settings-impact-pane-body")
    return [
        _static_text(widget)
        for widget in body.query(".destination-section")
        if widget.display
    ]


@pytest.mark.asyncio
@private_profile_test
async def test_applies_to_row_sits_under_the_default_model_and_says_no_chat_is_open(
    request,
):
    """AC#1: directly under Default model; with no Console chat, it says so."""
    host = _SettingsCssHarness(_app(), "settings")
    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        model_row = screen.query_one("#settings-model-row")
        applies_row = screen.query_one("#settings-model-applies-row")

        assert applies_row.region.y == model_row.region.bottom
        assert applies_row.region.height == 1
        assert _static_text(applies_row.query_one(".settings-input-label")) == (
            "Applies to"
        )
        assert _text(screen, "#settings-model-applies-to") == (
            f"{_NEW_CHATS} No Console chat is open."
        )


@pytest.mark.asyncio
@private_profile_test
async def test_a_long_chat_title_wraps_the_row_rather_than_cut_the_pair(request):
    """AC#1: the pair is the point of the row; a long title gives way to it
    and the row wraps instead of cutting it off."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    app = _app()
    store = ConsoleChatStore()
    own = ConsoleSessionSettings(
        provider="anthropic", model="claude-sonnet-4-5-20250929"
    )
    session = store.create_session(
        title="Refactor the settings screen into region modules",
        settings=own,
        canonical_settings_baseline=own,
    )
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    app.console_runtime.set_chat_store(store)
    host = _SettingsCssHarness(app, "settings")
    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        text = _text(screen, "#settings-model-applies-to")
        assert text == (
            f"{_NEW_CHATS} Open chat “Refactor the settings s…” keeps "
            "Anthropic · claude-sonnet-4-5-20250929."
        )
        row = screen.query_one("#settings-model-applies-row")
        painted = " ".join(line.strip() for line in _region_rows(screen, row))
        assert "claude-sonnet-4-5-20250929." in painted, painted
        assert row.region.height == 2


@pytest.mark.asyncio
@private_profile_test
async def test_inspector_leads_with_reach_then_the_next_new_chat_then_the_field(
    request,
):
    """AC#2, AC#3: Applies to, Next new chat will use, the focused field, Key
    (spec mock (c) order); the next chat reads saved config, and a dirty form
    says its edits apply only after save."""
    app = _app()
    host = _SettingsCssHarness(app, "settings")
    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)

        assert _inspector_sections(screen)[:4] == [
            "Applies to",
            "Next new chat will use",
            "Focused field guide",
            "Key",
        ]
        assert [
            _text(screen, f"#settings-provider-applies-{row}")
            for row in ("new", "unused", "work", "switch")
        ] == [
            "New chats: yes",
            "Unused open chats: follow the saved default",
            "Chats with work: keep their own; switch there with Alt+M",
            "Model defaults: chats that switch to this model pick them up",
        ]

        saved = blank_console_session_settings(app.app_config)
        assert (saved.model, saved.temperature, saved.max_tokens) == (
            "gpt-4o",
            0.4,
            2048,
        )
        assert _text(screen, "#settings-provider-next-chat-pair") == "OpenAI · gpt-4o"
        assert _text(screen, "#settings-provider-next-chat-values") == (
            "T 0.4 · max 2048 · stream On"
        )
        note = screen.query_one("#settings-provider-next-chat-note", Static)
        assert note.display is False

        screen.query_one("#settings-model-value", Input).value = "gpt-4.1"
        await pilot.pause()

        assert screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        assert note.display is True
        assert _static_text(note) == "Unsaved edits apply only after save (s)."
        # The next new chat still gets the saved pair until s is pressed.
        assert _text(screen, "#settings-provider-next-chat-pair") == "OpenAI · gpt-4o"


@pytest.mark.asyncio
@private_profile_test
async def test_focused_field_shows_help_and_range_and_its_key_only_when_opened(
    request,
):
    """AC#4, AC#5: the guide shows help and range; the config key -- and the
    catalog, key-policy, manual-entry, sampling-route and endpoint-key facts
    the card no longer prints -- sit in one closed disclosure, which keeps
    naming the focused field's key when it is opened."""
    host = _SettingsCssHarness(_app(), "settings")
    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        card = screen.query_one("#settings-providers-models-card")
        for gone in (
            "#settings-provider-catalog",
            "#settings-provider-catalog-policy",
            "#settings-provider-manual-entry-policy",
            "#settings-provider-sampling-route",
            "#settings-provider-endpoint-key",
        ):
            assert not card.query(gone), gone

        screen.query_one("#settings-provider-endpoint-value", Input).focus()
        await pilot.pause()

        guide = [
            _text(screen, f"#settings-provider-field-guide-{index}")
            for index in range(4)
        ]
        assert guide[0] == "Focused setting: Endpoint"
        assert guide[1] == "Purpose: Server address the requests go to."
        assert guide[3].startswith("Validation: an http:// or https:// address")
        assert not screen.query("#settings-provider-field-guide-4")
        assert not any(row.startswith("Saved as") for row in guide)

        disclosure = screen.query_one("#settings-provider-config-key", Collapsible)
        assert disclosure.collapsed is True
        assert str(disclosure.title) == "config key"
        assert disclosure.region.height == 1
        inside = [
            "".join(_text(screen, selector).split())
            for selector in (
                "#settings-provider-config-key-saved-as",
                "#settings-provider-endpoint-key",
                "#settings-provider-catalog",
                "#settings-provider-catalog-policy",
                "#settings-provider-manual-entry-policy",
                "#settings-provider-sampling-route",
            )
        ]
        assert inside[0] == "Savedas:api_settings.openai.api_base_url"
        assert inside[1] == "Endpointkey:api_settings.openai.api_base_url"
        assert inside[2].startswith("Providercatalog")
        for selector in (
            "#settings-provider-config-key-saved-as",
            "#settings-provider-catalog",
        ):
            assert disclosure in screen.query_one(selector).ancestors

        screen.query_one("#settings-impact-pane-body").scroll_to_widget(
            disclosure, animate=False
        )
        await pilot.pause()
        await pilot.click("#settings-provider-config-key CollapsibleTitle")
        await _wait_until(pilot, lambda: not disclosure.collapsed, "the disclosure")

        assert pilot.app.focused.parent is disclosure
        assert "".join(
            _text(screen, "#settings-provider-config-key-saved-as").split()
        ) == ("Savedas:api_settings.openai.api_base_url")
        assert _text(screen, "#settings-provider-field-guide-0") == (
            "Focused setting: Endpoint"
        )


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("work", ["untouched", "message"])
async def test_saving_default_b_states_its_reach_and_a_new_chat_takes_it(
    request, monkeypatch, work
):
    """AC#7: the whole app on a private profile -- a live Console store whose
    active chat is on provider A, the real Settings save writer, and Ctrl+T
    in Console. The Applies-to row must say what the Console then does."""
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.provider_readiness import provider_config_key
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
    from Tests.UI.test_console_session_settings import (
        _build_live_config_test_app,
        _wait_for_screen,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    adapter = SettingsConfigAdapter()
    for section, values in (
        ("splash_screen", {"enabled": False}),
        ("first_run", {"setup_completed": True}),
        ("chat_defaults", {"provider": "llama_cpp", "model": "model-a"}),
        ("api_settings.llama_cpp", {"api_url": "http://127.0.0.1:9099"}),
        ("api_settings.openai", {"api_key": _FAKE_KEY}),
    ):
        assert adapter.save_values(section, values), section
    config_module.load_settings(force_reload=True)
    app = _build_live_config_test_app()

    async with app.run_test(size=_SIZE) as pilot:
        app.providers_models = {"OpenAI": ["gpt-4.1"], "llama_cpp": ["model-a"]}
        app.post_message(NavigateToScreen("chat"))
        console = await _wait_for_screen(app, pilot, "ChatScreen")
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        chat_id = store.active_session_id
        chat = console._session._ensure_active_console_session_settings()
        assert (provider_config_key(chat.provider), chat.model) == (
            "llama_cpp",
            "model-a",
        )
        title = next(s.title for s in store.sessions() if s.id == chat_id)
        if work == "message":
            store.append_message(
                chat_id, role=ConsoleMessageRole.USER, content="Keep my settings."
            )

        app.post_message(NavigateToScreen("settings", {"category": PROVIDERS_MODELS}))
        screen = await _wait_for_screen(app, pilot, "SettingsScreen")
        await _wait_for_selector(screen, pilot, "#settings-model-applies-to")
        await pilot.pause()
        keeps = f"{_NEW_CHATS} Open chat “{title}” keeps llama.cpp · model-a."
        if work == "untouched":
            assert _text(screen, "#settings-model-applies-to") == (
                f"{_NEW_CHATS} Open chat “{title}” is unused and will use "
                "llama.cpp · model-a."
            )
        else:
            assert _text(screen, "#settings-model-applies-to") == keeps

        screen.query_one("#settings-provider-value", Select).value = "openai"
        await pilot.pause()
        field = screen.query_one(
            "#settings-model-picker #model-search-picker-input", Input
        )
        field.focus()
        await pilot.pause()
        await pilot.press(*"gpt-4.1", "enter")
        await _wait_until(
            pilot,
            lambda: screen.query_one("#settings-model-value", Input).value == "gpt-4.1",
            "the picker to stage gpt-4.1",
        )
        screen.set_focus(None)
        await pilot.pause()
        await pilot.press("s")
        await _wait_until(
            pilot,
            lambda: not screen._category_has_unsaved_changes(PROVIDERS_MODELS),
            "the save to finish",
        )
        await pilot.pause()
        saved = config_module.load_settings(force_reload=True)["chat_defaults"]
        assert (saved["provider"], saved["model"]) == ("openai", "gpt-4.1")

        if work == "untouched":
            assert _text(screen, "#settings-model-applies-to") == (
                f"{_NEW_CHATS} Open chat “{title}” is unused and will use "
                "OpenAI · gpt-4.1."
            )
        else:
            assert _text(screen, "#settings-model-applies-to") == keeps
        assert _text(screen, "#settings-provider-next-chat-pair") == (
            "OpenAI · gpt-4.1"
        )

        app.post_message(NavigateToScreen("chat"))
        console = await _wait_for_screen(app, pilot, "ChatScreen")
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        assert store.active_session_id == chat_id
        held = console._session._ensure_active_console_session_settings()
        expected_held = (
            ("openai", "gpt-4.1") if work == "untouched" else ("llama_cpp", "model-a")
        )
        assert (provider_config_key(held.provider), held.model) == expected_held

        await pilot.press("ctrl+t")
        await _wait_until(
            pilot,
            lambda: store.active_session_id not in (None, chat_id),
            "Ctrl+T to open a new chat",
        )
        new_chat = store.session_settings(store.active_session_id)
        assert (provider_config_key(new_chat.provider), new_chat.model) == (
            "openai",
            "gpt-4.1",
        )
