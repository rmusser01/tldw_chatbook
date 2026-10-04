"""Settings ▸ Console Behavior ▸ Global fallback defaults: the shared rows (TASK-33007.7).

Console Behavior owns the global fallbacks (ADR-006, ADR-052) and now draws
them with Model defaults' row grammar: core rows first, Sampling in one closed
one-row disclosure, and streaming as an On/Off Select from the same family as
the model default's Inherit/On/Off Select (ADR-095). Mounted with the real
application stylesheet at 211x44.

The last test runs the whole app on a private profile: the real Settings save
writer, Providers & Models' inherited streaming row, and Ctrl+T in Console.
"""

from __future__ import annotations

import pytest
from textual.containers import Horizontal
from textual.widgets import Collapsible, Input, Select, Static
from textual.widgets._collapsible import CollapsibleTitle

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen, _static_text
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.Chat.console_provider_support import (
    MODEL_CONFIG_FIELDS,
    MODEL_FIELD_LABELS,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import (
    MODEL_PROFILE_STREAMING_SELECT_OPTIONS,
)
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    CORE_FIELDS,
    SAMPLING_FIELDS,
)

_SIZE = (211, 44)
CONSOLE_BEHAVIOR = SettingsCategoryId.CONSOLE_BEHAVIOR


def _app(chat_defaults=None):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "openai",
        "model": "gpt-4.1",
        **(chat_defaults or {}),
    }
    return app


async def _open(host, pilot):
    await _settle_settings(pilot)
    await _click_settings_category(pilot, "console-behavior")
    await host.workers.wait_for_complete()
    await pilot.pause()
    return _active_destination_screen(host)


def _cid(name: str) -> str:
    return "settings-console-default-" + name.replace("_", "-")


def _text(screen, selector: str) -> str:
    return _static_text(screen.query_one(selector, Static))


def _row_copy(screen, name: str) -> tuple[str, str]:
    return _text(screen, f"#{_cid(name)}-source"), _text(screen, f"#{_cid(name)}-help")


async def _wait_until(pilot, predicate, what: str, *, attempts: int = 250) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.asyncio
@private_profile_test
async def test_fallbacks_use_model_defaults_rows_core_first_then_closed_sampling(
    request,
):
    """AC#1/AC#2: Temperature, Max tokens, Streaming, then reasoning and
    thinking, each a label, a one-row control, a Source word and a help line;
    the six samplers sit in one closed one-row Sampling disclosure; streaming
    is an On/Off Select of the model default's family."""
    host = _SettingsCssHarness(_app({"top_p": 0.95}), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        fallbacks = screen.query_one("#settings-console-fallbacks")
        rows = [child for child in fallbacks.children if isinstance(child, Horizontal)]
        assert [row.id for row in rows] == [f"{_cid(name)}-row" for name in CORE_FIELDS]
        for name, row in zip(CORE_FIELDS, rows):
            label, control, source, help_line = row.children
            assert _static_text(label) == MODEL_FIELD_LABELS[name]
            assert label.has_class("settings-input-label")
            assert control.id == _cid(name)
            assert source.id == f"{_cid(name)}-source" and _static_text(source)
            assert help_line.id == f"{_cid(name)}-help" and _static_text(help_line)
            assert row.region.height == 1, (name, row.region)
            assert control.region.height == 1, (name, control.region)

        sampling = screen.query_one("#settings-console-sampling", Collapsible)
        assert list(fallbacks.children)[-1] is sampling
        assert sampling.collapsed is True
        assert sampling.region.height == 1
        assert sampling.query_one(CollapsibleTitle).region.height == 1
        assert str(sampling.title) == "Sampling · Top P 0.95"
        inner = [
            child
            for child in sampling.query_one("Contents").children
            if isinstance(child, Horizontal)
        ]
        assert [row.id for row in inner] == [
            f"{_cid(name)}-row" for name in SAMPLING_FIELDS
        ]

        # The model default's Select offers the same two options plus a
        # blank "Inherit"; the global fallback has nothing to inherit from.
        streaming = screen.query_one(f"#{_cid('streaming')}", Select)
        assert tuple(map(tuple, streaming._options)) == (
            MODEL_PROFILE_STREAMING_SELECT_OPTIONS
        )
        assert streaming.value == "true"
        assert streaming.has_class("settings-compact-select")
        assert streaming.parent.has_class("settings-select-row")

        # '/' still reaches a sampler: landing opens the closed disclosure.
        screen._land_search_focus_on_field(_cid("seed"), "Seed")
        await pilot.pause()
        await pilot.pause()
        assert sampling.collapsed is False
        assert screen.app.focused is screen.query_one(f"#{_cid('seed')}", Input)


@pytest.mark.asyncio
@private_profile_test
async def test_each_fallback_says_its_source_and_a_blank_says_what_it_sends(request):
    """AC#1 (spec §6 grammar): a saved fallback reads "Console Behavior" and
    the field's help, an edited one "edited *", and a blank optional one says
    the provider decides; the Sampling title follows the fields."""
    host = _SettingsCssHarness(
        _app({"temperature": 0.4, "top_p": 0.9, "streaming": False}), "settings"
    )

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        assert _row_copy(screen, "temperature") == (
            "Console Behavior",
            MODEL_CONFIG_FIELDS["temperature"].help,
        )
        assert _row_copy(screen, "streaming") == (
            "Console Behavior",
            MODEL_CONFIG_FIELDS["streaming"].help,
        )
        assert screen.query_one(f"#{_cid('streaming')}", Select).value == "false"
        assert _row_copy(screen, "seed") == ("provider", "blank = provider default")
        assert _row_copy(screen, "max_tokens") == (
            "provider",
            "blank = provider default",
        )

        temperature = screen.query_one(f"#{_cid('temperature')}", Input)
        temperature.focus()
        await pilot.press("end", *["backspace"] * 4, *"0.8")
        sampling = screen.query_one("#settings-console-sampling", Collapsible)
        sampling.collapsed = False
        await pilot.pause()
        seed = screen.query_one(f"#{_cid('seed')}", Input)
        seed.focus()
        await pilot.press(*"7")
        await pilot.pause()

        assert _row_copy(screen, "temperature") == (
            "edited *",
            MODEL_CONFIG_FIELDS["temperature"].help,
        )
        assert _row_copy(screen, "seed") == (
            "edited *",
            MODEL_CONFIG_FIELDS["seed"].help,
        )
        assert str(sampling.title) == "Sampling · Top P 0.9 · Seed 7"


@pytest.mark.asyncio
@private_profile_test
async def test_legacy_enable_streaming_is_still_read_and_shown(request):
    """AC#3: with only the legacy enable_streaming key the Select still shows
    its value (chat_defaults.streaming is read first when present)."""
    host = _SettingsCssHarness(_app({"enable_streaming": False}), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        assert screen.query_one(f"#{_cid('streaming')}", Select).value == "false"


@pytest.mark.asyncio
@private_profile_test
async def test_the_streaming_key_fact_lives_in_the_inspector_config_key_disclosure(
    request,
):
    """AC#4: the raw 'chat_defaults.streaming is canonical...' line is gone
    from Settings' own copy; the Inspector's closed "config key" disclosure
    holds it."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        detail = screen.query_one("#settings-detail-pane-body")
        assert not [
            _static_text(widget)
            for widget in detail.query(Static)
            if "enable_streaming" in _static_text(widget)
        ]
        disclosure = screen.query_one(
            "#settings-console-behavior-config-key", Collapsible
        )
        assert disclosure.collapsed is True
        assert str(disclosure.title) == "config key"
        assert disclosure.region.height == 1
        body = screen.query_one("#settings-impact-pane-body")
        holders = [
            widget
            for widget in body.query(Static)
            if "enable_streaming" in _static_text(widget)
        ]
        assert holders
        assert all(disclosure in widget.ancestors for widget in holders)
        fact = " ".join(_static_text(widget) for widget in holders).replace("\n", "")
        assert "chat_defaults.streaming" in fact


@pytest.mark.asyncio
@private_profile_test
async def test_saving_global_streaming_off_reaches_a_new_chat_and_model_defaults(
    request, monkeypatch
):
    """AC#3/AC#6: the whole app on a private profile -- the real Settings save
    writer sets chat_defaults.streaming = false, a model default left at
    Inherit then says "inherits Off · Console Behavior", and Ctrl+T's new chat
    resolves streaming Off."""
    from Tests.UI.test_console_session_settings import (
        _build_live_config_test_app,
        _wait_for_screen,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook import config as config_module
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    adapter = SettingsConfigAdapter()
    for section, values in (
        ("splash_screen", {"enabled": False}),
        ("first_run", {"setup_completed": True}),
        (
            "chat_defaults",
            {"provider": "llama_cpp", "model": "model-a", "streaming": True},
        ),
        ("api_settings.llama_cpp", {"api_url": "http://127.0.0.1:9099"}),
    ):
        assert adapter.save_values(section, values), section
    config_module.load_settings(force_reload=True)
    app = _build_live_config_test_app()

    async with app.run_test(size=_SIZE) as pilot:
        app.providers_models = {"llama_cpp": ["model-a"]}
        app.post_message(NavigateToScreen("settings", {"category": CONSOLE_BEHAVIOR}))
        screen = await _wait_for_screen(app, pilot, "SettingsScreen")
        await _wait_for_selector(screen, pilot, f"#{_cid('streaming')}")
        await pilot.pause()
        streaming = screen.query_one(f"#{_cid('streaming')}", Select)
        assert streaming.value == "true"
        streaming.focus()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("down", "enter")
        await _wait_until(
            pilot,
            lambda: streaming.value == "false",
            "the Select to choose Off",
        )
        assert screen._category_has_unsaved_changes(CONSOLE_BEHAVIOR)
        screen.set_focus(None)
        await pilot.pause()
        await pilot.press("s")
        await _wait_until(
            pilot,
            lambda: not screen._category_has_unsaved_changes(CONSOLE_BEHAVIOR),
            "the save to finish",
        )
        saved = config_module.load_settings(force_reload=True)["chat_defaults"]
        assert saved["streaming"] is False
        assert _row_copy(screen, "streaming")[0] == "Console Behavior"

        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await _wait_for_selector(
            screen, pilot, "#settings-model-profile-streaming-help"
        )
        await pilot.pause()
        await pilot.pause()
        inherit = screen.query_one("#settings-model-profile-streaming", Select)
        assert inherit.value is Select.NULL
        assert _text(screen, "#settings-model-profile-streaming-help") == (
            "inherits Off · Console Behavior"
        )

        app.post_message(NavigateToScreen("chat"))
        console = await _wait_for_screen(app, pilot, "ChatScreen")
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        before = store.active_session_id
        await pilot.press("ctrl+t")
        await _wait_until(
            pilot,
            lambda: store.active_session_id not in (None, before),
            "Ctrl+T to open a new chat",
        )
        assert store.session_settings(store.active_session_id).streaming is False
