"""Context and ambient effects remain readable, durable, and recoverable."""

from pathlib import Path

import pytest
from textual.widgets import Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _category, _painted
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness


@pytest.mark.parametrize("raw", ["inf", "-inf", "nan"])
@private_profile_test
async def test_nonfinite_background_rate_keeps_configuration_loadable(request, raw):
    """A legal TOML nonfinite float must not stop config/Settings startup."""
    import toml

    from tldw_chatbook import config

    path = Path(config.get_cli_config_path())
    values = toml.loads(path.read_text())
    values.setdefault("console", {})["background_effects"] = {
        "enabled": True,
        "effect": "rain",
        "fps": float(raw),
    }
    path.write_text(toml.dumps(values))
    before = path.read_bytes()
    loaded = config.load_settings(force_reload=True)
    assert loaded["console"]["background_effects"]["fps"] == 6
    assert loaded["console"]["background_effects"]["effect"] == "rain"
    assert path.read_bytes() == before


@private_profile_test
async def test_context_percentage_label_is_fully_painted(request):
    app = _build_test_app()
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        await _category(host, pilot, "Console Behavior")
        control = host.screen.query_one(
            "#settings-console-context-target-percent", Input
        )
        control.focus()
        await _settle(host, pilot)
        label = control.parent.query_one(".settings-input-label", Static)
        _assert_painted(host.screen, label)
        assert str(label.renderable) in _painted(host, label)


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.timeout(240)
@private_profile_test
async def test_context_and_effects_keyboard_save_revert_retry(
    request, monkeypatch, theme, size
):
    import tomllib

    from textual.widgets import Button

    from Tests.UI.test_settings_provider_keyboard_journeys import (
        _edit,
        _revert,
        _tab_to,
    )
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_context_policy import merge_context_policy
    from tldw_chatbook.UI.Screens import settings_screen as module

    app = _build_test_app()
    host = _StyledDestinationHarness(app, "settings")
    host.theme = theme
    path = Path(config.get_cli_config_path())
    prefix = "#settings-console-context-"
    effect_prefix = "#settings-console-background-effect-"
    expected = {
        "conversation_budget_mode": "custom",
        "conversation_budget_tokens": 64000,
        "compaction_mode": "automatic",
        "compaction_representation": "hybrid",
        "compaction_trigger_ratio": 0.85,
        "compaction_target_ratio": 0.55,
        "compaction_summary_max_tokens": 1536,
        "compaction_failure_behavior": "omit_older_context",
        "compaction_carry_forward_mode": "memory_with_latest_exchange",
    }
    background = {
        "enabled": True,
        "effect": "matrix",
        "scope": "transcript",
        "intensity": "high",
        "fps": 9,
    }

    async def focus(selector):
        widget = host.screen.query_one(selector)
        widget.focus()
        await _settle(host, pilot)
        _assert_painted(host.screen, widget)
        return widget

    def label_is_painted(control):
        label = control.parent.query_one(".settings-input-label", Static)
        _assert_painted(host.screen, label)
        assert str(label.renderable) in _painted(host, label)

    async def choose(selector, index, expected_value):
        control = await _tab_to(host, pilot, selector)
        label_is_painted(control)
        await pilot.press("enter", "home", *(["down"] * index), "enter")
        await _settle(host, pilot)
        assert control.value == expected_value
        _assert_painted(host.screen, control)

    async def edit(selector, value):
        await _edit(host, pilot, selector, value)
        label_is_painted(host.screen.query_one(selector))

    async def stage():
        await focus(prefix + "budget-mode")
        await choose(prefix + "budget-mode", 1, "custom")
        await edit(prefix + "budget-tokens", "64000")
        await choose(prefix + "compaction-mode", 1, "automatic")
        await choose(prefix + "compaction-representation", 2, "hybrid")
        await edit(prefix + "trigger-percent", "85")
        await edit(prefix + "target-percent", "55")
        await edit(prefix + "summary-max-tokens", "1536")
        await choose(prefix + "failure-behavior", 1, "omit_older_context")
        await choose(prefix + "carry-forward-mode", 1, "memory_with_latest_exchange")
        toggle = await focus(effect_prefix + "enabled")
        assert str(toggle.label) in _painted(host, toggle)
        if not toggle.value:
            await pilot.press("space")
            await _settle(host, pilot)
        await choose(effect_prefix + "type", 3, "matrix")
        # Unsupported scope explains its fallback and returns to Transcript.
        await choose(effect_prefix + "scope", 1, "transcript")
        assert (
            "Workbench scope is not available" in host.screen._console_behavior_result
        )
        await choose(effect_prefix + "intensity", 2, "high")
        await edit(effect_prefix + "fps", "9")

    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, "Console Behavior")
        screen = host.screen
        before = path.read_bytes()
        original = screen.query_one(effect_prefix + "fps", Input).value
        await focus(prefix + "target-percent")
        await edit(prefix + "target-percent", "75")
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert "at least 15 percentage points" in screen._console_behavior_result
        assert path.read_bytes() == before
        assert screen.query_one(prefix + "target-percent", Input).value == "75"
        await stage()
        for invalid in ("0", "13"):
            await edit(effect_prefix + "fps", invalid)
            await pilot.press("escape", "s")
            await _settle(host, pilot)
            assert "between 1 and 12" in screen._console_behavior_result
            assert screen.query_one(effect_prefix + "fps", Input).value == invalid
            assert path.read_bytes() == before
            await focus(effect_prefix + "fps")
        await edit(effect_prefix + "fps", "9")
        await _category(host, pilot, "Overview")
        await _category(host, pilot, "Console Behavior")
        assert screen.query_one(prefix + "budget-tokens", Input).value == "64000"
        assert screen.query_one(effect_prefix + "fps", Input).value == "9"
        await _revert(host, pilot, discard=False)
        assert screen.query_one(effect_prefix + "fps", Input).value == "9"
        await _revert(host, pilot, discard=True)
        assert screen.query_one(effect_prefix + "fps", Input).value == original
        assert not screen._category_has_unsaved_changes(screen.active_category)
        assert path.read_bytes() == before
        await stage()
        writer = config.apply_settings_mutation_to_cli_config

        def refuse(*args, **kwargs):
            return config.ConfigMutationResult(False, False, "before_replace")

        monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", refuse)
        monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", refuse)
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert path.read_bytes() == before
        assert screen._category_has_unsaved_changes(screen.active_category)
        assert "Failed" in screen._console_behavior_result
        assert screen.query_one(prefix + "budget-tokens", Input).value == "64000"
        monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", writer)
        monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", writer)
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        saved = tomllib.loads(path.read_text())["console"]
        assert {key: saved[key] for key in expected} == expected
        assert saved["background_effects"] == background
        assert app.app_config["console"]["background_effects"] == background
        runtime = merge_context_policy(
            global_overrides=ConsoleChatController._global_context_policy_overrides(
                None
            )
        )
        assert runtime.custom_budget_tokens == 64000
        assert runtime.budget_mode.value == "custom"
        assert runtime.compaction_mode.value == "automatic"
        assert runtime.summary_max_tokens == 1536
        assert runtime.failure_behavior.value == "omit_older_context"
        assert runtime.trigger_ratio == 0.85
        assert runtime.target_ratio == 0.55
        assert runtime.compaction_representation.value == "hybrid"
        assert runtime.carry_forward_mode.value == "memory_with_latest_exchange"
        assert not screen._category_has_unsaved_changes(screen.active_category)
        receipt = screen.query_one("#settings-console-behavior-result", Static)
        receipt.scroll_visible(animate=False)
        await _settle(host, pilot)
        assert "settings saved" in " ".join(_painted(host, receipt).lower().split())
        await focus(prefix + "edit-summary-prompt")
        await pilot.press("enter")
        await _settle(host, pilot)
        assert screen.active_category == "internal-prompts"
        assert (
            screen.query_one("#internal-prompts-search", Input).value
            == "console.rewind_summarize"
        )
        assert screen.focused.id == "prompt-row-console__rewind_summarize"
        await _category(host, pilot, "Console Behavior")
        await focus(prefix + "target-percent")
        for resized in ((80, 24), (170, 48), size):
            await pilot.resize_terminal(*resized)
            await _settle(host, pilot)
            control = screen.query_one(prefix + "target-percent", Input)
            assert screen.focused is control
            _assert_painted(screen, control)
            label_is_painted(control)
            assert control.value == "55"
        assert screen.query_one("#settings-save-category", Button).disabled


@pytest.mark.parametrize("leave_while_saving", [False, True])
@private_profile_test
async def test_saved_background_reaches_existing_console_and_stops(
    request, monkeypatch, leave_while_saving
):
    """Saving on a covered or newly resumed Console updates the same effect child."""
    import asyncio
    import threading

    from textual.screen import Screen
    from textual.widgets import Checkbox

    from Tests.UI.test_destination_shells import _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    app = _build_test_app()
    host = ConsoleHarness(app)
    entered, release = threading.Event(), threading.Event()
    finished = threading.Event()
    outcome = {}
    real_writer = config.apply_settings_mutation_to_cli_config

    def delayed_writer(*args, **kwargs):
        entered.set()
        assert release.wait(15)
        try:
            outcome["result"] = real_writer(*args, **kwargs)
            return outcome["result"]
        finally:
            finished.set()

    async with host.run_test(size=(170, 48)) as pilot:
        console = host.screen
        await _wait_for_selector(
            console, pilot, "#console-transcript-background-effect"
        )
        effect = console.query_one("#console-transcript-background-effect")
        transcript = console.query_one("#console-native-transcript")
        from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        for role, content in (
            (ConsoleMessageRole.USER, "Keep this sample question"),
            (ConsoleMessageRole.ASSISTANT, "Keep this sample answer"),
        ):
            store.append_message(session.id, role=role, content=content)
        await console._sync_native_console_chat_ui()
        await transcript.refresh_messages()
        text_before = transcript.to_plain_text()
        assert "Keep this sample question" in text_before
        assert "Keep this sample answer" in text_before
        assert not effect.is_effect_active
        await host.push_screen(module.SettingsScreen(app))
        await _category(host, pilot, "Console Behavior")
        toggle = host.screen.query_one(
            "#settings-console-background-effect-enabled", Checkbox
        )
        toggle.focus()
        await _settle(host, pilot)
        await pilot.press("space", "tab", "enter", "end", "enter")
        await _settle(host, pilot)
        monkeypatch.setattr(
            config, "apply_settings_mutation_to_cli_config", delayed_writer
        )
        monkeypatch.setattr(
            module, "apply_settings_mutation_to_cli_config", delayed_writer
        )
        try:
            await pilot.press("escape", "s")
            assert await asyncio.to_thread(entered.wait, 3)
            assert not effect.is_effect_active
            if leave_while_saving:
                await host.pop_screen()
                await pilot.pause()
        finally:
            release.set()
        assert await asyncio.to_thread(finished.wait, 3)
        # Popping Settings cancels its Textual worker handle, but the already
        # running file-write thread still has to publish its result to the app.
        for _ in range(150):
            if getattr(app, "_console_appearance_refresh_generation", 0):
                break
            await pilot.pause(0.02)
        await _settle(host, pilot)
        assert outcome["result"].file_replaced
        assert outcome["result"].caches_reloaded
        assert effect.is_effect_active
        assert effect.settings.effect == "matrix"
        assert effect._timer is not None
        if not leave_while_saving:
            await host.pop_screen()
        await _settle(host, pilot)
        assert console.query_one("#console-native-transcript") is transcript
        assert transcript.to_plain_text() == text_before
        timer = effect._timer
        await host.push_screen(Screen())
        await host.pop_screen()
        await _settle(host, pilot)
        assert effect._timer is timer
        await host.push_screen(module.SettingsScreen(app))
        await _category(host, pilot, "Console Behavior")
        toggle = host.screen.query_one(
            "#settings-console-background-effect-enabled", Checkbox
        )
        toggle.focus()
        await _settle(host, pilot)
        await pilot.press("space", "escape", "s")
        await _settle(host, pilot)
        assert not effect.is_effect_active
        assert effect._timer is None
        await host.pop_screen()
        await _settle(host, pilot)
        assert console.query_one("#console-native-transcript") is transcript
        assert transcript.to_plain_text() == text_before
