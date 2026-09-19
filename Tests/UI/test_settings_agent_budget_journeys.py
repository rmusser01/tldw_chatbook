"""Console budget saves must agree with the next run and remain editable."""

import tomllib
from pathlib import Path

import pytest
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _category, _painted
from Tests.UI.test_settings_provider_keyboard_journeys import (
    _edit,
    _revert,
    _settle,
    _tab_to,
)
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness

STEPS = "#settings-console-agent-max-steps"
BUDGET_VALUES = (
    ("agent_max_total_tokens", "3219876", "max_total_tokens", 3219876),
    ("agent_max_wall_seconds", "120.5", "max_wall_seconds", 120.5),
    ("agent_max_tool_call_seconds", "0", "max_tool_call_seconds", 0),
    ("agent_max_model_turns", "200", "max_model_turns", 200),
    ("agent_max_steps", "400", "max_steps", 400),
)


@pytest.mark.asyncio
@private_profile_test
async def test_over_limit_steps_are_refused_without_saving_a_different_runtime_budget(
    request,
):
    """A 200000-step save must not report success then run with 25000 steps."""
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_agent_bridge import console_run_budget

    app = _build_test_app()
    host = _StyledDestinationHarness(app, "settings")
    path = Path(config.get_cli_config_path())
    async with host.run_test(size=(80, 24)) as pilot:
        await _category(host, pilot, "Console Behavior")
        original = path.read_bytes()
        original_runtime = console_run_budget().max_steps
        await _edit(host, pilot, STEPS, "200000")
        await pilot.press("escape", "s")
        await _settle(host, pilot)

        assert path.read_bytes() == original, (
            "Settings wrote a step budget that the runtime silently replaces"
        )
        assert console_run_budget().max_steps == original_runtime
        assert host.screen.query_one(STEPS, Input).value == "200000"
        assert host.screen._category_has_unsaved_changes(host.screen.active_category)
        assert "199999" in host.screen._console_behavior_result.replace(",", "")


@pytest.mark.asyncio
@private_profile_test
async def test_legacy_over_limit_steps_display_the_runtime_fallback(request):
    """The field must not promise a legacy raw value that the run cannot use."""
    from Tests.UI.test_settings_agent_run_budget import _Screen
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_agent_bridge import console_run_budget
    from tldw_chatbook.UI.Screens.settings_screen import AGENT_BUDGET_FIELDS_BY_KEY

    result = config.apply_settings_mutation_to_cli_config(
        {"console": {"agent_max_steps": 200000}}
    )
    assert result.file_replaced
    assert console_run_budget().max_steps == 25000
    field = AGENT_BUDGET_FIELDS_BY_KEY["agent_max_steps"]
    assert (
        _Screen({"agent_max_steps": 200000})._loaded_agent_budget_value(field) == 25000
    )


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@private_profile_test
async def test_budget_keyboard_save_revert_recovery_and_next_run_agree(
    request, monkeypatch, theme, size
):
    """A painted edit must survive navigation and retry, then reach the real resolver."""
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_agent_bridge import (
        UNLIMITED_TOOL_CALL_DEADLINE_SECONDS,
        console_run_budget,
    )
    from tldw_chatbook.UI.Screens import settings_screen as module

    app = _build_test_app()
    host = _StyledDestinationHarness(app, "settings")
    host.theme = theme
    path = Path(config.get_cli_config_path())
    fields = module.AGENT_BUDGET_FIELDS_BY_KEY

    def assert_values(screen):
        for key, _text, _attribute, value in BUDGET_VALUES:
            assert (
                float(screen.query_one(f"#{fields[key].widget_id}", Input).value)
                == value
            )

    async def edit_values(pilot):
        for key, text, _attribute, _value in BUDGET_VALUES:
            selector = f"#{fields[key].widget_id}"
            await _edit(host, pilot, selector, text)
            control = host.screen.query_one(selector, Input)
            label = control.parent.query_one(".settings-input-label", Static)
            _assert_painted(host.screen, label)
            assert str(label.renderable) in _painted(host, label)
            assert control.value in _painted(host, control)

    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, "Console Behavior")
        screen = host.screen
        before = path.read_bytes()
        initial_budget = console_run_budget()
        initial_steps = screen.query_one(STEPS, Input).value
        assert not screen._category_has_unsaved_changes(screen.active_category)

        # An empty or below-floor field keeps the exact draft and cannot write.
        wall = f"#{fields['agent_max_wall_seconds'].widget_id}"
        await _edit(host, pilot, wall, "0")
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert path.read_bytes() == before
        assert screen.query_one(wall, Input).value == "0"
        assert "at least 1" in screen._console_behavior_result

        await edit_values(pilot)
        assert path.read_bytes() == before
        assert console_run_budget() == initial_budget
        warning = screen.query_one(
            "#settings-console-agent-budget-step-warning", Static
        )
        assert "134 tool-calling rounds, not 200" in str(warning.renderable)

        await _category(host, pilot, "Overview")
        await _category(host, pilot, "Console Behavior")
        assert_values(screen)
        await _revert(host, pilot, discard=False)
        assert_values(screen)
        await _revert(host, pilot, discard=True)
        assert screen.query_one(STEPS, Input).value == initial_steps
        assert not screen._category_has_unsaved_changes(screen.active_category)
        assert path.read_bytes() == before

        await edit_values(pilot)
        real_writer = config.apply_settings_mutation_to_cli_config

        def fail(*args, **kwargs):
            return config.ConfigMutationResult(False, False, "before_replace")

        monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", fail)
        monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", fail)
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert path.read_bytes() == before
        assert console_run_budget() == initial_budget
        assert screen._category_has_unsaved_changes(screen.active_category)
        assert "Failed" in screen._console_behavior_result
        assert_values(screen)

        monkeypatch.setattr(
            config, "apply_settings_mutation_to_cli_config", real_writer
        )
        monkeypatch.setattr(
            module, "apply_settings_mutation_to_cli_config", real_writer
        )
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        saved = tomllib.loads(path.read_text())["console"]
        runtime = console_run_budget()
        for key, _text, attribute, value in BUDGET_VALUES:
            assert saved[key] == value
            assert app.app_config["console"][key] == value
            expected = (
                UNLIMITED_TOOL_CALL_DEADLINE_SECONDS
                if key == "agent_max_tool_call_seconds"
                else value
            )
            assert getattr(runtime, attribute) == expected
        assert not screen._category_has_unsaved_changes(screen.active_category)
        receipt = screen.query_one("#settings-console-behavior-result", Static)
        receipt.scroll_visible(animate=False)
        await _settle(host, pilot)
        _assert_painted(screen, receipt)
        assert "console behavior settings saved." in " ".join(
            _painted(host, receipt).lower().split()
        )

        # The supported maximum is usable, and zero keeps the token unlimited mode.
        await _edit(host, pilot, STEPS, "199999")
        tokens = f"#{fields['agent_max_total_tokens'].widget_id}"
        await _edit(host, pilot, tokens, "0")
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert console_run_budget().max_steps == 199999
        assert console_run_budget().max_total_tokens == 0
        assert tomllib.loads(path.read_text())["console"]["agent_max_steps"] == 199999
        await _category(host, pilot, "Overview")
        await _category(host, pilot, "Console Behavior")
        control = await _tab_to(host, pilot, STEPS)
        assert control.value == "199999"
        help_text = screen.query_one(f"{STEPS}-help", Static)
        assert "199,999" in str(help_text.renderable)
        assert screen.query_one("#settings-save-category", Button).disabled
        for resized in ((80, 24), (170, 48), size):
            await pilot.resize_terminal(*resized)
            await _settle(host, pilot)
            focused = screen.query_one(STEPS, Input)
            assert screen.focused is focused
            _assert_painted(screen, focused)
            assert "199999" in _painted(host, focused)
