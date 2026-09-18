"""The immediate thinking preference survives Settings lifetime boundaries."""

import asyncio
import threading
import tomllib
from pathlib import Path

import pytest
from textual.screen import Screen
from textual.widgets import Checkbox, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_overview_search_journeys import _category
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness

SELECTOR = "#settings-console-show-model-thinking"


def _setup():
    from tldw_chatbook import config

    result = config.apply_settings_mutation_to_cli_config(
        {"console": {"show_model_thinking": False}}
    )
    assert result.failure_phase is None
    return (
        _StyledDestinationHarness(_build_test_app(), "settings"),
        Path(config.get_cli_config_path()),
    )


async def _toggle(host, pilot):
    checkbox = host.screen.query_one(SELECTOR, Checkbox)
    checkbox.focus()
    await pilot.wait_for_scheduled_animations()
    await pilot.press("space")
    await pilot.pause()
    return checkbox


@pytest.mark.asyncio
@private_profile_test
async def test_thinking_visibility_rebases_after_config_reload(request):
    from tldw_chatbook import config

    host, path = _setup()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        await _toggle(host, pilot)
        await _settle(host, pilot)
        assert tomllib.loads(path.read_text())["console"]["show_model_thinking"]
        config.apply_settings_mutation_to_cli_config(
            {"console": {"show_model_thinking": False}}
        )
        assert host.screen._reload_current_config() == "Config reload: loaded"
        await _category(host, pilot, "Overview")
        await _category(host, pilot, "Console Behavior")
        assert not host.screen.query_one(SELECTOR, Checkbox).value
        await _toggle(host, pilot)
        await _settle(host, pilot)
        assert tomllib.loads(path.read_text())["console"]["show_model_thinking"]
        assert host.app_instance.app_config["console"]["show_model_thinking"]


@pytest.mark.asyncio
@pytest.mark.parametrize("edit_after_return", [False, True])
@pytest.mark.parametrize("cache_failure", [False, True])
@private_profile_test
async def test_thinking_visibility_drains_across_settings_recreation(
    request, monkeypatch, edit_after_return, cache_failure
):
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    host, path = _setup()
    real_writer = config.apply_settings_mutation_to_cli_config
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def gated(payload):
        calls.append(payload)
        ordinal = len(calls)
        if ordinal == 1:
            entered.set()
            assert release.wait(15)

        def fail_publication():
            raise OSError("test publication fault")

        try:
            return real_writer(
                payload,
                after_replace=fail_publication
                if cache_failure and ordinal == 1
                else None,
            )
        finally:
            finished.set()

    monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", gated)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        try:
            await _toggle(host, pilot)
            assert await asyncio.to_thread(entered.wait, 3)
            await _toggle(host, pilot)
            old = host.screen
            await host.switch_screen(Screen())
            assert not old.is_attached
            await host.switch_screen(module.SettingsScreen(host.app_instance))
            await pilot.pause()
            await pilot.press("escape", "/", *"Console Behavior", "enter")
            await pilot.pause()
            if edit_after_return:
                host.screen._console_behavior_result = "Could not save earlier choice."
                await _toggle(host, pilot)
        finally:
            release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await _settle(host, pilot)
        expected = edit_after_return
        assert (
            tomllib.loads(path.read_text())["console"]["show_model_thinking"]
            is expected
        )
        assert (
            host.app_instance.app_config["console"]["show_model_thinking"] is expected
        )
        assert host.screen.query_one(SELECTOR, Checkbox).value is expected
        assert "saved" in host.screen._console_behavior_result
        assert "Could not save" not in host.screen._console_behavior_result
        if cache_failure and edit_after_return:
            assert "refresh" in host.screen._console_behavior_result


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(190, 55), (80, 24)])
@pytest.mark.parametrize("failure", ["before-replace", "conflict", "exception"])
@private_profile_test
async def test_thinking_visibility_failed_write_is_readable_and_retriable(
    request, monkeypatch, size, failure
):
    from Tests.UI.test_library_rag_result_focus import _assert_painted
    from Tests.UI.test_settings_overview_search_journeys import _painted
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    host, path = _setup()
    real_writer = config.apply_settings_mutation_to_cli_config

    def fail(payload):
        assert payload == {"console": {"show_model_thinking": True}}
        if failure == "exception":
            raise OSError("test unavailable writer")
        return config.ConfigMutationResult(
            False,
            False,
            "before_replace" if failure == "before-replace" else None,
            conflict=failure == "conflict",
        )

    monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", fail)
    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, "Console Behavior")
        before = path.read_bytes()
        checkbox = await _toggle(host, pilot)
        await _settle(host, pilot)
        assert path.read_bytes() == before
        assert checkbox.value is False
        assert host.app_instance.app_config["console"]["show_model_thinking"] is False
        assert "prior setting was restored" in host.screen._console_behavior_result
        assert host.screen.focused is checkbox
        _assert_painted(host.screen, checkbox)
        assert "Show model thinking (Off)" in _painted(host, checkbox)
        receipt = host.screen.query_one(
            "#settings-console-model-thinking-result", Static
        )
        _assert_painted(host.screen, receipt)
        assert "Save failed; setting restored." in _painted(host, receipt)
        assert "Toggle again to retry." in _painted(host, receipt)
        monkeypatch.setattr(
            module, "apply_settings_mutation_to_cli_config", real_writer
        )
        await pilot.press("space")
        await _settle(host, pilot)
        assert tomllib.loads(path.read_text())["console"]["show_model_thinking"] is True
        assert checkbox.value is True
        assert "Show model thinking (On)" in _painted(host, checkbox)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["saved", "failed", "cache-warning"])
@private_profile_test
async def test_thinking_visibility_reconciles_while_settings_is_absent(
    request, monkeypatch, outcome
):
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    host, path = _setup()
    real_writer = config.apply_settings_mutation_to_cli_config
    entered, release = threading.Event(), threading.Event()

    def gated(payload):
        entered.set()
        assert release.wait(15)
        if outcome == "failed":
            return config.ConfigMutationResult(False, False, "before_replace")
        if outcome == "cache-warning":

            def fail_publication():
                raise OSError("test publication fault")

            return real_writer(payload, after_replace=fail_publication)
        return real_writer(payload)

    monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", gated)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        try:
            await _toggle(host, pilot)
            assert await asyncio.to_thread(entered.wait, 3)
            old = host.screen
            await host.switch_screen(Screen())
            assert not old.is_attached
        finally:
            release.set()
        await _settle(host, pilot)
        fails = outcome == "failed"
        assert (
            tomllib.loads(path.read_text())["console"]["show_model_thinking"]
            is not fails
        )
        assert (
            host.app_instance.app_config["console"]["show_model_thinking"] is not fails
        )
        await host.switch_screen(module.SettingsScreen(host.app_instance))
        await _category(host, pilot, "Console Behavior")
        assert host.screen.query_one(SELECTOR, Checkbox).value is not fails
        message = str(
            host.screen.query_one(
                "#settings-console-model-thinking-result", Static
            ).render()
        )
        assert "have not been saved" not in message
        if fails:
            assert "Save failed; setting restored." in message
            assert "Toggle again to retry." in message
        elif outcome == "cache-warning":
            assert "Saved. Reload settings to refresh." in message
        else:
            assert "visibility saved" in message


@pytest.mark.asyncio
@private_profile_test
async def test_thinking_setting_updates_mounted_console_without_altering_evidence(
    request,
):
    from Tests.UI.test_console_thinking_disclosures import _displayable
    from Tests.UI.test_destination_shells import _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.thinking_blocks import ThinkingEnvelope
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.Widgets.Console.console_assistant_turn import (
        ConsoleActivityDisclosure,
    )
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    host, path = _setup()
    host = ConsoleHarness(host.app_instance)
    async with host.run_test(size=(190, 55)) as pilot:
        console = host.screen
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Public sample answer",
        )
        assistant = store.replace_message_thinking(
            assistant.id,
            ThinkingEnvelope((_displayable("Synthetic displayable sample"),)),
        )
        assert assistant is not None
        original_thinking = assistant.thinking
        original_config = tomllib.loads(path.read_text())
        await console._sync_native_console_chat_ui()
        transcript = console.query_one(ConsoleTranscript)
        await transcript.refresh_messages()
        assert not transcript.query(ConsoleActivityDisclosure)
        await host.push_screen(SettingsScreen(host.app_instance))
        await _category(host, pilot, "Console Behavior")
        await _toggle(host, pilot)
        await _settle(host, pilot)
        await transcript.refresh_messages()
        assert transcript.query(ConsoleActivityDisclosure)
        await _toggle(host, pilot)
        await _settle(host, pilot)
        await transcript.refresh_messages()
        assert not transcript.query(ConsoleActivityDisclosure)
        preserved = store.get_message(assistant.id)
        assert preserved.thinking == original_thinking
        assert preserved.content == "Public sample answer"
        saved = tomllib.loads(path.read_text())["console"]
        for key in (
            "show_model_thinking",
            "exchange_capture",
            "reasoning_history",
            "thinking_history_policy_default",
        ):
            assert saved.get(key) == original_config["console"].get(key)
