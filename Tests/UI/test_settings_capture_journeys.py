"""Console capture Settings exercise real disclosure and persistence boundaries."""

import asyncio
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _category, _painted
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


def _setup():
    from tldw_chatbook import config

    result = config.apply_settings_mutation_to_cli_config(
        {
            "console": {
                "exchange_capture": False,
                "exchange_capture_pii_redaction": False,
                "trace_viewer_profile": "safe",
                "trace_viewer_profile_version": 1,
            }
        }
    )
    assert result.failure_phase is None
    return (
        _StyledDestinationHarness(_build_test_app(), "settings"),
        Path(config.get_cli_config_path()),
    )


async def _choose_viewer(host, pilot, value):
    viewer = host.screen.query_one("#settings-console-trace-viewer-profile", Select)
    viewer.focus()
    await pilot.wait_for_scheduled_animations()
    await pilot.press("enter", "end" if value == "full" else "home", "enter")
    assert viewer.value == value


async def _apply(host, pilot):
    button = host.screen.query_one("#settings-console-exchange-capture-apply", Button)
    button.focus()
    await pilot.wait_for_scheduled_animations()
    _assert_painted(host.screen, button)
    assert str(button.label) in _painted(host, button)
    await pilot.press("enter")
    await pilot.pause()
    return button


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(190, 55), (80, 24)])
@pytest.mark.parametrize("choice", ["escape", "cancel-button", "confirm-button"])
@private_profile_test
async def test_full_trace_view_requires_real_keyboard_confirmation(
    request, size, choice
):
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    host, path = _setup()
    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, "Console Behavior")
        screen = host.screen
        await _choose_viewer(host, pilot, "full")
        before = path.read_bytes()
        apply = await _apply(host, pilot)
        assert isinstance(host.screen, ConfirmationDialog)
        assert "PII detectors missed" in host.screen.message
        _assert_painted(host.screen, host.screen.query_one(".dialog-message"))
        for selector in ("#cancel-button", "#confirm-button"):
            button = host.screen.query_one(selector, Button)
            _assert_painted(host.screen, button)
            assert str(button.label) in _painted(host, button)
        assert path.read_bytes() == before
        if choice == "escape":
            await pilot.press("escape")
        else:
            host.screen.query_one("#" + choice, Button).focus()
            await pilot.press("enter")
        await _settle(host, pilot)
        assert host.screen is screen
        assert screen.focused is apply
        _assert_painted(screen, apply)
        expected = "full" if choice == "confirm-button" else "safe"
        if expected == "safe":
            assert path.read_bytes() == before
        console = tomllib.loads(path.read_text())["console"]
        assert console["trace_viewer_profile"] == expected
        assert console["exchange_capture"] is False
        assert console["exchange_capture_pii_redaction"] is False
        assert config.runtime_capture_policy().viewer_profile == expected
        await host.switch_screen(Screen())
        await host.switch_screen(SettingsScreen(host.app_instance))
        await _category(host, pilot, "Console Behavior")
        assert (
            host.screen.query_one(
                "#settings-console-trace-viewer-profile", Select
            ).value
            == expected
        )


@pytest.mark.asyncio
@private_profile_test
async def test_duplicate_full_apply_opens_one_dialog_and_commits_once(
    request, monkeypatch
):
    from tldw_chatbook.UI.Screens import settings_screen as module

    host, path = _setup()
    real_apply = module.apply_console_capture_settings
    writes = []

    def apply(**kwargs):
        writes.append(kwargs)
        return real_apply(**kwargs)

    monkeypatch.setattr(module, "apply_console_capture_settings", apply)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        screen = host.screen
        await _choose_viewer(host, pilot, "full")
        button = await _apply(host, pilot)
        screen.handle_console_exchange_capture_apply(
            SimpleNamespace(stop=lambda: None, button=button)
        )
        await pilot.pause()
        assert (
            len([s for s in host.screen_stack if isinstance(s, ConfirmationDialog)])
            == 1
        )
        assert writes == []
        host.screen.query_one("#confirm-button", Button).focus()
        await pilot.press("enter")
        await _settle(host, pilot)
        assert len(writes) == 1
        assert (
            tomllib.loads(path.read_text())["console"]["trace_viewer_profile"] == "full"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["before-replace", "exception", "cache-warning", "stale"]
)
@private_profile_test
async def test_full_view_save_failure_or_stale_consent_is_retriable(
    request, monkeypatch, failure
):
    from tldw_chatbook import config

    host, path = _setup()
    real_writer = config.apply_settings_mutation_to_cli_config
    before = path.read_bytes()

    def write(*args, **kwargs):
        if failure == "exception":
            raise OSError("test unavailable writer")
        if failure == "before-replace":
            return config.ConfigMutationResult(False, False, "before_replace")
        if failure == "cache-warning":
            publish_policy = kwargs.get("after_replace")

            def fail_cache():
                if publish_policy is not None:
                    publish_policy()
                raise OSError("test publication fault")

            kwargs["after_replace"] = fail_cache
        return real_writer(*args, **kwargs)

    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        screen = host.screen
        await _choose_viewer(host, pilot, "full")
        await _apply(host, pilot)
        if failure == "stale":
            real_writer({"console": {"max_parallel_runs": 3}})
            before = path.read_bytes()
        monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", write)
        host.screen.query_one("#confirm-button", Button).focus()
        await pilot.press("enter")
        await _settle(host, pilot)
        status = str(
            screen.query_one(
                "#settings-console-exchange-capture-status", Static
            ).render()
        )
        saved = failure == "cache-warning"
        assert ("Saved and active" in status) is saved
        if saved:
            assert "refresh degraded" in status
        else:
            assert "Failed" in status
            assert path.read_bytes() == before
        assert tomllib.loads(path.read_text())["console"]["trace_viewer_profile"] == (
            "full" if saved else "safe"
        )
        assert config.runtime_capture_policy().viewer_profile == (
            "full" if saved else "safe"
        )
        monkeypatch.setattr(
            config, "apply_settings_mutation_to_cli_config", real_writer
        )
        assert screen._reload_current_config() == "Config reload: loaded"
        await _category(host, pilot, "Overview")
        await _category(host, pilot, "Console Behavior")
        if not saved:
            await _choose_viewer(host, pilot, "full")
            await _apply(host, pilot)
            host.screen.query_one("#confirm-button", Button).focus()
            await pilot.press("enter")
            await _settle(host, pilot)
        assert (
            tomllib.loads(path.read_text())["console"]["trace_viewer_profile"] == "full"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_full_view_confirmation_uses_the_live_controller(request):
    from Tests.Chat.test_console_chat_controller_exchanges import _new_controller

    host, path = _setup()
    controller = _new_controller()
    controller.store.ensure_session()
    host.app_instance.console_runtime = SimpleNamespace(chat_controller=controller)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        screen = host.screen
        await _choose_viewer(host, pilot, "full")
        await _apply(host, pilot)
        host.screen.query_one("#confirm-button", Button).focus()
        await pilot.press("enter")
        await _settle(host, pilot)
        assert screen._console_capture_status == "Saved and active"
        assert (
            tomllib.loads(path.read_text())["console"]["trace_viewer_profile"] == "full"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_diagnostics_keyboard_reload_rebases_capture_controls_and_consent(
    request,
):
    from tldw_chatbook import config

    host, path = _setup()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        screen = host.screen
        result = config.apply_settings_mutation_to_cli_config(
            {"console": {"exchange_capture_pii_redaction": True}}
        )
        assert result.failure_phase is None
        await _category(host, pilot, "Diagnostics")
        await pilot.press("t")
        await _settle(host, pilot)
        assert screen._diagnostics_reload_result.startswith("Config reload: loaded")
        await _category(host, pilot, "Console Behavior")
        assert screen.query_one("#settings-console-trace-pii-redaction", Checkbox).value
        await _choose_viewer(host, pilot, "full")
        await _apply(host, pilot)
        host.screen.query_one("#confirm-button", Button).focus()
        await pilot.press("enter")
        await _settle(host, pilot)
        console = tomllib.loads(path.read_text())["console"]
        assert console["trace_viewer_profile"] == "full"
        assert console["exchange_capture_pii_redaction"] is True
        assert config.runtime_capture_policy().viewer_profile == "full"


@pytest.mark.asyncio
@private_profile_test
async def test_departed_settings_cannot_apply_a_late_full_confirmation(
    request, monkeypatch
):
    host, path = _setup()
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_confirmation(_dialog):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return True

    monkeypatch.setattr(host, "push_screen_wait", delayed_confirmation)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        await _choose_viewer(host, pilot, "full")
        before = path.read_bytes()
        await _apply(host, pilot)
        await asyncio.wait_for(entered.wait(), timeout=3)
        await host.switch_screen(Screen())
        release.set()
        await _settle(host, pilot)
        assert path.read_bytes() == before
