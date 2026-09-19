"""Permission-summary preferences survive slow writes and save recovery."""

import asyncio
import threading
import tomllib
from pathlib import Path

import pytest
from textual.screen import Screen
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _category, _painted
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness


def _setup(monkeypatch, writer=None):
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    config.apply_settings_mutation_to_cli_config(
        {
            "permission_summary": {
                "mode": "off",
                "provider": "OpenAI",
                "model": "review-model",
                "max_input_chars": 1234,
            }
        }
    )
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    if writer:
        monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", writer)
        monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", writer)
    return host, Path(config.get_cli_config_path())


async def _depart(host, pilot, departure):
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    if departure == "category":
        await pilot.press("escape", "/", *"Storage", "enter")
        await pilot.pause()
    else:
        old = host.screen
        await host.switch_screen(Screen())
        assert not old.is_attached
        await host.switch_screen(SettingsScreen(host.app_instance))
        await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("departure", ["category", "screen"])
@pytest.mark.parametrize(
    "field,first,last",
    [
        ("mode", "always", "off"),
        ("model", "first-model", "last-model"),
        ("provider", "Anthropic", "OpenAI"),
    ],
)
@private_profile_test
async def test_latest_permission_summary_edit_survives_a_held_write(
    request, monkeypatch, departure, field, first, last
):
    """A write already in progress must not override the latest UI choice."""
    from tldw_chatbook import config

    real_writer = config.apply_settings_mutation_to_cli_config
    entered, release = threading.Event(), threading.Event()

    def gated(payload, *args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(15)
        return real_writer(payload, *args, **kwargs)

    host, path = _setup(monkeypatch, gated)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        control = host.screen.query_one(f"#settings-permission-summary-{field}")
        try:
            control.value = first
            await pilot.pause()
            assert await asyncio.to_thread(entered.wait, 3)
            control.value = last
            await pilot.pause()
            await _depart(host, pilot, departure)
        finally:
            release.set()
        await _settle(host, pilot)
        await _category(host, pilot, "Console Behavior")
        saved = tomllib.loads(path.read_text())["permission_summary"]
        assert saved[field] == last
        assert saved["max_input_chars"] == 1234
        assert (
            host.screen.query_one(f"#settings-permission-summary-{field}").value == last
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("departure", ["category", "screen"])
@pytest.mark.parametrize("size", [(190, 55), (80, 24)])
@private_profile_test
async def test_permission_summary_failure_retains_values_and_keyboard_retry(
    request, monkeypatch, departure, size
):
    """Failed opt-in stays visibly unsaved and retries without losing edits."""
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    real_writer = config.apply_settings_mutation_to_cli_config

    def fail(*args, **kwargs):
        return config.ConfigMutationResult(False, False, "before_replace")

    host, path = _setup(monkeypatch, fail)
    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, "Console Behavior")
        before = path.read_bytes()
        host.screen.query_one(
            "#settings-permission-summary-mode", Select
        ).value = "always"
        await _settle(host, pilot)
        assert path.read_bytes() == before
        assert "not saved" in str(
            host.screen.query_one(
                "#settings-permission-summary-result", Static
            ).render()
        )
        await _depart(host, pilot, departure)
        await _category(host, pilot, "Console Behavior")
        assert (
            host.screen.query_one("#settings-permission-summary-mode", Select).value
            == "always"
        )
        assert (
            config.get_runtime_config_snapshot().values["permission_summary"]["mode"]
            == "off"
        )
        retry = host.screen.query_one("#settings-permission-summary-retry", Button)
        retry.focus()
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        _assert_painted(host.screen, retry)
        assert "Retry" in _painted(host, retry)
        guide = host.screen.query_one(
            "#settings-console-behavior-field-guide-3", Static
        )
        assert "no Save step" in str(guide.render())
        monkeypatch.setattr(
            config, "apply_settings_mutation_to_cli_config", real_writer
        )
        monkeypatch.setattr(
            module, "apply_settings_mutation_to_cli_config", real_writer
        )
        await pilot.press("enter")
        await _settle(host, pilot)
        assert tomllib.loads(path.read_text())["permission_summary"]["mode"] == "always"
        assert not retry.display
        assert host.screen.focused is host.screen.query_one(
            "#settings-permission-summary-mode"
        )
        assert "not saved" not in str(
            host.screen.query_one(
                "#settings-permission-summary-result", Static
            ).render()
        )


@pytest.mark.asyncio
@private_profile_test
async def test_permission_summary_open_and_revisit_do_not_write(request, monkeypatch):
    """Mount-time events must neither persist defaults nor rewrite saved values."""
    host, path = _setup(monkeypatch)
    async with host.run_test(size=(190, 55)) as pilot:
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        await _category(host, pilot, "Console Behavior")
        await _depart(host, pilot, "screen")
        await _category(host, pilot, "Console Behavior")
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
        assert (
            host.screen.query_one("#settings-permission-summary-provider", Input).value
            == "OpenAI"
        )
        assert (
            host.screen.query_one("#settings-permission-summary-model", Input).value
            == "review-model"
        )
        disclosure = str(
            host.screen.query_one(
                "#settings-permission-summary-disclosure", Static
            ).render()
        )
        assert "excerpt of this conversation" in disclosure
        assert "you designate" in disclosure


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_timing", ["none", "before_receipt", "after_receipt"])
@private_profile_test
async def test_permission_summary_reports_replaced_file_when_cache_reload_fails(
    request, monkeypatch, refresh_timing
):
    """A saved file with stale live caches must not be reported as fully applied."""
    from tldw_chatbook import config

    real_writer = config.apply_settings_mutation_to_cli_config

    def partial(payload, *args, **kwargs):
        def fail_publication():
            raise OSError("test publication fault")

        result = real_writer(payload, *args, after_replace=fail_publication, **kwargs)
        assert result.file_replaced and not result.caches_reloaded
        if refresh_timing == "before_receipt":
            real_writer(
                {"permission_summary": {"mode": "off", "model": "reloaded-model"}}
            )
        return result

    host, path = _setup(monkeypatch, partial)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        host.screen.query_one(
            "#settings-permission-summary-mode", Select
        ).value = "always"
        await _settle(host, pilot)
        assert tomllib.loads(path.read_text())["permission_summary"]["mode"] == (
            "off" if refresh_timing == "before_receipt" else "always"
        )
        if refresh_timing == "after_receipt":
            real_writer(
                {"permission_summary": {"mode": "off", "model": "reloaded-model"}}
            )
        before = path.stat().st_mtime_ns
        await _depart(host, pilot, "screen")
        await _category(host, pilot, "Console Behavior")
        assert host.screen.query_one(
            "#settings-permission-summary-mode", Select
        ).value == ("always" if refresh_timing == "none" else "off")
        message = str(
            host.screen.query_one(
                "#settings-permission-summary-result", Static
            ).render()
        )
        if refresh_timing != "none":
            assert (
                host.screen.query_one("#settings-permission-summary-model", Input).value
                == "reloaded-model"
            )
            assert "could not be refreshed" not in message
        else:
            assert "Saved" in message and "could not be refreshed" in message
        assert not host.screen.query_one(
            "#settings-permission-summary-retry", Button
        ).display
        assert path.stat().st_mtime_ns == before


@pytest.mark.asyncio
@private_profile_test
async def test_permission_summary_revisit_uses_reloaded_config(request, monkeypatch):
    """A successful old edit cannot conceal an Advanced Config change on return."""
    from tldw_chatbook import config
    from tldw_chatbook.Chat.permission_summary_service import resolve_permission_summary

    host, path = _setup(monkeypatch)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        host.screen.query_one(
            "#settings-permission-summary-mode", Select
        ).value = "always"
        await _settle(host, pilot)
        await _depart(host, pilot, "category")
        config.apply_settings_mutation_to_cli_config(
            {"permission_summary": {"mode": "fallback", "model": "reloaded-model"}}
        )
        before = path.stat().st_mtime_ns
        await _category(host, pilot, "Console Behavior")
        assert (
            host.screen.query_one("#settings-permission-summary-mode", Select).value
            == "fallback"
        )
        assert (
            host.screen.query_one("#settings-permission-summary-model", Input).value
            == "reloaded-model"
        )
        resolution = resolve_permission_summary(
            config.get_runtime_config_snapshot().values
        )
        assert resolution.mode == "fallback"
        assert resolution.model == "reloaded-model"
        assert path.stat().st_mtime_ns == before
        host.screen.query_one("#settings-permission-summary-mode", Select).value = "off"
        await _settle(host, pilot)
        assert tomllib.loads(path.read_text())["permission_summary"]["mode"] == "off"
