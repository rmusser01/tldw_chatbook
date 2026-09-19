"""Settings credential projections remain responsive and follow current selection."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_settings_configuration_hub import (
    _open_settings_category,
    _settle_settings_mount_storm,
)
from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


@pytest.fixture
def credential_io(monkeypatch, tmp_path):
    path = tmp_path / "credential.json"
    clock = [100.0]
    monotonic_clock = [100.0]
    payload = {"claudeAiOauth": {"accessToken": "private-token", "expiresAt": 101000}}
    path.write_text(json.dumps(payload))
    entered, release = threading.Event(), threading.Event()
    readers = []
    original_read = Path.read_text

    def gated_read(target, *args, **kwargs):
        if target == path:
            readers.append(threading.get_ident())
            if not entered.is_set():
                entered.set()
                release.wait(5)
        return original_read(target, *args, **kwargs)

    cache = subscription._SubscriptionReadinessCache()
    monkeypatch.setattr(subscription, "_SUBSCRIPTION_READINESS_CACHE", cache)
    monkeypatch.setattr(subscription, "DEFAULT_CREDENTIALS_PATH", path)
    monkeypatch.setattr(subscription, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        subscription,
        "time",
        SimpleNamespace(time=lambda: clock[0], monotonic=lambda: monotonic_clock[0]),
    )
    monkeypatch.setattr(Path, "read_text", gated_read)
    yield SimpleNamespace(
        path=path,
        clock=clock,
        monotonic_clock=monotonic_clock,
        entered=entered,
        release=release,
        readers=readers,
        cache=cache,
    )
    release.set()
    if cache._worker is not None:
        cache._worker.join(timeout=3)
        assert not cache._worker.is_alive()


def settings_app():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "Anthropic",
        "model": "claude-sonnet-4-5",
    }
    app.app_config["api_settings"] = {
        "anthropic": {
            "auth_source": "claude_subscription",
            "model": "claude-sonnet-4-5",
        },
        "openai": {"api_key": "configured-other-key", "model": "gpt-4o"},
    }
    return app


async def wait_for_copy(screen, selector, expected):
    async with asyncio.timeout(3):
        while expected not in str(screen.query_one(selector, Static).renderable):
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@private_profile_test
async def test_settings_mount_and_overview_refresh_without_another_edit(
    credential_io, request
):
    host = DestinationHarness(settings_app(), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        assert credential_io.entered.is_set()
        assert len(credential_io.readers) == 1
        assert credential_io.readers[0] != threading.get_ident()
        await wait_for_copy(
            screen, "#settings-overview-configuration", "Checking Claude"
        )
        heartbeats = []
        timer = screen.set_interval(0.01, lambda: heartbeats.append(None))
        async with asyncio.timeout(1):
            while len(heartbeats) < 3:
                await asyncio.sleep(0.01)
        assert not credential_io.release.is_set()
        timer.stop()
        credential_io.release.set()
        await wait_for_copy(screen, "#settings-overview-configuration", "Status: Ready")
        credential_io.entered.clear()
        credential_io.release.clear()
        credential_io.monotonic_clock[0] = 106.0
        await wait_for_copy(
            screen, "#settings-overview-configuration", "Checking Claude"
        )
        assert credential_io.entered.is_set()
        assert len(credential_io.readers) == 2
        credential_io.release.set()
        await wait_for_copy(screen, "#settings-overview-configuration", "Status: Ready")
        credential_io.clock[0] = 102.0
        await wait_for_copy(screen, "#settings-overview-configuration", "expired")


@pytest.mark.asyncio
@private_profile_test
async def test_settings_fast_completion_is_visible_on_first_poll(
    credential_io, monkeypatch, request
):
    original_set_interval = SettingsScreen.set_interval

    def start_paused(screen, interval, callback, *args, **kwargs):
        timer = original_set_interval(screen, interval, callback, *args, **kwargs)
        if getattr(callback, "__name__", "") == "_poll_subscription_readiness":
            timer.pause()
        return timer

    monkeypatch.setattr(SettingsScreen, "set_interval", start_paused)
    host = DestinationHarness(settings_app(), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        await wait_for_copy(
            screen, "#settings-overview-configuration", "Checking Claude"
        )
        assert screen._subscription_readiness_observation is None
        credential_io.release.set()
        async with asyncio.timeout(3):
            while credential_io.cache.revision == 0:
                await asyncio.sleep(0.01)
        screen._subscription_readiness_timer.resume()
        await wait_for_copy(screen, "#settings-overview-configuration", "Status: Ready")


@pytest.mark.asyncio
@private_profile_test
async def test_settings_provider_check_refreshes_and_preserves_unsaved_fields(
    credential_io,
    request,
):
    host = DestinationHarness(settings_app(), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        assert credential_io.readers[0] != threading.get_ident()
        await wait_for_copy(
            screen, "#settings-provider-credential-status", "Checking Claude"
        )
        model = screen.query_one("#settings-model-value", Input)
        model.value = "unsaved-model"
        await pilot.pause()
        screen.action_settings_test_category(allow_text_entry_focus=True)
        await wait_for_copy(screen, "#settings-provider-test-result", "Checking Claude")
        credential_io.release.set()
        await wait_for_copy(
            screen, "#settings-provider-credential-status", "Claude subscription"
        )
        await wait_for_copy(
            screen, "#settings-provider-test-result", "configuration=complete"
        )
        assert model.value == "unsaved-model"
        assert "model" in screen._provider_draft().dirty_keys
        assert (
            host.app_instance.app_config["chat_defaults"]["model"]
            == "claude-sonnet-4-5"
        )
        assert "private-token" not in str(
            screen.query_one("#settings-provider-test-result", Static).renderable
        )
        credential_io.entered.clear()
        credential_io.release.clear()
        credential_io.monotonic_clock[0] = 106.0
        await wait_for_copy(screen, "#settings-provider-test-result", "Checking Claude")
        assert credential_io.entered.is_set()
        credential_io.release.set()
        await wait_for_copy(
            screen, "#settings-provider-test-result", "configuration=complete"
        )
        credential_io.clock[0] = 102.0
        await wait_for_copy(screen, "#settings-provider-credential-status", "expired")
        await wait_for_copy(screen, "#settings-provider-test-result", "expired")
        assert "configuration=blocked" in str(
            screen.query_one("#settings-provider-test-result", Static).renderable
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["provider", "model"])
@private_profile_test
async def test_settings_late_subscription_result_does_not_replace_edited_draft(
    credential_io, changed_field, request
):
    host = DestinationHarness(settings_app(), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.action_settings_test_category(allow_text_entry_focus=True)
        if changed_field == "provider":
            screen.query_one("#settings-provider-value", Select).value = "openai"
        else:
            screen.query_one("#settings-model-value", Input).value = "another-model"
        await pilot.pause()
        before = screen._provider_test_result
        credential_io.release.set()
        async with asyncio.timeout(3):
            while credential_io.cache.revision == 0:
                await asyncio.sleep(0.01)
        await pilot.pause(0.3)
        status = str(
            screen.query_one("#settings-provider-credential-status", Static).renderable
        )
        assert screen._provider_test_result == before
        if changed_field == "provider":
            assert "local config key saved" in status
            assert "Claude" not in status
            assert (
                screen.query_one("#settings-provider-value", Select).value == "openai"
            )
        else:
            assert "Claude subscription" in status
            assert (
                screen.query_one("#settings-model-value", Input).value
                == "another-model"
            )


@pytest.mark.asyncio
@private_profile_test
async def test_settings_missing_subscription_reports_owner_recovery_after_completion(
    credential_io,
    request,
):
    host = DestinationHarness(settings_app(), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.action_settings_test_category(allow_text_entry_focus=True)
        credential_io.path.unlink()
        credential_io.release.set()
        await wait_for_copy(
            screen,
            "#settings-provider-credential-status",
            "missing; log in with Claude Code",
        )
        await wait_for_copy(
            screen, "#settings-provider-test-result", "configuration=blocked"
        )
        result = str(
            screen.query_one("#settings-provider-test-result", Static).renderable
        )
        assert "Checking Claude" not in result
        assert "private-token" not in result
        assert str(credential_io.path) not in result


@pytest.mark.asyncio
@private_profile_test
async def test_settings_dismissal_stops_projection_of_pending_credentials(
    credential_io,
    request,
):
    host = DestinationHarness(settings_app(), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.action_settings_test_category(allow_text_entry_focus=True)
        before = screen._provider_test_result
        await host.pop_screen()
        await pilot.pause()
        assert not screen.is_attached
        assert screen._subscription_readiness_timer is None
        credential_io.release.set()
        async with asyncio.timeout(3):
            while credential_io.cache.revision == 0:
                await asyncio.sleep(0.01)
        await pilot.pause(0.3)
        assert screen._provider_test_result == before
