"""Mounted Console controls remain usable while credentials are being read."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_provider_apply_defaults_flow import (
    _console_app,
    _ConsoleFlowHarness,
)
from Tests.UI.test_console_session_settings import ModalHarness
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal


@pytest.fixture
def expiring_credential(monkeypatch, tmp_path):
    """Control credential expiry and cache TTL without advancing the UI clock."""
    path = tmp_path / "credential.json"
    path.write_text("{}")
    cache = subscription._SubscriptionReadinessCache()
    state = SimpleNamespace(
        wall=1000.0,
        monotonic=0.0,
        expires_at_ms=1001000,
        release=threading.Event(),
        readers=[],
        cache=cache,
    )
    state.release.set()
    monkeypatch.setattr(subscription, "_SUBSCRIPTION_READINESS_CACHE", cache)
    monkeypatch.setattr(subscription, "DEFAULT_CREDENTIALS_PATH", path)
    monkeypatch.setattr(subscription, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        subscription,
        "time",
        SimpleNamespace(time=lambda: state.wall, monotonic=lambda: state.monotonic),
    )
    original_read = Path.read_text

    def read_credential(target, *args, **kwargs):
        if target != path:
            return original_read(target, *args, **kwargs)
        state.readers.append(threading.get_ident())
        assert state.release.wait(5)
        return json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "private-token",
                    "expiresAt": state.expires_at_ms,
                }
            }
        )

    monkeypatch.setattr(Path, "read_text", read_credential)
    yield state
    state.release.set()
    if cache._worker is not None:
        cache._worker.join(timeout=3)
        assert not cache._worker.is_alive()


async def _wait_for_status(widget, text):
    try:
        async with asyncio.timeout(3):
            while text not in str(widget.renderable):
                await asyncio.sleep(0.01)
    except TimeoutError:
        pytest.fail(f"Expected {text!r}; rendered {str(widget.renderable)!r}")


@pytest.mark.asyncio
async def test_modal_refreshes_expiry_and_renews_stale_snapshot_without_edits(
    expiring_credential,
):
    app = ModalHarness()
    app.app_config = {
        "api_settings": {"anthropic": {"auth_source": "claude_subscription"}}
    }
    modal = ConsoleSettingsModal(
        settings=ConsoleSessionSettings(
            provider="anthropic", model="claude-sonnet-4-5"
        ),
        app_config=app.app_config,
        providers_models={"anthropic": ["claude-sonnet-4-5"]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
    )
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(modal)
        status = modal.query_one("#console-settings-readiness", Static)
        await _wait_for_status(status, "Ready to send")
        await pilot.pause(0.3)
        revision = expiring_credential.cache.revision
        expiring_credential.wall = 1002
        await _wait_for_status(status, "expired")
        assert expiring_credential.cache.revision == revision
        assert modal.query_one("#console-settings-configure-credential", Button).display

        expiring_credential.release.clear()
        expiring_credential.expires_at_ms = 2000000
        expiring_credential.monotonic = subscription._KEYCHAIN_TTL_S + 1
        await _wait_for_status(status, "Checking Claude")
        assert len(expiring_credential.readers) == 2
        assert all(
            reader != threading.get_ident() for reader in expiring_credential.readers
        )
        assert not modal.query_one(
            "#console-settings-configure-credential", Button
        ).display
        expiring_credential.release.set()
        await _wait_for_status(status, "Ready to send")
        assert "private-token" not in str(status.renderable)


@pytest.mark.asyncio
@private_profile_test
async def test_console_send_refreshes_expiry_and_renews_stale_snapshot_without_edits(
    expiring_credential,
    request,
):
    app = _console_app()
    app.app_config["api_settings"]["anthropic"] = {"auth_source": "claude_subscription"}
    host = _ConsoleFlowHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen
        console._provider_readiness_app_config = lambda: app.app_config
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        store.replace_session_settings(
            store.active_session_id,
            ConsoleSessionSettings(provider="anthropic", model="claude-sonnet-4-5"),
        )
        await console._sync_native_console_chat_ui()
        composer = console.query_one("#console-native-composer")
        composer.load_draft("hello")
        send = console.query_one("#console-send-message", Button)
        async with asyncio.timeout(3):
            while send.disabled:
                await asyncio.sleep(0.01)
        await pilot.pause(0.3)
        revision = expiring_credential.cache.revision
        status = console.query_one("#console-send-disabled-reason", Static)
        expiring_credential.wall = 1002
        await _wait_for_status(status, "log in with Claude Code")
        assert send.disabled
        assert expiring_credential.cache.revision == revision

        expiring_credential.release.clear()
        expiring_credential.expires_at_ms = 2000000
        expiring_credential.monotonic = subscription._KEYCHAIN_TTL_S + 1
        await _wait_for_status(status, "Checking Claude")
        assert send.disabled
        assert len(expiring_credential.readers) == 2
        assert all(
            reader != threading.get_ident() for reader in expiring_credential.readers
        )
        expiring_credential.release.set()
        async with asyncio.timeout(3):
            while send.disabled:
                await asyncio.sleep(0.01)
        assert "private-token" not in str(status.renderable)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["file", "keychain"])
async def test_streaming_control_responds_and_readiness_refreshes_after_slow_read(
    monkeypatch, tmp_path, source
):
    path = tmp_path / "credential.json"
    payload = json.dumps({"claudeAiOauth": {"accessToken": "private-token"}})
    if source == "file":
        path.write_text(payload)
    cache = subscription._SubscriptionReadinessCache()
    monkeypatch.setattr(subscription, "_SUBSCRIPTION_READINESS_CACHE", cache)
    monkeypatch.setattr(subscription, "_KEYCHAIN_CACHE", None)
    monkeypatch.setattr(subscription, "DEFAULT_CREDENTIALS_PATH", path)
    monkeypatch.setattr(subscription, "sys", SimpleNamespace(platform="darwin"))
    entered = threading.Event()
    release = threading.Event()
    readers = []
    original_read = Path.read_text

    def block_read():
        readers.append(threading.get_ident())
        entered.set()
        assert release.wait(5)

    def gated_file(target, *args, **kwargs):
        if target == path and source == "file":
            block_read()
        return original_read(target, *args, **kwargs)

    def security(command, **kwargs):
        assert command[0] == "/usr/bin/security"
        assert kwargs["timeout"] <= 5
        if source == "keychain":
            block_read()
        return SimpleNamespace(returncode=0, stdout=payload)

    monkeypatch.setattr(Path, "read_text", gated_file)
    monkeypatch.setattr(subscription.subprocess, "run", security)
    app = ModalHarness()
    app.app_config = {
        "api_settings": {"anthropic": {"auth_source": "claude_subscription"}}
    }
    modal = ConsoleSettingsModal(
        settings=ConsoleSessionSettings(
            provider="anthropic", model="claude-sonnet-4-5"
        ),
        app_config=app.app_config,
        providers_models={"anthropic": ["claude-sonnet-4-5"]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
    )
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            assert entered.is_set()
            assert readers == [readers[0]]
            assert readers[0] != threading.get_ident()
            initial = modal._build_draft().streaming
            modal.query_one("#console-settings-streaming", Button).press()
            await pilot.pause()
            assert modal._build_draft().streaming is not initial
            assert not release.is_set()
            status = modal.query_one("#console-settings-readiness", Static)
            assert "Checking Claude" in str(status.renderable)
            assert not modal.query_one(
                "#console-settings-configure-credential", Button
            ).display
            release.set()
            # No user action after release: completion must reach the UI itself.
            async with asyncio.timeout(3):
                while "Ready to send" not in str(status.renderable):
                    await asyncio.sleep(0.01)
            assert "private-token" not in str(status.renderable)
            assert len(readers) == 1
    finally:
        release.set()
        if cache._worker is not None:
            cache._worker.join(timeout=3)
            assert not cache._worker.is_alive()


@pytest.mark.asyncio
@private_profile_test
async def test_console_send_unblocks_when_background_credential_read_finishes(
    monkeypatch, tmp_path, request
):
    path = tmp_path / "credential.json"
    path.write_text(json.dumps({"claudeAiOauth": {"accessToken": "private-token"}}))
    cache = subscription._SubscriptionReadinessCache()
    monkeypatch.setattr(subscription, "_SUBSCRIPTION_READINESS_CACHE", cache)
    monkeypatch.setattr(subscription, "DEFAULT_CREDENTIALS_PATH", path)
    monkeypatch.setattr(subscription, "sys", SimpleNamespace(platform="linux"))
    release = threading.Event()
    entered = threading.Event()
    original_read = Path.read_text

    def gated_read(target, *args, **kwargs):
        if target == path:
            entered.set()
            assert release.wait(5)
        return original_read(target, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", gated_read)
    app = _console_app()
    app.app_config["api_settings"]["anthropic"] = {"auth_source": "claude_subscription"}
    host = _ConsoleFlowHarness(app)
    try:
        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen
            console._provider_readiness_app_config = lambda: app.app_config
            await _wait_for_selector(console, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            store.replace_session_settings(
                store.active_session_id,
                ConsoleSessionSettings(provider="anthropic", model="claude-sonnet-4-5"),
            )
            await console._sync_native_console_chat_ui()
            composer = console.query_one("#console-native-composer")
            composer.load_draft("hello")
            await pilot.pause()
            assert entered.is_set()
            send = console.query_one("#console-send-message", Button)
            assert send.disabled
            assert "Checking Claude" in str(
                console.query_one("#console-send-disabled-reason").renderable
            )
            release.set()
            async with asyncio.timeout(3):
                while send.disabled:
                    await asyncio.sleep(0.01)
            assert console._console_send_blocked_reason() == ""
        assert console._console_credential_poll_timer is None
    finally:
        release.set()
        if cache._worker is not None:
            cache._worker.join(timeout=3)
            assert not cache._worker.is_alive()
