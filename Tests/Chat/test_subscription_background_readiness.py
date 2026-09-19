"""UI snapshots never read credentials inline or change send authentication."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription

CONFIG = {"api_settings": {"anthropic": {"auth_source": "claude_subscription"}}}


def snapshot():
    return get_provider_readiness("Anthropic", CONFIG, background_credentials=True)


@pytest.fixture
def credential_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subscription, "DEFAULT_CREDENTIALS_PATH", tmp_path / "credential.json"
    )
    # Never fall back to the host Keychain, including a failing test's teardown.
    monkeypatch.setattr(subscription, "sys", SimpleNamespace(platform="linux"))
    cache = subscription._SubscriptionReadinessCache()
    monkeypatch.setattr(subscription, "_SUBSCRIPTION_READINESS_CACHE", cache)
    yield cache
    worker = cache._worker
    if worker is not None:
        worker.join(timeout=3)
        assert not worker.is_alive()


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_ui_snapshot_does_not_block_on_file_io_and_refreshes(
    monkeypatch, tmp_path
):
    """The first check must return while the file read is still gated."""
    # Keep this test independent of the new cache class to reproduce the public defect.
    path = tmp_path / "credential.json"
    path.write_text(json.dumps({"claudeAiOauth": {"accessToken": "private-token"}}))
    monkeypatch.setattr(subscription, "DEFAULT_CREDENTIALS_PATH", path)
    monkeypatch.setattr(subscription, "sys", SimpleNamespace(platform="linux"))
    entered = threading.Event()
    release = threading.Event()
    read_threads = []
    original_read = Path.read_text

    def gated_read(target, *args, **kwargs):
        if target == path:
            read_threads.append(threading.get_ident())
            entered.set()
            assert release.wait(3)
        return original_read(target, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", gated_read)
    try:
        pending = snapshot()
        await wait_until(entered.is_set)
        assert not pending.ready
        assert "Checking" in pending.reason
        assert "private-token" not in repr(pending)
        assert read_threads == [read_threads[0]]
        assert read_threads[0] != threading.get_ident()
        # Multiple UI updates must not create more blocked file readers.
        for _ in range(20):
            assert not snapshot().ready
        heartbeat = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(heartbeat.set_result, True)
        assert await heartbeat
        assert not release.is_set()
    finally:
        release.set()
    await wait_until(lambda: snapshot().ready)
    assert len(read_threads) == 1
    assert snapshot().api_key is None


@pytest.mark.asyncio
async def test_cache_ttl_begins_after_slow_missing_read(credential_cache, monkeypatch):
    now = [100.0]
    attempts = []
    monkeypatch.setattr(
        subscription,
        "time",
        SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[0]),
    )

    def missing():
        attempts.append(True)
        now[0] += 5.0

    monkeypatch.setattr(subscription, "read_claude_code_credential", missing)
    snapshot()
    await wait_until(lambda: "No Claude" in snapshot().reason)
    now[0] += 4.9
    assert "No Claude" in snapshot().reason
    assert len(attempts) == 1
    now[0] += 0.2
    snapshot()
    await wait_until(lambda: len(attempts) == 2)


@pytest.mark.asyncio
async def test_readiness_detects_expiry_within_cache_ttl(credential_cache, monkeypatch):
    wall_time = [100.0]
    monkeypatch.setattr(
        subscription,
        "time",
        SimpleNamespace(monotonic=lambda: 0.0, time=lambda: wall_time[0]),
    )
    subscription.DEFAULT_CREDENTIALS_PATH.write_text(
        json.dumps(
            {"claudeAiOauth": {"accessToken": "private-token", "expiresAt": 101_000}}
        )
    )
    snapshot()
    await wait_until(lambda: snapshot().ready)
    wall_time[0] = 102.0
    expired = snapshot()
    assert not expired.ready
    assert "expired" in expired.reason


@pytest.mark.asyncio
async def test_reader_error_is_secret_free_and_cached(credential_cache, monkeypatch):
    attempts = []

    def broken():
        attempts.append(True)
        raise OSError("private-token-secret-in-error")

    monkeypatch.setattr(subscription, "read_claude_code_credential", broken)
    revision = subscription.subscription_readiness_revision()
    snapshot()
    await wait_until(lambda: subscription.subscription_readiness_revision() > revision)
    result = snapshot()
    assert not result.ready
    assert "private-token" not in repr(result)
    assert "private-token" not in result.user_message
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_sync_send_readiness_resolves_credentials_while_ui_cache_pending(
    credential_cache,
):
    subscription.DEFAULT_CREDENTIALS_PATH.write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "private-token"}})
    )
    # A send can arrive before any UI snapshot was populated.
    readiness = get_provider_readiness("Anthropic", CONFIG)
    assert readiness.ready
    assert readiness.api_key_source == "subscription:claude_code"
    assert readiness.api_key is None
    assert subscription.read_claude_code_credential().access_token == "private-token"


@pytest.mark.asyncio
async def test_async_send_waits_for_real_credentials_without_blocking_loop(
    credential_cache, monkeypatch
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    path = subscription.DEFAULT_CREDENTIALS_PATH
    path.write_text(json.dumps({"claudeAiOauth": {"accessToken": "private-token"}}))
    entered = threading.Event()
    release = threading.Event()
    read_threads = []
    original_read = Path.read_text

    def gated_read(target, *args, **kwargs):
        if target == path:
            read_threads.append(threading.get_ident())
            entered.set()
            release.wait(3)
        return original_read(target, *args, **kwargs)

    def unexpected_network(request):
        raise AssertionError(f"Unexpected network route: {request.url.path}")

    monkeypatch.setattr(Path, "read_text", gated_read)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(unexpected_network)
    ) as client:
        gateway = ConsoleProviderGateway(
            http_client=client, config_provider=lambda: CONFIG, environ={}
        )
        operation = asyncio.create_task(
            gateway.resolve_for_send(
                ConsoleProviderSelection(
                    provider="anthropic", explicit_model="claude-sonnet-4-5"
                )
            )
        )
        try:
            await wait_until(entered.is_set)
            assert not operation.done(), (
                "The event loop must run while the credential read waits"
            )
            assert read_threads[0] != threading.get_ident()
        finally:
            release.set()
            resolution = await operation
        assert resolution.ready
        assert resolution.api_key is None
        assert resolution.api_key_source == "subscription:claude_code"


@pytest.mark.parametrize("state", ["ready", "expired", "missing"])
def test_subscription_summary_names_the_actual_credential_source(
    credential_cache, state
):
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSessionSettings,
        ConsoleSettingsContextEstimate,
        build_console_settings_readiness,
        build_console_settings_summary_state,
    )
    from tldw_chatbook.Widgets.Console.console_settings_summary import (
        build_console_readiness_presentation,
    )

    if state != "missing":
        subscription.DEFAULT_CREDENTIALS_PATH.write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "private-token",
                        "expiresAt": 1 if state == "expired" else 0,
                    }
                }
            )
        )
    settings = ConsoleSessionSettings(provider="anthropic", model="claude-sonnet-4-5")
    readiness = build_console_settings_readiness(
        settings, app_config=CONFIG, background_credentials=False
    )
    presentation = build_console_readiness_presentation(readiness)
    summary = build_console_settings_summary_state(
        settings,
        readiness=readiness,
        context_estimate=ConsoleSettingsContextEstimate(0, 200000, "0 / 200k"),
    )
    assert "Claude subscription" in summary.credential_row
    assert "local config" not in summary.credential_row
    assert "API key" not in presentation.primary_label
    assert "private-token" not in repr(summary)
    if state != "ready":
        assert "Claude Code" in presentation.primary_label
