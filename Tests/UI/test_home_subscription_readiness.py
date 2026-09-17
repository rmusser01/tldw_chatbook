"""Home publishes completed subscription readiness without blocking its UI."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_home_screen import HOME_TEST_SIZE, HomeHarness
from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription


@pytest.fixture
def credential_io(monkeypatch, tmp_path):
    path = tmp_path / "credential.json"
    cache = subscription._SubscriptionReadinessCache()
    entered, release = threading.Event(), threading.Event()
    readers = []
    original_read = Path.read_text

    def gated_read(target, *args, **kwargs):
        if target == path:
            readers.append(threading.get_ident())
            entered.set()
            release.wait(5)
        return original_read(target, *args, **kwargs)

    monkeypatch.setattr(subscription, "_SUBSCRIPTION_READINESS_CACHE", cache)
    monkeypatch.setattr(subscription, "DEFAULT_CREDENTIALS_PATH", path)
    monkeypatch.setattr(subscription, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(Path, "read_text", gated_read)
    yield SimpleNamespace(
        path=path, cache=cache, entered=entered, release=release, readers=readers
    )
    release.set()
    if cache._worker is not None:
        cache._worker.join(timeout=3)
        assert not cache._worker.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("credential_state", ["ready", "missing", "expired"])
@private_profile_test
async def test_home_subscription_completion_updates_badge_without_blocking(
    credential_io, credential_state, request
):
    """A pending first paint must not become the visit's final readiness."""
    if credential_state != "missing":
        credential_io.path.write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "private-home-token",
                        "expiresAt": 1 if credential_state == "expired" else 0,
                    }
                }
            )
        )
    app = _build_test_app(
        config_overrides={
            "chat_defaults": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
            },
            "api_settings": {"anthropic": {"auth_source": "claude_subscription"}},
        }
    )
    host = HomeHarness(app)
    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause()
        home = host.screen
        assert credential_io.entered.is_set()
        assert "Model: Blocked" in str(home.query_one("#home-details-body").renderable)
        assert home._current_dashboard.next_action.action_id == "fix_model_setup"
        heartbeats = []
        timer = home.set_interval(0.01, lambda: heartbeats.append(None))
        async with asyncio.timeout(1):
            while len(heartbeats) < 3:
                await asyncio.sleep(0.01)
        timer.stop()
        assert all(reader != threading.get_ident() for reader in credential_io.readers)
        assert not credential_io.release.is_set()

        credential_io.release.set()
        async with asyncio.timeout(3):
            while credential_io.cache.revision == 0:
                await asyncio.sleep(0.01)
            await host.workers.wait_for_complete()
        await pilot.pause()

        status = str(home.query_one("#home-details-body").renderable)
        expected_badge = (
            "Model: Ready" if credential_state == "ready" else "Model: Blocked"
        )
        assert expected_badge in status
        assert home._home_content_snapshot.console_ready is (
            credential_state == "ready"
        )
        assert (home._current_dashboard.next_action.action_id == "fix_model_setup") is (
            credential_state != "ready"
        )
        assert "private-home-token" not in status
        assert str(credential_io.path) not in status
