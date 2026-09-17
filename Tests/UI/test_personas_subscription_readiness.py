"""Roleplay handoff readiness never waits for credential storage on the UI thread."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_personas_workbench import (
    PersonasScreen,
    PersonasTestApp,
    _install_conversation_db,
    _mounted,
    character_handler_module,
    stub_characters,  # noqa: F401 - shared fixture
)
from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription

pytestmark = pytest.mark.usefixtures("stub_characters")


@pytest.fixture
def handoff_app(mock_app_instance, monkeypatch):
    monkeypatch.setattr(
        character_handler_module, "_default_character_db", lambda: object()
    )
    _install_conversation_db(monkeypatch, [])
    mock_app_instance.chat_dictionary_scope_service = None
    mock_app_instance.app_config = {
        "chat_defaults": {"provider": "anthropic", "model": "claude-sonnet-4-5"},
        "api_settings": {"anthropic": {"auth_source": "claude_subscription"}},
    }
    return PersonasTestApp(mock_app_instance)


@pytest.fixture
def credential_io(monkeypatch, tmp_path):
    path = tmp_path / "credential.json"
    cache = subscription._SubscriptionReadinessCache()
    state = SimpleNamespace(
        path=path,
        cache=cache,
        release=threading.Event(),
        entered=threading.Event(),
        readers=[],
        source="file",
        outcome="ready",
        expires_at_ms=0,
        wall=1000.0,
        monotonic=0.0,
    )
    path.write_text("{}")
    monkeypatch.setattr(subscription, "_SUBSCRIPTION_READINESS_CACHE", cache)
    monkeypatch.setattr(subscription, "_KEYCHAIN_CACHE", None)
    monkeypatch.setattr(subscription, "DEFAULT_CREDENTIALS_PATH", path)
    monkeypatch.setattr(subscription, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(
        subscription,
        "time",
        SimpleNamespace(time=lambda: state.wall, monotonic=lambda: state.monotonic),
    )
    original_read = Path.read_text

    def read_payload():
        state.readers.append(threading.get_ident())
        state.entered.set()
        # Bounded even against the original synchronous bug, so red cannot hang.
        state.release.wait(3)
        if state.outcome == "error":
            raise OSError("private-token must never appear in status")
        if state.outcome == "missing":
            return "{}"
        return json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "private-token",
                    "expiresAt": state.expires_at_ms,
                }
            }
        )

    def gated_read(target, *args, **kwargs):
        if target == path:
            if state.source == "keychain":
                return "{}"
            return read_payload()
        return original_read(target, *args, **kwargs)

    def security(command, **kwargs):
        assert command[0] == "/usr/bin/security"
        assert kwargs["timeout"] <= 5
        return SimpleNamespace(
            returncode=0 if state.source == "keychain" else 1,
            stdout=read_payload() if state.source == "keychain" else "",
        )

    monkeypatch.setattr(Path, "read_text", gated_read)
    monkeypatch.setattr(subscription.subprocess, "run", security)
    yield state
    state.release.set()
    if cache._worker is not None:
        cache._worker.join(timeout=3)
        assert not cache._worker.is_alive()


def _readiness_text(screen):
    return str(screen.query_one("#personas-readiness-console", Static).renderable)


async def _wait_for_text(screen, expected):
    async with asyncio.timeout(3):
        while expected not in _readiness_text(screen):
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "outcome"),
    [("file", "ready"), ("keychain", "ready"), ("file", "missing"), ("file", "error")],
)
async def test_handoff_readiness_keeps_ui_responsive_and_finishes_automatically(
    handoff_app, credential_io, source, outcome, monkeypatch
):
    credential_io.source = source
    credential_io.outcome = outcome
    original_poll = PersonasScreen._poll_console_handoff_readiness
    allow_poll = False

    def delayed_first_poll(screen):
        if allow_poll:
            original_poll(screen)

    # Force completion before the first interval check: the comparison must
    # track the painted pending status, even when no poll has run yet.
    monkeypatch.setattr(
        PersonasScreen, "_poll_console_handoff_readiness", delayed_first_poll
    )
    async with handoff_app.run_test(size=(160, 50)) as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert credential_io.entered.is_set()
        assert credential_io.readers[0] != threading.get_ident()
        assert credential_io.readers == [credential_io.readers[0]]
        assert "Checking Claude" in _readiness_text(screen)
        assert "No Claude" not in _readiness_text(screen)
        assert screen.query_one("#personas-start-chat", Button).disabled
        assert not screen.query_one("#personas-attach-to-console", Button).disabled

        heartbeat = asyncio.Event()
        handoff_app.set_timer(0.01, heartbeat.set)
        async with asyncio.timeout(1):
            await heartbeat.wait()
        assert not credential_io.release.is_set()
        credential_io.release.set()
        async with asyncio.timeout(3):
            while credential_io.cache.revision == 0:
                await asyncio.sleep(0.01)
        allow_poll = True
        expected = "Ready to chat" if outcome == "ready" else "No Claude subscription"
        await _wait_for_text(screen, expected)
        assert screen.query_one("#personas-start-chat", Button).disabled is (
            outcome != "ready"
        )
        assert "private-token" not in _readiness_text(screen)
        assert screen.preview.gateway is None
        assert len(credential_io.readers) == 1


@pytest.mark.asyncio
async def test_completion_between_header_and_inspector_reads_refreshes_both_surfaces(
    handoff_app, credential_io, monkeypatch
):
    credential_io.release.set()
    async with handoff_app.run_test(size=(160, 50)) as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await _wait_for_text(screen, "Ready to chat")
        reads = []

        def completing_snapshot(*, background=False):
            assert background
            reads.append(True)
            return "pending" if len(reads) == 1 else "ready"

        monkeypatch.setattr(
            subscription, "subscription_credential_status", completing_snapshot
        )
        # Finish after the header read but before the inspector read in this
        # synchronous render. No further edit should be needed to reconcile it.
        screen._sync_title_and_console_actions()
        header_status = screen.query_one(
            "#personas-header #workbench-header-status", Static
        )
        assert len(reads) >= 2
        async with asyncio.timeout(3):
            while str(header_status.renderable) != "Ready":
                await asyncio.sleep(0.01)
        assert "Ready to chat" in _readiness_text(screen)
        assert not screen.query_one("#personas-start-chat", Button).disabled


@pytest.mark.asyncio
async def test_handoff_readiness_refreshes_expiry_and_stale_snapshot_without_edits(
    handoff_app, credential_io
):
    credential_io.expires_at_ms = 1001000
    credential_io.release.set()
    async with handoff_app.run_test(size=(160, 50)) as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await _wait_for_text(screen, "Ready to chat")
        completed_revision = credential_io.cache.revision

        # Expiry occurs within the snapshot TTL, without another completed read.
        credential_io.wall = 1002
        await _wait_for_text(screen, "credential is expired")
        assert screen.query_one("#personas-start-chat", Button).disabled
        assert credential_io.cache.revision == completed_revision

        # An idle mounted screen starts and observes the next bounded refresh.
        credential_io.release.clear()
        credential_io.entered.clear()
        credential_io.expires_at_ms = 2000000
        credential_io.monotonic = subscription._KEYCHAIN_TTL_S + 1
        await _wait_for_text(screen, "Checking Claude")
        assert credential_io.entered.is_set()
        assert len(credential_io.readers) == 2
        credential_io.release.set()
        await _wait_for_text(screen, "Ready to chat")
        assert not screen.query_one("#personas-start-chat", Button).disabled


@pytest.mark.asyncio
async def test_credential_completion_uses_current_provider_and_selection(
    handoff_app, credential_io
):
    async with handoff_app.run_test(size=(160, 50)) as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await _wait_for_text(screen, "Checking Claude")
        handoff_app.app_config["chat_defaults"] = {
            "provider": "openai",
            "model": "gpt-4o",
        }
        await pilot.click("#personas-library-row-character-2")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.state.selected_entity_id == "2"
        current_text = _readiness_text(screen)
        assert "openai" in current_text.lower()
        assert "Claude" not in current_text
        credential_io.release.set()
        async with asyncio.timeout(3):
            while credential_io.cache.revision == 0:
                await asyncio.sleep(0.01)
        await pilot.pause(0.3)
        assert _readiness_text(screen) == current_text
        assert screen.state.selected_entity_id == "2"
        assert screen.query_one("#personas-start-chat", Button).disabled


@pytest.mark.asyncio
async def test_unmount_stops_handoff_refresh_before_credential_completion(
    handoff_app, credential_io, monkeypatch
):
    async with handoff_app.run_test(size=(160, 50)) as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await _wait_for_text(screen, "Checking Claude")
        await handoff_app.pop_screen()
        await pilot.pause()
        assert not screen.is_attached
        assert screen._console_readiness_poll_timer is None

        def forbidden_refresh():
            raise AssertionError("credential completion repainted an unmounted screen")

        monkeypatch.setattr(
            screen, "_sync_title_and_console_actions", forbidden_refresh
        )
        credential_io.release.set()
        async with asyncio.timeout(3):
            while credential_io.cache.revision == 0:
                await asyncio.sleep(0.01)
        await pilot.pause(0.3)
        screen._poll_console_handoff_readiness()
