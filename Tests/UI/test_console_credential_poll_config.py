"""Credential polling must not wait on another thread's native config locks."""

import asyncio
import threading

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.console_fixture_ownership import owned_console_apps  # noqa: F401
from Tests.UI.test_console_mcp_approval import _wait_for_production_console_ready
from tldw_chatbook import config
from tldw_chatbook.Chat.console_session_settings import console_send_connection
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderProbeResult,
    ProviderTestEvidenceStore,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _ready_app():
    app = _build_test_app(
        configured_default="chat",
        config_overrides={"splash_screen": {"enabled": False}},
    )
    assert config.save_settings_to_cli_config(
        {
            # load_settings(force_reload=True) below drops the factory's
            # override, so the splash must be off on disk too.
            "splash_screen": {"enabled": False},
            "first_run": {"setup_completed": True},
            "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
            "api_settings.llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "local-model",
            },
        }
    )
    app.app_config = config.load_settings(force_reload=True)
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    return app


@pytest.mark.parametrize("lock_name", ["_settings_rebuild_lock", "_config_file_lock"])
async def test_credential_poll_defers_native_config_contention(lock_name):
    """A blocking getter or lost trailing refresh fails on the mounted owner."""
    app = _ready_app()
    async with app.run_test(size=(120, 50)) as pilot:
        screen = await _wait_for_production_console_ready(app, pilot)
        screen._stop_console_credential_poll_timer()
        settings, _ = screen._active_console_settings_readiness()
        identity = console_send_connection(
            settings, app_config=screen._provider_readiness_app_config()
        )
        evidence = ProviderTestEvidenceStore(lambda: app)
        evidence.settle(
            evidence.begin(identity),
            ProviderProbeResult("unreachable", (), "timeout"),
        )
        # A changed gate must not be consumed until the real projection can run.
        screen._console_credential_snapshot = None
        screen._console_credential_pre_gate = None
        held = threading.Event()
        release = threading.Event()
        expired = threading.Event()

        def hold():
            with getattr(config, lock_name)():
                held.set()
                if not release.wait(2):
                    expired.set()

        owner = threading.Thread(target=hold)
        owner.start()
        try:
            assert held.wait(3)
            screen._poll_console_credential_readiness()
            assert not expired.is_set(), "Credential poll waited on native config"
            assert owner.is_alive() and not release.is_set()
            assert screen._console_credential_snapshot is None
            assert screen._console_credential_pre_gate is None
            for _ in range(3):
                screen._poll_console_credential_readiness()
                assert screen._console_credential_snapshot is None
            assert not config._settings_rebuild_lock()._is_owned()
            assert not config._config_file_lock()._is_owned()
        finally:
            release.set()
            owner.join(3)
            assert not owner.is_alive()
        # Restore the native 250 ms poll after the controlled lock hold.
        screen._console_credential_poll_timer = screen.set_interval(
            0.25, screen._poll_console_credential_readiness
        )
        async with asyncio.timeout(10):
            # TASK-42475: the screen's own pending-work signal, not a copied list.
            while (
                screen._console_credential_snapshot is None
                or screen.console_pending_startup_work()
            ):
                await asyncio.sleep(0.01)
        assert "Not ready · timed out" in str(
            screen.query_one("#console-settings-readiness-row").render()
        )
        assert screen.query_one("#console-send-message").disabled
        assert not screen._console_control_bar_sync_scheduled
        screen._poll_console_credential_readiness()
        assert screen._console_credential_snapshot is not None


async def test_credential_poll_builds_no_workspace_context(monkeypatch):
    """A completed credential poll re-derives the send controls with no context build.

    Formerly ``test_credential_poll_builds_current_workspace_once`` (dev
    95ca4bf47b): the poll's settings-summary and control-bar syncs each built
    the workspace context, and the poll's live build scope made them share
    ONE build instead of two. TASK-36571.1: neither sync builds it any more
    (the Agent section reads its open flag; the rail and the Inspector read
    the workspace label), so the poll builds none, and its scope still
    closes behind it.
    """
    from tldw_chatbook.UI.Console_Modules import workspace

    app = _ready_app()
    async with app.run_test(size=(120, 50)) as pilot:
        screen = await _wait_for_production_console_ready(app, pilot)
        screen._stop_console_credential_poll_timer()
        built = {"workspace state": 0, "context": 0}
        observing = False
        original_state = workspace.build_console_workspace_state
        controller = screen._workspace
        # Every context build runs the browser pipeline once, memo hit or not.
        original_context = controller._with_console_conversation_browser_state

        def build_state(*args, **kwargs):
            if observing:
                built["workspace state"] += 1
            return original_state(*args, **kwargs)

        def build_context(*args, **kwargs):
            if observing:
                built["context"] += 1
            return original_context(*args, **kwargs)

        monkeypatch.setattr(workspace, "build_console_workspace_state", build_state)
        monkeypatch.setattr(
            controller, "_with_console_conversation_browser_state", build_context
        )
        screen._console_credential_snapshot = None
        async with asyncio.timeout(10):
            # A changed snapshot runs both syncs. Retry until a pass completes
            # them: lock contention, a cold read or a deferred control bar
            # leaves the snapshot unset. Every attempt is counted.
            while screen._console_credential_snapshot is None:
                screen._console_credential_pre_gate = None
                observing = True
                try:
                    screen._poll_console_credential_readiness()
                finally:
                    observing = False
                if screen._console_credential_snapshot is None:
                    await asyncio.sleep(0.01)
        assert built == {"workspace state": 0, "context": 0}, (
            f"a credential poll built {built}"
        )
        assert screen._workspace._console_tick_builds is None
