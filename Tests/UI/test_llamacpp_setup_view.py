from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Input, Select

from tldw_chatbook.LLM_Management.llamacpp_connection import LlamaCppProbeResult
from tldw_chatbook.LLM_Management.llamacpp_profiles import LlamaCppProfileRepository
from tldw_chatbook.UI.LLM_Management.llamacpp_setup_view import LlamaCppSetupView
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)


class Harness(App):
    def __init__(self, path):
        super().__init__()
        self.pending_handoffs = PendingHandoffStore()
        self.path = path

    def compose(self) -> ComposeResult:
        yield LlamaCppSetupView(self, repository=LlamaCppProfileRepository(self.path))


@pytest.mark.asyncio
async def test_profile_save_select_reload_and_delete(tmp_path):
    app = Harness(tmp_path / "profiles.json")
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(LlamaCppSetupView)
        view.query_one(Collapsible).collapsed = False
        await pilot.pause()
        view.query_one("#llamacpp-context-size", Input).focus()
        await pilot.press("4", "0", "9", "6")
        view.query_one("#llamacpp-profile-name", Input).focus()
        await pilot.press("L", "a", "p", "t", "o", "p")
        view.query_one("#llamacpp-profile-save", Button).focus()
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.02)
            if view._profiles.profiles:
                break
        assert len(view._profiles.profiles) == 1
        assert (
            app._llamacpp_lab_draft["profile_id"]
            == view._profiles.profiles[0].profile_id
        )
        view.query_one("#llamacpp-profile-select", Select).value = "defaults"
        await pilot.pause()
        assert view.tuning().context_size is None
        profile = view._profiles.profiles[0]
        view.query_one("#llamacpp-profile-select", Select).value = profile.profile_id
        await pilot.pause()
        assert view.tuning().context_size == 4096
        await view.profile_action("reload")
        await view.profile_action("delete")
        assert not view._profiles.profiles


@pytest.mark.asyncio
async def test_verified_check_and_stale_target_handoff(tmp_path, monkeypatch):
    from tldw_chatbook.UI.LLM_Management import llamacpp_setup_view as module

    async def probe(request, **kwargs):
        return LlamaCppProbeResult(request, "ready", ("org/model",), "org/model")

    monkeypatch.setattr(module, "probe_llamacpp_target", probe)
    app = Harness(tmp_path / "profiles.json")
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(LlamaCppSetupView)
        view.query_one("#llamacpp-existing-url", Input).value = "http://127.0.0.1:8181"
        await pilot.pause()
        await view.check_connection()
        assert not view.query_one("#llamacpp-use-console", Button).disabled
        assert view.stage_handoff(default=False)
        assert app.pending_handoffs.has_pending(HandoffChannel.LLAMACPP_CONSOLE)
        view.query_one("#llamacpp-existing-url", Input).value = "http://127.0.0.1:9191"
        await pilot.pause()
        assert view.query_one("#llamacpp-use-console", Button).disabled
        assert not view.stage_handoff(default=True)


@pytest.mark.asyncio
async def test_remount_rebinds_exact_owned_claim_and_death_invalidates(
    tmp_path, monkeypatch
):
    import threading
    from types import SimpleNamespace

    from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
        reserve_server_launch,
    )
    from tldw_chatbook.UI.LLM_Management import llamacpp_setup_view as module

    app = Harness(tmp_path / "profiles.json")
    app._llm_server_lifecycle_lock = threading.RLock()
    app._llm_server_launch_claims = {}
    process = SimpleNamespace(poll=lambda: None)
    app.llamacpp_server_process = None
    claim = reserve_server_launch(app, "llamacpp")
    claim._connection_url = "http://127.0.0.1:8181"
    app.llamacpp_server_process = process
    app._llamacpp_lab_draft = {"endpoint": claim._connection_url}

    async def probe(request, **kwargs):
        return LlamaCppProbeResult(
            request, "ready", ("chatbook-llamacpp",), "chatbook-llamacpp"
        )

    monkeypatch.setattr(module, "probe_llamacpp_target", probe)
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(LlamaCppSetupView)
        await view.check_connection()
        assert view.owner.snapshot().target.runtime_owner == "lab_process"
        process.poll = lambda: 1
        assert view.owner.snapshot().target is None


@pytest.mark.asyncio
async def test_invalid_endpoint_never_enters_app_draft(tmp_path):
    app = Harness(tmp_path / "profiles.json")
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(LlamaCppSetupView)
        view.query_one(
            "#llamacpp-existing-url", Input
        ).value = "http://user:PRIVATE@127.0.0.1:8181/?token=SECRET"
        await pilot.pause()
        await view.check_connection()
        assert "PRIVATE" not in repr(app._llamacpp_lab_draft)
        assert "SECRET" not in repr(app._llamacpp_lab_draft)


@pytest.mark.asyncio
async def test_initial_select_event_does_not_erase_retained_draft(tmp_path):
    app = Harness(tmp_path / "profiles.json")
    app._llamacpp_lab_draft = {"context_size": "4096", "name": "Laptop draft"}
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(LlamaCppSetupView)
        assert view.tuning().context_size == 4096
        assert view.query_one("#llamacpp-profile-name", Input).value == "Laptop draft"


@pytest.mark.asyncio
async def test_real_models_pane_preserves_sources_snapshots_and_navigates_to_console(
    monkeypatch,
):
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.config import get_cli_setting
    from tldw_chatbook.Constants import TAB_LLM
    from tldw_chatbook.UI.LLM_Management import llamacpp_setup_view as module
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if (section, key) == ("splash_screen", "enabled")
            else get_cli_setting(section, key, default)
        ),
    )

    async def probe(request, **kwargs):
        return LlamaCppProbeResult(request, "ready", ("org/model",), "org/model")

    monkeypatch.setattr(module, "probe_llamacpp_target", probe)
    app = _build_test_app()
    async with app.run_test(size=(160, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if getattr(app, "_initial_screen_pushed", False):
                break
        await app.handle_screen_navigation(NavigateToScreen(TAB_LLM))
        for _ in range(100):
            await pilot.pause(0.02)
            if (
                list(app.screen.query(LlamaCppSetupView))
                and list(app.screen.query("#llamacpp-gguf-source-mode"))
                and list(app.screen.query("#llamacpp-snapshot-manager"))
                and not app.screen.query_one(LlamaCppSetupView)._hydrating
            ):
                break
        view = app.screen.query_one(LlamaCppSetupView)
        assert app.screen.query_one("#llamacpp-gguf-source-mode", Select)
        assert app.screen.query_one("#llamacpp-snapshot-manager")
        view.query_one("#llamacpp-existing-url", Input).value = "http://127.0.0.1:8181"
        await pilot.pause()
        view.query_one("#llamacpp-check", Button).press()
        for _ in range(30):
            await pilot.pause(0.02)
            if view.owner.snapshot().target is not None:
                break
        assert view.owner.snapshot().target is not None
        view.query_one("#llamacpp-use-console", Button).press()
        for _ in range(100):
            await pilot.pause(0.02)
            if isinstance(
                app.screen, ChatScreen
            ) and not app.pending_handoffs.has_pending(HandoffChannel.LLAMACPP_CONSOLE):
                break
        assert isinstance(app.screen, ChatScreen)
        store = app.screen._ensure_console_chat_store()
        settings = store.effective_session_settings(store.active_session_id)
        assert (settings.provider, settings.model, settings.base_url) == (
            "llama_cpp",
            "org/model",
            "http://127.0.0.1:8181",
        )


@pytest.mark.asyncio
async def test_navigation_failure_discards_exact_staged_intent(tmp_path, monkeypatch):
    from tldw_chatbook.UI.LLM_Management import llamacpp_setup_view as module

    async def probe(request, **kwargs):
        return LlamaCppProbeResult(request, "ready", ("model",), "model")

    monkeypatch.setattr(module, "probe_llamacpp_target", probe)
    app = Harness(tmp_path / "profiles.json")
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(LlamaCppSetupView)
        view.query_one("#llamacpp-existing-url", Input).value = "http://127.0.0.1:8181"
        await pilot.pause()
        await view.check_connection()

        def fail(_message):
            raise RuntimeError("navigation unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(view, "post_message", fail)
            assert not view.stage_handoff(default=True)
        assert not app.pending_handoffs.has_pending(HandoffChannel.LLAMACPP_DEFAULT)


def test_credentials_are_exact_endpoint_scoped_and_malformed_config_is_ignored(
    tmp_path,
):
    app = Harness(tmp_path / "profiles.json")
    view = LlamaCppSetupView(app)
    app.app_config = {
        "api_settings": {
            "llama_cpp": {"api_url": "http://127.0.0.1:8181", "api_key": "PRIVATE"}
        }
    }
    assert view._credential("http://127.0.0.1:9191") is None
    assert view._credential("http://127.0.0.1:8181") == "PRIVATE"
    app.app_config = {"api_settings": {"llama_cpp": "malformed"}}
    assert view._credential("http://127.0.0.1:8181") is None
