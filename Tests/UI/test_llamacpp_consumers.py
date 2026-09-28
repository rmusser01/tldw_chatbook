"""Verified llama.cpp targets reach Console and Settings through bounded handoffs."""

import tomllib

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_console_provider_apply_defaults_flow import (
    _console_app,
    _ConsoleFlowHarness,
    _install_ready_vllm_target,
    _SettingsFlowHarness,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.config import get_cli_config_path
from tldw_chatbook.LLM_Management.llamacpp_connection import (
    LlamaCppConnectionOwner,
    LlamaCppProbeResult,
)
from tldw_chatbook.UI.Navigation.llamacpp_handoff import (
    LlamaCppConsoleIntent,
    LlamaCppDefaultIntent,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.UI.Navigation.vllm_handoff import VllmDefaultIntent
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId


def _ready_target(
    app,
    *,
    api_url: str = "http://127.0.0.1:8001",
    model_id: str = "served-model",
):
    owner = LlamaCppConnectionOwner()
    request = owner.begin(
        api_url,
        runtime_owner="external_server",
        model_id=model_id,
    )
    assert owner.accept(LlamaCppProbeResult(request, "ready", (model_id,), model_id))
    app._llamacpp_connection_owner = owner
    target = owner.snapshot().target
    assert target is not None
    return owner, target


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change_model", "change_endpoint"),
    ((False, False), (True, False), (False, True)),
)
async def test_settings_acknowledges_llamacpp_even_when_fields_are_already_saved(
    change_model: bool, change_endpoint: bool
):
    """A clean or partly clean prefill must settle its exact handoff."""

    app = _console_app()
    config_path = get_cli_config_path()
    before = config_path.read_bytes()
    async with _SettingsFlowHarness(app).run_test(size=(180, 50)) as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#settings-provider-save-result")
        loaded = screen._provider_loaded_setting_values()
        assert loaded["provider"] == "llama_cpp"
        model_id = "other-served-model" if change_model else str(loaded["model"])
        api_url = (
            "http://127.0.0.1:8001" if change_endpoint else str(loaded["endpoint"])
        )
        _, target = _ready_target(app, api_url=api_url, model_id=model_id)
        app.pending_handoffs.stage(
            HandoffChannel.LLAMACPP_DEFAULT,
            LlamaCppDefaultIntent.from_target(target),
        )
        assert screen._consume_pending_llamacpp_default_intent() is True
        await pilot.pause(0.2)
        assert not app.pending_handoffs.has_pending(HandoffChannel.LLAMACPP_DEFAULT)
        assert screen._vllm_default_claim is None
        values = screen._provider_setting_values_mapping()
        assert (values["provider"], values["model"], values["endpoint"]) == (
            "llama_cpp",
            model_id,
            api_url,
        )
        draft = screen._provider_draft()
        assert (draft is not None) == (change_model or change_endpoint)
        assert config_path.read_bytes() == before


@pytest.mark.asyncio
async def test_console_adopts_verified_llamacpp_for_active_session_only():
    app = _console_app()
    _, target = _ready_target(app)
    app.pending_handoffs.stage(
        HandoffChannel.LLAMACPP_CONSOLE, LlamaCppConsoleIntent.from_target(target)
    )
    async with _ConsoleFlowHarness(app).run_test(size=(120, 42)) as pilot:
        console = pilot.app.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await pilot.pause(0.3)
        settings = console._session._active_console_session_settings()
        assert settings is not None
        assert (settings.provider, settings.model) == ("llama_cpp", "served-model")
        assert settings.base_url != target.base_url
        store = console._ensure_console_chat_store()
        assert (
            store.effective_session_settings(store.active_session_id).base_url
            == target.base_url
        )
        assert (
            app.app_config["api_settings"]["llama_cpp"]["api_url"]
            == "http://127.0.0.1:9099"
        )
        assert not app.pending_handoffs.has_pending(HandoffChannel.LLAMACPP_CONSOLE)


@pytest.mark.asyncio
async def test_settings_stages_verified_llamacpp_without_persisting():
    app = _console_app()
    _, target = _ready_target(app)
    config_path = get_cli_config_path()
    before = config_path.read_bytes()
    app.pending_handoffs.stage(
        HandoffChannel.LLAMACPP_DEFAULT, LlamaCppDefaultIntent.from_target(target)
    )
    async with _SettingsFlowHarness(app).run_test(size=(180, 50)) as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#settings-provider-save-result")
        await pilot.pause(0.3)
        draft = screen._provider_draft()
        assert draft is not None
        assert screen._provider_setting_values_mapping()["provider"] == "llama_cpp"
        assert draft.values["model"] == "served-model"
        assert draft.values["endpoint"] == target.base_url
        assert (
            screen.query_one("#settings-provider-endpoint-value", Input).value
            == target.base_url
        )
        assert (
            app.app_config["api_settings"]["llama_cpp"]["api_url"]
            == "http://127.0.0.1:9099"
        )
        assert config_path.read_bytes() == before
        assert not app.pending_handoffs.has_pending(HandoffChannel.LLAMACPP_DEFAULT)
        screen._revert_category(SettingsCategoryId.PROVIDERS_MODELS)
        assert screen._provider_draft() is None
        assert config_path.read_bytes() == before


@pytest.mark.asyncio
async def test_llamacpp_default_persists_only_after_settings_save():
    app = _console_app()
    _, target = _ready_target(app)
    config_path = get_cli_config_path()
    before = config_path.read_bytes()
    app.pending_handoffs.stage(
        HandoffChannel.LLAMACPP_DEFAULT, LlamaCppDefaultIntent.from_target(target)
    )
    async with _SettingsFlowHarness(app).run_test(size=(180, 50)) as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#settings-provider-save-result")
        await pilot.pause(0.3)
        assert config_path.read_bytes() == before
        assert screen.query_one("#settings-save-category", Button).disabled is False
        await pilot.click("#settings-save-category")
        await pilot.pause()
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["chat_defaults"]["provider"] == "llama_cpp"
    assert saved["chat_defaults"]["model"] == "served-model"
    assert saved["api_settings"]["llama_cpp"]["api_url"] == target.base_url


@pytest.mark.asyncio
async def test_stale_llamacpp_target_cannot_change_console_or_settings():
    app = _console_app()
    owner, target = _ready_target(app)
    app.pending_handoffs.stage(
        HandoffChannel.LLAMACPP_CONSOLE, LlamaCppConsoleIntent.from_target(target)
    )
    app.pending_handoffs.stage(
        HandoffChannel.LLAMACPP_DEFAULT, LlamaCppDefaultIntent.from_target(target)
    )
    owner.invalidate("target_changed")
    async with _ConsoleFlowHarness(app).run_test(size=(120, 42)) as pilot:
        console = pilot.app.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await pilot.pause(0.3)
        settings = console._session._active_console_session_settings()
        assert settings is not None and settings.model == "model-a"
        assert app.pending_handoffs.has_pending(HandoffChannel.LLAMACPP_CONSOLE)
    async with _SettingsFlowHarness(app).run_test(size=(180, 50)) as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#settings-provider-save-result")
        await pilot.pause(0.3)
        assert screen._provider_draft() is None
        assert app.pending_handoffs.has_pending(HandoffChannel.LLAMACPP_DEFAULT)


@pytest.mark.asyncio
async def test_llamacpp_settings_late_owner_expiry_compensates_before_ack(monkeypatch):
    app = _console_app()
    owner, target = _ready_target(app)
    async with _SettingsFlowHarness(app).run_test(size=(180, 50)) as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#settings-provider-save-result")
        deferred = []
        original = screen.call_after_refresh
        monkeypatch.setattr(
            screen,
            "call_after_refresh",
            lambda callback, *args: deferred.append((callback, args)),
        )
        app.pending_handoffs.stage(
            HandoffChannel.LLAMACPP_DEFAULT, LlamaCppDefaultIntent.from_target(target)
        )
        assert screen._consume_pending_llamacpp_default_intent() is True
        assert screen._vllm_default_claim is not None
        owner.invalidate("process_unavailable")
        callback, args = deferred[0]
        callback(*args)
        monkeypatch.setattr(screen, "call_after_refresh", original)
        assert screen._provider_draft() is None
        assert app.pending_handoffs.has_pending(HandoffChannel.LLAMACPP_DEFAULT)
        assert (
            screen.query_one("#settings-provider-endpoint-value", Input).value
            != target.base_url
        )


@pytest.mark.asyncio
async def test_llamacpp_console_adoption_failure_restores_session_and_requeues(
    monkeypatch,
):
    app = _console_app()
    _, target = _ready_target(app)
    async with _ConsoleFlowHarness(app).run_test(size=(120, 42)) as pilot:
        console = pilot.app.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        session_store = console._ensure_console_chat_store()
        session_id = session_store.active_session_id
        before = session_store.session_settings(session_id)
        app.pending_handoffs.stage(
            HandoffChannel.LLAMACPP_CONSOLE, LlamaCppConsoleIntent.from_target(target)
        )
        monkeypatch.setattr(
            console,
            "_sync_console_chat_core_state",
            lambda: (_ for _ in ()).throw(
                RuntimeError("controlled post-adoption failure")
            ),
        )
        assert console.consume_pending_llamacpp_console_intent() is False
        assert session_store.session_settings(session_id) == before
        assert app.pending_handoffs.has_pending(HandoffChannel.LLAMACPP_CONSOLE)


@pytest.mark.asyncio
async def test_llamacpp_default_recovery_blocks_other_default_until_released(
    monkeypatch,
):
    app = _console_app()
    llama_owner, llama_target = _ready_target(app)
    _, vllm_target = _install_ready_vllm_target(app)
    async with _SettingsFlowHarness(app).run_test(size=(180, 50)) as pilot:
        screen = pilot.app.screen
        await _wait_for_selector(screen, pilot, "#settings-provider-save-result")
        original_release = app.pending_handoffs.release
        monkeypatch.setattr(app.pending_handoffs, "release", lambda claim: False)
        app.pending_handoffs.stage(
            HandoffChannel.LLAMACPP_DEFAULT,
            LlamaCppDefaultIntent.from_target(llama_target),
        )
        llama_owner.invalidate("target_changed")
        assert screen._consume_pending_llamacpp_default_intent() is False
        assert (
            app.pending_handoffs.release_recovery(HandoffChannel.LLAMACPP_DEFAULT)
            is not None
        )
        recovery_button = screen.query_one("#settings-vllm-handoff-recovery", Button)
        assert recovery_button.display is True
        assert str(recovery_button.label) == "Retry provider handoff cleanup"
        app.pending_handoffs.stage(
            HandoffChannel.VLLM_DEFAULT, VllmDefaultIntent.from_target(vllm_target)
        )
        assert screen._consume_pending_vllm_default_intent() is False
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_DEFAULT)
        monkeypatch.setattr(app.pending_handoffs, "release", original_release)
        assert screen.recover_vllm_default_handoff() is True
        assert (
            app.pending_handoffs.release_recovery(HandoffChannel.LLAMACPP_DEFAULT)
            is None
        )
        assert screen._consume_pending_vllm_default_intent() is True
        await pilot.pause(0.2)
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_DEFAULT)
