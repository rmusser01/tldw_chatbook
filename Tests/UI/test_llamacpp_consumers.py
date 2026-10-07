"""Verified llama.cpp targets reach Console and Settings through bounded handoffs."""

import tomllib

import pytest

from Tests.private_profile import private_profile_test
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


@private_profile_test
@pytest.mark.asyncio
async def test_console_adopts_verified_llamacpp_for_active_session_only(request):
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


@private_profile_test
@pytest.mark.asyncio
async def test_stale_llamacpp_target_cannot_change_console_or_settings(request):
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


@private_profile_test
@pytest.mark.asyncio
async def test_llamacpp_console_adoption_failure_restores_session_and_requeues(
    request,
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
        entered = []
        original_adopt = session_store.adopt_session_ephemeral_endpoint
        original_sync = console._sync_console_chat_core_state

        def adopt_endpoint(*args, **kwargs):
            receipt = original_adopt(*args, **kwargs)
            entered.append(session_store.effective_session_settings(session_id))
            return receipt

        def sync_after_adoption():
            original_sync()
            if entered:
                raise RuntimeError("controlled post-adoption failure")

        monkeypatch.setattr(
            session_store, "adopt_session_ephemeral_endpoint", adopt_endpoint
        )
        monkeypatch.setattr(
            console, "_sync_console_chat_core_state", sync_after_adoption
        )
        assert console._session.consume_pending_llamacpp_console_intent() is False
        assert len(entered) == 1
        assert (entered[0].provider, entered[0].model, entered[0].base_url) == (
            "llama_cpp",
            target.model_id,
            target.base_url,
        )
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


def _verified_console_target(app, provider):
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmConsoleIntent

    if provider == "llama_cpp":
        owner, target = _ready_target(app)
        return owner, target, HandoffChannel.LLAMACPP_CONSOLE, LlamaCppConsoleIntent
    owner, target = _install_ready_vllm_target(app)
    return owner, target, HandoffChannel.VLLM_CONSOLE, VllmConsoleIntent


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("provider", ("llama_cpp", "vllm"))
@pytest.mark.parametrize(
    "outcome", ("success", "rollback", "cancel", "stale", "wrong_owner", "wrong_type")
)
async def test_verified_console_adoption_exact_authority_and_compensation(
    request, monkeypatch, provider, outcome
):
    """Reach the real mutation, or prove exact authority rejects it first."""
    import asyncio

    app = _console_app()
    owner, target, channel, intent_type = _verified_console_target(app, provider)
    config_path = get_cli_config_path()
    async with _ConsoleFlowHarness(app).run_test(size=(120, 42)) as pilot:
        console = pilot.app.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await pilot.pause(0.3)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        before = store.session_settings(session_id)
        conversation_id = store.persist_session_if_needed(session_id)
        row_before = app.chachanotes_db.get_conversation_by_id(conversation_id)
        metadata_before = row_before.get("metadata")
        config_before = config_path.read_bytes()
        entered = []
        original_sync = console._sync_console_chat_core_state
        cancellation = asyncio.CancelledError("controlled after adoption")

        adoption_completed = False
        original_adopt = store.adopt_session_ephemeral_endpoint

        def adopt_endpoint(*args, **kwargs):
            nonlocal adoption_completed
            receipt = original_adopt(*args, **kwargs)
            adoption_completed = True
            return receipt

        def sync_after_adoption():
            original_sync()
            if not adoption_completed:
                return
            entered.append(store.effective_session_settings(session_id))
            if len(entered) == 1 and outcome == "rollback":
                raise RuntimeError("controlled after adoption")
            if len(entered) == 1 and outcome == "cancel":
                raise cancellation

        monkeypatch.setattr(store, "adopt_session_ephemeral_endpoint", adopt_endpoint)

        monkeypatch.setattr(
            console, "_sync_console_chat_core_state", sync_after_adoption
        )
        app.pending_handoffs.stage(channel, intent_type.from_target(target))
        if outcome == "stale":
            owner.invalidate("target_changed")
        elif outcome == "wrong_owner":
            other = "vllm" if provider == "llama_cpp" else "llama_cpp"
            wrong_owner, _, _, _ = _verified_console_target(app, other)
            attribute = (
                "_llamacpp_connection_owner"
                if provider == "llama_cpp"
                else "_vllm_connection_owner"
            )
            monkeypatch.setattr(app, attribute, wrong_owner)
        elif outcome == "wrong_type":
            claim = app.pending_handoffs.claim(channel)
            # Corrupt only the detached payload, retaining the exact real token
            # so the production release still proves claim ownership.
            object.__setattr__(claim, "value", object())
            monkeypatch.setattr(
                app.pending_handoffs,
                "claim",
                lambda selected: claim if selected is channel else None,
            )
        consume = (
            console._session.consume_pending_llamacpp_console_intent
            if provider == "llama_cpp"
            else console._session.consume_pending_vllm_console_intent
        )
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError) as caught:
                consume()
            assert caught.value is cancellation
        else:
            assert consume() is (outcome == "success")
        target_url = target.base_url if provider == "llama_cpp" else target.api_url
        if outcome in {"success", "rollback", "cancel"}:
            assert entered, "the injected post-adoption boundary was not reached"
            assert (entered[0].provider, entered[0].model, entered[0].base_url) == (
                provider,
                target.model_id,
                target_url,
            )
        else:
            assert entered == []
        if outcome == "success":
            assert not app.pending_handoffs.has_pending(channel)
            assert store.effective_session_settings(session_id).base_url == target_url
        else:
            assert store.session_settings(session_id) == before
            assert store.session_ephemeral_endpoint_policy(session_id) is None
            assert app.pending_handoffs.has_pending(channel)
            assert (
                app.chachanotes_db.get_conversation_by_id(conversation_id).get(
                    "metadata"
                )
                == metadata_before
            )
        assert config_path.read_bytes() == config_before
        assert target_url not in str(
            app.chachanotes_db.get_conversation_by_id(conversation_id).get("metadata")
            or ""
        )


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("provider", ("llama_cpp", "vllm"))
async def test_verified_console_adoption_dispatches_on_real_resume(request, provider):
    from textual.screen import Screen

    app = _console_app()
    _, target, channel, intent_type = _verified_console_target(app, provider)
    async with _ConsoleFlowHarness(app).run_test(size=(120, 42)) as pilot:
        console = pilot.app.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await pilot.pause(0.3)
        await pilot.app.push_screen(Screen())
        app.pending_handoffs.stage(channel, intent_type.from_target(target))
        pilot.app.pop_screen()
        for _ in range(40):
            await pilot.pause(0.025)
            if not app.pending_handoffs.has_pending(channel):
                break
        assert pilot.app.screen is console
        assert not app.pending_handoffs.has_pending(channel)
        settings = console._session._active_console_session_settings()
        assert (settings.provider, settings.model) == (provider, target.model_id)


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("provider", ("llama_cpp", "vllm"))
async def test_verified_console_adoption_rebases_a_dropped_sampler(request, provider):
    """A valid source with no Top P can adopt a target that accepts it."""
    from dataclasses import replace
    from tldw_chatbook.Chat.console_session_settings import (
        validate_console_session_settings,
    )

    app = _console_app()
    _, target, channel, intent_type = _verified_console_target(app, provider)
    async with _ConsoleFlowHarness(app).run_test(size=(120, 42)) as pilot:
        console = pilot.app.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await pilot.pause(0.3)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        source = replace(
            store.session_settings(session_id),
            provider="custom-openai-api-2",
            model="source-model",
            top_p=None,
            source="user",
        )
        assert (
            validate_console_session_settings(source, app_config=app.app_config) == []
        )
        store.replace_session_settings(session_id, source)
        config_before = get_cli_config_path().read_bytes()
        app.pending_handoffs.stage(channel, intent_type.from_target(target))
        consume = (
            console._session.consume_pending_llamacpp_console_intent
            if provider == "llama_cpp"
            else console._session.consume_pending_vllm_console_intent
        )
        assert consume() is True
        effective = store.effective_session_settings(session_id)
        assert (effective.provider, effective.model) == (provider, target.model_id)
        assert effective.top_p is not None
        assert (
            validate_console_session_settings(effective, app_config=app.app_config)
            == []
        )
        assert not app.pending_handoffs.has_pending(channel)
        assert get_cli_config_path().read_bytes() == config_before


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("provider", ("llama_cpp", "vllm"))
@pytest.mark.parametrize("same_pair", (False, True))
async def test_verified_console_adoption_uses_canonical_target_defaults(
    request, provider, same_pair
):
    from dataclasses import replace
    from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
    from tldw_chatbook.Chat.console_session_settings import (
        build_target_default_console_session_settings,
    )

    app = _console_app()
    _, target, channel, intent_type = _verified_console_target(app, provider)
    async with _ConsoleFlowHarness(app).run_test(size=(120, 42)) as pilot:
        console = pilot.app.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await pilot.pause(0.3)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        defaults = build_target_default_console_session_settings(
            app.app_config, provider, target.model_id
        )
        source = replace(
            store.session_settings(session_id),
            provider=provider if same_pair else "llama_cpp",
            model=target.model_id if same_pair else "other-source-model",
            temperature=0.17,
            top_p=0.42,
            system_prompt="retained prompt",
            character_label="Retained character",
            pinned_prefill="retained prefill",
            source="user",
        )
        policy = ConsoleContextPolicyOverrides(custom_budget_tokens=2048)
        store.replace_session_settings(session_id, source)
        store.set_session_context_policy_overrides(session_id, policy)
        conversation_id = store.persist_session_if_needed(session_id)
        config_before = get_cli_config_path().read_bytes()
        app.pending_handoffs.stage(channel, intent_type.from_target(target))
        consume = (
            console._session.consume_pending_llamacpp_console_intent
            if provider == "llama_cpp"
            else console._session.consume_pending_vllm_console_intent
        )
        assert consume() is True
        settings = store.session_settings(session_id)
        expected = source if same_pair else defaults
        assert (settings.temperature, settings.top_p) == (
            expected.temperature,
            expected.top_p,
        )
        assert (
            settings.system_prompt,
            settings.character_label,
            settings.pinned_prefill,
        ) == (source.system_prompt, source.character_label, source.pinned_prefill)
        assert store.session_context_policy_overrides(session_id) == policy
        assert settings.base_url == defaults.base_url
        assert settings.source == "user"
        target_url = target.base_url if provider == "llama_cpp" else target.api_url
        assert store.effective_session_settings(session_id).base_url == target_url
        assert target_url not in str(
            app.chachanotes_db.get_conversation_by_id(conversation_id).get("metadata")
            or ""
        )
        assert get_cli_config_path().read_bytes() == config_before


@pytest.mark.asyncio
@private_profile_test
async def test_verified_console_adoption_captures_rollback_after_controller_sync(
    request, monkeypatch
):
    from dataclasses import replace

    app = _console_app()
    _, target, channel, intent_type = _verified_console_target(app, "llama_cpp")
    async with _ConsoleFlowHarness(app).run_test(size=(120, 42)) as pilot:
        console = pilot.app.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await pilot.pause(0.3)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        synchronized = replace(
            store.session_settings(session_id), temperature=0.27, source="user"
        )
        original_ensure = console._ensure_console_chat_controller
        original_sync = console._sync_console_chat_core_state
        original_adopt = store.adopt_session_ephemeral_endpoint
        entered = []
        ensured = []
        controller_slot = type(console)._console_chat_controller
        missing_read = [True]

        def initially_absent(screen):
            if screen is console and missing_read[0]:
                missing_read[0] = False
                return None
            return controller_slot.fget(screen)

        # Exercise only the absent-owner observation; the real initializer
        # then resolves the existing runtime and performs its normal sync.
        monkeypatch.setattr(
            type(console),
            "_console_chat_controller",
            property(initially_absent, controller_slot.fset),
        )

        def ensure_with_sync():
            first = not ensured
            ensured.append(bool(entered))
            controller = original_ensure()
            if first:
                store.replace_session_settings(session_id, synchronized)
                original_sync()
            return controller

        def adopt_endpoint(*args, **kwargs):
            receipt = original_adopt(*args, **kwargs)
            entered.append(store.effective_session_settings(session_id))
            return receipt

        def fail_after_adoption():
            original_sync()
            if entered:
                raise RuntimeError("controlled post-adoption failure")

        monkeypatch.setattr(
            console, "_ensure_console_chat_controller", ensure_with_sync
        )
        monkeypatch.setattr(store, "adopt_session_ephemeral_endpoint", adopt_endpoint)
        monkeypatch.setattr(
            console, "_sync_console_chat_core_state", fail_after_adoption
        )
        app.pending_handoffs.stage(channel, intent_type.from_target(target))
        assert console._session.consume_pending_llamacpp_console_intent() is False
        assert ensured and ensured[0] is False
        assert missing_read == [False]
        assert len(entered) == 1 and entered[0].base_url == target.base_url
        assert store.session_settings(session_id) == synchronized
        assert store.session_ephemeral_endpoint_policy(session_id) is None
        assert console._console_chat_controller.temperature == synchronized.temperature
        assert app.pending_handoffs.has_pending(channel)
