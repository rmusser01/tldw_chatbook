"""Named endpoint discovery through the mounted parent settings modal."""

import asyncio
from dataclasses import replace

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_session_settings import ModalHarness
from Tests.UI.test_console_settings_model_change import pick_mode_opener, real_rebase
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderDraftIdentity,
    ProviderProbeResult,
)
from tldw_chatbook.Widgets.Console.console_endpoint_template_modal import (
    ConsoleEndpointTemplateModal,
)
from tldw_chatbook.Widgets.Console.console_model_popover import (
    switcher_readiness_words,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal


@pytest.fixture(autouse=True)
def _isolate_context_metadata(monkeypatch):
    """Endpoint UI tests never consult a real server for unrelated context data."""
    from tldw_chatbook.Chat.console_context_window import resolve_context_window
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    async def resolve(_gateway, settings):
        return resolve_context_window(settings.provider, settings.model or "")

    monkeypatch.setattr(ConsoleProviderGateway, "resolve_context_window", resolve)


def _modal(app, *, provider="llama_cpp", tester=None):
    providers_models = {"llama_cpp": ["model-a"]}
    return ConsoleSettingsModal(
        settings=ConsoleSessionSettings(
            provider=provider, model="model-a", base_url="http://127.0.0.1:9099"
        ),
        app_config=app.app_config,
        providers_models=providers_models,
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
        connection_tester=tester,
        # TASK-33006.4: the Console's wiring; a pair changes only by a pick.
        draft_rebaser=real_rebase,
        model_picker=pick_mode_opener(app, app.app_config, providers_models),
    )


async def _pick_in_pick_mode(app, pilot, *keys: str) -> None:
    """Pick mode is open: type ``keys`` (if any) in Find, then Enter."""
    from tldw_chatbook.Widgets.Console.console_model_popover import (
        ConsoleModelPopover,
    )

    for _ in range(30):
        await pilot.pause(0.02)
        if isinstance(app.screen, ConsoleModelPopover):
            break
    assert isinstance(app.screen, ConsoleModelPopover) and app.screen._pick_only
    await app.workers.wait_for_complete()
    await pilot.press(*keys, "enter")
    await pilot.pause()


async def _settled(modal, pilot):
    for _ in range(30):
        await pilot.pause(0.02)
        identity = modal._current_connection_probe_identity()
        evidence = (
            modal._connection_evidence_store.evidence_for(identity)
            if identity
            else None
        )
        if evidence and evidence.endpoint != "testing":
            return evidence
    pytest.fail("Discovery never settled into current evidence")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        ProviderProbeResult("reachable", ("served-model",)),
        ProviderProbeResult("unreachable", (), "timeout"),
    ],
)
@private_profile_test
async def test_create_endpoint_returns_to_responsive_settings_with_settled_evidence(
    request,
    outcome,
):
    async def connection(identity):
        assert type(identity) is ProviderDraftIdentity
        assert identity.custom_endpoint_id == "custom-ep:report-probe"
        assert identity.provider_key == "llama_cpp"
        assert identity.connection_identity == ("llama_cpp", "http://127.0.0.1:9999")
        return outcome

    app = ModalHarness()
    modal = _modal(app, tester=connection)
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-endpoint-new", Button).press()
        await pilot.pause()
        template = app.screen
        assert isinstance(template, ConsoleEndpointTemplateModal)
        template.query_one("#endpoint-template-name", Input).value = "Report probe"
        template.query_one(
            "#endpoint-template-url", Input
        ).value = "http://127.0.0.1:9999"
        await pilot.pause()
        template.query_one("#endpoint-template-create", Button).press()
        # TASK-33006.4: the created entry opens pick mode on itself; the
        # pick (the entry with the template's model) lands on a pair.
        await _pick_in_pick_mode(app, pilot)
        for _ in range(30):
            await pilot.pause(0.02)
            if (
                app.screen is modal
                and modal._active_provider == "custom-ep:report-probe"
            ):
                break
        assert app.screen is modal
        assert modal._active_provider == "custom-ep:report-probe"
        evidence = await _settled(modal, pilot)
        assert evidence.endpoint == outcome.endpoint
        assert evidence.model_ids == outcome.model_ids
        assert evidence.category == outcome.category
        assert evidence.generation == "not_tested"
        assert not modal.query_one("#console-settings-model-discover", Button).disabled
        assert "Testing connection" not in str(
            modal.query_one("#console-settings-model-discover-status", Static).content
        )
        assert "report-probe" in app.app_config["custom_endpoints"]
        # TASK-33006.1: Streaming is an On/Off Select; it still responds.
        streaming = modal.query_one("#console-settings-streaming", Select)
        before = streaming.value
        streaming.value = "off" if before == "on" else "on"
        await pilot.pause()
        assert modal._streaming_draft is (streaming.value == "on") and (
            streaming.value != before
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "family,key,endpoint",
    [
        ("llama_cpp", "llama_cpp", "http://127.0.0.1:9099"),
        ("openai_compatible", "custom", "http://127.0.0.1:9099/v1/chat/completions"),
        ("ollama", "ollama", "http://127.0.0.1:9099/v1/chat/completions"),
    ],
)
async def test_named_entry_uses_family_contract_and_own_credential_source(
    monkeypatch, family, key, endpoint
):
    monkeypatch.setenv("ENTRY_PROBE_KEY", "fixture-entry-key")
    app = ModalHarness()
    app.app_config = {
        "api_settings": {key: {"api_key": "wrong-family-key"}},
        "custom_endpoints": {
            "one": {
                "display_name": "One",
                "family": family,
                "base_url": "http://127.0.0.1:9099",
                "api_key_env": "ENTRY_PROBE_KEY",
                "models": ["model-a"],
            }
        },
    }
    modal = _modal(app, provider="custom-ep:one", tester=lambda _: None)
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        assert modal._provider_supports_model_discovery("custom-ep:one")
        identity = modal._current_connection_probe_identity()
        assert identity is not None
        assert identity.provider_key == key
        assert identity.custom_endpoint_id == "custom-ep:one"
        assert identity.connection_identity == (key, endpoint)
        assert identity.credential_source == "environment"
        assert "fixture-entry-key" not in repr(identity)
        monkeypatch.setenv("ENTRY_PROBE_KEY", "replacement-entry-key")
        assert modal._current_connection_probe_identity() != identity
        assert replace(identity, custom_endpoint_id="custom-ep:two") != identity


@pytest.mark.asyncio
async def test_late_result_cannot_publish_after_same_family_entry_switch():
    started = asyncio.Event()
    release = asyncio.Event()
    current_started = asyncio.Event()
    current_release = asyncio.Event()
    old_returned = asyncio.Event()

    async def connection(identity):
        if identity.custom_endpoint_id == "custom-ep:one":
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            old_returned.set()
            return ProviderProbeResult("reachable", ("stale-model",))
        current_started.set()
        await current_release.wait()
        return ProviderProbeResult("reachable", ("current-model",))

    app = ModalHarness()
    app.app_config = {
        "custom_endpoints": {
            slug: {
                "display_name": slug.title(),
                "family": "llama_cpp",
                "base_url": "http://127.0.0.1:9099",
                "models": ["model-a"],
            }
            for slug in ("one", "two")
        }
    }
    modal = _modal(app, provider="custom-ep:one", tester=connection)
    async with app.run_test(size=(160, 48)) as pilot:
        try:
            await app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#console-settings-model-discover", Button).press()
            await asyncio.wait_for(started.wait(), 2)
            modal._model_picked(("custom-ep:two", "model-a"))  # pick mode's result
            await pilot.pause()
            modal.query_one("#console-settings-model-discover", Button).press()
            await asyncio.wait_for(current_started.wait(), 2)
            release.set()
            await asyncio.wait_for(old_returned.wait(), 2)
            await pilot.pause()
            assert modal.query_one("#console-settings-model-discover", Button).disabled
            current_release.set()
            evidence = await _settled(modal, pilot)
            assert evidence.model_ids == ("current-model",)
            await pilot.pause()
            # TASK-33006.4: a listing never picks the model; Change does.
            assert modal._current_model_value() == "model-a"
            assert modal._connection_evidence_store.evidence_for(
                modal._current_connection_probe_identity()
            ).model_ids == ("current-model",)
        finally:
            release.set()
            current_release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["endpoint", "credential"])
async def test_changed_entry_discards_pending_result_and_restores_retry(
    monkeypatch, change
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def connection(_identity):
        started.set()
        await release.wait()
        return ProviderProbeResult("reachable", ("obsolete-model",))

    monkeypatch.setenv("ENTRY_PROBE_KEY", "fixture-first-key")
    app = ModalHarness()
    entry = {
        "display_name": "One",
        "family": "llama_cpp",
        "base_url": "http://127.0.0.1:9099",
        "api_key_env": "ENTRY_PROBE_KEY",
        "models": ["model-a"],
    }
    app.app_config = {"custom_endpoints": {"one": entry}}
    modal = _modal(app, provider="custom-ep:one", tester=connection)
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        button = modal.query_one("#console-settings-model-discover", Button)
        button.press()
        await asyncio.wait_for(started.wait(), 2)
        assert button.disabled
        if change == "endpoint":
            entry["base_url"] = "http://127.0.0.1:9999"
        else:
            monkeypatch.setenv("ENTRY_PROBE_KEY", "fixture-second-key")
        release.set()
        await pilot.pause()
        assert not button.disabled
        assert modal._current_model_value() == "model-a"
        assert (
            modal._connection_evidence_store.evidence_for(
                modal._current_connection_probe_identity()
            )
            is None
        )
        assert "Testing connection" not in str(
            modal.query_one("#console-settings-model-discover-status", Static).content
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("display_name", "provider_id", "base_url", "model_id"),
    [
        (
            "New live endpoint",
            "custom-ep:new-live-endpoint",
            "http://127.0.0.1:9999",
            "live-served-model",
        ),
        pytest.param(
            "Llama local 2",
            "custom-ep:llama-local-2",
            "http://127.0.0.1:9090",
            r"E:\LLM-Models\Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf",
            id="reported-windows-crashes",
        ),
    ],
)
@private_profile_test
async def test_create_endpoint_with_live_controller_rebase_settles(
    request, monkeypatch, display_name, provider_id, base_url, model_id
):
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover

    calls = []

    async def connection(identity, *, app_config):
        from tldw_chatbook.Chat.custom_endpoint_registry import entry_for

        calls.append(
            (
                identity.custom_endpoint_id,
                entry_for(app_config, provider_id) is not None,
            )
        )
        assert identity.provider_key == "llama_cpp"
        assert identity.connection_identity == ("llama_cpp", base_url)
        return ProviderProbeResult("reachable", (model_id,))

    monkeypatch.setattr(
        ChatScreen, "_test_console_connection", staticmethod(connection)
    )
    app = _persisted_console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await console._open_console_settings(focus_model=True)
        await pilot.pause()
        modal = harness.screen
        modal.query_one("#console-settings-endpoint-new", Button).press()
        await pilot.pause()
        template = harness.screen
        template.query_one("#endpoint-template-name", Input).value = display_name
        template.query_one("#endpoint-template-url", Input).value = base_url
        await pilot.pause()
        template.query_one("#endpoint-template-create", Button).press()
        # TASK-33006.4: the created entry opens the real pick-only switcher
        # on itself; typing the served id picks the pair, then it is probed.
        for _ in range(60):
            await pilot.pause(0.02)
            if isinstance(harness.screen, ConsoleModelPopover):
                break
        picker = harness.screen
        assert isinstance(picker, ConsoleModelPopover) and picker._pick_only
        find = picker.query_one("#console-popover-find", Input)
        assert find.value == f"{display_name} "
        await harness.workers.wait_for_complete()
        find.value = f"{display_name} {model_id}"
        await pilot.pause()
        row = picker.highlighted_row()
        assert (row.provider, row.model) == (provider_id, model_id)
        await pilot.press("enter")
        for _ in range(30):
            await pilot.pause(0.02)
            if harness.screen is modal:
                break
        assert harness.screen is modal
        evidence = await _settled(modal, pilot)
        assert evidence.endpoint == "reachable"
        assert evidence.model_ids == (model_id,)
        # Review round 1: Create lists the entry first (so pick mode offers
        # what it serves), then the pick's evidence probe follows.
        assert calls == [(provider_id, True)] * 2
        assert modal._current_model_value() == model_id
        assert modal._current_draft_discovery_identity().provider_key == provider_id

        # Creation persists the entry, but Cancel leaves the conversation on
        # its original provider. Selecting the new entry must run the real
        # quick-picker rebase before Apply commits the exact registry ID.
        # TASK-33003.5: the switch onto the entry is an unapplied edit, so
        # Cancel asks first; Discard is the choice that closes unchanged.
        modal.query_one("#console-settings-cancel", Button).press()
        await pilot.pause()
        assert "Model" in str(  # the pair is one field (TASK-33006.4)
            modal.query_one("#console-settings-close-message", Static).renderable
        )
        await pilot.press("d")
        await pilot.pause()
        assert harness.screen is console
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert store.session_settings(session_id).provider == "llama_cpp"
        await console.action_open_console_model_popover()
        await pilot.pause()
        quick = harness.screen
        assert isinstance(quick, ConsoleModelPopover)
        await harness.workers.wait_for_complete()
        # TASK-33004.4: Find + the entry's pair row replace the provider
        # Select and model picker; the entry's name picks the provider.
        quick.query_one("#console-popover-find", Input).value = (
            f"{display_name} {model_id}"
        )
        await pilot.pause()
        row = quick.highlighted_row()
        assert (row.provider, row.model) == (provider_id, model_id)
        apply = quick.query_one("#console-popover-apply", Button)
        assert not apply.disabled
        apply.press()
        await pilot.pause()
        assert harness.screen is console
        settings = store.session_settings(session_id)
        assert settings.provider == provider_id
        assert settings.model == model_id
        assert settings.base_url == base_url


@pytest.mark.asyncio
@private_profile_test
async def test_endpoint_command_create_lands_on_a_pair_in_pick_mode(
    request, monkeypatch
):
    """Review round 1 (finding 1): ``/endpoint`` -> Create lands the entry
    on a pair as New endpoint… does. Pick mode opens on the entry over the
    exact Chat settings the command opened, listing the model it serves
    though the template named none, and the pick rebases that draft."""
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover

    async def connection(identity, *, app_config):
        del app_config
        assert identity.custom_endpoint_id == "custom-ep:command-box"
        return ProviderProbeResult("reachable", ("served-z",))

    monkeypatch.setattr(
        ChatScreen, "_test_console_connection", staticmethod(connection)
    )
    harness = _ConsoleFlowHarness(_persisted_console_app())
    async with harness.run_test(size=(160, 48)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        console.action_open_console_new_endpoint()
        for _ in range(60):
            await pilot.pause(0.02)
            if isinstance(harness.screen, ConsoleEndpointTemplateModal):
                break
        template = harness.screen
        assert isinstance(template, ConsoleEndpointTemplateModal)
        modal = harness.screen_stack[-2]
        assert isinstance(modal, ConsoleSettingsModal)
        template.query_one("#endpoint-template-name", Input).value = "Command box"
        template.query_one(
            "#endpoint-template-url", Input
        ).value = "http://127.0.0.1:9998"
        template.query_one("#endpoint-template-models", Input).value = ""
        await pilot.pause()
        template.query_one("#endpoint-template-create", Button).press()
        for _ in range(60):
            await pilot.pause(0.02)
            if isinstance(harness.screen, ConsoleModelPopover):
                break
        picker = harness.screen
        assert isinstance(picker, ConsoleModelPopover) and picker._pick_only
        assert picker.query_one("#console-popover-find", Input).value == "Command box "
        await harness.workers.wait_for_complete()
        await pilot.press(*"served-z", "enter")
        for _ in range(30):
            await pilot.pause(0.02)
            if harness.screen is modal:
                break
        assert harness.screen is modal
        assert (modal._active_provider, modal._current_model_value()) == (
            "custom-ep:command-box",
            "served-z",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("display_name", "slug", "base_url", "model_id"),
    [
        (
            "New live endpoint",
            "new-live-endpoint",
            "http://127.0.0.1:9999",
            "live-served-model",
        ),
        pytest.param(
            "Llama local 2",
            "llama-local-2",
            "http://127.0.0.1:9090",
            r"E:\LLM-Models\Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf",
            id="reported-windows-crashes",
        ),
    ],
)
@private_profile_test
async def test_switch_model_applies_a_saved_entry_through_the_live_rebase(
    request, display_name, slug, base_url, model_id
):
    """The Switch model half of the test above, on its own (TASK-33004.4).

    That test's modal half fails on dev at its ``provider_key`` assertion
    before it reaches Switch model, so this case seeds the saved entry
    directly and runs the real controller rebase and live commit.
    """
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _open_provider_popover,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

    provider_id = f"custom-ep:{slug}"
    app = _persisted_console_app()
    assert SettingsConfigAdapter().save_sections(
        {
            f"custom_endpoints.{slug}": {
                "display_name": display_name,
                "family": "llama_cpp",
                "base_url": base_url,
                "models": [model_id],
            }
        }
    )
    app.app_config = load_settings(force_reload=True)
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert store.session_settings(session_id).provider == "llama_cpp"
        quick = await _open_provider_popover(console, harness, pilot)
        # The entry's display name picks the provider; the row is a pair.
        quick.query_one("#console-popover-find", Input).value = (
            f"{display_name} {model_id}"
        )
        await pilot.pause()
        row = quick.highlighted_row()
        assert (row.kind, row.provider, row.model) == ("pair", provider_id, model_id)
        values = str(quick.query_one("#console-popover-values-label", Static).render())
        # TASK-33004.5: the label names the pair, display name included.
        assert values == f"Values for {model_id} · {display_name}"
        quick.query_one("#console-popover-apply", Button).press()
        await pilot.pause()
        assert harness.screen is console
        settings = store.session_settings(session_id)
        assert settings.provider == provider_id
        assert settings.model == model_id
        assert settings.base_url == base_url


@pytest.mark.asyncio
async def test_completed_entry_listing_becomes_unverified_when_credential_changes(
    monkeypatch,
):
    async def connection(_identity):
        return ProviderProbeResult("reachable", ("model-a",))

    monkeypatch.setenv("ENTRY_PROBE_KEY", "fixture-first-key")
    app = ModalHarness()
    app.app_config = {
        "custom_endpoints": {
            "one": {
                "display_name": "One",
                "family": "llama_cpp",
                "base_url": "http://127.0.0.1:9099",
                "api_key_env": "ENTRY_PROBE_KEY",
                "models": ["model-a"],
            }
        }
    }
    modal = _modal(app, provider="custom-ep:one", tester=connection)
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-model-discover", Button).press()
        await _settled(modal, pilot)
        assert modal._current_model_discovery_matches_current_draft()
        monkeypatch.setenv("ENTRY_PROBE_KEY", "fixture-second-key")
        modal._sync_readiness_display()
        assert not modal._current_model_discovery_matches_current_draft()


@pytest.mark.asyncio
@private_profile_test
async def test_make_default_retains_hyphenated_entry_and_endpoint_on_reload(
    request,
):
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _drain_settings_tasks,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.Chat.console_session_settings import (
        build_default_console_session_settings,
    )
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

    app = _persisted_console_app()
    assert SettingsConfigAdapter().save_sections(
        {
            "custom_endpoints.gpu-node": {
                "display_name": "GPU node",
                "family": "llama_cpp",
                "base_url": "http://127.0.0.1:9099",
                "models": ["registry-model"],
            }
        }
    )
    app.app_config = load_settings(force_reload=True)
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await console._open_console_settings(focus_model=True)
        await pilot.pause()
        modal = harness.screen
        # TASK-33006.4: Change opens the real pick-only switcher (the
        # Console's open_model_picker); the entry's name picks its row.
        assert harness.focused is modal.query_one(
            "#console-settings-model-change", Button
        )
        await pilot.press("enter")
        await _pick_in_pick_mode(harness, pilot, *"GPU node registry")
        assert harness.screen is modal
        assert modal._active_provider == "custom-ep:gpu-node"
        assert modal._current_model_value() == "registry-model"
        button = modal.query_one("#console-settings-make-default", Button)
        assert not button.disabled
        button.press()
        await pilot.pause()
        await _drain_settings_tasks(app)
        loaded = load_settings(force_reload=True)
        assert loaded["chat_defaults"]["provider"] == "custom-ep:gpu-node"
        defaults = build_default_console_session_settings(loaded)
        assert defaults.provider == "custom-ep:gpu-node"
        assert defaults.base_url == "http://127.0.0.1:9099"
        assert defaults.model == "registry-model"


@pytest.mark.asyncio
async def test_editing_builtin_endpoint_updates_bound_dirty_draft():
    app = ModalHarness()
    modal = _modal(app, tester=lambda _: None)
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one(
            "#console-settings-base-url", Input
        ).value = "http://127.0.0.1:9999"
        await pilot.pause()
        assert modal._endpoint_draft.value == "http://127.0.0.1:9999"
        assert modal._endpoint_draft.bound_provider_config_key == "llama_cpp"
        assert modal._endpoint_draft.dirty is True
        assert modal._endpoint_draft.checked is False


@pytest.mark.asyncio
async def test_a_models_listing_never_changes_the_model():
    """TASK-33006.4 (spec rule 1): Test connection lists models but never
    picks one, even a sole listed model; only Change's pick mode does."""

    async def connection(_identity):
        return ProviderProbeResult("reachable", ("served-model",))

    app = ModalHarness()
    modal = _modal(app, tester=connection)
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-model-discover", Button).press()
        evidence = await _settled(modal, pilot)
        assert evidence.model_ids == ("served-model",)
        assert modal._current_model_value() == "model-a"
        assert modal._current_model_discovery_matches_current_draft()
        status = modal.query_one("#console-settings-model-discover-status", Static)
        assert str(status.content) == "1 model listed"


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["endpoint"])  # no model adapter since TASK-33006.4
async def test_named_entry_adapter_echo_does_not_cancel_pending_probe(adapter):
    started = asyncio.Event()
    release = asyncio.Event()

    async def connection(_identity):
        started.set()
        await release.wait()
        return ProviderProbeResult("reachable", ("model-a",))

    app = ModalHarness()
    app.app_config = {
        "custom_endpoints": {
            "one": {
                "display_name": "One",
                "family": "llama_cpp",
                "base_url": "http://127.0.0.1:9099",
                "models": ["model-a"],
            }
        }
    }
    modal = _modal(app, provider="custom-ep:one", tester=connection)
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-model-discover", Button).press()
        await asyncio.wait_for(started.wait(), 2)
        endpoint = modal.query_one("#console-settings-base-url", Input)
        endpoint.post_message(Input.Changed(endpoint, endpoint.value))
        await pilot.pause()
        assert modal._active_connection_probe_token is not None
        release.set()
        evidence = await _settled(modal, pilot)
        assert evidence.endpoint == "reachable"


def _stub_llama_network(monkeypatch, server: dict[str, bool]) -> list[str]:
    """Replace only the network under the real probe (TASK-33005.2).

    Everything above it -- the modal's tester, the Console's retry, the
    category mapping -- is production code. Returns the probed URLs.
    """
    import errno

    import httpx

    import tldw_chatbook.UI.Screens.settings_endpoint_probe as probe_module

    real_probe = probe_module.probe_settings_endpoint
    probed: list[str] = []

    async def network(request: httpx.Request) -> httpx.Response:
        if not server["up"]:
            raise httpx.ConnectError("refused") from OSError(
                errno.ECONNREFUSED, "refused"
            )
        return httpx.Response(200, json={"data": [{"id": "model-a"}]})

    async def probe(base_url, **kwargs):
        probed.append(base_url)
        async with httpx.AsyncClient(transport=httpx.MockTransport(network)) as client:
            return await real_probe(base_url, http_client=client, **kwargs)

    monkeypatch.setattr(probe_module, "probe_settings_endpoint", probe)
    return probed


async def _console_settled(console, pilot, predicate) -> None:
    """Let the Console's 0.25 s idle poll pick up a settled result."""
    for _ in range(40):
        await pilot.pause(0.05)
        if predicate(console._active_console_settings_readiness()[1]):
            await pilot.pause(0.3)  # One more tick: the poll repaints.
            return
    pytest.fail("Console readiness never moved")


def _rail_text(console, selector: str) -> str:
    widget = console.query_one(selector)
    return str(getattr(widget, "label", None) or widget.render())


@pytest.mark.asyncio
@private_profile_test
async def test_refused_chat_settings_test_blocks_console_until_one_retry(
    request, monkeypatch
):
    """TASK-33005.2 (AC#1/#2/#3/#7): the real Chat settings modal's refused
    test of the active llama.cpp turns the real Console Not ready with a
    retry, and one Retry after the server starts restores Ready."""
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector

    server = {"up": False}
    probed = _stub_llama_network(monkeypatch, server)
    harness = _ConsoleFlowHarness(_persisted_console_app())
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        assert console._active_console_settings_readiness()[1].blocker is None
        # Review round 1: the header badge is the status row's word.
        assert _rail_text(console, "#workbench-header-status") == "Ready · not tested"

        await console._open_console_settings(focus_model=False)
        await pilot.pause()
        modal = harness.screen
        modal.query_one("#console-settings-model-discover", Button).press()
        assert (await _settled(modal, pilot)).category == "connection_refused"
        await pilot.pause()
        refused = "Not ready · refused :9099"  # TASK-33005.3: one word everywhere
        assert _rail_text(modal, "#console-settings-readiness").startswith(
            f"{refused}\n"
        )  # Chat settings readiness
        modal.query_one("#console-settings-cancel", Button).press()
        await pilot.pause()
        assert harness.screen is console
        await _console_settled(console, pilot, lambda r: r.blocker is not None)

        readiness = console._active_console_settings_readiness()[1]
        assert (readiness.blocker, readiness.recovery_action) == (
            "endpoint_unreachable",
            "retry_connection",
        )
        # Model section (left rail), setup card, Inspector and the
        # Conversation settings summary all read the same readiness.
        # TASK-33005.3 (AC#8): ... in the same word, with the header badge
        # and the switcher rows; only Not ready paints the rail line red.
        recovery = console.query_one("#console-model-section-recovery")
        assert _rail_text(console, "#console-model-section-recovery") == refused
        assert recovery.has_class("conversation-attention-error")
        assert _rail_text(console, "#console-settings-readiness-row") == refused
        assert refused in _rail_text(console, "#console-setup-step-1")
        assert switcher_readiness_words(
            console._console_default_readiness("llama_cpp", "model-a")
        ) == refused
        assert "connection refused" in _rail_text(
            console, "#console-settings-endpoint-row"
        )
        retry = console.query_one("#console-setup-modal-action", Button)
        assert str(retry.label) == "Retry connection"
        # Review finding 6: the "status row" also covers the header word and
        # the composer's reason strip; Send itself is disabled.
        assert _rail_text(console, "#workbench-header-status") == refused
        assert console.query_one("#console-send-message", Button).disabled
        assert _rail_text(console, "#console-send-disabled-reason") == (
            "Send blocked — retry the connection to continue ›"
        )
        assert "endpoint unreachable" in console._console_provider_blocker_copy()
        assert "Retry connection" in console._console_setup_blocked_reason()
        default = console._console_default_readiness("llama_cpp", "model-a")
        assert default.blocker == "endpoint_unreachable"  # AC#3
        reads = len(probed)
        console._active_console_settings_readiness()
        console._poll_console_credential_readiness()
        assert len(probed) == reads  # AC#4: reading never probes.

        # The Conversation settings button (Inspector rail) re-tests too,
        # with no settings opened; a still-down server stays blocked.
        summary_button = console.query_one("#console-settings-open", Button)
        await console.on_console_settings_open(Button.Pressed(summary_button))
        await console.workers.wait_for_complete()
        await pilot.pause()
        assert harness.screen is console
        assert len(probed) == reads + 1
        assert console._active_console_settings_readiness()[1].blocker == (
            "endpoint_unreachable"
        )

        server["up"] = True
        retry.press()
        await _console_settled(console, pilot, lambda r: r.blocker is None)

        assert harness.screen is console  # Retry opened nothing.
        assert len(probed) == reads + 2
        assert console._console_setup_blocked_reason() == ""
        ready = console._active_console_settings_readiness()[1]
        assert ready.endpoint == "reachable"
        reachable = f"Ready · reachable {ready.observed_at.astimezone():%H:%M}"
        for selector in (
            "#console-model-section-recovery",
            "#console-settings-readiness-row",
            "#workbench-header-status",
        ):
            assert _rail_text(console, selector) == reachable, selector
        assert not recovery.has_class("conversation-attention-error")
        assert switcher_readiness_words(
            console._console_default_readiness("llama_cpp", "model-a")
        ) == reachable


@pytest.mark.asyncio
@private_profile_test
async def test_shared_evidence_change_refreshes_the_console_once(request):
    """TASK-33005.2 (AC#6): a result settled elsewhere reaches the Console
    through the idle poll's existing gate, in exactly one refresh."""
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )
    from tldw_chatbook.Chat.provider_test_evidence import (
        ProviderTestEvidenceStore,
    )

    harness = _ConsoleFlowHarness(_persisted_console_app())
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        console._stop_console_credential_poll_timer()
        console._poll_console_credential_readiness()
        refreshes = []
        real_sync = console._sync_console_settings_summary
        console._sync_console_settings_summary = lambda: (
            refreshes.append(1),
            real_sync(),
        )

        store = ProviderTestEvidenceStore(lambda: harness)
        store.settle(
            store.begin(
                ProviderDraftIdentity(
                    provider_key="llama_cpp",
                    connection_identity=canonical_connection_identity(
                        "llama_cpp", "http://127.0.0.1:9099"
                    ),
                    credential_source="none",
                    credential_revision=0,
                    draft_generation=0,
                )
            ),
            ProviderProbeResult("unreachable", (), "timeout"),
        )
        for _ in range(3):
            console._poll_console_credential_readiness()
        assert len(refreshes) == 1  # Counted before any other tick can run.

        await pilot.pause()
        assert _rail_text(console, "#console-settings-readiness-row") == (
            "Not ready · timed out"
        )


_KEYED_VLLM_CONFIG = {
    "api_settings": {"vllm": {"api_url": "http://127.0.0.1:8000", "api_key": "sk-good"}}
}


def _stub_keyed_server(monkeypatch, server_key: str) -> list[str | None]:
    """A vLLM started with ``--api-key server_key``, under the real probe.

    Returns the Authorization header of every request it answered.
    """
    import httpx

    import tldw_chatbook.UI.Screens.settings_endpoint_probe as probe_module

    sent: list[str | None] = []

    async def server(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers.get("Authorization"))
        if sent[-1] != f"Bearer {server_key}":
            return httpx.Response(401)
        return httpx.Response(200, json={"data": [{"id": "model-a"}]})

    real_probe = probe_module.probe_settings_endpoint

    async def probe(base_url, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            return await real_probe(base_url, http_client=client, **kwargs)

    monkeypatch.setattr(probe_module, "probe_settings_endpoint", probe)
    return sent


def _vllm_identity(key: str) -> ProviderDraftIdentity:
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )
    from tldw_chatbook.Chat.provider_test_evidence import (
        connection_credential_revision,
    )

    return ProviderDraftIdentity(
        provider_key="vllm",
        connection_identity=canonical_connection_identity(
            "vllm", "http://127.0.0.1:8000"
        ),
        credential_source="stored",
        credential_revision=connection_credential_revision(key),
        draft_generation=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_key", "blocker"),
    [("sk-good", None), ("sk-other", "credential_rejected")],
    ids=["right-key", "wrong-key"],
)
@private_profile_test
async def test_a_keyed_local_server_is_tested_with_the_key_a_send_uses(
    request, monkeypatch, server_key, blocker
):
    """TASK-33005.2 review I-1: the probe sent no key, so a vLLM started with
    ``--api-key`` answered 401 and the Console blocked on "key rejected" for
    a key it never sent, which no re-test could clear. The probe now carries
    the saved key, so the verdict is about the key that was sent."""
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )
    from tldw_chatbook.Chat.provider_test_evidence import shared_connection_evidence
    from tldw_chatbook.UI.Console_Modules.connection_probe import (
        settle_connection_probe,
    )

    sent = _stub_keyed_server(monkeypatch, server_key)
    app = SimpleNamespace()
    identity = _vllm_identity("sk-good")

    await settle_connection_probe(app, identity, app_config=_KEYED_VLLM_CONFIG)
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="vllm", model="model-a"),
        app_config=_KEYED_VLLM_CONFIG,
        connection_evidence=shared_connection_evidence(lambda: app),
    )

    assert sent == ["Bearer sk-good"]
    assert readiness.connection == identity  # The Console's own key for it.
    assert readiness.blocker == blocker
    assert (readiness.operability == "ready_to_send") is (blocker is None)


@pytest.mark.asyncio
@private_profile_test
async def test_a_probe_for_a_replaced_key_never_sends_the_current_one(
    request, monkeypatch
):
    """The saved key changed after the identity was keyed: sending the new
    key would record its verdict under the old key's connection."""
    from tldw_chatbook.UI.Console_Modules.connection_probe import (
        probe_console_connection,
    )

    sent = _stub_keyed_server(monkeypatch, "sk-good")

    result = await probe_console_connection(
        _vllm_identity("sk-replaced"), app_config=_KEYED_VLLM_CONFIG
    )

    assert sent == []
    assert result == ProviderProbeResult("unreachable", (), "connection_error")


@pytest.mark.asyncio
@private_profile_test
async def test_an_unconfigured_llama_cpp_is_one_connection_in_chat_settings_and_console(
    request,
):
    """TASK-33005.2 review finding 5: with nothing saved, Chat settings
    prefills the default llama.cpp origin and tests it, and the Console keys
    a new chat's llama.cpp on that same origin -- so the refused test blocks
    the chat that would send there."""
    from tldw_chatbook.Chat import console_session_settings as session_settings
    from tldw_chatbook.Chat.console_provider_endpoints import (
        DEFAULT_LLAMACPP_BASE_URL,
    )
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )
    from tldw_chatbook.Chat.provider_test_evidence import shared_connection_evidence

    tested: list[ProviderDraftIdentity] = []

    async def refused(identity):
        tested.append(identity)
        return ProviderProbeResult("unreachable", (), "connection_refused")

    app = ModalHarness()
    app.app_config = {"api_settings": {"llama_cpp": {}}}
    settings = session_settings.build_target_default_console_session_settings(
        app.app_config, "llama_cpp", "model-a"
    )
    assert settings.base_url is None  # Nothing saved anywhere.
    modal = ConsoleSettingsModal(
        settings=settings,
        app_config=app.app_config,
        providers_models={"llama_cpp": ["model-a"]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
        connection_tester=refused,
    )
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        prefill = modal.query_one("#console-settings-base-url", Input).value
        modal.query_one("#console-settings-model-discover", Button).press()
        await _settled(modal, pilot)

    readiness = session_settings.build_console_settings_readiness(
        settings,
        app_config=app.app_config,
        connection_evidence=shared_connection_evidence(lambda: app),
    )
    default = canonical_connection_identity("llama_cpp", DEFAULT_LLAMACPP_BASE_URL)
    assert prefill == DEFAULT_LLAMACPP_BASE_URL
    assert [identity.connection_identity for identity in tested] == [default]
    assert readiness.blocker == "endpoint_unreachable"
    assert readiness.connection.connection_identity == default


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "notified"),
    [("connection_refused", True), ("timeout", True), ("unauthorized", False)],
)
async def test_retry_says_still_unreachable_only_for_a_server_still_down(
    monkeypatch, category, notified
):
    """Review M-1: "Start it, then retry" after a rejected key contradicted
    the Console's own "Configure API key" recovery."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules import connection_probe

    async def probe(_identity, *, app_config=None):
        return ProviderProbeResult("unreachable", (), category)

    monkeypatch.setattr(connection_probe, "probe_console_connection", probe)
    notes: list[str] = []
    app = SimpleNamespace(notify=lambda message, **_kwargs: notes.append(message))

    await connection_probe._retry(app, _vllm_identity("sk-good"), "vLLM", {})

    assert notes == (
        ["vLLM is still unreachable. Start it, then retry."] if notified else []
    )


def _timed_out_openai(app):
    """OpenAI whose explicit Settings 't' key check timed out, as the Console reads it."""
    from tldw_chatbook.Chat import console_session_settings as session_settings
    from tldw_chatbook.Chat.provider_test_evidence import (
        ProviderTestEvidenceStore,
        shared_connection_evidence,
    )

    config = {"api_settings": {"openai": {"api_key": "sk-cloud"}}}
    settings = session_settings.build_target_default_console_session_settings(
        config, "openai", "gpt-5.1"
    )
    identity = session_settings.console_send_connection(settings, app_config=config)
    store = ProviderTestEvidenceStore(lambda: app)
    store.settle(store.begin(identity), ProviderProbeResult("unreachable", (), "timeout"))

    def readiness(_provider=None, _model=None):
        return session_settings.build_console_settings_readiness(
            settings,
            app_config=config,
            connection_evidence=shared_connection_evidence(lambda: app),
        )

    return settings, readiness


@pytest.mark.asyncio
async def test_a_cloud_key_check_failure_retries_in_settings_not_chat_settings():
    """TASK-33005 final review I-4: a timed-out OpenAI key check has no Console
    probe (D2), so Retry connection opens Providers & Models at OpenAI and says
    to press t -- never Chat settings, which cannot test a cloud provider."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules.connection_probe import (
        retry_console_connection,
    )
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    notes: list[str] = []
    app = SimpleNamespace(notify=lambda message, **_kwargs: notes.append(message))
    settings, readiness = _timed_out_openai(app)
    assert readiness().recovery_action == "retry_connection"
    posted: list[object] = []
    chat_settings: list[object] = []

    async def open_chat_settings(**kwargs):
        chat_settings.append(kwargs)

    screen = SimpleNamespace(
        app=app,
        _active_console_settings_readiness=lambda: (settings, readiness()),
        _console_default_readiness=readiness,
        _open_console_settings=open_chat_settings,
        post_message=posted.append,
    )

    await retry_console_connection(screen)

    assert chat_settings == []
    [message] = posted
    assert isinstance(message, NavigateToScreen)
    assert message.screen_context["category"] == "providers-models"
    assert message.screen_context["provider"] == "openai"
    assert message.screen_context["model"] == "gpt-5.1"
    assert notes == ["Press t to test OpenAI again."]
