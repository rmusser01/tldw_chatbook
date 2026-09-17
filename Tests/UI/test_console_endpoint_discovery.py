"""Named endpoint discovery through the mounted parent settings modal."""

import asyncio
from dataclasses import replace

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_session_settings import ModalHarness
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
    return ConsoleSettingsModal(
        settings=ConsoleSessionSettings(
            provider=provider, model="model-a", base_url="http://127.0.0.1:9099"
        ),
        app_config=app.app_config,
        providers_models={"llama_cpp": ["model-a"]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
        connection_tester=tester,
    )


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
        before = str(modal.query_one("#console-settings-streaming", Button).label)
        modal.query_one("#console-settings-streaming", Button).press()
        await pilot.pause()
        assert (
            str(modal.query_one("#console-settings-streaming", Button).label) != before
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
            modal.query_one(
                "#console-settings-provider", Select
            ).value = "custom-ep:two"
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
            assert modal._current_model_value() == "current-model"
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
@private_profile_test
async def test_create_endpoint_with_live_controller_rebase_settles(
    request, monkeypatch
):
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    calls = []

    async def connection(identity, *, app_config):
        from tldw_chatbook.Chat.custom_endpoint_registry import entry_for

        calls.append(
            (
                identity.custom_endpoint_id,
                entry_for(app_config, "custom-ep:new-live-endpoint") is not None,
            )
        )
        return ProviderProbeResult("reachable", ("live-served-model",))

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
        template.query_one("#endpoint-template-name", Input).value = "New live endpoint"
        template.query_one(
            "#endpoint-template-url", Input
        ).value = "http://127.0.0.1:9999"
        await pilot.pause()
        template.query_one("#endpoint-template-create", Button).press()
        for _ in range(30):
            await pilot.pause(0.02)
            if harness.screen is modal:
                break
        assert harness.screen is modal
        evidence = await _settled(modal, pilot)
        assert evidence.endpoint == "reachable"
        assert evidence.model_ids == ("live-served-model",)
        assert calls == [("custom-ep:new-live-endpoint", True)]
        assert modal._current_model_value() == "live-served-model"
        assert (
            modal._current_draft_discovery_identity().provider_key
            == "custom-ep:new-live-endpoint"
        )


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
        assert not modal._current_discovered_model_ids


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
        modal.query_one(
            "#console-settings-provider", Select
        ).value = "custom-ep:gpu-node"
        await pilot.pause()
        assert modal._active_provider == "custom-ep:gpu-node"
        picker = modal.query_one("#console-settings-model-picker")
        picker.set_model_value("registry-model")
        picker.post_message(picker.ModelSelected("registry-model"))
        await pilot.pause()
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
async def test_stale_model_adapter_event_cannot_undo_discovered_selection():
    async def connection(_identity):
        return ProviderProbeResult("reachable", ("served-model",))

    app = ModalHarness()
    modal = _modal(app, tester=connection)
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-model-discover", Button).press()
        await _settled(modal, pilot)
        select = modal.query_one("#console-settings-model-select", Select)
        assert select.value == "served-model"
        select.post_message(Select.Changed(select, "model-a"))
        await pilot.pause()
        assert modal._current_model_value() == "served-model"
        assert modal._current_model_discovery_matches_current_draft()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["model", "endpoint"])
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
        if adapter == "model":
            select = modal.query_one("#console-settings-model-select", Select)
            select.post_message(Select.Changed(select, select.value))
        else:
            endpoint = modal.query_one("#console-settings-base-url", Input)
            endpoint.post_message(Input.Changed(endpoint, endpoint.value))
        await pilot.pause()
        assert modal._active_connection_probe_token is not None
        release.set()
        evidence = await _settled(modal, pilot)
        assert evidence.endpoint == "reachable"
