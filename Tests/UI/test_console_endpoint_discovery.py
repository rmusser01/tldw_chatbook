"""Named endpoint discovery through the mounted parent settings modal."""

import asyncio
from dataclasses import replace

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_session_settings import ModalHarness
from Tests.UI.test_console_settings_core_first import (
    CoreFirstHarness,
    _open,
    _settings,
)
from Tests.UI.test_console_settings_model_change import (
    pick,
    pick_modal,
    pick_mode_opener,
    press_change,
    real_rebase,
    settle,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    build_target_default_console_session_settings,
    console_send_connection,
)
from tldw_chatbook.Chat.provider_catalog import SAVED_LIST_NOTE
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderDraftIdentity,
    ProviderProbeResult,
    ProviderTestEvidence,
    provider_connection_evidence,
)
from tldw_chatbook.UI.Console_Modules import connection_probe
from tldw_chatbook.Widgets.Console.console_endpoint_template_modal import (
    ConsoleEndpointTemplateModal,
)
from tldw_chatbook.Widgets.Console.console_model_popover import (
    switcher_readiness_words,
)
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    CREATED_ENTRY_PICK_LATER_COPY,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal

# Bound at import: Tests/conftest.py's ``_no_switcher_probe`` replaces the
# module attribute for every test, so a test that runs the switcher's own
# probe sets this back (as Tests/UI/test_console_switcher_local_probe.py does).
_REAL_SWITCHER_PROBER = connection_probe.switcher_connection_prober


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
        identity = modal._current_draft_discovery_identity()
        assert identity is not None
        assert identity.provider_key == "llama_cpp"
        assert identity.entry_id == provider_id

        # Creation persists the entry, but Cancel leaves the conversation on
        # its original provider. Selecting the new entry must run the real
        # quick-picker rebase before Apply commits the exact registry ID.
        # TASK-33003.5: the switch onto the entry is an unapplied edit, so
        # closing asks first; Discard is the choice that closes unchanged.
        # Esc: Cancel shows only with the recovery actions (capture finding 20).
        await pilot.press("escape")
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
        # The listing's served model is a row before anything is typed.
        served = ("custom-ep:command-box", "served-z")
        for _ in range(30):
            await pilot.pause(0.02)
            pairs = {(row.provider, row.model) for row in picker._rows}
            if served in pairs:
                break
        assert served in pairs, pairs
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

    Seed a saved entry directly to exercise the controller rebase and live
    commit independently of endpoint creation.
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
        assert str(status.content) == f"1 model listed. {SAVED_LIST_NOTE}"


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


def _stub_llama_network(
    monkeypatch, server: dict[str, bool], listings=None
) -> list[str]:
    """Replace only the network under the real probe (TASK-33005.2).

    Everything above it -- the modal's tester, the Console's retry, the
    category mapping -- is production code. Every server lists model-a, or
    what ``listings`` names for its port (None: refused). Returns the probed
    URLs.
    """
    import errno

    import httpx

    import tldw_chatbook.UI.Screens.settings_endpoint_probe as probe_module

    real_probe = probe_module.probe_settings_endpoint
    probed: list[str] = []

    async def network(request: httpx.Request) -> httpx.Response:
        models = (listings or {}).get(request.url.port, ("model-a",))
        if not server["up"] or models is None:
            raise httpx.ConnectError("refused") from OSError(
                errno.ECONNREFUSED, "refused"
            )
        return httpx.Response(200, json={"data": [{"id": m} for m in models]})

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
        await pilot.press("escape")  # Cancel shows only with recovery actions
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
        provider_connection_evidence,
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

        snapshot = console._console_credential_snapshot
        console._console_sync_maintenance_close_admission()
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
        assert not refreshes
        assert console._console_credential_snapshot == snapshot
        console._console_sync_maintenance_resume()
        for _ in range(3):
            console._poll_console_credential_readiness()
        assert len(refreshes) == 1  # Counted before any other tick can run.

        await pilot.pause()
        assert _rail_text(console, "#console-settings-readiness-row") == (
            "Not ready · timed out"
        )

        from tldw_chatbook import config
        from tldw_chatbook.Backup_Recovery import storage_admission

        identity = console._active_console_settings_readiness()[1].connection
        assert identity is not None
        assert console._console_config_snapshot_is_disk_loaded(
            console._provider_readiness_app_config()
        )
        store.settle(
            store.begin(identity), ProviderProbeResult("reachable", ("model-a",))
        )
        snapshot = console._console_credential_snapshot
        pauses, control_results = [], []
        real_control = console._sync_console_control_bar

        def summary_then_pause():
            real_sync()
            pauses.append(storage_admission._begin_local_pause())

        def checked_control():
            result = real_control()
            control_results.append(result)
            return result

        console._sync_console_settings_summary = summary_then_pause
        console._sync_console_control_bar = checked_control
        try:
            console._poll_console_credential_readiness()
            assert len(pauses) == 1
            assert control_results == [False]
            assert console._console_credential_snapshot == snapshot
        finally:
            console._sync_console_settings_summary = real_sync
            console._sync_console_control_bar = real_control
            for pause in reversed(pauses):
                pause.resume()
        console._poll_console_credential_readiness()
        assert (
            console._console_credential_snapshot[2]
            == provider_connection_evidence(harness).version
        )

        # A genuine cold reader still defers at native storage admission.
        snapshot = console._console_credential_snapshot
        store.settle(
            store.begin(identity), ProviderProbeResult("unreachable", (), "timeout")
        )
        pause = storage_admission._begin_local_pause()
        try:
            config._invalidate_config_caches()
            assert not console._console_sync_maintenance_paused
            console._poll_console_credential_readiness()
            assert console._console_credential_snapshot == snapshot
        finally:
            pause.resume()
        console._poll_console_credential_readiness()
        assert (
            console._console_credential_snapshot[2]
            == provider_connection_evidence(harness).version
        )

        # A cold lookup cannot adopt a different selected configuration source.
        import os

        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        selected = os.environ["TLDW_CONFIG_PATH"]
        snapshot = console._console_credential_snapshot
        os.environ["TLDW_CONFIG_PATH"] = selected + ".other"
        try:
            with pytest.raises(RecoveryRequired):
                console._poll_console_credential_readiness()
            assert console._console_credential_snapshot == snapshot
        finally:
            os.environ["TLDW_CONFIG_PATH"] = selected
        console._poll_console_credential_readiness()
        assert console._console_credential_snapshot == snapshot


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
    from tldw_chatbook.UI.Navigation.conversation_settings_navigation import (
        ConsoleProviderReturnIntent,
    )
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        PendingHandoffStore,
    )

    notes: list[str] = []
    app = SimpleNamespace(notify=lambda message, **_kwargs: notes.append(message))
    settings, readiness = _timed_out_openai(app)
    assert readiness().recovery_action == "retry_connection"
    posted: list[object] = []
    chat_settings: list[object] = []

    async def open_chat_settings(**kwargs):
        chat_settings.append(kwargs)

    handoffs = PendingHandoffStore()
    screen = SimpleNamespace(
        app=app,
        app_instance=SimpleNamespace(pending_handoffs=handoffs),
        _active_console_settings_readiness=lambda: (settings, readiness()),
        _console_default_readiness=readiness,
        _ensure_console_chat_store=lambda: SimpleNamespace(
            active_session_id="session-1"
        ),
        _open_console_settings=open_chat_settings,
        post_message=posted.append,
    )

    await retry_console_connection(screen)

    assert chat_settings == []
    [message] = posted
    assert isinstance(message, NavigateToScreen)
    # TASK-33008.1 (R32): the retry leaves with a way back to this chat; a
    # key check names no field, so Settings puts focus on the return.
    assert message.screen_context == {
        "category": "providers-models",
        "provider": "openai",
        "model": "gpt-5.1",
        "field": None,
        "return_revision": 1,
        "session_id": "session-1",
    }
    claim = handoffs.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert claim is not None
    assert claim.value == ConsoleProviderReturnIntent("session-1")
    assert notes == ["Press t to test OpenAI again."]


async def _pick_mode_pairs(pilot, app, modal) -> set[tuple[str, str]]:
    """Press Change, read the pairs pick mode offers, and Esc back."""
    switcher = await press_change(pilot, app, modal)
    pairs = {(row.provider, row.model) for row in switcher._rows}
    await pilot.press("escape")
    await settle(pilot, app)
    assert app.screen is modal
    return pairs


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_a_listing_is_offered_in_pick_mode_only_for_the_endpoint_that_listed_it():
    """TASK-33006.19: Test connection's listing is kept with the endpoint it
    listed. Once the Endpoint changes, pick mode no longer offers what the
    old one served; back on the endpoint that listed it, it is offered again."""

    async def connection(_identity):
        return ProviderProbeResult("reachable", ("model-a", "served-y"))

    app = CoreFirstHarness()
    modal = pick_modal(app, _settings(), connection_tester=connection)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        modal.query_one("#console-settings-model-discover", Button).press()
        await settle(pilot, app)
        served = ("llama_cpp", "served-y")
        assert served in await _pick_mode_pairs(pilot, app, modal)

        base_url = modal.query_one("#console-settings-base-url", Input)
        base_url.value = "http://127.0.0.1:9100"
        await settle(pilot, app)
        assert served not in await _pick_mode_pairs(pilot, app, modal)

        base_url.value = "http://127.0.0.1:9099"
        await settle(pilot, app)
        assert served in await _pick_mode_pairs(pilot, app, modal)


async def _list_served_y(pilot, app, modal) -> None:
    """Press Test connection; it lists model-a and served-y."""
    modal.query_one("#console-settings-model-discover", Button).press()
    await settle(pilot, app)


def _served_y_modal(app, settings, **kwargs):
    async def connection(_identity):
        return ProviderProbeResult("reachable", ("model-a", "served-y"))

    return pick_modal(app, settings, connection_tester=connection, **kwargs)


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_a_listing_is_not_offered_once_its_pick_restores_another_endpoint():
    """TASK-33006.19 review round 1: pick mode asks where a pick lands. Back
    from OpenAI, a pick of llama.cpp·served-y restores that pair's edited
    Endpoint (:9100), which never listed served-y, so it is not offered."""
    app = CoreFirstHarness()
    modal = _served_y_modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        await _list_served_y(pilot, app, modal)
        await pick(pilot, app, modal, "served-y")
        assert modal._current_model_value() == "served-y"
        base_url = modal.query_one("#console-settings-base-url", Input)
        base_url.value = "http://127.0.0.1:9100"
        await settle(pilot, app)
        await pick(pilot, app, modal, "gpt-5")
        assert modal._draft.settings.provider == "openai"

        assert ("llama_cpp", "served-y") not in await _pick_mode_pairs(
            pilot, app, modal
        )


#: TASK-35190: this chat's own Endpoint, a second local server beside the
#: configured :9099 (CoreFirstHarness, ``_persisted_console_app``).
SESSION_URL = "http://127.0.0.1:9200"


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_a_listing_is_offered_only_while_its_pick_keeps_the_listed_endpoint():
    """TASK-35190 AC#2 (until then this pinned TASK-33006.19's reset): a chat
    on :9200 while the config says :9099 lists served-y at :9200. A pick of
    it keeps the unedited Endpoint on :9200, so pick mode offers it without
    retyping the Endpoint, and picking it keeps the chat on :9200."""
    app = CoreFirstHarness()
    modal = _served_y_modal(app, replace(_settings(), base_url=SESSION_URL))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        base_url = modal.query_one("#console-settings-base-url", Input)
        assert base_url.value == SESSION_URL
        await _list_served_y(pilot, app, modal)
        assert ("llama_cpp", "served-y") in await _pick_mode_pairs(pilot, app, modal)

        await pick(pilot, app, modal, "served-y")
        assert modal._current_model_value() == "served-y"
        assert base_url.value == SESSION_URL
        assert modal._draft.settings.base_url == SESSION_URL


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detour",
    [None, ("gpt-5", "openai", "gpt-5"), ("GPU box", "custom-ep:gpu-box", "model-a")],
    ids=["direct", "after-openai", "after-an-entry"],
)
async def test_a_pick_on_the_chats_own_provider_keeps_its_session_endpoint(detour):
    """TASK-35190 AC#1 and AC#3: a chat on :9200 while the config says :9099.
    A pick of the saved model-b keeps :9200, also after a pick onto another
    provider in between. That pick takes the other provider's own endpoint
    (OpenAI's, the GPU box entry's URL), never :9200."""
    app = CoreFirstHarness()
    modal = pick_modal(app, replace(_settings(), base_url=SESSION_URL))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        base_url = modal.query_one("#console-settings-base-url", Input)
        if detour is not None:
            find, provider, model = detour
            await pick(pilot, app, modal, find)
            own = build_target_default_console_session_settings(
                app.app_config, provider, model
            ).base_url
            assert (modal._draft.settings.provider, modal._current_model_value()) == (
                provider,
                model,
            )
            assert modal._draft.settings.base_url == own != SESSION_URL
        await pick(pilot, app, modal, "model-b")
        assert (modal._draft.settings.provider, modal._current_model_value()) == (
            "llama_cpp",
            "model-b",
        )
        assert base_url.value == SESSION_URL
        assert modal._draft.settings.base_url == SESSION_URL


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("detour", [False, True], ids=["direct", "after-openai"])
async def test_a_pick_keeps_the_endpoint_its_provider_had_last(detour):
    """TASK-35190 review round 1: an unseen pick takes the Endpoint of the
    most recently remembered draft on its provider. On :9200, served-y is
    picked, then :9300 typed for it, then model-a picked back (restoring
    :9200). A pick of model-b then keeps :9200, also after a pick of OpenAI
    in between: a re-remembered pair moves to the end of the drafts, so the
    older served-y draft and its :9300 edit are not the latest."""

    async def connection(_identity):
        return ProviderProbeResult("reachable", ("model-a", "model-b", "served-y"))

    app = CoreFirstHarness()
    settings = replace(_settings(), base_url=SESSION_URL)
    modal = pick_modal(app, settings, connection_tester=connection)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        base_url = modal.query_one("#console-settings-base-url", Input)
        await _list_served_y(pilot, app, modal)
        await pick(pilot, app, modal, "served-y")
        assert base_url.value == SESSION_URL
        base_url.value = "http://127.0.0.1:9300"
        await settle(pilot, app)
        await pick(pilot, app, modal, "model-a")
        assert modal._current_model_value() == "model-a"
        assert base_url.value == SESSION_URL
        if detour:
            await pick(pilot, app, modal, "gpt-5")
        await pick(pilot, app, modal, "model-b")
        assert (modal._draft.settings.provider, modal._current_model_value()) == (
            "llama_cpp",
            "model-b",
        )
        assert base_url.value == SESSION_URL
        assert modal._draft.settings.base_url == SESSION_URL
        assert not modal._draft.endpoint_draft.dirty


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    ("provider", "kept"),
    [("llama_cpp", True), ("custom-ep:gpu-box", False)],
    ids=["server", "registry-entry"],
)
def test_an_unseen_pick_keeps_the_chats_url_except_on_a_registry_entry(provider, kept):
    """TASK-35190: the rebaser keeps the URL the chat holds for an unseen
    model on its own provider, but an entry's URL is its endpoint (ADR-146):
    a pick of another of its models takes the entry's saved URL, never one
    the chat's settings still hold."""
    from tldw_chatbook.Chat.console_settings_apply import FULL_MODEL_DEFAULT_FIELDS

    app_config = CoreFirstHarness().app_config
    chat_url = "http://127.0.0.1:9201"
    state = ConsoleSettingsModal._initial_full_draft(
        ConsoleSessionSettings(provider=provider, model="model-a", base_url=chat_url)
    )
    rebased = real_rebase(
        state,
        provider=provider,
        model="model-b",
        app_config=app_config,
        exposed_fields=FULL_MODEL_DEFAULT_FIELDS,
    )
    entry_url = build_target_default_console_session_settings(
        app_config, provider, "model-b"
    ).base_url
    expected = chat_url if kept else entry_url
    assert expected and rebased.settings.base_url == expected


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("typed", "listed", "detour", "offered"),
    [
        (False, True, False, {"model-a", "served-y"}),
        (True, True, False, {"model-a", "served-y"}),
        (True, False, False, {"model-a", "model-b"}),
        (False, False, True, {"model-a", "model-b"}),
    ],
    ids=["chats-endpoint", "typed", "typed-unlisted", "after-openai"],
)
async def test_a_listing_at_this_chats_endpoint_replaces_the_configured_servers(
    typed, listed, detour, offered
):
    """TASK-35190, for TASK-33005.12's listing: pick mode reads the configured
    :9099's readiness, as these tests' opener does, and :9099's own listing
    (model-a, only-at-9099) is in the shared evidence. A pick on llama.cpp
    lands on this chat's Endpoint, :9200, the chat's own or one typed, so
    :9099's only-at-9099 is never offered. llama.cpp's rows are what Test
    connection listed at :9200 (model-a, served-y), without the saved
    model-b, which no server listed (TASK-33005.12 AC#1); with no listing
    there they are the saved models (its AC#2). After a pick of OpenAI in
    between (review round 1), a llama.cpp pick still lands on :9200, by the
    rebaser's own rule, so :9099's listing is still not llama.cpp's rows."""
    app = CoreFirstHarness()
    owner = provider_connection_evidence(app)
    configured = console_send_connection(
        build_target_default_console_session_settings(
            app.app_config, "llama_cpp", None
        ),
        app_config=app.app_config,
    )
    listing = ("model-a", "only-at-9099")
    owner.publish(ProviderTestEvidence(configured, "reachable", listing), order=1)
    picker = pick_mode_opener(app, app.app_config, connection_evidence=owner)
    assert (
        await _llama_models_listed_at_9200(
            app, typed=typed, listed=listed, detour=detour, model_picker=picker
        )
        == offered
    )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_a_listing_at_this_chats_endpoint_replaces_a_loaded_catalog():
    """TASK-35190, for the Wave 6 integration review round 2's case: with no
    listing of the configured :9099, llama.cpp's rows were its loaded catalog
    (model-a, model-b), served-y after it. A pick lands on :9200, so they are
    what Test connection listed there; model-b, which :9200 did not list, is
    not offered (TASK-33005.12 AC#1)."""
    assert await _llama_models_listed_at_9200(CoreFirstHarness()) == {
        "model-a",
        "served-y",
    }


async def _llama_models_listed_at_9200(
    app, *, typed=False, listed=True, detour=False, **kwargs
) -> set[str]:
    """Chat settings on :9200, this chat's Endpoint or (``typed``) one typed
    over the configured :9099, lists served-y there (unless not ``listed``)
    and picks OpenAI's gpt-5 (if ``detour``); return the llama.cpp models
    Change offers."""
    settings = _settings() if typed else replace(_settings(), base_url=SESSION_URL)
    modal = _served_y_modal(app, settings, **kwargs)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        if typed:
            modal.query_one("#console-settings-base-url", Input).value = SESSION_URL
            await settle(pilot, app)
        if listed:
            await _list_served_y(pilot, app, modal)
        if detour:
            await pick(pilot, app, modal, "gpt-5")
            assert modal._draft.settings.provider == "openai"
        pairs = await _pick_mode_pairs(pilot, app, modal)
    return {model for provider, model in pairs if provider == "llama_cpp"}


async def _probed(switcher, pilot, provider):
    """Wait for the switcher's own probe of ``provider``; return its readiness."""
    for _ in range(60):
        readiness = switcher._readiness.get(provider)
        if readiness is not None and readiness.evidence is not None:
            return readiness
        await pilot.pause(0.05)
    pytest.fail(f"Switch model never probed {provider}")


def _on_9200(monkeypatch, listed, configured=None):
    """Run the switcher's real probe over a network where the configured
    :9099 lists ``configured`` and this chat's :9200 lists ``listed`` (None:
    refuses); return the probed URLs."""
    monkeypatch.setattr(
        connection_probe, "switcher_connection_prober", _REAL_SWITCHER_PROBER
    )
    listings = {9099: configured, 9200: listed}
    return _stub_llama_network(monkeypatch, {"up": True}, listings)


async def _console_on_9200(harness, pilot):
    """The Console, with its llama.cpp chat moved to :9200; return it."""
    from Tests.UI.test_destination_shells import _wait_for_selector

    console = harness.screen
    await _wait_for_selector(console, pilot, "#console-settings-summary")
    store = console._ensure_console_chat_store()
    console._session._replace_active_console_session_settings(
        replace(store.session_settings(store.active_session_id), base_url=SESSION_URL)
    )
    return console


@pytest.mark.asyncio
@private_profile_test
async def test_switch_model_keeps_the_chats_session_endpoint_and_saves_none(
    request, monkeypatch
):
    """TASK-35190 AC#4: Switch model (Alt+M) on a llama.cpp chat on :9200
    while the configured :9099 is down. A pick on llama.cpp lands on :9200,
    so Switch model's readiness and its rows (the real probe, over a stubbed
    network) read :9200, never :9099 (review round 1): llama.cpp is READY and
    its rows are what :9200 listed, served-y and not the saved model-c.
    Picked after a detour through vLLM, model-b keeps :9200, as Chat settings
    does (AC#1). New chats use :9099, so the open probes it too (review round
    2), and Make default for new chats on a READY row is refused in words
    that name :9099, as is Enter in a list reopened from Settings, where it
    sets the default too while none works. Nothing is saved; the save itself
    (no Endpoint) is
    test_make_default_for_new_chats_reads_the_configured_endpoint's."""
    import tomllib

    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _drain_settings_tasks,
        _highlight,
        _open_provider_popover,
        _persisted_console_app,
    )
    from tldw_chatbook.config import get_cli_config_path
    from tldw_chatbook.UI.Console_Modules.model_switcher import reopen_model_switcher

    listed = ("model-a", "model-b", "served-y")
    probed = _on_9200(monkeypatch, listed)
    app = _persisted_console_app()
    app.providers_models = {
        "llama_cpp": ["model-a", "model-b", "model-c"],
        "vllm": ["vendor/model:b"],
    }
    config_path = get_cli_config_path()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = await _console_on_9200(harness, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id

        quick = await _open_provider_popover(console, harness, pilot)
        llama = await _probed(quick, pilot, "llama_cpp")
        assert (llama.blocker, llama.evidence.model_ids) == (None, listed)
        assert quick._models_for("llama_cpp") == listed
        assert (await _highlight(quick, pilot, "vllm")).provider == "vllm"
        row = await _highlight(quick, pilot, "model-b")
        assert (row.kind, row.provider, row.model) == ("pair", "llama_cpp", "model-b")
        quick.query_one("#console-popover-apply", Button).press()
        await pilot.pause()
        await _drain_settings_tasks(app)
        assert harness.screen is console
        applied = store.session_settings(session_id)
        assert (applied.provider, applied.model, applied.base_url) == (
            "llama_cpp",
            "model-b",
            SESSION_URL,
        )

        config = console._provider_readiness_app_config()
        configured = console_send_connection(
            build_target_default_console_session_settings(config, "llama_cpp", None),
            app_config=config,
        )
        quick = await _open_provider_popover(console, harness, pilot)
        for _ in range(60):  # The open's probe of :9099, by no hand publish.
            if provider_connection_evidence(harness).evidence_for(configured):
                break
            await pilot.pause(0.05)
        row = await _highlight(quick, pilot, "served-y")
        assert (row.kind, row.provider, row.model) == ("pair", "llama_cpp", "served-y")
        quick.action_make_new_chat_default()
        await pilot.pause()
        assert harness.screen is quick
        refusal = (
            "Unavailable for new chats (http://127.0.0.1:9099): llama.cpp failed "
            "its last connection test (connection refused)."
        )
        assert quick._error == refusal
        llama = quick._readiness["llama_cpp"]
        assert llama.blocker is None
        assert llama.connection.connection_identity[1] == SESSION_URL
        assert quick._models_for("llama_cpp") == listed

        await pilot.press("escape")
        await pilot.pause()
        assert harness.screen is console
        reopen_model_switcher(console, "llama_cpp", "model-b")
        await pilot.pause()
        await harness.workers.wait_for_complete()
        await pilot.pause()
        quick = harness.screen
        assert quick._enter_sets_default
        row = await _highlight(quick, pilot, "model-b")
        assert (row.kind, row.provider, row.model) == ("pair", "llama_cpp", "model-b")
        quick._activate(row)  # Enter
        await pilot.pause()
        assert harness.screen is quick
        assert quick._error == refusal
        applied = store.session_settings(session_id)
        assert (applied.model, applied.base_url) == ("model-b", SESSION_URL)
    assert any(":9099" in url for url in probed)
    saved = config_path.read_text(encoding="utf-8")
    assert tomllib.loads(saved)["chat_defaults"]["model"] == "model-a"
    assert tomllib.loads(saved)["api_settings"]["llama_cpp"]["api_url"] == (
        "http://127.0.0.1:9099"
    )
    assert "127.0.0.1:9200" not in saved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_url",
    ["http://127.0.0.1:9099", "http://8.8.8.8:9099"],
    ids=["local", "public"],
)
@private_profile_test
async def test_make_default_for_new_chats_reads_the_configured_endpoint(
    request, monkeypatch, configured_url
):
    """TASK-35190 review round 2: Switch model on a llama.cpp chat whose :9200
    is down, while the configured server lists model-a and model-b. The rows
    read :9200, so llama.cpp is under NEEDS SETUP, but new chats use the
    configured server, so Make default for new chats on model-b saves the
    pair, and no Endpoint (AC#4). The open probes the configured server too,
    under the switcher's own rules: a local one is checked, a public one is
    never contacted."""
    import tomllib

    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _drain_settings_tasks,
        _highlight,
        _open_provider_popover,
        _persisted_console_app,
    )
    from tldw_chatbook.config import get_cli_config_path
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

    probed = _on_9200(monkeypatch, None, configured=("model-a", "model-b"))
    app = _persisted_console_app()
    app.providers_models = {"llama_cpp": ["model-a", "model-b"]}
    assert SettingsConfigAdapter().save_sections(
        {"api_settings.llama_cpp": {"api_url": configured_url}}
    )
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = await _console_on_9200(harness, pilot)
        config = console._provider_readiness_app_config()
        assert config["api_settings"]["llama_cpp"]["api_url"] == configured_url
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id

        quick = await _open_provider_popover(console, harness, pilot)
        assert (await _probed(quick, pilot, "llama_cpp")).blocker is not None
        await harness.workers.wait_for_complete()  # Both of the open's probes.
        row = await _highlight(quick, pilot, "model-b")
        assert (row.kind, row.provider, row.model) == ("setup", "llama_cpp", "model-b")
        quick.action_make_new_chat_default()
        await pilot.pause()
        await _drain_settings_tasks(app)
        assert harness.screen is console
        applied = store.session_settings(session_id)
        assert (applied.model, applied.base_url) == ("model-b", SESSION_URL)
    assert any(":9200" in url for url in probed)
    if configured_url.startswith("http://127."):
        assert any(configured_url in url for url in probed)
    assert not any("8.8.8.8" in url for url in probed)
    saved = get_cli_config_path().read_text(encoding="utf-8")
    assert tomllib.loads(saved)["chat_defaults"]["model"] == "model-b"
    assert tomllib.loads(saved)["api_settings"]["llama_cpp"]["api_url"] == (
        configured_url
    )
    assert "127.0.0.1:9200" not in saved


@pytest.mark.asyncio
@private_profile_test
async def test_pick_mode_reads_this_chats_endpoint_while_the_configured_one_is_down(
    request, monkeypatch
):
    """TASK-35190 review round 1: Chat settings on a llama.cpp chat on :9200
    while the configured :9099 is down. Pick mode, opened by the Console's
    own opener, reads llama.cpp at :9200, where a pick of it lands, so its
    probe never tries :9099: llama.cpp is READY with served-y, which Test
    connection listed at :9200, and picking it keeps :9200."""
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _persisted_console_app,
    )

    probed = _on_9200(monkeypatch, ("model-a", "served-y"))
    harness = _ConsoleFlowHarness(_persisted_console_app())
    async with harness.run_test(size=(211, 44)) as pilot:
        console = await _console_on_9200(harness, pilot)
        await console._open_console_settings(focus_model=False)
        await settle(pilot, harness)
        modal = harness.screen
        base_url = modal.query_one("#console-settings-base-url", Input)
        assert base_url.value == SESSION_URL
        await _list_served_y(pilot, harness, modal)

        switcher = await press_change(pilot, harness, modal)
        assert switcher._readiness["llama_cpp"].blocker is None
        assert ("llama_cpp", "served-y") in {
            (row.provider, row.model) for row in switcher._rows if row.kind == "pair"
        }
        await pilot.press(*"served-y")
        await settle(pilot, harness)
        await pilot.press("enter")
        await settle(pilot, harness)
        assert harness.screen is modal
        assert modal._current_model_value() == "served-y"
        assert base_url.value == SESSION_URL
        assert modal._draft.settings.base_url == SESSION_URL
    assert all(":9099" not in url for url in probed)


@pytest.mark.asyncio
@private_profile_test
async def test_a_settings_round_trip_keeps_the_chats_endpoint_for_its_provider(
    request, monkeypatch
):
    """TASK-35190 review round 1: Chat settings on a llama.cpp chat on :9200
    picks vLLM, and its draft is suspended (as a pick whose key is set in
    Settings suspends it) and reopened. The reopened draft still remembers
    the chat's own pair, so a pick of model-b keeps :9200, as it does with no
    round trip (AC#1)."""
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _persisted_console_app,
    )

    app = _persisted_console_app()
    app.providers_models = {
        "llama_cpp": ["model-a", "model-b"],
        "vllm": ["vendor/model:b"],
    }
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = await _console_on_9200(harness, pilot)
        await console._open_console_settings(focus_model=False)
        await settle(pilot, harness)
        modal = harness.screen
        await pick(pilot, harness, modal, "vendor/model:b")
        assert modal._draft.settings.provider == "vllm"
        snapshot = modal.capture_suspended_draft()
        await harness.pop_screen()
        await settle(pilot, harness)
        assert harness.screen is console

        await console._open_console_settings(suspended_draft=snapshot)
        await settle(pilot, harness)
        modal = harness.screen
        assert modal._draft.settings.provider == "vllm"
        await pick(pilot, harness, modal, "model-b")
        assert (modal._draft.settings.provider, modal._current_model_value()) == (
            "llama_cpp",
            "model-b",
        )
        assert modal.query_one("#console-settings-base-url", Input).value == (
            SESSION_URL
        )
        assert modal._draft.settings.base_url == SESSION_URL


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_a_created_entry_listed_under_the_close_guard_says_how_to_pick_it():
    """TASK-33006.25: Esc during a new entry's listing shows the unsaved-
    changes prompt, and pick mode cannot open over it. When the listing
    ends, a notice says the entry is saved and how to pick its model; Keep
    editing, then Change, offers what it served."""
    release = asyncio.Event()

    async def slow_listing(_identity):
        await release.wait()
        return ProviderProbeResult("reachable", ("served-x",))

    app = CoreFirstHarness()
    modal = pick_modal(app, _settings(), connection_tester=slow_listing)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        temperature = modal.query_one("#console-settings-temperature", Input)
        temperature.focus()
        temperature.value = "0.9"  # an unsaved edit, so Esc asks first
        await settle(pilot, app)
        modal.pick_created_endpoint("custom-ep:gpu-box")
        for _ in range(4):
            await pilot.pause()
        await pilot.press("escape")
        for _ in range(4):
            await pilot.pause()
        guard = modal.query_one("#console-settings-close-guard")
        assert guard.display, "precondition: the close guard is showing"

        release.set()
        await settle(pilot, app)

        assert app.screen is modal and guard.display
        assert [note.message for note in app._notifications] == [
            CREATED_ENTRY_PICK_LATER_COPY.format(name="GPU box")
        ]
        await pilot.press("escape")  # Keep editing
        await settle(pilot, app)
        assert not guard.display
        assert ("custom-ep:gpu-box", "served-x") in await _pick_mode_pairs(
            pilot, app, modal
        )
