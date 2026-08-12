"""TASK-388: switching the provider updates the dependent fields atomically.

The review saw the form assert a stale provider/model/readiness combination for
~1-3s after selecting a new provider. In the current code the provider
Select.Changed handler updates every dependent field synchronously, so there is
no window where the form shows the previous provider. This locks that atomicity.
"""

import time

import pytest
from textual.widgets import Input, Select, Static

from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
)
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.UI.Screens import settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
    SettingsEndpointProbeOutcome,
)


@pytest.mark.asyncio
async def test_provider_switch_updates_dependent_fields_with_no_stale_window():
    """Selecting a new provider flips readiness/source/model in the same tick."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4o"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": "fake-key-not-real"},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await pilot.pause()

        provider_select = screen.query_one("#settings-provider-value", Select)
        # Drive the real Select.Changed handler, then read the dependent fields
        # WITHOUT pumping the event loop -- any staleness would show here.
        screen.handle_provider_value_changed(
            Select.Changed(provider_select, "llama.cpp")
        )

        readiness = str(
            screen.query_one("#settings-provider-readiness", Static).renderable
        )
        source = str(
            screen.query_one("#settings-provider-source", Static).renderable
        )
        model_value = screen.query_one("#settings-model-value", Input).value

        # The dependent fields reflect the NEW provider immediately...
        assert "llama.cpp" in readiness
        assert "draft" in source.lower()
        # ...and never assert the previous provider/model combination.
        assert "gpt-4o" not in readiness
        assert model_value != "gpt-4o"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        ConfigMutationResult(False, False, "before_replace"),
        ConfigMutationResult(True, False, "cache_reload"),
    ],
    ids=["before-replace", "cache-reload"],
)
async def test_provider_save_partial_failure_keeps_memory_and_draft(
    monkeypatch, result
):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4o"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": "fake-key-not-real"},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    before = {
        "chat_defaults": dict(app.app_config["chat_defaults"]),
        "llama_cpp": dict(app.app_config["api_settings"]["llama_cpp"]),
    }
    calls = []

    def fake_writer(section_values, *, delete_keys=None):
        calls.append((section_values, delete_keys))
        return result

    monkeypatch.setattr(
        settings_screen_module,
        "apply_settings_mutation_to_cli_config",
        fake_writer,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)
        provider.value = "llama_cpp"
        screen.handle_provider_value_changed(Select.Changed(provider, "llama_cpp"))
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://127.0.0.1:8080/v1/models"
        model = screen.query_one("#settings-model-value", Input)
        model.value = "qwen"
        await pilot.pause()

        await pilot.click("#settings-save-category")

        assert SettingsCategoryId.PROVIDERS_MODELS in screen._settings_drafts
        save_copy = str(
            screen.query_one("#settings-provider-save-result", Static).renderable
        )
        assert "not fully applied" in save_copy.lower()
        assert "saved" not in save_copy.lower().replace("not fully applied", "")
        if result.file_replaced:
            assert "file was written" in save_copy.lower()
            assert "restart" in save_copy.lower()
        else:
            assert "file was not written" in save_copy.lower()
            assert "retry" in save_copy.lower()

    assert len(calls) == 1
    assert app.app_config["chat_defaults"] == before["chat_defaults"]
    assert app.app_config["api_settings"]["llama_cpp"] == before["llama_cpp"]


@pytest.mark.asyncio
async def test_provider_save_uses_one_atomic_writer_for_all_owned_values(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4o"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": "fake-key-not-real"},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "old"},
    }
    calls = []

    def fake_writer(section_values, *, delete_keys=None):
        calls.append((section_values, delete_keys))
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        settings_screen_module,
        "apply_settings_mutation_to_cli_config",
        fake_writer,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Select)
        provider.value = "llama_cpp"
        screen.handle_provider_value_changed(Select.Changed(provider, "llama_cpp"))
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://127.0.0.1:8080/v1/chat/completions"
        model = screen.query_one("#settings-model-value", Input)
        model.value = "qwen"
        await pilot.pause()

        await pilot.click("#settings-save-category")

    assert len(calls) == 1
    sections, deletes = calls[0]
    assert sections["chat_defaults"] == {"provider": "llama_cpp", "model": "qwen"}
    assert sections["api_settings.llama_cpp"] == {
        "api_url": "http://127.0.0.1:8080",
        "model": "qwen",
    }
    assert sections["provider_setup.confirmed"] == {"llama_cpp": True}
    assert deletes in ({}, None)
    assert app.app_config["chat_defaults"] == {"provider": "llama_cpp", "model": "qwen"}
    assert app.app_config["api_settings"]["llama_cpp"]["api_url"] == (
        "http://127.0.0.1:8080"
    )
    assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts


@pytest.mark.asyncio
async def test_exact_probe_evidence_survives_returned_model_selection_and_save(
    monkeypatch,
):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:8080", "model": "model-a"}
    }

    async def fake_probe(base_url, **kwargs):
        return SettingsEndpointProbeOutcome(
            state="reachable",
            summary="reachable (2 models)",
            model_ids=("model-a", "model-b"),
        )

    monkeypatch.setattr(settings_screen_module, "probe_settings_endpoint", fake_probe)
    monkeypatch.setattr(
        settings_screen_module,
        "apply_settings_mutation_to_cli_config",
        lambda section_values, *, delete_keys=None: ConfigMutationResult(
            True, True, None
        ),
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.action_settings_test_category()
        deadline = time.monotonic() + 4.0
        while (
            time.monotonic() < deadline
            and "endpoint reachable" not in screen._provider_test_result
        ):
            await pilot.pause(0.01)

        tested_identity = screen._provider_current_draft_identity()
        assert tested_identity is not None
        assert screen._provider_test_evidence_store.evidence_for(tested_identity) is not None

        model = screen.query_one("#settings-model-value", Input)
        model.value = "model-b"
        await pilot.pause()
        selected_identity = screen._provider_current_draft_identity()
        assert selected_identity == tested_identity
        assert screen._provider_test_evidence_store.evidence_for(selected_identity) is not None

        await pilot.click("#settings-save-category")

        saved_identity = screen._provider_current_draft_identity()
        assert saved_identity is not None
        rebound = screen._provider_test_evidence_store.evidence_for(saved_identity)
        assert rebound is not None
        assert rebound.model_ids == ("model-a", "model-b")
