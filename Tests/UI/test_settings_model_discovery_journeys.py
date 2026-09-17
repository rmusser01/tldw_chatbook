"""Model discovery selection and async receipts belong to the active form."""

import asyncio

import pytest
from textual.widgets import Button, Input, Select, SelectionList, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_configuration_hub import (
    FakeSettingsModelDiscoveryScope,
    ModelDiscoveryResult,
    PersistenceResult,
    _discovered_model,
    _open_settings_category,
)
from Tests.UI.test_settings_provider_keyboard_journeys import (
    CATEGORY,
    ProviderSettingsHarness,
    _settle,
    _tab_to,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

MODELS = ("model-alpha", "model-beta")


def _app_scope(*, model="", gate=None, failure=False):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": model}
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://first.invalid/v1"},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    app.providers_models = {"OpenAI": [], "llama_cpp": []}
    scope = DiscoveryScope(gate=gate, failure=failure)
    app.llm_provider_catalog_scope_service = scope
    return app, scope


class DiscoveryScope(FakeSettingsModelDiscoveryScope):
    def __init__(self, *, gate=None, failure=False):
        super().__init__(
            result=ModelDiscoveryResult(
                provider="openai",
                provider_list_key="OpenAI",
                endpoint_fingerprint="fixture",
                status="success",
                models=tuple(_discovered_model(name) for name in MODELS),
            ),
            persistence_result=PersistenceResult(
                provider="openai",
                provider_list_key="OpenAI",
                status="saved",
                saved_model_ids=(MODELS[1],),
                message="Saved 1 discovered model.",
            ),
        )
        self.gate = gate
        self.failure = failure
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def discover_models(self, **kwargs):
        self.discover_calls.append(kwargs)
        if self.gate == "discover":
            self.started.set()
            await self.release.wait()
        if self.failure:
            raise OSError("fixture failure")
        return self.result

    async def clear_discovered_models(self, **kwargs):
        self.clear_calls.append(kwargs)
        if self.gate == "clear":
            self.started.set()
            await self.release.wait()
        if self.failure:
            raise OSError("fixture failure")

    async def persist_discovered_models_to_settings(self, **kwargs):
        self.persist_calls.append(kwargs)
        if self.gate == "save":
            self.started.set()
            await self.release.wait()
        if self.failure:
            raise OSError("fixture failure")
        return self.persistence_result


async def _open(host, pilot):
    await _open_settings_category(pilot, "#settings-category-providers-models")
    await _settle(host, pilot)
    return host.screen


def _list(screen):
    return screen.query_one("#settings-discovered-models-list", SelectionList)


def _status(screen):
    return str(screen.query_one("#settings-model-discovery-status", Static).renderable)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_discovery_keyboard_selection_survives_rebuild_and_save(theme, size):
    app, scope = _app_scope()
    host = ProviderSettingsHarness(app, "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = await _open(host, pilot)
        await _tab_to(host, pilot, "#settings-discover-provider-models")
        await pilot.press("enter")
        await _settle(host, pilot)
        assert _list(screen).option_count == 2
        assert not _list(screen).selected
        assert not scope.persist_calls
        await _tab_to(host, pilot, "#settings-discovered-models-list")
        await pilot.press("home", "down", "space")
        await _settle(host, pilot)
        assert _list(screen).selected == [MODELS[1]]
        assert screen._model_discovery_selected_model_ids == {MODELS[1]}
        # This is the existing pane-rebuild path used by form layout changes.
        screen.mutate_reactive(SettingsScreen.active_category)
        await _settle(host, pilot)
        assert _list(screen).selected == [MODELS[1]]
        await _tab_to(host, pilot, "#settings-save-discovered-provider-models")
        await pilot.press("enter")
        await _settle(host, pilot)
        assert scope.persist_calls == [
            {"mode": "local", "provider": "openai", "model_ids": [MODELS[1]]}
        ]
        assert app.providers_models == {"OpenAI": [MODELS[1]], "llama_cpp": []}
        assert screen.query_one("#settings-model-value", Input).value == MODELS[1]
        assert screen._category_has_unsaved_changes(CATEGORY)
        assert app.app_config["chat_defaults"]["model"] == ""
        assert "Saved 1" in _status(screen)
        await _tab_to(host, pilot, "#settings-clear-discovered-provider-models")
        await pilot.press("enter")
        await _settle(host, pilot)
        assert _list(screen).option_count == 0
        assert screen.query_one("#settings-model-value", Input).suggester is None
        assert app.providers_models["OpenAI"] == [MODELS[1]]
        assert "cleared" in _status(screen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["provider", "endpoint", "endpoint_roundtrip", "credential_env", "api_key"],
)
@pytest.mark.parametrize("failure", [False, True])
async def test_late_discovery_cannot_publish_into_a_changed_form(change, failure):
    app, scope = _app_scope(gate="discover", failure=failure)
    host = ProviderSettingsHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        screen._discover_provider_models_worker()
        try:
            await asyncio.wait_for(scope.started.wait(), timeout=5)
            if change == "provider":
                screen.query_one("#settings-provider-value", Select).value = "llama_cpp"
            elif change == "credential_env":
                screen.query_one(
                    "#settings-provider-credential-env-var", Input
                ).value = "FIXTURE_KEY"
            elif change == "api_key":
                screen.query_one(
                    "#settings-provider-api-key", Input
                ).value = "fixture-secret"
            else:
                screen.query_one(
                    "#settings-provider-endpoint-value", Input
                ).value = "https://second.invalid/v1"
            await pilot.pause()
            if change == "endpoint_roundtrip":
                screen.query_one(
                    "#settings-provider-endpoint-value", Input
                ).value = "https://first.invalid/v1"
                await pilot.pause()
            expected_status = _status(screen)
        finally:
            scope.release.set()
        await _settle(host, pilot)
        assert _list(screen).option_count == 0
        assert _status(screen) == expected_status
        assert not screen._model_discovery_models
        assert screen.query_one(
            "#settings-save-discovered-provider-models", Button
        ).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("change", ["provider", "endpoint_roundtrip"])
async def test_late_save_keeps_receipt_and_activation_out_of_a_changed_form(
    failure, change
):
    app, scope = _app_scope(gate="save")
    host = ProviderSettingsHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        await screen._discover_provider_models()
        _list(screen).select(MODELS[1])
        await pilot.pause()
        scope.failure = failure
        screen._save_selected_discovered_provider_models_worker()
        try:
            await asyncio.wait_for(scope.started.wait(), timeout=5)
            if change == "provider":
                screen.query_one("#settings-provider-value", Select).value = "llama_cpp"
                await pilot.pause()
            else:
                field = screen.query_one("#settings-provider-endpoint-value", Input)
                field.value = "https://second.invalid/v1"
                await pilot.pause()
                field.value = "https://first.invalid/v1"
                await pilot.pause()
            expected_model = screen.query_one("#settings-model-value", Input).value
            expected_status = _status(screen)
        finally:
            scope.release.set()
        await _settle(host, pilot)
        assert screen.query_one("#settings-model-value", Input).value == expected_model
        assert _status(screen) == expected_status
        assert app.providers_models["OpenAI"] == ([] if failure else [MODELS[1]])
        assert app.providers_models["llama_cpp"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["discover", "save", "clear"])
async def test_discovery_operations_report_failure_and_allow_retry(operation):
    app, scope = _app_scope(model="existing-active-model")
    host = ProviderSettingsHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = await _open(host, pilot)
        if operation != "discover":
            await screen._discover_provider_models()
            _list(screen).select(MODELS[1])
            await pilot.pause()
        run = {
            "discover": screen._discover_provider_models,
            "save": screen._save_selected_discovered_provider_models,
            "clear": screen._clear_discovered_provider_models,
        }[operation]
        scope.failure = True
        await run()
        await _settle(host, pilot)
        assert "OSError" in _status(screen)
        assert "again" in _status(screen)
        assert "cleared" not in _status(screen)
        if operation != "discover":
            assert _list(screen).selected == [MODELS[1]]
            assert _list(screen).option_count == 2
        scope.failure = False
        await run()
        await _settle(host, pilot)
        assert "OSError" not in _status(screen)
        assert (
            screen.query_one("#settings-model-value", Input).value
            == "existing-active-model"
        )
        if operation == "save":
            assert app.providers_models["OpenAI"] == [MODELS[1]]
        elif operation == "clear":
            assert _list(screen).option_count == 0
            assert screen.query_one("#settings-model-value", Input).suggester is None
        else:
            assert _list(screen).option_count == 2
        assert app.app_config["chat_defaults"]["model"] == "existing-active-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_late_clear_cannot_overwrite_a_changed_form(failure):
    app, scope = _app_scope(gate="clear")
    host = ProviderSettingsHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        await screen._discover_provider_models()
        scope.failure = failure
        screen._clear_discovered_provider_models_worker()
        try:
            await asyncio.wait_for(scope.started.wait(), timeout=5)
            screen.query_one("#settings-provider-value", Select).value = "llama_cpp"
            await pilot.pause()
            expected_status = _status(screen)
        finally:
            scope.release.set()
        await _settle(host, pilot)
        assert _status(screen) == expected_status
        assert _list(screen).option_count == 0


@pytest.mark.asyncio
async def test_rebuild_keeps_selection_with_connection_drafts():
    app, scope = _app_scope()
    host = ProviderSettingsHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        for selector, value in (
            ("#settings-provider-endpoint-value", "https://draft.invalid/v1"),
            ("#settings-provider-credential-env-var", "FIXTURE_KEY"),
            ("#settings-provider-api-key", "fixture-secret"),
        ):
            screen.query_one(selector, Input).value = value
            await pilot.pause()
        await screen._discover_provider_models()
        _list(screen).select(MODELS[1])
        await pilot.pause()
        screen.mutate_reactive(SettingsScreen.active_category)
        await _settle(host, pilot)
        assert _list(screen).selected == [MODELS[1]]
        assert _list(screen).option_count == 2
        assert not scope.persist_calls
