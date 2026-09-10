"""Mounted Settings contracts for device-local local-reasoning replay policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Checkbox, Select

import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
)
from Tests.UI.test_settings_configuration_hub import _open_settings_category
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.local_reasoning import (
    ReasoningReplayPolicy,
    reasoning_override_key,
)
from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML, validate_config_keys


def _app_with_local_console_target(
    *,
    provider: str = "local_vllm",
    endpoint: str = "http://localhost:9099",
    model: str = "alias",
):
    app = _build_test_app()
    app.app_config["console"] = {}
    app.app_config.setdefault("api_settings", {})[provider] = {"api_url": endpoint}
    store = ConsoleChatStore()
    store.ensure_session(
        settings=ConsoleSessionSettings(provider=provider, model=model, base_url=None)
    )
    gateway = ConsoleProviderGateway(config_provider=lambda: app.app_config)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider=provider,
        model=model,
        base_url=endpoint,
    )
    app.console_runtime = SimpleNamespace(
        chat_store=store,
        chat_controller=controller,
        provider_gateway=gateway,
    )
    return app


def test_reasoning_override_maps_are_known_freeform_console_config() -> None:
    """Catches saved target digests being reported as unknown config keys."""

    key = reasoning_override_key("local_vllm", "http://localhost:9099", "alias")

    assert DEFAULT_CONFIG_FROM_TOML["console"]["reasoning_history_overrides"] == {}
    assert DEFAULT_CONFIG_FROM_TOML["console"]["reasoning_native_tool_overrides"] == {}
    assert (
        validate_config_keys(
            {
                "console": {
                    "reasoning_history_overrides": {key: "current"},
                    "reasoning_native_tool_overrides": {key: True},
                }
            }
        )
        == []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("saved_mode", ["current", "all", "off"])
async def test_reasoning_history_defaults_automatic_and_persists_each_mode(
    monkeypatch, saved_mode: str
) -> None:
    """Catches a missing mode option or saving replay under conversation policy."""

    app = _app_with_local_console_target()
    saved: list[dict[str, dict[str, object]]] = []

    class FakeAdapter:
        def save_sections(self, values):
            saved.append(values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        default = screen.query_one("#settings-console-reasoning-history", Select)

        assert default.value == "auto"
        assert [(str(label), value) for label, value in default._options] == [
            ("Automatic (recommended)", "auto"),
            ("Current exchange", "current"),
            ("All available", "all"),
            ("Off", "off"),
        ]
        visible = _visible_text(screen)
        assert "Conversation Auto" in visible
        assert "Include and Exclude" in visible
        assert "Required" in visible
        assert "server template can still omit older reasoning" in visible

        default.value = saved_mode
        await pilot.pause()
        await pilot.click("#settings-save-category")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert saved == [{"console": {"reasoning_history": saved_mode}}]
        assert app.app_config["console"]["reasoning_history"] == saved_mode


@pytest.mark.asyncio
async def test_reasoning_history_remembers_normalized_target_and_clears_override(
    monkeypatch,
) -> None:
    """Catches storing a raw URL/model or making native-tool support implicit."""

    endpoint = "HTTP://user:secret@LOCALHOST:9099/v1/chat/completions?token=private"
    app = _app_with_local_console_target(endpoint=endpoint, model="  alias  ")
    saved: list[dict[str, dict[str, object]]] = []

    class FakeAdapter:
        def save_sections(self, values):
            saved.append(values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        override = screen.query_one("#settings-console-reasoning-override", Select)
        native = screen.query_one("#settings-console-reasoning-native-tools", Checkbox)

        assert override.value == "inherit"
        assert next((str(label), value) for label, value in override._options) == (
            "Use default",
            "inherit",
        )
        assert native.value is False

        override.value = "all"
        native.value = True
        await pilot.pause()
        await pilot.click("#settings-save-category")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        key = reasoning_override_key("local_vllm", endpoint, "alias")
        expected_first = {
            "console": {
                "reasoning_history_overrides": {key: "all"},
                "reasoning_native_tool_overrides": {key: True},
            }
        }
        assert saved == [expected_first]
        assert app.app_config["console"]["reasoning_history_overrides"] == {key: "all"}
        assert app.app_config["console"]["reasoning_native_tool_overrides"] == {
            key: True
        }
        assert "secret" not in key and "private" not in key and "alias" not in key
        target_copy = next(
            str(widget.renderable)
            for widget in screen.query(".settings-detail-row")
            if "local_vllm / alias" in str(getattr(widget, "renderable", ""))
        )
        assert "secret" not in target_copy and "private" not in target_copy

        override.value = "inherit"
        await pilot.pause()
        await pilot.click("#settings-save-category")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert saved[-1] == {"console": {"reasoning_history_overrides": {}}}
        assert app.app_config["console"]["reasoning_history_overrides"] == {}
        assert app.app_config["console"]["reasoning_native_tool_overrides"] == {
            key: True
        }


@pytest.mark.asyncio
async def test_gemma_status_explains_fenced_tool_round_limit() -> None:
    app = _app_with_local_console_target(
        provider="local_vllm",
        endpoint="http://localhost:9099",
        model="gemma-4",
    )
    key = reasoning_override_key(
        "local_vllm", "http://localhost:9099", "gemma-4"
    )
    app.console_runtime.provider_gateway.reasoning_policies = {
        key: ReasoningReplayPolicy(
            "current",
            "Auto",
            "Gemma 4",
            supports_preserve=True,
            verified=True,
            native_tools=False,
        )
    }

    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)

        assert (
            "Native server tools are needed to retain Gemma thinking across tool rounds"
            in _visible_text(screen)
        )
