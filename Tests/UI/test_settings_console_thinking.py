"""The thinking display preference uses canonical settings persistence."""

import pytest
from textual.widgets import Checkbox, Select
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
)
from Tests.UI.test_settings_configuration_hub import _open_settings_category
import tldw_chatbook.UI.Screens.settings_screen as settings_module


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["show_thinking"])
async def test_thinking_default_on_saves_off_and_reopens_off(monkeypatch, setting):
    app = _build_test_app()
    app.app_config["console"] = {}
    saved = []

    class Adapter:
        def save_sections(self, values):
            saved.append(values)
            return True

    monkeypatch.setattr(settings_module, "SettingsConfigAdapter", Adapter)
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        toggle = screen.query_one(
            "#settings-console-" + setting.replace("_", "-"), Checkbox
        )
        assert toggle.value is True
        toggle.focus()
        await pilot.press("space")
        await pilot.click("#settings-save-category")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert saved == [{"console": {setting: False}}]
        assert app.app_config["console"][setting] is False
        screen._sync_console_behavior_widgets()
        assert toggle.value is False


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["llama_cpp", "local_vllm", "local_ollama"])
async def test_reasoning_history_defaults_auto_and_remembers_current_model_override(
    monkeypatch, provider
):
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.local_reasoning import reasoning_override_key

    app = _build_test_app()
    app.app_config["console"] = {}
    store = ConsoleChatStore()
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    app.app_config.setdefault("api_settings", {})[provider] = {
        "api_url": "http://localhost:9099"
    }
    store.ensure_session(
        settings=ConsoleSessionSettings(provider=provider, model="alias", base_url=None)
    )
    gateway = ConsoleProviderGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider=provider,
        model="alias",
        base_url="http://localhost:9099",
    )
    app.console_runtime = SimpleNamespace(
        chat_store=store, chat_controller=controller, provider_gateway=gateway
    )
    saved = []

    class Adapter:
        def save_sections(self, values):
            saved.append(values)
            return True

    monkeypatch.setattr(settings_module, "SettingsConfigAdapter", Adapter)
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        default = screen.query_one("#settings-console-reasoning-history", Select)
        assert default.value == "auto"
        override = screen.query_one("#settings-console-reasoning-override", Select)
        assert override.value == "inherit"
        override.value = "all"
        await pilot.pause()
        await pilot.click("#settings-save-category")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        key = reasoning_override_key(
            provider,
            "http://localhost:9099"
            if provider != "llama_cpp"
            else "http://127.0.0.1:9099",
            "alias",
        )
        assert app.app_config["console"]["reasoning_history_overrides"] == {key: "all"}
        assert saved == [{"console": {"reasoning_history_overrides": {key: "all"}}}]
        screen._sync_console_behavior_widgets()
        assert override.value == "all"
        native = screen.query_one("#settings-console-reasoning-native-tools", Checkbox)
        native.value = True
        override.value = "inherit"
        default.value = "off"
        await pilot.pause()
        await pilot.click("#settings-save-category")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert app.app_config["console"]["reasoning_history_overrides"] == {}
        assert app.app_config["console"]["reasoning_history"] == "off"
        assert app.app_config["console"]["reasoning_native_tool_overrides"] == {
            key: True
        }


@pytest.mark.asyncio
async def test_console_screen_refreshes_thinking_before_answer_text():
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from textual.widgets import Static
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleThinkingBlock

    app = _build_test_app()
    host = ConsoleHarness(app)
    host.app_config = app.app_config
    async with host.run_test(size=(160, 45)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        store = screen._ensure_console_chat_store()
        session = store.ensure_session()
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="Question"
        )
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        await screen._sync_native_console_transcript()
        await pilot.pause()
        store.append_thinking_chunk(message.id, "Visible before any answer")
        await screen._sync_native_console_transcript()
        await pilot.pause()
        row = screen.query_one(f"#console-message-{message.id}")
        block = row.query_one(ConsoleThinkingBlock)
        assert block.display and not block.collapsed
        assert "Visible before any answer" in str(
            block.query_one(".console-thinking-text", Static).render()
        )
        store.append_thinking_chunk(message.id, ", and updates live")
        await screen._sync_native_console_transcript()
        await pilot.pause()
        assert "updates live" in str(
            block.query_one(".console-thinking-text", Static).render()
        )

        app.app_config.setdefault("console", {})["show_thinking"] = False
        await screen._sync_native_console_transcript()
        await pilot.pause()
        assert not block.display
        assert store.get_message(message.id).thinking_text.endswith("updates live")
        app.app_config["console"]["show_thinking"] = True
        await screen._sync_native_console_transcript()
        await pilot.pause()
        assert block.display
