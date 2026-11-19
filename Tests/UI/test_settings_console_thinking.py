"""Console reasoning-history settings use canonical settings persistence.

TASK-41509 deleted two tests written against APIs that never shipped (not on
dev, not in the Codex snapshot bf364a4bf7): a ``console.show_thinking``
setting saved by the category Save, and ``ConsoleThinkingBlock`` with
``ConsoleChatStore.append_thinking_chunk``. The shipped setting is
``console.show_model_thinking``, saved at once; see
``test_settings_context_memory_controls.py`` and
``test_settings_thinking_visibility_journeys.py``. Live thinking is the
transcript's activity disclosure; see ``test_console_thinking_disclosures.py``.
"""

import pytest
from textual.widgets import Checkbox, Select

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
)
from Tests.UI.test_settings_configuration_hub import _open_settings_category
import tldw_chatbook.UI.Screens.settings_screen as settings_module


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["llama_cpp", "local_vllm", "local_ollama"])
@private_profile_test
async def test_reasoning_history_defaults_auto_and_remembers_current_model_override(
    request, monkeypatch, provider
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
