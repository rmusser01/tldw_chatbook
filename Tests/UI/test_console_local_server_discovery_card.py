"""Setup-card local-server auto-detect affordance tests (task-188)."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
from tldw_chatbook.Chat.console_onboarding_state import (
    ConsoleSetupCardState,
    ConsoleSetupStep,
    build_console_detected_server_action,
)
from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID,
    CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION,
    ConsoleSetupModal,
)


_BLOCKED_CARD_STATE = ConsoleSetupCardState(
    mode="card",
    steps=(
        ConsoleSetupStep(state="active", label="Connect a provider (API key or local server)"),
        ConsoleSetupStep(state="pending", label="Pick a model"),
        ConsoleSetupStep(state="pending", label="Send your first message"),
    ),
)
_DETECTED_SERVER = DiscoveredLocalServer(
    provider_key="llama_cpp",
    base_url="http://127.0.0.1:8080",
    model_ids=("srv-model-a", "srv-model-b"),
)


class SetupModalHarness(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.workbench_actions: list[str] = []

    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")

    async def on_mount(self) -> None:
        self.query_one("#console-setup-modal", ConsoleSetupModal).sync_card_state(
            _BLOCKED_CARD_STATE,
            action_label="Set up provider",
            action_tooltip="Open provider settings.",
        )

    def on_workbench_action_requested(self, event: WorkbenchActionRequested) -> None:
        event.stop()
        self.workbench_actions.append(event.action_id)


def _is_displayed(widget) -> bool:
    current = widget
    while current is not None:
        if current.display is False or current.styles.display == "none":
            return False
        current = getattr(current, "parent", None)
    return True


@pytest.mark.asyncio
async def test_setup_modal_offers_and_routes_detected_server_action() -> None:
    app = SetupModalHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        detected = app.query_one(f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}", Button)
        # No offer yet: the card stays exactly as before (no noise).
        assert not _is_displayed(detected)

        modal.sync_detected_server_action(
            build_console_detected_server_action(_DETECTED_SERVER, card_mode="card")
        )
        await pilot.pause()

        assert _is_displayed(detected)
        assert str(detected.label) == "Use detected llama.cpp (127.0.0.1:8080)"
        assert "srv-model-a" in str(detected.tooltip)

        detected.press()
        await pilot.pause()
        assert app.workbench_actions == [CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION]

        # Withdrawing the offer hides the affordance again.
        modal.sync_detected_server_action(None)
        await pilot.pause()
        assert not _is_displayed(detected)


@pytest.mark.asyncio
async def test_setup_modal_hides_detected_action_once_card_unblocks() -> None:
    app = SetupModalHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        modal.sync_detected_server_action(
            build_console_detected_server_action(_DETECTED_SERVER, card_mode="card")
        )
        await pilot.pause()
        detected = app.query_one(f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}", Button)
        assert _is_displayed(detected)

        modal.sync_card_state(ConsoleSetupCardState(mode="quiet"))
        await pilot.pause()
        assert not _is_displayed(detected)


class ConsoleHarness(App[None]):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _blocked_console_app(detected_servers: tuple[DiscoveredLocalServer, ...]):
    """Build a test app whose Console blocks on setup with a discovery fake."""
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    discovery_calls: list[object] = []

    async def fake_discovery(config):
        discovery_calls.append(config)
        return detected_servers

    app.console_local_server_discovery = fake_discovery
    app._discovery_calls = discovery_calls
    return app


async def _wait_for_displayed(console, pilot, selector: str) -> None:
    for _ in range(60):
        widgets = list(console.query(selector))
        if widgets and _is_displayed(widgets[0]):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"{selector} never became visible")


@pytest.mark.asyncio
async def test_blocked_card_runs_discovery_once_and_offers_detected_server() -> None:
    app = _blocked_console_app((_DETECTED_SERVER,))
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        await _wait_for_displayed(
            console, pilot, f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}"
        )

        detected = console.query_one(
            f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}", Button
        )
        assert str(detected.label) == "Use detected llama.cpp (127.0.0.1:8080)"
        assert len(app._discovery_calls) == 1


@pytest.mark.asyncio
async def test_pressing_detected_server_saves_config_and_unlocks_card(monkeypatch) -> None:
    app = _blocked_console_app((_DETECTED_SERVER,))
    saved_sections: list[dict] = []

    def fake_save(sections):
        saved_sections.append(sections)
        # Simulate the persisted write: the fresh-config readiness path reads
        # the injected mapping verbatim in tests, so apply the sections there.
        for section_path, values in sections.items():
            target = app.app_config
            for part in section_path.split("."):
                target = target.setdefault(part, {})
            target.update(values)
        return True

    monkeypatch.setattr(chat_screen_module, "save_settings_to_cli_config", fake_save)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        await _wait_for_displayed(
            console, pilot, f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}"
        )

        console.query_one(f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}", Button).press()
        for _ in range(60):
            modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
            if not modal.is_blocking:
                break
            await pilot.pause(0.05)

        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.is_blocking is False
        assert saved_sections == [
            {
                "api_settings.llama_cpp": {
                    "api_url": "http://127.0.0.1:8080",
                    "model": "srv-model-a",
                },
                "chat_defaults": {
                    "provider": "llama_cpp",
                    "model": "srv-model-a",
                },
            }
        ]
        settings = console._active_console_session_settings()
        assert settings is not None
        assert settings.provider == "llama_cpp"
        assert settings.model == "srv-model-a"
        assert settings.source == "user"


@pytest.mark.asyncio
async def test_blocked_card_stays_quiet_when_no_server_is_found() -> None:
    app = _blocked_console_app(())
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        for _ in range(10):
            await pilot.pause(0.05)

        assert len(app._discovery_calls) == 1
        detected = console.query_one(
            f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}", Button
        )
        assert not _is_displayed(detected)
