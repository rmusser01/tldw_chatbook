"""Setup-card local-server auto-detect affordance tests (task-188).

TASK-33008.3: "Use detected <server>" opens Switch model on that provider
instead of adopting the server's first model on one press. Enter there adopts
the highlighted model; Esc saves nothing. The mounted tests are
``private_profile_test`` cases: a factory-built app needs its own profile.
"""

import asyncio

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_model_switcher import line_with, list_lines
from Tests.UI.test_console_switch_model_entry_points import _switcher_ready, _until
from Tests.UI.test_console_switch_model_keys import _drain
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
import tldw_chatbook.UI.Console_Modules.model_switcher as model_switcher_module
from tldw_chatbook.Chat.console_onboarding_state import (
    ConsoleSetupCardState,
    ConsoleSetupStep,
    build_console_detected_server_action,
)
from tldw_chatbook.Chat.console_session_settings import (
    build_target_default_console_session_settings,
    console_send_connection,
)
from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderProbeResult,
    ProviderTestEvidenceStore,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_ACTION_ID,
    CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID,
    CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION,
    ConsoleSetupModal,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleModelDiscoveryIdentity,
)


_BLOCKED_CARD_STATE = ConsoleSetupCardState(
    mode="card",
    steps=(
        ConsoleSetupStep(
            state="active", label="Connect a provider (API key or local server)"
        ),
        ConsoleSetupStep(state="pending", label="Pick a model"),
        ConsoleSetupStep(state="pending", label="Send your first message"),
    ),
)
_DETECTED_SERVER = DiscoveredLocalServer(
    provider_key="llama_cpp",
    base_url="http://127.0.0.1:8080",
    model_ids=("srv-model-a", "srv-model-b"),
)


def test_modal_discovery_identity_does_not_collapse_returned_provider_drafts() -> None:
    """Returning to a provider uses a later generation, never its old probe identity."""
    first_visit = ConsoleModelDiscoveryIdentity(
        provider_key="llama_cpp",
        connection_identity=("llama_cpp", "http://127.0.0.1:8080"),
        draft_generation=2,
    )
    returned_visit = ConsoleModelDiscoveryIdentity(
        provider_key="llama_cpp",
        connection_identity=("llama_cpp", "http://127.0.0.1:8080"),
        draft_generation=4,
    )

    assert returned_visit != first_visit


class SetupModalHarness(ConsolidatedCSSApp):
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
        # TASK-33008.3: the tooltip names the server, never a model it would
        # pick for the user.
        assert "llama.cpp at 127.0.0.1:8080" in str(detected.tooltip)
        assert "srv-model-a" not in str(detected.tooltip)

        detected.press()
        await pilot.pause()
        assert app.workbench_actions == [CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION]

        # Withdrawing the offer hides the affordance again.
        modal.sync_detected_server_action(None)
        await pilot.pause()
        assert not _is_displayed(detected)


@pytest.mark.asyncio
async def test_setup_modal_first_focus_is_the_detected_action_when_one_is_offered() -> (
    None
):
    """TASK-33008.3 AC#1: ``focus_primary_action`` lands on "Use detected
    <server>" while it is offered, and on the primary action otherwise. The
    button order on the card does not change."""
    app = SetupModalHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        primary = app.query_one(f"#{CONSOLE_SETUP_MODAL_ACTION_ID}", Button)
        detected = app.query_one(f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}", Button)

        modal.focus_primary_action()
        await pilot.pause()
        assert app.focused is primary

        modal.sync_detected_server_action(
            build_console_detected_server_action(_DETECTED_SERVER, card_mode="card")
        )
        modal.focus_primary_action()
        await pilot.pause()
        assert app.focused is detected
        assert modal.query(Button).first() is primary

        modal.sync_detected_server_action(None)
        modal.focus_primary_action()
        await pilot.pause()
        assert app.focused is primary


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


class ConsoleHarness(ConsolidatedCSSApp):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _blocked_console_app(
    detected_servers: tuple[DiscoveredLocalServer, ...],
    llama_cpp: dict | None = None,
):
    """Build a test app whose Console blocks on setup with a discovery fake."""
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    if llama_cpp is not None:
        app.app_config["api_settings"]["llama_cpp"] = llama_cpp
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    discovery_calls: list[object] = []

    async def fake_discovery(config):
        discovery_calls.append(config)
        return detected_servers

    app.console_local_server_discovery = fake_discovery
    app._discovery_calls = discovery_calls
    return app


def _capture_saves(app, monkeypatch) -> list[dict]:
    """Stand in for the config writer: record each save and apply it to the
    injected mapping, which the fresh-config readiness path reads verbatim."""
    saved_sections: list[dict] = []

    def fake_save(sections):
        saved_sections.append(sections)
        for section_path, values in sections.items():
            target = app.app_config
            for part in section_path.split("."):
                target = target.setdefault(part, {})
            target.update(values)
        return True

    # The adoption writer lives beside the card's switcher opener; a patch of
    # chat_screen's old name would now raise instead of capturing nothing.
    monkeypatch.setattr(model_switcher_module, "save_settings_to_cli_config", fake_save)
    return saved_sections


async def _wait_for_displayed(console, pilot, selector: str) -> None:
    for _ in range(60):
        widgets = list(console.query(selector))
        if widgets and _is_displayed(widgets[0]):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"{selector} never became visible")


async def _detected_action_focused(host, console, pilot) -> Button:
    """AC#1: discovery ends after the card's first focus; the guidance sync
    that follows moves focus to "Use detected <server>"."""
    await _wait_for_selector(console, pilot, "#console-setup-modal")
    await _wait_for_displayed(
        console, pilot, f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}"
    )
    detected = console.query_one(f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}", Button)
    await _until(pilot, lambda: host.focused is detected, "the detected action's focus")
    return detected


def _pair(console) -> tuple[str, str | None]:
    settings = console._session._active_console_session_settings()
    assert settings is not None
    return (settings.provider, settings.model)


@pytest.mark.asyncio
@private_profile_test
async def test_blocked_card_runs_discovery_once_and_offers_detected_server(
    request,
) -> None:
    app = _blocked_console_app((_DETECTED_SERVER,))
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        detected = await _detected_action_focused(host, console, pilot)

        assert str(detected.label) == "Use detected llama.cpp (127.0.0.1:8080)"
        assert len(app._discovery_calls) == 1


@pytest.mark.asyncio
@private_profile_test
async def test_detected_server_enter_enter_saves_config_and_unlocks_card(
    monkeypatch,
    request,
) -> None:
    """TASK-33008.3 AC#1, #2, #6, #7. Rewritten on purpose from
    ``test_pressing_detected_server_saves_config_and_unlocks_card``, which
    pinned the one-press adoption of the server's first model.

    Two keys from the card: Enter opens Switch model on the detected provider,
    with the server's first model highlighted, every model it lists visible
    and a scope line that names the new-chat default; nothing is saved yet.
    Enter there saves the pair and the endpoint and unlocks the card."""
    app = _blocked_console_app((_DETECTED_SERVER,))
    saved_sections = _capture_saves(app, monkeypatch)
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _detected_action_focused(host, console, pilot)
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        before = _pair(console)
        assert before[0] != "llama_cpp"

        await pilot.press("enter")
        switcher = await _switcher_ready(host, pilot)
        assert switcher.query_one("#console-popover-find", Input).value == "llama.cpp "
        row = switcher.highlighted_row()
        assert row is not None
        assert (row.kind, row.provider, row.model) == (
            "pair",
            "llama_cpp",
            "srv-model-a",
        )
        lines = list_lines(host, switcher)
        assert "llama.cpp" in line_with(lines, "srv-model-a")
        assert "llama.cpp" in line_with(lines, "srv-model-b")
        scope = switcher.query_one("#console-popover-scope", Static)
        assert (
            str(scope.render()) == model_switcher_module.SCOPE_THIS_CHAT_AND_NEW_CHATS
        )
        assert saved_sections == [] and modal.is_blocking
        assert _pair(console) == before

        await pilot.press("enter")
        await _until(pilot, lambda: host.screen is console, "the switcher to close")
        await _drain(app)
        await _until(pilot, lambda: not modal.is_blocking, "the card to unlock")

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
        settings = console._session._active_console_session_settings()
        assert settings is not None
        assert settings.provider == "llama_cpp"
        assert settings.model == "srv-model-a"
        assert settings.base_url == "http://127.0.0.1:8080"
        assert settings.source == "user"


@pytest.mark.asyncio
@private_profile_test
async def test_esc_from_the_detected_switcher_returns_to_the_card_with_nothing_saved(
    monkeypatch,
    request,
) -> None:
    """TASK-33008.3 AC#3: Esc closes the card-opened switcher; the card still
    blocks with focus on "Use detected <server>", and nothing was written."""
    app = _blocked_console_app((_DETECTED_SERVER,))
    saved_sections = _capture_saves(app, monkeypatch)
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        detected = await _detected_action_focused(host, console, pilot)
        before = dict(app.app_config["chat_defaults"])
        pair = _pair(console)

        await pilot.press("enter")
        await _switcher_ready(host, pilot)
        await pilot.press("down")  # Moving the highlight applies nothing either.
        await pilot.press("escape")
        await _until(pilot, lambda: host.screen is console, "Esc to close it")
        await pilot.pause()

        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.is_blocking
        assert host.focused is detected
        assert saved_sections == []
        assert app.app_config["chat_defaults"] == before
        assert "llama_cpp" not in app.app_config["api_settings"]
        assert _pair(console) == pair


@pytest.mark.asyncio
@private_profile_test
async def test_detected_server_keeps_a_different_saved_endpoint_for_this_chat_only(
    monkeypatch,
    request,
) -> None:
    """TASK-33008.3 AC#2 (task-16476) and ruling R12, a different endpoint
    saved. The saved endpoint's last test was refused, which would list the
    provider under NEEDS SETUP; the card's switcher reads the detected
    endpoint instead, so Enter on the second listed model applies it. The
    saved endpoint is kept, with today's warning, and the detected one is
    used for this chat only."""
    saved_endpoint = "http://127.0.0.1:9099"
    app = _blocked_console_app(
        (_DETECTED_SERVER,), llama_cpp={"api_url": saved_endpoint, "model": "old-model"}
    )
    # A saved model list for the provider describes another endpoint: the
    # rows are the models this server listed.
    app.providers_models = {"llama_cpp": ["old-model"]}
    saved_sections = _capture_saves(app, monkeypatch)
    notices: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notices.append(
            (str(message), kwargs.get("severity"))
        ),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _detected_action_focused(host, console, pilot)
        store = ProviderTestEvidenceStore(lambda: host)
        store.settle(
            store.begin(
                console_send_connection(
                    build_target_default_console_session_settings(
                        app.app_config, "llama_cpp", "old-model"
                    ),
                    app_config=app.app_config,
                )
            ),
            ProviderProbeResult("unreachable", (), "connection_refused"),
        )
        assert (
            console._console_default_readiness("llama_cpp", "old-model").blocker
            == "endpoint_unreachable"
        )

        await pilot.press("enter")
        switcher = await _switcher_ready(host, pilot)
        lines = list_lines(host, switcher)
        assert "old-model" not in "\n".join(lines)  # only what the server lists
        await pilot.press("down")
        row = switcher.highlighted_row()
        assert row is not None
        assert (row.kind, row.provider, row.model) == (
            "pair",
            "llama_cpp",
            "srv-model-b",
        )
        assert saved_sections == []

        await pilot.press("enter")
        await _until(pilot, lambda: host.screen is console, "the switcher to close")
        await _drain(app)

        assert saved_sections == [
            {
                "api_settings.llama_cpp": {"model": "srv-model-b"},
                "chat_defaults": {"provider": "llama_cpp", "model": "srv-model-b"},
            }
        ]
        assert app.app_config["api_settings"]["llama_cpp"]["api_url"] == saved_endpoint
        warning = (
            "Keeping the saved endpoint http://127.0.0.1:9099 for llama_cpp; "
            "using the detected server for this session only."
        )
        assert (warning, "warning") in notices
        settings = console._session._active_console_session_settings()
        assert (settings.provider, settings.model) == ("llama_cpp", "srv-model-b")
        assert settings.base_url == "http://127.0.0.1:8080"


@pytest.mark.asyncio
@private_profile_test
async def test_a_failed_adoption_holds_no_shutdown_admission(
    monkeypatch,
    request,
) -> None:
    """The adoption runs inside the switcher's commit, after the live commit
    took a durability admission. If it raises, the switcher reports it and
    delivers no result, so the commit itself must release the admission:
    ``close_and_drain`` (app shutdown) waits for every one still held."""
    app = _blocked_console_app((_DETECTED_SERVER,))

    def broken_writer(*_args) -> None:
        raise RuntimeError("config writer exploded")

    monkeypatch.setattr(
        model_switcher_module, "adopt_detected_local_server", broken_writer
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _detected_action_focused(host, console, pilot)
        await pilot.press("enter")
        switcher = await _switcher_ready(host, pilot)
        await pilot.press("enter")
        await pilot.pause()

        assert host.screen is switcher
        assert switcher.query_one("#console-popover-error", Static).display
        await asyncio.wait_for(
            app.console_settings_durability_owner.close_and_drain(), timeout=2
        )


@pytest.mark.asyncio
@private_profile_test
async def test_blocked_card_stays_quiet_when_no_server_is_found(request) -> None:
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
