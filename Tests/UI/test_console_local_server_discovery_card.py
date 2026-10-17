"""Setup-card local-server auto-detect affordance tests (task-188).

TASK-33008.3: "Use detected <server>" opens Switch model on that provider
instead of adopting the server's first model on one press. Enter there adopts
the highlighted model; Esc saves nothing. The mounted tests are
``private_profile_test`` cases: a factory-built app needs its own profile.
"""

import asyncio
import threading
import time
from dataclasses import replace

import pytest
from textual.app import ComposeResult
from textual.screen import Screen

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
from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
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
    **api_settings: dict,
):
    """Build a test app whose Console blocks on setup with a discovery fake."""
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}, **api_settings},
    }
    if llama_cpp is not None:
        app.app_config["api_settings"]["llama_cpp"] = llama_cpp
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    discovery_calls: list[object] = []
    discovery_threads: list[int] = []

    async def fake_discovery(config):
        discovery_calls.append(config)
        discovery_threads.append(threading.get_ident())
        return detected_servers

    app.console_local_server_discovery = fake_discovery
    app._discovery_calls = discovery_calls
    app._discovery_threads = discovery_threads
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


async def _until_clock(pilot, predicate, what: str, *, timeout: float = 20.0) -> None:
    """``_until`` on a clock deadline: a count of pauses ran out early at load
    average 30-60 (TASK-33008 review round 4; it was 60 x 0.05 s here)."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await pilot.pause(0.05)


async def _wait_for_displayed(console, pilot, selector: str) -> None:
    def shown() -> bool:
        widgets = list(console.query(selector))
        return bool(widgets) and _is_displayed(widgets[0])

    await _until_clock(pilot, shown, f"{selector} to be shown")


async def _first_look_done(app, console, pilot) -> None:
    """The card's first discovery ran, and so did the mount's own resume.

    The first look can end before that resume runs (it waits on the attach
    reconciliation). A return in that window is merged into it and reads as
    the mount's, so it re-arms no discovery; a test that leaves and comes
    back waits for it, as a user's return comes after the first visit
    (TASK-33008 review round 4).
    """
    await _wait_for_selector(console, pilot, "#console-setup-modal")
    await _until(pilot, lambda: len(app._discovery_calls) == 1, "discovery")
    await _until_clock(
        pilot,
        lambda: not console._console_mount_visit_refreshed,
        "the mount's own resume",
    )


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
        # Capture finding 12 (02a): step 1's word is the chat's OpenAI pair's,
        # and it says so beside the detected server's action.
        step = console.query_one("#console-setup-step-1", Static)
        assert "\n     OpenAI: Not ready · no key" in str(step.render())
        assert len(app._discovery_calls) == 1
        # Owner ruling (Phase 5): it probes in its own thread and loop. On the
        # UI loop a boot-time storage scope of another task leaked into the
        # recovery guard, which refused a plain launch's only attempt.
        assert app._discovery_threads != [threading.get_ident()]


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
async def test_two_presses_of_use_detected_open_one_switch_model(request) -> None:
    """Final review: a double click posts two ``Button.Pressed`` before the
    first one's Switch model has the screen. Each opened a dialog, and the
    lower one stayed open on stale state after the top one applied. Only the
    first opens; one Esc is back on the card."""
    app = _blocked_console_app((_DETECTED_SERVER,))
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        detected = await _detected_action_focused(host, console, pilot)

        detected.post_message(Button.Pressed(detected))
        detected.post_message(Button.Pressed(detected))
        await _switcher_ready(host, pilot)
        await pilot.pause()

        opened = [type(screen).__name__ for screen in host.screen_stack]
        assert opened[-2:] == ["ChatScreen", "ConsoleModelPopover"], opened
        await pilot.press("escape")
        await _until(pilot, lambda: host.screen is console, "one Esc to close it")


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
            "Keeping the saved endpoint http://127.0.0.1:9099 for llama.cpp; "
            "using the detected server for this session only."
        )
        assert (warning, "warning") in notices
        # The detected endpoint is this chat's session-only endpoint: never
        # conversation metadata, and the send path uses it.
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        settings = store.session_settings(session_id)
        assert (settings.provider, settings.model) == ("llama_cpp", "srv-model-b")
        assert settings.base_url != "http://127.0.0.1:8080"
        assert (
            store.effective_session_settings(session_id).base_url
            == "http://127.0.0.1:8080"
        )
        assert store.session_ephemeral_endpoint_policy(session_id).base_url == (
            "http://127.0.0.1:8080"
        )
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        await _until(pilot, lambda: not modal.is_blocking, "the card to unlock")


def _record_notices(app, monkeypatch) -> list[tuple[str, str | None]]:
    notices: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notices.append(
            (str(message), kwargs.get("severity"))
        ),
    )
    return notices


async def _enter_enter(host, console, pilot) -> None:
    """Press Enter on "Use detected <server>", then in Switch model; drain."""
    await _detected_action_focused(host, console, pilot)
    await pilot.press("enter")
    await _switcher_ready(host, pilot)
    await pilot.press("enter")
    await _until(pilot, lambda: host.screen is console, "the switcher to close")
    await _drain(host.app_instance)


@pytest.mark.asyncio
@private_profile_test
async def test_detected_ollama_keeps_a_different_saved_endpoint_and_unlocks_the_card(
    monkeypatch,
    request,
) -> None:
    """TASK-33008.3 AC#2 and parent AC#1 for a provider off the direct
    llama.cpp path. Ollama has a different address saved (a LAN box), so it is
    kept, and the detected server is this chat's session-only endpoint. That
    endpoint is what a send uses, so readiness reads it as configured and the
    card unlocks: before, readiness said "endpoint not saved" and the card
    stayed up after Enter, Enter."""
    lan = "http://192.168.1.50:11434/v1/chat/completions"
    detected = DiscoveredLocalServer(
        provider_key="ollama",
        base_url="http://127.0.0.1:11434",
        model_ids=("llama3.2", "qwen3"),
    )
    app = _blocked_console_app((detected,), ollama={"api_url": lan, "model": ""})
    saved_sections = _capture_saves(app, monkeypatch)
    notices = _record_notices(app, monkeypatch)
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _enter_enter(host, console, pilot)
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        await _until(pilot, lambda: not modal.is_blocking, "the card to unlock")

        assert saved_sections == [
            {
                "api_settings.ollama": {"model": "llama3.2"},
                "chat_defaults": {"provider": "ollama", "model": "llama3.2"},
            }
        ]
        assert app.app_config["api_settings"]["ollama"]["api_url"] == lan
        assert any(
            message.startswith("Keeping the saved endpoint") and severity == "warning"
            for message, severity in notices
        )
        settings, readiness = console._active_console_settings_readiness()
        assert (settings.provider, settings.model) == ("ollama", "llama3.2")
        assert settings.base_url == detected.base_url
        assert readiness.native_send_supported, readiness.blocker
        selection = console._build_console_provider_selection()
        assert selection.base_url == detected.base_url
        assert not selection.configured_endpoint_fallback_allowed


@pytest.mark.asyncio
@private_profile_test
async def test_a_failed_default_save_says_the_pick_applies_to_this_chat_only(
    monkeypatch,
    request,
) -> None:
    """TASK-33008.3: the scope line promised the new-chat default, so a save
    that fails says so instead of only logging it."""
    app = _blocked_console_app((_DETECTED_SERVER,))
    monkeypatch.setattr(
        model_switcher_module, "save_settings_to_cli_config", lambda _sections: False
    )
    notices = _record_notices(app, monkeypatch)
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _enter_enter(host, console, pilot)

        assert (
            "Default not saved for llama.cpp/srv-model-a; it applies to this chat only.",
            "warning",
        ) in notices
        assert _pair(console) == ("llama_cpp", "srv-model-a")


_DETECTED_OLLAMA = DiscoveredLocalServer(
    provider_key="ollama",
    base_url="http://127.0.0.1:11434",
    model_ids=("llama3.2", "qwen3"),
)


def _shipped_ollama(**edits: object) -> dict:
    """The template's Ollama table (``http://localhost:11434/v1/...``)."""
    return {**DEFAULT_CONFIG_FROM_TOML["api_settings"]["ollama"], **edits}


@pytest.mark.asyncio
@private_profile_test
async def test_detected_ollama_replaces_its_shipped_endpoint_after_its_model_changed(
    monkeypatch,
    request,
) -> None:
    """TASK-33008.3: only the endpoint decides whether the user chose one. An
    edited model leaves Ollama's endpoint the template's, so the detected
    address is saved, with no "Keeping the saved endpoint" warning."""
    app = _blocked_console_app(
        (_DETECTED_OLLAMA,), ollama=_shipped_ollama(model="qwen3")
    )
    saved_sections = _capture_saves(app, monkeypatch)
    notices = _record_notices(app, monkeypatch)
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _enter_enter(host, console, pilot)

        assert saved_sections == [
            {
                "api_settings.ollama": {
                    "model": "llama3.2",
                    "api_url": _DETECTED_OLLAMA.base_url,
                },
                "chat_defaults": {"provider": "ollama", "model": "llama3.2"},
            }
        ]
        assert not [message for message, _ in notices if "Keeping" in message]


@pytest.mark.asyncio
@private_profile_test
async def test_a_failed_save_keeps_the_detected_ollama_for_this_chat_and_unlocks(
    monkeypatch,
    request,
) -> None:
    """TASK-33008.3: when the default cannot be saved, the detected endpoint
    is not on disk either, so it is this chat's session-only endpoint. Before,
    it sat on the chat as an unsaved endpoint: Ollama read "endpoint not
    saved" and the card stayed up although the notice said the pick applies."""
    app = _blocked_console_app((_DETECTED_OLLAMA,), ollama=_shipped_ollama())
    monkeypatch.setattr(
        model_switcher_module, "save_settings_to_cli_config", lambda _sections: False
    )
    notices = _record_notices(app, monkeypatch)
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _enter_enter(host, console, pilot)
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        await _until(pilot, lambda: not modal.is_blocking, "the card to unlock")

        assert (
            "Default not saved for Ollama/llama3.2; it applies to this chat only.",
            "warning",
        ) in notices
        store = console._ensure_console_chat_store()
        policy = store.session_ephemeral_endpoint_policy(store.active_session_id)
        assert policy.base_url == _DETECTED_OLLAMA.base_url
        settings, readiness = console._active_console_settings_readiness()
        assert (settings.provider, settings.model) == ("ollama", "llama3.2")
        assert readiness.native_send_supported, readiness.blocker
        selection = console._build_console_provider_selection()
        assert selection.base_url == _DETECTED_OLLAMA.base_url
        assert not selection.configured_endpoint_fallback_allowed


@pytest.mark.asyncio
@private_profile_test
async def test_a_failed_adoption_keeps_the_applied_pair_and_warns(
    monkeypatch,
    request,
) -> None:
    """TASK-33008.3. Rewritten on purpose from
    ``test_a_failed_adoption_holds_no_shutdown_admission``, which pinned the
    switcher reporting "nothing changed" after the live commit had applied
    the pair. The adoption runs after that commit: when it raises, the commit
    stands, the switcher delivers it (which releases the durability
    admission ``close_and_drain`` waits for), and a warning says what is not
    done."""
    app = _blocked_console_app((_DETECTED_SERVER,))

    def broken_writer(*_args) -> None:
        raise RuntimeError("config writer exploded")

    monkeypatch.setattr(
        model_switcher_module, "adopt_detected_local_server", broken_writer
    )
    notices = _record_notices(app, monkeypatch)
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _enter_enter(host, console, pilot)

        assert _pair(console) == ("llama_cpp", "srv-model-a")
        assert (
            (
                "The detected server was not fully applied: check this chat's "
                "endpoint in Chat settings."
            ),
            "warning",
        ) in notices
        await asyncio.wait_for(
            app.console_settings_durability_owner.close_and_drain(), timeout=2
        )


@pytest.mark.asyncio
@private_profile_test
async def test_a_detected_legacy_alias_server_lists_its_models(
    monkeypatch,
    request,
) -> None:
    """TASK-33008.3: discovery also probes the shipped legacy-alias endpoints
    (llama.cpp on :8001). Legacy aliases get no readiness unless used, so the
    card's switcher was empty for such a server; serving models is use."""
    detected = DiscoveredLocalServer(
        provider_key="local_llamacpp",
        base_url="http://127.0.0.1:8001",
        model_ids=("alias-a", "alias-b"),
    )
    app = _blocked_console_app((detected,))
    host = ConsoleHarness(app)

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _detected_action_focused(host, console, pilot)
        await pilot.press("enter")
        switcher = await _switcher_ready(host, pilot)
        await host.workers.wait_for_complete()
        await pilot.pause()

        lines = list_lines(host, switcher)
        for model in detected.model_ids:
            assert "legacy alias" in line_with(lines, model)
        row = switcher.highlighted_row()
        assert row is not None
        assert (row.provider, row.model) == ("local_llamacpp", "alias-a")


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


@pytest.mark.asyncio
@private_profile_test
async def test_a_card_resumed_with_nothing_detected_runs_discovery_again(
    request,
) -> None:
    """Controller ruling of 2026-10-12 (the final review's item 1, Option A).

    Discovery ran once per screen, and the Console screen is reused, so a
    server started after the first look was never offered until Chatbook
    restarted. Coming back to Console while the card still blocks and nothing
    was found runs it again; once a server is found, a return does not."""
    app = _blocked_console_app(())
    host = ConsoleHarness(app)

    async def leave_and_come_back(pilot) -> None:
        await host.push_screen(Screen())
        await pilot.pause()
        await host.pop_screen()
        await pilot.pause()

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _first_look_done(app, console, pilot)

        async def server_started(config):
            app._discovery_calls.append(config)
            return (_DETECTED_SERVER,)

        app.console_local_server_discovery = server_started
        await leave_and_come_back(pilot)
        detected = await _detected_action_focused(host, console, pilot)
        assert str(detected.label) == "Use detected llama.cpp (127.0.0.1:8080)"
        assert len(app._discovery_calls) == 2

        await leave_and_come_back(pilot)
        for _ in range(10):
            await pilot.pause(0.05)
        assert len(app._discovery_calls) == 2


@pytest.mark.asyncio
@private_profile_test
async def test_a_card_unblocked_and_broken_while_away_runs_discovery_on_return(
    monkeypatch,
    request,
) -> None:
    """Final fix wave 3, item 2. Discovery left the card with nothing found;
    the card then unblocked without a resume (an apply whose callback runs on
    the screen) and broke again while the user was away (a key removed in
    Settings). The return runs discovery again.

    A plain return re-syncs the Console before the resume's own code runs
    (the attach reconciliation), so the card already reads blocking there.
    The resume's code read the card's state before its own sync, so where it
    runs without that re-sync first it read "not blocking", kept discovery
    spent, and the sync then found the card blocking. It no longer reads it."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _blocked_console_app(())
    host = ConsoleHarness(app)
    openai = app.app_config["api_settings"]["openai"]

    async with host.run_test(size=(211, 44)) as pilot:
        console = host.screen_stack[-1]
        await _first_look_done(app, console, pilot)
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id

        async def unblock() -> None:
            # An apply's callback: a key and a model, and the card's own sync.
            openai["api_key"] = "sk-test-key-for-the-card"
            store.replace_session_settings(
                session_id,
                replace(
                    store.session_settings(session_id),
                    provider="openai",
                    model="gpt-x",
                    source="user",
                ),
            )
            console._sync_console_transcript_guidance()
            await _until(pilot, lambda: not modal.is_blocking, "the card to unlock")

        await unblock()
        await host.push_screen(Screen())
        await pilot.pause()
        openai["api_key"] = ""  # the key removed in Settings
        await host.pop_screen()
        await _until(pilot, lambda: modal.is_blocking, "the card to block again")
        await _until(
            pilot, lambda: len(app._discovery_calls) == 2, "discovery to run again"
        )

        await unblock()
        openai["api_key"] = ""
        console.on_screen_resume()  # its own code, with no re-sync before it
        await _until(
            pilot, lambda: len(app._discovery_calls) == 3, "discovery on the resume"
        )
