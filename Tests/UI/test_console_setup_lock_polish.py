"""TASK-2154.8: Console setup/lock-state polish (FR-03, FR-06, FR-09, FR-10).

Pins the four lock-state fixes from the 2026-08 Console UX review:

- FR-03: the quiet empty transcript offers the provider recovery action in
  place when the provider is blocked (and never when nothing is broken or the
  blocking setup card already covers the transcript).
- FR-06: the footer hides its misleading "Enter send" hint while the setup
  modal locks the composer, swapping in the accurate "continue setup" hint.
- FR-09: typing while the composer is locked raises one informational toast
  per blocking episode instead of vanishing silently.
- FR-10: the setup card explains "provider" in plain terms via a subtitle.
"""

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_CARD_SUBTITLE,
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


def _widget_text(widget) -> str:
    renderable = getattr(widget, "renderable", None)
    if renderable is None:
        return str(widget.render())
    return str(getattr(renderable, "plain", renderable))


class EmptyTranscriptActionHarness(ConsolidatedCSSApp):
    """Bare transcript harness recording Workbench action requests."""

    def __init__(self):
        super().__init__()
        self.workbench_actions: list[str] = []

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")

    def on_workbench_action_requested(self, event: WorkbenchActionRequested) -> None:
        event.stop()
        self.workbench_actions.append(event.action_id)


class SetupModalActionHarness(ConsolidatedCSSApp):
    """Bare setup-modal harness recording actions and notifications."""

    def __init__(self, state: ConsoleSetupCardState):
        super().__init__()
        self._state = state
        self.workbench_actions: list[str] = []
        self.notifications: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")

    async def on_mount(self) -> None:
        self.query_one("#console-setup-modal", ConsoleSetupModal).sync_card_state(
            self._state,
            action_label="Configure API",
            action_tooltip="Open provider settings.",
        )

    def on_workbench_action_requested(self, event: WorkbenchActionRequested) -> None:
        event.stop()
        self.workbench_actions.append(event.action_id)

    def notify(self, message, *, title="", severity="information", timeout=None):
        # Record instead of raising real toasts; the toast machinery is not
        # what is under test here.
        self.notifications.append((str(message), severity))


class ConsoleHarness(ConsolidatedCSSApp):
    """Real ChatScreen harness (mirrors test_console_workbench_contract)."""

    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _blocked_openai_app():
    """Test app whose Console provider is blocked (empty OpenAI key)."""
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4.1-2025-04-14"},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    return app


async def _wait_for(predicate, pilot, attempts: int = 300) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition was not met in time")


# ---------------------------------------------------------------------------
# FR-03: empty transcript recovery action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quiet_empty_state_with_blocker_offers_recovery_action():
    app = EmptyTranscriptActionHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.sync_empty_state(
            ConsoleSetupCardState(mode="quiet"),
            provider_action_label="Configure API",
            provider_action_tooltip="Open provider settings.",
        )
        await pilot.pause()
        await pilot.pause()

        action = transcript.query_one("#console-empty-provider-action", Button)
        assert _widget_text(action) == "Configure API"
        assert action.tooltip == "Open provider settings."

        await pilot.click("#console-empty-provider-action")
        await pilot.pause()
        assert app.workbench_actions == ["provider-recovery"]


@pytest.mark.asyncio
async def test_empty_state_hides_action_when_not_blocked_or_covered():
    app = EmptyTranscriptActionHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)

        # Provider ready (no recovery label synced): no button, no dead end.
        transcript.sync_empty_state(ConsoleSetupCardState(mode="quiet"))
        await pilot.pause()
        await pilot.pause()
        assert not transcript.query("#console-empty-provider-action")

        # Blocking card mode: the modal covers the transcript, so the in-panel
        # action must not render even though a label exists.
        transcript.sync_empty_state(
            ConsoleSetupCardState(
                mode="card",
                steps=(ConsoleSetupStep(state="active", label="Add an API key"),),
            ),
            provider_action_label="Configure API",
            provider_action_tooltip="Open provider settings.",
        )
        await pilot.pause()
        await pilot.pause()
        assert not transcript.query("#console-empty-provider-action")

        # Transition back to a blocked quiet state: the action appears.
        transcript.sync_empty_state(
            ConsoleSetupCardState(mode="quiet"),
            provider_action_label="Configure API",
        )
        await pilot.pause()
        await pilot.pause()
        assert transcript.query("#console-empty-provider-action")

        # And is removed again when the block clears.
        transcript.sync_empty_state(ConsoleSetupCardState(mode="quiet"))
        await pilot.pause()
        await pilot.pause()
        assert not transcript.query("#console-empty-provider-action")


# ---------------------------------------------------------------------------
# FR-09: typing while the composer is locked yields visible feedback
# ---------------------------------------------------------------------------


def _blocking_card_state() -> ConsoleSetupCardState:
    return ConsoleSetupCardState(
        mode="card",
        steps=(
            ConsoleSetupStep(state="active", label="Add an API key"),
            ConsoleSetupStep(state="pending", label="Pick a model"),
        ),
    )


@pytest.mark.asyncio
async def test_typing_while_locked_toasts_once_per_blocking_episode():
    app = SetupModalActionHarness(_blocking_card_state())

    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.is_blocking
        modal.focus_primary_action()
        await pilot.pause()

        await pilot.press("x")
        await pilot.pause()
        assert len(app.notifications) == 1
        message, severity = app.notifications[0]
        assert severity == "information"
        assert "Typing is locked" in message

        # Further typing in the same episode does not spam toasts.
        await pilot.press("x", "j", "1")
        await pilot.pause()
        assert len(app.notifications) == 1

        # Enter still activates the card's primary action (unchanged behavior).
        await pilot.press("enter")
        await pilot.pause()
        assert app.workbench_actions == ["provider-recovery"]

        # Block lifts: typing is no longer consumed and no toast fires.
        modal.sync_card_state(ConsoleSetupCardState(mode="quiet"))
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()
        assert len(app.notifications) == 1

        # A NEW blocking episode re-arms the one-shot toast.
        modal.sync_card_state(
            _blocking_card_state(),
            action_label="Configure API",
            action_tooltip="Open provider settings.",
        )
        modal.focus_primary_action()
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()
        assert len(app.notifications) == 2


# ---------------------------------------------------------------------------
# FR-10: the setup card explains "provider" in plain terms
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_card_shows_plain_language_provider_subtitle():
    app = SetupModalActionHarness(_blocking_card_state())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        subtitle = app.query_one("#console-setup-modal-subtitle", Static)
        assert subtitle.display is True
        assert _widget_text(subtitle) == CONSOLE_SETUP_CARD_SUBTITLE

        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        modal.sync_card_state(ConsoleSetupCardState(mode="quiet"))
        await pilot.pause()
        assert subtitle.display is False


# ---------------------------------------------------------------------------
# FR-06: footer send hint hidden while the composer is locked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_footer_hides_send_hint_while_setup_locks_composer(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    host = ConsoleHarness(_blocked_openai_app())

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for(
            lambda: console._console_setup_modal_blocking(), pilot
        )
        footer = console.query_one(AppFooterStatus)
        await _wait_for(
            lambda: "continue setup" in footer.shortcut_text, pilot
        )
        assert "Enter send" not in footer.shortcut_text

        # Block lifts (or never applied): the normal send hint returns.
        monkeypatch.setattr(
            console, "_console_setup_modal_blocking", lambda: False
        )
        console._register_console_footer_shortcuts()
        await pilot.pause()
        assert "continue setup" not in footer.shortcut_text
        assert "Enter send" in footer.shortcut_text

        # Re-blocked: swapped again.
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: True)
        console._register_console_footer_shortcuts()
        await pilot.pause()
        assert "continue setup" in footer.shortcut_text
        assert "Enter send" not in footer.shortcut_text
