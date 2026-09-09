"""Task 7 (group onboarding): first-run hand-off to Library notes.

task-32140: neither the wizard Summary step nor the Console "Get started"
card offered a path into Notes for a local-first user without a provider
configured -- everything pointed at provider setup. These tests pin the new
"Write your first note" / "Write a note in Library" affordances and their
routing to Library's New note view (LIBRARY_NAV_CONTEXT_NOTES_CREATE), the
same destination the existing command-palette "new_note" quick action uses.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import ComposeResult
from textual.widgets import Button

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_NOTES_ACTION_LABEL,
)
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_NOTES_CREATE, TAB_LIBRARY
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_NOTES_ACTION_ID,
    CONSOLE_SETUP_MODAL_NOTES_WORKBENCH_ACTION,
    ConsoleSetupModal,
)


def test_wizard_summary_offers_write_your_first_note() -> None:
    """The Summary step offers a notes path beside "Add your first document"."""
    from tldw_chatbook.UI.Wizards import FirstRunSetupWizard as wizard_module

    source = inspect.getsource(wizard_module)
    assert '"Write your first note", id="setup-exit-library-notes"' in source
    assert '@on(Button.Pressed, "#setup-exit-library-notes")' in source


def test_wizard_exit_route_notes_navigates_to_library_new_note() -> None:
    """The Summary's exit_route sentinel resolves to Library's New note view."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import EXIT_ROUTE_LIBRARY_NOTES

    events: list[str] = []
    worker_coroutines: list = []

    async def record_navigation(_message) -> None:
        events.append("navigation-complete")

    def capture_worker(work, **kwargs) -> None:
        worker_coroutines.append(work)

    receiver = SimpleNamespace(
        current_tab=TAB_LIBRARY,
        handle_screen_navigation=AsyncMock(side_effect=record_navigation),
        _schedule_startup_model_catalog_refresh=MagicMock(),
        post_message=MagicMock(
            side_effect=AssertionError("completed navigation must use its worker")
        ),
        run_worker=capture_worker,
    )
    try:
        TldwCli._handle_first_run_wizard_result(
            receiver,
            {"completed": True, "exit_route": EXIT_ROUTE_LIBRARY_NOTES},
        )

        assert len(worker_coroutines) == 1
        import asyncio

        asyncio.run(worker_coroutines[0])

        receiver.handle_screen_navigation.assert_awaited_once()
        message = receiver.handle_screen_navigation.await_args.args[0]
        assert isinstance(message, NavigateToScreen)
        assert message.screen_name == TAB_LIBRARY
        assert message.screen_context == {LIBRARY_NAV_CONTEXT_NOTES_CREATE: True}
        assert events == ["navigation-complete"]
    finally:
        for worker in worker_coroutines:
            worker.close()


def test_wizard_exit_route_notes_dropped_without_completion() -> None:
    """An incomplete/cancelled result never navigates (untrusted-payload guard)."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import EXIT_ROUTE_LIBRARY_NOTES

    receiver = SimpleNamespace(post_message=MagicMock())
    TldwCli._handle_first_run_wizard_result(
        receiver,
        {"completed": False, "exit_route": EXIT_ROUTE_LIBRARY_NOTES},
    )
    receiver.post_message.assert_not_called()


class _SetupModalHarness(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.workbench_actions: list[str] = []

    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")

    async def on_mount(self) -> None:
        from tldw_chatbook.Chat.console_onboarding_state import (
            ConsoleSetupCardState,
            ConsoleSetupStep,
        )

        self.query_one("#console-setup-modal", ConsoleSetupModal).sync_card_state(
            ConsoleSetupCardState(
                mode="card",
                steps=(
                    ConsoleSetupStep(
                        state="active",
                        label="Connect a provider (API key or local server)",
                    ),
                    ConsoleSetupStep(state="pending", label="Pick a model"),
                    ConsoleSetupStep(state="pending", label="Send your first message"),
                ),
            ),
            action_label="Set up provider",
            action_tooltip="Open provider settings.",
        )

    def on_workbench_action_requested(self, event: WorkbenchActionRequested) -> None:
        event.stop()
        self.workbench_actions.append(event.action_id)


@pytest.mark.asyncio
async def test_setup_modal_offers_write_a_note_action() -> None:
    """A no-provider profile still gets a real, needs-no-provider first action."""
    app = _SetupModalHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notes_button = app.query_one(f"#{CONSOLE_SETUP_MODAL_NOTES_ACTION_ID}", Button)
        assert notes_button.display is True
        assert str(notes_button.label) == CONSOLE_SETUP_NOTES_ACTION_LABEL

        notes_button.press()
        await pilot.pause()
        assert app.workbench_actions == [CONSOLE_SETUP_MODAL_NOTES_WORKBENCH_ACTION]


@pytest.mark.asyncio
async def test_console_workbench_action_write_note_navigates_to_library() -> None:
    """ChatScreen routes the modal's notes action to Library's New note view."""
    receiver = SimpleNamespace(post_message=MagicMock())
    event = WorkbenchActionRequested(CONSOLE_SETUP_MODAL_NOTES_WORKBENCH_ACTION)

    await ChatScreen.on_console_workbench_action_requested(receiver, event)

    receiver.post_message.assert_called_once()
    message = receiver.post_message.call_args.args[0]
    assert isinstance(message, NavigateToScreen)
    assert message.screen_name == TAB_LIBRARY
    assert message.screen_context == {LIBRARY_NAV_CONTEXT_NOTES_CREATE: True}
