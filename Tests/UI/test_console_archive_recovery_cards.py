"""Mounted one-shot archive recovery controls."""

import time

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Select

from tldw_chatbook.Chat.archive_recovery.contracts import (
    ArchiveTarget,
    ArchiveWorkspaceSnapshot,
    RecoveryConfirmation,
)
from tldw_chatbook.Widgets.Chat_Widgets.archive_recovery_cards import (
    ArchiveRecoveryConfirmCard,
)

# UI autouse fixtures import the app, whose config participant binds at collection.
pytestmark = pytest.mark.bootstrap_profile


class RecoveryCardApp(App):
    def __init__(self, confirmation):
        super().__init__()
        self.confirmation = confirmation
        self.decisions = []

    def compose(self) -> ComposeResult:
        yield ArchiveRecoveryConfirmCard(self.confirmation)

    def on_archive_recovery_confirm_card_recovery_decided(self, event):
        self.decisions.append(event)


def confirmation(*, expired=False):
    return RecoveryConfirmation(
        "request",
        "operation",
        "source",
        ArchiveTarget(
            "original",
            3,
            "[bold]Original\x1b title",
            True,
            "project",
            ArchiveWorkspaceSnapshot("project", "Project", True, None, 1),
        ),
        ("chat_only", "chat_and_workspace"),
        None,
        None,
        time.monotonic() + (-1 if expired else 300),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("keyboard", [False, True])
async def test_visible_scope_is_exact_one_shot_decision(keyboard):
    app = RecoveryCardApp(confirmation())
    async with app.run_test(size=(70, 30)) as pilot:
        app.query_one("#recovery-scope", Select).value = "chat_and_workspace"
        await pilot.pause()
        if keyboard:
            app.query_one("#recovery-scope").focus()
            await pilot.press("tab", "enter")
        else:
            await pilot.click("#recovery-confirm")
        await pilot.pause()
        assert len(app.decisions) == 1
        event = app.decisions[0]
        assert (event.session_id, event.request_id, event.allow, event.scope) == (
            "source",
            "request",
            True,
            "chat_and_workspace",
        )
        assert app.query_one("#recovery-confirm", Button).disabled
        assert not app.query("Checkbox")


@pytest.mark.asyncio
async def test_expired_card_cannot_emit_allow():
    app = RecoveryCardApp(confirmation(expired=True))
    async with app.run_test() as pilot:
        await pilot.click("#recovery-confirm")
        assert not app.decisions


@pytest.mark.asyncio
async def test_queued_old_control_event_cannot_answer_replacement(monkeypatch):
    from dataclasses import replace

    from tldw_chatbook.Widgets.Chat_Widgets.archive_recovery_cards import (
        RecoveryDecisionButton,
    )

    app = RecoveryCardApp(confirmation())
    async with app.run_test() as pilot:
        card = app.query_one(ArchiveRecoveryConfirmCard)
        app.query_one("#recovery-scope", Select).value = "chat_only"
        await pilot.pause()
        emitted = []
        button = app.query_one("#recovery-confirm", RecoveryDecisionButton)
        with monkeypatch.context() as capture:
            capture.setattr(button, "post_message", emitted.append)
            button.press()
        pressed = [
            event
            for event in emitted
            if isinstance(event, RecoveryDecisionButton.Pressed)
        ]
        assert len(pressed) == 1
        old_event = pressed[0]
        card.set_confirmation(
            replace(
                confirmation(),
                request_id="replacement",
                session_id="other",
                selected_scope="chat_only",
            )
        )
        card.post_message(old_event)
        await pilot.pause()
        assert not app.decisions
        # Native Button ignores both mouse and Enter while its active effect runs.
        ready_deadline = time.monotonic() + 2
        while button.has_class("-active"):
            assert time.monotonic() < ready_deadline
            await pilot.pause(0.05)
        button.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.decisions) == 1
        assert (app.decisions[0].session_id, app.decisions[0].request_id) == (
            "other",
            "replacement",
        )


@pytest.mark.asyncio
async def test_stale_scope_event_cannot_override_visible_choice():
    app = RecoveryCardApp(confirmation())
    async with app.run_test() as pilot:
        card = app.query_one(ArchiveRecoveryConfirmCard)
        scope = app.query_one("#recovery-scope", Select)
        from dataclasses import replace

        old_event = Select.Changed(scope, "chat_only")
        card.set_confirmation(
            replace(
                confirmation(),
                request_id="replacement",
                session_id="other",
                selected_scope="chat_and_workspace",
            )
        )
        card.post_message(old_event)
        await pilot.pause()
        assert "other active chats" in str(app.query_one("#recovery-effects").content)
        await pilot.click("#recovery-confirm")
        assert app.decisions[0].scope == scope.value == "chat_and_workspace"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["result", "imported"])
async def test_library_recovery_navigation_can_await_a_leave_dialog(route):
    import asyncio
    from types import SimpleNamespace

    from textual import on
    from textual.screen import ModalScreen

    from tldw_chatbook.UI.Console_Modules.archive_recovery import (
        ConsoleArchiveRecoverySurface,
        ConsoleArchiveRecoveryView,
    )
    from tldw_chatbook.Widgets.Chat_Widgets.archive_recovery_cards import (
        ArchiveRecoveryResultsCard,
    )

    completed = asyncio.Event()

    class LeaveDialog(ModalScreen[bool]):
        def compose(self):
            yield Button("Leave", id="confirm-leave")

        @on(Button.Pressed)
        def confirm(self):
            self.dismiss(True)

    class NavigationApp(App):
        async def navigate(self, *args, **kwargs):
            assert await self.push_screen_wait(LeaveDialog())
            completed.set()

        def compose(self):
            view = ConsoleArchiveRecoveryView(SimpleNamespace(app=self))
            view.open_library = self.navigate
            view.is_answerable = lambda session_id: session_id == "source"
            button = Button("Search Library", id="recovery-manual-library")
            button.source_session_id = "source"
            with ConsoleArchiveRecoverySurface(view):
                yield button

    app = NavigationApp()
    async with app.run_test(size=(80, 25)) as pilot:
        surface = app.query_one(ConsoleArchiveRecoverySurface)
        if route == "result":
            surface.post_message(
                ArchiveRecoveryResultsCard.ActionRequested(
                    "source",
                    "page",
                    None,
                    "library",
                )
            )
        else:
            app.query_one("#recovery-manual-library", Button).press()
        async with asyncio.timeout(2):
            while not isinstance(app.screen, LeaveDialog):
                await pilot.pause(0.01)
        await pilot.click("#confirm-leave")
        await asyncio.wait_for(completed.wait(), timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("lose_during_mount", [False, True])
async def test_queued_results_retire_on_source_loss_and_remount_on_return(
    monkeypatch, lose_during_mount
):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveSearchRequest
    from tldw_chatbook.Chat.archive_recovery.runtime import ArchiveRecoveryPresentation
    from tldw_chatbook.UI.Console_Modules.archive_recovery import (
        ConsoleArchiveRecoverySurface,
        ConsoleArchiveRecoveryView,
    )
    from tldw_chatbook.Widgets.Chat_Widgets.archive_recovery_cards import (
        ArchiveRecoveryResultsCard,
    )

    view = ConsoleArchiveRecoveryView(SimpleNamespace())

    class ResultApp(App):
        def compose(self):
            yield ConsoleArchiveRecoverySurface(view)

    app = ResultApp()
    answerable = lose_during_mount
    view._session_id = "source"
    view.is_answerable = lambda session_id: answerable
    record = ArchiveRecoveryPresentation(
        "source",
        ArchiveSearchRequest(),
        {"result_set_id": "page", "items": [], "cursor": None},
        False,
        0,
    )
    async with app.run_test(size=(80, 25)):
        surface = app.query_one(ConsoleArchiveRecoverySurface)
        view._surface = lambda: surface
        mount = surface.mount

        async def mount_and_lose_source(*args, **kwargs):
            nonlocal answerable
            await mount(*args, **kwargs)
            answerable = False

        monkeypatch.setattr(surface, "mount", mount_and_lose_source)
        await view._mount_results(record)
        assert not surface.query(ArchiveRecoveryResultsCard)
        assert "page" not in view._rendered_sets
        answerable = True
        monkeypatch.setattr(surface, "mount", mount)
        await view._mount_results(record)
        assert len(surface.query(ArchiveRecoveryResultsCard)) == 1
