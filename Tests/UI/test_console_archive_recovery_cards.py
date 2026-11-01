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


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_sync_waits_for_recovery_root_mount_then_initializes_controls(
    monkeypatch, concurrent
):
    import asyncio
    from types import SimpleNamespace

    from textual.containers import Vertical

    from tldw_chatbook.UI.Console_Modules.archive_recovery import (
        ConsoleArchiveRecoveryView,
    )

    class RecoveryRoot(Vertical):
        async def on_mount(self):
            assert not self.is_mounted
            await self.app.view.sync()
            assert not self.query("#console-archive-recovery-controls")
            self.app.checked_pending_mount = True

    class MountApp(App):
        checked_pending_mount = False

        def compose(self):
            store = SimpleNamespace(
                active_session_id="source", messages_for_session=lambda _: []
            )
            runtime = SimpleNamespace(
                presentations=lambda _: [],
                coordinator=SimpleNamespace(
                    snapshot=lambda _: None, receipts=lambda _: []
                ),
            )
            self.screen._console_chat_store = store
            self.screen._console_runtime = lambda: SimpleNamespace(
                ensure_archive_recovery=lambda: runtime
            )
            self.view = ConsoleArchiveRecoveryView(self.screen)
            yield RecoveryRoot(id="console-archive-recovery")

    app = MountApp()
    async with app.run_test():
        assert app.checked_pending_mount
        assert app.query_one("#console-archive-recovery").is_mounted
        if concurrent:
            root = app.query_one("#console-archive-recovery")
            mount = root.mount
            entered, release = asyncio.Event(), asyncio.Event()
            mount_calls = []

            async def blocked_mount(*args, **kwargs):
                mount_calls.append(args)
                entered.set()
                await release.wait()
                await mount(*args, **kwargs)

            monkeypatch.setattr(root, "mount", blocked_mount)
            first = asyncio.create_task(app.view.sync())
            await asyncio.wait_for(entered.wait(), 2)
            second = asyncio.create_task(app.view.sync())
            try:
                await asyncio.sleep(0)
                assert len(mount_calls) == 1
            finally:
                release.set()
                await asyncio.gather(first, second)
        else:
            await app.view.sync()
        assert len(app.query(ArchiveRecoveryConfirmCard)) == 1
        assert app.query_one(ArchiveRecoveryConfirmCard).is_mounted
        assert app.view._session_id == "source"


@pytest.mark.asyncio
async def test_sync_ignores_recovery_root_during_app_exit():
    from textual.containers import Vertical

    from tldw_chatbook.UI.Console_Modules.archive_recovery import (
        ConsoleArchiveRecoveryView,
    )

    class MountApp(App):
        def compose(self):
            yield Vertical(id="console-archive-recovery")

    app = MountApp()
    async with app.run_test():
        root = app.query_one("#console-archive-recovery")
        assert root.is_mounted and not root._pruning and not root._closing
        view = ConsoleArchiveRecoveryView(app.screen)
        app._exit = True
        try:
            assert not root.is_attached
            await view.sync()
            assert not root.children
        finally:
            app._exit = False


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_during_mount", [False, True])
async def test_recovery_controls_are_inert_when_app_exits(
    monkeypatch, exit_during_mount
):
    from types import SimpleNamespace

    from textual.containers import Vertical

    from tldw_chatbook.UI.Console_Modules.archive_recovery import (
        ConsoleArchiveRecoveryView,
    )

    class MountApp(App):
        def compose(self):
            self.screen._console_chat_store = SimpleNamespace(
                active_session_id="source", messages_for_session=lambda _: []
            )
            runtime = SimpleNamespace(
                presentations=lambda _: [],
                coordinator=SimpleNamespace(
                    snapshot=lambda _: None, receipts=lambda _: []
                ),
            )
            self.screen._console_runtime = lambda: SimpleNamespace(
                view=self.screen, ensure_archive_recovery=lambda: runtime
            )
            self.view = ConsoleArchiveRecoveryView(self.screen)
            yield Vertical(id="console-archive-recovery")

    app = MountApp()
    async with app.run_test():
        root = app.query_one("#console-archive-recovery")
        mount = root.mount

        async def mount_then_exit(*args, **kwargs):
            await mount(*args, **kwargs)
            app._exit = True

        if exit_during_mount:
            monkeypatch.setattr(root, "mount", mount_then_exit)
        try:
            await app.view.sync()
            if exit_during_mount:
                assert not app.query(ArchiveRecoveryConfirmCard)
            else:
                assert app.view.is_answerable("source")
                app._exit = True
            assert not app.view.is_answerable("source")
        finally:
            app._exit = False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,own_operation", [("busy", False), ("expired", False), ("expired", True)]
)
async def test_receipt_only_retires_its_own_mounted_confirmation(status, own_operation):
    from types import SimpleNamespace

    from textual.containers import Vertical

    from tldw_chatbook.Chat.archive_recovery.contracts import RecoveryOperationSnapshot
    from tldw_chatbook.Chat.archive_recovery.operations import recovery_outcome
    from tldw_chatbook.UI.Console_Modules.archive_recovery import (
        ConsoleArchiveRecoveryView,
    )

    pending = confirmation()
    snapshot = RecoveryOperationSnapshot(
        "operation", "source", "awaiting_confirmation", pending, None
    )
    receipts = []

    class ReceiptApp(App):
        def compose(self):
            self.screen._console_chat_store = SimpleNamespace(
                active_session_id="source", messages_for_session=lambda _: []
            )
            runtime = SimpleNamespace(
                presentations=lambda _: [],
                coordinator=SimpleNamespace(
                    snapshot=lambda _: snapshot, receipts=lambda _: receipts
                ),
            )
            self.screen._console_runtime = lambda: SimpleNamespace(
                view=self.screen, ensure_archive_recovery=lambda: runtime
            )
            self.view = ConsoleArchiveRecoveryView(self.screen)
            with Vertical(id="console-archive-recovery"):
                yield Button("Refresh", id="recovery-refresh")

    app = ReceiptApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await app.view.sync()
        card = app.query_one(ArchiveRecoveryConfirmCard)
        scope = app.query_one("#recovery-scope", Select)
        scope.value = "chat_and_workspace"
        await pilot.pause()
        button = app.query_one("#recovery-confirm", Button)
        button.focus()
        await pilot.pause()
        assert app.focused is button
        assert card.confirmation is pending and card.display
        receipt = RecoveryOperationSnapshot(
            pending.operation_id if own_operation else "other-operation",
            "source",
            "settled",
            None,
            recovery_outcome(status, pending.target, next_action="refresh"),
        )
        assert await app.view.show_receipt(receipt)
        receipts.append(receipt)
        if own_operation:
            snapshot = None
            assert card.confirmation is None and not card.display
            assert app.focused is app.query_one("#recovery-refresh")
        else:
            assert card.confirmation is pending
            assert card.display and card.visible
            assert scope.value == "chat_and_workspace"
            assert app.focused is button
            await app.view.sync()
            await pilot.pause()
            assert card.confirmation is pending
            assert card.display and card.visible
            assert scope.value == "chat_and_workspace"
            assert app.focused is button
            assert not button.disabled
