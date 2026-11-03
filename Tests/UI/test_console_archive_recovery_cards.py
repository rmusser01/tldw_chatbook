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
@pytest.mark.parametrize("boundary", ["compose", "scope_mount"])
@pytest.mark.parametrize("updates", ["empty", "replace", "clear"])
async def test_pending_controls_render_latest_confirmation_on_mount(boundary, updates):
    from dataclasses import replace

    from textual.events import Compose, Mount

    from tldw_chatbook.Widgets.Chat_Widgets.archive_recovery_cards import (
        RecoveryDecisionButton,
    )

    first = confirmation()
    latest = replace(
        first,
        session_id="latest-source",
        request_id="latest-request",
        selected_scope="chat_only",
        target=replace(first.target, title="Latest target"),
    )
    payloads = {
        "empty": [None],
        "replace": [first, latest],
        "clear": [first, latest, None],
    }[updates]
    card = ArchiveRecoveryConfirmCard()
    observed = []

    class PendingCardApp(RecoveryCardApp):
        def compose(self):
            yield card

    def update_pending_controls(message):
        sender = message._sender
        at_boundary = (
            isinstance(message, Compose) and sender is card
            if boundary == "compose"
            else isinstance(message, Mount)
            and isinstance(sender, Select)
            and sender.id == "recovery-scope"
        )
        if not at_boundary:
            return
        assert not card.is_mounted
        for payload in payloads:
            assert card.set_confirmation(payload) is False
        observed.append(True)

    app = PendingCardApp(None)
    async with app.run_test(message_hook=update_pending_controls) as pilot:
        assert observed == [True]
        assert not app.decisions
        expected = latest if updates == "replace" else None
        assert card.confirmation is expected
        assert card.display == (expected is not None)
        button = card.query_one("#recovery-confirm", RecoveryDecisionButton)
        for control in card.query(RecoveryDecisionButton):
            assert control.is_mounted
            assert control.binding == (
                (latest.session_id, latest.request_id) if expected else None
            )
        if expected is None:
            assert button.disabled
            return
        assert card.query_one("#recovery-scope", Select).value == "chat_only"
        assert str(button.label) == "Confirm restore · Latest target"
        assert not button.disabled
        await pilot.click("#recovery-confirm")
        await pilot.pause()
        assert len(app.decisions) == 1
        decision = app.decisions[0]
        assert (
            decision.session_id,
            decision.request_id,
            decision.allow,
            decision.scope,
        ) == ("latest-source", "latest-request", True, "chat_only")
        assert button.disabled


@pytest.mark.asyncio
async def test_queryable_retiring_confirmation_clears_without_children():
    from textual.containers import Vertical

    from tldw_chatbook.UI.Console_Modules.archive_recovery import (
        ConsoleArchiveRecoveryView,
    )

    observed = []

    class RetiringCard(ArchiveRecoveryConfirmCard):
        def on_unmount(self):
            assert self.is_mounted
            assert not self.children
            assert self.app.view._confirmation() is self
            observed.append(self.app.view.show_confirmation(None))
            assert self.confirmation is None
            assert not self.display

    class RetiringCardApp(RecoveryCardApp):
        def compose(self):
            self.view = ConsoleArchiveRecoveryView(self.screen)
            with Vertical(id="console-archive-recovery-controls"):
                yield RetiringCard(self.confirmation)

    app = RetiringCardApp(confirmation())
    async with app.run_test():
        await app.query_one(RetiringCard).remove()
        assert observed == [False]
        assert not app.decisions


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["target_removed", "scope_unmount"])
async def test_partial_controls_cannot_acknowledge_or_answer_replacement(boundary):
    from dataclasses import replace
    from types import SimpleNamespace

    from textual.containers import Vertical
    from textual.events import Unmount

    from tldw_chatbook.UI.Console_Modules.archive_recovery import (
        ConsoleArchiveRecoveryView,
    )
    from tldw_chatbook.Widgets.Chat_Widgets.archive_recovery_cards import (
        RecoveryDecisionButton,
    )

    first = replace(confirmation(), selected_scope="chat_only")
    latest = replace(first, request_id="replacement")
    observed = []
    card = scope = old_event = None

    class PartialCardApp(RecoveryCardApp):
        def compose(self):
            self.screen._console_chat_store = SimpleNamespace(
                active_session_id="source"
            )
            self.screen._console_runtime = lambda: SimpleNamespace(view=self.screen)
            self.view = ConsoleArchiveRecoveryView(self.screen)
            with Vertical(id="console-archive-recovery-controls"):
                yield ArchiveRecoveryConfirmCard(first)

    def check_replacement():
        assert app.view.is_answerable("source")
        observed.append(app.view.show_confirmation(latest))
        assert observed == [False]
        assert card.confirmation is latest
        card.post_message(old_event)

    def observe_retirement(message):
        if (
            boundary == "scope_unmount"
            and isinstance(message, Unmount)
            and message._sender is scope
        ):
            assert scope.is_mounted
            assert not scope.children
            assert len(card.children) == 6
            check_replacement()

    app = PartialCardApp(first)
    async with app.run_test(message_hook=observe_retirement) as pilot:
        card = app.query_one(ArchiveRecoveryConfirmCard)
        scope = card.query_one("#recovery-scope", Select)
        button = card.query_one("#recovery-confirm", RecoveryDecisionButton)
        old_event = RecoveryDecisionButton.Pressed(button)
        if boundary == "target_removed":
            await card.query_one("#recovery-target").remove()
            check_replacement()
        else:
            await scope.remove()
        await pilot.pause()
        assert observed == [False]
        assert not app.decisions


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


@pytest.mark.asyncio
@pytest.mark.parametrize("during", ["controls", "imported"])
async def test_sync_stops_when_its_chat_is_replaced_mid_sync(monkeypatch, during):
    """TASK-38652: a sync must not read or write for a chat replaced under it.

    Console's first-chat handoff switches to its own chat and closes the
    untouched one, and it can run while ``sync`` awaits a surface mount.
    During the controls mount, the old sync then asked the store for the
    closed chat's messages and raised KeyError, which ends the app. During
    the imported-recovery mount, it cleared the confirmation just presented
    for the new chat. The next sync must show the new chat.
    """
    import json
    from dataclasses import replace
    from types import SimpleNamespace

    from textual.containers import Vertical

    from tldw_chatbook.Chat.archive_recovery.contracts import RecoveryOperationSnapshot
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.UI.Console_Modules.archive_recovery import (
        ConsoleArchiveRecoverySurface,
        ConsoleArchiveRecoveryView,
    )

    store = ConsoleChatStore()
    untouched = store.create_session()
    handoff = store.create_session(activate=False)
    if during == "imported":
        store.append_message(
            untouched.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=json.dumps({"result_set_id": "page", "result_ref": "ref"}),
        )
    pending = replace(confirmation(), session_id=handoff.id)
    snapshots = {
        handoff.id: RecoveryOperationSnapshot(
            "operation", handoff.id, "awaiting_confirmation", pending, None
        )
    }

    class HandoffApp(App):
        def compose(self):
            self.screen._console_chat_store = store
            runtime = SimpleNamespace(
                presentations=lambda _: (),
                coordinator=SimpleNamespace(
                    snapshot=snapshots.get, receipts=lambda _: ()
                ),
            )
            self.screen._console_runtime = lambda: SimpleNamespace(
                view=self.screen, ensure_archive_recovery=lambda: runtime
            )
            self.view = ConsoleArchiveRecoveryView(self.screen)
            yield Vertical(id="console-archive-recovery")

    app = HandoffApp()
    mount = ConsoleArchiveRecoverySurface.mount
    trigger = ArchiveRecoveryConfirmCard if during == "controls" else Vertical

    async def mount_then_hand_off(surface, *widgets, **kwargs):
        await mount(surface, *widgets, **kwargs)
        if store.active_session_id == untouched.id and type(widgets[0]) is trigger:
            store.switch_session(handoff.id)
            store.close_session(untouched.id)
            if during == "imported":
                assert app.view.show_confirmation(pending)

    monkeypatch.setattr(ConsoleArchiveRecoverySurface, "mount", mount_then_hand_off)
    async with app.run_test(size=(80, 30)):
        await app.view.sync()
        assert store.active_session_id == handoff.id
        if during == "imported":
            assert app.query_one(ArchiveRecoveryConfirmCard).confirmation is pending

        await app.view.sync()
        assert app.view._session_id == handoff.id
        card = app.query_one(ArchiveRecoveryConfirmCard)
        assert card.confirmation is pending and card.display
        assert not app.query("#recovery-manual-library")
