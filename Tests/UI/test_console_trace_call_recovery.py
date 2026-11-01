"""Mounted recovery actions for pre-dispatch trace persistence failures."""

from __future__ import annotations

import asyncio

import pytest
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_context_budget_copy import MODEL_WINDOW_SETTING
from tldw_chatbook.Chat.console_trace_call_recovery import TraceCallProviderContact
from tldw_chatbook.UI.Console_Modules.trace_call_recovery import (
    TraceCallRecoveryCallout,
    TraceCallRecoveryState,
    dispatch_trace_call_recovery_action,
)

pytestmark = pytest.mark.bootstrap_profile


class _Controller:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def retry_library_preparation(self, preparation_id: str) -> object:
        self.actions.append(("retry", preparation_id))
        self.started.set()
        await self.release.wait()
        return object()

    async def send_without_capture(self, preparation_id: str) -> object:
        self.actions.append(("send_without_capture", preparation_id))
        self.started.set()
        await self.release.wait()
        return object()

    def cancel_library_preparation(self, preparation_id: str) -> object:
        self.actions.append(("cancel", preparation_id))
        return object()


class _RecoveryApp(ConsolidatedCSSApp):
    def __init__(self, controller: _Controller) -> None:
        super().__init__()
        self.controller = controller

    def compose(self) -> ComposeResult:
        yield TraceCallRecoveryCallout(
            # TASK-33621.5 AC#2: the controller records whether the send
            # reached the provider; this card's send never crossed dispatch.
            state=TraceCallRecoveryState(
                "preparation-1", provider_contact=TraceCallProviderContact("no")
            ),
            on_action=lambda action, preparation_id: (
                dispatch_trace_call_recovery_action(
                    self.controller, action, preparation_id
                )
            ),
        )


async def test_trace_recovery_callout_is_wrapped_labeled_and_keyboard_focusable() -> (
    None
):
    controller = _Controller()
    app = _RecoveryApp(controller)
    async with app.run_test(size=(34, 18)) as pilot:
        await pilot.pause()
        rendered = "\n".join(
            str(widget.render())
            for widget in app.screen.query("TraceCallRecoveryCallout *")
            if hasattr(widget, "render")
        )
        assert "Owner: This Console send" in rendered
        assert "Problem: Trace capture could not be saved." in rendered
        assert "Impact: The provider was not contacted." in rendered
        assert "Retry capture" in rendered
        assert "Send without capture" in rendered
        assert "Cancel send" in rendered

        retry = app.screen.query_one("#console-trace-retry", Button)
        retry.focus()
        await pilot.pause()
        assert app.focused is retry
        await pilot.press("enter")
        await asyncio.wait_for(controller.started.wait(), timeout=1)
        assert all(button.disabled for button in app.screen.query(Button))
        controller.release.set()
        await app.workers.wait_for_complete()


async def test_trace_recovery_action_is_consumed_before_double_press() -> None:
    controller = _Controller()
    app = _RecoveryApp(controller)
    async with app.run_test(size=(52, 16)) as pilot:
        retry = app.screen.query_one("#console-trace-retry", Button)
        retry.focus()
        await pilot.press("enter", "enter")
        await asyncio.wait_for(controller.started.wait(), timeout=1)
        await pilot.pause()

        assert controller.actions == [("retry", "preparation-1")]
        status = app.screen.query_one("#console-trace-status", Static)
        assert "Working" in str(status.render())
        controller.release.set()
        await app.workers.wait_for_complete()


async def test_trace_recovery_cancel_calls_terminal_controller_entrypoint() -> None:
    controller = _Controller()
    app = _RecoveryApp(controller)
    async with app.run_test(size=(42, 18)) as pilot:
        cancel = app.screen.query_one("#console-trace-cancel", Button)
        cancel.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        assert controller.actions == [("cancel", "preparation-1")]


async def test_trace_recovery_refusal_reenables_all_actions() -> None:
    class RefusingApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield TraceCallRecoveryCallout(
                state=TraceCallRecoveryState("preparation-1"),
                on_action=lambda *_args: False,
            )

    app = RefusingApp()
    async with app.run_test(size=(42, 18)) as pilot:
        retry = app.screen.query_one("#console-trace-retry", Button)
        retry.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        callout = app.screen.query_one(TraceCallRecoveryCallout)
        assert callout.display
        assert all(not button.disabled for button in callout.query(Button))
        assert "did not complete" in str(
            callout.query_one("#console-trace-status", Static).render()
        )


async def test_trace_refusal_copy_survives_transcript_sync_and_is_painted() -> None:
    """An unchanged trace pause must display the controller's actual refusal."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
    from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
        ProviderContinuationTranscriptRegion,
    )

    class RefusingRegionApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield ProviderContinuationTranscriptRegion(
                session_surface_builder=lambda: Static("Conversation"),
                recovery_message_builder=lambda: None,
                on_recovery_action=lambda *_args: False,
                trace_recovery_state_builder=lambda: TraceCallRecoveryState(
                    "preparation-1"
                ),
                on_trace_recovery_action=lambda *_args: ConsoleSubmitResult(
                    False, False, "Trace storage is not writable."
                ),
            )

    app = RefusingRegionApp()
    async with app.run_test(size=(80, 32)) as pilot:
        card = app.screen.query_one(TraceCallRecoveryCallout)
        card.query_one("#console-trace-retry", Button).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.screen.query_one(ProviderContinuationTranscriptRegion).sync_recovery()
        await pilot.pause()
        assert card.display
        assert "Trace storage is not writable." in str(
            card.query_one("#console-trace-status", Static).render()
        )
        assert (
            "Trace&#160;storage&#160;is&#160;not&#160;writable."
            in app.export_screenshot()
        )


# --- TASK-34350: the same card holds a send at the compaction threshold ---


class _HoldController:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str]] = []

    async def compact_and_send(self, preparation_id: str) -> object:
        self.actions.append(("compact_and_send", preparation_id))
        return object()

    async def send_without_compacting(self, preparation_id: str) -> object:
        self.actions.append(("send_without_compacting", preparation_id))
        return object()

    def cancel_library_preparation(self, preparation_id: str) -> object:
        self.actions.append(("cancel", preparation_id))
        return object()


class _HoldApp(ConsolidatedCSSApp):
    def __init__(self, controller: _HoldController, *, estimated: bool) -> None:
        super().__init__()
        self.controller = controller
        self.estimated = estimated

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Chat.console_turn_preparation import ContextCompactionHold

        yield TraceCallRecoveryCallout(
            state=TraceCallRecoveryState(
                "preparation-1",
                context_hold=ContextCompactionHold(
                    session_id="session-1",
                    used_tokens=1_650,
                    trigger_tokens=1_440,
                    budget_tokens=1_800,
                    estimated=self.estimated,
                ),
            ),
            on_action=lambda action, preparation_id: (
                dispatch_trace_call_recovery_action(
                    self.controller, action, preparation_id
                )
            ),
        )


def _painted(app) -> str:
    return "\n".join(
        str(widget.render())
        for widget in app.screen.query("TraceCallRecoveryCallout *")
        if hasattr(widget, "render") and widget.display
    )


async def test_a_compaction_hold_shows_its_numbers_and_four_choices() -> None:
    app = _HoldApp(_HoldController(), estimated=False)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        painted = _painted(app)
        assert "1,650 of 1,800 tokens" in painted
        # TASK-33621.4 AC#1: the trigger, and /rewind's own preview.
        assert "1,440 tokens" in painted
        assert "/rewind" in painted
        assert "Nothing was sent" in painted
        visible = {
            str(button.label) for button in app.screen.query(Button) if button.display
        }
        assert visible == {
            "Compact and send",
            "Send without compacting",
            "Change policy",
            "Cancel send",
        }
        assert "Problem:" not in painted
        assert "Owner:" not in painted
        assert "estimate" not in painted


async def test_change_policy_opens_chat_settings_at_context_and_memory() -> None:
    """TASK-33621.4 AC#1: a one-step route to the policy, not a recovery action:
    the send stays held, the card stays usable, the controller is not called."""
    controller = _HoldController()
    app = _HoldApp(controller, estimated=False)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        opened: list[dict] = []

        async def open_settings(**kwargs) -> bool:
            opened.append(kwargs)
            return True

        app.screen._open_console_settings = open_settings
        card = app.screen.query_one(TraceCallRecoveryCallout)
        card.query_one("#console-trace-change-policy", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert opened == [{"focus_context": True}]
        assert controller.actions == []
        assert card.display
        assert not card.query_one("#console-trace-compact-send", Button).disabled


async def test_change_policy_is_offered_only_for_a_compaction_hold() -> None:
    app = _RecoveryApp(_Controller())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        card = app.screen.query_one(TraceCallRecoveryCallout)
        assert not card.query_one("#console-trace-change-policy", Button).display
        assert "Owner: This Console send" in _painted(app)


async def test_a_compaction_hold_on_an_estimated_window_says_so() -> None:
    app = _HoldApp(_HoldController(), estimated=True)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # Rewritten on purpose (capture finding 17): the pointer is the
        # constant both Chat settings views spell.
        assert MODEL_WINDOW_SETTING in _painted(app)


async def test_each_compaction_hold_button_reaches_its_controller_action() -> None:
    for button_id, action in (
        ("#console-trace-compact-send", "compact_and_send"),
        ("#console-trace-send-uncompacted", "send_without_compacting"),
        ("#console-trace-cancel", "cancel"),
    ):
        controller = _HoldController()
        app = _HoldApp(controller, estimated=False)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app.screen.query_one(button_id, Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
        assert controller.actions == [(action, "preparation-1")]


@pytest.mark.parametrize("kind", ["trace", "continuation"])
@pytest.mark.parametrize("cleared", [False, True])
async def test_recovery_sync_waits_for_nested_mount_and_replays_current_state(
    monkeypatch, kind, cleared
):
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
        ProviderContinuationRecoveryCallout,
        ProviderContinuationTranscriptRegion,
    )

    def message(identity, version):
        return ConsoleChatMessage(
            id=identity,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            generation_projection_quarantined=True,
            generation_projection_quarantine_version=version,
            generation_projection_quarantine_reason="Reload current generation.",
        )

    trace = [TraceCallRecoveryState("initial") if kind == "trace" else None]
    continuation = [message("initial", 1) if kind == "continuation" else None]
    latest_trace = TraceCallRecoveryState("latest", temporary_capture=True)
    latest_message = message("latest", 2)
    actions = []
    region = ProviderContinuationTranscriptRegion(
        session_surface_builder=lambda: Static("Conversation"),
        recovery_message_builder=lambda: continuation[0],
        on_recovery_action=lambda *args: (
            actions.append(("continuation", *args)) or False
        ),
        trace_recovery_state_builder=lambda: trace[0],
        on_trace_recovery_action=lambda *args: (
            actions.append(("trace", *args)) or False
        ),
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original_mount = Widget.mount_composed_widgets
    held_id = f"console-{kind}-actions"

    async def hold_action_children(owner, widgets):
        if owner.id == held_id:
            entered.set()
            await asyncio.wait_for(release.wait(), timeout=5)
        await original_mount(owner, widgets)

    monkeypatch.setattr(Widget, "mount_composed_widgets", hold_action_children)

    class MountOrderApp(ConsolidatedCSSApp):
        def compose(self):
            yield region

    async def publish_during_mount():
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            assert not region.is_mounted
            card_type = (
                TraceCallRecoveryCallout
                if kind == "trace"
                else ProviderContinuationRecoveryCallout
            )
            card = region.query_one(card_type)
            assert card.query_one(f"#console-{kind}-title", Static)
            assert not card.query(Button), "the original nested buttons must be pending"
            trace[0] = latest_trace if kind == "trace" and not cleared else None
            continuation[0] = (
                latest_message if kind == "continuation" and not cleared else None
            )
            region.sync_recovery()
        finally:
            release.set()

    probe = asyncio.create_task(publish_during_mount())
    app = MountOrderApp()
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await probe
            await pilot.pause()
            trace_card = region.query_one(TraceCallRecoveryCallout)
            continuation_card = region.query_one(ProviderContinuationRecoveryCallout)
            if cleared:
                assert not trace_card.display and not continuation_card.display
                trace[0] = latest_trace if kind == "trace" else None
                continuation[0] = latest_message if kind == "continuation" else None
                region.sync_recovery()
            if kind == "trace":
                assert trace_card.recovery_state is latest_trace
                target = trace_card.query_one("#console-trace-save-send", Button)
                expected = ("trace", "save_and_send", "latest")
            else:
                assert continuation_card.recovery_state.message_id == "latest"
                assert continuation_card.recovery_state.message_version == 2
                target = continuation_card.query_one(
                    "#console-continuation-reload", Button
                )
                expected = ("continuation", "reload", "latest", 2)
            assert target.display and not target.disabled
            target.focus()
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            assert actions == [expected]
            assert region.is_mounted
            await trace_card.remove()
            with pytest.raises(NoMatches):
                region.sync_recovery()
    finally:
        release.set()
        await asyncio.gather(probe, return_exceptions=True)
