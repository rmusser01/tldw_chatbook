"""Visible and fail-closed admission for temporary Capture-On sends."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_chat_controller import (
    CapturePolicySnapshot,
    CapturePurgeAvailability,
)
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    CapturePolicyResolution,
    CapturePolicySource,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    TemporaryCaptureRequiresSave,
    require_durable_capture_admission,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
from tldw_chatbook.UI.Console_Modules.trace_call_recovery import (
    TraceCallRecoveryCallout,
    dispatch_trace_call_recovery_action,
    trace_call_recovery_state,
)
from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import (
    CapturePolicyBindings,
    ConsoleCapturePolicyDialog,
)


def test_gateway_rejects_temporary_capture_on_before_adapter_admission() -> None:
    with pytest.raises(TemporaryCaptureRequiresSave, match="Save & Send"):
        require_durable_capture_admission(
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            ephemeral=True,
        )

    require_durable_capture_admission(
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_OFF,
        ephemeral=True,
    )


class _TemporaryRecoveryApp(ConsolidatedCSSApp):
    def __init__(self, on_action) -> None:
        super().__init__()
        preparation = SimpleNamespace(
            preparation_id="preparation-1",
            state=ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.TEMPORARY_CAPTURE,
        )
        self.state = trace_call_recovery_state(preparation)
        self.on_action = on_action

    def compose(self) -> ComposeResult:
        yield TraceCallRecoveryCallout(
            state=self.state,
            on_action=self.on_action,
        )


async def test_temporary_capture_callout_offers_save_or_explicit_capture_off() -> None:
    actions: list[tuple[str, str]] = []
    app = _TemporaryRecoveryApp(
        lambda action, preparation_id: actions.append((action, preparation_id))
        or True
    )
    async with app.run_test(size=(52, 18)) as pilot:
        await pilot.pause()
        save = app.screen.query_one("#console-trace-save-send", Button)
        retry = app.screen.query_one("#console-trace-retry", Button)
        capture_off = app.screen.query_one("#console-trace-send-without", Button)
        rendered = "\n".join(
            str(widget.render())
            for widget in app.screen.query("TraceCallRecoveryCallout Static")
        )

        assert save.display and capture_off.display
        assert not retry.display
        assert "Temporary chats cannot store durable captures" in rendered
        save.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert actions == [("save_and_send", "preparation-1")]


async def test_save_and_send_dispatches_only_the_explicit_controller_action() -> None:
    class Controller:
        def __init__(self) -> None:
            self.actions: list[str] = []

        async def save_and_send(self, preparation_id: str) -> bool:
            self.actions.append(preparation_id)
            return True

    controller = Controller()
    result = await dispatch_trace_call_recovery_action(
        controller,
        "save_and_send",
        "preparation-1",
    )

    assert result is True
    assert controller.actions == ["preparation-1"]


class _DialogApp(ConsolidatedCSSApp):
    pass


async def test_capture_policy_explains_temporary_save_and_send_boundary() -> None:
    snapshot = CapturePolicySnapshot(
        session_id="temporary-session",
        conversation_id=None,
        conversation_title="Temporary chat",
        enabled=True,
        next_detail=None,
        conversation_detail=CaptureDetail.SAFE,
        global_detail=CaptureDetail.SAFE,
        effective=CapturePolicyResolution(
            True,
            CaptureDetail.SAFE,
            CapturePolicySource.CONVERSATION,
            (),
        ),
        policy_revision=1,
        config_generation=1,
        capture_revision=1,
        active_run_detail=None,
        queued_consumer=False,
        save_pending=False,
        error_code=None,
    )

    async def count_full() -> int:
        return 0

    bindings = CapturePolicyBindings(
        target_session_id="temporary-session",
        target_conversation_id=None,
        read=lambda: snapshot,
        apply_next=lambda *_args: None,  # type: ignore[arg-type]
        apply_conversation=lambda *_args: None,  # type: ignore[arg-type]
        apply_global=lambda *_args: None,  # type: ignore[arg-type]
        count_full=count_full,
        purge_full=lambda *_args: None,  # type: ignore[arg-type]
        capture_revision=lambda: 1,
        purge_availability=lambda: CapturePurgeAvailability(False, "target_missing"),
    )
    app = _DialogApp()
    async with app.run_test() as pilot:
        await app.push_screen(ConsoleCapturePolicyDialog(bindings))
        await pilot.pause()
        guidance = app.screen.query_one(
            "#capture-policy-temporary-guidance", Static
        )

        assert "Save & Send" in str(guidance.render())
        assert "Send without capture" in str(guidance.render())
