"""Original Console decision widgets backed by the selected command owner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ...Chat.console_execution_process import ConsoleChannelLost, ConsoleExecutionClient
from ...Chat.console_execution_decision_head import ConsoleDecisionHead
from ...Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from ...Widgets.Console.console_project_instructions import (
    ProjectInstructionNoticeModal, ProjectInstructionSetupModal, ProjectInstructionSetupResult,
)
from ..Screens.chat_screen import ChatScreen
from .execution_projection import _definition, _require_methods
from .execution_state import selected_client


_SCREEN_METHODS = tuple(
    (name, _definition(getattr(ChatScreen, name)))
    for name in (
        "_set_console_pending_decision", "set_task_resume_state", "sync_task_resume_state",
    )
)
_CLIENT_METHODS = tuple(
    (name, _definition(getattr(ConsoleExecutionClient, name)))
    for name in ("submit", "read_decision_head")
)


@dataclass(eq=False, slots=True)
class _ProjectModal:
    client: object
    screen: object
    visit: int
    source_revision: int
    head: object
    modal: object
    resolving: bool = False


def is_owned_project_modal(client, screen, visit: int) -> bool:
    """Recognize only this exact original modal above its owning Console."""
    state = client._project_modal
    if type(state) is not _ProjectModal:
        return False
    stack = client._app.screen_stack
    return bool(
        state.client is client and state.screen is screen and state.visit == visit
        and client._view is screen and client._visit == visit
        and state.source_revision == client._source_revision
        and type(state.modal) in (ProjectInstructionNoticeModal, ProjectInstructionSetupModal)
        and len(stack) >= 2 and stack[-1] is state.modal and stack[-2] is screen
        and screen._session._project_instruction_decision_modals.get(state.head.decision_id)
        is state.modal
    )


def _current(screen, client, visit) -> None:
    if selected_client(screen) is not client or client._visit != visit:
        raise ConsoleChannelLost("Console decision presentation visit changed")
    _require_methods(screen, _SCREEN_METHODS)
    _require_methods(client, _CLIENT_METHODS)


async def present_selected_decision(screen, client, visit: int) -> bool:
    """Render the exact published mixed head through the original task cards."""
    _current(screen, client, visit)
    projection = _current_head(client, visit)
    decision = projection.decision
    if decision is not None and decision.session_id != projection.session_id:
        raise RuntimeError("Console decision presentation session changed")
    shown = screen._set_console_pending_decision(decision)
    if shown and decision is not None and decision.decision_type == "approval" and screen.is_current:
        call_ids = tuple(call["call_id"] for call in decision.payload["calls"])
        key = (visit, client._source_revision, projection.session_id, projection.incarnation_id, decision.decision_id, call_ids)
        if (
            decision.payload.get("phase") != "finishing"
            and getattr(screen, "_console_process_answerable_approval", None) != key
        ):
            receipt = client.submit("ResolveDecision", {"decision": {
                "kind": "ApprovalPresentationDecision", "round_id": decision.decision_id,
                "call_ids": list(call_ids), "attachment_id": client._attachment_id, "visit": visit,
            }}, session_id=projection.session_id, incarnation_id=projection.incarnation_id)
            try:
                answerable = await asyncio.shield(receipt.applied)
            finally:
                if await asyncio.shield(receipt.retired) is not True:
                    raise ConsoleChannelLost("Console approval presentation retirement is unproven")
            _current(screen, client, visit)
            if answerable.get("applied") is True:
                screen._console_process_answerable_approval = key
    else:
        screen._console_process_answerable_approval = None
    head = projection.project
    state = client._project_modal
    if state is not None:
        if type(state) is not _ProjectModal:
            raise RuntimeError("Console project modal owner changed")
        if state.resolving:
            return shown
        if state.head == head and is_owned_project_modal(client, screen, visit):
            return shown
        _dismiss_project_modal(state)
    if head is None or not screen.is_current:
        return shown
    modal = (
        ProjectInstructionNoticeModal(head.notice)
        if head.kind == "dispatch" else ProjectInstructionSetupModal(head.choices)
    )
    state = _ProjectModal(client, screen, visit, client._source_revision, head, modal)
    modal._console_process_modal_owner = state
    client._project_modal = state
    screen._session._project_instruction_decision_modals[head.decision_id] = modal

    def finish(result):
        _finish_project_modal(state, result)

    try:
        hidden = _project_checkpoint(state, "hidden")
        screen.run_worker(
            _observe_resolution(screen, client, visit, hidden, refresh=False),
            group="console-process-decision", exclusive=False,
        )
        await screen.app.push_screen(modal, callback=finish)
        if client._project_modal is not state:
            return shown
        _current(screen, client, visit)
        receipt = _project_checkpoint(state, "mounted")
        try:
            mounted = await asyncio.shield(receipt.applied)
        finally:
            if await asyncio.shield(receipt.retired) is not True:
                raise ConsoleChannelLost("Console project mount retirement is unproven")
        if mounted.get("applied") is not True and client._project_modal is state:
            _dismiss_project_modal(state)
    except BaseException:
        if client._project_modal is state:
            _dismiss_project_modal(state)
        raise
    return shown


def _current_head(client, visit):
    head = client.decision_head
    frame = client._projection
    if (
        type(head) is not ConsoleDecisionHead
        or (head.source_revision, head.session_id, head.incarnation_id,
            head.attachment_id, head.visit) != (
            client._source_revision, frame.session_id, frame.incarnation_id,
            client._attachment_id, visit,
        )
    ):
        raise ConsoleChannelLost("Console decision publication scope changed")
    return head


def _project_checkpoint(state, action):
    return state.client.submit(
        "ProjectDecisionPresentation",
        {
            "attachment_id": state.client._attachment_id, "visit": state.visit,
            "decision_id": state.head.decision_id, "action": action,
        },
        session_id=state.head.session_id, incarnation_id=state.head.incarnation_id,
    )


def _forget_project_modal(state) -> None:
    state.resolving = True
    if state.client._project_modal is state:
        state.client._project_modal = None
    modals = state.screen._session._project_instruction_decision_modals
    if modals.get(state.head.decision_id) is state.modal:
        del modals[state.head.decision_id]
    state.modal._console_process_modal_owner = None


def project_modal_visibility_changed(modal, visible: bool) -> None:
    """Report only the actual registered original modal's Textual visibility."""
    state = getattr(modal, "_console_process_modal_owner", None)
    if type(state) is not _ProjectModal or state.modal is not modal:
        return
    client, screen = state.client, state.screen
    if (
        client._project_modal is not state or client._view is not screen
        or client._visit != state.visit or client._shutdown_task is not None
    ):
        return
    if visible and (
        not modal.is_mounted or not is_owned_project_modal(client, screen, state.visit)
    ):
        return
    receipt = _project_checkpoint(state, "mounted" if visible else "hidden")
    screen.run_worker(
        _observe_resolution(screen, client, state.visit, receipt, refresh=False),
        group="console-process-decision", exclusive=False,
    )


def _dismiss_project_modal(state) -> None:
    """Retire stale display only; a dismissal never decides its original head."""
    state.resolving = True
    receipt = _project_checkpoint(state, "hidden")
    state.screen.run_worker(
        _observe_resolution(state.screen, state.client, state.visit, receipt, refresh=False),
        group="console-process-decision", exclusive=False,
    )
    result = ProjectInstructionSetupResult("cancel") if state.head.kind == "binding" else "cancel"
    if state.modal.is_mounted:
        state.modal.dismiss(result)
    else:
        _forget_project_modal(state)


def _finish_project_modal(state, result) -> None:
    client, screen = state.client, state.screen
    if client._project_modal is not state:
        return
    if state.resolving:
        _forget_project_modal(state)
        if screen.is_current and client._view is screen and client._visit == state.visit:
            from .execution_state import refresh_selected_view

            screen.run_worker(
                refresh_selected_view(screen, checkpoint="visible"),
                group="console-process-decision", exclusive=False,
            )
        return
    _forget_project_modal(state)
    hidden = _project_checkpoint(state, "hidden")
    screen.run_worker(
        _observe_resolution(screen, client, state.visit, hidden, refresh=False),
        group="console-process-decision", exclusive=False,
    )
    if client._view is not screen or client._visit != state.visit or not screen.is_current:
        return
    _current(screen, client, state.visit)
    if client._source_revision != state.source_revision:
        return
    if state.head.kind == "dispatch":
        if result not in {"proceed", "cancel", "disable"}:
            return
        choice = {"kind": "ProjectDispatchDecision", "decision_id": state.head.decision_id, "action": result}
    else:
        if type(result) is not ProjectInstructionSetupResult:
            return
        choice = {
            "kind": "ProjectBindingDecision", "decision_id": state.head.decision_id,
            "action": result.action, "binding_id": result.binding_id,
        }
    receipt = client.submit(
        "ResolveDecision", {"decision": choice},
        session_id=state.head.session_id, incarnation_id=state.head.incarnation_id,
    )
    screen.run_worker(
        _observe_resolution(screen, client, state.visit, receipt, checkpoint="visible"),
        group="console-process-decision", exclusive=False,
    )


def handle_selected_approval_event(screen, event) -> bool:
    """Forward a real card's captured round and ordered call IDs unchanged.

    An event for a replaced head cannot decide the currently shown round.
    The backend still checks the original round registry and permission owner.
    """
    client = selected_client(screen)
    if client is None:
        return False
    _current(screen, client, client._visit)
    if type(event) is not ChatApprovalCard.ApprovalDecided:
        raise TypeError("Original Console approval decision event is required")
    projection = _current_head(client, client._visit)
    decision = projection.decision
    if (
        decision is None or decision.decision_type != "approval"
        or decision.decision_id != event.round_id
    ):
        return True
    payload = decision.payload
    if payload["round_id"] != event.round_id:
        raise RuntimeError("Console approval round presentation changed")
    call_ids = tuple(call["call_id"] for call in payload["calls"])
    if (
        not call_ids or any(not call_id for call_id in call_ids)
        or len(set(call_ids)) != len(call_ids)
        or set(event.decisions) - set(call_ids)
    ):
        raise RuntimeError("Console approval call identity changed")
    # These are the existing card's explicit choices, never inferred grants.
    receipt = client.submit(
        "ResolveDecision",
        {"decision": {
            "kind": "ApprovalRoundDecision",
            "round_id": event.round_id,
            "call_ids": list(call_ids),
            "decisions": dict(event.decisions),
            "denial_reasons": dict(event.decisions.denial_reasons),
        }},
        session_id=projection.session_id,
        incarnation_id=projection.incarnation_id,
    )
    screen.run_worker(
        _observe_resolution(screen, client, client._visit, receipt),
        group="console-process-decision", exclusive=False,
    )
    return True


async def _observe_resolution(screen, client, visit, receipt, *, refresh=True, checkpoint=None) -> None:
    """Join the issued decision before reconciling its actual current view."""
    try:
        await asyncio.shield(receipt.applied)
    finally:
        if await asyncio.shield(receipt.retired) is not True:
            raise ConsoleChannelLost("Console decision retirement is unproven")
    if refresh and client._view is screen and client._visit == visit and screen.is_current:
        _current(screen, client, visit)
        from .execution_state import refresh_selected_view, start_selected_decision_refresh

        if checkpoint is not None:
            # Restoring actual visibility retains its original FULL lifecycle
            # and native retirement. The finite decision read cannot seed it.
            await refresh_selected_view(screen, checkpoint=checkpoint)
        else:
            start_selected_decision_refresh(screen, force=True)
