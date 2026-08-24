from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from textual.widgets import Static

import tldw_chatbook.Chat.console_chat_models as recovery_models
from tldw_chatbook.Chat.console_chat_models import ConsoleControllerActivity
from tldw_chatbook.Chat.console_prompt_queue import PromptQueueSnapshot
from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    derive_prompt_queue_presentation,
)


def _model_symbols():
    required = (
        "ConsoleDispatchRecoveryAction",
        "ConsoleDispatchRecoveryActionId",
        "ConsoleDispatchRecoveryKind",
        "ConsoleDispatchRecoveryState",
    )
    missing = [name for name in required if not hasattr(recovery_models, name)]
    assert not missing, f"dispatch recovery model is missing: {', '.join(missing)}"
    return tuple(getattr(recovery_models, name) for name in required)


def _ui_module():
    try:
        module = importlib.import_module(
            "tldw_chatbook.UI.Console_Modules.dispatch_recovery"
        )
    except ModuleNotFoundError:
        pytest.fail("Textual dispatch recovery projection is missing", pytrace=False)
    required = (
        "ConsoleDispatchRecoveryRegion",
        "derive_dispatch_recovery_presentation",
    )
    missing = [name for name in required if not hasattr(module, name)]
    assert not missing, f"dispatch recovery UI is missing: {', '.join(missing)}"
    return module


def _state(*, started: bool = False, in_flight: bool = False, remote: bool = False):
    action_type, action_id, kind, state_type = _model_symbols()
    if remote:
        return state_type(
            kind=(kind.REMOTE_DISPATCH_STARTED if started else kind.REMOTE_ACCEPTED),
            assistant_message_id="assistant-1",
            conversation_id="conversation-1",
            visible_copy=(
                "Response delivery status is unknown on the source device."
                if started
                else "Response accepted on another device; waiting for dispatch."
            ),
            actions=(),
        )
    actions = (
        action_type(
            action_id=(action_id.RETRY_ANYWAY if started else action_id.RETRY_RESPONSE),
            label="Retry anyway" if started else "Retry response",
            enabled=not in_flight,
            disabled_reason="Recovery action is already in progress."
            if in_flight
            else "",
        ),
        action_type(
            action_id=action_id.DISCARD,
            label="Discard",
            enabled=not in_flight,
            disabled_reason="Recovery action is already in progress."
            if in_flight
            else "",
        ),
    )
    return state_type(
        kind=kind.DISPATCH_STARTED if started else kind.ACCEPTED,
        assistant_message_id="assistant-1",
        conversation_id="conversation-1",
        visible_copy=(
            "Response delivery status is unknown on the source device."
            if started
            else "Response accepted; waiting for dispatch."
        ),
        warning=(
            "Retry anyway may send a duplicate request because delivery status "
            "is unknown."
            if started
            else ""
        ),
        actions=actions,
        in_flight=in_flight,
    )


def test_presentation_projects_model_owned_accepted_labels_literally() -> None:
    module = _ui_module()

    presentation = module.derive_dispatch_recovery_presentation(_state())

    assert presentation.visible is True
    assert presentation.visible_copy == "Response accepted; waiting for dispatch."
    assert presentation.warning == ""
    assert presentation.markup is False
    assert [
        (action.action_id, action.label, action.enabled, action.disabled_reason)
        for action in presentation.actions
    ] == [
        ("retry_response", "Retry response", True, ""),
        ("discard", "Discard", True, ""),
    ]


def test_dispatch_started_projects_literal_duplicate_warning() -> None:
    module = _ui_module()

    presentation = module.derive_dispatch_recovery_presentation(_state(started=True))

    assert presentation.visible_copy == (
        "Response delivery status is unknown on the source device."
    )
    assert presentation.warning == (
        "Retry anyway may send a duplicate request because delivery status is unknown."
    )
    assert [action.label for action in presentation.actions] == [
        "Retry anyway",
        "Discard",
    ]


@pytest.mark.parametrize("started", [False, True])
def test_in_flight_state_disables_repeated_actions_idempotently(started: bool) -> None:
    module = _ui_module()
    state = _state(started=started, in_flight=True)

    first = module.derive_dispatch_recovery_presentation(state)
    second = module.derive_dispatch_recovery_presentation(state)

    assert first == second
    assert all(action.enabled is False for action in first.actions)
    assert {action.disabled_reason for action in first.actions} == {
        "Recovery action is already in progress."
    }


@pytest.mark.parametrize("started", [False, True])
def test_remote_or_imported_recovery_is_visible_but_has_no_source_action(
    started: bool,
) -> None:
    module = _ui_module()

    presentation = module.derive_dispatch_recovery_presentation(
        _state(started=started, remote=True)
    )

    assert presentation.visible is True
    assert presentation.actions == ()
    assert presentation.markup is False


def test_widget_constructs_dynamic_copy_with_markup_disabled() -> None:
    module = _ui_module()
    region = module.ConsoleDispatchRecoveryRegion(_state(started=True))

    children = list(region.compose())
    dynamic = [child for child in children if isinstance(child, Static)]

    assert len(dynamic) == 2
    assert all(item._render_markup is False for item in dynamic)
    assert str(dynamic[0].render()) == (
        "Response delivery status is unknown on the source device."
    )
    assert str(dynamic[1].render()) == (
        "Retry anyway may send a duplicate request because delivery status is unknown."
    )


def _empty_queue_snapshot() -> PromptQueueSnapshot:
    return PromptQueueSnapshot(
        session_id="session-1",
        revision=0,
        entries=(),
        waiting_count=0,
        claimed_count=0,
        total_count=0,
        mode=SimpleNamespace(value="paused"),
        pause_reason=None,
        reservation=SimpleNamespace(value="released"),
        expected_context_epoch=0,
        closing=False,
    )


def _activity() -> ConsoleControllerActivity:
    return ConsoleControllerActivity(
        session_id="session-1",
        occupies_slot=False,
        preparing_before_acceptance=False,
        accepted_live_turn=True,
        needs_approval=False,
        queued_count=0,
        queue_paused=True,
        terminal_notification_eligible=False,
    )


def test_queue_presentation_cannot_offer_resume_while_dispatch_recovery_blocks() -> (
    None
):
    state = _state()

    presentation = derive_prompt_queue_presentation(
        _empty_queue_snapshot(),
        _activity(),
        dispatch_recovery=state,
    )

    assert presentation.state_label == "Response accepted; waiting for dispatch."
    assert presentation.pause_enabled is False
    assert presentation.primary_action == "dispatch-recovery"


def test_unreconstructable_reason_is_model_owned_and_markup_neutral() -> None:
    action_type, action_id, kind, state_type = _model_symbols()
    reason = (
        "Retry response is unavailable because one-shot prefill or transient evidence "
        "cannot be reconstructed exactly."
    )
    state = state_type(
        kind=kind.ACCEPTED,
        assistant_message_id="assistant-1",
        conversation_id="conversation-1",
        visible_copy="Response accepted; waiting for dispatch.",
        actions=(
            action_type(
                action_id=action_id.RETRY_RESPONSE,
                label="Retry response",
                enabled=False,
                disabled_reason=reason,
            ),
            action_type(
                action_id=action_id.DISCARD,
                label="Discard",
                enabled=True,
            ),
        ),
    )

    presentation = _ui_module().derive_dispatch_recovery_presentation(state)

    assert presentation.actions[0].disabled_reason == reason
    assert presentation.markup is False
