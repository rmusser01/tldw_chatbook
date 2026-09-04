from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tldw_chatbook.Canvas.models import CanvasConflictResult, CanvasScope
from tldw_chatbook.Chat.console_canvas_controller import (
    CanvasRunState,
    ConsoleCanvasController,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscript,
    canvas_card_presentations,
)

SESSION_ID = "session-1"
CONVERSATION_ID = "conversation-1"
RUN_ID = "run-1"
ASSISTANT_ID = "assistant-1"
SOURCE_SENTINEL = "CANVAS_SOURCE_SENTINEL_3_3"


def _scope(**changes: object) -> CanvasScope:
    values = {
        "session_id": SESSION_ID,
        "conversation_id": CONVERSATION_ID,
        "active_message_ids": ("user-1",),
        "selected_canvas_id": None,
        "selected_revision_id": None,
        "run_id": RUN_ID,
    }
    values.update(changes)
    return CanvasScope(**values)


def _controller() -> ConsoleCanvasController:
    controller = ConsoleCanvasController()
    controller.register_run(
        _scope(), assistant_message_id=ASSISTANT_ID, temporary=False
    )
    return controller


def test_successful_finalization_exposes_one_source_private_contribution() -> None:
    controller = _controller()
    result = controller.create_canvas(
        _scope(),
        tool_call_id="call-1",
        title="Status board",
        html=f"<main>{SOURCE_SENTINEL}</main>",
    )

    settlement = controller.finish_run(RUN_ID, "done")

    assert settlement is not None
    assert settlement.state is CanvasRunState.READY
    assert settlement.cards[0].revision_id == result.revision.revision_id
    assert settlement.cards[0].status == "updated"
    assert SOURCE_SENTINEL not in settlement.metadata_json
    assert SOURCE_SENTINEL not in repr(settlement)
    assert settlement.contribution.revision_count == 1


def test_canvas_only_success_keeps_the_registered_assistant_anchor() -> None:
    controller = _controller()
    controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Only Canvas", html="<p>x</p>"
    )

    settlement = controller.finish_run(RUN_ID, "done")

    assert settlement is not None
    assert settlement.assistant_message_id == ASSISTANT_ID
    assert settlement.contribution.origin_message_id == ASSISTANT_ID


def test_successful_run_without_canvas_calls_can_close_without_a_contribution() -> None:
    controller = _controller()

    settlement = controller.finish_run(RUN_ID, "done")

    assert settlement is not None
    assert settlement.state is CanvasRunState.READY
    assert settlement.contribution is None
    assert controller.confirm_settlement(RUN_ID) is True
    assert controller.settlement_for_assistant(ASSISTANT_ID).state is (
        CanvasRunState.COMMITTED
    )


@pytest.mark.parametrize("terminal", ["cancelled", "error", "stuck"])
def test_non_success_terminal_discards_exact_run_once(terminal: str) -> None:
    controller = _controller()
    controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Draft", html="<p>draft</p>"
    )

    first = controller.finish_run(RUN_ID, terminal)
    duplicate = controller.finish_run(RUN_ID, terminal)

    assert first is duplicate
    assert first is not None
    assert first.state is CanvasRunState.DISCARDED
    assert first.cards[0].status == "discarded"
    assert first.cards[0].reopenable is False
    assert first.contribution is None


def test_same_tool_call_replay_is_idempotent() -> None:
    controller = _controller()
    first = controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Draft", html="<p>x</p>"
    )
    second = controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Draft", html="<p>x</p>"
    )

    assert second == first
    assert controller.run_revision_count(RUN_ID) == 1


def test_sequential_same_turn_updates_preserve_parent_ancestry() -> None:
    controller = _controller()
    created = controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Draft", html="<p>one</p>"
    )
    scope = replace(
        _scope(),
        selected_canvas_id=created.revision.canvas_id,
        selected_revision_id=created.revision.revision_id,
    )
    first = controller.update_canvas(
        scope,
        tool_call_id="call-2",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        html="<p>two</p>",
    )
    second = controller.update_canvas(
        replace(scope, selected_revision_id=first.revision.revision_id),
        tool_call_id="call-3",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=first.revision.revision_id,
        html="<p>three</p>",
    )

    assert first.revision.parent_revision_id == created.revision.revision_id
    assert second.revision.parent_revision_id == first.revision.revision_id
    assert second.revision.sequence == 3


def test_parallel_same_parent_retry_fails_boundedly_without_mutation() -> None:
    controller = _controller()
    created = controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Draft", html="<p>one</p>"
    )
    scope = replace(
        _scope(),
        selected_canvas_id=created.revision.canvas_id,
        selected_revision_id=created.revision.revision_id,
    )
    controller.update_canvas(
        scope,
        tool_call_id="call-2",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        html="<p>two</p>",
    )

    conflict = controller.update_canvas(
        scope,
        tool_call_id="call-3",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        html="<p>parallel</p>",
    )

    assert isinstance(conflict, CanvasConflictResult)
    assert conflict.code == "ambiguous_ancestry"
    assert controller.run_revision_count(RUN_ID) == 2


def test_resume_reuses_settlement_without_duplicating_revision() -> None:
    controller = _controller()
    result = controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Draft", html="<p>x</p>"
    )
    first = controller.finish_run(RUN_ID, "done")

    resumed = controller.resume_run(_scope(), assistant_message_id=ASSISTANT_ID)
    replay = controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Draft", html="<p>x</p>"
    )

    assert resumed is first
    assert replay.revision.revision_id == result.revision.revision_id
    assert controller.run_revision_count(RUN_ID) == 1


def test_metadata_serialization_never_contains_source() -> None:
    controller = _controller()
    controller.create_canvas(
        _scope(),
        tool_call_id="call-1",
        title="Draft",
        html=f"<script>{SOURCE_SENTINEL}</script>",
    )
    settlement = controller.finish_run(RUN_ID, "done")

    assert settlement is not None
    payload = json.loads(settlement.metadata_json)
    assert SOURCE_SENTINEL not in json.dumps(payload)
    assert set(payload["canvas_cards"][0]) == {
        "canvas_id",
        "revision_id",
        "title",
        "sequence",
        "digest",
        "status",
        "origin",
        "reopenable",
        "error_code",
    }


def test_bridge_terminal_binding_refuses_a_different_agent_run_id() -> None:
    controller = _controller()
    controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Draft", html="<p>x</p>"
    )

    settlement = controller.finish_assistant_run(
        ASSISTANT_ID, actual_run_id="different-run", terminal_status="done"
    )

    assert settlement is not None
    assert settlement.state is CanvasRunState.DISCARDED
    assert settlement.cards[0].error_code == "run_identity_changed"


def test_app_shutdown_discards_ready_stage_without_touching_other_run() -> None:
    controller = _controller()
    controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Draft", html="<p>x</p>"
    )
    controller.finish_run(RUN_ID, "done")
    other_scope = _scope(run_id="run-2")
    controller.register_run(
        other_scope, assistant_message_id="assistant-2", temporary=False
    )

    controller.discard_session(SESSION_ID)

    assert (
        controller.settlement_for_assistant(ASSISTANT_ID).state
        is CanvasRunState.DISCARDED
    )
    assert (
        controller.settlement_for_assistant("assistant-2").state
        is CanvasRunState.DISCARDED
    )


def test_transcript_restores_metadata_only_canvas_card() -> None:
    controller = _controller()
    controller.create_canvas(
        _scope(),
        tool_call_id="call-1",
        title="Safe card",
        html=f"<p>{SOURCE_SENTINEL}</p>",
    )
    settlement = controller.finish_run(RUN_ID, "done")
    metadata = MessageMetadata.from_json(settlement.metadata_json)
    message = ConsoleChatMessage(
        id=ASSISTANT_ID,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        metadata=metadata,
    )

    cards = canvas_card_presentations(message)
    transcript = ConsoleTranscript()
    transcript.set_messages((message,))

    assert len(cards) == 1
    assert cards[0].label == "Safe card · revision 1 · updated"
    assert cards[0].reopenable is True
    assert SOURCE_SENTINEL not in repr(cards)
    assert "Canvas · Safe card · revision 1 · updated" in transcript.to_plain_text()
    rows = transcript._transcript_rows()
    assert any(
        row.kind == "canvas-card"
        or any(nested.kind == "canvas-card" for nested in row.nested_rows)
        for row in rows
    )


def test_temporary_committed_run_remains_available_for_atomic_promotion() -> None:
    controller = ConsoleCanvasController()
    controller.activate_session(SESSION_ID)
    controller.register_run(_scope(), assistant_message_id=ASSISTANT_ID, temporary=True)
    controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Temporary", html="<p>temporary</p>"
    )
    controller.finish_run(RUN_ID, "done")

    assert controller.confirm_settlement(RUN_ID) is True
    contribution = controller.promotion_contribution(SESSION_ID)

    assert contribution is not None
    assert contribution.revision_count == 1
    assert (
        controller.settlement_for_assistant(ASSISTANT_ID).cards[0].status == "temporary"
    )
    assert controller.abort_contribution(SESSION_ID, contribution) is True
    retry = controller.promotion_contribution(SESSION_ID)
    assert retry is not None
    assert retry.revision_count == 1


def test_next_temporary_run_reads_and_extends_committed_session_history() -> None:
    controller = ConsoleCanvasController()
    controller.activate_session(SESSION_ID)
    controller.register_run(_scope(), assistant_message_id=ASSISTANT_ID, temporary=True)
    created = controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Temporary", html="<p>one</p>"
    )
    controller.finish_run(RUN_ID, "done")
    assert controller.confirm_settlement(RUN_ID) is True

    second_scope = _scope(
        run_id="run-2",
        selected_canvas_id=created.revision.canvas_id,
        selected_revision_id=created.revision.revision_id,
    )
    controller.register_run(
        second_scope, assistant_message_id="assistant-2", temporary=True
    )
    restored = controller.read_canvas(second_scope, created.revision.canvas_id)
    updated = controller.update_canvas(
        second_scope,
        tool_call_id="call-2",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        html="<p>two</p>",
    )

    assert restored.source == "<p>one</p>"
    assert updated.revision.parent_revision_id == created.revision.revision_id
    assert updated.revision.sequence == 2


def test_temporary_promotion_refuses_an_unsettled_run() -> None:
    controller = ConsoleCanvasController()
    controller.activate_session(SESSION_ID)
    controller.register_run(_scope(), assistant_message_id=ASSISTANT_ID, temporary=True)
    controller.create_canvas(
        _scope(), tool_call_id="call-1", title="Temporary", html="<p>draft</p>"
    )

    with pytest.raises(RuntimeError, match="canvas_turns_not_settled"):
        controller.promotion_contribution(SESSION_ID)
