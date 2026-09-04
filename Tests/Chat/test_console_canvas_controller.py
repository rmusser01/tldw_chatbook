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
        active_message_ids=("user-1", ASSISTANT_ID, "user-2"),
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


def test_conflict_replay_is_stable_after_later_updates() -> None:
    controller = _controller()
    created = controller.create_canvas(
        _scope(), tool_call_id="create", title="Draft", html="<p>one</p>"
    )
    selected = replace(
        _scope(),
        selected_canvas_id=created.revision.canvas_id,
        selected_revision_id=created.revision.revision_id,
    )
    first = controller.update_canvas(
        selected,
        tool_call_id="update",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        html="<p>two</p>",
    )
    conflict = controller.update_canvas(
        selected,
        tool_call_id="conflict",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        html="<p>parallel</p>",
    )
    controller.update_canvas(
        replace(selected, selected_revision_id=first.revision.revision_id),
        tool_call_id="later",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=first.revision.revision_id,
        html="<p>three</p>",
    )

    assert (
        controller.update_canvas(
            selected,
            tool_call_id="conflict",
            canvas_id=created.revision.canvas_id,
            expected_parent_revision_id=created.revision.revision_id,
            html="<p>parallel</p>",
        )
        is conflict
    )


def test_promotion_lease_blocks_stage_after_snapshot() -> None:
    controller = ConsoleCanvasController()
    controller.activate_session(SESSION_ID)
    controller.register_run(_scope(), assistant_message_id=ASSISTANT_ID, temporary=True)
    controller.create_canvas(
        _scope(), tool_call_id="create", title="Temporary", html="<p>x</p>"
    )
    controller.finish_run(RUN_ID, "done")
    controller.confirm_settlement(RUN_ID)
    contribution = controller.promotion_contribution(SESSION_ID)

    with pytest.raises(RuntimeError, match="canvas_promotion_in_flight"):
        controller.register_run(
            _scope(run_id="run-after-snapshot"),
            assistant_message_id="assistant-after-snapshot",
            temporary=True,
        )
    with pytest.raises(RuntimeError, match="canvas_promotion_in_flight"):
        controller.activate_session(SESSION_ID)
    with pytest.raises(RuntimeError, match="canvas_scope_unavailable"):
        controller.create_canvas(
            _scope(), tool_call_id="create", title="Temporary", html="<p>x</p>"
        )
    controller.discard_session(SESSION_ID)
    assert contribution is not None
    assert controller.abort_contribution(SESSION_ID, contribution) is True


def test_late_bound_handle_is_inert_after_same_id_session_reactivation() -> None:
    controller = ConsoleCanvasController()
    controller.activate_session(SESSION_ID)
    old = controller.register_run(
        _scope(), assistant_message_id=ASSISTANT_ID, temporary=True
    )
    controller.activate_session(SESSION_ID)
    new = controller.register_run(
        _scope(), assistant_message_id=ASSISTANT_ID, temporary=True
    )

    assert (
        old.finish_assistant_run(
            ASSISTANT_ID, actual_run_id=RUN_ID, terminal_status="error"
        )
        is None
    )
    with pytest.raises(RuntimeError, match="canvas_scope_unavailable"):
        old.create_canvas(
            _scope(), tool_call_id="late", title="Late", html="<p>late</p>"
        )
    assert new.is_scope_current(_scope()) is True


def test_temporary_history_is_branch_scoped_and_honors_selected_revision() -> None:
    controller = ConsoleCanvasController()
    controller.activate_session(SESSION_ID)
    first_scope = _scope(active_message_ids=("root", "assistant-a"))
    controller.register_run(
        first_scope, assistant_message_id="assistant-a", temporary=True
    )
    created = controller.create_canvas(
        first_scope, tool_call_id="create", title="Branch", html="<p>one</p>"
    )
    controller.finish_run(RUN_ID, "done")
    controller.confirm_settlement(RUN_ID)

    sibling_scope = _scope(
        run_id="run-b",
        active_message_ids=("root", "assistant-b"),
        selected_canvas_id=created.revision.canvas_id,
        selected_revision_id=created.revision.revision_id,
    )
    controller.register_run(
        sibling_scope, assistant_message_id="assistant-b", temporary=True
    )
    assert controller.list_canvases(sibling_scope) == ()
    with pytest.raises(RuntimeError, match="canvas_base_unavailable"):
        controller.read_canvas(sibling_scope, created.revision.canvas_id)

    descendant_scope = replace(
        sibling_scope,
        run_id="run-c",
        active_message_ids=("root", "assistant-a", "user-c", "assistant-c"),
    )
    controller.register_run(
        descendant_scope, assistant_message_id="assistant-c", temporary=True
    )
    assert (
        controller.read_canvas(descendant_scope, created.revision.canvas_id).source
        == "<p>one</p>"
    )


def test_temporary_branch_switching_resolves_each_reachable_head() -> None:
    controller = ConsoleCanvasController()
    controller.activate_session(SESSION_ID)
    root_scope = _scope(active_message_ids=("root", "assistant-root"))
    controller.register_run(
        root_scope, assistant_message_id="assistant-root", temporary=True
    )
    root = controller.create_canvas(
        root_scope, tool_call_id="root", title="Branches", html="<p>root</p>"
    )
    controller.finish_run(RUN_ID, "done")
    controller.confirm_settlement(RUN_ID)

    def commit_branch(run_id: str, assistant_id: str, source: str):
        scope = _scope(
            run_id=run_id,
            active_message_ids=("root", "assistant-root", assistant_id),
            selected_canvas_id=root.revision.canvas_id,
            selected_revision_id=root.revision.revision_id,
        )
        controller.register_run(
            scope, assistant_message_id=assistant_id, temporary=True
        )
        result = controller.update_canvas(
            scope,
            tool_call_id=f"update-{run_id}",
            canvas_id=root.revision.canvas_id,
            expected_parent_revision_id=root.revision.revision_id,
            html=source,
        )
        controller.finish_run(run_id, "done")
        controller.confirm_settlement(run_id)
        return result

    left = commit_branch("run-left", "assistant-left", "<p>left</p>")
    right = commit_branch("run-right", "assistant-right", "<p>right</p>")
    left_view = _scope(
        run_id="view-left",
        active_message_ids=("root", "assistant-root", "assistant-left", "view-left"),
    )
    right_view = _scope(
        run_id="view-right",
        active_message_ids=("root", "assistant-root", "assistant-right", "view-right"),
    )
    controller.register_run(left_view, assistant_message_id="view-left", temporary=True)
    controller.register_run(
        right_view, assistant_message_id="view-right", temporary=True
    )

    assert (
        controller.read_canvas(left_view, root.revision.canvas_id).source
        == "<p>left</p>"
    )
    assert (
        controller.read_canvas(right_view, root.revision.canvas_id).source
        == "<p>right</p>"
    )
    assert left.revision.parent_revision_id == root.revision.revision_id
    assert right.revision.parent_revision_id == root.revision.revision_id


def test_production_temporary_scope_never_queries_nonexistent_durable_owner(
    tmp_path,
) -> None:
    from tldw_chatbook.Canvas.service import CanvasService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "temporary-production-scope.sqlite", "temp")
    try:
        controller = ConsoleCanvasController(durable_service=CanvasService(db))
        controller.activate_session(SESSION_ID)
        run = controller.register_run(
            _scope(), assistant_message_id=ASSISTANT_ID, temporary=True
        )

        assert run.list_canvases(_scope()) == ()
        with pytest.raises(RuntimeError, match="canvas_base_unavailable"):
            run.read_canvas(_scope(), "11111111-1111-4111-8111-111111111111")
        with pytest.raises(RuntimeError, match="canvas_base_unavailable"):
            run.update_canvas(
                _scope(),
                tool_call_id="missing-update",
                canvas_id="11111111-1111-4111-8111-111111111111",
                expected_parent_revision_id=(
                    "22222222-2222-4222-8222-222222222222"
                ),
                html="<p>must not stage</p>",
            )
    finally:
        db.close_connection()


def test_temporary_sequence_ignores_other_committed_canvas_rows() -> None:
    controller = ConsoleCanvasController()
    controller.activate_session(SESSION_ID)

    def commit_create(run_id: str, assistant_id: str, title: str):
        scope = _scope(
            run_id=run_id,
            active_message_ids=(assistant_id,),
        )
        controller.register_run(scope, assistant_message_id=assistant_id, temporary=True)
        created = controller.create_canvas(
            scope,
            tool_call_id=f"create-{run_id}",
            title=title,
            html=f"<p>{title}</p>",
        )
        controller.finish_run(run_id, "done")
        controller.confirm_settlement(run_id)
        return created

    first = commit_create("run-a", "assistant-a", "A")
    commit_create("run-b", "assistant-b", "B")
    update_scope = _scope(
        run_id="run-a-update",
        active_message_ids=("assistant-a", "user-next", "assistant-a-update"),
        selected_canvas_id=first.revision.canvas_id,
        selected_revision_id=first.revision.revision_id,
    )
    controller.register_run(
        update_scope, assistant_message_id="assistant-a-update", temporary=True
    )

    updated = controller.update_canvas(
        update_scope,
        tool_call_id="update-a",
        canvas_id=first.revision.canvas_id,
        expected_parent_revision_id=first.revision.revision_id,
        html="<p>A2</p>",
    )

    assert updated.revision.sequence == 2


def test_two_open_runs_cannot_mutate_the_same_durable_parent() -> None:
    controller = _controller()
    created = controller.create_canvas(
        _scope(), tool_call_id="create", title="Draft", html="<p>one</p>"
    )
    first_scope = replace(
        _scope(),
        selected_canvas_id=created.revision.canvas_id,
        selected_revision_id=created.revision.revision_id,
    )
    second_scope = replace(first_scope, run_id="run-second")
    controller.register_run(
        second_scope, assistant_message_id="assistant-second", temporary=False
    )
    first = controller.update_canvas(
        first_scope,
        tool_call_id="first",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        html="<p>first</p>",
    )
    conflict = controller.update_canvas(
        second_scope,
        tool_call_id="second",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        html="<p>second</p>",
    )

    assert first.revision.parent_revision_id == created.revision.revision_id
    assert conflict.code == "ambiguous_ancestry"
    assert controller.run_revision_count("run-second") == 0


def test_durable_historical_branch_uses_owner_global_next_sequence(tmp_path) -> None:
    from tldw_chatbook.Canvas.repository import CanvasRepository
    from tldw_chatbook.Canvas.service import CanvasService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "canvas-controller-branch.sqlite", "branch")
    try:
        conversation_id = db.add_conversation({"title": "Branch"})
        root_message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "root",
            }
        )
        leaf_message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "parent_message_id": root_message_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "leaf",
            }
        )
        assert conversation_id and root_message_id and leaf_message_id
        repository = CanvasRepository(db)
        root = repository.create_canvas(
            conversation_id,
            title="Durable branch",
            source="<p>root</p>",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id=root_message_id,
            origin_turn_id="root-run",
        )
        repository.append_revision(
            conversation_id,
            root.identity.canvas_id,
            parent_revision_id=root.revision.revision_id,
            title="Durable branch",
            source="<p>newer</p>",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id=leaf_message_id,
            origin_turn_id="newer-run",
        )
        scope = CanvasScope(
            session_id="durable-session",
            conversation_id=conversation_id,
            active_message_ids=(root_message_id, leaf_message_id),
            selected_canvas_id=root.identity.canvas_id,
            selected_revision_id=root.revision.revision_id,
            run_id="historical-run",
        )
        controller = ConsoleCanvasController(durable_service=CanvasService(db))
        controller.register_run(
            scope, assistant_message_id=leaf_message_id, temporary=False
        )

        branch = controller.update_canvas(
            scope,
            tool_call_id="historical-update",
            canvas_id=root.identity.canvas_id,
            expected_parent_revision_id=root.revision.revision_id,
            html="<p>branch</p>",
        )

        assert branch.revision.parent_revision_id == root.revision.revision_id
        assert branch.revision.sequence == 3
    finally:
        db.close_connection()


def test_promotion_remaps_card_and_revision_origins_together(tmp_path) -> None:
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "canvas-origin-remap.sqlite", "canvas-origin")
    controller = ConsoleCanvasController()
    store = ConsoleChatStore(
        persistence=ChatPersistenceService(db),
        canvas_promotion_participant=controller,
        canvas_turn_controller=controller,
    )
    try:
        session = store.create_session(ephemeral=True)
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        scope = _scope(
            session_id=session.id,
            conversation_id=session.id,
            active_message_ids=(assistant.id,),
        )
        controller.register_run(
            scope, assistant_message_id=assistant.id, temporary=True
        )
        created = controller.create_canvas(
            scope,
            tool_call_id="origin-remap",
            title="Remap",
            html="<p>private origin source</p>",
        )
        settlement = controller.finish_run(RUN_ID, "done")
        live_assistant = store._message_or_raise(assistant.id)
        live_assistant.metadata = MessageMetadata.from_json(settlement.metadata_json)
        live_assistant.status = "complete"
        live_assistant.assistant_generation_state = "complete"
        controller.confirm_exact_settlement(settlement)

        conversation_id = store.promote_ephemeral_session(session.id)
        promoted = store._message_or_raise(assistant.id)
        durable_id = promoted.persisted_message_id
        assert durable_id is not None
        assert promoted.metadata.canvas_cards[0].origin.message_id == durable_id
        store.set_message_feedback(assistant.id, "up")
        raw = db.get_message_by_id(durable_id)["metadata_json"]
        from tldw_chatbook.Chat.chat_conversation_service import (
            ChatConversationService,
        )
        from tldw_chatbook.Chat.console_conversation_hydration import (
            console_messages_from_conversation_tree,
        )

        tree = ChatConversationService(db).get_conversation_tree(
            conversation_id, root_limit=100, depth_cap=100
        )
        nodes = console_messages_from_conversation_tree(tree, db=db)
        restarted_store = ConsoleChatStore(
            persistence=ChatPersistenceService(db),
            canvas_promotion_participant=ConsoleCanvasController(),
        )
        restarted = restarted_store.restore_persisted_session(
            title="Restarted Canvas",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=nodes,
            active_leaf_persisted_id=durable_id,
        )
        restarted_message = restarted_store.messages_for_session(restarted.id)[0]
        hydrated = restarted_message.metadata
        revision_origin = (
            db.get_connection()
            .execute(
                "SELECT origin_message_id FROM canvas_revisions WHERE id = ?",
                (created.revision.revision_id,),
            )
            .fetchone()[0]
        )

        assert conversation_id is not None
        assert hydrated is not None
        assert hydrated.canvas_cards[0].origin.message_id == durable_id
        assert revision_origin == durable_id
        assert "private origin source" not in raw
    finally:
        db.close_connection()
