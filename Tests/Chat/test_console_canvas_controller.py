from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tldw_chatbook.Canvas.limits import CanvasLimitError, CanvasRepositoryLimits
from tldw_chatbook.Canvas.models import (
    CanvasConflictResult,
    CanvasQuotaUsage,
    CanvasRenderPlan,
    CanvasScope,
    CanvasSourceIdentity,
    RenderNode,
)
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


def _synthetic_plan(source: str) -> CanvasRenderPlan:
    """Build a source-exact inert plan so admission tests isolate ownership."""

    return CanvasRenderPlan(
        runtime_profile="canvas-v1",
        source_identity=CanvasSourceIdentity.from_source(source),
        root=RenderNode("synthetic-root", "html"),
    )


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


def test_production_owner_enforces_exact_default_canvas_and_revision_counts() -> None:
    controller = _controller()
    scope = _scope()
    created = []
    for index in range(10):
        created.append(
            controller.create_canvas(
                scope,
                tool_call_id=f"create-{index}",
                title=f"Canvas {index}",
                html="<p>x</p>",
            )
        )

    with pytest.raises(CanvasLimitError, match="canvas_count_limit"):
        controller.create_canvas(
            scope,
            tool_call_id="create-overflow",
            title="Canvas overflow",
            html="<p>x</p>",
        )
    assert controller.run_revision_count(RUN_ID) == 10

    revision_controller = _controller()
    current = revision_controller.create_canvas(
        scope,
        tool_call_id="root",
        title="Revision boundary",
        html="<p>0</p>",
    )
    for sequence in range(2, 101):
        current = revision_controller.update_canvas(
            scope,
            tool_call_id=f"update-{sequence}",
            canvas_id=current.revision.canvas_id,
            expected_parent_revision_id=current.revision.revision_id,
            html=f"<p>{sequence}</p>",
        )

    with pytest.raises(CanvasLimitError, match="revision_count_limit"):
        revision_controller.update_canvas(
            scope,
            tool_call_id="update-overflow",
            canvas_id=current.revision.canvas_id,
            expected_parent_revision_id=current.revision.revision_id,
            html="<p>101</p>",
        )
    assert revision_controller.run_revision_count(RUN_ID) == 100


def test_production_owner_counts_concurrent_bytes_and_abort_releases_them() -> None:
    limits = CanvasRepositoryLimits(
        max_canvases_per_conversation=10,
        max_revisions_per_canvas=100,
        max_source_bytes_per_conversation=8,
        max_source_bytes_per_revision=8,
    )
    controller = ConsoleCanvasController(repository_limits=limits)
    scopes = [
        _scope(run_id=f"run-{index}", active_message_ids=(f"assistant-{index}",))
        for index in range(3)
    ]
    runs = [
        controller.register_run(
            scope, assistant_message_id=f"assistant-{index}", temporary=False
        )
        for index, scope in enumerate(scopes)
    ]
    runs[0].create_canvas(scopes[0], tool_call_id="first", title="First", html="1234")
    runs[1].create_canvas(scopes[1], tool_call_id="second", title="Second", html="5678")

    with pytest.raises(CanvasLimitError, match="conversation_source_bytes_limit"):
        runs[2].create_canvas(scopes[2], tool_call_id="third", title="Third", html="x")
    assert controller.run_revision_count("run-2") == 0

    assert controller.abort_settlement("run-1", "cancelled") is True
    runs[2].create_canvas(scopes[2], tool_call_id="third", title="Third", html="x")
    assert controller.run_revision_count("run-2") == 1


def test_temporary_owner_enforces_exact_default_session_bytes_across_scopes(
    monkeypatch,
) -> None:
    """The default 8 MiB ceiling spans one temporary session incarnation."""

    source = "x" * (512 * 1024)
    plan = _synthetic_plan(source)
    controller = ConsoleCanvasController()
    controller.activate_session(SESSION_ID)

    first_scope = _scope(
        run_id="import-first",
        conversation_id="temporary-conversation-a",
        active_message_ids=("message-a-0",),
    )
    first = controller.interactive_create_canvas(
        first_scope,
        origin_message_id="message-a-0",
        title="Imported",
        html=source,
        temporary=True,
        _prepared_plan=plan,
    )
    renamed = controller.interactive_rename_canvas(
        replace(
            first_scope,
            run_id="rename-first",
            active_message_ids=("message-a-0", "message-a-1"),
            selected_canvas_id=first.revision.canvas_id,
            selected_revision_id=first.revision.revision_id,
        ),
        origin_message_id="message-a-1",
        canvas_id=first.revision.canvas_id,
        expected_parent_revision_id=first.revision.revision_id,
        title="Renamed",
        temporary=True,
    )
    assert renamed.revision.sequence == 2

    for index in range(8):
        message_id = f"message-a-{index + 2}"
        controller.interactive_create_canvas(
            _scope(
                run_id=f"import-a-{index}",
                conversation_id="temporary-conversation-a",
                active_message_ids=(message_id,),
            ),
            origin_message_id=message_id,
            title=f"Imported A {index}",
            html=source,
            temporary=True,
            _prepared_plan=plan,
        )

    for index in range(5):
        message_id = f"message-b-{index}"
        controller.interactive_create_canvas(
            _scope(
                run_id=f"import-b-{index}",
                conversation_id="temporary-conversation-b",
                active_message_ids=(message_id,),
            ),
            origin_message_id=message_id,
            title=f"Imported B {index}",
            html=source,
            temporary=True,
            _prepared_plan=plan,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_canvas_controller.compile_canvas_document",
        _synthetic_plan,
    )
    exact_scope = _scope(
        run_id="reserved-exact-boundary",
        conversation_id="temporary-conversation-b",
        active_message_ids=("reserved-message",),
    )
    exact = controller.register_run(
        exact_scope, assistant_message_id="reserved-message", temporary=True
    )
    exact.create_canvas(
        exact_scope, tool_call_id="exact", title="Exact boundary", html=source
    )

    overflow_scope = replace(
        exact_scope,
        run_id="reserved-overflow",
        active_message_ids=("overflow-message",),
    )
    overflow = controller.register_run(
        overflow_scope, assistant_message_id="overflow-message", temporary=True
    )
    with pytest.raises(CanvasLimitError, match="session_source_bytes_limit"):
        overflow.create_canvas(
            overflow_scope, tool_call_id="overflow", title="Overflow", html="x"
        )
    assert controller.run_revision_count("reserved-overflow") == 0

    assert controller.abort_settlement("reserved-exact-boundary", "cancelled") is True
    admitted = overflow.create_canvas(
        overflow_scope, tool_call_id="overflow", title="Overflow", html="x"
    )
    assert admitted.revision.source_bytes == 1


def test_durable_committed_stage_is_not_double_counted_after_persistence() -> None:
    class DurableUsage:
        usage = CanvasQuotaUsage((), (), 0)

        def quota_usage(self, _scope):
            return self.usage

    service = DurableUsage()
    limits = CanvasRepositoryLimits(max_source_bytes_per_conversation=4)
    controller = ConsoleCanvasController(
        durable_service=service, repository_limits=limits
    )
    first_scope = _scope(run_id="durable-first", active_message_ids=("first",))
    first_run = controller.register_run(
        first_scope, assistant_message_id="first", temporary=False
    )
    first = first_run.create_canvas(
        first_scope, tool_call_id="first", title="First", html="12"
    )
    settlement = first_run.finish_assistant_run(
        "first", actual_run_id="durable-first", terminal_status="done"
    )
    assert settlement is not None
    service.usage = CanvasQuotaUsage(
        (first.revision.canvas_id,), ((first.revision.canvas_id, 1),), 2
    )
    assert controller.confirm_exact_settlement(settlement) is True

    second_scope = _scope(run_id="durable-second", active_message_ids=("second",))
    second_run = controller.register_run(
        second_scope, assistant_message_id="second", temporary=False
    )
    second_run.create_canvas(
        second_scope, tool_call_id="second", title="Second", html="34"
    )
    assert controller.run_revision_count("durable-second") == 1


def test_temporary_import_and_rename_share_admission_without_partial_stage() -> None:
    limits = CanvasRepositoryLimits(
        max_canvases_per_conversation=1,
        max_revisions_per_canvas=2,
        max_source_bytes_per_conversation=100,
        max_source_bytes_per_revision=100,
    )
    controller = ConsoleCanvasController(repository_limits=limits)
    controller.activate_session(SESSION_ID)
    create_scope = _scope(run_id="import-create", active_message_ids=("user-1",))
    created = controller.interactive_create_canvas(
        create_scope,
        origin_message_id="user-1",
        title="Imported",
        html="<p>one</p>",
        temporary=True,
    )
    rename_scope = _scope(
        run_id="import-rename",
        active_message_ids=("user-1", "user-2"),
        selected_canvas_id=created.revision.canvas_id,
        selected_revision_id=created.revision.revision_id,
    )
    renamed = controller.interactive_rename_canvas(
        rename_scope,
        origin_message_id="user-2",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        title="Renamed",
        temporary=True,
    )
    overflow_scope = replace(
        rename_scope,
        run_id="import-overflow",
        active_message_ids=("user-1", "user-2", "user-3"),
        selected_revision_id=renamed.revision.revision_id,
    )

    with pytest.raises(CanvasLimitError, match="revision_count_limit"):
        controller.interactive_update_canvas(
            overflow_scope,
            origin_message_id="user-3",
            canvas_id=created.revision.canvas_id,
            expected_parent_revision_id=renamed.revision.revision_id,
            html="<p>three</p>",
            temporary=True,
        )

    assert "import-overflow" not in controller._runs


def test_durable_import_counts_concurrent_stage_and_existing_history(tmp_path) -> None:
    from tldw_chatbook.Canvas.service import CanvasService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "canvas-admission.sqlite", "admission")
    limits = CanvasRepositoryLimits(max_canvases_per_conversation=1)
    try:
        conversation_id = db.add_conversation({"title": "Admission"})
        first_message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "first",
            }
        )
        second_message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "parent_message_id": first_message_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "second",
            }
        )
        assert conversation_id and first_message_id and second_message_id
        controller = ConsoleCanvasController(
            durable_service=CanvasService(db), repository_limits=limits
        )
        tool_scope = CanvasScope(
            session_id="durable-session",
            conversation_id=conversation_id,
            active_message_ids=(first_message_id, second_message_id),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id="tool-stage",
        )
        run = controller.register_run(
            tool_scope, assistant_message_id=first_message_id, temporary=False
        )
        run.create_canvas(
            tool_scope, tool_call_id="reserved", title="Reserved", html="<p>x</p>"
        )
        import_scope = replace(tool_scope, run_id="durable-import")

        with pytest.raises(CanvasLimitError, match="canvas_count_limit"):
            controller.interactive_create_canvas(
                import_scope,
                origin_message_id=second_message_id,
                title="Imported",
                html="<p>imported</p>",
                temporary=False,
            )

        assert controller.abort_settlement("tool-stage", "cancelled") is True
        imported = controller.interactive_create_canvas(
            import_scope,
            origin_message_id=second_message_id,
            title="Imported",
            html="<p>imported</p>",
            temporary=False,
        )
        assert imported.revision.title == "Imported"

        next_scope = replace(tool_scope, run_id="tool-after-import")
        next_run = controller.register_run(
            next_scope, assistant_message_id=second_message_id, temporary=False
        )
        with pytest.raises(CanvasLimitError, match="canvas_count_limit"):
            next_run.create_canvas(
                next_scope,
                tool_call_id="existing-overflow",
                title="Overflow",
                html="<p>overflow</p>",
            )
        assert controller.run_revision_count("tool-after-import") == 0
    finally:
        db.close_connection()


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


def test_tool_mutation_publication_waits_for_postcommit_and_is_retry_safe() -> None:
    controller = _controller()
    publications = []
    attempts = 0

    def flaky_listener(publication) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient UI publication failure")
        publications.append(publication)

    controller.add_settlement_listener(flaky_listener)
    result = controller.create_canvas(
        _scope(),
        tool_call_id="postcommit-create",
        title="Postcommit",
        html=f"<p>{SOURCE_SENTINEL}</p>",
    )
    settlement = controller.finish_run(RUN_ID, "done")

    assert publications == []
    assert settlement is not None
    assert controller.confirm_exact_settlement(settlement) is True
    assert publications == []
    assert controller.confirm_exact_settlement(settlement) is True
    assert len(publications) == 1
    assert publications[0].scope == _scope()
    assert publications[0].revisions == (result.revision,)
    assert SOURCE_SENTINEL not in repr(publications[0])
    assert controller.confirm_exact_settlement(settlement) is True
    assert len(publications) == 1


@pytest.mark.parametrize("terminal_status", ["failed", "cancelled", "stopped"])
def test_failed_tool_mutation_never_publishes_phantom_selection(
    terminal_status: str,
) -> None:
    controller = _controller()
    publications = []
    controller.add_settlement_listener(publications.append)
    controller.create_canvas(
        _scope(),
        tool_call_id="failed-create",
        title="Not committed",
        html=f"<p>{SOURCE_SENTINEL}</p>",
    )

    settlement = controller.finish_run(RUN_ID, terminal_status)

    assert settlement is not None
    assert settlement.state is CanvasRunState.DISCARDED
    assert publications == []


def test_durable_transaction_rollback_cannot_publish_and_retry_publishes_once(
    tmp_path,
) -> None:
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    class FailingContribution:
        def write(self, *, writer, conversation_id, message_ids) -> None:
            raise RuntimeError("forced rollback")

    db = CharactersRAGDB(tmp_path / "canvas-publication-rollback.sqlite", "canvas")
    try:
        service = ChatPersistenceService(db)
        conversation_id = service.create_conversation(
            assistant_kind="generic", assistant_id="console"
        )
        persisted_message_id = service.create_message(
            conversation_id=conversation_id,
            sender="assistant",
            content="pending",
        )
        initial = db.get_message_by_id(persisted_message_id)
        controller = ConsoleCanvasController()
        scope = _scope(
            conversation_id=conversation_id,
            active_message_ids=(ASSISTANT_ID,),
        )
        controller.register_run(
            scope,
            assistant_message_id=ASSISTANT_ID,
            temporary=False,
        )
        result = controller.create_canvas(
            scope,
            tool_call_id="durable-rollback",
            title="Postcommit",
            html=f"<p>{SOURCE_SENTINEL}</p>",
        )
        settlement = controller.finish_run(RUN_ID, "done")
        publications = []
        controller.add_settlement_listener(publications.append)
        write_args = {
            "native_message_id": ASSISTANT_ID,
            "message_id": persisted_message_id,
            "content": "complete",
            "thinking_blocks_json": None,
            "provider_continuation_json": None,
            "assistant_generation_state": "complete",
            "usage_json": None,
            "metadata_json": settlement.metadata_json,
            "update_metadata": True,
            "expected_version": initial["version"],
        }

        with pytest.raises(RuntimeError, match="forced rollback"):
            service.replace_assistant_generation_projection_with_contributions(
                **write_args,
                contributions=(settlement.contribution, FailingContribution()),
                on_durable_commit=lambda: controller.confirm_exact_settlement(
                    settlement
                ),
            )

        assert publications == []
        assert controller.settlement_for_assistant(ASSISTANT_ID).state is (
            CanvasRunState.READY
        )
        assert db.get_message_by_id(persisted_message_id)["content"] == "pending"
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM canvas_revisions")
            .fetchone()[0]
            == 0
        )

        service.replace_assistant_generation_projection_with_contributions(
            **write_args,
            contributions=(settlement.contribution,),
            on_durable_commit=lambda: controller.confirm_exact_settlement(settlement),
        )

        assert [item.revisions for item in publications] == [(result.revision,)]
        assert controller.settlement_for_assistant(ASSISTANT_ID).state is (
            CanvasRunState.COMMITTED
        )
        assert SOURCE_SENTINEL not in repr(publications)
    finally:
        db.close_connection()


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
