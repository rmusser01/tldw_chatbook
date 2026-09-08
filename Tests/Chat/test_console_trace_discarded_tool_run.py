"""Capture recovery after an explicitly discarded, interrupted tool run."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_project_instruction_traces import _project_console
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatMessage,
    ConsoleChatStore,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_trace_models import new_opaque_id
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer


def _reload_console(app):
    conversation = app.session.persisted_conversation_id
    nodes = [
        ConsoleChatMessage(
            id=str(row["id"]),
            role=ConsoleMessageRole(str(row["role"])),
            content=str(row.get("content") or ""),
            persisted_message_id=str(row["id"]),
            parent_message_id=row.get("parent_message_id"),
        )
        for row in app.db.get_messages_for_conversation(conversation, limit=100)
    ]
    store = ConsoleChatStore(persistence=app.store.persistence)
    session = store.restore_persisted_session(
        title=app.session.title,
        workspace_id="project-trace",
        persisted_conversation_id=conversation,
        all_nodes=nodes,
        active_leaf_persisted_id=app.db.get_conversation_active_leaf(conversation),
        settings=app.session.settings,
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=app.runs, store=store, provider_gateway=app.gateway
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=app.gateway,
        agent_runtime_enabled=True,
        agent_bridge=bridge,
        confirm_project_instruction_dispatch=lambda _: "proceed",
    )
    controller.app = SimpleNamespace(workspace_registry_service=app.registry)
    app.store, app.session, app.controller = store, session, controller
    app.restart_factory()


@pytest.mark.parametrize("cold_after_discard", [False, True])
@pytest.mark.parametrize("project_enabled", [False, True])
async def test_following_captured_sends_after_discarded_tool_run(
    tmp_path,
    monkeypatch,
    cold_after_discard,
    project_enabled,
    tamper=None,
    current_transform=False,
    failed_followups=0,
):
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_args: 100_000)
    monkeypatch.setattr(
        agent_service, "catalog_schema_tokens", lambda *_args, **_kwargs: 10_001
    )
    async with _project_console(
        tmp_path, monkeypatch, tools=2, discover_tools=True
    ) as app:
        app.store.set_session_project_instruction_state(
            app.session.id,
            replace(
                app.session.project_instruction_state,
                project_instructions_enabled=project_enabled,
            ),
        )
        if current_transform:
            assert app.store.persist_session_if_needed(app.session.id)
            app.controller._chat_dictionary_applier = lambda _conversation, text: (
                text.replace("calculator", "calculator tool")
            )
        original = type(app.factory.service).prepare_current_surface_delta

        def reject_third(instance, *args, **kwargs):
            if len(app.http_payloads) == 2:
                raise ValueError("unsupported_surface_change")
            return original(instance, *args, **kwargs)

        # Recreate the previously shipped third-call failure, retaining all real
        # writes from the successful first two provider requests.
        with monkeypatch.context() as fault:
            fault.setattr(
                type(app.factory.service), "prepare_current_surface_delta", reject_third
            )
            first = await app.controller.submit_draft(
                "Find and load the calculator", session_id=app.session.id
            )
        assert first.accepted
        assert app.reservation_errors == ["unsupported_surface_change"]
        assert len(app.http_payloads) == 2
        if current_transform:
            assert any(
                row.get("content") == "Find and load the calculator tool"
                for row in app.http_payloads[0]["messages"]
            )
        prior_user = app.store.get_message(first.user_message_id).persisted_message_id
        reader = ConsoleTraceNativeReader(app.db)

        def prior_requests():
            with app.db.transaction() as cursor:
                ids = cursor.execute(
                    "SELECT call_id FROM console_trace_calls WHERE turn_id = ? ORDER BY call_sequence",
                    (prior_user,),
                ).fetchall()
                return tuple(
                    reader._reconstruct_request(
                        cursor, reader.repository.get_call(cursor, row[0])
                    )
                    for row in ids
                )

        previous = prior_requests()
        assert len(previous) == 2
        with app.db.transaction() as cursor:
            if current_transform:
                origin_id = cursor.execute(
                    "SELECT call_id FROM console_trace_calls WHERE turn_id = ? AND call_sequence = 0",
                    (prior_user,),
                ).fetchone()[0]
                origin = reader.repository.get_call(cursor, origin_id)
                assert (
                    reader.service._call_message_tail(cursor, origin).component_kind
                    == "active_request"
                )
                source_id = cursor.execute(
                    "SELECT semantic_revision_id FROM console_trace_events WHERE call_id = ? AND event_type = 'call_boundary'",
                    (origin_id,),
                ).fetchone()[0]
                source = reader.repository.get_semantic_revision(cursor, source_id)
                assert source.source_message_id == prior_user
                assert (
                    source.source_conversation_id
                    == app.session.persisted_conversation_id
                )
                assert source.normalized_role == "user"
            prior_records = tuple(
                tuple(row)
                for row in cursor.execute(
                    "SELECT * FROM console_trace_calls WHERE turn_id = ? ORDER BY call_sequence",
                    (prior_user,),
                )
            )
            assert [
                row[0]
                for row in cursor.execute(
                    "SELECT state FROM console_trace_calls WHERE turn_id = ?",
                    (prior_user,),
                )
            ] == ["response_started", "response_started"]
        with app.db.transaction() as cursor:
            result_rows = cursor.execute(
                "SELECT * FROM console_dispatch_checkpoints WHERE user_message_id = ?",
                (prior_user,),
            )
            checkpoint = dict(
                zip(
                    (item[0] for item in result_rows.description),
                    result_rows.fetchone(),
                    strict=True,
                )
            )
        _reload_console(app)
        result = await app.controller.discard_dispatch_recovery(app.session.id)
        assert result.accepted
        if cold_after_discard:
            _reload_console(app)
        for followup_index in range(failed_followups):

            def reject_successor(instance, *args, **kwargs):
                raise ValueError("surface_replacement_checkpoint_unavailable")

            with monkeypatch.context() as fault:
                fault.setattr(
                    type(app.factory.service),
                    "prepare_current_surface_delta",
                    reject_successor,
                )
                failed = await app.controller.submit_draft(
                    f"Earlier failed follow-up {followup_index}",
                    session_id=app.session.id,
                )
            assert failed.accepted
            assert len(app.http_payloads) == 2
            _reload_console(app)
            assert (
                await app.controller.discard_dispatch_recovery(app.session.id)
            ).accepted
        checks = []
        if tamper:
            validator = type(app.factory.service)._validate_completed_tool_turn

            def tampered_validator(service, cursor, **kwargs):
                witness = kwargs["witness"]
                if (
                    witness.discarded_assistant_message_id is None
                    or kwargs.get("reserved_call") is None
                ):
                    return validator(service, cursor, **kwargs)
                checks.append(tamper)
                assistant_id = witness.discarded_assistant_message_id
                if tamper.startswith("followup_"):
                    revision_id, followup_assistant = witness.discarded_followups[0]
                    if tamper == "followup_not_discarded":
                        cursor.execute(
                            "UPDATE messages SET assistant_generation_state = 'accepted' WHERE id = ?",
                            (followup_assistant,),
                        )
                    elif tamper == "followup_owner":
                        kwargs["witness"] = replace(
                            witness,
                            discarded_followups=((revision_id, new_opaque_id()),),
                        )
                    elif tamper == "followup_source":
                        kwargs["witness"] = replace(
                            witness,
                            discarded_followups=(
                                (witness.user_revision_id, followup_assistant),
                            ),
                        )
                    elif tamper == "followup_duplicate":
                        kwargs["witness"] = replace(
                            witness, discarded_followups=witness.discarded_followups * 2
                        )
                    elif tamper == "followup_gap":
                        kwargs["witness"] = replace(witness, discarded_followups=())
                    elif tamper == "followup_value":
                        kwargs["values"] = (
                            {**kwargs["values"][0], "content": "unowned history"},
                            *kwargs["values"][1:],
                        )
                    elif tamper == "followup_over_limit":
                        kwargs["witness"] = replace(
                            witness,
                            discarded_followups=witness.discarded_followups * 257,
                        )
                elif tamper == "not_discarded":
                    cursor.execute(
                        "UPDATE messages SET assistant_generation_state = 'accepted' WHERE id = ?",
                        (assistant_id,),
                    )
                elif tamper == "deleted":
                    cursor.execute(
                        "UPDATE messages SET deleted = 1 WHERE id = ?", (assistant_id,)
                    )
                elif tamper == "wrong_parent":
                    cursor.execute(
                        "UPDATE messages SET parent_message_id = ? WHERE id = ?",
                        (prior_user, kwargs["current_turn_id"]),
                    )
                elif tamper == "wrong_prior_turn":
                    cursor.execute(
                        "UPDATE messages SET parent_message_id = ? WHERE id = ?",
                        (kwargs["current_turn_id"], assistant_id),
                    )
                elif tamper in {"sibling_owner", "deleted_sibling_owner"}:
                    sibling_id = app.db.add_message(
                        {
                            "conversation_id": app.session.persisted_conversation_id,
                            "sender": "assistant",
                            "role": "assistant",
                            "content": "Another response attempt",
                            "parent_message_id": prior_user,
                        }
                    )
                    if tamper == "deleted_sibling_owner":
                        cursor.execute(
                            "UPDATE messages SET deleted = 1 WHERE id = ?",
                            (sibling_id,),
                        )
                elif tamper == "active_checkpoint":
                    columns = ",".join(checkpoint)
                    placeholders = ",".join("?" for _ in checkpoint)
                    cursor.execute(
                        f"INSERT INTO console_dispatch_checkpoints ({columns}) VALUES ({placeholders})",
                        tuple(checkpoint.values()),
                    )
                elif tamper == "forged_owner":
                    kwargs["witness"] = replace(
                        witness, discarded_assistant_message_id=new_opaque_id()
                    )
                elif tamper == "wrong_policy":
                    kwargs["current_policy_id"] = new_opaque_id()
                elif tamper == "unknown_call":
                    # Exercise the state proof without violating immutable SQL
                    # transition constraints in this fault-injection test.
                    from tldw_chatbook.Chat.console_trace_models import TraceCallState

                    get_call = service.repository.get_call

                    def unknown_call(inner_cursor, call_id):
                        call = get_call(inner_cursor, call_id)
                        return (
                            replace(call, state=TraceCallState.DISPATCH_UNKNOWN)
                            if call_id == witness.terminal_call_id
                            else call
                        )

                    with monkeypatch.context() as fault:
                        fault.setattr(service.repository, "get_call", unknown_call)
                        return validator(service, cursor, **kwargs)
                return validator(service, cursor, **kwargs)

            monkeypatch.setattr(
                type(app.factory.service),
                "_validate_completed_tool_turn",
                tampered_validator,
            )
        for question in ("Next question", "And one more"):
            before = len(app.http_payloads)
            result = await app.controller.submit_draft(
                question, session_id=app.session.id
            )
            assert result.accepted
            if tamper:
                assert checks == [tamper]
                assert app.controller.run_state.status.value == "blocked"
                assert len(app.http_payloads) == before
                assert prior_requests() == previous
                return
            assert app.controller.run_state.status.value == "completed", (
                app.reservation_errors
            )
            assert len(app.http_payloads) == before + 1
            current_user = app.store.get_message(
                result.user_message_id
            ).persisted_message_id
            captures = reader.read_calls(current_user)
            assert len(captures) == 1
            rows = [
                {
                    key: value
                    for key, value in row.items()
                    if key != "_chatbook_ephemeral_origin"
                }
                for row in captures[0].capture.request["messages_payload"]
            ]
            assert (
                rows
                == CredentialSanitizer()
                .sanitize(app.http_payloads[-1]["messages"])
                .value
            )
            assert prior_requests() == previous
            with app.db.transaction() as cursor:
                assert (
                    tuple(
                        tuple(row)
                        for row in cursor.execute(
                            "SELECT * FROM console_trace_calls WHERE turn_id = ? ORDER BY call_sequence",
                            (prior_user,),
                        )
                    )
                    == prior_records
                )
        assert app.reservation_errors == (
            ["unsupported_surface_change"]
            + ["surface_replacement_checkpoint_unavailable"] * failed_followups
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "not_discarded",
        "deleted",
        "wrong_parent",
        "wrong_prior_turn",
        "sibling_owner",
        "deleted_sibling_owner",
        "active_checkpoint",
        "forged_owner",
        "wrong_policy",
        "unknown_call",
    ],
)
async def test_discard_evidence_is_revalidated_before_dispatch(
    tmp_path, monkeypatch, tamper
):
    await test_following_captured_sends_after_discarded_tool_run(
        tmp_path,
        monkeypatch,
        cold_after_discard=True,
        project_enabled=True,
        tamper=tamper,
    )


@pytest.mark.parametrize("project_enabled", [False, True])
async def test_discard_restores_the_exact_transformed_user_source(
    tmp_path, monkeypatch, project_enabled
):
    await test_following_captured_sends_after_discarded_tool_run(
        tmp_path,
        monkeypatch,
        cold_after_discard=True,
        project_enabled=project_enabled,
        current_transform=True,
    )


@pytest.mark.parametrize("project_enabled", [False, True])
@pytest.mark.parametrize("current_transform", [False, True])
@pytest.mark.parametrize("failed_followups", [1, 2])
async def test_discard_after_already_failed_followups(
    tmp_path,
    monkeypatch,
    project_enabled,
    current_transform,
    failed_followups,
):
    await test_following_captured_sends_after_discarded_tool_run(
        tmp_path,
        monkeypatch,
        cold_after_discard=True,
        project_enabled=project_enabled,
        current_transform=current_transform,
        failed_followups=failed_followups,
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "followup_not_discarded",
        "followup_owner",
        "followup_source",
        "followup_duplicate",
        "followup_gap",
        "followup_value",
        "followup_over_limit",
    ],
)
async def test_each_discarded_followup_is_revalidated(tmp_path, monkeypatch, tamper):
    await test_following_captured_sends_after_discarded_tool_run(
        tmp_path,
        monkeypatch,
        cold_after_discard=True,
        project_enabled=True,
        failed_followups=1,
        tamper=tamper,
    )
