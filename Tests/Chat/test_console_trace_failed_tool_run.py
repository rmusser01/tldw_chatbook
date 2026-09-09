"""Captured history can advance past a closed, unanswered agent tool run."""

from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from Tests.Chat.test_console_project_instruction_traces import _project_console
from Tests.Chat.test_console_trace_discarded_tool_run import _reload_console
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentTraceRequestFactory
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer


def _privacy(app, enabled):
    app.controller.set_next_trace_privacy(
        app.session.id,
        capture_enabled=enabled,
        pii_redaction_enabled=False,
        expected_policy_revision=app.controller.capture_policy_snapshot(
            app.session.id
        ).policy_revision,
    )


@asynccontextmanager
async def _failed_console(
    tmp_path,
    monkeypatch,
    *,
    project_enabled,
    bypass,
    transform=False,
    failed_followups=0,
):
    async with _project_console(tmp_path, monkeypatch, tools=2) as app:
        app.store.set_session_project_instruction_state(
            app.session.id,
            replace(
                app.session.project_instruction_state,
                project_instructions_enabled=project_enabled,
            ),
        )
        if transform:
            assert app.store.persist_session_if_needed(app.session.id)
            app.controller._chat_dictionary_applier = lambda _conversation, text: (
                text.replace("calculator", "calculator tool")
            )
        build = ConsoleAgentTraceRequestFactory.build

        def fail_third(factory, *args, **kwargs):
            if len(app.http_payloads) == 2:
                with app.db.transaction() as cursor:
                    rows = cursor.execute("SELECT * FROM console_dispatch_checkpoints")
                    app.original_checkpoint = dict(
                        zip(
                            (item[0] for item in rows.description),
                            rows.fetchone(),
                            strict=True,
                        )
                    )
                raise ValueError("trace provenance category mismatch: tool_loop")
            return build(factory, *args, **kwargs)

        # Recreate the shipped request-builder failure after actual successful
        # requests; let the real agent/controller settle the failed owner.
        with monkeypatch.context() as fault:
            fault.setattr(ConsoleAgentTraceRequestFactory, "build", fail_third)
            first = await app.controller.submit_draft(
                "Use calculator twice", session_id=app.session.id
            )
        assert first.accepted
        if transform:
            assert any(
                row.get("content") == "Use calculator tool twice"
                for row in app.http_payloads[0]["messages"]
            )
        app.prior_user = app.store.get_message(
            first.user_message_id
        ).persisted_message_id
        app.failed_assistant = app.store.get_message(
            first.assistant_message_id
        ).persisted_message_id
        with app.db.transaction() as cursor:
            assert tuple(
                cursor.execute(
                    "SELECT assistant_generation_state, content FROM messages WHERE id = ?",
                    (app.failed_assistant,),
                ).fetchone()
            ) == ("failed", "")
            assert [
                row[0]
                for row in cursor.execute(
                    "SELECT state FROM console_trace_calls WHERE turn_id = ?",
                    (app.prior_user,),
                )
            ] == ["complete", "complete"]
            assert (
                cursor.execute(
                    "SELECT count(*) FROM console_dispatch_checkpoints WHERE user_message_id = ?",
                    (app.prior_user,),
                ).fetchone()[0]
                == 0
            )
        _reload_console(app)
        assert not (
            await app.controller.discard_dispatch_recovery(app.session.id)
        ).accepted
        app.untraced_users = []
        for index in range(failed_followups):

            def reject(service, *args, **kwargs):
                raise ValueError("unsupported_surface_change")

            with monkeypatch.context() as fault:
                fault.setattr(
                    type(app.factory.service), "prepare_current_surface_delta", reject
                )
                failed = await app.controller.submit_draft(
                    f"Earlier failed follow-up {index}", session_id=app.session.id
                )
            assert failed.accepted and len(app.http_payloads) == 2
            app.untraced_users.append(
                app.store.get_message(failed.user_message_id).persisted_message_id
            )
            _reload_console(app)
            assert (
                await app.controller.discard_dispatch_recovery(app.session.id)
            ).accepted
            _reload_console(app)
        for index in range(bypass):
            _privacy(app, False)
            result = await app.controller.submit_draft(
                f"Successful uncaptured follow-up {index}", session_id=app.session.id
            )
            assert result.accepted and len(app.http_payloads) == 3 + index
            app.untraced_users.append(
                app.store.get_message(result.user_message_id).persisted_message_id
            )
            _reload_console(app)
        _privacy(app, True)
        yield app


def _prior_evidence(app):
    reader = ConsoleTraceNativeReader(app.db)
    with app.db.transaction() as cursor:
        calls = tuple(
            tuple(row)
            for row in cursor.execute(
                "SELECT * FROM console_trace_calls WHERE turn_id = ? ORDER BY call_sequence",
                (app.prior_user,),
            )
        )
        requests = tuple(
            reader._reconstruct_request(
                cursor, reader.repository.get_call(cursor, row[0])
            )
            for row in calls
        )
    return calls, requests


@pytest.mark.parametrize("project_enabled", [False, True])
@pytest.mark.parametrize("bypass", [0, 1])
async def test_captured_sends_after_failed_empty_tool_run(
    tmp_path, monkeypatch, project_enabled, bypass, transform=False, failed_followups=0
):
    async with _failed_console(
        tmp_path,
        monkeypatch,
        project_enabled=project_enabled,
        bypass=bypass,
        transform=transform,
        failed_followups=failed_followups,
    ) as app:
        original = _prior_evidence(app)
        for question in ("Hello after failed run", "And after recovery"):
            before = len(app.http_payloads)
            result = await app.controller.submit_draft(
                question, session_id=app.session.id
            )
            assert result.accepted
            assert len(app.http_payloads) == before + 1, app.reservation_errors
            current = app.store.get_message(result.user_message_id).persisted_message_id
            captures = ConsoleTraceNativeReader(app.db).read_calls(current)
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
            assert _prior_evidence(app) == original
            with app.db.transaction() as cursor:
                for user_id in app.untraced_users:
                    assert (
                        cursor.execute(
                            "SELECT count(*) FROM console_trace_calls WHERE turn_id = ?",
                            (user_id,),
                        ).fetchone()[0]
                        == 0
                    )
            _reload_console(app)
        assert (
            app.reservation_errors == ["unsupported_surface_change"] * failed_followups
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "active",
        "unknown_delivery",
        "response_started",
        "no_response",
        "partial",
        "image",
        "thinking",
        "continuation",
        "attachment",
        "metadata",
        "deleted",
        "wrong_parent",
        "sibling",
        "deleted_sibling",
        "checkpoint",
        "policy",
        "owner",
        "gap",
        "duplicate",
        "over_limit",
        "descriptor_limit",
        "value",
        "followup_active",
        "followup_checkpoint",
        "followup_call",
        "followup_revision",
        "followup_retired",
    ],
)
async def test_closed_history_is_revalidated_before_dispatch(
    tmp_path, monkeypatch, tamper
):
    from tldw_chatbook.Chat.console_trace_models import TraceCallState, new_opaque_id

    async with _failed_console(
        tmp_path, monkeypatch, project_enabled=True, bypass=1
    ) as app:
        original = _prior_evidence(app)
        validator = type(app.factory.service)._validate_completed_tool_turn
        applied = []
        mutation_errors = []

        def tampered(service, cursor, **kwargs):
            witness = kwargs["witness"]
            if (
                witness.closed_assistant_message_id is None
                or kwargs.get("reserved_call") is None
            ):
                return validator(service, cursor, **kwargs)
            assistant_id = witness.closed_assistant_message_id
            user_revision, followup_assistant, response_revision = (
                witness.closed_followups[0]
            )

            def mutate(message_id, sql, parameters):
                from tldw_chatbook.Chat.console_semantic_revision import (
                    SemanticRevisionCoordinator,
                )

                try:
                    SemanticRevisionCoordinator(app.db).mutate_message(
                        cursor,
                        message_id=message_id,
                        creation_reason="edit",
                        mutate=lambda inner: inner.execute(sql, parameters),
                    )
                except Exception as exc:
                    mutation_errors.append(str(exc))
                    raise

            if tamper in {"unknown_delivery", "response_started", "no_response"}:
                get_call = service.repository.get_call

                def changed_call(inner_cursor, call_id):
                    call = get_call(inner_cursor, call_id)
                    if call_id != witness.terminal_call_id:
                        return call
                    if tamper == "no_response":
                        return replace(call, response_started_at=None)
                    return replace(
                        call,
                        state=(
                            TraceCallState.DISPATCH_UNKNOWN
                            if tamper == "unknown_delivery"
                            else TraceCallState.RESPONSE_STARTED
                        ),
                    )

                with monkeypatch.context() as fault:
                    fault.setattr(service.repository, "get_call", changed_call)
                    applied.append(tamper)
                    return validator(service, cursor, **kwargs)
            if tamper == "active":
                mutate(
                    assistant_id,
                    "UPDATE messages SET assistant_generation_state = 'dispatch_started' WHERE id = ?",
                    (assistant_id,),
                )
            elif tamper == "partial":
                mutate(
                    assistant_id,
                    "UPDATE messages SET content = 'Partial answer' WHERE id = ?",
                    (assistant_id,),
                )
            elif tamper == "image":
                mutate(
                    assistant_id,
                    "UPDATE messages SET image_data = ?, image_mime_type = 'image/png' WHERE id = ?",
                    (b"image", assistant_id),
                )
            elif tamper == "thinking":
                mutate(
                    assistant_id,
                    "UPDATE messages SET thinking_blocks_json = '{}' WHERE id = ?",
                    (assistant_id,),
                )
            elif tamper == "continuation":
                mutate(
                    assistant_id,
                    "UPDATE messages SET provider_continuation_json = '{}' WHERE id = ?",
                    (assistant_id,),
                )
            elif tamper == "attachment":
                mutate(
                    assistant_id,
                    "INSERT INTO message_attachments (message_id, position, data, mime_type, display_name) VALUES (?, 1, ?, 'image/png', 'image')",
                    (assistant_id, b"image"),
                )
            elif tamper == "metadata":
                cursor.execute(
                    "UPDATE messages SET metadata_json = ? WHERE id = ?",
                    ('{"canvas_cards":[{}]}', assistant_id),
                )
            elif tamper == "deleted":
                cursor.execute(
                    "UPDATE messages SET deleted = 1 WHERE id = ?", (assistant_id,)
                )
            elif tamper == "wrong_parent":
                mutate(
                    assistant_id,
                    "UPDATE messages SET parent_message_id = ? WHERE id = ?",
                    (kwargs["current_turn_id"], assistant_id),
                )
            elif tamper in {"sibling", "deleted_sibling"}:
                sibling = app.db.add_message(
                    {
                        "conversation_id": app.session.persisted_conversation_id,
                        "sender": "assistant",
                        "role": "assistant",
                        "content": "Other attempt",
                        "parent_message_id": app.prior_user,
                    }
                )
                if tamper == "deleted_sibling":
                    cursor.execute(
                        "UPDATE messages SET deleted = 1 WHERE id = ?", (sibling,)
                    )
            elif tamper in {"checkpoint", "followup_checkpoint"}:
                checkpoint = dict(app.original_checkpoint)
                if tamper == "followup_checkpoint":
                    checkpoint["assistant_message_id"] = followup_assistant
                    checkpoint["user_message_id"] = app.untraced_users[0]
                columns = ",".join(checkpoint)
                slots = ",".join("?" for _ in checkpoint)
                cursor.execute(
                    f"INSERT INTO console_dispatch_checkpoints ({columns}) VALUES ({slots})",
                    tuple(checkpoint.values()),
                )
            elif tamper == "policy":
                kwargs["current_policy_id"] = new_opaque_id()
            elif tamper == "owner":
                kwargs["witness"] = replace(
                    witness, closed_assistant_message_id=new_opaque_id()
                )
            elif tamper == "gap":
                kwargs["witness"] = replace(witness, closed_followups=())
            elif tamper in {"duplicate", "over_limit", "descriptor_limit"}:
                applied.append(tamper)
                kwargs["witness"] = replace(
                    witness,
                    closed_followups=witness.closed_followups
                    * (
                        2
                        if tamper == "duplicate"
                        else 128
                        if tamper == "descriptor_limit"
                        else 257
                    ),
                )
                return validator(service, cursor, **kwargs)
            elif tamper == "value":
                kwargs["values"] = (
                    {**kwargs["values"][0], "content": "Changed saved text"},
                    *kwargs["values"][1:],
                )
            elif tamper == "followup_active":
                mutate(
                    followup_assistant,
                    "UPDATE messages SET assistant_generation_state = 'dispatch_started' WHERE id = ?",
                    (followup_assistant,),
                )
            elif tamper == "followup_call":
                service.repository.reserve_call(
                    cursor,
                    owner_id=kwargs["owner_id"],
                    segment_id=kwargs["segment_id"],
                    turn_id=app.untraced_users[0],
                    run_id=new_opaque_id(),
                    call_sequence=0,
                    idempotency_key=new_opaque_id(),
                    policy_id=kwargs["current_policy_id"],
                )
            elif tamper == "followup_revision":
                kwargs["witness"] = replace(
                    witness,
                    closed_followups=(
                        (user_revision, followup_assistant, user_revision),
                    ),
                )
            elif tamper == "followup_retired":
                get_revision = service.repository.get_semantic_revision

                def retired(inner_cursor, revision_id):
                    revision = get_revision(inner_cursor, revision_id)
                    return (
                        replace(revision, live_message_id=None)
                        if revision_id == response_revision
                        else revision
                    )

                with monkeypatch.context() as fault:
                    fault.setattr(service.repository, "get_semantic_revision", retired)
                    applied.append(tamper)
                    return validator(service, cursor, **kwargs)
            applied.append(tamper)
            return validator(service, cursor, **kwargs)

        monkeypatch.setattr(
            type(app.factory.service), "_validate_completed_tool_turn", tampered
        )
        before = len(app.http_payloads)
        result = await app.controller.submit_draft(
            "Next captured question", session_id=app.session.id
        )
        assert result.accepted
        assert applied == [tamper], mutation_errors
        assert len(app.http_payloads) == before
        assert app.controller.run_state.status.value == "blocked"
        assert _prior_evidence(app) == original


@pytest.mark.parametrize("project_enabled", [False, True])
@pytest.mark.parametrize("transform", [False, True])
async def test_failed_history_with_discarded_and_uncaptured_followups(
    tmp_path, monkeypatch, project_enabled, transform
):
    await test_captured_sends_after_failed_empty_tool_run(
        tmp_path,
        monkeypatch,
        project_enabled,
        bypass=2,
        transform=transform,
        failed_followups=2,
    )


@pytest.mark.parametrize("project_enabled", [False, True])
@pytest.mark.parametrize("cold", [False, True])
async def test_guard_stopped_search_allows_next_capture(
    tmp_path, monkeypatch, project_enabled, cold
):
    import json

    import httpx

    from tldw_chatbook.Agents.local_tool_provider import (
        LocalToolProvider,
        _default_specs,
    )
    from tldw_chatbook.MCP.permission_store import EffectiveToolState
    from tldw_chatbook.Tools import web_tool_impls

    original_transport = httpx.MockTransport
    searches = []

    def transport(handler):
        def respond(request):
            response = handler(request)
            payload = response.json()
            message = payload["choices"][0]["message"]
            if '"name": "calculator"' in message["content"]:
                query = f"query {len(searches)}"
                message["content"] = (
                    "```tool_call\n"
                    + json.dumps({"name": "web_search", "arguments": {"query": query}})
                    + "\n```"
                )
            return httpx.Response(200, json=payload)

        return original_transport(respond)

    def unavailable(**kwargs):
        searches.append(kwargs["search_query"])
        return {"processing_error": "DuckDuckGo returned an anti-bot challenge"}

    monkeypatch.setattr(httpx, "MockTransport", transport)
    monkeypatch.setattr(
        "tldw_chatbook.Web_Scraping.WebSearch_APIs.perform_websearch", unavailable
    )
    web_tool_impls._reset_state_for_tests()
    try:
        async with _project_console(tmp_path, monkeypatch, tools=3) as app:
            app.store.set_session_project_instruction_state(
                app.session.id,
                replace(
                    app.session.project_instruction_state,
                    project_instructions_enabled=project_enabled,
                ),
            )
            provider = LocalToolProvider(
                workspace_root=tmp_path,
                specs=[
                    spec
                    for spec in _default_specs(tmp_path, workspace_executor=None)
                    if spec.name == "web_search"
                ],
                resolve_state=lambda _hub: EffectiveToolState(
                    state="allow", origin="tool_override"
                ),
            )
            monkeypatch.setattr(
                app.controller,
                "_compose_local_provider",
                lambda *_args, **_kwargs: (provider, lambda _calls: {}),
            )
            first = await app.controller.submit_draft(
                "Search for this three times", session_id=app.session.id
            )
            assert first.accepted
            assert len(searches) == len(app.http_payloads) == 3, (
                searches,
                app.reservation_errors,
            )
            runs = app.runs.list_runs(app.session.persisted_conversation_id)
            assert runs[0]["status"] == "stuck", runs
            prior_user = app.store.get_message(
                first.user_message_id
            ).persisted_message_id
            assistant = app.store.get_message(
                first.assistant_message_id
            ).persisted_message_id
            reader = ConsoleTraceNativeReader(app.db)
            original = reader.read_calls(prior_user)
            assert len(original) == 3
            with app.db.transaction() as cursor:
                closure = tuple(
                    cursor.execute(
                        "SELECT assistant_generation_state, content FROM messages WHERE id = ?",
                        (assistant,),
                    ).fetchone()
                )
                assert closure == ("failed", ""), closure
            if cold:
                _reload_console(app)
            second = await app.controller.submit_draft(
                "Now answer directly", session_id=app.session.id
            )
            assert second.accepted
            assert len(app.http_payloads) == 4, app.reservation_errors
            assert app.controller.run_state.status.value == "completed"
            current_user = app.store.get_message(
                second.user_message_id
            ).persisted_message_id
            assert len(reader.read_calls(current_user)) == 1
            assert reader.read_calls(prior_user) == original
    finally:
        web_tool_impls._reset_state_for_tests()
