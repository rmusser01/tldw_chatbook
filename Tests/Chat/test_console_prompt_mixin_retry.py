"""Provider-boundary original Retry versus current-choice send behavior."""

from dataclasses import replace

import pytest

from Tests.Chat.test_console_prompt_mixin_admission import A, Scope
from Tests.Chat.test_console_trace_current_turn_transforms import (
    trace_harness as trace_harness,  # noqa: PLC0414 - shared native fixture
)
from tldw_chatbook.Chat.console_prompt_mixin_service import ConsolePromptMixinService
from tldw_chatbook.Chat.console_prompt_mixins import MixinPreferences, MixinSelection

pytestmark = pytest.mark.bootstrap_profile


def select(harness, monkeypatch, *, enabled=True, session_id="session-1"):
    scope = Scope()
    scope.release.set()
    scope.records[A.record_uuid]["prompt_definition"]["body"] = "body A"
    service = ConsolePromptMixinService(
        store=harness.store,
        prompt_scope_service=scope,
        persistence=harness.store.persistence,
    )
    monkeypatch.setattr(harness.controller, "prompt_mixin_service", lambda: service)
    service.replace_preferences(
        session_id,
        expected_revision=0,
        value=MixinPreferences(entries=(MixinSelection(A, enabled, "A", "before"),)),
    )
    return service, scope


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [False, True])
async def test_retry_original_regenerate_current_and_clean_history(
    trace_harness, monkeypatch, capture
):
    h = trace_harness
    capture_mode(monkeypatch, capture)
    service, scope = select(h, monkeypatch)
    h.adapter_error = True
    result = await h.controller.submit_draft("input", session_id="session-1")
    assert h.entries[-1][-1]["content"] == "body A\n\ninput"
    user = h.store.get_message(result.user_message_id)
    assert user.content == "input"
    reply = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role.value == "assistant"
    )
    scope.records[A.record_uuid]["prompt_definition"]["body"] = "body B"
    scope.records[A.record_uuid]["version"] = 2
    prefs = service.preferences("session-1")
    service.replace_preferences(
        "session-1",
        expected_revision=prefs.revision,
        value=replace(prefs, entries=(replace(prefs.entries[0], enabled=False),)),
    )
    h.adapter_error = False
    retry = await h.controller.retry_message(reply.id)
    assert len(h.entries) == 2, retry
    assert h.entries[-1][-1]["content"] == "body A\n\ninput"
    prefs = service.preferences("session-1")
    service.replace_preferences(
        "session-1",
        expected_revision=prefs.revision,
        value=replace(prefs, entries=(replace(prefs.entries[0], enabled=True),)),
    )
    regenerated = await h.controller.regenerate_message(reply.id)
    assert len(h.entries) == 3, regenerated
    assert h.entries[-1][-1]["content"] == "body B\n\ninput"
    subsequent = await h.controller.submit_draft("next", session_id="session-1")
    assert len(h.entries) == 4, (subsequent, h.failures)
    assert h.entries[-1][0] == {"role": "user", "content": "input"}
    assert h.entries[-1][-1]["content"] == "body B\n\nnext"


def capture_mode(monkeypatch, enabled):
    from tldw_chatbook.Chat import console_chat_controller as module
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
    from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparation

    def preparation(**kwargs):
        kwargs["capture_mode"] = (
            ConsoleTraceCaptureMode.CAPTURE_ON
            if enabled
            else ConsoleTraceCaptureMode.CAPTURE_OFF
        )
        return ConsoleTurnPreparation(**kwargs)

    monkeypatch.setattr(module, "ConsoleTurnPreparation", preparation)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("capture", [False, True])
async def test_original_retry_survives_reopen_deleted_source_and_latest_loss(
    trace_harness, monkeypatch, enabled, capture
):
    from Tests.Chat.test_console_turn_resend import _restore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    h = trace_harness
    capture_mode(monkeypatch, capture)
    service, scope = select(h, monkeypatch, enabled=enabled)
    h.adapter_error = True
    first = await h.controller.submit_draft("original", session_id="session-1")
    assert len(h.entries) == 1
    original_text = "body A\n\noriginal" if enabled else "original"
    assert h.entries[-1][-1]["content"] == original_text
    assistant = h.store.get_message(first.assistant_message_id)
    original_attempt = assistant.metadata.mixin_attempt_id
    prefs = service.preferences("session-1")
    service.replace_preferences(
        "session-1",
        expected_revision=prefs.revision,
        value=replace(prefs, entries=(replace(prefs.entries[0], enabled=True),)),
    )
    scope.records[A.record_uuid]["prompt_definition"]["body"] = "B"
    await h.controller.retry_message(assistant.id)
    assert len(h.entries) == 2 and h.entries[-1][-1]["content"] == original_text
    retried = h.store.get_message(assistant.id)
    assert retried.metadata.mixin_attempt_id != original_attempt
    conversation = h.store._sessions["session-1"].persisted_conversation_id
    path = h.db.db_path
    await h.store.persist_prompt_mixin_preferences("session-1")
    h.db.close()
    database = CharactersRAGDB(path, "mixin-reopen")
    try:
        restored = _restore(database, conversation)
        restored.controller.provider_gateway = h.gateway
        new_service = ConsolePromptMixinService(
            store=restored.store,
            prompt_scope_service=scope,
            persistence=restored.store.persistence,
        )
        monkeypatch.setattr(
            restored.controller, "prompt_mixin_service", lambda: new_service
        )
        scope.records.clear()
        reply = next(
            row
            for row in restored.store.messages_for_session(restored.session_id)
            if row.persisted_message_id == assistant.persisted_message_id
        )
        h.adapter_error = False
        retried_after_restart = await restored.controller.retry_message(reply.id)
        assert len(h.entries) == 3 and h.entries[-1][-1]["content"] == original_text, (
            retried_after_restart,
            reply.status,
        )
        latest = restored.store.get_message(reply.id).metadata.mixin_attempt_id
        assert latest not in {original_attempt, retried.metadata.mixin_attempt_id}
        with database.transaction() as cursor:
            records = cursor.execute(
                "SELECT attempt_id,retry_of_attempt_id FROM console_prompt_mixin_attempts WHERE conversation_id=? ORDER BY rowid",
                (conversation,),
            ).fetchall()
            assert [tuple(row) for row in records] == [
                (original_attempt, None),
                (retried.metadata.mixin_attempt_id, original_attempt),
                (latest, retried.metadata.mixin_attempt_id),
            ]
            cursor.execute(
                "DELETE FROM console_prompt_mixin_attempts WHERE conversation_id=? AND attempt_id=?",
                (conversation, latest),
            )
        # The previous valid record must never rescue a missing latest attempt.
        restored.store._message_or_raise(reply.id).status = "failed"
        refusal = await restored.controller.retry_message(reply.id)
        assert "unavailable" in refusal.visible_copy and len(h.entries) == 3
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_harness", [False, True], indirect=True)
async def test_current_choice_edit_continue_preview_and_clean_history(
    trace_harness, monkeypatch
):
    h = trace_harness
    service, scope = select(h, monkeypatch)
    capture_mode(monkeypatch, False)
    first = await h.controller.submit_draft("input", session_id="session-1")
    assert len(h.entries) == 1
    scope.records[A.record_uuid]["prompt_definition"]["body"] = "current {{literal}}"
    scope.records[A.record_uuid]["version"] = 2
    preferences_before_preview = service.preferences("session-1")

    def forbid_save(*args, **kwargs):
        pytest.fail("preview must not schedule preference persistence")

    with monkeypatch.context() as patch:
        patch.setattr(h.store, "schedule_prompt_mixin_preferences_save", forbid_save)
        preview = await h.controller.build_context_snapshot(
            "draft", session_id="session-1"
        )
    assert service.preferences("session-1") == preferences_before_preview
    assert (
        preview.next_send_payload["messages"][-1]["content"]
        == "current {{literal}}\n\ndraft"
    )
    assert len(h.entries) == 1
    edited = await h.controller.edit_and_resend_message(first.user_message_id, "edited")
    assert len(h.entries) == 2, edited
    assert h.entries[-1][-1]["content"] == "current {{literal}}\n\nedited"
    reply = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role.value == "assistant"
    )
    continued = await h.controller.continue_from_message(reply.id)
    assert len(h.entries) == 3, continued
    assert (
        h.entries[-1][-1]["content"]
        == "current {{literal}}\n\nContinue and extend the selected message."
    )
    assert h.entries[-1][0]["content"] == "edited"


@pytest.mark.asyncio
async def test_snapshot_write_failure_blocks_initial_adapter_and_preserves_draft(
    trace_harness, monkeypatch
):
    from tldw_chatbook.DB import console_prompt_mixins as records

    h = trace_harness
    select(h, monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic storage failure")

    monkeypatch.setattr(records, "write_attempt", fail)
    h.store.set_session_draft("session-1", "keep me")
    result = await h.controller.submit_draft("keep me", session_id="session-1")
    assert not result.provider_started and not h.entries
    assert h.store.session_draft("session-1") == "keep me"
    assert (
        h.db.get_connection()
        .execute("SELECT count(*) FROM console_prompt_mixin_attempts")
        .fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_harness", [False, True], indirect=True)
@pytest.mark.parametrize("character", [False, True])
async def test_plain_character_streaming_and_nonstreaming_keep_bodies_literal(
    trace_harness, monkeypatch, character
):
    h = trace_harness
    _service, scope = select(h, monkeypatch)
    scope.records[A.record_uuid]["prompt_definition"]["body"] = (
        "literal alias {{x}} /command"
    )
    if character:
        h.store._sessions["session-1"].assistant_kind = "character"
    result = await h.controller.submit_draft("alias input", session_id="session-1")
    assert len(h.entries) == 1, result
    assert (
        h.entries[0][-1]["content"] == "literal alias {{x}} /command\n\nexpanded input"
    )
    assert h.store.get_message(result.user_message_id).content == "alias input"
    assert all(
        not any(key.startswith("_tldw_mixin") for key in row) for row in h.entries[0]
    )


@pytest.mark.asyncio
async def test_hook_user_row_is_never_the_mixin_anchor(trace_harness, monkeypatch):
    h = trace_harness
    select(h, monkeypatch)
    capture_mode(monkeypatch, False)

    async def hook(*args):
        return [{"role": "user", "content": "unrelated hook"}]

    monkeypatch.setattr(h.controller, "_submission_hook_input", hook)
    result = await h.controller.submit_draft("input", session_id="session-1")
    assert len(h.entries) == 1, result
    assert h.entries[0][-2:] == [
        {"role": "user", "content": "body A\n\ninput"},
        {"role": "user", "content": "unrelated hook"},
    ]


@pytest.mark.asyncio
async def test_temporary_attempts_promote_with_retry_lineage(
    trace_harness, monkeypatch
):
    import asyncio

    from tldw_chatbook.Chat.console_prompt_mixins import MixinAttemptWrite

    h = trace_harness
    capture_mode(monkeypatch, False)
    # Reuse the real controller/gateway with a genuinely temporary session.
    temporary = h.store.create_session(session_id="temp", ephemeral=True)
    scope = Scope()
    scope.release.set()
    scope.records[A.record_uuid]["prompt_definition"]["body"] = "temporary"
    service = ConsolePromptMixinService(
        store=h.store, prompt_scope_service=scope, persistence=h.store.persistence
    )
    monkeypatch.setattr(h.controller, "prompt_mixin_service", lambda: service)
    service.replace_preferences(
        "temp",
        expected_revision=0,
        value=MixinPreferences(entries=(MixinSelection(A, True, "A", "before"),)),
    )
    h.adapter_error = True
    result = await h.controller.submit_draft("input", session_id="temp")
    assert len(h.entries) == 1, result
    first = h.store.get_message(result.assistant_message_id)
    await h.controller.retry_message(first.id)
    assert len(h.entries) == 2
    await h.controller.retry_message(first.id)
    assert len(h.entries) == 3
    batches = tuple(temporary.prompt_mixin_attempts.values())
    assert len(batches) == 3 and all(
        type(batch) is MixinAttemptWrite for batch in batches
    )
    assert batches[1].retry_of_attempt_id == batches[0].attempt_id
    scope.records.clear()
    from tldw_chatbook.DB import console_prompt_mixins

    original_write = console_prompt_mixins.write_attempt

    def fail_latest(cursor, conversation_id, *, batch, native_message_ids):
        original_write(
            cursor, conversation_id, batch=batch, native_message_ids=native_message_ids
        )
        if batch.attempt_id == batches[-1].attempt_id:
            raise RuntimeError("forced promotion rollback")

    with monkeypatch.context() as patch:
        patch.setattr(console_prompt_mixins, "write_attempt", fail_latest)
        with pytest.raises(RuntimeError, match="forced promotion rollback"):
            await asyncio.to_thread(h.store.promote_ephemeral_session, "temp")
    assert temporary.ephemeral and temporary.persisted_conversation_id is None
    assert tuple(temporary.prompt_mixin_attempts.values()) == batches
    assert (
        h.db.get_connection()
        .execute("SELECT count(*) FROM console_prompt_mixin_attempts")
        .fetchone()[0]
        == 0
    )
    conversation = await asyncio.to_thread(h.store.promote_ephemeral_session, "temp")
    assert conversation
    rows = (
        h.db.get_connection()
        .execute(
            "SELECT attempt_id,retry_of_attempt_id FROM console_prompt_mixin_attempts WHERE conversation_id=? ORDER BY rowid",
            (conversation,),
        )
        .fetchall()
    )
    assert [tuple(row) for row in rows] == [
        (batches[0].attempt_id, None),
        (batches[1].attempt_id, batches[0].attempt_id),
        (batches[2].attempt_id, batches[1].attempt_id),
    ]
    promoted = h.store.get_message(first.id)
    assert promoted.metadata.mixin_attempt_id == batches[-1].attempt_id
    assert (
        h.store.persistence.get_conversation_mixin_attempt(
            conversation,
            assistant_message_id=promoted.persisted_message_id,
            attempt_id=batches[-1].attempt_id,
        ).snapshot
        == batches[-1].snapshot
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "count",
        "foreign",
        "parent",
        "revision",
        "pending",
        "marker",
        "stale",
        "sibling",
        "dispatch",
        "wrong_origin",
        "unsealed",
        "cycle",
    ],
)
async def test_replacement_trace_proof_requires_exact_operational_owner(
    trace_harness, monkeypatch, damage
):
    from uuid import uuid4

    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

    h = trace_harness
    select(h, monkeypatch)
    first = await h.controller.submit_draft("input", session_id="session-1")
    await h.controller.regenerate_message(first.assistant_message_id)
    assistant = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role is ConsoleMessageRole.ASSISTANT
    )
    current = h.store.append_message(
        "session-1", role=ConsoleMessageRole.USER, content="next", persist=True
    )
    conversation = h.store._sessions["session-1"].persisted_conversation_id
    service = ConsoleTraceService()
    with h.db.transaction() as cursor:
        revision = cursor.execute(
            "SELECT revision_id FROM console_trace_semantic_revisions WHERE live_message_id=? ORDER BY revision_sequence DESC LIMIT 1",
            (assistant.persisted_message_id,),
        ).fetchone()[0]
        kwargs = {
            "conversation_id": conversation,
            "source_user_id": h.store.get_message(
                first.user_message_id
            ).persisted_message_id,
            "current_user_id": current.persisted_message_id,
            "assistant_revision_id": revision,
            "origin_call_id": cursor.execute(
                "SELECT trace_call_id FROM console_prompt_mixin_attempts WHERE conversation_id=? AND assistant_message_id=? ORDER BY rowid LIMIT 1",
                (
                    conversation,
                    h.store.get_message(
                        first.assistant_message_id
                    ).persisted_message_id,
                ),
            ).fetchone()[0],
        }
        assert (
            service.operational_replacement_attempt(cursor, **kwargs)
            == assistant.metadata.mixin_attempt_id
        )
        if damage == "wrong_origin":
            kwargs["origin_call_id"] = str(uuid4())
        elif damage in {"unsealed", "cycle"}:
            cursor.execute("DROP TRIGGER console_prompt_mixin_attempt_immutable")
            if damage == "unsealed":
                cursor.execute(
                    "UPDATE console_prompt_mixin_attempts SET result_revision_id=NULL WHERE attempt_id=?",
                    (assistant.metadata.mixin_attempt_id,),
                )
            else:
                cursor.execute(
                    "UPDATE console_prompt_mixin_attempts SET replaces_attempt_id=attempt_id WHERE attempt_id=?",
                    (assistant.metadata.mixin_attempt_id,),
                )
        elif damage == "missing":
            cursor.execute(
                "DELETE FROM console_prompt_mixin_attempts WHERE attempt_id=?",
                (assistant.metadata.mixin_attempt_id,),
            )
        elif damage == "count":
            cursor.execute(
                "DELETE FROM console_prompt_mixin_attempt_items WHERE attempt_id=?",
                (assistant.metadata.mixin_attempt_id,),
            )
        elif damage == "foreign":
            kwargs["conversation_id"] = str(uuid4())
        elif damage == "parent":
            kwargs["current_user_id"] = first.user_message_id
        elif damage == "sibling":
            kwargs["assistant_revision_id"] = cursor.execute(
                "SELECT revision_id FROM console_trace_semantic_revisions WHERE live_message_id=? ORDER BY revision_sequence DESC LIMIT 1",
                (h.store.get_message(first.assistant_message_id).persisted_message_id,),
            ).fetchone()[0]
        elif damage == "dispatch":
            cursor.execute(
                "INSERT INTO console_dispatch_checkpoints(assistant_message_id,user_message_id,conversation_id,preparation_id,attempt_id,state,user_message_version,assistant_message_version,origin,frozen_authority_json,resolved_destination_json,reconstructability_json) VALUES(?,?,?,?,?,'dispatch_started',1,1,'manual','{}','{}','{}')",
                (
                    assistant.persisted_message_id,
                    kwargs["source_user_id"],
                    conversation,
                    str(uuid4()),
                    str(uuid4()),
                ),
            )
        elif damage == "stale":
            from tldw_chatbook.Chat.console_prompt_mixins import MixinAttemptWrite
            from tldw_chatbook.DB.console_prompt_mixins import (
                read_attempt,
                write_attempt,
            )

            original_id = assistant.metadata.mixin_attempt_id
            snapshot = read_attempt(
                cursor, conversation, attempt_id=original_id
            ).snapshot
            successor = MixinAttemptWrite(
                "newer",
                kwargs["source_user_id"],
                assistant.persisted_message_id,
                snapshot,
                original_id,
            )
            write_attempt(
                cursor,
                conversation,
                batch=successor,
                native_message_ids={
                    successor.user_message_id: successor.user_message_id,
                    successor.assistant_message_id: successor.assistant_message_id,
                },
            )
            cursor.execute("DROP TRIGGER console_prompt_mixin_message_marker_preserve")
            cursor.execute(
                "UPDATE messages SET metadata_json=? WHERE id=?",
                (assistant.metadata.to_json(), assistant.persisted_message_id),
            )
        elif damage == "revision":
            kwargs["assistant_revision_id"] = str(uuid4())
        elif damage == "pending":
            h.db.update_message(
                assistant.persisted_message_id,
                {"assistant_generation_state": "accepted"},
                h.db.get_message_by_id(assistant.persisted_message_id)["version"],
            )
        else:
            cursor.execute("DROP TRIGGER console_prompt_mixin_message_marker_preserve")
            cursor.execute(
                "UPDATE messages SET metadata_json=? WHERE id=?",
                (
                    '{"mixin_attempt_id":"forged","mixin_snapshot_schema_version":1}',
                    assistant.persisted_message_id,
                ),
            )
        assert service.operational_replacement_attempt(cursor, **kwargs) is None


@pytest.mark.asyncio
async def test_regenerate_attempt_failure_rolls_back_empty_native_owner(
    trace_harness, monkeypatch
):
    from tldw_chatbook.DB import console_prompt_mixins

    h = trace_harness
    capture_mode(monkeypatch, False)
    select(h, monkeypatch)
    first = await h.controller.submit_draft("input", session_id="session-1")
    before = (
        h.db.get_connection()
        .execute("SELECT id FROM messages WHERE role='assistant'")
        .fetchall()
    )

    def fail(*args, **kwargs):
        raise RuntimeError("forced contribution failure")

    monkeypatch.setattr(console_prompt_mixins, "write_attempt", fail)
    await h.controller.regenerate_message(first.assistant_message_id)
    assert len(h.entries) == 1
    after = (
        h.db.get_connection()
        .execute("SELECT id FROM messages WHERE role='assistant'")
        .fetchall()
    )
    assert [tuple(row) for row in after] == [tuple(row) for row in before]


@pytest.mark.asyncio
async def test_capture_masks_credentials_but_operational_body_is_exact(
    trace_harness, monkeypatch
):
    h = trace_harness
    service, scope = select(h, monkeypatch)
    secret = "Bearer sk-proj-" + "x" * 48
    scope.records[A.record_uuid]["prompt_definition"]["body"] = secret
    first = await h.controller.submit_draft("input", session_id="session-1")
    assert h.entries[0][-1]["content"] == secret + "\n\ninput"
    assert (
        await service.retry_snapshot(
            "session-1", assistant_message_id=first.assistant_message_id
        )
    ).items[0].definition.body == secret
    artifact_values = (
        h.db.get_connection()
        .execute("SELECT sanitized_bytes FROM console_trace_artifacts")
        .fetchall()
    )
    assert secret not in repr([tuple(row) for row in artifact_values])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        '{"mixin_attempt_id":"unknown"}',
        '{"mixin_attempt_id":"unknown","mixin_snapshot_schema_version":true}',
        "[]",
    ],
)
async def test_raw_corrupt_marker_refuses_before_provider_even_if_typed_metadata_is_absent(
    trace_harness, monkeypatch, metadata
):
    h = trace_harness
    capture_mode(monkeypatch, False)
    select(h, monkeypatch)
    h.adapter_error = True
    first = await h.controller.submit_draft("input", session_id="session-1")
    assistant = h.store.get_message(first.assistant_message_id)
    with h.db.transaction() as cursor:
        cursor.execute("DROP TRIGGER console_prompt_mixin_message_marker_preserve")
        cursor.execute(
            "UPDATE messages SET metadata_json=? WHERE id=?",
            (metadata, assistant.persisted_message_id),
        )
    h.store._message_or_raise(assistant.id).metadata = None
    result = await h.controller.retry_message(assistant.id)
    assert "unavailable" in result.visible_copy and len(h.entries) == 1


@pytest.mark.asyncio
async def test_stopped_empty_resend_uses_original_attempt(trace_harness, monkeypatch):
    from tldw_chatbook.Chat.console_turn_resend import resend_turn

    h = trace_harness
    capture_mode(monkeypatch, False)
    _service, scope = select(h, monkeypatch)
    h.adapter_error = True
    first = await h.controller.submit_draft("input", session_id="session-1")
    assistant = h.store._message_or_raise(first.assistant_message_id)
    original = assistant.metadata.mixin_attempt_id
    assistant.status = "stopped"
    scope.records[A.record_uuid]["prompt_definition"]["body"] = "changed"
    scope.records.clear()
    h.adapter_error = False
    result = await resend_turn(h.controller, first.user_message_id)
    assert len(h.entries) == 2, result
    assert h.entries[-1][-1]["content"] == "body A\n\ninput"
    row = (
        h.db.get_connection()
        .execute(
            "SELECT retry_of_attempt_id FROM console_prompt_mixin_attempts ORDER BY rowid DESC LIMIT 1"
        )
        .fetchone()
    )
    assert row[0] == original


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_harness", [False, True], indirect=True)
@pytest.mark.parametrize("capture", [False, True])
async def test_real_agent_tool_loop_reuses_once_composed_request(
    trace_harness, monkeypatch, tmp_path, capture
):
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_prepared_request import thaw_json
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    h = trace_harness
    capture_mode(monkeypatch, capture)
    select(h, monkeypatch)
    runs = AgentRunsDB(tmp_path / "mixin-agent.sqlite", client_id="mixin-test")
    try:
        bridge = ConsoleAgentBridge(
            agent_runs_db=runs,
            store=h.store,
            provider_gateway=h.gateway,
            native_tools_enabled=lambda: False,
        )
        h.controller._agent_bridge = bridge
        h.controller._agent_runtime_enabled = True

        def adapter(**kwargs):
            h.entries.append(thaw_json(kwargs["messages_payload"]))
            text = (
                '```tool_call\n{"name":"calculator","arguments":{"expression":"6*7"}}\n```'
                if len(h.entries) == 1
                else "42"
            )
            if kwargs.get("streaming"):
                return iter([{"choices": [{"delta": {"content": text}}]}])
            return {"choices": [{"message": {"content": text}}]}

        monkeypatch.setattr(h.gateway, "_chat_api_call_fn", adapter)
        result = await h.controller.submit_draft("input", session_id="session-1")
        assert len(h.entries) == 2, (result, h.failures)
        for rows in h.entries:
            assert sum(str(row.get("content", "")).count("body A") for row in rows) == 1
            assert any(row.get("content") == "body A\n\ninput" for row in rows)
            assert all("_tldw_mixin_attempt" not in row for row in rows)
        assert h.store.get_message(result.user_message_id).content == "input"
        with h.db.transaction() as cursor:
            binding = cursor.execute(
                "SELECT trace_call_id FROM console_prompt_mixin_attempts"
            ).fetchone()[0]
            if capture:
                calls = cursor.execute(
                    "SELECT call_id FROM console_trace_calls ORDER BY call_sequence"
                ).fetchall()
                assert len(calls) == 2 and binding == calls[0][0]
            else:
                assert binding is None
    finally:
        runs.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("action", ["retry", "regenerate"])
async def test_continue_replay_composes_only_synthetic_current_user(
    trace_harness, monkeypatch, restart, action
):
    from Tests.Chat.test_console_turn_resend import _restore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    h = trace_harness
    capture_mode(monkeypatch, False)
    service, scope = select(h, monkeypatch)
    first = await h.controller.submit_draft("canonical user", session_id="session-1")
    h.adapter_error = True
    await h.controller.continue_from_message(first.assistant_message_id)
    failed = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role.value == "assistant"
    )
    assert h.entries[-1][0]["content"] == "canonical user"
    assert (
        h.entries[-1][-1]["content"]
        == "body A\n\nContinue and extend the selected message."
    )
    scope.records[A.record_uuid]["prompt_definition"]["body"] = "body B"
    controller = h.controller
    database = None
    if restart:
        conversation = h.store._sessions["session-1"].persisted_conversation_id
        await h.store.persist_prompt_mixin_preferences("session-1")
        path = h.db.db_path
        h.db.close()
        database = CharactersRAGDB(path, "continue-reopen")
        restored = _restore(database, conversation)
        controller = restored.controller
        controller.provider_gateway = h.gateway
        service = ConsolePromptMixinService(
            store=restored.store,
            prompt_scope_service=scope,
            persistence=restored.store.persistence,
        )
        monkeypatch.setattr(controller, "prompt_mixin_service", lambda: service)
        failed = next(
            row
            for row in restored.store.all_messages_for_session(restored.session_id)
            if row.persisted_message_id == failed.persisted_message_id
        )
        restored.store.set_active_leaf(restored.session_id, failed.id)
    try:
        h.adapter_error = False
        result = await (
            controller.retry_message(failed.id)
            if action == "retry"
            else controller.regenerate_message(failed.id)
        )
        assert len(h.entries) == 3, result
        assert h.entries[-1][0]["content"] == "canonical user"
        body = "body A" if action == "retry" else "body B"
        assert (
            h.entries[-1][-1]["content"]
            == body + "\n\nContinue and extend the selected message."
        )
    finally:
        if database is not None:
            database.close()


@pytest.mark.asyncio
async def test_cancel_during_attempt_write_drains_before_publication_and_no_dispatch(
    trace_harness, monkeypatch
):
    import asyncio
    from threading import Event

    h = trace_harness
    capture_mode(monkeypatch, False)
    select(h, monkeypatch)
    h.adapter_error = True
    first = await h.controller.submit_draft("input", session_id="session-1")
    original = h.store.get_message(first.assistant_message_id).metadata.mixin_attempt_id
    entered, release = Event(), Event()
    save = h.store.persistence.save_conversation_mixin_attempt

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return save(*args, **kwargs)

    monkeypatch.setattr(h.store.persistence, "save_conversation_mixin_attempt", delayed)
    task = asyncio.create_task(h.controller.retry_message(first.assistant_message_id))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert (
            h.store.get_message(first.assistant_message_id).metadata.mixin_attempt_id
            == original
        )
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        current = h.store.get_message(first.assistant_message_id)
        assert current.metadata.mixin_attempt_id != original
        record = h.store.persistence.get_conversation_mixin_attempt(
            h.store._sessions["session-1"].persisted_conversation_id,
            assistant_message_id=current.persisted_message_id,
            attempt_id=current.metadata.mixin_attempt_id,
        )
        assert record.status == "recorded"
        assert len(h.entries) == 1
    finally:
        release.set()
        if not task.done():
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage", ["missing_link", "unequal_link", "edited_first", "edited_replacement"]
)
async def test_marker_cannot_authorize_unverified_or_edited_result(
    trace_harness, monkeypatch, damage
):
    from uuid import uuid4

    h = trace_harness
    select(h, monkeypatch)
    first = await h.controller.submit_draft("input", session_id="session-1")
    assistant_id = first.assistant_message_id
    if damage == "edited_replacement":
        await h.controller.regenerate_message(assistant_id)
        assistant_id = next(
            row.id
            for row in reversed(h.store.messages_for_session("session-1"))
            if row.role.value == "assistant"
        )
        seal = (
            h.db.get_connection()
            .execute(
                "SELECT result_revision_id FROM console_prompt_mixin_attempts WHERE attempt_id=?",
                (h.store.get_message(assistant_id).metadata.mixin_attempt_id,),
            )
            .fetchone()[0]
        )
        assert seal is not None
    if damage.startswith("edited"):
        h.store.update_message_content(assistant_id, "user edited response")
    else:
        original_read = h.factory.repository.get_response_link

        def broken(cursor, call_id):
            link = original_read(cursor, call_id)
            if link is None or damage == "missing_link":
                return None
            return replace(link, semantic_revision_id=str(uuid4()))

        monkeypatch.setattr(h.factory.repository, "get_response_link", broken)
    before = len(h.entries)
    result = await h.controller.submit_draft("next", session_id="session-1")
    assert len(h.entries) == before, (result, h.failures)
    assert h.failures


@pytest.mark.asyncio
async def test_result_seal_guards_rollback_gc_and_fork_authority(
    trace_harness, monkeypatch
):
    import asyncio
    import sqlite3

    from Tests.Chat.test_console_trace_graph_gc import _finish_legacy_migration
    from tldw_chatbook.DB.console_prompt_mixins import copy_conversation_mixins

    h = trace_harness
    service, _scope = select(h, monkeypatch)
    first = await h.controller.submit_draft("input", session_id="session-1")
    await h.controller.regenerate_message(first.assistant_message_id)
    assistant = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role.value == "assistant"
    )
    conversation = h.store._sessions["session-1"].persisted_conversation_id
    with h.db.transaction() as cursor:
        rows = cursor.execute(
            "SELECT attempt_id,replaces_attempt_id,result_revision_id,trace_call_id FROM console_prompt_mixin_attempts ORDER BY rowid"
        ).fetchall()
        assert rows[0][3] is not None and rows[0][2] is None
        assert rows[1][1] == rows[0][0] and rows[1][2] is not None
        for column in ("trace_call_id", "result_revision_id"):
            owner = rows[0][0] if column == "trace_call_id" else rows[1][0]
            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute(
                    f"UPDATE console_prompt_mixin_attempts SET {column}=NULL WHERE attempt_id=?",
                    (owner,),
                )
    fork = h.db.add_conversation({"title": "snapshot fork"})
    user = h.db.add_message(
        {"conversation_id": fork, "sender": "user", "content": "input"}
    )
    reply = h.db.add_message(
        {
            "conversation_id": fork,
            "sender": "assistant",
            "content": "answer",
            "parent_message_id": user,
        }
    )
    with h.db.transaction() as cursor:
        copy_conversation_mixins(
            cursor,
            source_conversation_id=conversation,
            destination_conversation_id=fork,
            message_ids={
                first.user_message_id: user,
                assistant.persisted_message_id: reply,
            },
        )
        assert all(
            tuple(row) == (None, None, None)
            for row in cursor.execute(
                "SELECT replaces_attempt_id,result_revision_id,trace_call_id FROM console_prompt_mixin_attempts WHERE conversation_id=?",
                (fork,),
            ).fetchall()
        )
    _finish_legacy_migration(h.db)
    await asyncio.to_thread(
        h.store.persistence.purge_console_trace,
        conversation_id=conversation,
        request_id="mixin-proof-gc",
        detached_at="2026-11-01T22:00:00Z",
    )
    assert (
        h.db.get_connection()
        .execute(
            "SELECT trace_call_id FROM console_prompt_mixin_attempts WHERE conversation_id=? AND attempt_id=?",
            (conversation, rows[0][0]),
        )
        .fetchone()[0]
        is None
    )
    assert (
        await service.retry_snapshot("session-1", assistant_message_id=assistant.id)
    ).items


@pytest.mark.asyncio
async def test_result_seal_write_failure_rolls_back_terminal_projection(
    trace_harness, monkeypatch
):
    from tldw_chatbook.DB import console_prompt_mixins

    h = trace_harness
    select(h, monkeypatch)
    first = await h.controller.submit_draft("input", session_id="session-1")
    original = console_prompt_mixins.seal_attempt_result

    def fail(cursor, **kwargs):
        original(cursor, **kwargs)
        raise RuntimeError("forced seal rollback")

    monkeypatch.setattr(console_prompt_mixins, "seal_attempt_result", fail)
    await h.controller.regenerate_message(first.assistant_message_id)
    with h.db.transaction() as cursor:
        row = cursor.execute(
            "SELECT a.result_revision_id,m.assistant_generation_state FROM console_prompt_mixin_attempts a JOIN messages m ON m.id=a.assistant_message_id WHERE a.replaces_attempt_id IS NOT NULL"
        ).fetchone()
        assert row[0] is None and row[1] != "complete"


@pytest.mark.asyncio
async def test_promoted_history_gets_no_seal_and_first_durable_trace_can_retry_off(
    trace_harness, monkeypatch
):
    import asyncio

    from Tests.Chat.test_console_turn_resend import _restore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    h = trace_harness
    capture_mode(monkeypatch, False)
    h.store.create_session(session_id="temp", ephemeral=True)
    scope = Scope()
    scope.release.set()
    service = ConsolePromptMixinService(
        store=h.store, prompt_scope_service=scope, persistence=h.store.persistence
    )
    monkeypatch.setattr(h.controller, "prompt_mixin_service", lambda: service)
    service.replace_preferences(
        "temp",
        expected_revision=0,
        value=MixinPreferences(entries=(MixinSelection(A, True, "A", "before"),)),
    )
    first = await h.controller.submit_draft("temporary", session_id="temp")
    await h.controller.regenerate_message(first.assistant_message_id)
    reply = next(
        row
        for row in reversed(h.store.messages_for_session("temp"))
        if row.role.value == "assistant"
    )
    h.store.update_message_content(reply.id, "edited temporary result")
    conversation = await asyncio.to_thread(h.store.promote_ephemeral_session, "temp")
    with h.db.transaction() as cursor:
        assert all(
            tuple(row) == (None, None)
            for row in cursor.execute(
                "SELECT result_revision_id,trace_call_id FROM console_prompt_mixin_attempts WHERE conversation_id=?",
                (conversation,),
            )
        )
    capture_mode(monkeypatch, True)
    h.adapter_error = True
    durable = await h.controller.submit_draft("durable", session_id="temp")
    assert len(h.entries) == 3, durable
    original = h.entries[-1][-1]["content"]
    await h.store.persist_prompt_mixin_preferences("temp")
    path = h.db.db_path
    h.db.close()
    database = CharactersRAGDB(path, "promoted-reopen")
    try:
        restored = _restore(database, conversation)
        restored.controller.provider_gateway = h.gateway
        new_service = ConsolePromptMixinService(
            store=restored.store,
            prompt_scope_service=scope,
            persistence=restored.store.persistence,
        )
        monkeypatch.setattr(
            restored.controller, "prompt_mixin_service", lambda: new_service
        )
        scope.records.clear()
        capture_mode(monkeypatch, False)
        h.adapter_error = False
        failed = next(
            row
            for row in restored.store.all_messages_for_session(restored.session_id)
            if row.persisted_message_id == durable.assistant_message_id
        )
        restored.store.set_active_leaf(restored.session_id, failed.id)
        await restored.controller.retry_message(failed.id)
        assert len(h.entries) == 4 and h.entries[-1][-1]["content"] == original
        with database.transaction() as cursor:
            seal = cursor.execute(
                "SELECT result_revision_id FROM console_prompt_mixin_attempts WHERE attempt_id=?",
                (restored.store.get_message(failed.id).metadata.mixin_attempt_id,),
            ).fetchone()[0]
            revision = cursor.execute(
                "SELECT revision_id FROM console_trace_semantic_revisions WHERE live_message_id=? ORDER BY revision_sequence DESC LIMIT 1",
                (failed.persisted_message_id,),
            ).fetchone()[0]
            assert seal == revision
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("damage", ["attempt", "missing_items", "corrupt_items"])
async def test_regenerate_current_choices_recovers_unavailable_original(
    trace_harness, monkeypatch, capture, damage
):
    from Tests.Chat.test_console_prompt_mixin_admission import B
    from tldw_chatbook.DB.console_prompt_mixins import read_attempt

    h = trace_harness
    capture_mode(monkeypatch, capture)
    service, scope = select(h, monkeypatch)
    h.adapter_error = True
    first = await h.controller.submit_draft("input", session_id="session-1")
    original = h.store.get_message(first.assistant_message_id)
    old_attempt = original.metadata.mixin_attempt_id
    conversation = h.store._sessions["session-1"].persisted_conversation_id
    with h.db.transaction() as cursor:
        origin_call = cursor.execute(
            "SELECT trace_call_id FROM console_prompt_mixin_attempts WHERE attempt_id=?",
            (old_attempt,),
        ).fetchone()[0]
        assert (origin_call is not None) is capture
        if damage == "attempt":
            cursor.execute(
                "DELETE FROM console_prompt_mixin_attempts WHERE attempt_id=?",
                (old_attempt,),
            )
        else:
            item = cursor.execute(
                "SELECT conversation_id,attempt_id,snapshot_id FROM console_prompt_mixin_attempt_items WHERE attempt_id=?",
                (old_attempt,),
            ).fetchone()
            cursor.execute(
                "DELETE FROM console_prompt_mixin_attempt_items WHERE attempt_id=?",
                (old_attempt,),
            )
            if damage == "corrupt_items":
                cursor.execute(
                    "INSERT INTO console_prompt_mixin_attempt_items(conversation_id,attempt_id,snapshot_id,ordinal) VALUES(?,?,?,3)",
                    tuple(item),
                )
    # The original library source is gone; only a different current choice exists.
    del scope.records[A.record_uuid]
    scope.records[B.record_uuid]["prompt_definition"]["body"] = "current B"
    prefs = service.preferences("session-1")
    service.replace_preferences(
        "session-1",
        expected_revision=prefs.revision,
        value=replace(prefs, entries=(MixinSelection(B, True, "B", "before"),)),
    )
    h.adapter_error = False
    refused = await h.controller.retry_message(original.id)
    assert "unavailable" in refused.visible_copy and len(h.entries) == 1
    from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService

    refreshed = h.store.get_message(original.id)
    assert refreshed.mixin_usage_count is None
    assert (
        next(
            row
            for row in ConsoleMessageActionService().available_actions(refreshed)
            if row.action_id == "regenerate"
        ).label
        == "Regenerate with current mix-ins"
    )
    regenerated = await h.controller.regenerate_message(original.id)
    assert len(h.entries) == 2, (regenerated, h.failures)
    assert h.entries[-1][-1]["content"] == "current B\n\ninput"
    with h.db.transaction() as cursor:
        rows = cursor.execute(
            "SELECT attempt_id,retry_of_attempt_id,replaces_attempt_id,result_revision_id,trace_call_id FROM console_prompt_mixin_attempts WHERE attempt_id<>?",
            (old_attempt,),
        ).fetchall()
        assert len(rows) == 1
        new = rows[0]
        assert tuple(new)[1:4] == (None, None, None)
        assert new[4] is None  # Regenerate is not a fresh trace origin.
        assert (
            read_attempt(cursor, conversation, attempt_id=old_attempt).status
            == "unavailable"
        )
        assert (
            read_attempt(cursor, conversation, attempt_id=new[0])
            .snapshot.items[0]
            .definition.body
            == "current B"
        )


@pytest.mark.asyncio
async def test_regenerate_unavailable_original_still_requires_database_write(
    trace_harness, monkeypatch
):
    import sqlite3

    h = trace_harness
    capture_mode(monkeypatch, False)
    select(h, monkeypatch)
    h.adapter_error = True
    first = await h.controller.submit_draft("input", session_id="session-1")
    original = h.store.get_message(first.assistant_message_id)
    with h.db.transaction() as cursor:
        cursor.execute(
            "DELETE FROM console_prompt_mixin_attempts WHERE attempt_id=?",
            (original.metadata.mixin_attempt_id,),
        )
    calls = []

    def fail_read(*args, **kwargs):
        calls.append("read")
        raise sqlite3.OperationalError("database unavailable")

    def fail_write(*args, **kwargs):
        calls.append("write")
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(
        h.store.persistence, "get_conversation_mixin_attempt", fail_read
    )
    monkeypatch.setattr(
        h.store.persistence, "save_conversation_mixin_attempt", fail_write
    )
    h.adapter_error = False
    await h.controller.regenerate_message(original.id)
    assert calls == ["read", "write"]
    assert len(h.entries) == 1
    with h.db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_prompt_mixin_attempts"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize(
    ("count", "marked", "label", "action"),
    [
        (2, True, "Retry with original mix-ins", "retry"),
        (0, True, "Retry", "retry"),
        (0, False, "Retry", "retry"),
        (None, True, "Regenerate with current mix-ins", "regenerate"),
    ],
)
def test_retry_copy_uses_verified_usage_projection(count, marked, label, action):
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        status="failed",
        mixin_usage_count=count,
        metadata=MessageMetadata(
            mixin_attempt_id="original", mixin_snapshot_schema_version=1
        )
        if marked
        else None,
    )
    actions = ConsoleMessageActionService().available_actions(message)
    assert next(row for row in actions if row.action_id == action).label == label
    assert not any(
        row.action_id == ("regenerate" if action == "retry" else "retry")
        for row in actions
    )


@pytest.mark.asyncio
async def test_operational_inspection_and_display_count_never_authorize_retry(
    trace_harness, monkeypatch
):
    from tldw_chatbook.Chat.console_prompt_mixins import MixinSnapshotUnavailableError
    from tldw_chatbook.UI.Console_Modules.inspector_data import (
        build_mixin_snapshot_loader,
    )

    h = trace_harness
    _service, scope = select(h, monkeypatch)
    capture_mode(monkeypatch, False)
    h.adapter_error = True
    result = await h.controller.submit_draft("original", session_id="session-1")
    message = h.store.get_message(result.assistant_message_id)
    assert message.mixin_usage_count == 1
    loader = build_mixin_snapshot_loader(h.controller, "session-1", lambda: True)
    scope.records.clear()
    snapshot = await loader(message.id)
    assert snapshot.items[0].definition.body == "body A"
    conversation = h.store._sessions["session-1"].persisted_conversation_id
    with h.db.transaction() as cursor:
        cursor.execute(
            "DELETE FROM console_prompt_mixin_attempts WHERE conversation_id=?",
            (conversation,),
        )
    h.store._message_or_raise(message.id).mixin_usage_count = 999
    with pytest.raises(MixinSnapshotUnavailableError):
        await loader(message.id)
    calls = len(h.entries)
    retry = await h.controller.retry_message(message.id)
    assert len(h.entries) == calls
    assert not retry.accepted
    from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService

    refreshed = h.store.get_message(message.id)
    assert refreshed.mixin_usage_count is None
    actions = ConsoleMessageActionService().available_actions(refreshed)
    assert (
        next(row for row in actions if row.action_id == "regenerate").label
        == "Regenerate with current mix-ins"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("damage", [None, "missing", "corrupt_marker"])
async def test_off_loop_usage_hydration_qualifies_raw_receipts(
    trace_harness, monkeypatch, enabled, damage
):
    from types import SimpleNamespace

    from Tests.Chat.test_console_turn_resend import _restore
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_conversation_hydration import (
        prepare_console_session_data,
    )

    h = trace_harness
    select(h, monkeypatch, enabled=enabled)
    capture_mode(monkeypatch, False)
    h.adapter_error = True
    first = await h.controller.submit_draft("input", session_id="session-1")
    assistant = h.store.get_message(first.assistant_message_id)
    conversation = h.store._sessions["session-1"].persisted_conversation_id
    if damage:
        with h.db.transaction() as cursor:
            if damage == "missing":
                cursor.execute(
                    "DELETE FROM console_prompt_mixin_attempts WHERE conversation_id=?",
                    (conversation,),
                )
            else:
                cursor.execute(
                    "DROP TRIGGER console_prompt_mixin_message_marker_preserve"
                )
                cursor.execute(
                    "UPDATE messages SET metadata_json=? WHERE id=?",
                    (
                        '{"mixin_snapshot_schema_version":true}',
                        assistant.persisted_message_id,
                    ),
                )
    tree = ChatConversationService(h.db).get_conversation_tree(
        conversation, root_limit=100, depth_cap=100
    )
    prepared = await prepare_console_session_data(
        app=SimpleNamespace(chachanotes_db=h.db),
        store=h.store,
        conversation_id=conversation,
        tree=tree,
    )
    expected = int(enabled) if damage is None else None
    assert prepared.mixin_usage_counts[assistant.persisted_message_id] == expected
    restored = _restore(h.db, conversation)
    reply = next(
        row
        for row in restored.store.messages_for_session(restored.session_id)
        if row.persisted_message_id == assistant.persisted_message_id
    )
    assert reply.mixin_usage_count == expected
    restored.store.end_app_runtime()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "new_attempt_write"])
async def test_valid_original_failure_retains_retry_usage_and_exact_snapshot(
    trace_harness, monkeypatch, failure
):
    from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
    from tldw_chatbook.DB import console_prompt_mixins

    h = trace_harness
    service, _scope = select(h, monkeypatch)
    capture_mode(monkeypatch, False)
    h.adapter_error = True
    first = await h.controller.submit_draft("input", session_id="session-1")
    if failure == "new_attempt_write":

        def fail(*_args, **_kwargs):
            raise RuntimeError("injected new attempt write failure")

        monkeypatch.setattr(console_prompt_mixins, "write_attempt", fail)
        h.adapter_error = False
    await h.controller.retry_message(first.assistant_message_id)
    assert len(h.entries) == (2 if failure == "transport" else 1)
    reply = h.store.get_message(first.assistant_message_id)
    assert reply.mixin_usage_count == 1
    assert (
        next(
            row
            for row in ConsoleMessageActionService().available_actions(reply)
            if row.action_id == "retry"
        ).label
        == "Retry with original mix-ins"
    )
    assert (
        await service.retry_snapshot("session-1", assistant_message_id=reply.id)
    ).items[0].definition.body == "body A"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["regenerate", "edit", "continue"])
@pytest.mark.parametrize("boundary", ["resolution", "transform"])
async def test_current_choice_replay_refuses_source_rebind(
    trace_harness, monkeypatch, action, boundary
):
    import asyncio

    h = trace_harness
    capture_mode(monkeypatch, False)
    _, scope = select(h, monkeypatch)
    first = await h.controller.submit_draft("input", session_id="session-1")
    count = len(h.entries)
    before = h.store.active_path_message_ids("session-1")
    h.store.set_session_draft("session-1", "keep draft")
    name = (
        "_resolve_for_send_bounded"
        if boundary == "resolution"
        else "_apply_chat_dictionaries"
    )
    original = getattr(h.controller, name)
    entered, release = asyncio.Event(), asyncio.Event()

    async def rebind(*args, **kwargs):
        result = await original(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(h.controller, name, rebind)
    if action == "regenerate":
        operation = h.controller.regenerate_message(first.assistant_message_id)
    elif action == "edit":
        operation = h.controller.edit_and_resend_message(
            first.user_message_id, "edited"
        )
    else:
        operation = h.controller.continue_from_message(first.assistant_message_id)
    task = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        scope.source = object()
        scope.database = object()
    finally:
        release.set()
    result = await asyncio.wait_for(task, 5)
    assert not result.accepted
    assert len(h.entries) == count
    assert h.store._sessions["session-1"].draft == "keep draft"
    assert tuple(
        row
        for row in h.store.active_path_message_ids("session-1")
        if h.store.get_message(row).role.value != "system"
    ) == tuple(row for row in before if h.store.get_message(row).role.value != "system")


@pytest.mark.asyncio
async def test_unavailable_mixin_preview_has_bounded_recovery_copy(
    trace_harness, monkeypatch
):
    from loguru import logger

    h = trace_harness
    _, scope = select(h, monkeypatch)
    scope.records.clear()
    logs = []
    sink = logger.add(lambda message: logs.append(message.record))
    try:
        preview = await h.controller.build_context_snapshot(
            "private draft", session_id="session-1"
        )
    finally:
        logger.remove(sink)
    assert "Disable or remove" in preview.next_send_payload["tools"]["preview_note"]
    assert "internal error" not in str(preview.next_send_payload)
    assert not any(row["exception"] for row in logs)
    assert not any("private draft" in row["message"] for row in logs)
    assert h.entries == []


@pytest.mark.asyncio
async def test_continue_refuses_source_rebind_during_hook(trace_harness, monkeypatch):
    import asyncio

    h = trace_harness
    capture_mode(monkeypatch, False)
    _, scope = select(h, monkeypatch)
    first = await h.controller.submit_draft("input", session_id="session-1")
    entered, release = asyncio.Event(), asyncio.Event()
    original = h.controller._submission_hook_input

    async def held(*args, **kwargs):
        result = await original(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(h.controller, "_submission_hook_input", held)
    task = asyncio.create_task(
        h.controller.continue_from_message(first.assistant_message_id)
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        scope.source = object()
        scope.database = object()
    finally:
        release.set()
    result = await asyncio.wait_for(task, 5)
    assert not result.accepted and len(h.entries) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["hook", "provider", "native_commit"])
async def test_refused_native_send_keeps_usage_facts_until_acceptance(
    trace_harness, monkeypatch, refusal
):
    h = trace_harness
    capture_mode(monkeypatch, False)
    service, scope = select(h, monkeypatch)
    before = service.preferences("session-1")
    scope.records[A.record_uuid].update(version=2)
    scope.records[A.record_uuid]["prompt_definition"]["placement"] = "after"
    if refusal == "hook":
        original = h.controller.hook_admission_reason

        async def refuse():
            return "Hook admission refused"

        monkeypatch.setattr(h.controller, "hook_admission_reason", refuse)
    elif refusal == "native_commit":
        h.db.get_connection().execute(
            "CREATE TRIGGER review_fail_mixin_commit BEFORE INSERT ON console_dispatch_checkpoints "
            "BEGIN SELECT RAISE(ABORT, 'injected native transaction failure'); END"
        )
    else:
        original = h.gateway.resolve_for_send

        async def refuse(selection):
            return replace(
                await original(selection),
                ready=False,
                visible_copy="Provider unavailable",
            )

        monkeypatch.setattr(h.gateway, "resolve_for_send", refuse)
    result = await h.controller.submit_draft("input", session_id="session-1")
    assert not result.accepted and not result.provider_started
    assert not h.entries
    assert service.preferences("session-1") == before
    if refusal == "hook":
        monkeypatch.setattr(h.controller, "hook_admission_reason", original)
    elif refusal == "native_commit":
        h.db.get_connection().execute("DROP TRIGGER review_fail_mixin_commit")
    else:
        monkeypatch.setattr(h.gateway, "resolve_for_send", original)
    accepted = (
        await h.controller.retry_library_preparation(result.preparation_id)
        if refusal == "native_commit"
        else await h.controller.submit_draft("input", session_id="session-1")
    )
    assert accepted.accepted and accepted.provider_started
    facts = service.preferences("session-1")
    assert facts.revision == before.revision + 1
    assert facts.entries[0].last_used_version == 2
    assert facts.entries[0].placement == "after"
    assert h.entries[-1][-1]["content"] == "input\n\nbody A"


@pytest.mark.asyncio
@pytest.mark.parametrize("supplied", [False, True])
async def test_queue_usage_publishes_only_after_admission(
    trace_harness, monkeypatch, supplied
):
    h = trace_harness
    capture_mode(monkeypatch, False)
    service, scope = select(h, monkeypatch)
    before = service.preferences("session-1")
    scope.records[A.record_uuid]["version"] = 2
    registry = h.controller.prompt_queue_registry
    registry.begin_chain(
        "session-1",
        context_epoch=h.store.conversation_context_epoch("session-1"),
        expected_revision=0,
    )
    configuration = None
    if supplied:
        snapshot = await service.capture("session-1", expected_revision=before.revision)
        configuration = replace(
            h.controller.resolve_turn_configuration_snapshot("session-1"),
            prompt_mixins=snapshot,
        )
    revision = registry.snapshot("session-1").revision
    refused = await h.controller.queue_prompt(
        "session-1",
        text="queued",
        expected_revision=revision - 1,
        configuration=configuration,
    )
    assert not refused.applied
    assert service.preferences("session-1") == before
    accepted = await h.controller.queue_prompt(
        "session-1",
        text="queued",
        expected_revision=revision,
        configuration=configuration,
    )
    assert accepted.applied
    assert service.preferences("session-1").entries[0].last_used_version == 2
    assert not h.entries


@pytest.mark.asyncio
async def test_original_retry_ignores_unavailable_current_usage_preferences(
    trace_harness, monkeypatch
):
    from tldw_chatbook.Chat.console_prompt_mixins import MixinPreferenceRead

    h = trace_harness
    capture_mode(monkeypatch, False)
    _service, scope = select(h, monkeypatch)
    h.adapter_error = True
    sent = await h.controller.submit_draft("input", session_id="session-1")
    assert sent.accepted
    reply = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role.value == "assistant"
    )
    unavailable = MixinPreferenceRead("unsupported", None, {"schema_version": 99})
    h.store._sessions["session-1"].prompt_mixin_preferences = unavailable
    scope.records.clear()
    h.adapter_error = False
    await h.controller.retry_message(reply.id)
    assert len(h.entries) == 2
    assert h.entries[-1][-1]["content"] == "body A\n\ninput"
    assert h.store._sessions["session-1"].prompt_mixin_preferences is unavailable


@pytest.mark.asyncio
@pytest.mark.parametrize("current_state", ["unsupported", "removed"])
async def test_frozen_queue_acceptance_and_drain_ignore_missing_current_choices(
    trace_harness, monkeypatch, current_state
):
    from tldw_chatbook.Chat.console_prompt_mixins import MixinPreferenceRead
    from tldw_chatbook.Chat.console_prompt_queue import PromptQueuePauseReason

    h = trace_harness
    capture_mode(monkeypatch, False)
    service, scope = select(h, monkeypatch)
    baseline = await h.controller.submit_draft("baseline", session_id="session-1")
    assert baseline.provider_started
    h.entries.clear()
    snapshot = await service.capture("session-1", expected_revision=1)
    configuration = replace(
        h.controller.resolve_turn_configuration_snapshot("session-1"),
        prompt_mixins=snapshot,
    )
    session = h.store._sessions["session-1"]
    if current_state == "unsupported":
        choices = MixinPreferenceRead("unsupported", None, {"schema_version": 99})
    else:
        choices = MixinPreferenceRead("supported", MixinPreferences(revision=2), None)
    session.prompt_mixin_preferences = choices
    scope.records.clear()
    registry = h.controller.prompt_queue_registry
    registry.begin_chain(
        "session-1",
        context_epoch=h.store.conversation_context_epoch("session-1"),
        expected_revision=0,
    )
    accepted = await h.controller.queue_prompt(
        "session-1",
        text="queued",
        expected_revision=registry.snapshot("session-1").revision,
        configuration=configuration,
    )
    assert accepted.applied
    assert session.prompt_mixin_preferences is choices
    registry.pause(
        "session-1",
        reason=PromptQueuePauseReason.MANUAL,
        expected_revision=registry.snapshot("session-1").revision,
    )
    resumed = await h.controller.prompt_queue_coordinator.resume_and_drain("session-1")
    assert len(h.entries) == 1, resumed
    assert h.entries[-1][-1]["content"] == "body A\n\nqueued"
    assert session.prompt_mixin_preferences is choices


@pytest.mark.asyncio
@pytest.mark.parametrize("ephemeral", [False, True])
async def test_first_native_acceptance_publishes_usage(
    trace_harness, monkeypatch, ephemeral
):
    h = trace_harness
    capture_mode(monkeypatch, False)
    session = h.store.create_session(
        session_id="first-usage",
        ephemeral=ephemeral,
        settings=h.store._sessions["session-1"].settings,
    )
    _service, _scope = select(h, monkeypatch, session_id=session.id)
    accepted = await h.controller.submit_draft("input", session_id=session.id)
    assert accepted.accepted and len(h.entries) == 1, accepted
    assert (
        session.prompt_mixin_preferences.preferences.entries[0].last_used_version == 1
    )


@pytest.mark.asyncio
async def test_used_revision_flush_reopen_and_racing_selection_cas(
    trace_harness, monkeypatch
):
    import asyncio
    from threading import Event

    from Tests.Chat.test_console_turn_resend import _restore
    from tldw_chatbook.Chat.console_settings_durability import (
        ConsoleSettingsDurabilityOwner,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    h = trace_harness
    capture_mode(monkeypatch, False)
    service, scope = select(h, monkeypatch)
    await h.store.persist_prompt_mixin_preferences("session-1")
    intent_revision = service.preferences("session-1").revision
    accepted = await h.controller.submit_draft("input", session_id="session-1")
    assert accepted.accepted
    assert service.preferences("session-1").revision == intent_revision
    await h.store.persist_prompt_mixin_preferences("session-1")
    database = CharactersRAGDB(h.db.db_path, "review-reopen")
    restored = _restore(
        database, h.store._sessions["session-1"].persisted_conversation_id
    )
    owner = ConsoleSettingsDurabilityOwner()
    next_service = ConsolePromptMixinService(
        store=restored.store,
        prompt_scope_service=scope,
        persistence=restored.store.persistence,
        settings_durability_owner=owner,
    )
    entered, release = Event(), Event()
    try:
        prefs = next_service.preferences(restored.session_id)
        assert prefs.entries[0].last_used_version == 1
        assert prefs.revision > intent_revision
        scope.records[A.record_uuid]["version"] = 2
        snapshot = await next_service.capture(
            restored.session_id, expected_revision=prefs.revision
        )
        original = restored.store.persistence.save_conversation_mixin_preferences
        writes = []

        def held(**kwargs):
            writes.append(kwargs["value"])
            if len(writes) == 1:
                entered.set()
                assert release.wait(5)
            return original(**kwargs)

        monkeypatch.setattr(
            restored.store.persistence, "save_conversation_mixin_preferences", held
        )
        next_service.publish_accepted_usage(restored.session_id, snapshot)
        assert await asyncio.to_thread(entered.wait, 5)
        current = next_service.preferences(restored.session_id)
        assert current.revision == prefs.revision
        next_service.replace_preferences(
            restored.session_id,
            expected_revision=current.revision,
            value=replace(
                current, entries=(replace(current.entries[0], enabled=False),)
            ),
        )
        release.set()
        await owner.close_and_drain()
        assert len(writes) == 2
        saved = restored.store.persistence.get_conversation_mixin_preferences(
            restored.store._sessions[restored.session_id].persisted_conversation_id
        ).preferences
        assert saved.entries[0].last_used_version == 2 and not saved.entries[0].enabled
        assert next_service.preferences(restored.session_id).entries == saved.entries
        # The earlier same-intent save and later CAS each advanced SQL's own
        # durable revision without overwriting the newer live selection.
        assert saved.revision == prefs.revision + 2
    finally:
        release.set()
        await owner.close_and_drain()
        await restored.controller.shutdown()
        for session in restored.store.sessions():
            restored.store.close_session(session.id)
        restored.store.end_app_runtime()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reopen", [False, True])
async def test_successive_accepted_placement_observations_append_in_observed_order(
    trace_harness, monkeypatch, reopen
):
    from Tests.Chat.test_console_prompt_mixin_admission import B
    from Tests.Chat.test_console_turn_resend import _restore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    h = trace_harness
    capture_mode(monkeypatch, False)
    service, scope = select(h, monkeypatch)
    prefs = service.preferences("session-1")
    service.replace_preferences(
        "session-1",
        expected_revision=prefs.revision,
        value=replace(
            prefs, entries=(*prefs.entries, MixinSelection(B, True, "B", "before"))
        ),
    )
    scope.records[B.record_uuid]["prompt_definition"].update(
        body="body B", placement="after"
    )
    scope.records[B.record_uuid]["version"] = 2
    first = await h.controller.submit_draft("first", session_id="session-1")
    assert (
        first.accepted and h.entries[-1][-1]["content"] == "body A\n\nfirst\n\nbody B"
    )
    observed = service.preferences("session-1")
    assert observed.entries[-1].placement == "after"
    assert observed.revision == prefs.revision + 2
    await h.store.persist_prompt_mixin_preferences("session-1")
    database = restored = None
    controller, session_id = h.controller, "session-1"
    try:
        if reopen:
            database = CharactersRAGDB(h.db.db_path, "successive-layout-reopen")
            restored = _restore(
                database, h.store._sessions[session_id].persisted_conversation_id
            )
            controller, session_id = restored.controller, restored.session_id
            restored.store.replace_session_settings(
                session_id, h.store._sessions["session-1"].settings
            )
            await restored.store.hydrate_session_library_policy(session_id)
            controller.provider_gateway = h.gateway
            service = ConsolePromptMixinService(
                store=restored.store,
                prompt_scope_service=scope,
                persistence=restored.store.persistence,
            )
            monkeypatch.setattr(controller, "prompt_mixin_service", lambda: service)
            assert service.preferences(session_id).entries == observed.entries
        scope.records[A.record_uuid]["prompt_definition"]["placement"] = "after"
        scope.records[A.record_uuid]["version"] = 2
        second = await controller.submit_draft("second", session_id=session_id)
        assert second.accepted and second.provider_started, second
        assert h.entries[-1][-1]["content"] == "second\n\nbody B\n\nbody A"
        assert tuple(row.source for row in service.preferences(session_id).entries) == (
            B,
            A,
        )
    finally:
        if restored is not None:
            await restored.controller.shutdown()
            for session in restored.store.sessions():
                restored.store.close_session(session.id)
            restored.store.end_app_runtime()
        if database is not None:
            database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", ["retry", "supplied_send", "supplied_queue"])
async def test_frozen_replay_cannot_roll_current_accepted_layout_back(
    trace_harness, monkeypatch, replay
):
    from tldw_chatbook.Chat.console_prompt_queue import PromptQueuePauseReason

    h = trace_harness
    capture_mode(monkeypatch, False)
    service, scope = select(h, monkeypatch)
    snapshot = await service.capture("session-1", expected_revision=1)
    configuration = replace(
        h.controller.resolve_turn_configuration_snapshot("session-1"),
        prompt_mixins=snapshot,
    )
    h.adapter_error = True
    first = await h.controller.submit_draft("first", session_id="session-1")
    h.adapter_error = False
    scope.records[A.record_uuid]["version"] = 2
    scope.records[A.record_uuid]["prompt_definition"]["placement"] = "after"
    current = await h.controller.submit_draft("current", session_id="session-1")
    assert current.accepted
    before_replay = service.preferences("session-1")
    assert before_replay.entries[0].placement == "after"
    if replay == "retry":
        result = await h.controller.retry_message(first.assistant_message_id)
        assert result.accepted
        expected = "body A\n\nfirst"
    elif replay == "supplied_send":
        result = await h.controller.submit_draft(
            "frozen", session_id="session-1", configuration=configuration
        )
        assert result.accepted
        expected = "body A\n\nfrozen"
    else:
        registry = h.controller.prompt_queue_registry
        registry.begin_chain(
            "session-1",
            context_epoch=h.store.conversation_context_epoch("session-1"),
            expected_revision=registry.snapshot("session-1").revision,
        )
        result = await h.controller.queue_prompt(
            "session-1",
            text="frozen",
            expected_revision=registry.snapshot("session-1").revision,
            configuration=configuration,
        )
        assert result.applied
        assert service.preferences("session-1").entries[0].placement == "after"
        registry.pause(
            "session-1",
            reason=PromptQueuePauseReason.MANUAL,
            expected_revision=registry.snapshot("session-1").revision,
        )
        await h.controller.prompt_queue_coordinator.resume_and_drain("session-1")
        expected = "body A\n\nfrozen"
    assert h.entries[-1][-1]["content"] == expected
    facts = service.preferences("session-1")
    assert facts.revision == before_replay.revision
    assert facts.entries[0].placement == "after"


@pytest.mark.asyncio
async def test_current_queue_observes_layout_once_and_drain_preserves_later_intent(
    trace_harness, monkeypatch
):
    from tldw_chatbook.Chat.console_prompt_queue import PromptQueuePauseReason

    h = trace_harness
    capture_mode(monkeypatch, False)
    service, scope = select(h, monkeypatch)
    assert (
        await h.controller.submit_draft("baseline", session_id="session-1")
    ).accepted
    scope.records[A.record_uuid]["version"] = 2
    scope.records[A.record_uuid]["prompt_definition"]["placement"] = "after"
    registry = h.controller.prompt_queue_registry
    registry.begin_chain(
        "session-1",
        context_epoch=h.store.conversation_context_epoch("session-1"),
        expected_revision=registry.snapshot("session-1").revision,
    )
    accepted = await h.controller.queue_prompt(
        "session-1",
        text="queued",
        expected_revision=registry.snapshot("session-1").revision,
    )
    assert accepted.applied
    observed = service.preferences("session-1")
    assert observed.revision == 2 and observed.entries[0].placement == "after"
    later = service.replace_preferences(
        "session-1",
        expected_revision=observed.revision,
        value=replace(
            observed,
            entries=(replace(observed.entries[0], placement="before", enabled=False),),
        ),
    )
    registry.pause(
        "session-1",
        reason=PromptQueuePauseReason.MANUAL,
        expected_revision=registry.snapshot("session-1").revision,
    )
    await h.controller.prompt_queue_coordinator.resume_and_drain("session-1")
    assert h.entries[-1][-1]["content"] == "queued\n\nbody A"
    assert service.preferences("session-1") == later


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["regenerate", "edit", "continue"])
async def test_current_choice_routes_publish_accepted_layout(
    trace_harness, monkeypatch, route
):
    h = trace_harness
    capture_mode(monkeypatch, False)
    service, scope = select(h, monkeypatch)
    first = await h.controller.submit_draft("first", session_id="session-1")
    before = service.preferences("session-1")
    scope.records[A.record_uuid]["version"] = 2
    scope.records[A.record_uuid]["prompt_definition"]["placement"] = "after"
    if route == "regenerate":
        result = await h.controller.regenerate_message(first.assistant_message_id)
    elif route == "edit":
        result = await h.controller.edit_and_resend_message(
            first.user_message_id, "edited"
        )
    else:
        result = await h.controller.continue_from_message(first.assistant_message_id)
    assert result.accepted and len(h.entries) == 2
    assert h.entries[-1][-1]["content"].endswith("\n\nbody A")
    facts = service.preferences("session-1")
    assert facts.revision == before.revision + 1
    assert facts.entries[0].placement == "after"
    assert facts.entries[0].last_used_version == 2


@pytest.mark.asyncio
async def test_cancelled_preference_waiter_keeps_real_database_drain_owned(
    trace_harness, monkeypatch
):
    import asyncio
    from threading import Event

    h = trace_harness
    service, _scope = select(h, monkeypatch)
    entered, release, finished = Event(), Event(), Event()
    original = h.store.persistence.save_conversation_mixin_preferences

    def held(**kwargs):
        entered.set()
        assert release.wait(5)
        try:
            return original(**kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(
        h.store.persistence, "save_conversation_mixin_preferences", held
    )
    waiter = asyncio.create_task(h.store.persist_prompt_mixin_preferences("session-1"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not finished.is_set()
        release.set()
        outcome = await h.store.persist_prompt_mixin_preferences("session-1")
        assert not outcome.failed_components and not outcome.stale_components
        assert finished.is_set()
        saved = h.store.persistence.get_conversation_mixin_preferences(
            h.store._sessions["session-1"].persisted_conversation_id
        ).preferences
        assert saved.entries == service.preferences("session-1").entries
    finally:
        release.set()
        await asyncio.gather(waiter, return_exceptions=True)
