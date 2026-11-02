"""Integrated local-source journeys with real durable owners and adapter receipts."""

from dataclasses import replace

import pytest

from Tests.Chat.test_console_prompt_mixin_retry import capture_mode
from Tests.Chat.test_console_trace_current_turn_transforms import (
    trace_harness as trace_harness,  # noqa: PLC0414 - native fixture
)
from Tests.Chat.test_console_turn_resend import _restore
from tldw_chatbook.Chat.console_prompt_mixin_service import ConsolePromptMixinService
from tldw_chatbook.Chat.console_prompt_mixins import MixinSourceRef
from tldw_chatbook.Chat.console_prompt_queue import PromptQueuePauseReason
from tldw_chatbook.DB.console_prompt_mixins import read_attempt
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
async def local_journey(trace_harness, tmp_path, monkeypatch):
    h = trace_harness
    capture_mode(monkeypatch, False)
    database = PromptsDatabase(tmp_path / "prompts.sqlite", client_id="journey")
    h.scope = PromptScopeService(
        local_service=LocalPromptService(database), server_service=None
    )
    h.service = ConsolePromptMixinService(
        store=h.store, prompt_scope_service=h.scope, persistence=h.store.persistence
    )
    monkeypatch.setattr(h.controller, "prompt_mixin_service", lambda: h.service)
    h.record = await h.scope.save_prompt(
        mode="local",
        name="Scene",
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition={
            "kind": "text_mixin",
            "schema_version": 3,
            "body": "A literal {{scene}}",
            "placement": "before",
        },
    )
    source = MixinSourceRef(
        h.scope.mixin_source_identity(mode="local")[0], "local", h.record["uuid"]
    )
    await h.service.enable_reference(
        "session-1",
        source=source,
        target_incarnation_id=h.store._sessions["session-1"].incarnation_id,
        expected_revision=0,
    )
    try:
        yield h
    finally:
        database.close_connection()
        for session in h.store.sessions():
            h.store.close_session(session.id)
        h.store.end_app_runtime()


def toggle(h, enabled):
    prefs = h.service.preferences("session-1")
    return h.service.replace_preferences(
        "session-1",
        expected_revision=prefs.revision,
        value=replace(
            prefs, entries=tuple(replace(row, enabled=enabled) for row in prefs.entries)
        ),
    )


async def edit_source(h):
    h.record = await h.scope.save_prompt(
        mode="local",
        prompt_identifier=h.record["uuid"],
        expected_version=h.record["version"],
        name="Scene B",
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition={
            "kind": "text_mixin",
            "schema_version": 3,
            "body": "B literal {{scene}}",
            "placement": "before",
        },
    )


def receipt(
    h, assistant_id, expected_body, expected_version=None, *, current_input=None
):
    assistant = h.store.get_message(assistant_id)
    conversation = h.store._sessions["session-1"].persisted_conversation_id
    with h.db.transaction() as cursor:
        record = read_attempt(
            cursor, conversation, attempt_id=assistant.metadata.mixin_attempt_id
        )
        assert record.status == "recorded"
        anchors = cursor.execute(
            "SELECT user_message_id,assistant_message_id FROM console_prompt_mixin_attempts WHERE attempt_id=?",
            (assistant.metadata.mixin_attempt_id,),
        ).fetchone()
        assert anchors[1] == assistant.persisted_message_id
        clean = cursor.execute(
            "SELECT content FROM messages WHERE id=?", (anchors[0],)
        ).fetchone()[0]
    assert clean in {"input", "next", "Continue and extend the selected message."}
    assert [item.definition.body for item in record.snapshot.items] == (
        [expected_body] if expected_body else []
    )
    if expected_body:
        assert record.snapshot.items[0].record_version == expected_version
        assert h.entries[-1][-1]["content"] == expected_body + "\n\n" + (
            current_input or clean
        )
    else:
        assert h.entries[-1][-1]["content"] == clean
    return assistant.metadata.mixin_attempt_id


@pytest.mark.asyncio
@pytest.mark.parametrize("character", [False, True])
async def test_local_queue_retry_current_regenerate_reopen(local_journey, character):
    h = local_journey
    if character:
        h.store.set_session_character_name(
            "session-1", character_name="Detective", global_default="User"
        )
        h.store.seed_character_roleplay(
            "session-1",
            system_template="You are a detective.",
            greeting_template="",
            global_default="User",
        )
    toggle(h, False)
    baseline = await h.controller.submit_draft("baseline", session_id="session-1")
    assert baseline.provider_started
    h.entries.clear()
    toggle(h, True)
    other = h.store.create_session(title="Other", ephemeral=True)
    registry = h.controller.prompt_queue_registry
    registry.begin_chain(
        "session-1",
        context_epoch=h.store.conversation_context_epoch("session-1"),
        expected_revision=0,
    )
    queued = await h.controller.queue_prompt(
        "session-1",
        text="input",
        expected_revision=registry.snapshot("session-1").revision,
    )
    assert queued.applied
    registry.pause(
        "session-1",
        reason=PromptQueuePauseReason.MANUAL,
        expected_revision=registry.snapshot("session-1").revision,
    )
    await edit_source(h)
    toggle(h, False)
    h.store.switch_session(other.id)
    h.adapter_error = True
    resumed = await h.controller.prompt_queue_coordinator.resume_and_drain("session-1")
    assert len(h.entries) == 1, resumed
    reply = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role.value == "assistant"
    )
    original = receipt(h, reply.id, "A literal {{scene}}", 1)
    assert reply.mixin_usage_count == 1
    h.store.switch_session("session-1")
    h.adapter_error = False
    retried = await h.controller.retry_message(reply.id)
    assert retried.accepted and len(h.entries) == 2
    retried_id = receipt(h, reply.id, "A literal {{scene}}", 1)
    assert retried_id != original
    off = await h.controller.regenerate_message(reply.id)
    assert off.accepted and len(h.entries) == 3
    off_reply = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role.value == "assistant"
    )
    receipt(h, off_reply.id, None)
    toggle(h, True)
    current = await h.controller.regenerate_message(off_reply.id)
    assert current.accepted and len(h.entries) == 4
    current_reply = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role.value == "assistant"
    )
    receipt(h, current_reply.id, "B literal {{scene}}", 2)
    toggle(h, False)
    following = await h.controller.submit_draft("next", session_id="session-1")
    assert following.provider_started
    receipt(h, following.assistant_message_id, None)
    assert not any("literal {{scene}}" in str(row["content"]) for row in h.entries[-1])
    await h.store.persist_prompt_mixin_preferences("session-1")
    before = h.service.preferences("session-1")
    cid = h.store._sessions["session-1"].persisted_conversation_id
    restored = _restore(h.db, cid)
    try:
        service = ConsolePromptMixinService(
            store=restored.store,
            prompt_scope_service=h.scope,
            persistence=restored.store.persistence,
        )
        assert service.preferences(restored.session_id).entries == before.entries
        assert (
            service.preferences(
                restored.store.create_session(ephemeral=True).id
            ).entries
            == ()
        )
        assert (
            h.db.get_connection()
            .execute("SELECT COUNT(*) FROM console_trace_calls")
            .fetchone()[0]
            == 0
        )
        assert (
            h.db.get_connection()
            .execute("SELECT COUNT(*) FROM message_exchanges")
            .fetchone()[0]
            == 0
        )
        assert (
            h.db.get_connection()
            .execute("SELECT COUNT(*) FROM console_prompt_mixin_attempts")
            .fetchone()[0]
            == 6
        )
    finally:
        for session in restored.store.sessions():
            restored.store.close_session(session.id)
        restored.store.end_app_runtime()


@pytest.mark.asyncio
async def test_selected_generation_never_keeps_another_receipts_count(local_journey):
    h = local_journey
    result = await h.controller.submit_draft("input", session_id="session-1")
    original = h.store.get_message(result.assistant_message_id)
    assert original.mixin_usage_count == 1
    original_attempt_id = original.metadata.mixin_attempt_id
    manual = h.store.add_variant(original.id, "manual replacement")
    assert manual.mixin_usage_count is None
    selected = h.store.select_variant(original.id, 0)
    assert selected.metadata.mixin_attempt_id == original_attempt_id
    assert selected.mixin_usage_count == 1
    assert h.store.select_variant(original.id, 1).mixin_usage_count is None


@pytest.mark.asyncio
async def test_local_resend_continue_and_missing_original(local_journey, monkeypatch):
    from tldw_chatbook.Chat.console_turn_resend import resend_turn

    h = local_journey
    import asyncio
    from threading import Event

    from tldw_chatbook.Chat.console_prepared_request import thaw_json

    entered, release = Event(), Event()
    original_adapter = h.gateway._chat_api_call_fn

    def held(**kwargs):
        h.entries.append(thaw_json(kwargs["messages_payload"]))
        entered.set()
        assert release.wait(5)
        return {"choices": [{"message": {"content": "unused after stop"}}]}

    with monkeypatch.context() as patch:
        patch.setattr(h.gateway, "_chat_api_call_fn", held)
        sending = asyncio.create_task(
            h.controller.submit_draft("input", session_id="session-1")
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert h.controller.stop_active_run()
            release.set()
            first = await asyncio.wait_for(sending, 5)
        finally:
            release.set()
    assert h.gateway._chat_api_call_fn == original_adapter
    reply = h.store.get_message(first.assistant_message_id)
    assert reply.status == "stopped" and not reply.content
    original = receipt(h, reply.id, "A literal {{scene}}", 1)
    await edit_source(h)
    toggle(h, False)
    h.adapter_error = False
    resent = await resend_turn(h.controller, first.user_message_id)
    assert resent.accepted and len(h.entries) == 2
    reply = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role.value == "assistant"
    )
    assert receipt(h, reply.id, "A literal {{scene}}", 1) != original
    toggle(h, True)
    continued = await h.controller.continue_from_message(reply.id)
    assert continued.accepted and len(h.entries) == 3
    latest = next(
        row
        for row in reversed(h.store.messages_for_session("session-1"))
        if row.role.value == "assistant"
    )
    receipt(
        h,
        latest.id,
        "B literal {{scene}}",
        2,
        current_input="Continue and extend the selected message.",
    )
    h.adapter_error = True
    failed = await h.controller.submit_draft("input", session_id="session-1")
    failed_reply = h.store.get_message(failed.assistant_message_id)
    with h.db.transaction() as cursor:
        cursor.execute(
            "DELETE FROM console_prompt_mixin_attempts WHERE attempt_id=?",
            (failed_reply.metadata.mixin_attempt_id,),
        )
    blocked = await h.controller.retry_message(failed_reply.id)
    assert not blocked.accepted and "unavailable" in blocked.visible_copy
    assert len(h.entries) == 4


@pytest.mark.asyncio
async def test_local_snapshot_failure_never_enters_adapter(local_journey, monkeypatch):
    from tldw_chatbook.DB import console_prompt_mixins

    h = local_journey

    def fail(*args, **kwargs):
        raise RuntimeError("QA snapshot unavailable")

    monkeypatch.setattr(console_prompt_mixins, "write_attempt", fail)
    h.store.set_session_draft("session-1", "keep my draft")
    result = await h.controller.submit_draft("keep my draft", session_id="session-1")
    assert not result.provider_started and h.entries == []
    assert h.store.session_draft("session-1") == "keep my draft"
    assert (
        h.db.get_connection()
        .execute("SELECT COUNT(*) FROM console_prompt_mixin_attempts")
        .fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
async def test_local_postdispatch_settlement_keeps_uncertain_recovery(
    local_journey, monkeypatch
):
    h = local_journey

    def fail(*args, **kwargs):
        raise RuntimeError("QA settlement unavailable")

    monkeypatch.setattr(
        h.store.persistence.console_dispatch_repository, "settle_with_assistant", fail
    )
    result = await h.controller.submit_draft("input", session_id="session-1")
    assert len(h.entries) == 1
    receipt(h, result.assistant_message_id, "A literal {{scene}}", 1)
    recovery = h.store.dispatch_recovery_for_session("session-1")
    assert recovery is not None
    again = await h.controller.submit_draft("next", session_id="session-1")
    assert not again.accepted and len(h.entries) == 1


@pytest.mark.asyncio
async def test_local_body_bearing_tool_successor_reuses_one_attempt(
    local_journey, tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_prepared_request import thaw_json
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    h = local_journey
    runs = AgentRunsDB(tmp_path / "agent.sqlite", client_id="journey")
    try:
        h.controller._agent_bridge = ConsoleAgentBridge(
            agent_runs_db=runs,
            store=h.store,
            provider_gateway=h.gateway,
            native_tools_enabled=lambda: False,
        )
        h.controller._agent_runtime_enabled = True

        def adapter(**kwargs):
            h.entries.append(thaw_json(kwargs["messages_payload"]))
            text = (
                '```tool_call\n{"name":"calculator","arguments":{"expression":"6*7"}}\n```'
                if len(h.entries) == 1
                else "42"
            )
            return {"choices": [{"message": {"content": text}}]}

        monkeypatch.setattr(h.gateway, "_chat_api_call_fn", adapter)
        result = await h.controller.submit_draft("input", session_id="session-1")
        assert result.accepted and len(h.entries) == 2
        assert any("42" in str(row["content"]) for row in h.entries[1])
        for rows in h.entries:
            assert (
                sum(
                    str(row.get("content", "")).count("A literal {{scene}}")
                    for row in rows
                )
                == 1
            )
        assert h.store.get_message(result.user_message_id).content == "input"
        conversation = h.store._sessions["session-1"].persisted_conversation_id
        with h.db.transaction() as cursor:
            ids = cursor.execute(
                "SELECT attempt_id,user_message_id,assistant_message_id FROM console_prompt_mixin_attempts"
            ).fetchall()
            assert len(ids) == 1
            assert (
                ids[0][1]
                == h.store.get_message(result.user_message_id).persisted_message_id
            )
            assert (
                read_attempt(cursor, conversation, attempt_id=ids[0][0])
                .snapshot.items[0]
                .definition.body
                == "A literal {{scene}}"
            )
    finally:
        runs.close()
