"""Trajectory sidecar capture (schema v38, task 2).

Covers the Console persistence seam: every persisted Console message gets a
``user``/``assistant`` sidecar row, every TOOL marker gets ``tool_call`` +
``tool_result`` rows keyed to the parent assistant message (TOOL-marker
invariant: markers themselves are never persisted to ``messages``), and
streamed assistant rows carry step-start/first-token/completion timing.
"""

import json
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, TrajectoryRowWrite
from Tests.console_provider_doubles import provider_resolution


def _store_with_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    return db, store


class _TraceGateway:
    def __init__(self, outcome: str = "success") -> None:
        self.outcome = outcome

    async def resolve_for_send(self, _selection):
        return provider_resolution(
                   ready=True,
                   provider="llama_cpp",
                   model="test-model",
                   base_url="http://127.0.0.1:9099",
                   visible_copy="",
               )

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        if self.outcome == "error":
            raise RuntimeError("provider secret must not persist")
        if self.outcome == "partial_error":
            yield "partial"
            raise RuntimeError("provider secret must not persist")
        if self.outcome == "success":
            yield "done"


class _CancelGateway(_TraceGateway):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        self.started.set()
        yield "partial"
        await asyncio.Event().wait()


def test_persisted_user_message_produces_user_trajectory_row(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        user = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="hello",
            persist=True,
        )

        rows = db.get_trajectory_rows(conversation_id)
        assert [row.event_kind for row in rows] == ["user"]
        row = rows[0]
        assert row.message_id == user.persisted_message_id
        assert row.turn_id == store.get_message(user.id).turn_id
        assert row.payload_json is None
    finally:
        db.close()


def test_tool_marker_append_produces_tool_call_and_result_rows(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="list the files",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="working on it",
            persist=True,
        )

        marker = store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_list → (3 files)",
            tool_output_full="file-a\nfile-b\nfile-c",
        )
        assert marker.role is ConsoleMessageRole.TOOL
        # TOOL-marker invariant: the marker itself is never a persisted row.
        assert marker.persisted_message_id is None

        rows = db.get_trajectory_rows(conversation_id)
        tool_rows = [row for row in rows if row.event_kind.startswith("tool_")]
        assert sorted(row.event_kind for row in tool_rows) == [
            "tool_call",
            "tool_result",
        ]
        for row in tool_rows:
            assert row.message_id == assistant.persisted_message_id
            payload = json.loads(row.payload_json)
            assert payload["name"] == "fs_list"
            assert payload["result"] == ""
            assert payload["field_states"]["result"] == "omitted"
            assert payload.get("truncated") is not True
    finally:
        db.close()


def test_tool_marker_before_assistant_persist_flushes_on_assistant_persist(tmp_path):
    """A marker appended while the assistant row is still streaming is buffered
    and flushed (remapped to the persisted id) when the assistant persists."""
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="go",
            persist=True,
        )
        # Streaming assistant placeholder: NOT yet persisted.
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        assert assistant.persisted_message_id is None
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_read → preview",
            tool_output_full="full file contents",
        )
        # Nothing writable yet: the marker waits for the parent's durable id.
        assert db.get_trajectory_rows(conversation_id) == [] or all(
            row.event_kind == "user" for row in db.get_trajectory_rows(conversation_id)
        )

        store.append_stream_chunk(assistant.id, "done")
        completed = store.mark_message_complete(assistant.id)
        assert completed.persisted_message_id is not None

        tool_rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind.startswith("tool_")
        ]
        assert sorted(row.event_kind for row in tool_rows) == [
            "tool_call",
            "tool_result",
        ]
        for row in tool_rows:
            assert row.message_id == completed.persisted_message_id
    finally:
        db.close()


def test_streamed_assistant_row_carries_timing(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="hi",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )

        step_started = time.time()
        store.record_trajectory_timing(
            assistant.id,
            step_started_at=step_started,
            model="test-model",
            provider="test-provider",
        )
        time.sleep(0.01)
        store.append_stream_chunk(assistant.id, "first")
        time.sleep(0.01)
        store.record_trajectory_timing(assistant.id, completed_at=time.time())

        completed = store.mark_message_complete(assistant.id)
        rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind == "assistant"
        ]
        assert len(rows) == 1
        row = rows[0]
        assert row.message_id == completed.persisted_message_id
        assert row.model == "test-model"
        assert row.provider == "test-provider"
        assert row.step_started_at == pytest.approx(step_started, abs=1.0)
        assert row.first_token_at is not None
        assert row.first_token_at - row.step_started_at > 0
        assert row.completed_at >= row.first_token_at
        assert json.loads(row.payload_json) == {
            "model_status": "completed",
            "trace_version": 2,
        }
        snapshot = derive_trajectory(
            db.get_messages_for_conversation(conversation_id),
            {},
            rows,
            [],
            [],
            active_leaf_message_id=db.get_conversation_active_leaf(conversation_id),
        )
        records = [record for turn in snapshot.turns for record in turn.records]
        assert [record.kind for record in records] == [
            "user",
            "model_request_started",
            "model_first_token",
            "model_response_completed",
            "assistant",
        ]
        assert records[-1].parent_event_id == (
            f"model-timing:{completed.persisted_message_id}:completed"
        )
        assert records[-1].observed_at == pytest.approx(row.completed_at)
    finally:
        db.close()


def test_pending_trace_event_flushes_with_new_assistant_owner(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="go",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        assert assistant.persisted_message_id is None

        assert store.record_trace_event(
            session.id,
            anchor_message_id=assistant.id,
            event_kind="model_error",
            summary="Provider request failed",
            status="failed",
        )
        store.record_trajectory_timing(assistant.id, model_status="failed")
        store.mark_message_failed(assistant.id)

        rows = db.get_trajectory_rows(conversation_id)
        assert [row.event_kind for row in rows] == ["user", "model_error"]
        assert rows[-1].message_id == rows[0].message_id
        assert json.loads(rows[-1].payload_json)["field_states"] == {
            "payload": "omitted"
        }
    finally:
        db.close()


@pytest.mark.asyncio
async def test_real_submit_captures_retrieval_and_context_at_owner_seams(tmp_path):
    db, store = _store_with_db(tmp_path)
    secret_context = "credential=sk-never-persist-this"

    async def capture(_draft, _turn_context, **_kwargs):
        return SimpleNamespace(
            context=secret_context,
            citation_builder=None,
            prompt_evidence_set_id="prompt-set-1",
            citation_repair_contract=None,
        )

    try:
        session = store.ensure_session(title="Trace")
        controller = ConsoleChatController(
            store=store,
            provider_gateway=_TraceGateway(),
            rag_capture_provider=capture,
        )
        result = await controller.submit_draft("question")
        assert result.accepted

        conversation_id = next(
            item.persisted_conversation_id
            for item in store.sessions()
            if item.id == session.id
        )
        rows = db.get_trajectory_rows(conversation_id)
        kinds = [row.event_kind for row in rows]
        for kind in (
            "retrieval_started",
            "retrieval_candidates_selected",
            "retrieval_completed",
            "context_attached",
            "context_injected",
        ):
            assert kind in kinds
        assert secret_context not in repr(rows)
        context_rows = [row for row in rows if row.event_kind.startswith("context_")]
        assert all(
            json.loads(row.payload_json)["sensitivity"] == "system_context"
            for row in context_rows
        )
        snapshot = derive_trajectory(
            db.get_messages_for_conversation(conversation_id),
            {},
            rows,
            [],
            [],
        )
        records = [record for turn in snapshot.turns for record in turn.records]
        by_kind = {record.kind: record for record in records}
        chain = (
            "retrieval_started",
            "retrieval_candidates_selected",
            "retrieval_completed",
            "context_attached",
            "context_injected",
        )
        for parent_kind, child_kind in zip(chain, chain[1:]):
            assert by_kind[child_kind].parent_event_id == by_kind[parent_kind].event_id
            assert by_kind[child_kind].source_event_id == by_kind[parent_kind].event_id
        emitted_ids = {record.event_id for record in records}
        assert all(by_kind[kind].event_id in emitted_ids for kind in chain)
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("empty", "error"))
async def test_direct_provider_failure_is_truthful_and_has_no_completion(
    tmp_path, outcome: str
):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trace")
        controller = ConsoleChatController(
            store=store,
            provider_gateway=_TraceGateway(outcome),
        )
        result = await controller.submit_draft("question")
        assert result.accepted

        conversation_id = next(
            item.persisted_conversation_id
            for item in store.sessions()
            if item.id == session.id
        )
        rows = db.get_trajectory_rows(conversation_id)
        kinds = [row.event_kind for row in rows]
        assert "model_error" in kinds
        assert "model_response_completed" not in kinds
        assert "provider secret must not persist" not in repr(rows)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_direct_provider_cancel_emits_cancel_without_completion(tmp_path):
    db, store = _store_with_db(tmp_path)
    gateway = _CancelGateway()
    try:
        session = store.ensure_session(title="Trace")
        controller = ConsoleChatController(store=store, provider_gateway=gateway)
        task = asyncio.create_task(controller.submit_draft("question"))
        await gateway.started.wait()
        await asyncio.sleep(0)
        assert controller.stop_active_run(record_user_stop=False)
        result = await task
        assert result.accepted

        conversation_id = next(
            item.persisted_conversation_id
            for item in store.sessions()
            if item.id == session.id
        )
        rows = db.get_trajectory_rows(conversation_id)
        kinds = [row.event_kind for row in rows]
        assert "model_cancelled" in kinds
        assert "model_response_completed" not in kinds
    finally:
        db.close()


@pytest.mark.asyncio
async def test_user_retry_action_is_not_mislabeled_as_provider_retry(tmp_path):
    db, store = _store_with_db(tmp_path)
    gateway = _TraceGateway("empty")
    try:
        session = store.ensure_session(title="Trace")
        controller = ConsoleChatController(store=store, provider_gateway=gateway)
        failed = await controller.submit_draft("question")
        assert failed.assistant_message_id is not None

        gateway.outcome = "success"
        retried = await controller.retry_message(failed.assistant_message_id)
        assert retried.accepted

        conversation_id = next(
            item.persisted_conversation_id
            for item in store.sessions()
            if item.id == session.id
        )
        kinds = [row.event_kind for row in db.get_trajectory_rows(conversation_id)]
        assert "message_retry_requested" in kinds
        assert "model_retry" not in kinds
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_fails", (False, True))
async def test_real_llamacpp_fallback_retry_reaches_console_owner(
    tmp_path, monkeypatch, fallback_fails: bool
) -> None:
    class _EmptyStreamResponse:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            return
            yield  # pragma: no cover

    class _StreamCtx:
        async def __aenter__(self):
            return _EmptyStreamResponse()

        async def __aexit__(self, *_exc):
            return False

    class _FakeClient:
        def stream(self, *_args, **_kwargs):
            return _StreamCtx()

    db, store = _store_with_db(tmp_path)
    gateway = ConsoleProviderGateway()
    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        base_url="http://127.0.0.1:9099",
        model="model",
        ready=True,
        execution_key="llama_cpp",
        streaming=True,
    )

    async def resolve_for_send(_selection):
        return resolution

    gateway.resolve_for_send = resolve_for_send
    monkeypatch.setattr(
        ConsoleProviderGateway, "_active_http_client", lambda self: _FakeClient()
    )

    async def fake_complete(self, **_kwargs):
        if fallback_fails:
            raise RuntimeError("fallback failed")
        return "recovered"

    monkeypatch.setattr(ConsoleProviderGateway, "complete_llamacpp_chat", fake_complete)
    try:
        session = store.ensure_session(title="Trace")
        controller = ConsoleChatController(store=store, provider_gateway=gateway)
        result = await controller.submit_draft("question")
        assert result.accepted
        conversation_id = next(
            item.persisted_conversation_id
            for item in store.sessions()
            if item.id == session.id
        )
        rows = db.get_trajectory_rows(conversation_id)
        kinds = [row.event_kind for row in rows]
        assert kinds.count("model_retry") == 1
        terminal_kind = "model_error" if fallback_fails else "assistant"
        assert kinds.index("model_retry") < kinds.index(terminal_kind)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_regenerate_replacement_identity_resolves_after_persistence(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trace")
        store.persist_session_if_needed(session.id)
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="question", persist=True
        )
        original = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
            persist=True,
        )
        controller = ConsoleChatController(
            store=store,
            provider_gateway=_TraceGateway(),
        )

        result = await controller.regenerate_message(original.id)
        assert result.accepted

        conversation_id = next(
            item.persisted_conversation_id
            for item in store.sessions()
            if item.id == session.id
        )
        rows = db.get_trajectory_rows(conversation_id)
        row = next(row for row in rows if row.event_kind == "message_regenerated")
        payload = json.loads(row.payload_json)
        assert payload["replacement_event_id"].startswith("message:")
        assert payload["field_states"]["replacement_event_id"] == "observed"
        active_leaf_message_id = db.get_conversation_active_leaf(conversation_id)
        assert active_leaf_message_id is not None
        snapshot = derive_trajectory(
            db.get_messages_for_conversation(conversation_id),
            {},
            rows,
            [],
            [],
            active_leaf_message_id=active_leaf_message_id,
        )
        records = [record for turn in snapshot.turns for record in turn.records]
        emitted_ids = {record.event_id for record in records}
        regenerated = next(record for record in records if record.kind == "message_regenerated")
        assert regenerated.status == "completed"
        assert regenerated.replacement_event_id in emitted_ids
        assert f"message:{original.persisted_message_id}" not in emitted_ids
        ordered_ids = [record.event_id for record in records]
        assert ordered_ids.index(regenerated.event_id) < ordered_ids.index(
            regenerated.replacement_event_id
        )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_regenerate_partial_error_records_failed_replacement(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trace")
        store.persist_session_if_needed(session.id)
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="question", persist=True
        )
        original = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="answer", persist=True
        )
        controller = ConsoleChatController(
            store=store, provider_gateway=_TraceGateway("partial_error")
        )

        result = await controller.regenerate_message(original.id)
        assert result.accepted
        siblings, _index, _count = store.siblings_at(original.id)
        sibling = next(item for item in siblings if item.id != original.id)
        assert sibling.status == "failed"
        conversation_id = next(
            item.persisted_conversation_id
            for item in store.sessions()
            if item.id == session.id
        )
        row = next(
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind == "message_regenerated"
        )
        assert json.loads(row.payload_json)["status"] == "failed"
        snapshot = derive_trajectory(
            db.get_messages_for_conversation(conversation_id),
            {},
            db.get_trajectory_rows(conversation_id),
            [],
            [],
            active_leaf_message_id=db.get_conversation_active_leaf(conversation_id),
        )
        regenerated = next(
            record
            for turn in snapshot.turns
            for record in turn.records
            if record.kind == "message_regenerated"
        )
        assert regenerated.status == "failed"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_regenerate_cancel_records_stopped_replacement(tmp_path):
    db, store = _store_with_db(tmp_path)
    gateway = _CancelGateway()
    try:
        session = store.ensure_session(title="Trace")
        store.persist_session_if_needed(session.id)
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="question", persist=True
        )
        original = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="answer", persist=True
        )
        controller = ConsoleChatController(store=store, provider_gateway=gateway)

        task = asyncio.create_task(controller.regenerate_message(original.id))
        await asyncio.wait_for(gateway.started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert controller.stop_active_run() is True
        result = await asyncio.wait_for(task, timeout=1)
        assert result.accepted
        siblings, _index, _count = store.siblings_at(original.id)
        sibling = next(item for item in siblings if item.id != original.id)
        assert sibling.status == "stopped"
        conversation_id = next(
            item.persisted_conversation_id
            for item in store.sessions()
            if item.id == session.id
        )
        row = next(
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind == "message_regenerated"
        )
        assert json.loads(row.payload_json)["status"] == "stopped"
        snapshot = derive_trajectory(
            db.get_messages_for_conversation(conversation_id),
            {},
            db.get_trajectory_rows(conversation_id),
            [],
            [],
            active_leaf_message_id=db.get_conversation_active_leaf(conversation_id),
        )
        regenerated = next(
            record
            for turn in snapshot.turns
            for record in turn.records
            if record.kind == "message_regenerated"
        )
        assert regenerated.status == "stopped"
    finally:
        db.close()


def test_tool_result_uses_bounded_safe_summary_with_truncated_marker(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="go",
            persist=True,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
            persist=True,
        )
        huge = "safe " * (70 * 1024)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ generic_lookup → preview",
            tool_output_full=huge,
        )

        tool_rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind.startswith("tool_")
        ]
        assert len(tool_rows) == 2
        for row in tool_rows:
            payload = json.loads(row.payload_json)
            assert payload["truncated"] is True
            assert len(payload["result"]) <= 2100
            assert payload["result"] != huge
    finally:
        db.close()


def test_tool_result_cap_is_byte_safe_for_multibyte_content(tmp_path):
    """The cap is BYTES, not characters: 4-byte emoji content truncated by a
    character slice could leave the stored result up to 4x over budget."""
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="go",
            persist=True,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
            persist=True,
        )
        # U+1F600 encodes to 4 UTF-8 bytes per character: ~100k chars is
        # ~400 KiB, well over the 256 KiB byte cap.
        huge = "😀 " * (100 * 1024)
        assert len(huge) < 256 * 1024  # characters under, bytes over
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ generic_lookup → preview",
            tool_output_full=huge,
        )

        tool_rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind.startswith("tool_")
        ]
        assert len(tool_rows) == 2
        for row in tool_rows:
            payload = json.loads(row.payload_json)
            assert payload["truncated"] is True
            stored = payload["result"]
            assert len(stored) <= 2100
            # The split codepoint at the byte boundary was dropped cleanly.
            assert "�" not in stored
            assert stored != huge
    finally:
        db.close()


def test_tool_marker_scrubs_credentials_and_omits_file_content_durably(tmp_path):
    db, store = _store_with_db(tmp_path)
    secrets = (
        "ghp_" + "a" * 36,
        "AKIA" + "A" * 16,
        "eyJabcdefghij.abcdefghij.abcdefghij",
        "-----BEGIN PRIVATE KEY-----",
    )
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="go", persist=True
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
            persist=True,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ generic_lookup → preview",
            tool_output_full=" ".join(secrets),
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_read → preview",
            tool_output_full="/Users/alice/private.txt\nprivate file body",
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ generic_lookup → 3 matches",
            tool_output_full="chain of thought: private internal plan",
        )
        path_outputs = (
            "/private/var/db/secrets.txt",
            "~/private.txt",
            r"C:\Users\alice\secret.txt",
            r"\\server\share\secret.txt",
        )
        for path_output in path_outputs:
            store.append_message(
                session.id,
                role=ConsoleMessageRole.TOOL,
                content="⚙ generic_lookup → path",
                tool_output_full=path_output,
            )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ generic_lookup → 3 matches",
            tool_output_full="3 safe matches",
        )

        rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind == "tool_result"
        ]
        serialized = repr(rows)
        assert all(secret not in serialized for secret in secrets)
        assert "/Users/alice/private.txt" not in serialized
        assert "private file body" not in serialized
        assert "private internal plan" not in serialized
        assert all(path_output not in serialized for path_output in path_outputs)
        payloads = [json.loads(row.payload_json) for row in rows]
        generic_secret, file_payload, hidden_payload, *tail = payloads
        path_payloads, safe_payload = tail[:-1], tail[-1]
        assert generic_secret["field_states"]["result"] == "redacted"
        assert generic_secret["sensitivity"] == "tool_content"
        assert file_payload["result"] == ""
        assert file_payload["field_states"]["result"] == "omitted"
        assert hidden_payload["result"] == ""
        assert hidden_payload["field_states"]["result"] == "omitted"
        assert len(path_payloads) == len(path_outputs)
        assert all(payload["result"] == "" for payload in path_payloads)
        assert all(
            payload["field_states"]["result"] == "omitted" for payload in path_payloads
        )
        assert all(payload["sensitivity"] == "path" for payload in path_payloads)
        assert safe_payload["result"] == "3 safe matches"
        assert safe_payload["field_states"]["result"] == "observed"
    finally:
        db.close()


@pytest.mark.parametrize("diagnostic_fails", (False, True))
def test_sidecar_write_failure_attempts_one_nonrecursive_diagnostic(
    tmp_path, monkeypatch, diagnostic_fails: bool
) -> None:
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        user = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="go", persist=True
        )
        real_writer = store.persistence.write_trajectory_rows
        calls: list[list[str]] = []

        def selective_writer(rows):
            kinds = [row.event_kind for row in rows]
            calls.append(kinds)
            if kinds != ["capture_failed"] or diagnostic_fails:
                raise RuntimeError("secret diagnostic failure")
            return real_writer(rows)

        monkeypatch.setattr(store.persistence, "write_trajectory_rows", selective_writer)
        assert not store.record_trace_event(
            session.id,
            anchor_message_id=user.id,
            event_kind="model_error",
            summary="Provider request failed",
        )
        assert calls == [["model_error"], ["capture_failed"]]
        rows = db.get_trajectory_rows(conversation_id)
        diagnostics = [row for row in rows if row.event_kind == "capture_failed"]
        assert len(diagnostics) == (0 if diagnostic_fails else 1)
        if diagnostics:
            payload = json.loads(diagnostics[0].payload_json)
            assert payload["field_states"]["payload"] == "capture_failed"
            assert "secret diagnostic failure" not in repr(payload)

        # Retrying the same failed observation cannot create another diagnostic.
        store.record_trace_event(
            session.id,
            anchor_message_id=user.id,
            event_kind="model_error",
            summary="Provider request failed",
        )
        assert calls.count(["capture_failed"]) == 1
    finally:
        db.close()


def test_capture_failed_identity_deduplicates_after_restart_and_separates_sources(
    tmp_path, monkeypatch
) -> None:
    db_path = str(tmp_path / "chachanotes.sqlite")
    db = CharactersRAGDB(db_path, "test_client")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.ensure_session(title="Trajectory")
    conversation_id = store.persist_session_if_needed(session.id)
    user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="go", persist=True
    )
    assert user.persisted_message_id is not None

    def source_row(event_id: str) -> TrajectoryRowWrite:
        return TrajectoryRowWrite(
            message_id=user.persisted_message_id,
            conversation_id=conversation_id,
            turn_id=user.turn_id or user.persisted_message_id,
            seq=None,
            event_kind="model_error",
            payload_json=json.dumps(
                {
                    "event_id": event_id,
                    "summary": "Provider request failed",
                    "field_states": {"payload": "omitted"},
                }
            ),
        )

    first_source = source_row("source:first")
    second_source = source_row("source:second")

    def install_selective_writer(target_store):
        real_writer = target_store.persistence.write_trajectory_rows

        def selective_writer(rows):
            if [row.event_kind for row in rows] != ["capture_failed"]:
                raise RuntimeError("primary capture failed")
            return real_writer(rows)

        monkeypatch.setattr(
            target_store.persistence, "write_trajectory_rows", selective_writer
        )

    try:
        install_selective_writer(store)
        assert not store.write_trajectory_rows([first_source])
        first_diagnostic = next(
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind == "capture_failed"
        )
        first_event_id = json.loads(first_diagnostic.payload_json)["event_id"]
    finally:
        db.close()

    reopened = CharactersRAGDB(db_path, "test_client")
    try:
        reopened_store = ConsoleChatStore(persistence=ChatPersistenceService(reopened))
        install_selective_writer(reopened_store)
        assert not reopened_store.write_trajectory_rows([first_source])
        assert not reopened_store.write_trajectory_rows([second_source])

        diagnostics = [
            row
            for row in reopened.get_trajectory_rows(conversation_id)
            if row.event_kind == "capture_failed"
        ]
        assert len(diagnostics) == 2
        event_ids = [json.loads(row.payload_json)["event_id"] for row in diagnostics]
        assert event_ids.count(first_event_id) == 1
        assert len(set(event_ids)) == 2
        assert all(event_id.startswith("capture-failed:") for event_id in event_ids)
    finally:
        reopened.close()


def test_capture_failed_hydrates_durable_ids_once_for_many_failures() -> None:
    class FakeDB:
        def __init__(self) -> None:
            self.read_count = 0
            self.rows = [
                SimpleNamespace(event_kind="model", payload_json=None)
                for _ in range(5_000)
            ]

        def get_trajectory_rows(self, _conversation_id):
            self.read_count += 1
            return list(self.rows)

    db = FakeDB()

    def writer(rows):
        if [row.event_kind for row in rows] != ["capture_failed"]:
            return False
        db.rows.append(
            SimpleNamespace(
                event_kind="capture_failed", payload_json=rows[0].payload_json
            )
        )
        return True

    store = ConsoleChatStore(
        persistence=SimpleNamespace(db=db, write_trajectory_rows=writer)
    )
    for index in range(25):
        source = TrajectoryRowWrite(
            message_id="message-1",
            conversation_id="conversation-1",
            turn_id="turn-1",
            seq=None,
            event_kind="model_error",
            payload_json=json.dumps({"event_id": f"source:{index}"}),
        )
        assert not store.write_trajectory_rows([source])

    assert db.read_count == 1
    diagnostics = [row for row in db.rows if row.event_kind == "capture_failed"]
    assert len(diagnostics) == 25


def test_trajectory_write_failure_never_fails_the_turn(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        store.persist_session_if_needed(session.id)

        def exploding_writer(**kwargs):
            raise RuntimeError("sidecar unavailable")

        store.persistence.write_trajectory_rows = exploding_writer
        # The turn itself must still persist and complete normally.
        user = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="hello",
            persist=True,
        )
        assert user.persisted_message_id is not None
        marker = store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_read → preview",
            tool_output_full="ok",
        )
        assert marker.persisted_message_id is None
    finally:
        db.close()


def test_concurrent_upserts_produce_unique_seqs(tmp_path):
    """Two threads writing trajectory rows for one conversation must produce
    unique, gap-free seqs. Exercises the Console's write seam
    (``ChatPersistenceService.write_trajectory_rows``), whose bounded retry
    absorbs the transient write-write lock contention of concurrent turns;
    every row lands exactly once with a distinct per-conversation seq."""
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        service = ChatPersistenceService(db)
        conversation_id = db.add_conversation(
            {"chat_id": 1, "conversation_id": "traj-concurrency", "fragmentation": 0}
        )
        assert conversation_id is not None

        message_ids = [
            db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": f"message {index}",
                }
            )
            for index in range(50)
        ]

        def write_batch(prefix: str, ids: list) -> None:
            for index in range(25):
                assert service.write_trajectory_rows(
                    [
                        TrajectoryRowWrite(
                            message_id=ids.pop(),
                            conversation_id=conversation_id,
                            turn_id=f"{prefix}-turn",
                            seq=None,
                            event_kind="assistant",
                        )
                    ]
                )

        batch_a = message_ids[:25]
        batch_b = message_ids[25:]

        threads = [
            threading.Thread(target=write_batch, args=("t0", batch_a)),
            threading.Thread(target=write_batch, args=("t1", batch_b)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        rows = db.get_trajectory_rows(conversation_id)
        assert len(rows) == 50
        seqs = [row.seq for row in rows]
        assert sorted(seqs) == list(range(1, 51))
        assert len(set(seqs)) == 50
    finally:
        db.close()


def test_concurrent_direct_db_upserts_produce_unique_seqs(tmp_path):
    """The DB-layer upsert itself must be safe under cross-thread concurrency
    (BEGIN IMMEDIATE write lock): no thread's batch may roll back with the
    deferred-upgrade "database is locked" deadlock, and seqs stay unique."""
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        conversation_id = db.add_conversation(
            {"chat_id": 1, "conversation_id": "traj-db-concurrency", "fragmentation": 0}
        )
        assert conversation_id is not None
        message_ids = [
            db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": f"message {index}",
                }
            )
            for index in range(50)
        ]

        def write_direct_batch(ids: list) -> None:
            for message_id in ids:
                db.upsert_trajectory_rows(
                    [
                        TrajectoryRowWrite(
                            message_id=message_id,
                            conversation_id=conversation_id,
                            turn_id="db-turn",
                            seq=None,
                            event_kind="assistant",
                        )
                    ]
                )

        threads = [
            threading.Thread(target=write_direct_batch, args=(message_ids[:25],)),
            threading.Thread(target=write_direct_batch, args=(message_ids[25:],)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        rows = db.get_trajectory_rows(conversation_id)
        assert len(rows) == 50
        seqs = [row.seq for row in rows]
        assert sorted(seqs) == list(range(1, 51))
        assert len(set(seqs)) == 50
    finally:
        db.close()


# --- selection feedback events (task-17169, phase 4) ---------------------------
#
# Console selection feedback (Request changes / LGTM / Comment) was ephemeral:
# composed into the next user message and forgotten. Decision AC#4 was Option A
# -- the ADR-066 trajectory sidecar, because feedback is a chronological run
# event and the sidecar is local-only (a synced annotations table would drag in
# sync-schema implications for what is really an audit record).


def test_selection_feedback_persists_as_a_user_feedback_row(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Feedback")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="do it", persist=True
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="here is the patch",
            persist=True,
        )

        assert store.record_feedback_event(
            session.id,
            anchor_message_id=assistant.id,
            action="request-changes",
            quote="here is the patch",
            comment="use a context manager",
        )

        rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind == "user_feedback"
        ]
        assert len(rows) == 1
        row = rows[0]
        assert row.message_id == assistant.persisted_message_id
        assert row.turn_id == store.get_message(assistant.id).turn_id
        payload = json.loads(row.payload_json)
        assert payload["action"] == "request-changes"
        assert payload["quote"] == "here is the patch"
        assert payload["comment"] == "use a context manager"
    finally:
        db.close()


def test_feedback_without_a_comment_records_no_comment_key(tmp_path):
    """LGTM and Request-changes carry no comment; the payload must not
    fabricate an empty one for the viewer to render."""
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Feedback")
        conversation_id = store.persist_session_if_needed(session.id)
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="ok", persist=True
        )

        store.record_feedback_event(
            session.id,
            anchor_message_id=assistant.id,
            action="lgm",
            quote="ok",
            comment=None,
        )

        rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind == "user_feedback"
        ]
        assert json.loads(rows[0].payload_json) == {"action": "lgm", "quote": "ok"}
    finally:
        db.close()


def test_feedback_on_an_unpersisted_session_is_skipped_not_raised(tmp_path):
    """Ephemeral sessions have nothing to survive a restart -- the write is a
    silent no-op, and it must never take down the dispatch that triggered it."""
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Ephemeral")
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="ok", persist=False
        )

        assert (
            store.record_feedback_event(
                session.id,
                anchor_message_id=assistant.id,
                action="lgm",
                quote="ok",
                comment=None,
            )
            is False
        )
    finally:
        db.close()


def test_feedback_for_an_unknown_anchor_never_raises(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Feedback")
        store.persist_session_if_needed(session.id)

        assert (
            store.record_feedback_event(
                session.id,
                anchor_message_id="no-such-message",
                action="comment",
                quote="q",
                comment="c",
            )
            is False
        )
    finally:
        db.close()


def test_feedback_survives_a_restart(tmp_path):
    """AC#1: the whole point is durability. Reopen the DB from disk with a
    fresh store and the feedback is still there."""
    db_path = str(tmp_path / "chachanotes.sqlite")
    db = CharactersRAGDB(db_path, "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.ensure_session(title="Feedback")
        conversation_id = store.persist_session_if_needed(session.id)
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="ok", persist=True
        )
        store.record_feedback_event(
            session.id,
            anchor_message_id=assistant.id,
            action="comment",
            quote="ok",
            comment="revisit this",
        )
    finally:
        db.close()

    reopened = CharactersRAGDB(db_path, "test_client")
    try:
        rows = [
            row
            for row in reopened.get_trajectory_rows(conversation_id)
            if row.event_kind == "user_feedback"
        ]
        assert [json.loads(row.payload_json)["comment"] for row in rows] == [
            "revisit this"
        ]
    finally:
        reopened.close()


def test_message_edit_and_branch_selection_append_payload_free_trace_events(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Mutations")
        conversation_id = store.persist_session_if_needed(session.id)
        first = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="SECRET_ORIGINAL_CONTENT",
            persist=True,
        )
        second = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="second",
            persist=True,
        )

        store.update_message_content(second.id, "SECRET_EDITED_CONTENT")
        store.set_active_leaf(session.id, first.id)

        rows = db.get_trajectory_rows(conversation_id)
        by_kind = {
            row.event_kind: json.loads(row.payload_json)
            for row in rows
            if row.payload_json is not None
        }
        assert by_kind["message_edited"]["field_states"] == {"payload": "omitted"}
        assert by_kind["branch_selected"]["field_states"] == {"payload": "omitted"}
        assert "SECRET_ORIGINAL_CONTENT" not in repr(by_kind)
        assert "SECRET_EDITED_CONTENT" not in repr(by_kind)
    finally:
        db.close()


def test_trace_event_capture_failure_never_fails_the_user_mutation(
    tmp_path, monkeypatch
):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Capture failure")
        store.persist_session_if_needed(session.id)
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="before",
            persist=True,
        )
        monkeypatch.setattr(store, "write_trajectory_rows", lambda _rows: False)

        updated = store.update_message_content(message.id, "after")

        assert updated.content == "after"
    finally:
        db.close()
