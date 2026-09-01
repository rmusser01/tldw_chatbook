from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class _MessageLineageRepository(ConsoleTraceRepository):
    def __init__(self) -> None:
        super().__init__()
        self.requested_message_ids: list[str] = []

    def read_conversation_call_lineage(self, cursor, conversation_id):
        del cursor, conversation_id
        raise AssertionError("native reads must not load the whole conversation")

    def iter_message_call_lineage(self, cursor, conversation_id, message_id):
        self.requested_message_ids.append(message_id)
        yield from super().iter_message_call_lineage(
            cursor,
            conversation_id,
            message_id,
        )


@pytest.fixture
def database(tmp_path):
    db = CharactersRAGDB(tmp_path / "native-trace-reader.sqlite", "native-reader")
    yield db
    db.close()


def _message(
    database: CharactersRAGDB,
    conversation_id: str,
    *,
    sender: str,
    content: str,
) -> tuple[str, str]:
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": sender,
            "content": content,
        }
    )
    assert message_id is not None
    with database.transaction() as cursor:
        row = cursor.execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id,),
        ).fetchone()
    assert row is not None
    return message_id, str(row[0])


def test_native_reader_reconstructs_production_route_call_for_assistant_message(
    database: CharactersRAGDB,
) -> None:
    repository = _MessageLineageRepository()
    conversation_id = database.add_conversation({"title": "native reader"})
    assert conversation_id is not None
    user_id, user_revision = _message(
        database,
        conversation_id,
        sender="user",
        content="question",
    )
    assistant_id, assistant_revision = _message(
        database,
        conversation_id,
        sender="assistant",
        content="answer",
    )
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    occurred_at = "2026-09-01T00:00:00Z"

    with database.transaction(immediate=True) as cursor:
        repository.ensure_policy(cursor, policy)
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
        surface = repository.append_surface_node(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="message",
            reference=SemanticRevisionRef(user_revision),
        )
        header = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="fresh",
            endpoint_identity="https://api.openai.com/v1",
            generation_parameters={"streaming": False},
            adapter_defaults={},
            response_format={},
            reasoning_controls={},
            components=(),
        )
        call = repository.reserve_call(
            cursor,
            owner_id=owner.owner_id,
            segment_id=segment.segment_id,
            turn_id=user_id,
            run_id="run-native",
            call_sequence=0,
            idempotency_key="native-reader-call",
            policy_id=policy.policy_id,
        )
        repository.bind_call(
            cursor,
            call_id=call.call_id,
            surface_node_id=surface.node_id,
            request_header_id=header.header_id,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="fresh",
        )
        repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.DISPATCH_STARTED,
            occurred_at=occurred_at,
            integrity_state="complete",
        )
        repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.RESPONSE_STARTED,
            occurred_at=occurred_at,
            integrity_state="complete",
        )
        repository.store_response_link(
            cursor,
            call_id=call.call_id,
            response=SemanticRevisionRef(assistant_revision),
        )
        repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.COMPLETE,
            occurred_at=occurred_at,
            usage={"provider": "openai", "model": "gpt-test", "output": 1},
            integrity_state="complete",
        )
        repository.append_event(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            event_type="call_boundary",
            call_id=call.call_id,
        )

    calls = ConsoleTraceNativeReader(database, repository=repository).read_calls(
        assistant_id
    )

    assert len(calls) == 1
    native = calls[0]
    assert native.call_id == call.call_id
    assert native.verification_status == "verified"
    assert native.capture.run_tag == "run-native"
    assert native.capture.seq == 0
    assert native.capture.provider == "openai"
    assert native.capture.model == "gpt-test"
    assert native.capture.endpoint == "https://api.openai.com/v1"
    assert native.capture.status == "complete"
    assert native.capture.request["messages_payload"] == [
        {"role": "user", "content": "question"}
    ]
    assert native.capture.response == {"role": "assistant", "content": "answer"}
    assert repository.requested_message_ids == [assistant_id]


def test_native_reader_ignores_legacy_snapshot_routes(
    database: CharactersRAGDB,
) -> None:
    assert ConsoleTraceNativeReader(database).read_calls("missing-message") == ()
