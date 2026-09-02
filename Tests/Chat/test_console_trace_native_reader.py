from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    TraceCallState,
    TraceContentRef,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class _MessageLineageRepository(ConsoleTraceRepository):
    def __init__(self) -> None:
        super().__init__()
        self.requested_message_ids: list[str] = []
        self.requested_turn_ids: list[str | None] = []

    def read_conversation_call_lineage(self, cursor, conversation_id):
        del cursor, conversation_id
        raise AssertionError("native reads must not load the whole conversation")

    def iter_message_call_lineage(
        self,
        cursor,
        conversation_id,
        message_id,
        *,
        turn_id=None,
    ):
        self.requested_message_ids.append(message_id)
        self.requested_turn_ids.append(turn_id)
        yield from super().iter_message_call_lineage(
            cursor,
            conversation_id,
            message_id,
            turn_id=turn_id,
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
    parent_message_id: str | None = None,
) -> tuple[str, str]:
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": sender,
            "content": content,
            "parent_message_id": parent_message_id,
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


def test_native_reader_reconstructs_revision_and_artifact_calls_for_assistant(
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
        parent_message_id=user_id,
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

        artifact_call = repository.reserve_call(
            cursor,
            owner_id=owner.owner_id,
            segment_id=segment.segment_id,
            turn_id=user_id,
            run_id="run-artifact",
            call_sequence=1,
            idempotency_key="native-reader-artifact-call",
            policy_id=policy.policy_id,
        )
        repository.bind_call(
            cursor,
            call_id=artifact_call.call_id,
            surface_node_id=surface.node_id,
            request_header_id=header.header_id,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="fresh",
        )
        repository.advance_call_state(
            cursor,
            call_id=artifact_call.call_id,
            target=TraceCallState.DISPATCH_STARTED,
            occurred_at=occurred_at,
            integrity_state="complete",
        )
        repository.advance_call_state(
            cursor,
            call_id=artifact_call.call_id,
            target=TraceCallState.RESPONSE_STARTED,
            occurred_at=occurred_at,
            integrity_state="complete",
        )
        artifact = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=b'{"content":"transformed","role":"assistant"}',
            media_type="application/json",
            normalization_version="provider-response-v1",
        )
        repository.store_response_link(
            cursor,
            call_id=artifact_call.call_id,
            response=TraceContentRef(artifact.artifact_id, "provider_response"),
        )
        repository.advance_call_state(
            cursor,
            call_id=artifact_call.call_id,
            target=TraceCallState.COMPLETE,
            occurred_at=occurred_at,
            usage={"provider": "openai", "model": "gpt-test", "output": 1},
            integrity_state="complete",
        )
        repository.append_event(
            cursor,
            segment_id=segment.segment_id,
            sequence=1,
            event_type="call_boundary",
            call_id=artifact_call.call_id,
        )

    calls = ConsoleTraceNativeReader(database, repository=repository).read_calls(
        assistant_id
    )

    assert len(calls) == 2
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
    assert calls[1].call_id == artifact_call.call_id
    assert calls[1].capture.response == {
        "content": "transformed",
        "role": "assistant",
    }
    assert repository.requested_message_ids == [assistant_id]
    assert repository.requested_turn_ids == [user_id]


def test_native_reader_ignores_legacy_snapshot_routes(
    database: CharactersRAGDB,
) -> None:
    assert ConsoleTraceNativeReader(database).read_calls("missing-message") == ()


def test_native_continuation_uses_durable_policy_binding_without_regex_registry(
    database: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    conversation_id = database.add_conversation({"title": "durable continuation"})
    assert conversation_id is not None
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": "kimi-k3",
            "api_base_url": "https://api.moonshot.ai/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "answer",
                    "reasoning_blocks": ["customer-ABCDWXYZ"],
                    "calls": [],
                }
            ],
        }
    )
    encoded = dump_provider_continuation_json(checkpoint)
    assert encoded is not None
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "answer",
            "provider_continuation_json": encoded,
        }
    )
    assert message_id is not None
    policy = FrozenTracePolicy(
        new_opaque_id(),
        "credentials-v1",
        True,
        "44444444-4444-4444-8444-444444444444",
    )

    with database.transaction(immediate=True) as cursor:
        revision_row = cursor.execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ?""",
            (message_id,),
        ).fetchone()
        assert revision_row is not None
        revision_id = str(revision_row[0])
        repository.ensure_policy(cursor, policy)
        masked = json.loads(encoded)
        masked["rounds"][0]["reasoning_blocks"] = ["[PII omitted]"]
        artifact = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=json.dumps(
                {
                    "content": "answer",
                    "provider_continuation": masked,
                    "role": "assistant",
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            media_type="application/vnd.tldw.semantic-message+json",
            normalization_version="semantic-envelope-v1",
        )
        repository.bind_revision_policy(
            cursor,
            revision_id=revision_id,
            policy_id=policy.policy_id,
            artifact_id=artifact.artifact_id,
        )
        projected = ConsoleTraceNativeReader(
            database,
            repository=repository,
        )._project_continuation(
            cursor,
            call=SimpleNamespace(policy_id=policy.policy_id),
            revision_id=revision_id,
            expected_conversation_id=conversation_id,
        )

    assert projected["rounds"][0]["reasoning_blocks"] == ["[PII omitted]"]
    assert "customer-ABCDWXYZ" not in repr(projected)
