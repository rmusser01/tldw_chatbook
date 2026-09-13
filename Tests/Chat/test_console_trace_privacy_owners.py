"""Credential filtering at every durable and derived semantic-trace owner."""

from __future__ import annotations

import json

from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail, ExchangeCapture
from tldw_chatbook.Chat.console_exchange_export import project_exchange_export
from tldw_chatbook.Chat.console_semantic_revision import (
    SemanticRevisionCoordinator,
    project_semantic_revision_trace_message,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_redaction import CREDENTIAL_FILTER_VERSION
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.trace_export_profiles import TraceExportProfile
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


SECRET = "sk-live-task231136-super-secret"


def test_repository_filters_credentials_before_header_and_artifact_identity() -> None:
    database = CharactersRAGDB(":memory:", "trace-privacy-owner-test")
    repository = ConsoleTraceRepository()
    try:
        with database.transaction() as cursor:
            artifact = repository.store_sanitized_artifact(
                cursor,
                sanitized_bytes=json.dumps(
                    {"api_key": SECRET, "body": f"Bearer {SECRET}", "safe": "keep"}
                ).encode("utf-8"),
                media_type="application/json",
                normalization_version="canonical-json-v1",
            )
            header = repository.create_or_reuse_request_header(
                cursor,
                provider_name="openai",
                model_name="gpt-test",
                route_identity="responses",
                endpoint_identity=f"https://user:{SECRET}@example.invalid/v1?token={SECRET}",
                generation_parameters={"api_key": SECRET, "temperature": 0.2},
                adapter_defaults={"nested": {"authorization": f"Bearer {SECRET}"}},
                response_format={"type": "text"},
                reasoning_controls={"note": f"token={SECRET}"},
                components=(),
            )
            stored_header = cursor.execute(
                """SELECT endpoint_identity, generation_parameters_json,
                          adapter_defaults_json, reasoning_controls_json
                     FROM console_trace_request_headers WHERE header_id = ?""",
                (header.header_id,),
            ).fetchone()
            stored_artifact = cursor.execute(
                "SELECT sanitized_bytes FROM console_trace_artifacts WHERE artifact_id = ?",
                (artifact.artifact_id,),
            ).fetchone()[0]

        durable = " ".join(str(value) for value in stored_header) + bytes(
            stored_artifact
        ).decode("utf-8")
        assert SECRET not in durable
        assert "api_key" not in durable
        assert "https://example.invalid/v1" in durable
        assert "keep" in durable
    finally:
        database.close_connection()


def test_copy_and_export_reapply_mandatory_credential_projection() -> None:
    capture = ExchangeCapture(
        run_tag="run",
        seq=1,
        created_at="2026-08-31T00:00:00Z",
        provider="openai",
        model="gpt-test",
        endpoint=f"https://user:{SECRET}@example.invalid/v1?api_key={SECRET}",
        request={
            "messages_payload": [{"role": "user", "content": f"Bearer {SECRET}"}],
            "api_key": SECRET,
        },
        response={"content": f"token={SECRET}", "tool_calls": []},
        status="complete",
        usage_json=None,
        omitted_keys=(),
        capture_detail=CaptureDetail.FULL,
    )

    projection = project_exchange_export(capture, TraceExportProfile.FULL_TRACE)

    assert SECRET not in projection.json_text
    assert "api_key" not in projection.payload["request"]
    assert "https://example.invalid/v1" in projection.json_text


def test_live_canonical_revision_is_credential_safe_without_changing_message() -> None:
    database = CharactersRAGDB(":memory:", "trace-credential-canonical-test")
    repository = ConsoleTraceRepository()
    conversation_id = database.add_conversation({"title": "Credential projection"})
    assert conversation_id is not None
    source = f"Use Bearer {SECRET} for this request"
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": source,
        }
    )
    assert message_id is not None
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version=CREDENTIAL_FILTER_VERSION,
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )
    try:
        with database.transaction(immediate=True) as cursor:
            revision = SemanticRevisionCoordinator(database).ensure_current_revision(
                cursor,
                message_id=message_id,
            )
            repository.ensure_policy(cursor, policy)
            projected = project_semantic_revision_trace_message(
                cursor,
                revision_id=revision.revision_id,
                expected_conversation_id=conversation_id,
                policy_id=policy.policy_id,
            )
            canonical = cursor.execute(
                "SELECT content FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()[0]

        assert SECRET not in str(projected)
        assert "credential omitted" in str(projected)
        assert canonical == source
    finally:
        database.close_connection()
