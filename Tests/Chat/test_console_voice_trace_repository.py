"""Transaction and invariants for the Console semantic trace repository."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from tldw_chatbook.Chat.console_exchange_capture import (
    freeze_provisional_capture_eligibility,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    HeaderComponentRef,
    TraceIdentityConflict,
)
from tldw_chatbook.Chat.console_trace_service import (
    ConsoleTraceService,
    ProvisionalTraceImportRetryableError,
)
from tldw_chatbook.Chat.console_voice_trace_gateway import (
    ProvisionalTraceRegistry,
    VoiceTraceImportContext,
)
from tldw_chatbook.Chat.console_voice_trace_promotion import (
    ConfirmedPreCommitTraceImportError,
    MAX_PROMOTED_TRACE_BYTES,
    PostDispatchTraceArtifact,
    PostDispatchTraceCall,
    PostDispatchTraceHeaderComponent,
    PostDispatchTraceHeaderOmission,
    PostDispatchTraceImport,
    PostDispatchTraceResponse,
    PostDispatchTraceSurfaceComponent,
    PostDispatchTraceSystemComponent,
    derive_post_dispatch_trace_ids,
    derive_post_dispatch_trace_node_id,
    derive_post_dispatch_trace_replacement_id,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db() -> CharactersRAGDB:
    database = CharactersRAGDB(":memory:", "console-trace-repository-test")
    yield database
    database.close_connection()


@pytest.fixture
def repository() -> ConsoleTraceRepository:
    return ConsoleTraceRepository()


@pytest.fixture
def handoff_db(tmp_path):
    database = CharactersRAGDB(tmp_path / "voice-handoff.sqlite", "voice-handoff")
    try:
        yield database
    finally:
        database.close()


def _conversation_with_message(db: CharactersRAGDB) -> tuple[str, str]:
    conversation_id = db.add_conversation({"title": "semantic trace"})
    assert conversation_id is not None
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "must never enter semantic revision metadata",
        }
    )
    assert message_id is not None
    return conversation_id, message_id


def _untracked_message(
    db: CharactersRAGDB,
    *,
    conversation_id: str | None = None,
    content: str = "must never enter semantic revision metadata",
) -> tuple[str, str]:
    """Insert a message without invoking the semantic revision coordinator."""

    if conversation_id is None:
        conversation_id = db.add_conversation({"title": "semantic trace"})
        assert conversation_id is not None
    message_id = db._generate_uuid()
    now = db._get_current_utc_timestamp_iso()
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            """INSERT INTO messages(
                   id, conversation_id, sender, content, timestamp,
                   last_modified, client_id, version, deleted, role)
                 VALUES (?, ?, 'user', ?, ?, ?, ?, 1, 0, 'user')""",
            (message_id, conversation_id, content, now, now, db.client_id),
        )
    return conversation_id, message_id


def _policy() -> FrozenTracePolicy:
    return FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="cred-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )


def _retained_call_artifacts(
    call: PostDispatchTraceCall,
) -> dict[str, PostDispatchTraceArtifact]:
    artifacts: dict[str, PostDispatchTraceArtifact] = {}
    values = [
        *(component.artifact_value for component in call.request_surface),
        *(component.artifact_value for component in call.header_components),
        call.response.artifact_value if call.response is not None else None,
    ]
    for artifact in values:
        if artifact is None:
            continue
        assert artifacts.get(artifact.artifact_id, artifact) == artifact
        artifacts[artifact.artifact_id] = artifact
    return artifacts


def _retained_call_bytes(call: PostDispatchTraceCall) -> int:
    return sum(
        len(value.encode("utf-8"))
        for value in (
            call.generation_parameters_json,
            call.adapter_defaults_json,
            call.response_format_json,
            call.reasoning_controls_json,
            *((call.usage_json,) if call.usage_json is not None else ()),
        )
    ) + sum(
        artifact.retained_bytes for artifact in _retained_call_artifacts(call).values()
    )


def _seal_post_dispatch_import(
    request: PostDispatchTraceImport,
) -> PostDispatchTraceImport:
    calls = tuple(
        replace(call, sealed_payload_bytes=_retained_call_bytes(call))
        for call in request.calls
    )
    aggregate_artifacts: dict[str, PostDispatchTraceArtifact] = {}
    for call in calls:
        aggregate_artifacts.update(_retained_call_artifacts(call))
    aggregate_payload_bytes = sum(
        len(value.encode("utf-8"))
        for call in calls
        for value in (
            call.generation_parameters_json,
            call.adapter_defaults_json,
            call.response_format_json,
            call.reasoning_controls_json,
            *((call.usage_json,) if call.usage_json is not None else ()),
        )
    ) + sum(artifact.retained_bytes for artifact in aggregate_artifacts.values())
    return replace(
        request,
        calls=calls,
        aggregate_payload_bytes=aggregate_payload_bytes,
    )


def _post_dispatch_import(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    user_message_id: str,
    assistant_message_id: str,
    call_count: int = 2,
) -> PostDispatchTraceImport:
    connection = db.get_connection()
    revisions = {
        str(row["source_message_id"]): str(row["revision_id"])
        for row in connection.execute(
            """SELECT source_message_id, revision_id
                 FROM console_trace_semantic_revisions
                WHERE source_message_id IN (?, ?)""",
            (user_message_id, assistant_message_id),
        )
    }
    import_id = new_opaque_id()
    ids = derive_post_dispatch_trace_ids(import_id, call_count=call_count)
    request_surface = (
        PostDispatchTraceSurfaceComponent.revision(
            node_id=derive_post_dispatch_trace_node_id(import_id, 0, 0),
            component_kind="message",
            revision_id=revisions[user_message_id],
        ),
    )
    return _seal_post_dispatch_import(
        PostDispatchTraceImport(
            import_id=import_id,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            user_revision_id=revisions[user_message_id],
            assistant_message_id=assistant_message_id,
            assistant_revision_id=revisions[assistant_message_id],
            turn_id=user_message_id,
            run_id="voice-run",
            policy=FrozenTracePolicy(
                policy_id=new_opaque_id(),
                credential_filter_version="cred-v1",
                pii_redaction_enabled=False,
                pii_ruleset_revision_id=None,
            ),
            expected_call_count=call_count,
            calls=tuple(
                PostDispatchTraceCall(
                    call_id=ids.call_ids[sequence],
                    idempotency_key=f"voice-import-{import_id}-{sequence}",
                    call_sequence=sequence,
                    provider_name="openai",
                    model_name="gpt-test",
                    route_identity="chat_completions",
                    endpoint_identity="public_api",
                    generation_parameters_json="{}",
                    adapter_defaults_json="{}",
                    response_format_json="{}",
                    reasoning_controls_json="{}",
                    dispatch_started_at=f"2026-08-31T12:00:0{sequence * 3}Z",
                    response_started_at=(
                        f"2026-08-31T12:00:0{sequence * 3 + 1}Z"
                        if sequence == call_count - 1
                        else None
                    ),
                    settled_at=f"2026-08-31T12:00:0{sequence * 3 + 2}Z",
                    usage_json='{"input_tokens":1,"output_tokens":2}',
                    request_surface=request_surface,
                    response=(
                        PostDispatchTraceResponse.committed_revision(
                            revisions[assistant_message_id]
                        )
                        if sequence == call_count - 1
                        else PostDispatchTraceResponse.no_response(
                            "superseded_before_response"
                        )
                    ),
                    sealed_payload_bytes=0,
                    terminal_state=(
                        TraceCallState.COMPLETE
                        if sequence == call_count - 1
                        else TraceCallState.ERROR
                    ),
                )
                for sequence in range(call_count)
            ),
            aggregate_payload_bytes=0,
        )
    )


def _completed_pair(db: CharactersRAGDB) -> tuple[str, str, str]:
    conversation_id = db.add_conversation({"title": "promoted voice"})
    assert conversation_id is not None
    user_message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": "protected user transcript",
        }
    )
    assert user_message_id is not None
    assistant_message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": user_message_id,
            "sender": "assistant",
            "role": "assistant",
            "content": "protected assistant reply",
            "assistant_generation_state": "complete",
        }
    )
    assert assistant_message_id is not None
    return conversation_id, user_message_id, assistant_message_id


def test_import_post_dispatch_trace_atomically_creates_owner_and_ordered_calls(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )

    result = repository.import_post_dispatch_trace(db, request)

    assert result.already_imported is False
    assert result.conversation_id == conversation_id
    assert result.call_ids == tuple(call.call_id for call in request.calls)
    connection = db.get_connection()
    rows = connection.execute(
        """SELECT call_id, call_sequence, state, integrity_state,
                  reservation_provenance, import_reason_code,
                  dispatch_started_at, response_started_at, settled_at
             FROM console_trace_calls WHERE owner_id = ?
            ORDER BY call_sequence""",
        (result.owner_id,),
    ).fetchall()
    assert [tuple(row[0:6]) for row in rows] == [
        (
            call.call_id,
            call.call_sequence,
            call.terminal_state.value,
            "complete",
            "post_dispatch_promoted",
            "provisional_voice_promoted",
        )
        for call in request.calls
    ]
    assert [tuple(row[6:9]) for row in rows] == [
        (
            call.dispatch_started_at,
            call.response_started_at,
            call.settled_at,
        )
        for call in request.calls
    ]
    event_types = [
        str(row[0])
        for row in connection.execute(
            """SELECT event_type FROM console_trace_events
                 WHERE segment_id = ? ORDER BY sequence""",
            (result.segment_id,),
        )
    ]
    assert "request_header_selection" in event_types
    assert "response_selection" in event_types
    assert "call_outcome" in event_types
    assert event_types.count("call_boundary") == len(request.calls)
    assert not {"reserve", "bind", "dispatch_start"} & set(event_types)
    assert (
        connection.execute(
            """SELECT COUNT(*) FROM console_trace_response_links
             WHERE semantic_revision_id = ?""",
            (request.assistant_revision_id,),
        ).fetchone()[0]
        == 1
    )


def test_import_post_dispatch_trace_preserves_exact_typed_header_and_rejects_tamper(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        call_count=1,
    )
    system_artifact = PostDispatchTraceArtifact(
        artifact_id=new_opaque_id(),
        media_type="application/json",
        normalization_version="json-v1",
        sanitized_bytes=b'"rendered-system-canary"',
    )
    tool_artifact = PostDispatchTraceArtifact(
        artifact_id=new_opaque_id(),
        media_type="application/json",
        normalization_version="json-v1",
        sanitized_bytes=b'{"function":{"name":"lookup"},"type":"function"}',
    )
    literal_artifact = PostDispatchTraceArtifact(
        artifact_id=new_opaque_id(),
        media_type="application/json",
        normalization_version="json-v1",
        sanitized_bytes=b'{"model":"gpt-test","stream":true}',
    )
    system_composition = (
        PostDispatchTraceSystemComponent.transform_start("single_preamble"),
        PostDispatchTraceSystemComponent.revision(valid.user_revision_id),
        PostDispatchTraceSystemComponent.artifact(0),
        PostDispatchTraceSystemComponent.omission(
            source="project_instruction",
            reason_code="windowed",
        ),
        PostDispatchTraceSystemComponent.transform_end("single_preamble"),
    )
    header_components = (
        PostDispatchTraceHeaderComponent(
            "provider_literal_envelope", 0, literal_artifact
        ),
        PostDispatchTraceHeaderComponent("rendered_system_part", 0, system_artifact),
        PostDispatchTraceHeaderComponent("tool_schema", 0, tool_artifact),
    )
    header_omissions = (
        PostDispatchTraceHeaderOmission("tool_schema", 1, "source_unavailable"),
    )
    adapter_defaults_json = json.dumps(
        {
            "header_omissions": {"tool_schema:1": "source_unavailable"},
            "literal_surface_field": "messages",
            "system_composition": [
                component.as_json_object() for component in system_composition
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    call = replace(
        valid.calls[0],
        adapter_defaults_json=adapter_defaults_json,
        header_components=header_components,
        system_composition=system_composition,
        header_omissions=header_omissions,
    )
    retained_bytes = _retained_call_bytes(call)
    call = replace(call, sealed_payload_bytes=retained_bytes)
    request = replace(
        valid,
        calls=(call,),
        aggregate_payload_bytes=retained_bytes,
    )

    imported = repository.import_post_dispatch_trace(db, request)
    retried = repository.import_post_dispatch_trace(db, request)

    assert retried.already_imported is True
    identities = derive_post_dispatch_trace_ids(request.import_id, call_count=1)
    header = repository.get_request_header(
        db.get_connection().cursor(), identities.header_ids[0]
    )
    assert header is not None
    assert header.components == tuple(
        HeaderComponentRef(
            component.component_kind,
            component.ordinal,
            component.artifact_value.artifact_id,
        )
        for component in header_components
    )
    assert "rendered-system-canary" not in repr(request)
    assert "lookup" not in repr(request)

    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            """INSERT INTO console_trace_header_components(
                   header_id, component_kind, ordinal, artifact_id)
                 VALUES (?, 'tool_schema', 2, ?)""",
            (identities.header_ids[0], tool_artifact.artifact_id),
        )
    with pytest.raises(TraceIdentityConflict, match="post_dispatch_header"):
        repository.import_post_dispatch_trace(db, request)
    assert imported.call_ids == retried.call_ids


def test_import_post_dispatch_trace_rejects_header_artifact_identity_collision(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        call_count=1,
    )
    artifact = PostDispatchTraceArtifact(
        artifact_id=new_opaque_id(),
        media_type="application/json",
        normalization_version="json-v1",
        sanitized_bytes=b'{"type":"function"}',
    )
    request = _seal_post_dispatch_import(
        replace(
            valid,
            calls=(
                replace(
                    valid.calls[0],
                    header_components=(
                        PostDispatchTraceHeaderComponent("tool_schema", 0, artifact),
                    ),
                ),
            ),
        )
    )
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            """INSERT INTO console_trace_artifacts(
                   artifact_id, identity_digest, media_type,
                   normalization_version, sanitized_bytes, byte_length)
                 VALUES (?, ?, 'application/json', 'json-v1', ?, ?)""",
            (
                artifact.artifact_id,
                hashlib.sha256(b'{"different":true}').hexdigest(),
                sqlite3.Binary(b'{"different":true}'),
                len(b'{"different":true}'),
            ),
        )

    with pytest.raises(TraceIdentityConflict, match="post_dispatch_artifact"):
        repository.import_post_dispatch_trace(db, request)

    assert (
        repository.get_attached_owner_by_conversation(
            db.get_connection().cursor(), conversation_id
        )
        is None
    )


def test_import_post_dispatch_trace_preserves_per_call_surfaces_and_responses(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    connection = db.get_connection()
    revisions = {
        str(row["source_message_id"]): str(row["revision_id"])
        for row in connection.execute(
            """SELECT source_message_id, revision_id
                 FROM console_trace_semantic_revisions
                WHERE source_message_id IN (?, ?)""",
            (user_message_id, assistant_message_id),
        )
    }
    import_id = new_opaque_id()
    identities = derive_post_dispatch_trace_ids(import_id, call_count=2)
    first_response = PostDispatchTraceArtifact(
        artifact_id=new_opaque_id(),
        media_type="application/json",
        normalization_version="json-v1",
        sanitized_bytes=b'{"provider":"first-response"}',
    )
    user_component = PostDispatchTraceSurfaceComponent.revision(
        node_id=derive_post_dispatch_trace_node_id(import_id, 0, 0),
        component_kind="message",
        revision_id=revisions[user_message_id],
    )
    provider_component = PostDispatchTraceSurfaceComponent.artifact(
        node_id=derive_post_dispatch_trace_node_id(import_id, 1, 1),
        component_kind="provider_response",
        artifact=first_response,
    )
    calls = (
        PostDispatchTraceCall(
            call_id=identities.call_ids[0],
            idempotency_key=f"artifact-first-{import_id}",
            call_sequence=0,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="chat_completions",
            endpoint_identity="public_api",
            generation_parameters_json="{}",
            adapter_defaults_json="{}",
            response_format_json="{}",
            reasoning_controls_json="{}",
            dispatch_started_at="2026-08-31T12:00:00Z",
            response_started_at="2026-08-31T12:00:01Z",
            settled_at="2026-08-31T12:00:02Z",
            usage_json=None,
            request_surface=(user_component,),
            response=PostDispatchTraceResponse.artifact(first_response),
            sealed_payload_bytes=0,
            terminal_state=TraceCallState.ERROR,
        ),
        PostDispatchTraceCall(
            call_id=identities.call_ids[1],
            idempotency_key=f"revision-final-{import_id}",
            call_sequence=1,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="chat_completions",
            endpoint_identity="public_api",
            generation_parameters_json="{}",
            adapter_defaults_json="{}",
            response_format_json="{}",
            reasoning_controls_json="{}",
            dispatch_started_at="2026-08-31T12:00:03Z",
            response_started_at="2026-08-31T12:00:04Z",
            settled_at="2026-08-31T12:00:05Z",
            usage_json=None,
            request_surface=(user_component, provider_component),
            response=PostDispatchTraceResponse.committed_revision(
                revisions[assistant_message_id]
            ),
            sealed_payload_bytes=0,
        ),
    )
    request = _seal_post_dispatch_import(
        PostDispatchTraceImport(
            import_id=import_id,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            user_revision_id=revisions[user_message_id],
            assistant_message_id=assistant_message_id,
            assistant_revision_id=revisions[assistant_message_id],
            turn_id=user_message_id,
            run_id="voice-run",
            policy=_policy(),
            expected_call_count=2,
            calls=calls,
            aggregate_payload_bytes=0,
        )
    )

    first = repository.import_post_dispatch_trace(db, request)
    retried = repository.import_post_dispatch_trace(db, request)

    assert retried.already_imported is True
    assert retried.call_ids == first.call_ids
    surfaces = connection.execute(
        """SELECT semantic_revision_id, artifact_id
             FROM console_trace_surface_nodes WHERE segment_id = ?
            ORDER BY sequence""",
        (first.segment_id,),
    ).fetchall()
    assert [tuple(row) for row in surfaces] == [
        (revisions[user_message_id], None),
        (None, first_response.artifact_id),
    ]
    links = connection.execute(
        """SELECT call_id, link_kind, semantic_revision_id, artifact_id
             FROM console_trace_response_links
            WHERE call_id IN (?, ?) ORDER BY call_id""",
        first.call_ids,
    ).fetchall()
    by_call = {str(row[0]): tuple(row[1:]) for row in links}
    assert by_call[first.call_ids[0]] == (
        "artifact",
        None,
        first_response.artifact_id,
    )
    assert by_call[first.call_ids[1]] == (
        "revision",
        revisions[assistant_message_id],
        None,
    )
    assert "first-response" not in repr(request)


def test_import_post_dispatch_trace_preserves_terminal_no_response_without_link(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    first_call = replace(
        valid.calls[0],
        response_started_at=None,
        response=PostDispatchTraceResponse.no_response("provider_error_no_response"),
        terminal_state=TraceCallState.ERROR,
    )
    request = replace(valid, calls=(first_call, valid.calls[1]))

    result = repository.import_post_dispatch_trace(db, request)
    retried = repository.import_post_dispatch_trace(db, request)

    assert retried.already_imported is True
    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT 1 FROM console_trace_response_links WHERE call_id = ?",
            (result.call_ids[0],),
        ).fetchone()
        is None
    )
    assert tuple(
        connection.execute(
            """SELECT gap.event_type, gap.omission_reason_code
                 FROM console_trace_events AS gap
                 JOIN console_trace_events AS route
                   ON gap.segment_id = route.segment_id
                  AND gap.sequence = route.sequence + 1
                WHERE route.call_id = ?
                  AND route.event_type = 'provider_route_selection'
                  AND gap.call_id IS NULL AND gap.event_type = 'gap'""",
            (result.call_ids[0],),
        ).fetchone()
    ) == ("gap", "provider_error_no_response")


def test_import_post_dispatch_trace_preserves_divergent_rolling_surface(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    provisional_surface = (
        PostDispatchTraceSurfaceComponent.omission(
            node_id=derive_post_dispatch_trace_node_id(valid.import_id, 0, 0),
            component_kind="message",
            reason_code="provisional_revision_unavailable",
        ),
    )
    final_surface = (
        PostDispatchTraceSurfaceComponent.revision(
            node_id=derive_post_dispatch_trace_node_id(valid.import_id, 1, 0),
            component_kind="message",
            revision_id=valid.user_revision_id,
        ),
    )
    first_call = replace(
        valid.calls[0],
        response_started_at=None,
        request_surface=provisional_surface,
        response=PostDispatchTraceResponse.no_response("superseded_before_response"),
        terminal_state=TraceCallState.ERROR,
    )
    second_call = replace(valid.calls[1], request_surface=final_surface)
    request = replace(valid, calls=(first_call, second_call))

    result = repository.import_post_dispatch_trace(db, request)
    retried = repository.import_post_dispatch_trace(db, request)

    replacement_id = derive_post_dispatch_trace_replacement_id(valid.import_id, 1)
    connection = db.get_connection()
    assert retried.already_imported is True
    replacement_row = connection.execute(
        """SELECT predecessor_head_id, start_node_id, end_node_id,
                  replacement_node_id
             FROM console_trace_surface_replacements
            WHERE replacement_id = ?""",
        (replacement_id,),
    ).fetchone()
    assert replacement_row is not None
    assert tuple(replacement_row) == (
        provisional_surface[0].node_id,
        provisional_surface[0].node_id,
        provisional_surface[0].node_id,
        final_surface[0].node_id,
    )
    assert (
        connection.execute(
            """SELECT 1 FROM console_trace_events
            WHERE segment_id = ? AND event_type = 'surface_replace'
              AND surface_replacement_id = ?""",
            (result.segment_id, replacement_id),
        ).fetchone()
        is not None
    )


def test_import_post_dispatch_trace_retry_exactly_reconciles(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )

    first = repository.import_post_dispatch_trace(db, request)
    retried = repository.import_post_dispatch_trace(db, request)

    assert retried.owner_id == first.owner_id
    assert retried.segment_id == first.segment_id
    assert retried.call_ids == first.call_ids
    assert retried.already_imported is True


def test_import_post_dispatch_trace_retry_rejects_extra_call_event(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    imported = repository.import_post_dispatch_trace(db, request)
    with db.transaction(immediate=True) as cursor:
        tail = repository.get_event_tail(cursor, imported.segment_id)
        assert tail is not None
        repository.append_event(
            cursor,
            segment_id=imported.segment_id,
            sequence=tail.sequence + 1,
            event_type="usage",
            call_id=imported.call_ids[0],
        )

    with pytest.raises(TraceIdentityConflict, match="post_dispatch_events"):
        repository.import_post_dispatch_trace(db, request)


def test_post_dispatch_stopped_call_requires_observed_response_chronology(
    db: CharactersRAGDB,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )

    with pytest.raises(ValueError, match="requires response chronology"):
        replace(
            valid.calls[0],
            response_started_at=None,
            response=PostDispatchTraceResponse.no_response("stopped_before_response"),
            terminal_state=TraceCallState.STOPPED,
        )


def test_import_post_dispatch_trace_rejects_partial_declared_aggregate(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    partial = replace(valid, calls=(valid.calls[0],))

    with pytest.raises(ValueError, match="incomplete"):
        repository.import_post_dispatch_trace(db, partial)

    assert (
        repository.get_attached_owner_by_conversation(
            db.get_connection().cursor(), conversation_id
        )
        is None
    )


def test_import_post_dispatch_trace_rejects_inexact_aggregate_bytes(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )

    with pytest.raises(ValueError, match="aggregate byte accounting"):
        repository.import_post_dispatch_trace(
            db, replace(valid, aggregate_payload_bytes=1)
        )


def test_import_post_dispatch_trace_rejects_committed_revision_before_final_call(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    invalid = replace(
        valid,
        calls=(
            replace(
                valid.calls[0],
                response_started_at="2026-08-31T12:00:01Z",
                response=PostDispatchTraceResponse.committed_revision(
                    valid.assistant_revision_id
                ),
                terminal_state=TraceCallState.COMPLETE,
            ),
            valid.calls[1],
        ),
    )

    with pytest.raises(TraceIdentityConflict, match="promoted_nonfinal_response"):
        repository.import_post_dispatch_trace(db, invalid)

    assert (
        repository.get_attached_owner_by_conversation(
            db.get_connection().cursor(), conversation_id
        )
        is None
    )


@pytest.mark.parametrize(
    "terminal_state",
    [TraceCallState.ERROR, TraceCallState.STOPPED, TraceCallState.INTERRUPTED],
)
def test_import_post_dispatch_trace_rejects_noncomplete_final_revision(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    terminal_state: TraceCallState,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        call_count=1,
    )
    invalid = replace(
        valid,
        calls=(replace(valid.calls[0], terminal_state=terminal_state),),
    )

    with pytest.raises(TraceIdentityConflict, match="promoted_final_state"):
        repository.import_post_dispatch_trace(db, invalid)

    assert (
        repository.get_attached_owner_by_conversation(
            db.get_connection().cursor(), conversation_id
        )
        is None
    )


def test_import_post_dispatch_trace_counts_shared_artifact_once_per_retained_scope(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    shared_artifact = PostDispatchTraceArtifact(
        artifact_id=new_opaque_id(),
        media_type="application/json",
        normalization_version="json-v1",
        sanitized_bytes=b'{"shared":"provider-only"}',
    )
    header_component = PostDispatchTraceHeaderComponent(
        "tool_schema", 0, shared_artifact
    )
    provider_component = PostDispatchTraceSurfaceComponent.artifact(
        node_id=derive_post_dispatch_trace_node_id(valid.import_id, 1, 1),
        component_kind="provider_response",
        artifact=shared_artifact,
    )
    first_call = replace(
        valid.calls[0],
        header_components=(header_component,),
        response_started_at="2026-08-31T12:00:01Z",
        response=PostDispatchTraceResponse.artifact(shared_artifact),
        terminal_state=TraceCallState.ERROR,
    )
    second_call = replace(
        valid.calls[1],
        header_components=(header_component,),
        request_surface=(*valid.calls[1].request_surface, provider_component),
    )
    request = _seal_post_dispatch_import(
        replace(valid, calls=(first_call, second_call))
    )
    artifact_bytes = len(shared_artifact.sanitized_bytes)
    inline_payload_bytes = tuple(
        _retained_call_bytes(call) - artifact_bytes
        for call in (first_call, second_call)
    )

    assert tuple(call.sealed_payload_bytes for call in request.calls) == (
        inline_payload_bytes[0] + artifact_bytes,
        inline_payload_bytes[1] + artifact_bytes,
    )
    assert request.aggregate_payload_bytes == sum(inline_payload_bytes) + artifact_bytes
    repository.import_post_dispatch_trace(db, request)

    with pytest.raises(ValueError, match="call byte accounting"):
        repository.import_post_dispatch_trace(
            db,
            replace(
                request,
                calls=(
                    replace(
                        request.calls[0],
                        sealed_payload_bytes=request.calls[0].sealed_payload_bytes + 1,
                    ),
                    request.calls[1],
                ),
            ),
        )
    with pytest.raises(ValueError, match="aggregate byte accounting"):
        repository.import_post_dispatch_trace(
            db,
            replace(
                request,
                aggregate_payload_bytes=sum(
                    call.sealed_payload_bytes for call in request.calls
                ),
            ),
        )


def test_post_dispatch_trace_retained_boundary_with_scaled_quota(monkeypatch) -> None:
    from tldw_chatbook.Chat import console_trace_repository as repository_module

    assert MAX_PROMOTED_TRACE_BYTES == 64 * 1024 * 1024
    assert repository_module.MAX_PROMOTED_TRACE_BYTES == MAX_PROMOTED_TRACE_BYTES
    # Exercise the real aggregate validator with bounded valid JSON, leaving
    # credential authority and all production dataclass limits unchanged.
    quota = 1024
    monkeypatch.setattr(repository_module, "MAX_PROMOTED_TRACE_BYTES", quota)
    shared_bytes = json.dumps("x" * ((quota - 16) // 2 - 2)).encode("utf-8")
    conversation_id = new_opaque_id()
    user_message_id = new_opaque_id()
    assistant_message_id = new_opaque_id()
    user_revision_id = new_opaque_id()
    assistant_revision_id = new_opaque_id()
    import_id = new_opaque_id()
    identities = derive_post_dispatch_trace_ids(import_id, call_count=2)
    first_artifact = PostDispatchTraceArtifact(
        new_opaque_id(), "application/json", "canonical-json-v1", shared_bytes
    )
    second_artifact = PostDispatchTraceArtifact(
        new_opaque_id(), "application/json", "canonical-json-v1", shared_bytes
    )
    user_component = PostDispatchTraceSurfaceComponent.revision(
        node_id=derive_post_dispatch_trace_node_id(import_id, 0, 0),
        component_kind="message",
        revision_id=user_revision_id,
    )
    calls = (
        PostDispatchTraceCall(
            call_id=identities.call_ids[0],
            idempotency_key=f"boundary-{import_id}-0",
            call_sequence=0,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="chat_completions",
            endpoint_identity="public_api",
            generation_parameters_json="{}",
            adapter_defaults_json="{}",
            response_format_json="{}",
            reasoning_controls_json="{}",
            dispatch_started_at="2026-08-31T12:00:00Z",
            response_started_at="2026-08-31T12:00:01Z",
            settled_at="2026-08-31T12:00:02Z",
            request_surface=(
                user_component,
                PostDispatchTraceSurfaceComponent.artifact(
                    node_id=derive_post_dispatch_trace_node_id(import_id, 0, 1),
                    component_kind="provider_payload",
                    artifact=first_artifact,
                ),
            ),
            response=PostDispatchTraceResponse.artifact(first_artifact),
            terminal_state=TraceCallState.ERROR,
        ),
        PostDispatchTraceCall(
            call_id=identities.call_ids[1],
            idempotency_key=f"boundary-{import_id}-1",
            call_sequence=1,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="chat_completions",
            endpoint_identity="public_api",
            generation_parameters_json="{}",
            adapter_defaults_json="{}",
            response_format_json="{}",
            reasoning_controls_json="{}",
            dispatch_started_at="2026-08-31T12:00:03Z",
            response_started_at="2026-08-31T12:00:04Z",
            settled_at="2026-08-31T12:00:05Z",
            request_surface=(
                user_component,
                PostDispatchTraceSurfaceComponent.artifact(
                    node_id=derive_post_dispatch_trace_node_id(import_id, 1, 1),
                    component_kind="provider_payload",
                    artifact=second_artifact,
                ),
            ),
            response=PostDispatchTraceResponse.committed_revision(
                assistant_revision_id
            ),
        ),
    )
    request = _seal_post_dispatch_import(
        PostDispatchTraceImport(
            import_id=import_id,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            user_revision_id=user_revision_id,
            assistant_message_id=assistant_message_id,
            assistant_revision_id=assistant_revision_id,
            turn_id=user_message_id,
            run_id="boundary-run",
            policy=_policy(),
            expected_call_count=2,
            calls=calls,
        )
    )

    assert request.aggregate_payload_bytes == quota
    ConsoleTraceRepository._validate_post_dispatch_request(request)

    overflow_bytes = json.dumps(json.loads(shared_bytes) + "x").encode("utf-8")
    assert len(overflow_bytes) == len(shared_bytes) + 1
    overflow_artifact = PostDispatchTraceArtifact(
        new_opaque_id(), "application/json", "canonical-json-v1", overflow_bytes
    )
    overflow_call = replace(
        request.calls[1],
        request_surface=(
            user_component,
            PostDispatchTraceSurfaceComponent.artifact(
                node_id=derive_post_dispatch_trace_node_id(import_id, 1, 1),
                component_kind="provider_payload",
                artifact=overflow_artifact,
            ),
        ),
    )
    overflow_request = _seal_post_dispatch_import(
        replace(request, calls=(request.calls[0], overflow_call))
    )
    assert overflow_request.aggregate_payload_bytes == quota + 1
    with pytest.raises(ValueError, match="aggregate exceeds"):
        ConsoleTraceRepository._validate_post_dispatch_request(overflow_request)


def test_import_post_dispatch_trace_rejects_cross_call_chronology(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    overlapping_second = replace(
        valid.calls[1], dispatch_started_at="2026-08-31T12:00:01Z"
    )

    with pytest.raises(ValueError, match="chronology"):
        repository.import_post_dispatch_trace(
            db, replace(valid, calls=(valid.calls[0], overlapping_second))
        )


def test_import_post_dispatch_trace_rejects_preexisting_deterministic_residue(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    header_id = derive_post_dispatch_trace_ids(
        request.import_id, call_count=request.expected_call_count
    ).header_ids[0]
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            """INSERT INTO console_trace_request_headers(
                   header_id, provider_name, model_name, route_identity,
                   endpoint_identity, generation_parameters_json,
                   adapter_defaults_json, response_format_json,
                   reasoning_controls_json)
                 VALUES (?, 'other', 'other', 'other', 'other', '{}', '{}', '{}', '{}')""",
            (header_id,),
        )

    with pytest.raises(TraceIdentityConflict, match="post_dispatch_residue"):
        repository.import_post_dispatch_trace(db, request)

    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM console_trace_calls WHERE call_id IN (?, ?)",
            tuple(call.call_id for call in request.calls),
        )
        .fetchone()[0]
        == 0
    )


def test_import_post_dispatch_trace_reconciles_exception_after_commit(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )

    def fail_after_commit(_request: PostDispatchTraceImport) -> None:
        raise RuntimeError("synthetic_after_commit")

    monkeypatch.setattr(
        repository, "_after_post_dispatch_trace_commit", fail_after_commit
    )
    with pytest.raises(RuntimeError, match="synthetic_after_commit"):
        repository.import_post_dispatch_trace(db, request)

    retried = repository.import_post_dispatch_trace(db, request)

    assert retried.already_imported is True
    assert retried.call_ids == tuple(call.call_id for call in request.calls)


def test_import_post_dispatch_trace_classifies_transaction_body_rollback_as_confirmed(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )

    def fail_inside_transaction(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("transaction body failed")

    monkeypatch.setattr(
        repository,
        "_write_post_dispatch_trace",
        fail_inside_transaction,
    )

    with pytest.raises(ConfirmedPreCommitTraceImportError):
        repository.import_post_dispatch_trace(db, request)

    call_ids = derive_post_dispatch_trace_ids(
        request.import_id,
        call_count=request.expected_call_count,
    ).call_ids
    placeholders = ",".join("?" for _ in call_ids)
    rows = (
        db.get_connection()
        .execute(
            f"SELECT call_id FROM console_trace_calls WHERE call_id IN ({placeholders})",
            call_ids,
        )
        .fetchall()
    )
    assert rows == []


def test_import_post_dispatch_trace_rejects_borrowed_transaction_before_body(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )

    body_called = False

    def fail_inside_transaction(*args: object, **kwargs: object) -> None:
        nonlocal body_called
        del args, kwargs
        body_called = True
        raise RuntimeError("borrowed transaction body failed")

    monkeypatch.setattr(
        repository,
        "_write_post_dispatch_trace",
        fail_inside_transaction,
    )
    connection = db.get_connection()
    connection.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(TraceIdentityConflict, match="transaction_owner"):
            repository.import_post_dispatch_trace(db, request)
        assert body_called is False
        assert connection.in_transaction is True
    finally:
        connection.rollback()


def test_post_dispatch_import_rejects_managed_outer_transaction_before_writes(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    call_ids = tuple(call.call_id for call in request.calls)

    with db.transaction(immediate=True):
        with pytest.raises(TraceIdentityConflict, match="transaction_owner"):
            repository.import_post_dispatch_trace(db, request)
        placeholders = ",".join("?" for _ in call_ids)
        rows = (
            db.get_connection()
            .execute(
                f"SELECT call_id FROM console_trace_calls WHERE call_id IN ({placeholders})",
                call_ids,
            )
            .fetchall()
        )
        assert rows == []


def test_ordinary_repository_apis_still_join_caller_transaction(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    policy = _policy()
    with db.transaction(immediate=True) as cursor:
        created = repository.ensure_policy(cursor, policy)
        loaded = repository.get_policy(cursor, policy.policy_id)
    assert created == loaded


def test_service_retries_same_manifest_after_real_repository_confirmed_rollback(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    registry = ProvisionalTraceRegistry()
    attempt = registry.begin_attempt(
        promotion_id=request.import_id,
        attempt_id=new_opaque_id(),
        policy=request.policy,
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True,
            session_is_saved=True,
        ),
    )
    assert attempt is not None
    envelopes = tuple(
        registry._retain_gateway_call(attempt, call) for call in request.calls
    )
    assert all(envelope is not None for envelope in envelopes)
    manifest = registry.seal_attempt(
        attempt,
        expected_call_count=request.expected_call_count,
    )
    context = VoiceTraceImportContext(
        import_id=request.import_id,
        conversation_id=request.conversation_id,
        user_message_id=request.user_message_id,
        user_revision_id=request.user_revision_id,
        assistant_message_id=request.assistant_message_id,
        assistant_revision_id=request.assistant_revision_id,
        turn_id=request.turn_id,
        run_id=request.run_id,
        policy=request.policy,
    )
    original_write = repository._write_post_dispatch_trace
    should_fail = True

    def fail_once(*args: object, **kwargs: object):
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("transaction body failed once")
        return original_write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(repository, "_write_post_dispatch_trace", fail_once)
    service = ConsoleTraceService(repository=repository)
    retained_bytes = registry.retained_bytes

    with pytest.raises(ProvisionalTraceImportRetryableError):
        service.import_provisional_voice_trace(
            db,
            registry,
            manifest,
            envelopes,
            context,
        )
    assert registry.retained_bytes == retained_bytes

    result = service.import_provisional_voice_trace(
        db,
        registry,
        manifest,
        envelopes,
        context,
    )
    assert result.call_ids == tuple(call.call_id for call in request.calls)
    assert registry.retained_bytes == 0


@pytest.mark.asyncio
async def test_two_winning_imports_reconstruct_exact_full_surfaces(
    handoff_db, repository
):
    db = handoff_db
    conversation, user1, assistant1 = _completed_pair(db)
    first = _post_dispatch_import(
        db,
        conversation_id=conversation,
        user_message_id=user1,
        assistant_message_id=assistant1,
        call_count=1,
    )
    first_result = repository.import_post_dispatch_trace(db, first)
    user2 = db.add_message(
        {
            "conversation_id": conversation,
            "parent_message_id": assistant1,
            "sender": "user",
            "role": "user",
            "content": "second user",
        }
    )
    assistant2 = db.add_message(
        {
            "conversation_id": conversation,
            "parent_message_id": user2,
            "sender": "assistant",
            "role": "assistant",
            "content": "second reply",
            "assistant_generation_state": "complete",
        }
    )
    second = _post_dispatch_import(
        db,
        conversation_id=conversation,
        user_message_id=user2,
        assistant_message_id=assistant2,
        call_count=1,
    )
    revisions = (
        first.user_revision_id,
        first.assistant_revision_id,
        second.user_revision_id,
    )
    surface = tuple(
        PostDispatchTraceSurfaceComponent.revision(
            node_id=derive_post_dispatch_trace_node_id(second.import_id, 0, ordinal),
            component_kind="message",
            revision_id=revision,
        )
        for ordinal, revision in enumerate(revisions)
    )
    second = _seal_post_dispatch_import(
        replace(
            second,
            calls=(replace(second.calls[0], request_surface=surface),),
        )
    )
    result = repository.import_post_dispatch_trace(db, second)
    assert result.owner_id == first_result.owner_id
    with db.transaction() as cursor:
        service = ConsoleTraceService(repository=repository)
        for request, expected in (
            (first, (first.user_revision_id,)),
            (second, revisions),
        ):
            tail = repository.get_surface_node(
                cursor, request.calls[-1].request_surface[-1].node_id
            )
            projection = service._surface_projection(cursor, result.segment_id, tail)
            assert tuple(key[2] for _, key in projection.entries) == expected
    assert repository.import_post_dispatch_trace(db, first).already_imported
    assert repository.import_post_dispatch_trace(db, second).already_imported

    from Tests.Chat.test_console_trace_runtime import _semantic_request
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
        ConsoleProviderStreamSignals,
    )
    from tldw_chatbook.Chat.console_trace_provenance import (
        ConsoleRequestRoute,
        ConsoleTraceCaptureMode,
        SavedRevisionTraceProvenance,
    )
    from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory

    user3 = db.add_message(
        {
            "conversation_id": conversation,
            "parent_message_id": assistant2,
            "sender": "user",
            "role": "user",
            "content": "ordinary third user",
        }
    )
    message_ids = (user1, assistant1, user2, assistant2, user3)
    rows = [db.get_message_by_id(message_id) for message_id in message_ids]
    messages = [{"role": row["role"], "content": row["content"]} for row in rows]
    descriptors = [
        SavedRevisionTraceProvenance(
            str(
                db.get_connection()
                .execute(
                    "SELECT revision_id FROM console_trace_semantic_revisions WHERE source_message_id = ?",
                    (message_id,),
                )
                .fetchone()[0]
            )
        )
        for message_id in message_ids
    ]
    observed = []

    def inference(**kwargs):
        observed.append(kwargs["messages_payload"])
        return {"choices": [{"message": {"content": "ordinary reply"}}]}

    factory = ConsoleTraceBoundaryFactory(db)
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=inference, trace_call_boundary_factory=factory
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="openai",
        model="gpt-test",
        execution_key="openai",
        base_url="https://api.openai.com/v1",
        streaming=False,
    )
    try:
        prepared = gateway.prepare_chat_request(
            resolution,
            _semantic_request(messages, descriptors, _policy()),
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
        signals = ConsoleProviderStreamSignals()
        signals.bind_trace_settlement_sink(lambda handoff: handoff.settle(None))
        assert [
            item
            async for item in gateway.stream_chat(
                resolution,
                prepared,
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                signals=signals,
            )
        ] == ["ordinary reply"]
        assert observed == [tuple(messages)]
        with db.transaction() as cursor:
            tail = factory.service._effective_surface_tail(cursor, result.segment_id)
            projected = factory.service._surface_projection(
                cursor, result.segment_id, tail
            )
            assert tuple(key[2] for _, key in projected.entries) == tuple(
                item.revision_id for item in descriptors
            )
        assert repository.import_post_dispatch_trace(db, second).already_imported
    finally:
        await gateway.aclose()


@pytest.mark.parametrize("different", [False, True])
@pytest.mark.parametrize("retained_prefix", [False, True])
def test_retry_manifest_reconstructs_all_components_in_original_order(
    db, repository, different, retained_prefix
):
    conversation, user, assistant = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation,
        user_message_id=user,
        assistant_message_id=assistant,
        call_count=2,
    )
    calls = []
    for call in request.calls:
        surface = (
            PostDispatchTraceSurfaceComponent.omission(
                node_id=derive_post_dispatch_trace_node_id(
                    request.import_id, call.call_sequence, 0
                ),
                component_kind="provider_message",
                reason_code=(
                    "source_unavailable"
                    if different and call.call_sequence
                    else "trace_source_unavailable"
                ),
            ),
            PostDispatchTraceSurfaceComponent.revision(
                node_id=derive_post_dispatch_trace_node_id(
                    request.import_id, call.call_sequence, 1
                ),
                component_kind="provider_message",
                revision_id=request.user_revision_id,
            ),
        )
        if retained_prefix:
            surface = (
                PostDispatchTraceSurfaceComponent.omission(
                    node_id=derive_post_dispatch_trace_node_id(request.import_id, 0, 0),
                    component_kind="provider_message",
                    reason_code="prefix_unavailable",
                ),
                replace(
                    surface[0],
                    node_id=derive_post_dispatch_trace_node_id(
                        request.import_id, call.call_sequence, 1
                    ),
                ),
                replace(
                    surface[1],
                    node_id=derive_post_dispatch_trace_node_id(
                        request.import_id, call.call_sequence, 2
                    ),
                ),
            )
        calls.append(replace(call, request_surface=surface))
    request = _seal_post_dispatch_import(replace(request, calls=tuple(calls)))
    result = repository.import_post_dispatch_trace(db, request)
    with db.transaction() as cursor:
        service = ConsoleTraceService(repository=repository)
        events = tuple(repository.read_events(cursor, result.segment_id))
        for call in request.calls:
            tail = repository.get_surface_node(cursor, call.request_surface[-1].node_id)
            projection = service._surface_projection(cursor, result.segment_id, tail)
            assert tuple(key for _, key in projection.entries) == tuple(
                (
                    item.component_kind,
                    item.reference_kind,
                    item.revision_id or item.omission_reason_code,
                )
                for item in call.request_surface
            )
            boundary = next(
                event
                for event in events
                if event.event_type == "call_boundary" and event.call_id == call.call_id
            )
            assert (
                repository.surface_head_at_event_boundary(
                    cursor,
                    segment_id=result.segment_id,
                    through_sequence=boundary.sequence,
                )
                == call.request_surface[-1].node_id
            )
        replacements = {
            record.replacement_id: record.replacement
            for record in repository.read_surface_replacements(
                cursor, result.segment_id
            )
        }
        for index, event in enumerate(events):
            if event.event_type == "surface_replace":
                assert events[index - 1].event_type == "surface_append"
                assert (
                    events[index - 1].surface_node_id
                    == replacements[event.surface_replacement_id].replacement_node_id
                )
    assert repository.import_post_dispatch_trace(db, request).already_imported


def test_initial_surface_replacement_identity_is_stable_and_distinct():
    import_id = new_opaque_id()
    initial = derive_post_dispatch_trace_replacement_id(import_id, 0)
    assert initial == derive_post_dispatch_trace_replacement_id(import_id, 0)
    assert initial != derive_post_dispatch_trace_replacement_id(import_id, 1)
    for invalid in (-1, True, 1.0, "0"):
        with pytest.raises(ValueError, match="call_sequence"):
            derive_post_dispatch_trace_replacement_id(import_id, invalid)


def test_import_replaces_inherited_surface_without_changing_owner_root(db, repository):
    parent_conversation, parent_user, parent_assistant = _completed_pair(db)
    parent_request = _post_dispatch_import(
        db,
        conversation_id=parent_conversation,
        user_message_id=parent_user,
        assistant_message_id=parent_assistant,
        call_count=1,
    )
    repository.import_post_dispatch_trace(db, parent_request)
    conversation, user, assistant = _completed_pair(db)
    with db.transaction(immediate=True) as cursor:
        boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=parent_conversation,
            included_turn_ids=(parent_user,),
        )
        owner = repository.attach_fork_owner(
            cursor, conversation_id=conversation, boundary=boundary
        )
        assert repository.get_surface_tail(cursor, owner.root_segment_id) is None
    request = _post_dispatch_import(
        db,
        conversation_id=conversation,
        user_message_id=user,
        assistant_message_id=assistant,
        call_count=1,
    )
    result = repository.import_post_dispatch_trace(db, request)
    with db.transaction() as cursor:
        assert repository.get_owner(cursor, owner.owner_id) == owner
        node = repository.get_surface_node(
            cursor, request.calls[0].request_surface[0].node_id
        )
        assert node.predecessor_node_id == boundary.inherited_surface_head_id
        projected = ConsoleTraceService(repository=repository)._surface_projection(
            cursor, result.segment_id, node
        )
        assert tuple(key[2] for _, key in projected.entries) == (
            request.user_revision_id,
        )
    assert repository.import_post_dispatch_trace(db, request).already_imported


def test_import_fails_closed_on_oversized_active_physical_span_without_pair_changes(
    db, repository
):
    from tldw_chatbook.Chat.console_trace_models import (
        TraceOmission,
        SurfaceReplacement,
    )

    conversation, user, assistant = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation,
        user_message_id=user,
        assistant_message_id=assistant,
        call_count=1,
    )
    with db.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        repository.attach_owner(
            cursor, conversation_id=conversation, root_segment_id=segment.segment_id
        )
        nodes = []
        for sequence in range(257):
            nodes.append(
                repository.append_surface_node(
                    cursor,
                    segment_id=segment.segment_id,
                    sequence=sequence,
                    predecessor_node_id=nodes[-1].node_id if nodes else None,
                    component_kind="provider_message",
                    reference=TraceOmission("provider_message", "source_unavailable"),
                )
            )
        # Only two active nodes remain, but their physical span is257.
        repository.append_surface_replacement(
            cursor,
            segment_id=segment.segment_id,
            replacement=SurfaceReplacement(
                predecessor_head_id=nodes[255].node_id,
                start_node_id=nodes[1].node_id,
                start_sequence=1,
                end_node_id=nodes[255].node_id,
                end_sequence=255,
                replacement_node_id=nodes[256].node_id,
            ),
        )
    before = tuple(db.get_connection().iterdump())
    with pytest.raises(TraceIdentityConflict, match="post_dispatch_surface_span"):
        repository.import_post_dispatch_trace(db, request)
    assert tuple(db.get_connection().iterdump()) == before
    assert db.get_message_by_id(user)["content"] == "protected user transcript"
    assert db.get_message_by_id(assistant)["content"] == "protected assistant reply"


def test_import_post_dispatch_trace_reuses_existing_attached_owner(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    with db.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        call_count=1,
    )

    result = repository.import_post_dispatch_trace(db, request)

    assert result.owner_id == owner.owner_id
    assert result.segment_id == segment.segment_id


def test_import_post_dispatch_trace_reconciles_concurrent_ordinary_owner_race(
    tmp_path: Path,
) -> None:
    path = tmp_path / "owner-race.sqlite"
    setup_db = CharactersRAGDB(path, "voice-owner-race-setup")
    try:
        conversation_id, user_message_id, assistant_message_id = _completed_pair(
            setup_db
        )
        request = _post_dispatch_import(
            setup_db,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            call_count=1,
        )
    finally:
        setup_db.close_connection()

    ordinary_ready = threading.Event()
    importer_attempted = threading.Event()
    release_ordinary = threading.Event()
    outcomes: dict[str, str] = {}
    errors: list[BaseException] = []

    def create_ordinary_owner() -> None:
        ordinary_db = CharactersRAGDB(path, "voice-owner-race-ordinary")
        ordinary_repository = ConsoleTraceRepository()
        try:
            with ordinary_db.transaction(immediate=True) as cursor:
                segment = ordinary_repository.create_segment(cursor)
                owner = ordinary_repository.attach_owner(
                    cursor,
                    conversation_id=conversation_id,
                    root_segment_id=segment.segment_id,
                )
                outcomes["ordinary_owner_id"] = owner.owner_id
                outcomes["ordinary_segment_id"] = owner.root_segment_id
                ordinary_ready.set()
                if not release_ordinary.wait(5):
                    raise RuntimeError("ordinary owner race was not released")
        except BaseException as exc:  # pragma: no cover - asserted in main thread
            errors.append(exc)
        finally:
            ordinary_db.close_connection()

    def import_promoted_trace() -> None:
        if not ordinary_ready.wait(5):
            errors.append(RuntimeError("ordinary owner was not ready"))
            return
        importer_attempted.set()
        import_db = CharactersRAGDB(path, "voice-owner-race-importer")
        try:
            imported = ConsoleTraceRepository().import_post_dispatch_trace(
                import_db, request
            )
            outcomes["import_owner_id"] = imported.owner_id
            outcomes["import_segment_id"] = imported.segment_id
        except BaseException as exc:  # pragma: no cover - asserted in main thread
            errors.append(exc)
        finally:
            import_db.close_connection()

    ordinary_thread = threading.Thread(target=create_ordinary_owner)
    importer_thread = threading.Thread(target=import_promoted_trace)
    ordinary_thread.start()
    assert ordinary_ready.wait(5)
    importer_thread.start()
    assert importer_attempted.wait(5)
    release_ordinary.set()
    ordinary_thread.join(5)
    importer_thread.join(5)

    assert not ordinary_thread.is_alive()
    assert not importer_thread.is_alive()
    assert errors == []
    assert outcomes["import_owner_id"] == outcomes["ordinary_owner_id"]
    assert outcomes["import_segment_id"] == outcomes["ordinary_segment_id"]


@pytest.mark.parametrize("bad_sequences", [(0, 2), (1,), (1, 0)])
def test_import_post_dispatch_trace_rejects_sequence_gaps_without_residue(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    bad_sequences: tuple[int, ...],
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        call_count=len(bad_sequences),
    )
    request = PostDispatchTraceImport(
        import_id=valid.import_id,
        conversation_id=valid.conversation_id,
        user_message_id=valid.user_message_id,
        user_revision_id=valid.user_revision_id,
        assistant_message_id=valid.assistant_message_id,
        assistant_revision_id=valid.assistant_revision_id,
        turn_id=valid.turn_id,
        run_id=valid.run_id,
        policy=valid.policy,
        expected_call_count=valid.expected_call_count,
        calls=tuple(
            PostDispatchTraceCall(
                call_id=call.call_id,
                idempotency_key=call.idempotency_key,
                call_sequence=sequence,
                provider_name=call.provider_name,
                model_name=call.model_name,
                route_identity=call.route_identity,
                endpoint_identity=call.endpoint_identity,
                generation_parameters_json=call.generation_parameters_json,
                adapter_defaults_json=call.adapter_defaults_json,
                response_format_json=call.response_format_json,
                reasoning_controls_json=call.reasoning_controls_json,
                dispatch_started_at=call.dispatch_started_at,
                response_started_at=call.response_started_at,
                settled_at=call.settled_at,
                usage_json=call.usage_json,
                request_surface=call.request_surface,
                response=call.response,
                sealed_payload_bytes=call.sealed_payload_bytes,
                terminal_state=call.terminal_state,
            )
            for call, sequence in zip(valid.calls, bad_sequences, strict=True)
        ),
        aggregate_payload_bytes=valid.aggregate_payload_bytes,
    )

    with pytest.raises(ValueError, match="contiguous"):
        repository.import_post_dispatch_trace(db, request)

    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM console_trace_calls WHERE idempotency_key LIKE ?",
            (f"voice-import-{valid.import_id}-%",),
        ).fetchone()[0]
        == 0
    )


def test_import_post_dispatch_trace_rejects_revision_mismatch_without_owner(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        call_count=1,
    )
    request = PostDispatchTraceImport(
        import_id=valid.import_id,
        conversation_id=valid.conversation_id,
        user_message_id=valid.user_message_id,
        user_revision_id=valid.user_revision_id,
        assistant_message_id=valid.assistant_message_id,
        assistant_revision_id=new_opaque_id(),
        turn_id=valid.turn_id,
        run_id=valid.run_id,
        policy=valid.policy,
        expected_call_count=valid.expected_call_count,
        calls=valid.calls,
        aggregate_payload_bytes=valid.aggregate_payload_bytes,
    )

    with pytest.raises(TraceIdentityConflict, match="assistant_revision"):
        repository.import_post_dispatch_trace(db, request)

    assert (
        repository.get_attached_owner_by_conversation(
            db.get_connection().cursor(), conversation_id
        )
        is None
    )


def test_import_post_dispatch_trace_rejects_turn_outside_committed_pair_lineage(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    valid = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        call_count=1,
    )

    with pytest.raises(TraceIdentityConflict, match="turn_lineage"):
        repository.import_post_dispatch_trace(
            db,
            replace(valid, turn_id=new_opaque_id()),
        )

    assert (
        repository.get_attached_owner_by_conversation(
            db.get_connection().cursor(), conversation_id
        )
        is None
    )


def test_import_post_dispatch_trace_uses_user_message_turn_for_fork_selection(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, user_message_id, assistant_message_id = _completed_pair(db)
    request = _post_dispatch_import(
        db,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        call_count=1,
    )

    imported = repository.import_post_dispatch_trace(db, request)
    call = repository.get_call(db.get_connection().cursor(), imported.call_ids[0])
    boundary = repository.capture_fork_boundary(
        db.get_connection().cursor(),
        conversation_id=conversation_id,
        included_turn_ids=(user_message_id, assistant_message_id),
    )

    assert call is not None and call.turn_id == user_message_id
    assert boundary is not None
    assert boundary.source_owner_id == imported.owner_id
    assert boundary.parent_segment_id == imported.segment_id
    assert boundary.inherited_surface_head_id == call.surface_node_id


def _owned_root(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> tuple[str, str, str]:
    conversation_id, message_id = _conversation_with_message(db)
    with db.transaction() as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
    return conversation_id, message_id, owner.owner_id


def _surface_and_header(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    conversation_id: str,
    message_id: str,
    owner_id: str,
) -> tuple[str, str, str, str, str]:
    owner = repository.get_owner(cursor, owner_id)
    assert owner is not None
    policy = repository.ensure_policy(cursor, _policy())
    revision = repository.ensure_semantic_revision(
        cursor,
        source_conversation_id=conversation_id,
        source_message_id=message_id,
        revision_sequence=0,
        normalized_role="user",
        content_kind="text",
        creation_reason="message_create",
        live_message_id=message_id,
    )
    node = repository.append_surface_node(
        cursor,
        segment_id=owner.root_segment_id,
        sequence=0,
        predecessor_node_id=None,
        component_kind="message",
        reference=SemanticRevisionRef(revision.revision_id),
    )
    repository.append_event(
        cursor,
        segment_id=owner.root_segment_id,
        sequence=0,
        event_type="surface_append",
        surface_node_id=node.node_id,
    )
    header = repository.create_or_reuse_request_header(
        cursor,
        provider_name="openai",
        model_name="gpt-test",
        route_identity="chat_completions",
        endpoint_identity="public_api",
        generation_parameters={"temperature": 0},
        adapter_defaults={},
        response_format={},
        reasoning_controls={},
        components=(),
    )
    return (
        owner.root_segment_id,
        policy.policy_id,
        revision.revision_id,
        node.node_id,
        header.header_id,
    )


def _bound_call(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    conversation_id: str,
    message_id: str,
    owner_id: str,
    idempotency_key: str,
) -> str:
    segment_id, policy_id, _revision_id, node_id, header_id = _surface_and_header(
        cursor,
        repository,
        conversation_id=conversation_id,
        message_id=message_id,
        owner_id=owner_id,
    )
    call = repository.reserve_call(
        cursor,
        owner_id=owner_id,
        segment_id=segment_id,
        turn_id="turn-1",
        run_id="run-1",
        call_sequence=0,
        idempotency_key=idempotency_key,
        policy_id=policy_id,
    )
    return repository.bind_call(
        cursor,
        call_id=call.call_id,
        surface_node_id=node_id,
        request_header_id=header_id,
        provider_name="openai",
        model_name="gpt-test",
        route_identity="chat_completions",
    ).call_id
