"""Real-SQLite coverage for atomic Console chat-fork persistence."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, replace
from io import BytesIO
import json

import pytest
from PIL import Image as PILImage

from Tests.Chat.test_citation_trace_repository import (
    TEST_FINGERPRINT_CODEC,
    _identity,
    _sealed_write,
)
import tldw_chatbook.Chat.chat_persistence_service as chat_persistence_service
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationPersistenceUnavailable,
    CitationTraceRepository,
)
from tldw_chatbook.Chat.console_chat_fork import (
    ConsoleChatForkSnapshot,
    ConsoleForkCitationLink,
    ConsoleForkConfigurationSnapshot,
    ConsoleForkProjectedAttachment,
    ConsoleForkProjectedGeneration,
    ConsoleForkProjectedMessage,
    ConsoleForkProjectedVideoTombstone,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_conversation_hydration import (
    console_messages_from_conversation_tree,
    hydrate_console_generation_settings,
)
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    encode_project_context_json,
)
from tldw_chatbook.Chat.console_roleplay_metadata import (
    parse_console_roleplay_context,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_speech_preferences import (
    ConsoleSpeechPreferences,
    parse_console_speech_preferences,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    new_opaque_id,
)
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem, parse_scope
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata


def _configuration() -> ConsoleForkConfigurationSnapshot:
    return ConsoleForkConfigurationSnapshot(
        workspace_id="global",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="gpt-test",
            system_prompt="Keep it concise.",
        ),
        rag_scope=RagScope(
            items=(ScopeItem("note", "7"),),
            updated_at="2026-08-27T00:00:00Z",
        ),
        context_policy_overrides=ConsoleContextPolicyOverrides(
            compaction_mode=ContextCompactionMode.OFF,
        ),
        library_policy=ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        ),
        runtime_backend="local",
        assistant_kind="generic",
        assistant_id="console",
        assistant_authority_id=None,
        persona_memory_mode=None,
        character_id=None,
        character_name=None,
        user_display_name_override="Rowan",
        character_system_template="You are {{char}}.",
        speech_preferences=ConsoleSpeechPreferences(auto_speak=True),
        project_instruction_state=ProjectInstructionControlState(
            project_instructions_enabled=True,
            working_folder_binding_id="binding-7",
            working_folder_locator_fingerprint="locator-fingerprint",
            project_instruction_notice_key=None,
        ),
    )


def _seed_source(
    db: CharactersRAGDB,
    *,
    assistant_content: str = "Answer",
) -> tuple[int, tuple[dict, dict]]:
    db.add_conversation({"id": "source", "root_id": "root", "title": "Source"})
    db.add_message(
        {
            "id": "source-user",
            "conversation_id": "source",
            "sender": "user",
            "content": "Question",
            "client_id": db.client_id,
        }
    )
    db.add_message(
        {
            "id": "source-assistant",
            "conversation_id": "source",
            "parent_message_id": "source-user",
            "sender": "assistant",
            "content": assistant_content,
            "client_id": db.client_id,
        }
    )
    db.set_conversation_active_leaf("source", "source-assistant")
    conversation = db.get_conversation_by_id("source")
    messages = (
        db.get_message_by_id("source-user"),
        db.get_message_by_id("source-assistant"),
    )
    return conversation["version"], messages


def _snapshot(
    db: CharactersRAGDB,
    *,
    source_kind: str = "durable",
    assistant_content: str = "Answer",
) -> ConsoleChatForkSnapshot:
    if source_kind == "durable":
        source_version, source_rows = _seed_source(
            db,
            assistant_content=assistant_content,
        )
        source_conversation_id = "source"
        source_boundary_id = "source-assistant"
    else:
        source_version = None
        source_rows = (
            {"id": None, "version": None},
            {"id": None, "version": None},
        )
        source_conversation_id = None
        source_boundary_id = None
    durable = source_kind != "temporary"
    persisted_ids = ("fork-user", "fork-assistant") if durable else (None, None)
    source_ids = (
        ("source-user", "source-assistant")
        if source_kind == "durable"
        else (None, None)
    )
    messages = (
        ConsoleForkProjectedMessage(
            source_native_message_id="native-source-user",
            source_persisted_message_id=source_ids[0],
            source_persisted_revision=source_rows[0]["version"],
            source_persisted_content=("Question" if source_kind == "durable" else None),
            native_message_id="native-fork-user",
            persisted_message_id=persisted_ids[0],
            native_parent_id=None,
            persisted_parent_id=None,
            turn_id="fork-turn",
            visible_variant_id=None,
            role=ConsoleMessageRole.USER,
            status="complete",
            content="Question",
        ),
        ConsoleForkProjectedMessage(
            source_native_message_id="native-source-assistant",
            source_persisted_message_id=source_ids[1],
            source_persisted_revision=source_rows[1]["version"],
            source_persisted_content=(
                assistant_content if source_kind == "durable" else None
            ),
            native_message_id="native-fork-assistant",
            persisted_message_id=persisted_ids[1],
            native_parent_id="native-fork-user",
            persisted_parent_id=persisted_ids[0],
            turn_id="fork-turn",
            visible_variant_id=None,
            role=ConsoleMessageRole.ASSISTANT,
            status="complete",
            content=assistant_content,
        ),
    )
    citation_links = tuple(
        ConsoleForkCitationLink(
            source_persisted_message_id=source_id,
            source_revision=source_row["version"],
            state="none",
            trace_id=None,
        )
        for source_id, source_row in zip(source_ids, source_rows)
        if source_id is not None
    )
    return ConsoleChatForkSnapshot(
        fork_session_id="fork-session",
        fork_conversation_id="fork" if durable else None,
        title="Forked source",
        source_session_id="source-session",
        source_conversation_id=source_conversation_id,
        source_conversation_version=source_version,
        source_active_leaf_persisted_message_id=(
            "source-assistant" if source_kind == "durable" else None
        ),
        source_boundary_persisted_message_id=source_boundary_id,
        durable=durable,
        messages=messages,
        configuration=_configuration(),
        citation_links=citation_links,
    )


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _persist_store_source(
    db: CharactersRAGDB,
    store: ConsoleChatStore,
    session_id: str,
) -> None:
    db.add_conversation({"id": "source", "root_id": "root", "title": "Source"})
    session = store._sessions[session_id]
    session.persisted_conversation_id = "source"
    nodes = tuple(store._nodes_by_session[session_id].values())
    for index, message in enumerate(nodes, start=1):
        message.persisted_message_id = f"source-message-{index}"
    for message in nodes:
        native_parent = store._native_parent_by_message[message.id]
        persisted_parent = (
            store._nodes_by_session[session_id][native_parent].persisted_message_id
            if native_parent is not None
            else None
        )
        message.parent_message_id = persisted_parent
        db.add_message(
            {
                "id": message.persisted_message_id,
                "conversation_id": "source",
                "parent_message_id": persisted_parent,
                "sender": message.role.value,
                "content": message.content,
                "client_id": db.client_id,
            }
        )
    active_native = store._active_leaf_by_session[session_id]
    db.set_conversation_active_leaf(
        "source",
        store._nodes_by_session[session_id][active_native].persisted_message_id,
    )


def _snapshot_with_generated_images(
    db: CharactersRAGDB,
) -> ConsoleChatForkSnapshot:
    snapshot = _snapshot(db)
    first, second = snapshot.messages
    attachments = tuple(
        ConsoleForkProjectedAttachment(
            owner_native_message_id=first.native_message_id,
            owner_persisted_message_id=first.persisted_message_id,
            position=position,
            data=_png_bytes(color),
            mime_type="image/png",
            display_name=f"generated-{position}.png",
        )
        for position, color in enumerate(((255, 0, 0), (0, 0, 255)))
    )
    generation = tuple(
        ConsoleForkProjectedGeneration(
            owner_native_message_id=first.native_message_id,
            owner_persisted_message_id=first.persisted_message_id,
            position=position,
            prompt=f"generated {position}",
            negative_prompt="",
            backend="openai",
            model="image-test",
            seed=position,
            style=None,
            params_json='{"size":"small"}',
        )
        for position in range(2)
    )
    return replace(
        snapshot,
        messages=(
            replace(
                first,
                attachments=attachments,
                generation_metadata=generation,
            ),
            second,
        ),
    )


def _conversation_kwargs(snapshot: ConsoleChatForkSnapshot) -> dict[str, object]:
    configuration = snapshot.configuration
    return {
        "conversation_title": snapshot.title,
        "scope_type": "global",
        "workspace_id": None,
        "system_prompt": configuration.settings.system_prompt,
        "runtime_backend": configuration.runtime_backend,
        "assistant_kind": configuration.assistant_kind,
        "assistant_id": configuration.assistant_id,
        "assistant_authority_id": configuration.assistant_authority_id,
        "persona_memory_mode": configuration.persona_memory_mode,
        "character_id": configuration.character_id,
        "character_name": configuration.character_name,
        "speech_preferences": configuration.speech_preferences,
        "thinking_history_policy": configuration.thinking_history_policy,
    }


def _commit(service: ChatPersistenceService, snapshot: ConsoleChatForkSnapshot):
    return service.fork_console_conversation_bundle(
        snapshot=snapshot,
        conversation_kwargs=_conversation_kwargs(snapshot),
        policy_candidate=snapshot.configuration.library_policy,
        project_context_json=encode_project_context_json(
            snapshot.configuration.project_instruction_state
        ),
    )


def test_durable_fork_persists_thinking_history_policy(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "fork-thinking-policy.db", client_id="fork-test")
    snapshot = _snapshot(db)
    snapshot = replace(
        snapshot,
        configuration=replace(
            snapshot.configuration,
            thinking_history_policy="exclude",
        ),
    )

    committed = _commit(ChatPersistenceService(db), snapshot)

    assert committed is not None
    assert db.get_conversation_by_id("fork")["thinking_history_policy"] == "exclude"
    db.close_connection()


@contextmanager
def _raw_semantic_corruption(db: CharactersRAGDB):
    """Narrowly authorize deliberate corruption, then restore the real guard."""

    connection = db.get_connection()
    authorization = db._semantic_mutation_authorization_for_coordinator(connection)
    connection.create_function(
        "console_semantic_mutation_authorized", 2, lambda *_args: 1
    )
    try:
        yield
    finally:
        connection.create_function(
            "console_semantic_mutation_authorized",
            2,
            authorization._sqlite_authorized,
        )


def _counts(db: CharactersRAGDB) -> tuple[int, ...]:
    connection = db.get_connection()
    table_counts = tuple(
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "conversations",
            "messages",
            "message_attachments",
            "message_generation_metadata",
            "console_conversation_library_policy",
            "console_conversation_context_policy",
            "rag_message_trace_owners",
            "console_trace_semantic_revisions",
        )
    )
    epoch = connection.execute(
        "SELECT epoch FROM console_trace_graph_epoch WHERE singleton_id = 1"
    ).fetchone()[0]
    return (*table_counts, epoch)


def _source_state(db: CharactersRAGDB) -> tuple[object, ...]:
    conversation = db.get_conversation_by_id("source")
    messages = tuple(
        tuple(
            db.get_message_by_id(message_id)[key]
            for key in ("id", "content", "version")
        )
        for message_id in ("source-user", "source-assistant")
    )
    return (
        conversation["version"],
        conversation["active_leaf_message_id"],
        messages,
    )


def _active_citation_snapshot(
    db: CharactersRAGDB,
) -> tuple[ConsoleChatForkSnapshot, CitationTraceRepository]:
    snapshot = _snapshot(db, assistant_content="Answer [S1].")
    repository = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
        identity_context=_identity(db),
        fingerprint_codec=TEST_FINGERPRINT_CODEC,
    )
    prepared = repository.prepare_write(_sealed_write())
    assistant = snapshot.messages[-1]
    with db.transaction() as cursor:
        repository.write_prepared(
            cursor,
            prepared,
            message_id=assistant.source_persisted_message_id or "",
            message_revision=assistant.source_persisted_revision or 0,
            message_body=assistant.source_persisted_content or "",
        )
    links = tuple(
        replace(
            link,
            state=(
                "active_required"
                if link.source_persisted_message_id == "source-assistant"
                else "none"
            ),
            trace_id=(
                "trace-1"
                if link.source_persisted_message_id == "source-assistant"
                else None
            ),
        )
        for link in snapshot.citation_links
    )
    return replace(snapshot, citation_links=links), repository


def _replacement_sealed_write():
    original = _sealed_write()
    source_trace = original.trace
    source_run = source_trace.evidence_runs[0]
    source_prompt = source_trace.prompt_evidence_sets[0]
    source_entry = source_prompt.entries[0]
    source_attempt = source_trace.answer_attempts[0]
    source_occurrence = source_attempt.occurrences[0]
    run = source_run.model_copy(
        update={
            "run_id": "run-2",
            "request_id": "request-2",
            "payload_ref": "run-payload-2",
        }
    )
    prompt = source_prompt.model_copy(
        update={
            "prompt_set_id": "prompt-2",
            "entries": (
                source_entry.model_copy(
                    update={
                        "run_id": "run-2",
                        "snapshot_payload_ref": "snapshot-2",
                    }
                ),
            ),
        }
    )
    attempt = source_attempt.model_copy(
        update={
            "attempt_id": "attempt-2",
            "prompt_evidence_set_id": "prompt-2",
            "answer_payload_ref": "answer-payload-2",
            "occurrences": (
                source_occurrence.model_copy(update={"occurrence_id": "occurrence-2"}),
            ),
        }
    )
    trace = source_trace.model_copy(
        update={
            "trace_id": "trace-2",
            "request_id": "request-2",
            "generation_id": "generation-2",
            "evidence_runs": (run,),
            "prompt_evidence_sets": (prompt,),
            "answer_attempts": (attempt,),
            "selected_attempt_id": "attempt-2",
        }
    )
    run_payload = original.evidence_run_payloads[0].model_copy(
        update={"payload_id": "run-payload-2", "run_id": "run-2"}
    )
    snapshot_payload = original.evidence_snapshot_payloads[0].model_copy(
        update={"payload_id": "snapshot-2"}
    )
    answer_payload = original.answer_attempt_payloads[0].model_copy(
        update={"payload_id": "answer-payload-2", "attempt_id": "attempt-2"}
    )
    return type(original)(
        trace=trace,
        evidence_run_payloads=(run_payload,),
        evidence_snapshot_payloads=(snapshot_payload,),
        answer_attempt_payloads=(answer_payload,),
    )


@pytest.mark.parametrize("original_state", ("deleted", "body_mismatch"))
def test_fork_rejects_replacement_trace_after_exact_owner_confirmation(
    tmp_path,
    original_state,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"citation-switch-{original_state}.db",
        client_id="fork-test",
    )
    snapshot, repository = _active_citation_snapshot(db)
    service = ChatPersistenceService(db, citation_repository=repository)
    assistant = snapshot.messages[-1]
    replacement = repository.prepare_write(_replacement_sealed_write())
    with db.transaction() as cursor:
        cursor.execute(
            """
            UPDATE rag_message_trace_owners
            SET state = ?
            WHERE message_id = 'source-assistant' AND trace_id = 'trace-1'
            """,
            (original_state,),
        )
        repository.write_prepared(
            cursor,
            replacement,
            message_id="source-assistant",
            message_revision=assistant.source_persisted_revision or 0,
            message_body=assistant.source_persisted_content or "",
        )

    with pytest.raises(CitationPersistenceUnavailable):
        _commit(service, snapshot)

    assert db.get_conversation_by_id("fork") is None
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM rag_message_trace_owners WHERE message_id = ?",
            ("fork-assistant",),
        )
        .fetchone()[0]
        == 0
    )


def test_snapshot_carries_the_frozen_source_conversation_version() -> None:
    assert "source_conversation_version" in {
        field.name for field in fields(ConsoleChatForkSnapshot)
    }


def test_durable_fork_commits_ancestry_lineage_policy_and_context(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "fork.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)

    result = _commit(service, snapshot)

    conversation = db.get_conversation_by_id("fork")
    assert result.already_committed is False
    assert result.conversation_id == "fork"
    assert result.message_id_map == {
        "native-fork-user": "fork-user",
        "native-fork-assistant": "fork-assistant",
    }
    assert conversation["root_id"] == "root"
    assert conversation["parent_conversation_id"] == "source"
    assert conversation["forked_from_message_id"] == "source-assistant"
    assert conversation["scope_type"] == "global"
    assert conversation["workspace_id"] is None
    assert conversation["system_prompt"] == "Keep it concise."
    assert conversation["runtime_backend"] == "local"
    assert conversation["assistant_kind"] == "generic"
    assert conversation["assistant_id"] == "console"
    assert conversation["persona_memory_mode"] is None
    assert db.get_conversation_active_leaf("fork") == "fork-assistant"
    rows = db.get_messages_for_conversation("fork", limit=10)
    assert [row["id"] for row in rows] == ["fork-user", "fork-assistant"]
    assert rows[1]["parent_message_id"] == "fork-user"
    assert service.get_conversation_context_policy("fork").overrides == (
        snapshot.configuration.context_policy_overrides
    )
    assert db.get_conversation_console_project_context("fork") == (
        encode_project_context_json(snapshot.configuration.project_instruction_state)
    )
    metadata = json.loads(conversation["metadata"])
    assert metadata["console_session_settings"] == {
        "version": 1,
        "provider": "openai",
        "model": "gpt-test",
        "base_url": None,
        "temperature": 0.7,
        "top_p": 0.95,
        "min_p": None,
        "top_k": None,
        "max_tokens": None,
        "seed": None,
        "presence_penalty": None,
        "frequency_penalty": None,
        "reasoning_effort": None,
        "reasoning_summary": None,
        "verbosity": None,
        "thinking_effort": None,
        "thinking_budget_tokens": None,
        "streaming": True,
        "character_label": "",
        "system_prompt": "Keep it concise.",
        "source": "derived",
        "pinned_prefill": None,
        "persona_memory_mode": None,
    }
    restored_settings = hydrate_console_generation_settings({}, conversation).settings
    assert restored_settings == replace(snapshot.configuration.settings, source="user")
    assert parse_scope(metadata["rag_scope"]) == snapshot.configuration.rag_scope
    roleplay = parse_console_roleplay_context(metadata)
    assert roleplay.user_name_override == "Rowan"
    assert roleplay.character_system_template == "You are {{char}}."
    assert parse_console_speech_preferences(metadata) == (
        snapshot.configuration.speech_preferences
    )
    policy = service.console_library_policy_repository.read("fork").durable_policy
    assert policy is not None
    assert policy.policy_revision == 1
    assert db.get_conversation_by_id("source")["version"] == (
        snapshot.source_conversation_version
    )


def test_durable_fork_commits_shared_trace_prefix_without_payload_copy(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "fork-trace.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    repository = service.console_trace_repository
    with db.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id="source",
            root_segment_id=segment.segment_id,
        )
        policy = repository.ensure_policy(
            cursor,
            FrozenTracePolicy(
                policy_id=new_opaque_id(),
                credential_filter_version="cred-v1",
                pii_redaction_enabled=False,
                pii_ruleset_revision_id=None,
            ),
        )
        revision = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id="source",
            source_message_id="source-user",
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="message_create",
            live_message_id="source-user",
        )
        node = repository.append_surface_node(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="message",
            reference=SemanticRevisionRef(revision.revision_id),
        )
        repository.append_event(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            event_type="surface_append",
            surface_node_id=node.node_id,
        )
        call = repository.reserve_call(
            cursor,
            owner_id=owner.owner_id,
            segment_id=segment.segment_id,
            turn_id="fork-turn",
            run_id="run-1",
            call_sequence=0,
            idempotency_key="source-fork-call",
            policy_id=policy.policy_id,
        )
        repository.append_event(
            cursor,
            segment_id=segment.segment_id,
            sequence=1,
            event_type="call_boundary",
            call_id=call.call_id,
        )
        boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id="source",
            included_turn_ids=("fork-turn",),
        )
        assert boundary is not None
        payload_counts = {
            table: cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "console_trace_artifacts",
                "console_trace_surface_nodes",
                "console_trace_events",
                "console_trace_calls",
            )
        }

    result = _commit(service, replace(snapshot, trace_boundary=boundary))

    assert result is not None
    with db.transaction() as cursor:
        assert {
            table: cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in payload_counts
        } == payload_counts
        child_calls = repository.read_conversation_call_lineage(cursor, "fork")
        assert [item.call_id for item in child_calls] == [call.call_id]
        child_owner_row = cursor.execute(
            "SELECT root_segment_id FROM console_trace_owners WHERE conversation_id = ?",
            ("fork",),
        ).fetchone()
        assert child_owner_row is not None
        child_segment = repository.get_segment(cursor, child_owner_row[0])
        assert child_segment is not None
        assert child_segment.parent_segment_id == segment.segment_id
        assert child_segment.inherited_through_sequence == 1
    db.close_connection()


def test_same_target_retry_and_resolver_return_the_preallocated_identity(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "retry.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    first = _commit(service, snapshot)

    second = _commit(service, snapshot)
    resolved = service.resolve_console_fork_commit(snapshot)

    assert first.already_committed is False
    assert second.already_committed is True
    assert resolved is not None
    assert resolved.already_committed is True
    assert resolved.message_id_map == first.message_id_map
    assert _counts(db) == (2, 4, 0, 0, 1, 1, 0, 4, 4)


def test_exception_after_commit_is_resolved_to_the_same_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    db = CharactersRAGDB(tmp_path / "after-commit.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    original_transaction = db.transaction
    outer_open = False

    @contextmanager
    def raise_after_outer_commit(*, immediate=False):
        nonlocal outer_open
        is_outer_fork = immediate and not outer_open
        if is_outer_fork:
            outer_open = True
        try:
            with original_transaction(immediate=immediate) as cursor:
                yield cursor
        finally:
            if is_outer_fork:
                outer_open = False
        if is_outer_fork:
            raise RuntimeError("connection lost after commit")

    monkeypatch.setattr(db, "transaction", raise_after_outer_commit)

    with pytest.raises(RuntimeError, match="after commit"):
        _commit(service, snapshot)

    monkeypatch.setattr(db, "transaction", original_transaction)
    result = service.resolve_console_fork_commit(snapshot)
    assert result is not None
    assert result.already_committed is True
    assert result.message_id_map == {
        "native-fork-user": "fork-user",
        "native-fork-assistant": "fork-assistant",
    }


def test_resolver_distinguishes_absence_from_a_conflicting_target(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "resolve.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)

    assert service.resolve_console_fork_commit(snapshot) is None
    db.add_conversation({"id": "fork", "title": "Someone else's chat"})

    with pytest.raises(RuntimeError, match="collision"):
        service.resolve_console_fork_commit(snapshot)


@pytest.mark.parametrize(
    ("column", "changed_value"),
    (
        ("root_id", "fork"),
        ("parent_conversation_id", None),
        ("forked_from_message_id", "source-user"),
        ("title", "Different title"),
        ("active_leaf_message_id", "fork-user"),
    ),
)
def test_resolver_rejects_each_conflicting_fork_identity_field(
    tmp_path,
    column,
    changed_value,
) -> None:
    db = CharactersRAGDB(tmp_path / f"collision-{column}.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    _commit(service, snapshot)
    with db.transaction() as cursor:
        cursor.execute(
            f"UPDATE conversations SET {column} = ? WHERE id = 'fork'",
            (changed_value,),
        )

    with pytest.raises(RuntimeError, match="collision"):
        service.resolve_console_fork_commit(snapshot)


@pytest.mark.parametrize(
    "mutation",
    ("message-version", "message-body", "conversation", "active-leaf"),
)
def test_cursor_scoped_source_recheck_rejects_post_fence_races(
    tmp_path,
    mutation,
) -> None:
    db = CharactersRAGDB(tmp_path / f"race-{mutation}.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    if mutation == "active-leaf":
        before = db.get_conversation_by_id("source")["version"]
        db.set_conversation_active_leaf("source", "source-user")
        assert db.get_conversation_by_id("source")["version"] == before
    elif mutation == "conversation":
        source = db.get_conversation_by_id("source")
        db.update_conversation(
            "source", {"title": "Changed"}, expected_version=source["version"]
        )
    else:
        source = db.get_message_by_id("source-user")
        content = "Question" if mutation == "message-version" else "Changed"
        db.update_message(
            "source-user",
            {"content": content},
            expected_version=source["version"],
        )

    with pytest.raises(RuntimeError, match="source changed"):
        _commit(service, snapshot)

    assert db.get_conversation_by_id("fork") is None


def _snapshot_with_post_boundary_active_leaf(
    db: CharactersRAGDB,
    *,
    tail_length: int = 1,
) -> ConsoleChatForkSnapshot:
    snapshot = _snapshot(db)
    parent_id = "source-assistant"
    for index in range(tail_length):
        message_id = f"source-tail-{index}"
        db.add_message(
            {
                "id": message_id,
                "conversation_id": "source",
                "parent_message_id": parent_id,
                "sender": "user" if index % 2 == 0 else "assistant",
                "content": f"Post-boundary {index}",
                "client_id": db.client_id,
            }
        )
        parent_id = message_id
    db.set_conversation_active_leaf("source", parent_id)
    return replace(
        snapshot,
        source_conversation_version=db.get_conversation_by_id("source")["version"],
        source_active_leaf_persisted_message_id=parent_id,
    )


@pytest.mark.parametrize(
    "mutation",
    ("reparent", "missing", "deleted", "cross-conversation", "cycle"),
)
def test_active_leaf_lineage_recheck_rejects_post_fence_corruption(
    tmp_path,
    mutation,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"active-lineage-{mutation}.db",
        client_id="fork-test",
    )
    service = ChatPersistenceService(db)
    snapshot = _snapshot_with_post_boundary_active_leaf(db)
    if mutation == "cross-conversation":
        db.add_conversation({"id": "other", "root_id": "other-root", "title": "Other"})
    source_before = db.get_conversation_by_id("source")
    if mutation == "missing":
        connection = db.get_connection()
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            with _raw_semantic_corruption(db):
                connection.execute(
                    "DELETE FROM messages WHERE id = ?", ("source-tail-0",)
                )
                connection.commit()
        finally:
            connection.execute("PRAGMA foreign_keys = ON")
    else:
        with _raw_semantic_corruption(db), db.transaction() as cursor:
            if mutation == "reparent":
                cursor.execute(
                    "UPDATE messages SET parent_message_id = ? WHERE id = ?",
                    ("source-user", "source-tail-0"),
                )
            elif mutation == "deleted":
                cursor.execute(
                    "UPDATE messages SET deleted = 1 WHERE id = ?",
                    ("source-tail-0",),
                )
            elif mutation == "cross-conversation":
                cursor.execute(
                    "UPDATE messages SET conversation_id = ? WHERE id = ?",
                    ("other", "source-tail-0"),
                )
            else:
                cursor.execute(
                    "UPDATE messages SET parent_message_id = id WHERE id = ?",
                    ("source-tail-0",),
                )

    with pytest.raises(RuntimeError, match="source changed"):
        _commit(service, snapshot)

    source_after = db.get_conversation_by_id("source")
    assert source_after["version"] == source_before["version"]
    assert source_after["active_leaf_message_id"] == "source-tail-0"
    assert db.get_conversation_by_id("fork") is None


def test_active_leaf_lineage_recheck_is_bounded(tmp_path, monkeypatch) -> None:
    db = CharactersRAGDB(tmp_path / "active-lineage-depth.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot_with_post_boundary_active_leaf(db, tail_length=3)
    monkeypatch.setattr(
        chat_persistence_service,
        "CONSOLE_FORK_SOURCE_LINEAGE_MAX_DEPTH",
        2,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="source changed"):
        _commit(service, snapshot)

    assert db.get_conversation_by_id("fork") is None


@pytest.mark.parametrize("projection", ("visible-variant", "video-tombstone"))
def test_cursor_scoped_source_recheck_uses_exact_persisted_body_for_projections(
    tmp_path,
    projection,
) -> None:
    db = CharactersRAGDB(tmp_path / f"race-{projection}.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    first, second = snapshot.messages
    if projection == "visible-variant":
        second = replace(
            second,
            visible_variant_id="variant-2",
            content="A visible session-only answer",
        )
    else:
        second = replace(
            second,
            content=(
                "[video unavailable] The generated video expired; "
                "regenerate to recreate it."
            ),
            video_tombstone=ConsoleForkProjectedVideoTombstone(
                owner_native_message_id=second.native_message_id,
                owner_persisted_message_id=second.persisted_message_id,
                source_fingerprint="a" * 64,
                prompt="animate",
                negative_prompt="",
                backend="minimax",
                model="video-test",
                seed=7,
                duration_seconds=3.0,
                fps=24.0,
                width=640,
                height=360,
                ratio="16:9",
                source_image_message_id=None,
                container="mp4",
            ),
        )
    snapshot = replace(snapshot, messages=(first, second))
    before_version = db.get_message_by_id("source-assistant")["version"]
    with _raw_semantic_corruption(db), db.transaction() as cursor:
        cursor.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("Tampered without version", "source-assistant"),
        )
    assert db.get_message_by_id("source-assistant")["version"] == before_version

    with pytest.raises(RuntimeError, match="source changed"):
        _commit(service, snapshot)

    assert db.get_conversation_by_id("fork") is None


@pytest.mark.parametrize(
    ("field", "forged"),
    (
        ("conversation_title", "Forged title"),
        ("scope_type", "workspace"),
        ("workspace_id", "workspace-elsewhere"),
        ("system_prompt", "Forged system prompt"),
        ("runtime_backend", "remote"),
        ("assistant_kind", "character"),
        ("assistant_id", "forged-assistant"),
        ("assistant_authority_id", "forged-authority"),
        ("persona_memory_mode", "forged-memory"),
        ("character_id", "forged-character"),
        ("character_name", "Forged Character"),
        ("speech_preferences", ConsoleSpeechPreferences(auto_speak=False)),
    ),
)
def test_caller_cannot_override_snapshot_bound_configuration(
    tmp_path,
    field,
    forged,
) -> None:
    db = CharactersRAGDB(tmp_path / f"forged-{field}.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    kwargs = _conversation_kwargs(snapshot)
    kwargs[field] = forged

    with pytest.raises(ValueError, match="configuration"):
        service.fork_console_conversation_bundle(
            snapshot=snapshot,
            conversation_kwargs=kwargs,
            policy_candidate=snapshot.configuration.library_policy,
            project_context_json=encode_project_context_json(
                snapshot.configuration.project_instruction_state
            ),
        )

    assert db.get_conversation_by_id("fork") is None


@pytest.mark.parametrize(
    "failure_point",
    (
        "conversation",
        "message",
        "attachment",
        "generation",
        "citation",
        "policy",
        "context-policy",
        "project-context",
        "leaf",
    ),
)
def test_each_atomic_write_failure_rolls_back_the_complete_target(
    tmp_path,
    monkeypatch,
    failure_point,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"rollback-{failure_point}.db", client_id="fork-test"
    )
    service = ChatPersistenceService(db)
    snapshot = (
        _snapshot_with_generated_images(db)
        if failure_point in {"attachment", "generation"}
        else _snapshot(db)
    )
    before = _counts(db)
    source_before = _source_state(db)
    failure = RuntimeError(f"injected {failure_point}")
    if failure_point == "message":
        original_create = service.create_message
        calls = 0

        def fail_middle_message(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original_create(*args, **kwargs)
            if calls == 2:
                raise failure
            return result

        monkeypatch.setattr(service, "create_message", fail_middle_message)
    else:
        targets = {
            "conversation": (service, "create_conversation"),
            "attachment": (db, "_set_message_attachments_uncoordinated"),
            "generation": (db, "set_message_generation_metadata"),
            "citation": (service, "_link_console_fork_citations"),
            "policy": (service.console_library_policy_repository, "insert"),
            "context-policy": (service.context_repository, "save_policy"),
            "project-context": (db, "set_conversation_console_project_context"),
            "leaf": (db, "set_conversation_active_leaf"),
        }
        owner, name = targets[failure_point]
        original_write = getattr(owner, name)

        def fail_after_write(*args, **kwargs):
            original_write(*args, **kwargs)
            raise failure

        monkeypatch.setattr(owner, name, fail_after_write)

    with pytest.raises(RuntimeError, match=f"injected {failure_point}"):
        _commit(service, snapshot)

    assert _counts(db) == before
    assert _source_state(db) == source_before
    assert db.get_conversation_by_id("fork") is None
    assert db.get_conversation_active_leaf("source") == "source-assistant"


@pytest.mark.parametrize(
    "revocation",
    ("evidence-run", "snapshot", "answer", "tombstone"),
)
def test_citation_payload_revocation_rolls_back_the_integrated_fork_bundle(
    tmp_path,
    revocation,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"citation-race-{revocation}.db",
        client_id="fork-test",
    )
    snapshot, repository = _active_citation_snapshot(db)
    service = ChatPersistenceService(db, citation_repository=repository)
    with db.transaction() as cursor:
        if revocation == "evidence-run":
            cursor.execute(
                """
                UPDATE rag_evidence_runs
                SET redaction_state = 'purged', run_payload_json = NULL,
                    purged_at = '2026-08-27T00:00:00+00:00'
                WHERE trace_id = 'trace-1'
                """
            )
        elif revocation == "snapshot":
            cursor.execute(
                """
                UPDATE rag_evidence_snapshots
                SET redaction_state = 'redacted'
                WHERE payload_id = 'snapshot-1'
                """
            )
        elif revocation == "answer":
            cursor.execute(
                """
                UPDATE rag_answer_attempt_payloads
                SET redaction_state = 'purged', answer_body = NULL,
                    body_integrity_hmac = NULL,
                    purged_at = '2026-08-27T00:00:00+00:00'
                WHERE trace_id = 'trace-1'
                """
            )
        else:
            cursor.execute(
                """
                INSERT INTO rag_payload_tombstones VALUES (
                    ?, 'local_payload_v1', 'snapshot-1', 'snapshot-1',
                    'revoked', 'fork-race-policy',
                    '2026-08-27T00:00:00+00:00',
                    '2027-08-27T00:00:00+00:00'
                )
                """,
                (_identity(db).profile_id,),
            )
    before = _counts(db)
    source_before = _source_state(db)

    with pytest.raises(CitationPersistenceUnavailable):
        _commit(service, snapshot)

    assert _counts(db) == before
    assert _source_state(db) == source_before
    assert db.get_conversation_by_id("fork") is None
    owners = (
        db.get_connection()
        .execute("SELECT message_id FROM rag_message_trace_owners ORDER BY message_id")
        .fetchall()
    )
    assert [row["message_id"] for row in owners] == ["source-assistant"]


def test_failure_after_real_citation_owner_link_rolls_back_the_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    db = CharactersRAGDB(tmp_path / "citation-after-link.db", client_id="fork-test")
    snapshot, repository = _active_citation_snapshot(db)
    service = ChatPersistenceService(db, citation_repository=repository)
    before = _counts(db)
    source_before = _source_state(db)
    original_link = repository.link_fork_message_owner

    def fail_after_owner_link(*args, **kwargs):
        original_link(*args, **kwargs)
        raise RuntimeError("injected after citation owner link")

    monkeypatch.setattr(repository, "link_fork_message_owner", fail_after_owner_link)

    with pytest.raises(RuntimeError, match="after citation owner link"):
        _commit(service, snapshot)

    assert _counts(db) == before
    assert _source_state(db) == source_before
    assert db.get_conversation_by_id("fork") is None
    owners = (
        db.get_connection()
        .execute("SELECT message_id FROM rag_message_trace_owners ORDER BY message_id")
        .fetchall()
    )
    assert [row["message_id"] for row in owners] == ["source-assistant"]


@pytest.mark.parametrize(
    ("source_kind", "expected_writes", "expected_root"),
    (("unsaved", 1, "fork"), ("temporary", 0, None)),
)
def test_unsaved_and_temporary_sources_keep_their_distinct_durability(
    tmp_path,
    source_kind,
    expected_writes,
    expected_root,
) -> None:
    db = CharactersRAGDB(tmp_path / f"{source_kind}.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db, source_kind=source_kind)

    result = _commit(service, snapshot)

    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        == expected_writes
    )
    if source_kind == "temporary":
        assert result is None
    else:
        row = db.get_conversation_by_id("fork")
        assert row["root_id"] == expected_root
        assert row["parent_conversation_id"] is None
        assert row["forked_from_message_id"] is None


def test_missing_required_attachment_is_rejected_before_the_transaction(
    tmp_path,
    monkeypatch,
) -> None:
    db = CharactersRAGDB(tmp_path / "bad-media.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    first, second = snapshot.messages
    corrupt = ConsoleForkProjectedAttachment(
        owner_native_message_id=first.native_message_id,
        owner_persisted_message_id=first.persisted_message_id,
        position=0,
        data=b"not-an-image",
        mime_type="image/png",
        display_name="broken.png",
    )
    snapshot = replace(
        snapshot, messages=(replace(first, attachments=(corrupt,)), second)
    )
    original_transaction = db.transaction
    immediate_entries = 0

    @contextmanager
    def record_transaction(*, immediate=False):
        nonlocal immediate_entries
        immediate_entries += int(immediate)
        with original_transaction(immediate=immediate) as cursor:
            yield cursor

    monkeypatch.setattr(db, "transaction", record_transaction)

    with pytest.raises(ValueError, match="image"):
        _commit(service, snapshot)

    assert immediate_entries == 0
    assert db.get_conversation_by_id("fork") is None


def test_missing_durable_citation_state_is_rejected_before_the_transaction(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "missing-citation-state.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = replace(_snapshot(db), citation_links=())

    with pytest.raises(ValueError, match="citation states"):
        _commit(service, snapshot)

    assert db.get_conversation_by_id("fork") is None


@pytest.mark.parametrize(
    ("state", "trace_id"),
    (("active_required", None), ("none", "trace-1")),
)
def test_citation_state_and_frozen_trace_identity_must_match(
    tmp_path,
    state,
    trace_id,
) -> None:
    db = CharactersRAGDB(tmp_path / "citation-trace-shape.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    first, *remaining = snapshot.citation_links
    snapshot = replace(
        snapshot,
        citation_links=(replace(first, state=state, trace_id=trace_id), *remaining),
    )

    with pytest.raises(ValueError, match="citation states"):
        _commit(service, snapshot)

    assert db.get_conversation_by_id("fork") is None


def test_temporary_fork_rejects_governed_citation_identity(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "temporary-citation.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db, source_kind="temporary")
    first, second = snapshot.messages
    snapshot = replace(
        snapshot,
        messages=(
            replace(
                first,
                source_persisted_message_id="source-user",
                source_persisted_revision=1,
            ),
            second,
        ),
        citation_links=(
            ConsoleForkCitationLink(
                source_persisted_message_id="source-user",
                source_revision=1,
                state="none",
                trace_id=None,
            ),
        ),
    )

    with pytest.raises(ValueError, match="Temporary fork citation"):
        _commit(service, snapshot)


def test_corrupt_generated_image_metadata_is_rejected_before_the_transaction(
    tmp_path,
    monkeypatch,
) -> None:
    db = CharactersRAGDB(tmp_path / "bad-generation.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot_with_generated_images(db)
    first, second = snapshot.messages
    generation = (
        replace(first.generation_metadata[0], backend=""),
        first.generation_metadata[1],
    )
    snapshot = replace(
        snapshot,
        messages=(replace(first, generation_metadata=generation), second),
    )
    original_transaction = db.transaction
    immediate_entries = 0

    @contextmanager
    def record_transaction(*, immediate=False):
        nonlocal immediate_entries
        immediate_entries += int(immediate)
        with original_transaction(immediate=immediate) as cursor:
            yield cursor

    monkeypatch.setattr(db, "transaction", record_transaction)

    with pytest.raises(ValueError, match="generation metadata"):
        _commit(service, snapshot)

    assert immediate_entries == 0
    assert db.get_conversation_by_id("fork") is None


def test_generated_image_sidecars_round_trip_in_the_atomic_bundle(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "generated-image.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot_with_generated_images(db)

    _commit(service, snapshot)

    row = db.get_message_by_id("fork-user")
    assert row["image_data"] == snapshot.messages[0].attachments[0].data
    assert row["image_mime_type"] == "image/png"
    extras = db.get_attachments_for_messages(("fork-user",))["fork-user"]
    assert extras[0]["data"] == snapshot.messages[0].attachments[1].data
    generation = db.get_generation_metadata_for_messages(("fork-user",))["fork-user"]
    assert [item["prompt"] for item in generation] == ["generated 0", "generated 1"]


@pytest.mark.parametrize("status", ("stopped", "failed"))
def test_durable_reload_preserves_terminal_status_and_position_zero_label(
    tmp_path,
    status,
) -> None:
    db = CharactersRAGDB(tmp_path / f"reload-{status}.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    first, second = snapshot.messages
    attachment = ConsoleForkProjectedAttachment(
        owner_native_message_id=second.native_message_id,
        owner_persisted_message_id=second.persisted_message_id,
        position=0,
        data=_png_bytes((0, 255, 0)),
        mime_type="image/png",
        display_name="kept-position-zero.png",
    )
    snapshot = replace(
        snapshot,
        messages=(
            first,
            replace(second, status=status, attachments=(attachment,)),
        ),
    )

    _commit(service, snapshot)
    tree = ChatConversationService(db).get_conversation_tree("fork")
    hydrated = console_messages_from_conversation_tree(tree, db=db)
    restored = hydrated[-1]

    assert restored.status == status
    assert restored.attachments[0].display_name == "kept-position-zero.png"
    assert restored.attachment_label == "kept-position-zero.png"


@pytest.mark.parametrize("source_inside_snapshot", (True, False))
@pytest.mark.asyncio
async def test_video_reference_round_trips_from_store_projection(
    tmp_path,
    source_inside_snapshot,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"video-{source_inside_snapshot}.db",
        client_id="fork-test",
    )
    service = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=service)
    session = store.create_session(
        title="Video source",
        settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
    )
    if source_inside_snapshot:
        source_image = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Animate this image",
            attachments=(
                MessageAttachment(
                    _png_bytes((0, 255, 0)),
                    "image/png",
                    "source.png",
                    0,
                ),
            ),
        )
    else:
        root = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Choose a branch",
        )
        source_image = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Excluded image branch",
            attachments=(
                MessageAttachment(
                    _png_bytes((0, 255, 0)),
                    "image/png",
                    "excluded-source.png",
                    0,
                ),
            ),
        )
        store.create_sibling(
            source_image.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Selected branch",
        )
        assert root.id in store.active_path_message_ids(session.id)
    video = store.append_video_message(
        session.id,
        video_metadata=VideoGenerationMetadata(
            name="source-video-key",
            prompt="animate",
            backend="minimax",
            model="video-test",
            source_image_message_id=source_image.id,
        ),
    )
    _persist_store_source(db, store, session.id)

    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(video.id),
        title="Forked video",
        fork_session_id="fork-session",
        fork_conversation_id="fork",
    )
    projected_video = snapshot.messages[-1]
    assert projected_video.video_tombstone is not None
    projected_source = next(
        (
            message
            for message in snapshot.messages
            if message.source_native_message_id == source_image.id
        ),
        None,
    )
    expected_source_id = (
        projected_source.persisted_message_id if projected_source is not None else None
    )
    assert projected_video.video_tombstone.source_image_message_id == expected_source_id

    _commit(service, snapshot)

    tree = ChatConversationService(db).get_conversation_tree("fork")
    hydrated = console_messages_from_conversation_tree(tree, db=db)
    restored_video = next(message for message in hydrated if message.video_metadata)
    assert restored_video.video_metadata is not None
    assert restored_video.video_metadata.source_image_message_id == expected_source_id
    if projected_source is not None:
        restored_source = next(
            message
            for message in hydrated
            if message.persisted_message_id == projected_source.persisted_message_id
        )
        assert restored_source.attachments
        assert (
            restored_video.video_metadata.source_image_message_id
            == restored_source.persisted_message_id
        )

    row = db.get_message_by_id(projected_video.persisted_message_id)
    assert row["image_data"] is None
    metadata = VideoGenerationMetadata.from_json(row["metadata_json"])
    assert metadata is not None
    assert metadata.name == f"forked-video-{projected_video.native_message_id}"
    assert metadata.source_image_message_id == expected_source_id
    assert metadata.is_unavailable_tombstone is True
    assert "path" not in row["metadata_json"]
    assert "store" not in row["metadata_json"]

    resumed_store = ConsoleChatStore(persistence=service)
    resumed = resumed_store.restore_persisted_session(
        title="Forked video",
        workspace_id=None,
        persisted_conversation_id="fork",
        all_nodes=hydrated,
        active_leaf_persisted_id=db.get_conversation_active_leaf("fork"),
        settings=snapshot.configuration.settings,
        activate=False,
    )
    await resumed_store.hydrate_session_library_policy(resumed.id)
    resumed_video = next(
        message
        for message in resumed_store.messages_for_session(resumed.id)
        if message.video_metadata is not None
    )
    refork = resumed_store.stage_fork_snapshot(
        resumed_store.issue_fork_fence(resumed_video.id),
        title="Forked again",
        fork_session_id=f"refork-session-{source_inside_snapshot}",
        fork_conversation_id=f"refork-{source_inside_snapshot}",
    )
    assert refork.messages[-1].video_tombstone is not None
