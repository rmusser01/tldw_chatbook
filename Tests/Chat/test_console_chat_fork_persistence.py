"""Real-SQLite coverage for atomic Console chat-fork persistence."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, replace
from io import BytesIO
import json

import pytest
from PIL import Image as PILImage

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_fork import (
    ConsoleChatForkSnapshot,
    ConsoleForkCitationLink,
    ConsoleForkConfigurationSnapshot,
    ConsoleForkProjectedAttachment,
    ConsoleForkProjectedGeneration,
    ConsoleForkProjectedMessage,
    ConsoleForkProjectedVideoTombstone,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
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


def _seed_source(db: CharactersRAGDB) -> tuple[int, tuple[dict, dict]]:
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
            "content": "Answer",
            "client_id": db.client_id,
        }
    )
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
) -> ConsoleChatForkSnapshot:
    if source_kind == "durable":
        source_version, source_rows = _seed_source(db)
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
            native_message_id="native-fork-assistant",
            persisted_message_id=persisted_ids[1],
            native_parent_id="native-fork-user",
            persisted_parent_id=persisted_ids[0],
            turn_id="fork-turn",
            visible_variant_id=None,
            role=ConsoleMessageRole.ASSISTANT,
            status="complete",
            content="Answer",
        ),
    )
    citation_links = tuple(
        ConsoleForkCitationLink(
            source_persisted_message_id=source_id,
            source_revision=source_row["version"],
            state="none",
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


def _counts(db: CharactersRAGDB) -> tuple[int, ...]:
    connection = db.get_connection()
    return tuple(
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "conversations",
            "messages",
            "message_attachments",
            "message_generation_metadata",
            "console_conversation_library_policy",
            "console_conversation_context_policy",
            "rag_message_trace_owners",
        )
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
    }
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
    assert _counts(db) == (2, 4, 0, 0, 1, 1, 0)


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
    "mutation", ("message-version", "message-body", "conversation")
)
def test_cursor_scoped_source_recheck_rejects_post_fence_races(
    tmp_path,
    mutation,
) -> None:
    db = CharactersRAGDB(tmp_path / f"race-{mutation}.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot(db)
    if mutation == "conversation":
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
    failure = RuntimeError(f"injected {failure_point}")
    if failure_point == "message":
        original_create = service.create_message
        calls = 0

        def fail_middle_message(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise failure
            return original_create(*args, **kwargs)

        monkeypatch.setattr(service, "create_message", fail_middle_message)
    else:
        targets = {
            "conversation": (service, "create_conversation"),
            "attachment": (db, "set_message_attachments"),
            "generation": (db, "set_message_generation_metadata"),
            "citation": (service, "_link_console_fork_citations"),
            "policy": (service.console_library_policy_repository, "insert"),
            "context-policy": (service.context_repository, "save_policy"),
            "project-context": (db, "set_conversation_console_project_context"),
            "leaf": (db, "set_conversation_active_leaf"),
        }
        owner, name = targets[failure_point]
        monkeypatch.setattr(
            owner,
            name,
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )

    with pytest.raises(RuntimeError, match=f"injected {failure_point}"):
        _commit(service, snapshot)

    assert _counts(db) == before
    assert db.get_conversation_by_id("fork") is None
    assert db.get_conversation_active_leaf("source") is None


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


def test_video_tombstone_persists_only_regeneration_metadata(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "video.db", client_id="fork-test")
    service = ChatPersistenceService(db)
    snapshot = _snapshot_with_generated_images(db)
    first, second = snapshot.messages
    tombstone = ConsoleForkProjectedVideoTombstone(
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
        source_image_message_id="fork-user",
        container="mp4",
    )
    snapshot = replace(
        snapshot,
        messages=(
            first,
            replace(
                second,
                content="[video unavailable] The generated video expired; regenerate to recreate it.",
                video_tombstone=tombstone,
            ),
        ),
    )

    _commit(service, snapshot)

    row = db.get_message_by_id("fork-assistant")
    assert row["image_data"] is None
    metadata = VideoGenerationMetadata.from_json(row["metadata_json"])
    assert metadata is not None
    assert metadata.name == "forked-video-native-fork-assistant"
    assert metadata.source_image_message_id == "fork-user"
    assert "path" not in row["metadata_json"]
    assert "store" not in row["metadata_json"]
