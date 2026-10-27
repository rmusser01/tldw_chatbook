"""Captured C2 settings preserve native meaning and remain effect-free."""

import json
from contextlib import closing
from dataclasses import asdict

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import _seed_core, content_reads


def _seed_settings(connection, purpose):
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
        encode_project_context_json,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    _seed_core(connection, purpose)
    settings = ConsoleSessionSettings(
        provider="openai",
        model="history-model",
        temperature=0.4,
        system_prompt="Exact system note-a",
        source="user",
        pinned_prefill="Exact prefix",
    )
    metadata = {
        "console_session_settings": {"version": 1, **asdict(settings)},
        "active_dictionaries": [11],
        "pinned_response_prefill": "Exact prefix",
    }
    connection.execute(
        "UPDATE conversations SET metadata=?,thinking_history_policy='exclude',"
        "console_project_context_json=?,active_leaf_message_id='graph-variant',"
        "system_prompt='Exact system note-a',topic_last_tagged_message_id='graph-user' "
        "WHERE id='conversation'",
        (
            json.dumps(metadata),
            encode_project_context_json(ProjectInstructionControlState(False)),
        ),
    )
    connection.execute(
        "UPDATE messages SET usage_json=?,metadata_json=?,assistant_generation_state='complete' "
        "WHERE id='graph-answer'",
        (
            ProviderUsage(
                uncached_input=12, output=5, provider="openai", model="history-model"
            ).to_json(),
            MessageMetadata(origin="hook", root_fork=False).to_json(),
        ),
    )
    connection.execute(
        "INSERT INTO console_conversation_capture_policy VALUES ('conversation','safe',0,1,'2026-10-25')"
    )
    connection.execute(
        "INSERT INTO console_conversation_library_policy VALUES ('conversation',1,0,0,3,'2026-10-25')"
    )


def test_captured_settings_complete_native_fields_without_activation(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.console_capture_policy_repository import (
        ConsoleCapturePolicyRepository,
    )
    from tldw_chatbook.Chat.console_library_policy_repository import (
        ConsoleLibraryPolicyRepository,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        parse_persisted_console_session_settings,
    )
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.Chat.provider_usage import ProviderUsage
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    with content_reads(tmp_path, monkeypatch, _seed_settings) as reads:
        build = getattr(reads, "content_settings_plan", None)
        assert callable(build), "captured native reader needs C2 settings planning"
        plan = build()
        assert not plan.blockers
        cid = plan.identities.mapped("conversation", "conversation")
        mid = plan.identities.mapped("message", "graph-answer")
        conversation = next(
            dict(row.values)
            for row in plan.rows
            if row.table == "conversations" and row.original == ("conversation",)
        )
        message = next(
            dict(row.values)
            for row in plan.rows
            if row.table == "messages" and row.original == ("graph-answer",)
        )
        assert conversation["topic_last_tagged_message_id"] == plan.identities.mapped(
            "message", "graph-user"
        )
        assert conversation["thinking_history_policy"] == "exclude"
        assert conversation["system_prompt"] == "Exact system note-a"
        assert (
            parse_persisted_console_session_settings(
                conversation["metadata"]
            ).temperature
            == 0.4
        )
        assert json.loads(conversation["metadata"])["active_dictionaries"] == [
            plan.identities.mapped("dictionary", 11)
        ]
        assert MessageMetadata.from_json(message["metadata_json"]).origin == "hook"
        assert ProviderUsage.from_json(message["usage_json"]).total_tokens == 17
        assert not any(
            row.pending_columns
            for row in plan.rows
            if row.table in {"conversations", "messages"}
        )
        with closing(CharactersRAGDB(":memory:", "settings-consumer")) as db:
            db.add_conversation(
                {
                    "id": cid,
                    "metadata": conversation["metadata"],
                    "thinking_history_policy": conversation["thinking_history_policy"],
                }
            )
            with db.transaction() as cursor:
                for row in plan.rows:
                    if row.table in {
                        "console_conversation_capture_policy",
                        "console_conversation_library_policy",
                    }:
                        data = dict(row.values)
                        cursor.execute(
                            'INSERT INTO "'
                            + row.table
                            + '" ('
                            + ",".join(data)
                            + ") VALUES ("
                            + ",".join("?" for _ in data)
                            + ")",
                            tuple(data.values()),
                        )
            assert (
                ConsoleCapturePolicyRepository(db).read(cid).policy.capture_enabled
                is False
            )
            assert (
                ConsoleLibraryPolicyRepository(db)
                .read(cid)
                .durable_policy.policy_revision
                == 3
            )
            assert (
                db.get_conversation_by_id(cid)["thinking_history_policy"] == "exclude"
            )
            derived = (
                db.get_connection()
                .execute(
                    "SELECT dictionary_id FROM conversation_dictionary_attachments WHERE conversation_id=?",
                    (cid,),
                )
                .fetchall()
            )
            assert [row[0] for row in derived] == [
                plan.identities.mapped("dictionary", 11)
            ]
        assert mid != "graph-answer"
        with pytest.raises(ValueError, match="pending"):
            plan.require_complete()


def test_unknown_settings_and_blocked_character_remain_exact_dependencies(
    tmp_path, monkeypatch
):
    def seed(connection, purpose):
        _seed_settings(connection, purpose)
        if purpose == "legacy":
            connection.execute(
                'UPDATE conversations SET metadata=\'{"future_authority":{"secret":"do not expose"}}\' WHERE id=\'conversation\''
            )
            connection.execute(
                "UPDATE character_cards SET name='Default Assistant' WHERE id=1"
            )
            connection.execute(
                "UPDATE conversations SET character_id=1,assistant_kind='character',assistant_id='1' WHERE id<>'conversation'"
            )

    with content_reads(tmp_path, monkeypatch, seed) as reads:
        build = getattr(reads, "content_settings_plan", None)
        assert callable(build), "captured native reader needs C2 settings planning"
        plan = build()
        assert any(
            b.table == "conversations" and b.original == ("conversation",)
            for b in plan.blockers
        )
        assert any(
            b.table == "conversations"
            and b.dependency_domain == "character"
            and b.dependency_id == 1
            for b in plan.blockers
        )
        assert not any(row.table == "conversations" for row in plan.rows)
        assert "do not expose" not in repr(plan)
        assert any(
            "do not expose" in str(dict(row.values).get("metadata"))
            for row in plan.source_rows
            if row.table == "conversations"
        )


def test_native_appearance_and_local_authority_are_not_silently_completed(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.console_appearance import (
        ConsoleConversationAppearance,
        merge_console_conversation_appearance,
    )

    def seed(connection, purpose):
        _seed_settings(connection, purpose)
        metadata = connection.execute(
            "SELECT metadata FROM conversations WHERE id='conversation'"
        ).fetchone()[0]
        metadata = merge_console_conversation_appearance(
            metadata, ConsoleConversationAppearance(icon="★", color="#aBcDeF")
        )
        connection.execute(
            "UPDATE conversations SET metadata=?,character_id=1,assistant_kind='character',assistant_id='1',assistant_authority_id='source-context' WHERE id='conversation'",
            (json.dumps(metadata),),
        )

    with content_reads(tmp_path, monkeypatch, seed) as reads:
        plan = reads.content_settings_plan()
        row = next(
            (
                r
                for r in plan.rows
                if r.table == "conversations" and r.original == ("conversation",)
            ),
            None,
        )
        assert row is not None, plan.blockers
        values = dict(row.values)
        assert "assistant_authority_id" not in values
        assert row.pending_columns == ("assistant_authority_id",)
        assert values["assistant_id"] == str(plan.identities.mapped("character", 1))
        assert json.loads(values["metadata"])["console_appearance"] == {
            "icon": "★",
            "color": "#aBcDeF",
        }
        proof = next(
            p for p in plan.authority_slots if p.conversation_id == "conversation"
        )
        assert proof.original_authority_id == "source-context"
        assert proof.original_character_id == 1
        assert proof.mapped_character_id == plan.identities.mapped("character", 1)
        assert proof.context_name == "default"
        with pytest.raises(ValueError, match="pending"):
            plan.require_complete()


def test_native_video_terminal_metadata_keeps_exact_writer_shape():
    from tldw_chatbook.Backup_Recovery.mobile_content_settings import _message_metadata
    from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata

    receipt = "24f36b43-c412-4f46-84bd-56eb8b03e94f"
    video = VideoGenerationMetadata(
        name="historical",
        prompt="exact prompt",
        backend="minimax",
        is_unavailable_tombstone=True,
        terminal_receipt_id=receipt,
    )
    value = json.loads(video.to_json()) | {"terminal_receipt_id": receipt}
    encoded = _message_metadata(
        None,
        json.dumps(value),
        lambda *args: pytest.fail("tombstone has no physical member"),
    )
    assert json.loads(encoded) == value
    for malformed in (
        value | {"unknown_authority": "never drop"},
        value | {"terminal_receipt_id": "not-a-receipt"},
        value
        | {"video_generation": value["video_generation"] | {"future_authority": 7}},
    ):
        with pytest.raises(ValueError):
            _message_metadata(None, json.dumps(malformed), lambda *args: None)


def test_nonnull_native_authority_is_a_pending_value_slot(tmp_path, monkeypatch):
    source = {}

    def native_seed(db, purpose):
        authority = db.get_local_authority_id()
        source[purpose] = authority
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE conversations SET character_id=1,assistant_kind='character',assistant_id='1',assistant_authority_id=? WHERE id='conversation'",
                (authority,),
            )

    with content_reads(
        tmp_path, monkeypatch, _seed_settings, native_seed=native_seed
    ) as reads:
        plan = reads.content_settings_plan()
        row = next(
            r
            for r in plan.rows
            if r.table == "conversations" and r.original == ("conversation",)
        )
        assert "assistant_authority_id" not in dict(row.values)
        assert row.pending_columns == ("assistant_authority_id",)
        slot = next(
            p for p in plan.authority_slots if p.conversation_id == "conversation"
        )
        assert slot.original_authority_id == source["legacy"]
        assert slot.original_character_id == 1
        assert slot.mapped_character_id == plan.identities.mapped("character", 1)


def test_conversation_scalar_boundary_preserves_native_identity_without_coercion():
    from tldw_chatbook.Backup_Recovery.mobile_content_settings import (
        _validate_conversation_scalars,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    with closing(CharactersRAGDB(":memory:", "settings-scalars")) as db:
        cid = db.add_conversation({"title": "Exact title"})
        row = dict(db.get_conversation_by_id(cid))
        # Captured SQLite exposes stored scalars, without the native DB's
        # PARSE_DECLTYPES datetime converter on convenience reads.
        for name in ("created_at", "last_modified"):
            row[name] = (
                db.get_connection()
                .execute(
                    f"SELECT CAST({name} AS TEXT) FROM conversations WHERE id=?", (cid,)
                )
                .fetchone()[0]
            )
        _validate_conversation_scalars(row)
        server = row | {
            "runtime_backend": "server",
            "assistant_kind": "character",
            "assistant_id": "remote/7",
            "assistant_authority_id": "remote-authority",
        }
        _validate_conversation_scalars(server)
        for patch in (
            {"title": b"opaque bytes"},
            {"rating": 2.5},
            {"archived": "0"},
            {
                "assistant_authority_id": "x" * 257,
                "assistant_kind": "character",
                "character_id": 1,
                "assistant_id": "1",
            },
            {"assistant_kind": "character", "character_id": 1, "assistant_id": "01"},
            {"assistant_kind": "persona", "character_id": 1, "assistant_id": "persona"},
            {
                "runtime_backend": "server",
                "assistant_kind": "character",
                "character_id": 1,
                "assistant_id": "remote/7",
            },
        ):
            with pytest.raises(ValueError):
                _validate_conversation_scalars(row | patch)


def test_scope_rejects_legacy_truthiness_and_unknown_meaning():
    from tldw_chatbook.Backup_Recovery.mobile_content_settings import _scope

    original = {
        "version": 1,
        "items": [],
        "updated_at": "2026-10-25",
        "empty_is_scoped": False,
    }
    assert _scope(None, original) == original
    for patch in (
        {"empty_is_scoped": "false"},
        {"empty_is_scoped": 1},
        {"items": [{"source_type": "future", "source_id": "x"}]},
    ):
        with pytest.raises(ValueError):
            _scope(None, original | patch)


def test_video_live_member_requires_exact_native_physical_proof():
    from tldw_chatbook.Backup_Recovery.mobile_content_graph import _UnprovedReference
    from tldw_chatbook.Backup_Recovery.mobile_content_settings import _message_metadata
    from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata

    encoded = VideoGenerationMetadata(
        name="clip", prompt="exact", backend="minimax", container="webm"
    ).to_json()
    with pytest.raises(_UnprovedReference) as error:
        _message_metadata(
            None,
            encoded,
            lambda *args: pytest.fail("no invented file owner"),
            message_id="source-message",
        )
    assert error.value.domain == "generated_video_member"
    assert error.value.identity == "source-message/clip.webm"


def test_native_consumers_hydrate_private_fields_and_dictionary_derivations(
    tmp_path, monkeypatch
):
    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_conversation_hydration import (
        console_messages_from_conversation_tree,
    )
    from tldw_chatbook.Chat.console_generation_settings_metadata import (
        merge_console_generation_settings,
        parse_console_generation_settings,
        snapshot_from_session_settings,
    )
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
        decode_project_context_json,
        encode_project_context_json,
    )
    from tldw_chatbook.Chat.console_roleplay_metadata import (
        ConsoleRoleplayContext,
        merge_console_roleplay_context,
        parse_console_roleplay_context,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        parse_persisted_console_session_settings,
    )
    from tldw_chatbook.Chat.console_speech_preferences import (
        ConsoleSpeechPreferences,
        merge_console_speech_preferences,
        parse_console_speech_preferences,
    )
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )
    from tldw_chatbook.Chat.thinking_blocks import (
        DisplayableThinkingBlock,
        ThinkingEnvelope,
        dump_thinking_blocks_json,
        parse_thinking_blocks_json,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata

    thinking = ThinkingEnvelope(
        (
            DisplayableThinkingBlock(
                "opaque-block",
                0,
                "deepseek",
                "history-model",
                "responses",
                "text",
                "complete",
                "Exact private thinking note-a",
            ),
        )
    )
    continuation = json.dumps(_checkpoint())
    roleplay = ConsoleRoleplayContext(
        user_name_override="Exact User",
        character_system_template="Exact template note-a",
        persona_system_template="Exact persona",
    )
    speech = ConsoleSpeechPreferences(True, True, "sha256:" + "a" * 64, 1)
    project = ProjectInstructionControlState(
        True, "binding", "original-fingerprint", "original-notice"
    )
    receipt = "24f36b43-c412-4f46-84bd-56eb8b03e94f"

    def seed(connection, purpose):
        _seed_settings(connection, purpose)
        raw = connection.execute(
            "SELECT metadata FROM conversations WHERE id='conversation'"
        ).fetchone()[0]
        metadata = merge_console_generation_settings(
            raw,
            snapshot_from_session_settings(
                parse_persisted_console_session_settings(raw)
            ),
        )
        metadata = json.loads(merge_console_roleplay_context(metadata, roleplay))
        metadata = merge_console_speech_preferences(metadata, speech)
        metadata["active_dictionaries"] = [11, "11", 11]
        metadata["pinned_response_prefill"] = "literal active_dictionaries note-a"
        metadata["rag_scope"] = {
            "version": 2,
            "updated_at": "2026-10-25",
            "empty_is_scoped": True,
            "items": [{"source_type": "note", "source_id": "note-a"}],
        }
        connection.execute(
            "UPDATE conversations SET metadata=?,console_project_context_json=?,character_id=1,assistant_kind='character',assistant_id='1',assistant_authority_id=NULL,discovery_owner='ccp_character',discovery_entity_id='1' WHERE id='conversation'",
            (json.dumps(metadata), encode_project_context_json(project)),
        )
        connection.execute(
            "UPDATE messages SET provider_continuation_json=?,assistant_generation_state='continuation_active',thinking_blocks_json=? WHERE id='graph-answer'",
            (continuation, dump_thinking_blocks_json(thinking)),
        )
        video = VideoGenerationMetadata(
            name="historical",
            prompt="Exact video note-a",
            backend="minimax",
            source_image_message_id="graph-user",
            is_unavailable_tombstone=True,
            terminal_receipt_id=receipt,
        )
        connection.execute(
            "UPDATE messages SET metadata_json=? WHERE id='graph-variant'",
            (
                json.dumps(
                    json.loads(video.to_json()) | {"terminal_receipt_id": receipt}
                ),
            ),
        )

    from Tests.Backup_Recovery import mobile_consolidation_fixture

    original_seed = mobile_consolidation_fixture._seed
    source_workflow = {}

    def linked_chat_seed(db, legacy, result):
        source_workflow["legacy" if legacy else "canonical"] = result
        with db.transaction() as cursor:
            raw = cursor.execute(
                "SELECT metadata FROM conversations WHERE id='conversation'"
            ).fetchone()[0]
            metadata = json.loads(raw)
            metadata["console_agent_handoff"] = {
                "version": 2,
                "created_via": "new_chat",
                "state": "pending",
                "draft_revision": 1,
                "draft": "Exact draft note-a",
                "source_run_id": result["child_run_id"],
                "launch": {"mode": "draft", "status": "draft", "reason": None},
            }
            metadata["goal_launch"] = {
                "goal_id": result["goal_id"],
                "launch_id": "fixture-goal",
                "payload_hash": "a" * 64,
            }
            cursor.execute(
                "UPDATE conversations SET metadata=? WHERE id='conversation'",
                (json.dumps(metadata),),
            )

    def workflow_seed(stores, **kwargs):
        return original_seed(stores, **kwargs, linked_chat_seed=linked_chat_seed)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", workflow_seed)
    with content_reads(tmp_path, monkeypatch, seed) as reads:
        plan = reads.content_settings_plan()
        assert not plan.blockers
        rows = {(r.table, r.original): dict(r.values) for r in plan.rows}
        conversation = rows["conversations", ("conversation",)]
        cid = conversation["id"]
        assert conversation["assistant_authority_id"] is None
        assert not plan.authority_slots
        assert conversation["discovery_entity_id"] == str(
            plan.identities.mapped("character", 1)
        )
        assert parse_console_roleplay_context(conversation["metadata"]) == roleplay
        assert parse_console_speech_preferences(conversation["metadata"]) == speech
        assert (
            parse_console_generation_settings(conversation["metadata"]).snapshot.model
            == "history-model"
        )
        assert (
            decode_project_context_json(
                conversation["console_project_context_json"]
            ).project_instruction_notice_key
            == project.project_instruction_notice_key
        )
        assert decode_project_context_json(
            conversation["console_project_context_json"]
        ).working_folder_binding_id == plan.identities.mapped("binding", "binding")
        assert json.loads(conversation["metadata"])["rag_scope"]["items"][0][
            "source_id"
        ] == plan.identities.mapped("note", "note-a")
        assert {p.domain for p in plan.successors} >= {
            "binding",
            "character",
            "discovery_character",
            "provider_continuation",
            "dispatch_owner",
            "handoff_run",
            "run",
            "goal_launch",
        }
        metadata = json.loads(conversation["metadata"])
        assert metadata["console_agent_handoff"][
            "source_run_id"
        ] == plan.identities.mapped("run", source_workflow["legacy"]["child_run_id"])
        assert metadata["console_agent_handoff"]["draft"] == "Exact draft note-a"
        assert metadata["goal_launch"]["goal_id"] == plan.identities.mapped(
            "goal", source_workflow["legacy"]["goal_id"]
        )
        assert metadata["goal_launch"]["payload_hash"] == "a" * 64
        proof = next(
            p
            for p in plan.dictionary_derivations
            if p.conversation_id == "conversation"
        )
        assert proof.original_attachments == (11,)
        assert proof.original_unresolved and proof.mapped_unresolved
        assert proof.mapped_active == (plan.identities.mapped("dictionary", 11),) * 3
        with closing(CharactersRAGDB(":memory:", "full-settings-consumer")) as db:
            character = next(
                dict(r.values)
                for r in plan.graph.rows
                if r.table == "character_cards" and r.original == (1,)
            )
            with db.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO character_cards ("
                    + ",".join(character)
                    + ") VALUES ("
                    + ",".join("?" for _ in character)
                    + ")",
                    tuple(character.values()),
                )
            # Branch selectors are restored after native message insertion.
            insertion = conversation | {
                "active_leaf_message_id": None,
                "active_leaf_before_message_id": None,
                "topic_last_tagged_message_id": None,
            }
            assert db.add_conversation(insertion) == cid
            for source_id in ("graph-user", "graph-answer", "graph-variant"):
                assert db.add_message(rows["messages", (source_id,)])
            native = console_messages_from_conversation_tree(
                ChatConversationService(db).get_conversation_tree(cid), db=db
            )
            answer = next(
                m
                for m in native
                if m.persisted_message_id
                == plan.identities.mapped("message", "graph-answer")
            )
            assert answer.usage.total_tokens == 17
            assert answer.metadata.origin == "hook"
            stored = db.get_message_by_id(answer.persisted_message_id)
            assert parse_provider_continuation_json(
                stored["provider_continuation_json"]
            ) == parse_provider_continuation_json(continuation)
            assert (
                parse_thinking_blocks_json(stored["thinking_blocks_json"]) == thinking
            )
            assert stored["assistant_generation_state"] == "continuation_active"
            variant = next(
                m
                for m in native
                if m.persisted_message_id
                == plan.identities.mapped("message", "graph-variant")
            )
            assert (
                variant.video_metadata.source_image_message_id
                == plan.identities.mapped("message", "graph-user")
            )
            assert variant.video_metadata.prompt == "Exact video note-a"
            actual = (
                db.get_connection()
                .execute(
                    "SELECT dictionary_id FROM conversation_dictionary_attachments WHERE conversation_id=? ORDER BY dictionary_id",
                    (cid,),
                )
                .fetchall()
            )
            assert tuple(r[0] for r in actual) == proof.mapped_attachments
            assert (
                db.get_connection()
                .execute(
                    "SELECT 1 FROM conversation_dictionary_unresolved WHERE conversation_id=?",
                    (cid,),
                )
                .fetchone()
            )
        with pytest.raises(ValueError, match="pending"):
            plan.require_complete()


def test_fixed_native_adapters_reject_extra_nested_fields_and_scalar_coercion():
    from tldw_chatbook.Backup_Recovery import mobile_content_settings as settings
    from tldw_chatbook.Chat.message_metadata import (
        AgentChatStartMetadata,
        CanvasCardMetadata,
        CanvasCardOriginMetadata,
        CharacterEmoteEventMetadata,
        CharacterEmoteMetadata,
        MessageMetadata,
    )
    from tldw_chatbook.Chat.provider_usage import ProviderUsage
    from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata

    metadata = MessageMetadata(
        origin="agent_chat_start",
        agent_chat_start=AgentChatStartMetadata("attempt", "run", "conversation"),
        canvas_cards=(
            CanvasCardMetadata(
                "canvas",
                "revision",
                "Exact title",
                1,
                "a" * 64,
                "updated",
                CanvasCardOriginMetadata("message", "opaque-turn"),
                True,
            ),
        ),
        character_emote=CharacterEmoteMetadata(
            3,
            mood_label="happy",
            emote_events=(CharacterEmoteEventMetadata("happy", 2),),
            actor_kind="character",
            actor_id=1,
        ),
    )
    usage = ProviderUsage(uncached_input=3, output=2, transcription_seconds=1.5)
    video = VideoGenerationMetadata(
        "clip", "Exact prompt", "minimax", is_unavailable_tombstone=True
    )
    cases = (
        (settings._METADATA_JSON, asdict(metadata), metadata),
        (settings._USAGE_JSON, asdict(usage), usage),
        (settings._VIDEO_JSON, asdict(video), video),
    )
    for adapter, document, expected in cases:
        assert settings._typed(adapter, json.dumps(document)) == expected
        with pytest.raises(ValueError):
            settings._typed(
                adapter, json.dumps(document | {"future_authority": "never drop"})
            )
    malformed = (
        (
            settings._METADATA_JSON,
            asdict(metadata)
            | {"agent_chat_start": asdict(metadata.agent_chat_start) | {"extra": 1}},
        ),
        (
            settings._METADATA_JSON,
            asdict(metadata)
            | {
                "canvas_cards": [
                    {
                        **asdict(metadata.canvas_cards[0]),
                        "origin": {
                            "message_id": "message",
                            "run_id": "turn",
                            "extra": 1,
                        },
                    }
                ]
            },
        ),
        (
            settings._METADATA_JSON,
            asdict(metadata)
            | {
                "character_emote": {
                    **asdict(metadata.character_emote),
                    "emote_events": [{"state": "happy", "at_char": "2"}],
                }
            },
        ),
        (settings._USAGE_JSON, asdict(usage) | {"output": "2"}),
        (settings._VIDEO_JSON, asdict(video) | {"is_unavailable_tombstone": 1}),
    )
    for adapter, document in malformed:
        with pytest.raises(ValueError):
            settings._typed(adapter, json.dumps(document))
    assert all(
        not hasattr(kind, "__pydantic_config__")
        for kind in (
            MessageMetadata,
            AgentChatStartMetadata,
            CanvasCardMetadata,
            CanvasCardOriginMetadata,
            CharacterEmoteMetadata,
            CharacterEmoteEventMetadata,
            ProviderUsage,
            VideoGenerationMetadata,
        )
    )


@pytest.mark.parametrize(
    "metadata",
    [
        '{"pinned_response_prefill":"first","pinned_response_prefill":"second"}',
        '{"pinned_response_prefill":NaN}',
        '{"console_session_settings":{"version":999}}',
        '{"console_generation_settings":{"version":999}}',
        '{"console_roleplay_context":{"version":2,"unknown_authority":1}}',
        '{"console_speech":{"auto_speak":"false","paused":false,"consent_destination":null,"consent_version":1}}',
        '{"console_appearance":{"icon":"x","color":null,"new":true}}',
        '{"active_dictionaries":[true]}',
        '{"console_agent_handoff":{"draft":"keep","created_via":"new_chat","source_run_id":"run","extra":1}}',
    ],
)
def test_unknown_native_metadata_never_becomes_accepted_defaults(metadata):
    from tldw_chatbook.Backup_Recovery.mobile_content_settings import _metadata

    with pytest.raises(ValueError):
        _metadata(
            None,
            {"metadata": metadata},
            lambda *args: pytest.fail("invalid metadata reached successors"),
        )


def test_declared_persona_and_emote_dependencies_require_successor_proof(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.message_metadata import (
        CharacterEmoteMetadata,
        MessageMetadata,
    )

    def seed(connection, purpose):
        _seed_settings(connection, purpose)
        connection.execute(
            "UPDATE conversations SET discovery_owner='ccp_persona',discovery_entity_id='missing-local-persona' WHERE id<>'conversation'"
        )
        metadata = MessageMetadata(
            origin="hook",
            character_emote=CharacterEmoteMetadata(
                12,
                actor_kind="character",
                actor_id=11,
                expression_key="happy",
                expression_id=11,
            ),
        )
        connection.execute(
            "UPDATE messages SET metadata_json=? WHERE id='graph-answer'",
            (metadata.to_json(),),
        )

    with content_reads(tmp_path, monkeypatch, seed) as reads:
        plan = reads.content_settings_plan()
        row = next(
            r
            for r in plan.rows
            if r.table == "messages" and r.original == ("graph-answer",)
        )
        checks = {
            "persona_discovery_requires_actual_authority": any(
                b.table == "conversations"
                and b.dependency_domain == "persona"
                and b.dependency_id == "missing-local-persona"
                for b in plan.blockers
            ),
            "emote_actor_and_expression_require_C2c": {
                "C2c:character",
                "C2c:character_expression",
            }
            <= set(row.required_proofs),
        }
        assert all(checks.values()), checks
        metadata = MessageMetadata.from_json(dict(row.values)["metadata_json"])
        assert metadata.character_emote.actor_id == plan.identities.mapped(
            "character", 11
        )
        assert metadata.character_emote.expression_id == plan.identities.mapped(
            "character_expression", 11
        )
        with pytest.raises(ValueError, match="pending"):
            plan.require_complete()


def test_archive_preparation_precedes_sqlite_budget_and_first_open_rejects_tampering(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager
    from threading import Event

    from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from tldw_chatbook.Backup_Recovery import mobile_captured_sqlite
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    extract = mobile_captured_sqlite._extract
    native_open = mobile_captured_sqlite.open_recovery_validation
    completed = []
    reached = []

    def observed_extract(scope, sealed, members, root):
        yield from extract(scope, sealed, members, root)
        completed.append(sealed.digest)  # After the real final verify_sealed.

    @contextmanager
    def tampered_first_open(owner_id, candidate, **kwargs):
        assert len(completed) == 2, (
            "both archives must finish authentication before SQLite budgets start"
        )
        if not reached:
            original = candidate.read_bytes()
            changed = original.replace(
                b"private fixture prompt", b"changed fixture prompt"
            )
            assert changed != original
            candidate.write_bytes(changed)
            reached.append(owner_id)
        with native_open(owner_id, candidate, **kwargs) as opened:
            yield opened

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            before = _fingerprints(tmp_path / "profile")
            with journal._locked(exclusive=False) as fd:
                events = tuple(journal._records(fd))
            monkeypatch.setattr(mobile_captured_sqlite, "_extract", observed_extract)
            monkeypatch.setattr(
                mobile_captured_sqlite, "open_recovery_validation", tampered_first_open
            )
            with (
                pytest.raises(ValueError, match="candidate_changed"),
                captured.sqlite_reads(
                    password=b"test-private-mobile-rollback", cancel=Event()
                ),
            ):
                pytest.fail("changed first candidate reached the read scope")
            assert reached == ["db.chachanotes.primary"]
            assert len(set(completed)) == 2
            assert not tuple(output.glob(".mobile-read-*"))
            assert _fingerprints(tmp_path / "profile") == before
            with journal._locked(exclusive=False) as fd:
                assert tuple(journal._records(fd)) == events
