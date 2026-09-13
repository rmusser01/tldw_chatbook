"""Privacy inventory for model thinking and provider continuation surfaces."""

from __future__ import annotations

import io
import json
import logging
import shutil
import zipfile
from pathlib import Path

import pytest

from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    export_conversation_to_text,
    load_chat_history_from_file_and_save_to_db,
)
import tldw_chatbook.Character_Chat.Character_Chat_Lib as character_chat_module
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Chat.Chat_Functions import generate_chat_history_content
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    PROPRIETARY_THINKING_NOTICE,
    ConsoleChatMessage,
    ConsoleMessageRole,
    derive_console_session_title,
)
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.Chat.console_thinking_capture import ThinkingCapture
from tldw_chatbook.Chat.cost_display import build_provenance_line
from tldw_chatbook.Chat.document_generator import DocumentGenerator
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRound,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
)
from tldw_chatbook.Chat.trajectory_export import (
    build_trajectory_export,
    write_trajectory_export,
)
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.chat_outbox_producer import ChatSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload, generate_dataset_key
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


VISIBLE_ANSWER = "VISIBLE-ANSWER-TASK2-CANARY"
DISPLAYABLE_THINKING = "DISPLAYABLE-THINKING-TASK2-CANARY"
RAW_PRIVATE = "RAW-PRIVATE-CONTINUATION-TASK2-CANARY"
SENSITIVE_WARNING = (
    "This conversation export contains model thinking or private provider "
    "continuation. Treat it as sensitive conversation data."
)
PRIVATE_DIAGNOSTIC_CANARIES = (
    DISPLAYABLE_THINKING,
    RAW_PRIVATE,
    PROPRIETARY_THINKING_NOTICE,
)


def _thinking() -> ThinkingEnvelope:
    return ThinkingEnvelope(
        blocks=(
            DisplayableThinkingBlock(
                block_id="displayable-1",
                round_ordinal=0,
                provider="llama.cpp",
                model="local-reasoner",
                protocol="chat_completions",
                source_format="reasoning_content",
                status="complete",
                text=DISPLAYABLE_THINKING,
            ),
            ProprietaryThinkingBlock(
                block_id="proprietary-1",
                round_ordinal=1,
                provider="moonshot",
                model="private-reasoner",
                protocol="chat_completions",
                source_format="provider_evidence",
                status="complete",
            ),
        )
    )


def _continuation() -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k2.6",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content=VISIBLE_ANSWER,
                reasoning_blocks=(RAW_PRIVATE,),
                calls=(),
            ),
        ),
    )


def _canonical_thinking() -> str:
    value = dump_thinking_blocks_json(_thinking())
    assert value is not None
    return value


def _canonical_continuation() -> str:
    value = dump_provider_continuation_json(_continuation())
    assert value is not None
    return value


def _assert_answer_only(value: object) -> None:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    assert VISIBLE_ANSWER in text
    assert DISPLAYABLE_THINKING not in text
    assert RAW_PRIVATE not in text
    assert PROPRIETARY_THINKING_NOTICE not in text


def _assert_content_free_diagnostic(value: str) -> None:
    for canary in PRIVATE_DIAGNOSTIC_CANARIES:
        assert canary not in value


def _seed_source(database_path: Path) -> tuple[str, str]:
    database = CharactersRAGDB(database_path, "source-device")
    try:
        conversation_id = database.add_conversation(
            {
                "id": "thinking-privacy-conversation",
                "root_id": "thinking-privacy-conversation",
                "title": "Visible privacy title",
                "assistant_kind": "character",
                "character_id": 1,
                "discovery_owner": "ccp_character",
                "thinking_history_policy": "include",
            }
        )
        assert conversation_id == "thinking-privacy-conversation"
        assert database.add_message(
            {
                "id": "user-1",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "Visible question",
                "timestamp": "2026-08-26T00:00:00.000Z",
            }
        )
        message_id = database.add_message(
            {
                "id": "assistant-1",
                "conversation_id": conversation_id,
                "parent_message_id": "user-1",
                "sender": "assistant",
                "role": "assistant",
                "content": VISIBLE_ANSWER,
                "thinking_blocks_json": _canonical_thinking(),
                "provider_continuation_json": _canonical_continuation(),
                "assistant_generation_state": "complete",
                "timestamp": "2026-08-26T00:00:01.000Z",
            }
        )
        assert message_id == "assistant-1"
        database.set_conversation_active_leaf(conversation_id, message_id)
        return conversation_id, message_id
    finally:
        database.close_connection()


def test_answer_derivatives_and_capability_only_capture_use_visible_content() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content=VISIBLE_ANSWER,
        thinking=_thinking(),
        provider_continuation=_continuation(),
        usage=ProviderUsage(uncached_input=3, output=2),
    )
    actions = ConsoleMessageActionService()
    copy = actions.dispatch("copy", message)
    speech = actions.dispatch("speak", message)
    summary = ConsoleChatController._build_summary_span_text(
        ConsoleChatController.__new__(ConsoleChatController),
        [message],
        None,
        model="test-model",
    )
    provenance = build_provenance_line(
        provider="test-provider",
        model="test-model",
        usage=message.usage,
        cost=None,
        pricing_known=False,
    )

    assert copy.clipboard_text == VISIBLE_ANSWER
    assert speech.target_content == VISIBLE_ANSWER
    _assert_answer_only(summary)
    assert derive_console_session_title("Visible user draft") == "Visible user draft"
    assert provenance == "test-provider · test-model · 5 tok · pricing unknown"
    assert DISPLAYABLE_THINKING not in repr(message)
    assert RAW_PRIVATE not in repr(message)
    assert PROPRIETARY_THINKING_NOTICE not in repr(message)

    capture = ThinkingCapture(assistant_owner_id="capability-only")
    assert capture.observe(VISIBLE_ANSWER).envelope is None
    assert capture.settle("complete").envelope is None


def test_human_exporters_search_and_trajectory_are_answer_only_by_construction(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_path = tmp_path / "human-source.db"
    shutil.copyfile(chachanotes_template_db, source_path)
    conversation_id, _ = _seed_source(source_path)
    database = CharactersRAGDB(source_path, "thinking-privacy-human")
    try:
        text_export = export_conversation_to_text(database, conversation_id)
        assert text_export is not None
        _assert_answer_only(text_export)

        generator = DocumentGenerator.__new__(DocumentGenerator)
        generator.db = database
        context_rows = generator.get_conversation_context(conversation_id)
        _assert_answer_only(generator.format_context_for_llm(context_rows))
        assert all(
            "thinking_blocks_json" not in row
            and "provider_continuation_json" not in row
            for row in context_rows
        )

        # Negative control: even a caller-supplied mapping with sensitive
        # sidecars is projected through explicit visible fields.
        mapped = generator.format_context_for_llm(
            [
                {
                    "role": "assistant",
                    "content": VISIBLE_ANSWER,
                    "timestamp": "now",
                    "thinking_blocks_json": _canonical_thinking(),
                    "provider_continuation_json": _canonical_continuation(),
                }
            ]
        )
        _assert_answer_only(mapped)

        markdown = LocalCharacterPersonaService(database).export_chat_history(
            conversation_id, format="markdown"
        )
        _assert_answer_only(markdown)

        trajectory = build_trajectory_export(database, conversation_id)
        _assert_answer_only(trajectory)
        trajectory_path = write_trajectory_export(
            tmp_path / "trajectory.json", trajectory
        )
        _assert_answer_only(trajectory_path.read_text(encoding="utf-8"))

        assert len(database.search_messages_by_content(VISIBLE_ANSWER)) == 1
        assert database.search_messages_by_content(DISPLAYABLE_THINKING) == []
        assert database.search_messages_by_content(RAW_PRIVATE) == []
        assert database.search_messages_by_content(PROPRIETARY_THINKING_NOTICE) == []
    finally:
        database.close_connection()


def test_real_durable_owners_keep_thinking_and_private_continuation_separate(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_path = tmp_path / "durable-source.db"
    shutil.copyfile(chachanotes_template_db, source_path)
    conversation_id, message_id = _seed_source(source_path)
    source = CharactersRAGDB(source_path, "thinking-privacy-durable")
    target = CharactersRAGDB(tmp_path / "selected-json-target.db", "selected-target")
    state_repository = SyncStateRepository(tmp_path / "sync-state.db")
    state_repository.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        profile_mode="local_first",
        device_id="source-device",
        dataset_id="dataset-1",
    )
    dataset_key = generate_dataset_key()
    try:
        row = source.get_message_by_id(message_id)
        assert row is not None
        assert row["content"] == VISIBLE_ANSWER
        assert DISPLAYABLE_THINKING in row["thinking_blocks_json"]
        assert RAW_PRIVATE not in row["thinking_blocks_json"]
        assert RAW_PRIVATE in row["provider_continuation_json"]
        assert DISPLAYABLE_THINKING not in row["provider_continuation_json"]
        assert PROPRIETARY_THINKING_NOTICE not in row["thinking_blocks_json"]
        assert PROPRIETARY_THINKING_NOTICE not in row["provider_continuation_json"]

        sync_record = source.execute_query(
            "SELECT payload FROM sync_log WHERE entity = 'messages' "
            "AND entity_id = ? ORDER BY change_id DESC LIMIT 1",
            (message_id,),
        ).fetchone()
        assert sync_record is not None
        sync_projection = json.loads(sync_record["payload"])
        assert DISPLAYABLE_THINKING in sync_projection["thinking_blocks_json"]
        assert RAW_PRIVATE in sync_projection["provider_continuation_json"]
        assert PROPRIETARY_THINKING_NOTICE not in sync_record["payload"]

        expected_payload = {
            "assistant_generation_state": "complete",
            "content": VISIBLE_ANSWER,
            "provider_continuation_json": _canonical_continuation(),
            "role": "assistant",
            "thinking_blocks_json": _canonical_thinking(),
        }
        payload_hash = canonical_payload_hash(expected_payload)
        intent = source.read_committed_chat_sync_intent(
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
        )
        assert intent is not None
        result = ChatSyncV2OutboxProducer(
            state_repository=state_repository,
            dataset_keys={"dataset-1": dataset_key},
            source=source,
        ).reconcile_chat_message_intent(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            message_id=message_id,
            message_version=1,
            payload_hash=payload_hash,
        )
        assert result["status"] == "enqueued", result
        outbox_rows = state_repository.list_sync_v2_outbox_entries(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            dataset_id="dataset-1",
        )
        assert len(outbox_rows) == 1
        sync_payload = decrypt_sync_payload(
            json.loads(outbox_rows[0]["envelope"]["payload_ciphertext"]),
            key=dataset_key,
        )
        assert sync_payload == expected_payload

        rows = source.get_messages_for_conversation(conversation_id)
        selected_text, _ = generate_chat_history_content(
            rows, conversation_id, None, db_instance=source
        )
        selected = json.loads(selected_text)
        assert selected["sensitive_data_warning"] == SENSITIVE_WARNING
        assistant = next(
            item for item in selected["history"] if item["role"] == "assistant"
        )
        assert DISPLAYABLE_THINKING in json.dumps(assistant["thinking_blocks"])
        assert RAW_PRIVATE in json.dumps(assistant["_private"])
        assert PROPRIETARY_THINKING_NOTICE not in selected_text

        imported_id, _ = load_chat_history_from_file_and_save_to_db(
            target, io.BytesIO(selected_text.encode("utf-8"))
        )
        assert imported_id is not None
        imported = target.get_messages_for_conversation(imported_id)
        imported_assistant = next(row for row in imported if row["role"] == "assistant")
        assert DISPLAYABLE_THINKING in imported_assistant["thinking_blocks_json"]
        assert RAW_PRIVATE in imported_assistant["provider_continuation_json"]

        all_durable = json.dumps(
            [row, sync_projection, sync_payload, selected, imported_assistant],
            default=str,
        )
        assert PROPRIETARY_THINKING_NOTICE not in all_durable
    finally:
        source.close_connection()
        target.close_connection()


def test_chatbook_archive_and_import_preserve_only_approved_sensitive_fields(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_path = tmp_path / "chatbook-source.db"
    destination_path = tmp_path / "chatbook-target.db"
    shutil.copyfile(chachanotes_template_db, source_path)
    shutil.copyfile(chachanotes_template_db, destination_path)
    conversation_id, _ = _seed_source(source_path)
    archive_path = tmp_path / "thinking-privacy.chatbook.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path
    success, message, _ = creator.create_chatbook(
        name="Thinking privacy",
        description="Privacy inventory",
        content_selections={ContentType.CONVERSATION: [conversation_id]},
        output_path=archive_path,
    )
    assert success, message

    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        conversation_name = next(
            name
            for name in archive.namelist()
            if name.startswith("content/conversations/conversation_")
            and name.endswith(".json")
        )
        conversation = json.loads(archive.read(conversation_name))
    assistant = next(
        item for item in conversation["messages"] if item["role"] == "assistant"
    )
    assert DISPLAYABLE_THINKING in json.dumps(assistant["_thinking"])
    assert RAW_PRIVATE in json.dumps(assistant["_private"])
    assert SENSITIVE_WARNING in json.dumps(manifest)
    assert PROPRIETARY_THINKING_NOTICE not in json.dumps([manifest, conversation])

    status = ImportStatus()
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "chatbook-import"
    importer.temp_dir.mkdir()
    imported, import_message = importer.import_chatbook(
        archive_path, import_status=status
    )
    assert imported, import_message
    assert status.failed_items == 0
    destination = CharactersRAGDB(destination_path, "chatbook-assert")
    try:
        imported_conversation = destination.get_conversation_by_name(
            "Visible privacy title"
        )[0]
        rows = destination.get_messages_for_conversation(imported_conversation["id"])
        imported_assistant = next(row for row in rows if row["role"] == "assistant")
        assert DISPLAYABLE_THINKING in imported_assistant["thinking_blocks_json"]
        assert RAW_PRIVATE in imported_assistant["provider_continuation_json"]
        assert PROPRIETARY_THINKING_NOTICE not in json.dumps(
            imported_assistant, default=str
        )
    finally:
        destination.close_connection()


def test_malformed_human_export_log_is_content_free_for_all_privacy_canaries(
    caplog,
) -> None:
    malformed = {
        "thinking_blocks_json": _canonical_thinking(),
        "provider_continuation_json": _canonical_continuation(),
        "thinking_warning": PROPRIETARY_THINKING_NOTICE,
    }
    serialized_input = json.dumps(malformed)
    assert all(canary in serialized_input for canary in PRIVATE_DIAGNOSTIC_CANARIES)

    with caplog.at_level(logging.WARNING):
        payload, _ = generate_chat_history_content([malformed], None, None)

    assert json.loads(payload)["history"] == []
    assert "Unexpected item format" in caplog.text
    _assert_content_free_diagnostic(caplog.text)


def test_malformed_human_import_log_is_content_free_for_all_privacy_canaries() -> None:
    complete = json.dumps(
        {
            "thinking_blocks_json": DISPLAYABLE_THINKING,
            "provider_continuation_json": RAW_PRIVATE,
            "thinking_warning": PROPRIETARY_THINKING_NOTICE,
        }
    )
    malformed = complete[:-1].encode()
    assert all(canary.encode() in malformed for canary in PRIVATE_DIAGNOSTIC_CANARIES)
    diagnostics = io.StringIO()
    sink_id = character_chat_module.logger.add(diagnostics, format="{message}")
    database = CharactersRAGDB(":memory:", "thinking-privacy-malformed-import")
    try:
        assert load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(malformed)
        ) == (None, None)
    finally:
        character_chat_module.logger.remove(sink_id)
        database.close_connection()

    diagnostic = diagnostics.getvalue()
    assert "operation=chat_history_import" in diagnostic
    assert "category=JSONDecodeError" in diagnostic
    _assert_content_free_diagnostic(diagnostic)


def test_chatbook_rejects_proprietary_text_inside_thinking_without_echo() -> None:
    thinking = json.loads(_canonical_thinking())
    thinking["blocks"][1]["text"] = f"{RAW_PRIVATE}\n{PROPRIETARY_THINKING_NOTICE}"
    graph = {
        "messages": [
            {
                "id": "assistant-1",
                "parent_id": None,
                "variant_of": None,
                "order": 0,
                "role": "assistant",
                "content": VISIBLE_ANSWER,
                "deleted": False,
                "variant_number": 1,
                "is_selected_variant": True,
                "total_variants": 1,
                "_thinking": thinking,
            }
        ],
        "active_leaf_message_id": "assistant-1",
        "selected_path_message_ids": ["assistant-1"],
    }
    serialized_input = json.dumps(graph)
    assert all(canary in serialized_input for canary in PRIVATE_DIAGNOSTIC_CANARIES)

    with pytest.raises(ValueError, match="Invalid V2 conversation graph") as caught:
        ChatbookImporter._validate_v2_conversation_graph(graph)

    _assert_content_free_diagnostic(str(caught.value))
