"""Lossless graph and private-continuation chatbook round trips."""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import pytest

from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


PRIVATE_REASONING = "PRIVATE-REASONING-CANARY"
PRIVATE_ARGUMENTS = "PRIVATE-ARGUMENT-CANARY"


def _checkpoint_json(*, provider: str = "deepseek") -> str:
    protocol = "responses" if provider == "deepseek" else "chat_completions"
    raw = {
        "schema_version": 1,
        "checkpoint_revision": 1,
        "provider": provider,
        "protocol": protocol,
        "model": "deepseek-v4-flash" if provider == "deepseek" else "glm-5",
        "api_base_url": "https://api.deepseek.com"
        if provider == "deepseek"
        else "https://api.z.ai/api/paas/v4",
        "state": "active",
        "rounds": [
            {
                "assistant_content": "",
                "reasoning_blocks": [PRIVATE_REASONING],
                "calls": [
                    {
                        "call_id": "call_1",
                        "name": "calculator",
                        "arguments": json.dumps({"value": PRIVATE_ARGUMENTS}),
                        "state": "pending",
                    }
                ],
            }
        ],
    }
    return dump_provider_continuation_json(parse_provider_continuation_json(raw)) or ""


def _kimi_family_private(
    content: str,
    *,
    model: str = "kimi-k2.6",
    post_tool_only: bool = False,
) -> dict:
    rounds: list[dict] = []
    if post_tool_only:
        rounds.append(
            {
                "assistant_content": "",
                "reasoning_blocks": [PRIVATE_REASONING],
                "calls": [
                    {
                        "call_id": "call_1",
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                        "state": "completed",
                        "result": "4",
                    }
                ],
            }
        )
    else:
        rounds.append(
            {
                "assistant_content": content,
                "reasoning_blocks": [PRIVATE_REASONING],
                "calls": [],
            }
        )
    return {
        "schema_version": 1,
        "checkpoint_revision": 1,
        "provider": "moonshot",
        "protocol": "chat_completions",
        "model": model,
        "api_base_url": "https://api.moonshot.ai/v1",
        "state": "complete",
        "rounds": rounds,
    }


def test_imported_continuation_family_owner_rule_covers_versioned_kimi() -> None:
    """TASK-19170: the import keep/discard rule for complete preserved-thinking
    checkpoints follows the versioned-kimi family, with the pre-19170
    tool-only durable shape still kept."""
    status = ImportStatus()
    kept = ChatbookImporter._imported_continuation_json(
        {
            "role": "assistant",
            "content": "visible answer",
            "_private": {
                "provider_continuation": _kimi_family_private("visible answer")
            },
        },
        ordinal=1,
        status=status,
    )
    assert kept is not None
    assert parse_provider_continuation_json(kept).model == "kimi-k2.6"
    assert status.warnings == []

    dropped = ChatbookImporter._imported_continuation_json(
        {
            "role": "assistant",
            "content": "visible answer",
            "_private": {
                "provider_continuation": _kimi_family_private("does not match")
            },
        },
        ordinal=2,
        status=status,
    )
    assert dropped is None
    assert status.warnings == [
        "Exact tool continuation was discarded for message 2."
    ]


def test_imported_continuation_keeps_pre_19170_family_tool_only_shape() -> None:
    """Durable-data control: an old-style kimi-k2.6 complete checkpoint (no
    final reasoning round) must never be discarded by the owner rule -- its
    final round is a tool round whose content is not the visible answer."""
    status = ImportStatus()
    kept = ChatbookImporter._imported_continuation_json(
        {
            "role": "assistant",
            "content": "visible answer",
            "_private": {
                "provider_continuation": _kimi_family_private(
                    "visible answer", post_tool_only=True
                )
            },
        },
        ordinal=1,
        status=status,
    )
    assert kept is not None
    assert parse_provider_continuation_json(kept).rounds[-1].calls[0].state == (
        "completed"
    )
    assert status.warnings == []


def _source_graph(
    tmp_path: Path, chachanotes_template_db: Path
) -> tuple[dict[str, str], str, dict[str, str]]:
    db_path = tmp_path / "source.db"
    shutil.copyfile(chachanotes_template_db, db_path)
    db = CharactersRAGDB(str(db_path), "source")
    conversation_id = db.add_conversation(
        {
            "id": "conversation-source",
            "root_id": "conversation-source",
            "title": "Graph",
        }
    )
    assert conversation_id == "conversation-source"
    ids = {
        "user": "message-user",
        "base": "message-assistant-base",
        "selected": "message-assistant-selected",
        "deleted": "message-deleted-off-path",
    }
    assert db.add_message(
        {
            "id": ids["user"],
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Choose a branch",
            "timestamp": "2026-08-12T00:00:00+00:00",
        }
    )
    assert db.add_message(
        {
            "id": ids["base"],
            "conversation_id": conversation_id,
            "parent_message_id": ids["user"],
            "sender": "assistant",
            "content": "same visible answer",
            "timestamp": "2026-08-12T00:00:01+00:00",
        }
    )
    assert db.add_message(
        {
            "id": ids["selected"],
            "conversation_id": conversation_id,
            "parent_message_id": ids["user"],
            "sender": "assistant",
            "content": "same visible answer",
            "timestamp": "2026-08-12T00:00:02+00:00",
            "provider_continuation_json": _checkpoint_json(),
            "assistant_generation_state": "accepted",
        }
    )
    assert db.add_message(
        {
            "id": ids["deleted"],
            "conversation_id": conversation_id,
            "parent_message_id": ids["base"],
            "sender": "user",
            "content": "deleted branch",
            "timestamp": "2026-08-12T00:00:03+00:00",
        }
    )
    with db.transaction() as connection:
        connection.execute(
            "UPDATE messages SET variant_number = 1, is_selected_variant = 0, "
            "total_variants = 2 WHERE id = ?",
            (ids["base"],),
        )
        connection.execute(
            "UPDATE messages SET variant_of = ?, variant_number = 2, "
            "is_selected_variant = 1, total_variants = 2 WHERE id = ?",
            (ids["base"], ids["selected"]),
        )
    assert db.soft_delete_message(ids["deleted"], 1)
    db.set_conversation_active_leaf(conversation_id, ids["selected"])
    return {"ChaChaNotes": str(db_path)}, conversation_id, ids


def _create_export(
    tmp_path: Path, db_paths: dict[str, str], conversation_id: str
) -> Path:
    output = tmp_path / "graph.chatbook.zip"
    creator = ChatbookCreator(db_paths)
    creator.temp_dir = tmp_path
    success, message, _ = creator.create_chatbook(
        name="Graph",
        description="graph round trip",
        content_selections={ContentType.CONVERSATION: [conversation_id]},
        output_path=output,
    )
    assert success, message
    return output


def _rewrite_export(
    source: Path,
    destination: Path,
    *,
    mutate_manifest=None,
    mutate_conversation=None,
) -> Path:
    with zipfile.ZipFile(source) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files["manifest.json"])
    conversation_name = next(
        name
        for name in files
        if name.startswith("content/conversations/conversation_")
        and name.endswith(".json")
    )
    conversation = json.loads(files[conversation_name])
    if mutate_manifest:
        mutate_manifest(manifest)
    if mutate_conversation:
        mutate_conversation(conversation)
    files["manifest.json"] = json.dumps(manifest).encode()
    files[conversation_name] = json.dumps(conversation).encode()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return destination


def test_v2_export_preserves_graph_and_private_owner(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    db_paths, conversation_id, ids = _source_graph(tmp_path, chachanotes_template_db)

    export_path = _create_export(tmp_path, db_paths, conversation_id)

    with zipfile.ZipFile(export_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        readme = archive.read("README.md").decode()
        conversation = json.loads(
            archive.read("content/conversations/conversation_conversation-source.json")
        )
    assert manifest["version"] == "2.0"
    assert (
        manifest["content_items"][0]["metadata"][
            "contains_private_provider_continuation"
        ]
        is True
    )
    assert "private provider continuation data" in readme.lower()
    assert conversation["private_data_warning"] == (
        "This conversation contains private provider continuation data."
    )
    assert conversation["active_leaf_message_id"] == ids["selected"]
    assert conversation["selected_path_message_ids"] == [
        ids["user"],
        ids["selected"],
    ]
    by_id = {message["id"]: message for message in conversation["messages"]}
    assert set(by_id) == set(ids.values())
    assert by_id[ids["selected"]]["parent_id"] == ids["user"]
    assert by_id[ids["selected"]]["variant_of"] == ids["base"]
    assert by_id[ids["selected"]]["variant_number"] == 2
    assert by_id[ids["selected"]]["is_selected_variant"] is True
    assert by_id[ids["deleted"]]["deleted"] is True
    assert by_id[ids["selected"]]["_private"]["provider_continuation"] == json.loads(
        _checkpoint_json()
    )
    assert by_id[ids["selected"]]["assistant_generation_state"] == (
        "continuation_active"
    )
    assert by_id[ids["user"]]["assistant_generation_state"] is None
    assert "console_dispatch_checkpoints" not in json.dumps(conversation)
    assert "_private" not in by_id[ids["base"]]
    for member in (manifest, readme):
        serialized = json.dumps(member)
        assert PRIVATE_REASONING not in serialized
        assert PRIVATE_ARGUMENTS not in serialized


def test_v2_import_remaps_complete_graph_before_attaching_private_owner(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_paths, conversation_id, old_ids = _source_graph(
        tmp_path, chachanotes_template_db
    )
    export_path = _create_export(tmp_path, source_paths, conversation_id)
    destination_path = tmp_path / "destination.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    destination = CharactersRAGDB(str(destination_path), "destination")
    collision_conversation_id = destination.add_conversation(
        {"id": "collision-conversation", "root_id": "collision-conversation"}
    )
    assert collision_conversation_id
    for old_id in old_ids.values():
        assert destination.add_message(
            {
                "id": old_id,
                "conversation_id": collision_conversation_id,
                "sender": "user",
                "content": f"collision {old_id}",
            }
        )

    status = ImportStatus()
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "imports"
    importer.temp_dir.mkdir()
    success, message = importer.import_chatbook(
        export_path,
        conflict_resolution=ConflictResolution.RENAME,
        import_status=status,
    )

    assert success, message
    assert status.warnings == []
    imported_conversations = destination.get_conversation_by_name("Graph")
    assert len(imported_conversations) == 1
    imported_id = str(imported_conversations[0]["id"])
    rows = destination.execute_query(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp, rowid",
        (imported_id,),
    ).fetchall()
    imported = [dict(row) for row in rows]
    assert len(imported) == 4
    assert not set(old_ids.values()).intersection(str(row["id"]) for row in imported)
    user = next(row for row in imported if row["content"] == "Choose a branch")
    siblings = [row for row in imported if row["content"] == "same visible answer"]
    assert len(siblings) == 2
    assert {row["parent_message_id"] for row in siblings} == {user["id"]}
    assert [row["variant_number"] for row in siblings] == [1, 2]
    selected = next(row for row in siblings if row["is_selected_variant"])
    base = next(row for row in siblings if not row["is_selected_variant"])
    assert selected["variant_of"] == base["id"]
    assert selected["provider_continuation_json"] == _checkpoint_json()
    assert selected["assistant_generation_state"] == "continuation_active"
    checkpoint = parse_provider_continuation_json(
        selected["provider_continuation_json"]
    )
    assert checkpoint.state == "active"
    assert checkpoint.rounds[-1].calls[-1].state == "pending"
    deleted = next(row for row in imported if row["content"] == "deleted branch")
    assert deleted["deleted"] == 1
    assert deleted["parent_message_id"] == base["id"]
    assert destination.get_conversation_active_leaf(imported_id) == selected["id"]


@pytest.mark.parametrize(
    ("checkpoint", "persisted_state", "expected_state"),
    [
        (json.loads(_checkpoint_json()), "complete", "continuation_active"),
        (
            _kimi_family_private("same visible answer"),
            "complete",
            "complete",
        ),
    ],
    ids=["active-authoritative", "complete-preserved"],
)
def test_v2_import_uses_actual_preserved_continuation_state(
    tmp_path: Path,
    chachanotes_template_db: Path,
    checkpoint: dict,
    persisted_state: str,
    expected_state: str,
) -> None:
    source_paths, conversation_id, _ = _source_graph(
        tmp_path, chachanotes_template_db
    )
    export_path = _create_export(tmp_path, source_paths, conversation_id)

    def mutate(conversation: dict) -> None:
        selected = next(
            message
            for message in conversation["messages"]
            if message["role"] == "assistant" and message["is_selected_variant"]
        )
        selected["_private"] = {"provider_continuation": checkpoint}
        selected["assistant_generation_state"] = persisted_state

    rewritten = _rewrite_export(
        export_path,
        tmp_path / f"{expected_state}.chatbook.zip",
        mutate_conversation=mutate,
    )
    destination_path = tmp_path / f"destination-{expected_state}.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / f"imports-{expected_state}"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(
        rewritten,
        conflict_resolution=ConflictResolution.RENAME,
        import_status=ImportStatus(),
    )

    assert success, message
    destination = CharactersRAGDB(str(destination_path), "assertion")
    try:
        imported = destination.execute_query(
            "SELECT assistant_generation_state, provider_continuation_json "
            "FROM messages WHERE provider_continuation_json IS NOT NULL"
        ).fetchone()
        assert imported is not None
        assert imported["assistant_generation_state"] == expected_state
        assert (
            parse_provider_continuation_json(
                imported["provider_continuation_json"]
            ).state
            == checkpoint["state"]
        )
    finally:
        destination.close_connection()


def test_v2_import_rejects_continuation_active_with_complete_checkpoint(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_paths, conversation_id, _ = _source_graph(
        tmp_path, chachanotes_template_db
    )
    export_path = _create_export(tmp_path, source_paths, conversation_id)

    def mutate(conversation: dict) -> None:
        selected = next(
            message
            for message in conversation["messages"]
            if message["role"] == "assistant" and message["is_selected_variant"]
        )
        selected["_private"] = {
            "provider_continuation": _kimi_family_private("same visible answer")
        }
        selected["assistant_generation_state"] = "continuation_active"

    rewritten = _rewrite_export(
        export_path,
        tmp_path / "incompatible-state.chatbook.zip",
        mutate_conversation=mutate,
    )
    destination_path = tmp_path / "destination-incompatible.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "imports-incompatible"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(
        rewritten,
        conflict_resolution=ConflictResolution.RENAME,
        import_status=ImportStatus(),
    )

    assert success is False
    assert message == "Failed to import any items from chatbook"


def test_v2_import_rejects_continuation_active_without_checkpoint(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_paths, conversation_id, _ = _source_graph(
        tmp_path, chachanotes_template_db
    )
    export_path = _create_export(tmp_path, source_paths, conversation_id)

    def remove_checkpoint(conversation: dict) -> None:
        selected = next(
            message
            for message in conversation["messages"]
            if message["role"] == "assistant" and message["is_selected_variant"]
        )
        selected.pop("_private")
        selected["assistant_generation_state"] = "continuation_active"

    rewritten = _rewrite_export(
        export_path,
        tmp_path / "missing-active-checkpoint.chatbook.zip",
        mutate_conversation=remove_checkpoint,
    )
    destination_path = tmp_path / "destination-missing-active.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "imports-missing-active"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(
        rewritten,
        conflict_resolution=ConflictResolution.RENAME,
        import_status=ImportStatus(),
    )

    assert success is False
    assert message == "Failed to import any items from chatbook"


def _private_mutations() -> list:
    def payload(conversation):
        return next(
            message["_private"]["provider_continuation"]
            for message in conversation["messages"]
            if "_private" in message
        )

    def bad_version(conversation):
        payload(conversation)["schema_version"] = 99

    def bad_provider(conversation):
        payload(conversation)["provider"] = "openai"

    def bad_protocol(conversation):
        payload(conversation)["protocol"] = "unknown"

    def bad_pairing(conversation):
        payload(conversation)["rounds"][0]["calls"][0]["state"] = "completed"

    def bad_bound(conversation):
        payload(conversation)["model"] = "x" * 4097

    return [bad_version, bad_provider, bad_protocol, bad_pairing, bad_bound]


@pytest.mark.parametrize("mutate_private", _private_mutations())
def test_invalid_private_data_is_dropped_while_visible_graph_imports(
    tmp_path: Path,
    chachanotes_template_db: Path,
    mutate_private,
) -> None:
    source_paths, conversation_id, _ = _source_graph(tmp_path, chachanotes_template_db)
    export_path = _create_export(tmp_path, source_paths, conversation_id)

    def break_private_on_closed_owner(conversation: dict) -> None:
        selected = next(
            message
            for message in conversation["messages"]
            if message["role"] == "assistant" and message["is_selected_variant"]
        )
        selected["assistant_generation_state"] = "complete"
        mutate_private(conversation)

    broken = _rewrite_export(
        export_path,
        tmp_path / "broken.chatbook.zip",
        mutate_conversation=break_private_on_closed_owner,
    )
    destination_path = tmp_path / "destination.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    status = ImportStatus()
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "imports"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(broken, import_status=status)

    assert success, message
    assert status.warnings == ["Exact tool continuation was discarded for message 3."]
    diagnostic = json.dumps(status.to_dict())
    assert PRIVATE_REASONING not in diagnostic
    assert PRIVATE_ARGUMENTS not in diagnostic
    destination = CharactersRAGDB(str(destination_path), "verify")
    imported_id = str(destination.get_conversation_by_name("Graph")[0]["id"])
    rows = destination.execute_query(
        "SELECT content, provider_continuation_json FROM messages "
        "WHERE conversation_id = ?",
        (imported_id,),
    ).fetchall()
    assert len(rows) == 4
    assert all(row["provider_continuation_json"] is None for row in rows)


def _invalid_graph_mutations() -> list:
    def duplicate_id(conversation):
        conversation["messages"][1]["id"] = conversation["messages"][0]["id"]

    def dangling_parent(conversation):
        conversation["messages"][1]["parent_id"] = "missing-parent"

    def ambiguous_variant(conversation):
        conversation["messages"][2]["variant_of"] = conversation["messages"][2]["id"]

    def duplicate_order(conversation):
        conversation["messages"][1]["order"] = conversation["messages"][0]["order"]

    def deleted_active_ancestor(conversation):
        conversation["messages"][0]["deleted"] = True

    return [
        duplicate_id,
        dangling_parent,
        ambiguous_variant,
        duplicate_order,
        deleted_active_ancestor,
    ]


@pytest.mark.parametrize("mutate_graph", _invalid_graph_mutations())
def test_invalid_v2_graph_fails_without_partial_import(
    tmp_path: Path,
    chachanotes_template_db: Path,
    mutate_graph,
) -> None:
    source_paths, conversation_id, _ = _source_graph(tmp_path, chachanotes_template_db)
    export_path = _create_export(tmp_path, source_paths, conversation_id)
    broken = _rewrite_export(
        export_path,
        tmp_path / "broken-graph.chatbook.zip",
        mutate_conversation=mutate_graph,
    )
    destination_path = tmp_path / "destination.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    status = ImportStatus()
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "imports"
    importer.temp_dir.mkdir()

    success, _ = importer.import_chatbook(broken, import_status=status)

    assert success is False
    assert status.successful_items == 0
    assert len(status.errors) == 1
    assert "Invalid V2 conversation graph." in status.errors[0]
    diagnostic = json.dumps(status.to_dict())
    assert PRIVATE_REASONING not in diagnostic
    assert PRIVATE_ARGUMENTS not in diagnostic
    destination = CharactersRAGDB(str(destination_path), "verify")
    assert destination.get_conversation_by_name("Graph") == []


@pytest.mark.parametrize("identity_case", ["mismatch", "missing", "numeric", "blank"])
def test_v2_manifest_item_identity_mismatch_fails_without_partial_import(
    tmp_path: Path, chachanotes_template_db: Path, identity_case
) -> None:
    source_paths, conversation_id, _ = _source_graph(tmp_path, chachanotes_template_db)
    export_path = _create_export(tmp_path, source_paths, conversation_id)

    def mutate_manifest(manifest):
        item = next(
            item for item in manifest["content_items"] if item["type"] == "conversation"
        )
        if identity_case == "missing":
            item["id"] = "None"
        elif identity_case == "numeric":
            item["id"] = 1
        elif identity_case == "blank":
            item["id"] = ""

    def mutate_conversation(conversation):
        if identity_case == "mismatch":
            conversation["id"] = "other"
        elif identity_case == "missing":
            conversation.pop("id")
        elif identity_case == "numeric":
            conversation["id"] = 1
        elif identity_case == "blank":
            conversation["id"] = ""

    broken = _rewrite_export(
        export_path,
        tmp_path / "identity-mismatch.chatbook.zip",
        mutate_manifest=mutate_manifest,
        mutate_conversation=mutate_conversation,
    )
    destination_path = tmp_path / "destination.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    status = ImportStatus()
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "imports"
    importer.temp_dir.mkdir()

    success, _ = importer.import_chatbook(broken, import_status=status)

    assert success is False
    assert status.successful_items == 0
    destination = CharactersRAGDB(str(destination_path), "verify")
    assert destination.get_conversation_by_name("Graph") == []


def _linear_graph(count: int) -> dict:
    return {
        "messages": [
            {
                "id": f"m-{index}",
                "parent_id": None if index == 0 else f"m-{index - 1}",
                "variant_of": None,
                "order": index,
                "role": "user" if index % 2 == 0 else "assistant",
                "content": "x",
                "deleted": False,
                "variant_number": 1,
                "is_selected_variant": True,
                "total_variants": 1,
            }
            for index in range(count)
        ],
        "active_leaf_message_id": f"m-{count - 1}",
        "selected_path_message_ids": [f"m-{index}" for index in range(count)],
    }


def test_v2_graph_validates_long_chain_in_near_linear_time(monkeypatch) -> None:
    import tldw_chatbook.Chatbooks.chatbook_importer as importer_module

    count = 4_000
    monkeypatch.setattr(importer_module, "_MAX_V2_GRAPH_DEPTH", count)
    graph = _linear_graph(count)

    started = time.perf_counter()
    ordered = ChatbookImporter._validate_v2_conversation_graph(graph)

    assert len(ordered) == count
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    ("constant", "limit"),
    [
        ("_MAX_V2_GRAPH_MESSAGES", 0),
        ("_MAX_V2_MESSAGE_ID_CHARS", 2),
        ("_MAX_V2_TOTAL_ID_CHARS", 2),
        ("_MAX_V2_MESSAGE_CONTENT_CHARS", 0),
        ("_MAX_V2_TOTAL_CONTENT_CHARS", 0),
        ("_MAX_V2_GRAPH_DEPTH", 0),
    ],
)
def test_v2_graph_rejects_resource_bounds(monkeypatch, constant, limit) -> None:
    import tldw_chatbook.Chatbooks.chatbook_importer as importer_module

    monkeypatch.setattr(importer_module, constant, limit)
    with pytest.raises(ValueError, match="Invalid V2 conversation graph"):
        ChatbookImporter._validate_v2_conversation_graph(_linear_graph(1))


def test_v2_graph_accepts_exact_resource_limits(monkeypatch) -> None:
    import tldw_chatbook.Chatbooks.chatbook_importer as importer_module

    monkeypatch.setattr(importer_module, "_MAX_V2_GRAPH_MESSAGES", 1)
    monkeypatch.setattr(importer_module, "_MAX_V2_MESSAGE_ID_CHARS", 3)
    monkeypatch.setattr(importer_module, "_MAX_V2_MESSAGE_CONTENT_CHARS", 1)
    monkeypatch.setattr(importer_module, "_MAX_V2_TOTAL_CONTENT_CHARS", 1)
    monkeypatch.setattr(importer_module, "_MAX_V2_GRAPH_DEPTH", 1)

    assert len(ChatbookImporter._validate_v2_conversation_graph(_linear_graph(1))) == 1


def test_v2_private_bound_counts_utf8_not_ascii_escapes(monkeypatch) -> None:
    import tldw_chatbook.Chatbooks.chatbook_importer as importer_module

    graph = _linear_graph(1)
    checkpoint = json.loads(_checkpoint_json())
    checkpoint["rounds"][0]["reasoning_blocks"] = ["😀"]
    private = {"provider_continuation": checkpoint}
    graph["messages"][0]["role"] = "assistant"
    graph["messages"][0]["_private"] = private
    monkeypatch.setattr(
        importer_module,
        "_MAX_V2_TOTAL_PRIVATE_BYTES",
        len(json.dumps(private, separators=(",", ":"), ensure_ascii=False).encode()),
    )

    assert len(ChatbookImporter._validate_v2_conversation_graph(graph)) == 1


def test_v2_oversize_private_drops_without_rejecting_visible_graph(
    tmp_path: Path, chachanotes_template_db: Path, monkeypatch
) -> None:
    import tldw_chatbook.Chatbooks.chatbook_importer as importer_module

    source_paths, conversation_id, _ = _source_graph(tmp_path, chachanotes_template_db)
    export_path = _create_export(tmp_path, source_paths, conversation_id)

    def close_private_owner(conversation: dict) -> None:
        selected = next(
            message
            for message in conversation["messages"]
            if message["role"] == "assistant" and message["is_selected_variant"]
        )
        selected["assistant_generation_state"] = "complete"

    export_path = _rewrite_export(
        export_path,
        tmp_path / "closed-state-private-bound.chatbook.zip",
        mutate_conversation=close_private_owner,
    )
    monkeypatch.setattr(importer_module, "_MAX_V2_TOTAL_PRIVATE_BYTES", 1)
    destination_path = tmp_path / "destination-private-bound.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    status = ImportStatus()
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "imports-private-bound"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(export_path, import_status=status)

    assert success, message
    assert status.warnings == ["Exact tool continuation was discarded for message 3."]
    destination = CharactersRAGDB(str(destination_path), "verify")
    imported_id = str(destination.get_conversation_by_name("Graph")[0]["id"])
    rows = destination.execute_query(
        "SELECT provider_continuation_json FROM messages WHERE conversation_id = ?",
        (imported_id,),
    ).fetchall()
    assert len(rows) == 4
    assert all(row["provider_continuation_json"] is None for row in rows)


def test_v2_oversize_private_rejects_continuation_active_owner(
    tmp_path: Path, chachanotes_template_db: Path, monkeypatch
) -> None:
    import tldw_chatbook.Chatbooks.chatbook_importer as importer_module

    source_paths, conversation_id, _ = _source_graph(tmp_path, chachanotes_template_db)
    export_path = _create_export(tmp_path, source_paths, conversation_id)
    monkeypatch.setattr(importer_module, "_MAX_V2_TOTAL_PRIVATE_BYTES", 1)
    destination_path = tmp_path / "destination-active-private-bound.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "imports-active-private-bound"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(
        export_path, import_status=ImportStatus()
    )

    assert success is False
    assert message == "Failed to import any items from chatbook"


def test_v2_deep_private_reaches_canonical_parser_before_discard(
    tmp_path: Path, chachanotes_template_db: Path, monkeypatch
) -> None:
    import tldw_chatbook.Chatbooks.chatbook_importer as importer_module

    source_paths, conversation_id, _ = _source_graph(tmp_path, chachanotes_template_db)
    export_path = _create_export(tmp_path, source_paths, conversation_id)

    def make_private_deep(conversation):
        conversation["messages"][2]["assistant_generation_state"] = "complete"
        private_value = "PRIVATE-DEEP-V2-CANARY"
        for _ in range(40):
            private_value = [private_value]
        conversation["messages"][2]["_private"]["provider_continuation"] = private_value

    broken = _rewrite_export(
        export_path,
        tmp_path / "deep-private.chatbook.zip",
        mutate_conversation=make_private_deep,
    )

    def forbidden_raw_private_dump(*_args, **_kwargs):
        raise AssertionError("graph validation must not serialize raw private data")

    monkeypatch.setattr(importer_module.json, "dumps", forbidden_raw_private_dump)
    destination_path = tmp_path / "destination-deep-private.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    status = ImportStatus()
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "imports-deep-private"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(broken, import_status=status)

    assert success, message
    assert status.warnings == ["Exact tool continuation was discarded for message 3."]
    destination = CharactersRAGDB(str(destination_path), "verify")
    imported_id = str(destination.get_conversation_by_name("Graph")[0]["id"])
    assert (
        destination.execute_query(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (imported_id,)
        ).fetchone()[0]
        == 4
    )


def test_v1_flat_import_without_private_data_remains_supported(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_paths, conversation_id, _ = _source_graph(tmp_path, chachanotes_template_db)
    export_path = _create_export(tmp_path, source_paths, conversation_id)

    def make_v1(manifest):
        manifest["version"] = "1.0"

    def flatten(conversation):
        conversation["messages"] = [
            {"role": "user", "content": "legacy visible user"},
            {"role": "assistant", "content": "legacy visible assistant"},
        ]
        conversation.pop("active_leaf_message_id", None)
        conversation.pop("selected_path_message_ids", None)

    legacy = _rewrite_export(
        export_path,
        tmp_path / "legacy.chatbook.zip",
        mutate_manifest=make_v1,
        mutate_conversation=flatten,
    )
    destination_path = tmp_path / "destination.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    status = ImportStatus()
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / "imports"
    importer.temp_dir.mkdir()

    success, message = importer.import_chatbook(legacy, import_status=status)

    assert success, message
    assert status.warnings == []
    destination = CharactersRAGDB(str(destination_path), "verify-legacy-state")
    imported_id = str(destination.get_conversation_by_name("Graph")[0]["id"])
    rows = destination.execute_query(
        "SELECT assistant_generation_state FROM messages WHERE conversation_id = ?",
        (imported_id,),
    ).fetchall()
    assert len(rows) == 2
    assert all(row["assistant_generation_state"] is None for row in rows)
