"""Chatbook export/import carries message images (TASK-221).

Save Chatbook used to export only id/role/content/timestamp per message — a
Console conversation with image messages lost them (PR #621 rider). Both
storage tiers now round-trip: the legacy columns (position 0) and the v19
``message_attachments`` table (positions >= 1), as files under
``content/conversations/attachments/`` referenced by an ``attachments`` list
on the message JSON. Old chatbooks (no ``attachments`` key) import unchanged.
"""

import json
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

PNG_LEGACY = b"png-bytes-position-0"
PNG_POS1 = b"png-bytes-position-1"
PNG_POS2 = b"png-bytes-position-2"


@pytest.fixture
def source_env(tmp_path):
    """A source DB holding one conversation whose message carries 3 images."""
    db_dir = tmp_path / "source"
    db_dir.mkdir()
    db_paths = {
        "ChaChaNotes": str(db_dir / "chachanotes.db"),
        "Prompts": str(db_dir / "prompts.db"),
        "Media": str(db_dir / "media.db"),
        "Evals": str(db_dir / "evals.db"),
        "RAG": str(db_dir / "rag.db"),
    }
    db = CharactersRAGDB(db_paths["ChaChaNotes"], "test-source")
    conv_id = db.add_conversation({
        "title": "Image Round Trip",
    })
    db.add_message({
        "conversation_id": conv_id,
        "sender": "user",
        "content": "look at these",
        "timestamp": datetime.now().isoformat(),
        "image_data": PNG_LEGACY,
        "image_mime_type": "image/png",
    })
    messages = db.get_messages_for_conversation(conv_id)
    message_id = str(messages[0]["id"])
    db.set_message_attachments(message_id, [
        {
            "position": 1,
            "data": PNG_POS1,
            "mime_type": "image/png",
            "display_name": "second.png",
        },
        {
            "position": 2,
            "data": PNG_POS2,
            "mime_type": "image/jpeg",
            "display_name": "third.jpg",
        },
    ])
    return db_paths, conv_id


def _create_chatbook(db_paths, conv_id, tmp_path) -> Path:
    output = tmp_path / "round-trip.zip"
    creator = ChatbookCreator(db_paths)
    ok, message, _details = creator.create_chatbook(
        name="Image Round Trip Book",
        description="round trip",
        content_selections={ContentType.CONVERSATION: [conv_id]},
        output_path=output,
        auto_include_dependencies=False,
    )
    assert ok, message
    return output


def test_export_writes_attachment_files_and_manifest_entries(source_env, tmp_path):
    db_paths, conv_id = source_env
    output = _create_chatbook(db_paths, conv_id, tmp_path)

    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
        conv_json_name = next(
            name for name in names
            if name.startswith("content/conversations/") and name.endswith(".json")
        )
        conv_data = json.loads(zf.read(conv_json_name))
        message = conv_data["messages"][0]
        attachments = message.get("attachments")
        assert attachments is not None, "message JSON lost its attachments list"
        assert [entry["position"] for entry in attachments] == [0, 1, 2]
        by_position = {entry["position"]: entry for entry in attachments}
        assert by_position[1]["display_name"] == "second.png"
        assert by_position[2]["mime_type"] == "image/jpeg"
        assert zf.read(by_position[0]["file"]) == PNG_LEGACY
        assert zf.read(by_position[1]["file"]) == PNG_POS1
        assert zf.read(by_position[2]["file"]) == PNG_POS2


def test_import_restores_both_storage_tiers(source_env, tmp_path):
    db_paths, conv_id = source_env
    output = _create_chatbook(db_paths, conv_id, tmp_path)

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    dest_paths = {
        "ChaChaNotes": str(dest_dir / "chachanotes.db"),
        "Prompts": str(dest_dir / "prompts.db"),
        "Media": str(dest_dir / "media.db"),
        "Evals": str(dest_dir / "evals.db"),
        "RAG": str(dest_dir / "rag.db"),
    }
    importer = ChatbookImporter(dest_paths)
    ok, message = importer.import_chatbook(
        output,
        content_selections={ContentType.CONVERSATION: [conv_id]},
        prefix_imported=False,
    )
    assert ok, message

    dest_db = CharactersRAGDB(dest_paths["ChaChaNotes"], "test-dest")
    conversations = dest_db.get_conversation_by_name("Image Round Trip")
    assert conversations, "conversation missing after import"
    imported_messages = dest_db.get_messages_for_conversation(conversations[0]["id"])
    assert len(imported_messages) == 1
    imported = imported_messages[0]
    assert imported["image_data"] == PNG_LEGACY
    assert imported["image_mime_type"] == "image/png"
    extra = dest_db.get_attachments_for_messages([str(imported["id"])])
    rows = extra.get(str(imported["id"]), [])
    assert [(r["position"], r["data"], r["display_name"]) for r in rows] == [
        (1, PNG_POS1, "second.png"),
        (2, PNG_POS2, "third.jpg"),
    ]


def test_import_skips_traversal_attachment_paths(source_env, tmp_path):
    """A malicious chatbook naming a file outside the extraction root must be
    skipped (with the message still imported), never read."""
    db_paths, conv_id = source_env
    output = _create_chatbook(db_paths, conv_id, tmp_path)

    # Rewrite the conversation JSON so one attachment escapes the archive root.
    tampered = tmp_path / "tampered.zip"
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"outside-the-archive")
    with zipfile.ZipFile(output) as src, zipfile.ZipFile(tampered, "w") as dst:
        for item in src.infolist():
            payload = src.read(item.filename)
            if item.filename.startswith("content/conversations/") and item.filename.endswith(".json"):
                conv_data = json.loads(payload)
                conv_data["messages"][0]["attachments"][1]["file"] = "../../secret.bin"
                payload = json.dumps(conv_data).encode()
            dst.writestr(item, payload)

    dest_dir = tmp_path / "dest-tampered"
    dest_dir.mkdir()
    dest_paths = {
        "ChaChaNotes": str(dest_dir / "chachanotes.db"),
        "Prompts": str(dest_dir / "prompts.db"),
        "Media": str(dest_dir / "media.db"),
        "Evals": str(dest_dir / "evals.db"),
        "RAG": str(dest_dir / "rag.db"),
    }
    importer = ChatbookImporter(dest_paths)
    ok, message = importer.import_chatbook(
        tampered,
        content_selections={ContentType.CONVERSATION: [conv_id]},
    )
    assert ok, message
    dest_db = CharactersRAGDB(dest_paths["ChaChaNotes"], "test-dest")
    conversations = dest_db.get_conversation_by_name("Image Round Trip")
    assert conversations
    imported_messages = dest_db.get_messages_for_conversation(conversations[0]["id"])
    assert len(imported_messages) == 1
    rows = dest_db.get_attachments_for_messages(
        [str(imported_messages[0]["id"])]
    ).get(str(imported_messages[0]["id"]), [])
    # position 1 (the tampered entry) skipped; position 0 and 2 restored
    assert [r["position"] for r in rows] == [2]
    assert imported_messages[0]["image_data"] == PNG_LEGACY


def test_chatbook_without_attachments_key_imports_unchanged(source_env, tmp_path):
    """Backward compatibility: pre-TASK-221 chatbooks have no attachments key."""
    db_paths, conv_id = source_env
    output = _create_chatbook(db_paths, conv_id, tmp_path)

    stripped = tmp_path / "legacy.zip"
    with zipfile.ZipFile(output) as src, zipfile.ZipFile(stripped, "w") as dst:
        for item in src.infolist():
            if item.filename.startswith("content/conversations/attachments/"):
                continue
            payload = src.read(item.filename)
            if item.filename.startswith("content/conversations/") and item.filename.endswith(".json"):
                conv_data = json.loads(payload)
                for message in conv_data["messages"]:
                    message.pop("attachments", None)
                payload = json.dumps(conv_data).encode()
            dst.writestr(item, payload)

    dest_dir = tmp_path / "dest-legacy"
    dest_dir.mkdir()
    dest_paths = {
        "ChaChaNotes": str(dest_dir / "chachanotes.db"),
        "Prompts": str(dest_dir / "prompts.db"),
        "Media": str(dest_dir / "media.db"),
        "Evals": str(dest_dir / "evals.db"),
        "RAG": str(dest_dir / "rag.db"),
    }
    importer = ChatbookImporter(dest_paths)
    ok, message = importer.import_chatbook(
        stripped,
        content_selections={ContentType.CONVERSATION: [conv_id]},
    )
    assert ok, message
    dest_db = CharactersRAGDB(dest_paths["ChaChaNotes"], "test-dest")
    conversations = dest_db.get_conversation_by_name("Image Round Trip")
    assert conversations
    imported_messages = dest_db.get_messages_for_conversation(conversations[0]["id"])
    assert len(imported_messages) == 1
    assert imported_messages[0]["content"] == "look at these"
