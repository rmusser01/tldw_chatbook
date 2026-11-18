"""Server character portraits survive the public Chatbook import boundary."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.bootstrap_profile


# Hand-checked base64 for bytes b'\x00\xffportrait\x80', independent of the importer.
PORTRAIT = b"\x00\xffportrait\x80"
ENCODED_PORTRAIT = "AP9wb3J0cmFpdIA="
CHARACTER_NAME = "Server portrait character"


@pytest.fixture
def character_db_path(tmp_path, chachanotes_template_db):
    """Use the normal schema with fresh, real SQLite storage for every test."""
    path = tmp_path / "characters.db"
    shutil.copyfile(chachanotes_template_db, path)
    return path


def _server_character_archive(
    tmp_path: Path, *, version: str = "2.0", **image_fields
) -> Path:
    """Mirror the server's flat character JSON and manifest field layout."""
    archive_path = tmp_path / "server-character.zip"
    manifest = {
        "version": version,
        "name": "Server portrait import",
        "description": "Character compatibility fixture",
        "created_at": "2026-10-03T00:00:00",
        "updated_at": "2026-10-03T00:00:00",
        "author": None,
        "export_id": "portrait-fixture-export",
        "relationships": [],
        "configuration": {
            "include_media": False,
            "include_embeddings": False,
            "include_generated_content": True,
            "media_quality": "compressed",
            "max_file_size_mb": 100,
        },
        "statistics": {
            "total_conversations": 0,
            "total_notes": 0,
            "total_characters": 1,
            "total_media_items": 0,
            "total_prompts": 0,
            "total_evaluations": 0,
            "total_embeddings": 0,
            "total_world_books": 0,
            "total_dictionaries": 0,
            "total_documents": 0,
            "total_explainer_sessions": 0,
            "total_size_bytes": 0,
        },
        "metadata": {"tags": [], "categories": [], "language": "en", "license": None},
        "user_info": {"user_id": "portrait-fixture-user"},
        "content_items": [
            {
                "id": "42",
                "type": "character",
                "title": CHARACTER_NAME,
                "description": None,
                "created_at": None,
                "updated_at": None,
                "tags": [],
                "metadata": {},
                "file_path": "content/characters/character_42.json",
                "checksum": None,
            }
        ],
    }
    character = {
        "id": 42,
        "name": CHARACTER_NAME,
        "description": "A character exported by the server",
        "personality": "Helpful",
        "scenario": "Archive import",
        **image_fields,
    }
    character_json = json.dumps(character, indent=2, ensure_ascii=False).encode("utf-8")
    if version == "1.1.0":
        manifest.update(
            {
                "features_used": [
                    "content_envelopes",
                    "file_inventory",
                    "integrity_metadata",
                    "representations",
                    "lossiness_metadata",
                ],
                "producer": {"name": "tldw_server"},
                "source_instance": {},
                "compatibility": {
                    "min_reader_version": "1.1.0",
                    "recommended_reader_version": "1.1.0",
                    "unsupported_feature_behavior": "warn_lossy_import",
                    "v1_compatibility": {
                        "fallback": (
                            "Readers that only support v1.0 may use the core manifest fields "
                            "and ignore v1.1 metadata, with possible loss of representation "
                            "and integrity details."
                        ),
                    },
                },
                "file_inventory": [
                    {
                        "path": "content/characters/character_42.json",
                        "media_type": "application/json",
                        "size_bytes": len(character_json),
                        "integrity": {
                            "status": "verified",
                            "algorithm": "sha256",
                            "value": f"sha256:{hashlib.sha256(character_json).hexdigest()}",
                        },
                        "role": "payload",
                        "content_item_ids": [],
                    }
                ],
            }
        )
        manifest["statistics"].update(
            {
                "account_profiles": 0,
                "account_settings": 0,
                "conversations": 0,
                "notes": 0,
                "characters": 1,
                "media_records": 0,
                "prompts": 0,
                "evaluations": 0,
                "embeddings": 0,
                "world_books": 0,
                "dictionaries": 0,
                "generated_documents": 0,
                "explainer_sessions": 0,
                "media_transcripts": 0,
                "media_chunks": 0,
                "media_stored_artifacts": 0,
                "media_pointers": 0,
                "tags_categories_relationships": 0,
                "sensitive_user_values": 0,
            }
        )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("content/characters/character_42.json", character_json)
    return archive_path


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "2.0"])
def test_public_import_restores_server_portrait_bytes(
    character_db_path, tmp_path, version
):
    archive = _server_character_archive(
        tmp_path, version=version, image=ENCODED_PORTRAIT, image_encoding="base64"
    )
    status = ImportStatus()
    success, message = ChatbookImporter(
        {"ChaChaNotes": str(character_db_path)}
    ).import_chatbook(archive, import_status=status)
    assert success, message
    assert status.successful_items == 1

    # Open a new database owner after import so this asserts durable storage.
    db = CharactersRAGDB(character_db_path, "portrait-readback")
    try:
        card = db.get_character_card_by_name(CHARACTER_NAME)
        assert card["image"] == PORTRAIT
    finally:
        db.close_connection()


@pytest.mark.parametrize("version", ["1.1", "1.2.0", "2.0.0", "9.0.0"])
def test_unknown_manifest_version_is_rejected(character_db_path, tmp_path, version):
    archive = _server_character_archive(
        tmp_path, version=version, image=ENCODED_PORTRAIT, image_encoding="base64"
    )
    status = ImportStatus()
    success, _ = ChatbookImporter(
        {"ChaChaNotes": str(character_db_path)}
    ).import_chatbook(archive, import_status=status)
    assert not success
    assert status.errors
    assert status.successful_items == 0

    db = CharactersRAGDB(character_db_path, "portrait-readback")
    try:
        assert db.get_character_card_by_name(CHARACTER_NAME) is None
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "image_fields",
    [
        {"image": "AP9!wb3J0cmFpdIA=", "image_encoding": "base64"},
        {"image": "AP9wb3J0cmFpdIA", "image_encoding": "base64"},
        {"image": "AP9wb3J0cmFpdIA=\n", "image_encoding": "base64"},
        {"image": "非ASCII", "image_encoding": "base64"},
        {"image": None, "image_encoding": "base64"},
        {"image": 123, "image_encoding": "base64"},
        {"image_encoding": "base64"},
        {"image": ENCODED_PORTRAIT, "image_encoding": "hex"},
        {"image": ENCODED_PORTRAIT, "image_encoding": None},
    ],
)
def test_invalid_encoded_portrait_fails_without_creating_character(
    character_db_path, tmp_path, image_fields
):
    archive = _server_character_archive(tmp_path, **image_fields)
    status = ImportStatus()
    success, _ = ChatbookImporter(
        {"ChaChaNotes": str(character_db_path)}
    ).import_chatbook(archive, import_status=status)
    assert not success
    assert status.failed_items == 1
    assert status.errors

    db = CharactersRAGDB(character_db_path, "portrait-readback")
    try:
        assert db.get_character_card_by_name(CHARACTER_NAME) is None
    finally:
        db.close_connection()


@pytest.mark.parametrize("image_fields", [{}, {"image": None}])
def test_legacy_character_without_portrait_still_imports(
    character_db_path, tmp_path, image_fields
):
    archive = _server_character_archive(tmp_path, **image_fields)
    success, message = ChatbookImporter(
        {"ChaChaNotes": str(character_db_path)}
    ).import_chatbook(archive)
    assert success, message

    db = CharactersRAGDB(character_db_path, "portrait-readback")
    try:
        card = db.get_character_card_by_name(CHARACTER_NAME)
        assert card["description"] == "A character exported by the server"
        assert card["image"] is None
    finally:
        db.close_connection()


def test_renamed_import_preserves_portrait_and_existing_character(
    character_db_path, tmp_path
):
    db = CharactersRAGDB(character_db_path, "portrait-existing")
    try:
        original_id = db.add_character_card(
            {"name": CHARACTER_NAME, "image": b"existing-portrait"}
        )
    finally:
        db.close_connection()
    archive = _server_character_archive(
        tmp_path, image=ENCODED_PORTRAIT, image_encoding="base64"
    )
    success, message = ChatbookImporter(
        {"ChaChaNotes": str(character_db_path)}
    ).import_chatbook(archive, conflict_resolution=ConflictResolution.RENAME)
    assert success, message

    db = CharactersRAGDB(character_db_path, "portrait-readback")
    try:
        assert db.get_character_card_by_id(original_id)["image"] == b"existing-portrait"
        renamed = db.get_character_card_by_name(f"{CHARACTER_NAME} (1)")
        assert renamed["image"] == PORTRAIT
    finally:
        db.close_connection()
