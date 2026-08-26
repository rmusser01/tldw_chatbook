"""Real Chatbook Prompt export/import contract tests for TASK-197."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

import tldw_chatbook.Chatbooks.chatbook_creator as creator_module
import tldw_chatbook.Chatbooks.chatbook_importer as importer_module
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ContentItem,
    ContentType,
)
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_chatbook_record import (
    CHATBOOK_PROMPT_RECORD_KEYS,
    decode_chatbook_prompt_record,
)
from tldw_chatbook.Prompt_Management.prompt_scope_service import LocalPromptService


def _creator(database_path: Path, runtime_root: Path) -> ChatbookCreator:
    creator = ChatbookCreator({"Prompts": str(database_path)})
    creator.temp_dir = runtime_root / "creator"
    creator.temp_dir.mkdir(parents=True, exist_ok=True)
    return creator


def _importer(database_path: Path, runtime_root: Path) -> ChatbookImporter:
    importer = ChatbookImporter({"Prompts": str(database_path)})
    importer.temp_dir = runtime_root / "importer"
    importer.temp_dir.mkdir(parents=True, exist_ok=True)
    return importer


def _add_prompt(database: PromptsDatabase, **overrides: Any) -> tuple[int, str]:
    fields: dict[str, Any] = {
        "name": "Prompt",
        "author": "Author",
        "details": "Details",
        "system_prompt": "System",
        "user_prompt": "User",
        "keywords": ["Zulu", "alpha"],
        "prompt_format": "legacy",
        "prompt_schema_version": None,
        "prompt_definition": None,
        "artifact_type": "prompt",
    }
    fields.update(overrides)
    prompt_id, prompt_uuid, _ = database.add_prompt(**fields)
    assert prompt_id is not None
    assert prompt_uuid is not None
    return prompt_id, prompt_uuid


def _write_prompt_chatbook(
    archive_path: Path,
    payload: Any,
    *,
    item_id: str = "item-000001",
) -> None:
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Prompt fixture",
        description="Prompt fixture",
    )
    manifest.content_items.append(
        ContentItem(
            id=item_id,
            type=ContentType.PROMPT,
            title="Fixture Prompt",
            file_path=f"content/prompts/prompt_{item_id}.json",
        )
    )
    manifest.total_prompts = 1
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest.to_dict()))
        archive.writestr(
            f"content/prompts/prompt_{item_id}.json",
            json.dumps(payload, ensure_ascii=False),
        )


def _prompt_items(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in manifest["content_items"] if item["type"] == "prompt"]


def test_empty_prompt_selection_remains_a_noop_without_a_prompt_source(
    tmp_path: Path,
) -> None:
    creator = ChatbookCreator({})
    creator.temp_dir = tmp_path / "creator"
    creator.temp_dir.mkdir()
    output_path = tmp_path / "empty.zip"

    success, message, _ = creator.create_chatbook(
        name="Empty",
        description="Empty",
        content_selections={ContentType.PROMPT: []},
        output_path=output_path,
    )

    assert success is True, message
    assert output_path.exists()


def test_real_chatbook_round_trip_preserves_portable_prompt_records_only(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.db"
    destination_path = tmp_path / "destination.db"
    source = PromptsDatabase(source_path, client_id="source-client")
    try:
        legacy_id, legacy_uuid = _add_prompt(
            source,
            name="Legacy [bold]研究🙂",
            author=None,
            details="Initial details",
            system_prompt="System\nline",
            user_prompt="User\nمرحبا",
            keywords=["Zulu", "alpha"],
        )
        updated_id, updated_uuid, _ = source.add_prompt(
            name="Legacy [bold]研究🙂",
            author=None,
            details="Final details",
            system_prompt="System\nline",
            user_prompt="User\nمرحبا",
            keywords=["Zulu", "alpha"],
            overwrite=True,
            prompt_format="legacy",
            prompt_schema_version=None,
            prompt_definition=None,
            artifact_type="prompt",
        )
        assert updated_id == legacy_id
        assert updated_uuid == legacy_uuid
        recipe_definition = (
            '{"kind":"block_prompt","version":2,"literal":"[bold]研究🙂"}'
        )
        _add_prompt(
            source,
            name="Structured Recipe",
            details="",
            system_prompt=None,
            user_prompt="Recipe user lane",
            keywords=["recipe", "Unicode"],
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=recipe_definition,
            artifact_type="recipe",
        )
        foreign_definition = (
            '{ "kind": "foreign_prompt", "version": 1, "x": [3, 2, 1] }'
        )
        _add_prompt(
            source,
            name="Compatibility-only",
            author="Foreign",
            details=None,
            system_prompt="Opaque system",
            user_prompt=None,
            keywords=[],
            prompt_format="structured",
            prompt_schema_version=1,
            prompt_definition=foreign_definition,
            artifact_type="prompt",
        )
        for index in range(50):
            _add_prompt(source, name=f"Paged {index:02d}", keywords=[f"tag-{index}"])
        deleted_id, _ = _add_prompt(source, name="Deleted control")
        assert source.soft_delete_prompt(deleted_id) is True
        collection = LocalPromptService(source).create_prompt_collection(
            {"name": "Private collection", "prompt_ids": [legacy_id]}
        )
        assert type(collection["collection_id"]) is int
        assert source.get_prompt_history_count(legacy_uuid) == 2

        selected_ids = source.get_all_active_prompt_ids()
        source_snapshots = [
            source.fetch_prompt_chatbook_snapshot(prompt_id)
            for prompt_id in selected_ids
        ]
        assert len(source_snapshots) == 53
        assert all(snapshot is not None for snapshot in source_snapshots)

        archive_path = tmp_path / "prompts.zip"
        success, message, _ = _creator(source_path, tmp_path).create_chatbook(
            name="Prompt backup",
            description="Portable Prompts",
            content_selections={
                ContentType.PROMPT: [str(prompt_id) for prompt_id in selected_ids]
            },
            output_path=archive_path,
        )

        assert success is True, message
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            items = _prompt_items(manifest)
            expected_archive_ids = [
                f"item-{index:06d}" for index in range(1, len(selected_ids) + 1)
            ]
            assert [item["id"] for item in items] == expected_archive_ids
            assert [item["file_path"] for item in items] == [
                f"content/prompts/prompt_{item_id}.json"
                for item_id in expected_archive_ids
            ]
            assert all(item["created_at"] is None for item in items)
            assert all(item["updated_at"] is None for item in items)
            assert items[0]["title"] == "Legacy [bold]研究🙂"
            assert items[0]["description"] == "Final details"
            assert manifest["statistics"]["total_prompts"] == 53
            assert not any("history" in name.lower() for name in names)
            assert not any("collection" in name.lower() for name in names)
            payloads = [
                json.loads(archive.read(f"content/prompts/prompt_{item_id}.json"))
                for item_id in expected_archive_ids
            ]

        assert all(
            set(payload) == set(CHATBOOK_PROMPT_RECORD_KEYS) for payload in payloads
        )
        assert all(
            not {
                "id",
                "uuid",
                "client_id",
                "version",
                "last_modified",
                "created_at",
                "updated_at",
                "deleted",
            }
            & set(payload)
            for payload in payloads
        )
        assert [
            decode_chatbook_prompt_record(payload) for payload in payloads
        ] == source_snapshots
        assert payloads[1]["prompt_definition"] == recipe_definition
        assert payloads[2]["prompt_definition"] == foreign_definition

        status = ImportStatus()
        imported, import_message = _importer(
            destination_path, tmp_path
        ).import_chatbook(archive_path, import_status=status)
        assert imported is True, import_message
        assert status.successful_items == 53
        assert status.failed_items == 0
        destination = PromptsDatabase(destination_path, client_id="destination-reader")
        try:
            destination_ids = destination.get_all_active_prompt_ids()
            destination_snapshots = [
                destination.fetch_prompt_chatbook_snapshot(prompt_id)
                for prompt_id in destination_ids
            ]
            assert destination_snapshots == source_snapshots
            source_detail = source.get_prompt_by_id(legacy_id)
            destination_detail = destination.get_prompt_by_id(destination_ids[0])
            assert source_detail is not None
            assert destination_detail is not None
            assert destination_detail["uuid"] != source_detail["uuid"]
            assert destination_detail["client_id"] != source_detail["client_id"]
            assert destination_detail["version"] == 1
            assert destination_detail["last_modified"] != source_detail["last_modified"]
            assert destination.get_prompt_history_count(destination_detail["uuid"]) == 1
            collection_tables = (
                destination.get_connection()
                .execute(
                    """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name LIKE 'LocalPromptCollection%'
                """
                )
                .fetchall()
            )
            assert collection_tables == []
        finally:
            destination.close_connection()
    finally:
        source.close_connection()


@pytest.mark.parametrize(
    "failure_kind", ["missing", "database_init", "database", "encode", "write"]
)
def test_prompt_export_failure_is_atomic_bounded_and_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_kind: str,
) -> None:
    source_path = tmp_path / "source.db"
    source = PromptsDatabase(source_path, client_id="source-client")
    source.get_connection().execute(
        "INSERT OR REPLACE INTO sqlite_sequence(name, seq) VALUES ('Prompts', ?)",
        (971971970,),
    )
    source.get_connection().commit()
    prompt_id, _ = _add_prompt(
        source,
        name="TASK197_PROMPT_NAME_MUST_NOT_LEAK",
        system_prompt="TASK197_PROMPT_BODY_MUST_NOT_LEAK",
    )
    selected_id = prompt_id
    if failure_kind == "missing":
        assert source.soft_delete_prompt(prompt_id) is True
    elif failure_kind == "database_init":

        def fail_database_init(*_args: Any, **_kwargs: Any) -> None:
            raise sqlite3.DatabaseError("TASK197_DATABASE_EXCEPTION_MUST_NOT_LEAK")

        monkeypatch.setattr(creator_module, "PromptsDatabase", fail_database_init)
    elif failure_kind == "database":

        def fail_snapshot(_self: Any, _prompt_id: int) -> None:
            raise sqlite3.DatabaseError("TASK197_DATABASE_EXCEPTION_MUST_NOT_LEAK")

        monkeypatch.setattr(
            PromptsDatabase, "fetch_prompt_chatbook_snapshot", fail_snapshot
        )
    elif failure_kind == "encode":

        def fail_encode(_detail: Mapping[str, Any]) -> None:
            raise RuntimeError("TASK197_ENCODING_CATEGORY_MUST_NOT_LEAK")

        monkeypatch.setattr(
            creator_module,
            "encode_chatbook_prompt_record",
            fail_encode,
            raising=False,
        )
    else:
        real_dump: Callable[..., Any] = creator_module.json.dump

        def fail_prompt_write(value: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(value, Mapping) and value.get("record_schema"):
                raise RuntimeError("TASK197_WRITE_EXCEPTION_MUST_NOT_LEAK")
            return real_dump(value, *args, **kwargs)

        monkeypatch.setattr(creator_module.json, "dump", fail_prompt_write)

    output_path = tmp_path / "existing.zip"
    original = b"PREEXISTING_ARCHIVE_MUST_SURVIVE"
    output_path.write_bytes(original)
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        success, message, _ = _creator(source_path, tmp_path).create_chatbook(
            name="Prompt backup",
            description="Portable Prompts",
            content_selections={ContentType.PROMPT: [str(selected_id)]},
            output_path=output_path,
        )
    finally:
        logger.remove(sink)
        source.close_connection()

    rendered = "\n".join(messages + [record.getMessage() for record in caplog.records])
    assert success is False
    assert message == "Unable to export one or more Prompts."
    assert output_path.read_bytes() == original
    assert not output_path.with_name(output_path.name + ".partial").exists()
    for sentinel in (
        str(selected_id),
        "TASK197_PROMPT_NAME_MUST_NOT_LEAK",
        "TASK197_PROMPT_BODY_MUST_NOT_LEAK",
        "TASK197_DATABASE_EXCEPTION_MUST_NOT_LEAK",
        "TASK197_ENCODING_CATEGORY_MUST_NOT_LEAK",
        "TASK197_WRITE_EXCEPTION_MUST_NOT_LEAK",
        "Traceback",
    ):
        assert sentinel not in message
        assert sentinel not in rendered


def test_legacy_prompt_archive_imports_content_as_system_lane(tmp_path: Path) -> None:
    archive_path = tmp_path / "legacy.zip"
    _write_prompt_chatbook(
        archive_path,
        {
            "id": 917,
            "name": "Legacy Prompt",
            "description": "Legacy details",
            "content": "Legacy System\n[bold]研究🙂",
            "created_at": "2022-03-04T05:06:07",
            "updated_at": None,
        },
        item_id="917",
    )
    destination_path = tmp_path / "destination.db"
    status = ImportStatus()

    success, message = _importer(destination_path, tmp_path).import_chatbook(
        archive_path,
        prefix_imported=True,
        import_status=status,
    )

    assert success is True, message
    assert status.successful_items == 1
    destination = PromptsDatabase(destination_path, client_id="destination-reader")
    try:
        prompt_id = destination.get_all_active_prompt_ids()[0]
        assert destination.fetch_prompt_chatbook_snapshot(prompt_id) == {
            "name": "[Imported] Legacy Prompt",
            "author": None,
            "details": "Legacy details",
            "system_prompt": "Legacy System\n[bold]研究🙂",
            "user_prompt": None,
            "keywords": [],
            "artifact_type": "prompt",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        }
    finally:
        destination.close_connection()


def test_prompt_import_database_initialization_failure_is_bounded_and_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    archive_path = tmp_path / "valid.zip"
    _write_prompt_chatbook(
        archive_path,
        {
            "name": "TASK197_IMPORT_NAME_MUST_NOT_LEAK",
            "description": None,
            "content": "TASK197_IMPORT_BODY_MUST_NOT_LEAK",
        },
    )

    def fail_database_init(*_args: Any, **_kwargs: Any) -> None:
        raise sqlite3.DatabaseError("TASK197_IMPORT_DATABASE_MUST_NOT_LEAK")

    monkeypatch.setattr(importer_module, "PromptsDatabase", fail_database_init)
    status = ImportStatus()
    destination_path = tmp_path / "destination.db"
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        success, message = _importer(destination_path, tmp_path).import_chatbook(
            archive_path, import_status=status
        )
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages + [record.getMessage() for record in caplog.records])
    assert success is False
    assert message == "Failed to import any items from chatbook"
    assert status.processed_items == 1
    assert status.failed_items == 1
    assert status.errors == ["Unable to import Prompt item."]
    assert not destination_path.exists()
    for sentinel in (
        "TASK197_IMPORT_NAME_MUST_NOT_LEAK",
        "TASK197_IMPORT_BODY_MUST_NOT_LEAK",
        "TASK197_IMPORT_DATABASE_MUST_NOT_LEAK",
        "Traceback",
    ):
        assert sentinel not in rendered
        assert sentinel not in message
        assert sentinel not in "\n".join(status.errors)


def test_prompt_import_name_conflict_is_bounded_before_shared_database_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    name = "TASK197_CONFLICT_NAME_MUST_NOT_LEAK"
    body = "TASK197_CONFLICT_BODY_MUST_NOT_LEAK"
    archive_path = tmp_path / "conflict.zip"
    _write_prompt_chatbook(
        archive_path,
        {"name": name, "description": None, "content": body},
    )
    destination_path = tmp_path / "destination.db"
    destination = PromptsDatabase(destination_path, client_id="destination-seed")
    try:
        _add_prompt(destination, name=name, system_prompt="Existing")
    finally:
        destination.close_connection()
    status = ImportStatus()
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        success, message = _importer(destination_path, tmp_path).import_chatbook(
            archive_path,
            import_status=status,
        )
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages + [record.getMessage() for record in caplog.records])
    assert success is False
    assert message == "Failed to import any items from chatbook"
    assert status.processed_items == 1
    assert status.failed_items == 1
    assert status.errors == ["Unable to import Prompt item."]
    for sentinel in (name, body, "Traceback"):
        assert sentinel not in rendered
        assert sentinel not in message
        assert sentinel not in "\n".join(status.errors)


def test_prompt_import_reusing_keyword_does_not_log_archive_value(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    keyword = "task197_archive_keyword_must_not_leak"
    archive_path = tmp_path / "keyword.zip"
    payload = {
        "record_schema": "tldw-chatbook-prompt",
        "record_version": 1,
        "author": None,
        "details": None,
        "system_prompt": "System",
        "user_prompt": None,
        "keywords": [keyword],
        "artifact_type": "prompt",
        "prompt_format": "legacy",
        "prompt_schema_version": None,
        "prompt_definition": None,
    }
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Shared keyword fixture",
        description="Shared keyword fixture",
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        for index in (1, 2):
            item_id = f"item-{index:06d}"
            record = {**payload, "name": f"Imported Prompt {index}"}
            manifest.content_items.append(
                ContentItem(
                    id=item_id,
                    type=ContentType.PROMPT,
                    title=record["name"],
                    file_path=f"content/prompts/prompt_{item_id}.json",
                )
            )
            archive.writestr(
                f"content/prompts/prompt_{item_id}.json",
                json.dumps(record),
            )
        manifest.total_prompts = 2
        archive.writestr("manifest.json", json.dumps(manifest.to_dict()))

    destination_path = tmp_path / "destination.db"

    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        success, message = _importer(destination_path, tmp_path).import_chatbook(
            archive_path
        )
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages + [record.getMessage() for record in caplog.records])
    assert success is True, message
    assert keyword not in rendered
    assert "Traceback" not in rendered


def test_prompt_keyword_failures_log_only_fixed_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = PromptsDatabase(tmp_path / "keywords.db", client_id="keyword-owner")
    sentinel = "task197_keyword_failure_must_not_leak"
    prompt_id, _ = _add_prompt(database, name="Keyword target", keywords=[])

    def fail_keyword_update(*_args: Any, **_kwargs: Any) -> None:
        raise sqlite3.OperationalError(sentinel)

    monkeypatch.setattr(database, "_add_keyword_full", fail_keyword_update)
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        with pytest.raises(Exception):
            database.update_keywords_for_prompt(prompt_id, [sentinel])
    finally:
        logger.remove(sink)
        database.close_connection()

    rendered = "\n".join(messages + [record.getMessage() for record in caplog.records])
    assert (
        "Prompt keyword membership update failed category=OperationalError" in rendered
    )
    assert sentinel not in rendered
    assert "Traceback" not in rendered


def test_malformed_prompt_manifest_id_is_rejected_before_path_or_log_use(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    malicious_id = "../../TASK197_ARCHIVE_ID_MUST_NOT_LEAK\n"
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Malformed fixture",
        description="Malformed fixture",
    )
    manifest.content_items.append(
        ContentItem(
            id=malicious_id,
            type=ContentType.PROMPT,
            title="Prompt",
            file_path="content/prompts/placeholder.json",
        )
    )
    manifest.total_prompts = 1
    archive_path = tmp_path / "malformed-id.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest.to_dict()))
        archive.writestr(
            "content/prompts/placeholder.json",
            json.dumps({"name": "Should not load"}),
        )
    status = ImportStatus()
    destination_path = tmp_path / "destination.db"
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        success, message = _importer(destination_path, tmp_path).import_chatbook(
            archive_path, import_status=status
        )
    finally:
        logger.remove(sink)

    assert success is False
    assert message == "Failed to import any items from chatbook"
    assert status.processed_items == 1
    assert status.failed_items == 1
    assert status.errors == ["Unable to import Prompt item."]
    assert not destination_path.exists()
    rendered = "\n".join(messages + [record.getMessage() for record in caplog.records])
    assert "TASK197_ARCHIVE_ID_MUST_NOT_LEAK" not in rendered
    assert "Traceback" not in rendered


@pytest.mark.parametrize(
    "payload",
    [
        {
            "record_schema": "tldw-chatbook-prompt",
            "record_version": 99,
            "name": "TASK197_INVALID_PAYLOAD_MUST_NOT_LEAK",
            "author": None,
            "details": None,
            "system_prompt": "TASK197_INVALID_BODY_MUST_NOT_LEAK",
            "user_prompt": None,
            "keywords": [],
            "artifact_type": "prompt",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        },
        {"record_schema": "tldw-chatbook-prompt", "name": "Partial"},
        {
            "record_schema": "tldw-chatbook-prompt",
            "record_version": 1,
            "name": "Mixed",
            "description": "legacy",
            "content": "legacy",
        },
        {"name": "Legacy", "description": None, "content": 7},
        {
            "name": "Legacy",
            "description": None,
            "content": "System",
            "unknown": "TASK197_EXTRA_VALUE_MUST_NOT_LEAK",
        },
        ["not", "an", "object"],
    ],
)
def test_invalid_prompt_archive_fails_before_database_mutation_with_fixed_copy(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    payload: Any,
) -> None:
    archive_path = tmp_path / "invalid.zip"
    _write_prompt_chatbook(archive_path, payload)
    destination_path = tmp_path / "destination.db"
    status = ImportStatus()
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        success, _ = _importer(destination_path, tmp_path).import_chatbook(
            archive_path,
            import_status=status,
        )
    finally:
        logger.remove(sink)

    destination = PromptsDatabase(destination_path, client_id="destination-reader")
    try:
        assert destination.get_all_active_prompt_ids() == []
    finally:
        destination.close_connection()
    rendered = "\n".join(messages + [record.getMessage() for record in caplog.records])
    assert success is False
    assert status.processed_items == 1
    assert status.successful_items == 0
    assert status.failed_items == 1
    assert status.errors == ["Unable to import Prompt item."]
    for sentinel in (
        "TASK197_INVALID_PAYLOAD_MUST_NOT_LEAK",
        "TASK197_INVALID_BODY_MUST_NOT_LEAK",
        "TASK197_EXTRA_VALUE_MUST_NOT_LEAK",
        "Traceback",
    ):
        assert sentinel not in rendered
        assert sentinel not in "\n".join(status.errors)
