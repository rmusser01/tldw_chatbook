"""Connection-ownership regressions for Chatbook conversation I/O."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.Chatbooks import ChatbookCreator, ChatbookImporter
from tldw_chatbook.Chatbooks import chatbook_creator as creator_module
from tldw_chatbook.Chatbooks import chatbook_importer as importer_module
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class _CapturedCharactersDatabases:
    """Capture real operation-owned DBs without changing their behavior."""

    def __init__(self, factory: Callable[..., CharactersRAGDB]) -> None:
        self._factory = factory
        self.owners: list[CharactersRAGDB] = []
        self.connections: list[sqlite3.Connection] = []

    def __call__(self, *args: Any, **kwargs: Any) -> CharactersRAGDB:
        owner = self._factory(*args, **kwargs)
        self.owners.append(owner)
        self.connections.append(owner.get_connection())
        return owner

    def close_retained(self) -> None:
        """Keep an intentional RED from leaking descriptors into later tests."""

        for owner in self.owners:
            owner.close_connection()


def _assert_operation_released(
    *,
    observer: CharactersRAGDB,
    observer_connection: sqlite3.Connection,
    baseline: int,
    owned_connections: list[sqlite3.Connection],
) -> None:
    assert owned_connections
    assert observer_connection.execute("SELECT 1").fetchone()[0] == 1
    assert observer.registered_connection_count() == baseline
    for connection in owned_connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def _add_conversation_without_character(database: CharactersRAGDB) -> str:
    conversation_id = database.add_conversation(
        {
            "title": "Connection lifetime control",
            "character_id": None,
        }
    )
    assert conversation_id is not None
    return str(conversation_id)


def _create_conversation_archive(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_paths: dict[str, str],
    conversation_id: str,
    output_name: str,
) -> Path:
    """Create a real archive and explicitly clean its owner for importer tests."""

    output_path = tmp_path / output_name
    with monkeypatch.context() as scoped:
        scoped.setattr(creator_module, "get_user_data_dir", lambda: tmp_path)
        captured = _CapturedCharactersDatabases(creator_module.CharactersRAGDB)
        scoped.setattr(creator_module, "CharactersRAGDB", captured)
        try:
            success, message, _ = ChatbookCreator(db_paths).create_chatbook(
                name="Connection lifetime control",
                description="",
                content_selections={
                    ContentType.CONVERSATION: [conversation_id],
                },
                output_path=output_path,
                auto_include_dependencies=False,
            )
            assert success is True, message
        finally:
            captured.close_retained()
    return output_path


def test_creator_releases_owned_connection_after_each_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_db_paths: dict[str, str],
    populated_chachanotes_db: dict[str, Any],
) -> None:
    observer = populated_chachanotes_db["db"]
    observer_connection = observer.get_connection()
    baseline = observer.registered_connection_count()
    conversation_id = _add_conversation_without_character(observer)
    captured = _CapturedCharactersDatabases(creator_module.CharactersRAGDB)
    monkeypatch.setattr(creator_module, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(creator_module, "CharactersRAGDB", captured)

    try:
        creator = ChatbookCreator(mock_db_paths)
        for index in range(3):
            before = len(captured.connections)
            success, message, _ = creator.create_chatbook(
                name="Connection lifetime control",
                description="",
                content_selections={
                    ContentType.CONVERSATION: [conversation_id],
                },
                output_path=tmp_path / f"export-{index}.zip",
                auto_include_dependencies=False,
            )
            assert success is True, message
            _assert_operation_released(
                observer=observer,
                observer_connection=observer_connection,
                baseline=baseline,
                owned_connections=captured.connections[before:],
            )
    finally:
        captured.close_retained()
        observer.close_connection()


def test_creator_releases_owned_connection_when_service_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_db_paths: dict[str, str],
    populated_chachanotes_db: dict[str, Any],
) -> None:
    observer = populated_chachanotes_db["db"]
    observer_connection = observer.get_connection()
    baseline = observer.registered_connection_count()
    conversation_id = _add_conversation_without_character(observer)
    captured = _CapturedCharactersDatabases(creator_module.CharactersRAGDB)
    monkeypatch.setattr(creator_module, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(creator_module, "CharactersRAGDB", captured)
    monkeypatch.setattr(
        creator_module,
        "build_local_citation_conversation_service",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )

    try:
        success, _, _ = ChatbookCreator(mock_db_paths).create_chatbook(
            name="Connection lifetime control",
            description="",
            content_selections={
                ContentType.CONVERSATION: [conversation_id],
            },
            output_path=tmp_path / "failed-export.zip",
            auto_include_dependencies=False,
        )
        assert success is False
        _assert_operation_released(
            observer=observer,
            observer_connection=observer_connection,
            baseline=baseline,
            owned_connections=captured.connections,
        )
    finally:
        captured.close_retained()
        observer.close_connection()


def test_importer_releases_owned_connection_after_each_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_db_paths: dict[str, str],
    populated_chachanotes_db: dict[str, Any],
) -> None:
    conversation_id = _add_conversation_without_character(
        populated_chachanotes_db["db"]
    )
    archive_path = _create_conversation_archive(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        db_paths=mock_db_paths,
        conversation_id=conversation_id,
        output_name="import-source.zip",
    )
    destination_paths = dict(mock_db_paths)
    destination_paths["ChaChaNotes"] = str(tmp_path / "import-destination.db")
    observer = CharactersRAGDB(destination_paths["ChaChaNotes"], "observer")
    observer_connection = observer.get_connection()
    baseline = observer.registered_connection_count()
    captured = _CapturedCharactersDatabases(importer_module.CharactersRAGDB)
    monkeypatch.setattr(importer_module, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(importer_module, "CharactersRAGDB", captured)

    try:
        importer = ChatbookImporter(destination_paths)
        for _ in range(3):
            before = len(captured.connections)
            success, message = importer.import_chatbook(
                archive_path,
                conflict_resolution=ConflictResolution.SKIP,
            )
            assert success is True, message
            _assert_operation_released(
                observer=observer,
                observer_connection=observer_connection,
                baseline=baseline,
                owned_connections=captured.connections[before:],
            )
    finally:
        captured.close_retained()
        observer.close_connection()
        populated_chachanotes_db["db"].close_connection()


def test_importer_releases_owned_connection_when_service_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_db_paths: dict[str, str],
    populated_chachanotes_db: dict[str, Any],
) -> None:
    conversation_id = _add_conversation_without_character(
        populated_chachanotes_db["db"]
    )
    archive_path = _create_conversation_archive(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        db_paths=mock_db_paths,
        conversation_id=conversation_id,
        output_name="failed-import-source.zip",
    )
    destination_paths = dict(mock_db_paths)
    destination_paths["ChaChaNotes"] = str(tmp_path / "failed-import.db")
    observer = CharactersRAGDB(destination_paths["ChaChaNotes"], "observer")
    observer_connection = observer.get_connection()
    baseline = observer.registered_connection_count()
    captured = _CapturedCharactersDatabases(importer_module.CharactersRAGDB)
    monkeypatch.setattr(importer_module, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(importer_module, "CharactersRAGDB", captured)
    monkeypatch.setattr(
        importer_module,
        "build_local_citation_conversation_service",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )

    try:
        success, _ = ChatbookImporter(destination_paths).import_chatbook(archive_path)
        assert success is False
        _assert_operation_released(
            observer=observer,
            observer_connection=observer_connection,
            baseline=baseline,
            owned_connections=captured.connections,
        )
    finally:
        captured.close_retained()
        observer.close_connection()
        populated_chachanotes_db["db"].close_connection()
