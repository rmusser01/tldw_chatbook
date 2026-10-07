"""Canonical note language persists independently of exact source and versions."""

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(str(tmp_path / "notes.db"), client_id="test")
    yield database
    database.close_connection()


@pytest.mark.parametrize(
    "language", [None, "text", "markdown", "yaml", "json", "jsonl", "future/mode"]
)
def test_language_survives_create_read_list_and_copy(db, language):
    source = "\ufeffkey: 1\r\n# exact source  \r\n"
    note_id = db.add_note(title="Config", content=source, content_language=language)
    note = db.get_note_by_id(note_id)
    assert note["content"] == source
    assert note["content_language"] == language
    assert db.list_notes()[0]["content_language"] == language
    copy_id = db.add_note(
        title=note["title"],
        content=note["content"],
        content_language=note["content_language"],
    )
    assert db.get_note_by_id(copy_id)["content_language"] == language


def test_body_update_preserves_language_and_language_only_change_versions(db):
    note_id = db.add_note(title="Config", content="a: 1", content_language="yaml")
    db.update_note(note_id, {"content": "a: 2"}, expected_version=1)
    assert db.get_note_by_id(note_id)["content_language"] == "yaml"
    db.update_note(note_id, {"content_language": "future/mode"}, expected_version=2)
    note = db.get_note_by_id(note_id)
    assert (note["version"], note["content"], note["content_language"]) == (
        3,
        "a: 2",
        "future/mode",
    )
    with pytest.raises(ConflictError):
        db.update_note(note_id, {"content_language": None}, expected_version=2)
    assert db.get_note_by_id(note_id)["content_language"] == "future/mode"
    db.update_note(note_id, {"content_language": None}, expected_version=3)
    assert db.get_note_by_id(note_id)["content_language"] is None
    assert db.get_note_by_id(note_id)["version"] == 4


@pytest.mark.parametrize("language", ["", "x" * 101, 12, False, []])
def test_language_validation_rejects_invalid_metadata_before_write(db, language):
    with pytest.raises(InputError):
        db.add_note(title="Config", content="source", content_language=language)
    note_id = db.add_note(title="Config", content="source")
    with pytest.raises(InputError):
        db.update_note(note_id, {"content_language": language}, expected_version=1)
    assert db.get_note_by_id(note_id)["version"] == 1


def test_exact_title_is_preserved_on_create_and_update(db):
    note_id = db.add_note(title="  Config  ", content="source")
    assert db.get_note_by_id(note_id)["title"] == "  Config  "
    db.update_note(note_id, {"title": "  Next  "}, expected_version=1)
    assert db.get_note_by_id(note_id)["title"] == "  Next  "


@pytest.mark.parametrize("title", [" ", "x" * 256])
def test_title_boundary_rejects_without_rewriting(db, title):
    with pytest.raises(InputError):
        db.add_note(title=title, content="source")
    note_id = db.add_note(title="Valid", content="source")
    with pytest.raises(InputError):
        db.update_note(note_id, {"title": title}, expected_version=1)
    assert db.get_note_by_id(note_id)["version"] == 1


def test_server_payloads_and_response_preserve_language_omission():
    from tldw_chatbook.Notes.server_notes_workspace_service import (
        ServerNotesWorkspaceService,
    )

    service = ServerNotesWorkspaceService(None)
    assert "content_language" not in service.build_server_note_update_payload(
        content="a"
    ).model_dump(exclude_unset=True)
    cleared = service.build_server_note_update_payload(content_language=None)
    assert cleared.model_dump(exclude_unset=True) == {"content_language": None}
    assert (
        service.build_server_note_create_payload(
            title="A", content="a", content_language="yaml"
        ).content_language
        == "yaml"
    )
    assert (
        service.normalize_server_note({"id": "a", "content_language": "future/mode"})[
            "content_language"
        ]
        == "future/mode"
    )


def test_current_backup_restores_language_and_source(db, tmp_path):
    note_id = db.add_note(
        title="  Config  ", content="a: 1\r\n", content_language="future/mode"
    )
    backup = tmp_path / "current-backup.db"
    assert db.backup_database(str(backup))
    restored = CharactersRAGDB(str(backup), client_id="test")
    try:
        assert restored.get_note_by_id(note_id) == db.get_note_by_id(note_id)
    finally:
        restored.close_connection()


@pytest.mark.parametrize("historical_version", [73, 80])
def test_language_migration_and_old_backup_preserve_source(tmp_path, monkeypatch, historical_version):
    path = tmp_path / "historical.db"
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", historical_version)
        old = CharactersRAGDB(str(path), client_id="test")
        try:
            conn = old.get_connection()
            assert "content_language" not in {
                row["name"] for row in conn.execute("PRAGMA table_info(notes)")
            }
            conn.execute(
                "INSERT INTO notes(id, title, content, client_id) VALUES (?, ?, ?, ?)",
                ("old", "  Config  ", "a: 1\r\n", "test"),
            )
            conn.commit()
            backup = tmp_path / "old-backup.db"
            assert old.backup_database(str(backup))
        finally:
            old.close_connection()
    for candidate in (path, backup):
        upgraded = CharactersRAGDB(str(candidate), client_id="test")
        try:
            note = upgraded.get_note_by_id("old")
            assert note["content_language"] is None
            assert (note["title"], note["content"], note["version"]) == (
                "  Config  ",
                "a: 1\r\n",
                1,
            )
            assert (
                upgraded._get_db_version(upgraded.get_connection())
                == CharactersRAGDB._CURRENT_SCHEMA_VERSION
            )
        finally:
            upgraded.close_connection()


def test_exact_version_snapshot_contains_language(db):
    import json

    note_id = db.add_note(
        title="  Config  ", content="a: 1\r\n", content_language="yaml"
    )
    for version, language in [(1, "yaml"), (2, "json")]:
        if version == 2:
            db.update_note(note_id, {"content_language": language}, expected_version=1)
        row = db.execute_query(
            "SELECT version, payload FROM sync_log WHERE entity = 'notes' AND entity_id = ? ORDER BY version DESC LIMIT 1",
            (note_id,),
        ).fetchone()
        assert (row["version"], json.loads(row["payload"])["content_language"]) == (
            version,
            language,
        )
        assert json.loads(row["payload"])["content"] == "a: 1\r\n"


@pytest.mark.asyncio
async def test_local_scope_preserves_metadata_and_keeps_local_save_usable_with_pending_sync(
    db, tmp_path
):
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
    from tldw_chatbook.Notes.server_notes_workspace_service import (
        ServerNotesWorkspaceService,
    )

    local = NotesInteropService(tmp_path, "test", global_db_to_use=db)
    scope = NotesScopeService(
        local_notes_service=local,
        server_service=ServerNotesWorkspaceService(None),
        sync_v2_notes_producer=object(),
    )
    note_id = await scope.save_note(
        scope="local_note",
        user_id="test",
        title="Config",
        content="a: 1",
        content_language="yaml",
    )
    assert db.get_note_by_id(note_id)["content_language"] == "yaml"
    profile = {
        "server_profile_id": "server",
        "authenticated_principal_id": "test",
        "workspace_scope": None,
    }
    assert await scope.save_note(
        scope="local_note",
        user_id="test",
        note_id=note_id,
        version=1,
        title="Changed",
        content="a: 1",
        sync_v2_profile=profile,
    )
    assert db.get_note_by_id(note_id)["version"] == 2
    assert db.get_note_by_id(note_id)["content_language"] == "yaml"
    db.soft_delete_note(note_id, expected_version=2)
    restored = await scope.restore_note(
        scope="local_note",
        user_id="test",
        note_id=note_id,
        version=3,
        sync_v2_profile=profile,
    )
    assert restored["content_language"] == "yaml"
    assert not db.get_note_by_id(note_id)["deleted"]


@pytest.mark.asyncio
async def test_http_client_transmits_explicit_language_null_only_when_supplied():
    from unittest.mock import AsyncMock

    from tldw_chatbook.tldw_api.client import TLDWAPIClient
    from tldw_chatbook.tldw_api.notes_workspace_schemas import NoteUpdateRequest

    client = TLDWAPIClient("http://localhost:8000")
    client._request = AsyncMock(return_value={})
    await client.update_server_note(
        "note", NoteUpdateRequest(content_language=None), expected_version=1
    )
    assert client._request.call_args.kwargs["json_data"] == {"content_language": None}
    await client.update_server_note(
        "note", NoteUpdateRequest(content="body"), expected_version=1
    )
    assert client._request.call_args.kwargs["json_data"] == {"content": "body"}
