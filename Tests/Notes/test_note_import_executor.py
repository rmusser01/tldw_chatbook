"""Behavior tests for deterministic local Database Notes import target calls."""

from __future__ import annotations

import json
import sqlite3
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from datetime import UTC
from pathlib import Path
from queue import Queue
from uuid import UUID, uuid5

import pytest
from loguru import logger as loguru_logger

import tldw_chatbook.Notes.note_import_executor as note_import_executor_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.Notes.note_folder_models import (
    FolderCapabilityError,
    FolderCollisionError,
    FolderConflictError,
    FolderValidationError,
    NoteFolder,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.note_import_execution_models import (
    ImportEffectState,
    ImportItemOutcome,
    ImportSessionState,
    approve_note_import_plan,
)
from tldw_chatbook.Notes.note_import_executor import (
    ImportTargetConflictError,
    ImportTargetError,
    ImportTargetInternalError,
    ImportTargetPermanentError,
    ImportTargetRetryableError,
    LocalNoteImportTarget,
    NoteImportExecutor,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportAction,
    ImportBounds,
    ImportClassification,
    ImportMatch,
    ImportMatchKind,
    ImportPreviewItem,
    ImportSource,
    ImportSourceKind,
    NoteImportPlan,
    ParsedNotePayload,
    ProposedFolderMembership,
    RootCollisionChoice,
    RootCollisionState,
)
from tldw_chatbook.Notes.note_import_receipts import NoteImportReceiptRepository
from tldw_chatbook.Notes.Notes_Library import NotesInteropService

_FOLDER_ID = "00000000-0000-5000-8000-000000000001"
_OTHER_FOLDER_ID = "00000000-0000-5000-8000-000000000002"
_NOTE_ID = "00000000-0000-5000-8000-000000000101"
_EXECUTION_APPROVAL_ID = "00000000-0000-4000-8000-000000000041"


@pytest.fixture
def target_harness(
    tmp_path,
) -> Iterator[
    tuple[
        LocalNoteImportTarget,
        NotesInteropService,
        LocalNoteFolderRepository,
        CharactersRAGDB,
    ]
]:
    """Return the real local target stack over one temporary database."""
    db = CharactersRAGDB(tmp_path / "target.db", client_id="target-template")
    service = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="target-api",
        global_db_to_use=db,
    )
    target_db = service._get_db("target-user")
    folders = LocalNoteFolderRepository(target_db)
    target = LocalNoteImportTarget(
        db=target_db,
        folder_repository=folders,
    )
    yield target, service, folders, target_db
    service.close_all_user_connections()
    db.close_connection()


def _payload(
    *,
    title: str = "Imported title",
    content: str = "Imported body",
    keywords: tuple[str, ...] = ("Project", "draft"),
) -> ParsedNotePayload:
    return ParsedNotePayload(title=title, content=content, keywords=keywords)


def _execution_bounds() -> ImportBounds:
    return ImportBounds(
        max_files=50,
        max_file_bytes=1_000_000,
        max_total_bytes=5_000_000,
        max_depth=8,
        max_entries=1_000,
        max_notes_per_file=100,
        max_keywords_per_note=50,
    )


def _execution_item(
    *,
    item_id: str,
    payloads: tuple[ParsedNotePayload, ...],
    action: ImportAction,
    memberships: tuple[ProposedFolderMembership, ...] = (),
    match: ImportMatch | None = None,
    replace_content: bool = False,
    add_membership: bool = False,
    classification: ImportClassification | None = None,
) -> ImportPreviewItem:
    resolved_classification = classification
    if resolved_classification is None:
        resolved_classification = (
            ImportClassification.CHANGED_REPEAT
            if match is not None
            else ImportClassification.NEW
        )
    if resolved_classification in {
        ImportClassification.UNSUPPORTED,
        ImportClassification.FAILED,
    }:
        allowed_actions = (ImportAction.SKIP,)
        default_action = ImportAction.SKIP
    elif match is None:
        allowed_actions = (ImportAction.SKIP, ImportAction.CREATE_NEW)
        default_action = ImportAction.CREATE_NEW
    else:
        allowed_actions = (
            ImportAction.SKIP,
            ImportAction.CREATE_NEW,
            ImportAction.UPDATE_EXISTING,
        )
        default_action = ImportAction.CREATE_NEW
    return ImportPreviewItem(
        item_id=item_id,
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path=f"Selected/{item_id}.md",
            source_path=Path(f"/private/import/{item_id}.md"),
        ),
        payloads=payloads,
        memberships=memberships,
        classification=resolved_classification,
        reason="Approved import preview outcome.",
        default_action=default_action,
        selected_action=action,
        allowed_actions=allowed_actions,
        match=match,
        replace_content=replace_content,
        add_membership=add_membership,
    )


def _execution_plan(
    *items: ImportPreviewItem,
    proposed_folder_paths: tuple[tuple[str, ...], ...] = (),
    root_collision: RootCollisionState | None = None,
) -> NoteImportPlan:
    return NoteImportPlan(
        bounds=_execution_bounds(),
        items=items,
        proposed_folder_paths=proposed_folder_paths,
        root_collision=root_collision,
    )


def _approved_execution_plan(
    *items: ImportPreviewItem,
    proposed_folder_paths: tuple[tuple[str, ...], ...] = (),
    root_collision: RootCollisionState | None = None,
):
    return approve_note_import_plan(
        _execution_plan(
            *items,
            proposed_folder_paths=proposed_folder_paths,
            root_collision=root_collision,
        ),
        approval_id=_EXECUTION_APPROVAL_ID,
    )


@pytest.fixture
def real_executor(target_harness, tmp_path):
    target, service, folders, db = target_harness
    receipts = NoteImportReceiptRepository(tmp_path / "execution-receipts.sqlite3")
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=25,
    )
    return executor, target, receipts, service, folders, db


def _expected_note_id(item_id: str, payload_index: int) -> str:
    return str(
        uuid5(
            UUID(_EXECUTION_APPROVAL_ID),
            f"note:{item_id}:{payload_index}",
        )
    )


def _expected_folder_id(normalized_path: str) -> str:
    return str(
        uuid5(
            UUID(_EXECUTION_APPROVAL_ID),
            f"folder:{normalized_path}",
        )
    )


def _active_membership_count(
    db: CharactersRAGDB, *, folder_id: str, note_id: str
) -> int:
    row = (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM note_folder_memberships "
            "WHERE folder_id = ? AND note_id = ? AND deleted = 0",
            (folder_id, note_id),
        )
        .fetchone()
    )
    return int(row[0])


def test_target_constructor_rejects_a_different_repository_database_safely(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target_db = CharactersRAGDB(
        tmp_path / "private-target.db", client_id="private-target-user"
    )
    repository_db = CharactersRAGDB(
        tmp_path / "private-repository.db", client_id="private-repository-user"
    )
    folders = LocalNoteFolderRepository(repository_db)
    private_values = (
        target_db.db_path_str,
        repository_db.db_path_str,
        target_db.client_id,
        repository_db.client_id,
    )
    loguru_messages: list[str] = []
    sink_id = loguru_logger.add(lambda message: loguru_messages.append(str(message)))
    try:
        with pytest.raises(ImportTargetPermanentError) as caught:
            LocalNoteImportTarget(db=target_db, folder_repository=folders)
    finally:
        loguru_logger.remove(sink_id)
        target_db.close_connection()
        repository_db.close_connection()

    rendered = "".join(loguru_messages) + caplog.text + repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    for private_value in private_values:
        assert private_value not in rendered


def test_target_ensure_folder_is_deterministic_and_replay_safe(
    target_harness,
) -> None:
    target, _service, _folders, _db = target_harness

    first = target.ensure_folder(
        segments=("Imported",), folder_id=_FOLDER_ID, allow_existing=False
    )
    retry = target.ensure_folder(
        segments=("imported",), folder_id=_FOLDER_ID, allow_existing=False
    )

    assert first.folder_id == _FOLDER_ID
    assert retry == first


def test_target_folder_projection_is_frozen_usable_and_private_safe(
    target_harness,
) -> None:
    target, _service, _folders, _db = target_harness

    folder = target.ensure_folder(
        segments=("Private Imported",),
        folder_id=_FOLDER_ID,
        allow_existing=False,
    )

    folder_type = getattr(note_import_executor_module, "LocalTargetFolder", None)
    assert folder_type is not None
    assert isinstance(folder, folder_type)
    assert folder.folder_id == _FOLDER_ID
    assert folder.name == "Private Imported"
    assert folder.path == "/Private Imported"
    assert folder.normalized_path == "/private imported"
    with pytest.raises(FrozenInstanceError):
        folder.name = "changed"  # type: ignore[misc]
    for rendered in (repr(folder), str(folder)):
        for private_value in (
            _FOLDER_ID,
            "Private Imported",
            "/Private Imported",
            "/private imported",
        ):
            assert private_value not in rendered


@pytest.mark.parametrize("field", ["folder_id", "name", "path", "normalized_path"])
def test_target_folder_projection_rejects_non_exact_text_fields(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    target, _service, folders, _db = target_harness

    class TextSubclass(str):
        pass

    values = {
        "folder_id": _FOLDER_ID,
        "parent_id": None,
        "name": "Private Imported",
        "path": "/Private Imported",
        "normalized_path": "/private imported",
        "version": 1,
        "deleted": False,
    }
    values[field] = TextSubclass(values[field])
    hostile_folder = NoteFolder(**values)  # type: ignore[arg-type]
    monkeypatch.setattr(folders, "get_folder_by_path", lambda _segments: hostile_folder)

    with pytest.raises(ImportTargetInternalError) as caught:
        target.ensure_folder(
            segments=("Private Imported",),
            folder_id=_FOLDER_ID,
            allow_existing=False,
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "Private Imported" not in repr(caught.value)


def test_target_root_reuse_requires_explicit_allow_existing(target_harness) -> None:
    target, _service, folders, _db = target_harness
    existing = folders.create_folder(
        name="Imported", parent_id=None, folder_id=_OTHER_FOLDER_ID
    )

    with pytest.raises(ImportTargetConflictError):
        target.ensure_folder(
            segments=("Imported",), folder_id=_FOLDER_ID, allow_existing=False
        )

    reused = target.ensure_folder(
        segments=("Imported",), folder_id=_FOLDER_ID, allow_existing=True
    )
    assert reused.folder_id == existing.folder_id
    assert reused.name == existing.name
    assert reused.path == existing.path
    assert reused.normalized_path == existing.normalized_path


def test_target_deleted_deterministic_folder_identity_is_a_conflict(
    target_harness,
) -> None:
    target, _service, folders, db = target_harness
    folders.create_folder(name="Imported", parent_id=None, folder_id=_FOLDER_ID)
    db.get_connection().execute(
        "UPDATE note_folders SET deleted = 1 WHERE id = ?", (_FOLDER_ID,)
    )
    db.get_connection().commit()

    with pytest.raises(ImportTargetConflictError):
        target.ensure_folder(
            segments=("Imported",), folder_id=_FOLDER_ID, allow_existing=False
        )


def test_target_reconciles_a_concurrent_folder_create_winner(
    target_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, _service, folders, _db = target_harness
    original_create = folders.create_folder

    def concurrent_create(**kwargs):
        original_create(**kwargs)
        raise FolderCollisionError("hostile collision detail")

    monkeypatch.setattr(folders, "create_folder", concurrent_create)

    result = target.ensure_folder(
        segments=("Imported",), folder_id=_FOLDER_ID, allow_existing=False
    )

    assert result.folder_id == _FOLDER_ID


def test_target_concurrent_different_folder_identity_is_a_conflict(
    target_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, _service, folders, _db = target_harness
    original_create = folders.create_folder

    def concurrent_create(**_kwargs):
        original_create(name="Imported", parent_id=None, folder_id=_OTHER_FOLDER_ID)
        raise FolderCollisionError("hostile collision detail")

    monkeypatch.setattr(folders, "create_folder", concurrent_create)

    with pytest.raises(ImportTargetConflictError) as caught:
        target.ensure_folder(
            segments=("Imported",), folder_id=_FOLDER_ID, allow_existing=False
        )

    assert caught.value.__cause__ is None
    assert "hostile" not in repr(caught.value)


def test_target_concurrent_folder_id_winner_on_another_path_is_a_conflict(
    target_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, _service, folders, _db = target_harness
    original_create = folders.create_folder

    def concurrent_create(**_kwargs):
        original_create(name="Other", parent_id=None, folder_id=_FOLDER_ID)
        raise FolderValidationError("hostile identity detail")

    monkeypatch.setattr(folders, "create_folder", concurrent_create)

    with pytest.raises(ImportTargetConflictError) as caught:
        target.ensure_folder(
            segments=("Imported",), folder_id=_FOLDER_ID, allow_existing=False
        )

    assert caught.value.__cause__ is None
    assert "hostile" not in repr(caught.value)


@pytest.mark.parametrize("_race_attempt", range(3))
def test_target_real_two_connection_same_path_race_has_one_safe_winner(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    _race_attempt: int,
) -> None:
    database_path = tmp_path / "race.db"
    first_db = CharactersRAGDB(database_path, client_id="race-first")
    second_db = CharactersRAGDB(database_path, client_id="race-second")
    first_folders = LocalNoteFolderRepository(first_db)
    second_folders = LocalNoteFolderRepository(second_db)
    first_target = LocalNoteImportTarget(db=first_db, folder_repository=first_folders)
    second_target = LocalNoteImportTarget(
        db=second_db, folder_repository=second_folders
    )
    pre_read_barrier = threading.Barrier(2)

    def synchronize_first_pre_read(repository: LocalNoteFolderRepository) -> None:
        real_read = repository.get_folder_by_path
        is_first_read = True

        def synchronized_read(segments):
            nonlocal is_first_read
            result = real_read(segments)
            if is_first_read:
                is_first_read = False
                pre_read_barrier.wait(timeout=5)
            return result

        monkeypatch.setattr(repository, "get_folder_by_path", synchronized_read)

    synchronize_first_pre_read(first_folders)
    synchronize_first_pre_read(second_folders)
    outcomes: Queue[object] = Queue()

    def race(
        target: LocalNoteImportTarget,
        folder_id: str,
        database: CharactersRAGDB,
    ) -> None:
        try:
            outcomes.put(
                target.ensure_folder(
                    segments=("Racing path",),
                    folder_id=folder_id,
                    allow_existing=False,
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve the exact race outcome
            outcomes.put(exc)
        finally:
            database.close_connection()

    threads = (
        threading.Thread(
            target=race,
            args=(first_target, _FOLDER_ID, first_db),
            daemon=True,
        ),
        threading.Thread(
            target=race,
            args=(second_target, _OTHER_FOLDER_ID, second_db),
            daemon=True,
        ),
    )
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert all(not thread.is_alive() for thread in threads)
        observed = (outcomes.get_nowait(), outcomes.get_nowait())
        folder_type = getattr(note_import_executor_module, "LocalTargetFolder", None)
        assert folder_type is not None
        assert sum(isinstance(item, folder_type) for item in observed) == 1
        assert (
            sum(isinstance(item, ImportTargetConflictError) for item in observed) == 1
        )
        rows = (
            first_db.get_connection()
            .execute(
                "SELECT id FROM note_folders "
                "WHERE normalized_path = '/racing path' AND deleted = 0"
            )
            .fetchall()
        )
        assert len(rows) == 1
        assert rows[0]["id"] in {_FOLDER_ID, _OTHER_FOLDER_ID}
    finally:
        first_db.close_connection()
        second_db.close_connection()


def test_target_folder_conflict_precedes_following_note_mutation(
    target_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, _service, folders, _db = target_harness
    folders.create_folder(name="Imported", parent_id=None, folder_id=_OTHER_FOLDER_ID)
    real_insert_note = target._insert_note
    note_mutations = 0

    def recording_insert_note(*args, **kwargs):
        nonlocal note_mutations
        note_mutations += 1
        return real_insert_note(*args, **kwargs)

    monkeypatch.setattr(target, "_insert_note", recording_insert_note)

    with pytest.raises(ImportTargetConflictError):
        target.ensure_folder(
            segments=("Imported",), folder_id=_FOLDER_ID, allow_existing=False
        )
        target.create_note(note_id=_NOTE_ID, payload=_payload())

    assert note_mutations == 0
    assert target.read_note(note_id=_NOTE_ID) is None


@pytest.mark.parametrize(
    ("segments", "folder_id", "allow_existing"),
    [
        ((), _FOLDER_ID, False),
        (("Missing", "Child"), _FOLDER_ID, False),
        (("Imported",), "", False),
        (("Imported",), _FOLDER_ID, 1),
    ],
)
def test_target_folder_validation_failures_are_permanent(
    target_harness, segments, folder_id, allow_existing
) -> None:
    target, _service, _folders, _db = target_harness

    with pytest.raises(ImportTargetPermanentError):
        target.ensure_folder(
            segments=segments,
            folder_id=folder_id,
            allow_existing=allow_existing,
        )


def test_target_folder_segments_stop_at_the_bounded_overflow_probe(
    target_harness,
) -> None:
    target, _service, _folders, _db = target_harness
    next_calls = 0

    def guarded_segments():
        nonlocal next_calls
        while True:
            next_calls += 1
            if next_calls > 65:
                raise AssertionError("target consumed beyond its bounded probe")
            yield "segment"

    with pytest.raises(ImportTargetPermanentError):
        target.ensure_folder(
            segments=guarded_segments(),
            folder_id=_FOLDER_ID,
            allow_existing=False,
        )

    assert next_calls == 65


def test_target_read_note_returns_a_frozen_private_projection(target_harness) -> None:
    target, _service, _folders, _db = target_harness
    created = target.create_note(note_id=_NOTE_ID, payload=_payload())

    read_back = target.read_note(note_id=_NOTE_ID)

    assert read_back == created
    assert read_back is not None
    assert read_back.note_id == _NOTE_ID
    assert read_back.title == "Imported title"
    assert read_back.content == "Imported body"
    assert read_back.version == 1
    assert read_back.keywords == ("draft", "Project")
    with pytest.raises(FrozenInstanceError):
        read_back.title = "changed"  # type: ignore[misc]
    rendered = repr(read_back)
    for private_value in (_NOTE_ID, "Imported title", "Imported body", "Project"):
        assert private_value not in rendered


def test_target_create_note_persists_exact_payload_and_reconciles_retry(
    target_harness,
) -> None:
    target, _service, _folders, db = target_harness
    payload = _payload(title="  Imported title  ", keywords=(" Project ", "draft"))

    first = target.create_note(note_id=_NOTE_ID, payload=payload)
    retry = target.create_note(note_id=_NOTE_ID, payload=payload)

    assert first == retry
    assert first.title == "Imported title"
    assert first.content == payload.content
    assert target.keywords_match(note_id=_NOTE_ID, keywords=payload.keywords)
    count = (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM notes WHERE id = ?", (_NOTE_ID,))
        .fetchone()[0]
    )
    assert count == 1


def test_target_uses_injected_service_database_client_id_for_direct_writes(
    target_harness,
) -> None:
    _target, _service, folders, db = target_harness
    target = LocalNoteImportTarget(
        db=db,
        folder_repository=folders,
    )

    target.create_note(
        note_id=_NOTE_ID,
        payload=_payload(keywords=("canonical-client",)),
    )

    connection = db.get_connection()
    client_ids = {
        connection.execute(
            "SELECT client_id FROM notes WHERE id = ?", (_NOTE_ID,)
        ).fetchone()[0],
        connection.execute(
            "SELECT client_id FROM keywords WHERE keyword = 'canonical-client'"
        ).fetchone()[0],
        connection.execute(
            "SELECT client_id FROM sync_log "
            "WHERE entity = 'note_keywords' AND entity_id LIKE ?",
            (f"{_NOTE_ID}_%",),
        ).fetchone()[0],
    }
    assert client_ids == {"target-user"}


@pytest.mark.parametrize(
    "different",
    [
        _payload(title="Different"),
        _payload(content="Different"),
        _payload(keywords=("different",)),
    ],
)
def test_target_create_note_rejects_deterministic_id_payload_conflicts(
    target_harness, different: ParsedNotePayload
) -> None:
    target, _service, _folders, db = target_harness
    target.create_note(note_id=_NOTE_ID, payload=_payload())

    with pytest.raises(ImportTargetConflictError):
        target.create_note(note_id=_NOTE_ID, payload=different)

    count = (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM notes WHERE id = ?", (_NOTE_ID,))
        .fetchone()[0]
    )
    assert count == 1


def test_target_replace_note_pins_one_version_increment_and_reconciles_retry(
    target_harness,
) -> None:
    target, _service, _folders, _db = target_harness
    target.create_note(note_id=_NOTE_ID, payload=_payload())
    replacement = _payload(
        title="Replacement", content="Replacement body", keywords=("final",)
    )

    updated = target.replace_note(
        note_id=_NOTE_ID, expected_version=1, payload=replacement
    )
    retry = target.replace_note(
        note_id=_NOTE_ID, expected_version=1, payload=replacement
    )

    assert updated.version == 2
    assert retry == updated


def test_target_replace_note_rejects_missing_different_or_later_state(
    target_harness,
) -> None:
    target, service, _folders, _db = target_harness
    replacement = _payload(title="Replacement", keywords=("final",))
    with pytest.raises(ImportTargetConflictError):
        target.replace_note(note_id=_NOTE_ID, expected_version=1, payload=replacement)

    target.create_note(note_id=_NOTE_ID, payload=_payload())
    target.replace_note(note_id=_NOTE_ID, expected_version=1, payload=replacement)
    with pytest.raises(ImportTargetConflictError):
        target.replace_note(
            note_id=_NOTE_ID,
            expected_version=1,
            payload=_payload(title="Other replacement"),
        )

    assert service.update_note(
        "target-user",
        _NOTE_ID,
        {"title": "Concurrent later version", "content": "Later"},
        2,
    )
    with pytest.raises(ImportTargetConflictError):
        target.replace_note(note_id=_NOTE_ID, expected_version=1, payload=replacement)


def test_target_replace_note_translates_an_optimistic_race(
    target_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, _service, _folders, _db = target_harness
    target.create_note(note_id=_NOTE_ID, payload=_payload())
    real_update = target._update_note

    def racing_update(cursor, *, note_id, expected_version, payload):
        cursor.execute(
            """
            UPDATE notes
            SET title = ?, content = ?, last_modified = CURRENT_TIMESTAMP,
                version = version + 1
            WHERE id = ? AND version = ? AND deleted = 0
            """,
            ("Racer", "Racer body", note_id, expected_version),
        )
        return real_update(
            cursor,
            note_id=note_id,
            expected_version=expected_version,
            payload=payload,
        )

    monkeypatch.setattr(target, "_update_note", racing_update)

    with pytest.raises(ImportTargetConflictError):
        target.replace_note(
            note_id=_NOTE_ID, expected_version=1, payload=_payload(title="Winner")
        )


def test_target_create_rolls_back_note_on_internal_keyword_sync_failure(
    target_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, _service, _folders, db = target_harness

    def failing_sync(*_args, **_kwargs):
        raise ValueError("hostile private keyword detail")

    monkeypatch.setattr(target, "_sync_keywords", failing_sync)

    with pytest.raises(ImportTargetInternalError) as caught:
        target.create_note(note_id=_NOTE_ID, payload=_payload())

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None

    row = (
        db.get_connection()
        .execute("SELECT 1 FROM notes WHERE id = ?", (_NOTE_ID,))
        .fetchone()
    )
    assert row is None


def test_target_replace_rolls_back_text_on_internal_keyword_sync_failure(
    target_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, _service, _folders, _db = target_harness
    original = target.create_note(note_id=_NOTE_ID, payload=_payload())

    def failing_sync(*_args, **_kwargs):
        raise ValueError("hostile private keyword detail")

    monkeypatch.setattr(target, "_sync_keywords", failing_sync)

    with pytest.raises(ImportTargetInternalError) as caught:
        target.replace_note(
            note_id=_NOTE_ID,
            expected_version=1,
            payload=_payload(title="Must roll back", keywords=("replacement",)),
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None

    assert target.read_note(note_id=_NOTE_ID) == original


def test_target_keyword_sync_is_exact_canonical_and_idempotent(target_harness) -> None:
    target, service, _folders, db = target_harness
    target.create_note(note_id=_NOTE_ID, payload=_payload(keywords=()))
    stale_id = service.add_keyword("target-user", "stale")
    keep_id = service.add_keyword("target-user", "Keep")
    assert stale_id is not None and keep_id is not None
    assert service.link_note_to_keyword("target-user", _NOTE_ID, stale_id)
    assert service.link_note_to_keyword("target-user", _NOTE_ID, keep_id)

    target.sync_keywords(note_id=_NOTE_ID, keywords=("keep", " New ", "new"))
    first_links = (
        db.get_connection()
        .execute(
            "SELECT keyword_id FROM note_keywords WHERE note_id = ? ORDER BY keyword_id",
            (_NOTE_ID,),
        )
        .fetchall()
    )
    target.sync_keywords(note_id=_NOTE_ID, keywords=("KEEP", "New"))
    second_links = (
        db.get_connection()
        .execute(
            "SELECT keyword_id FROM note_keywords WHERE note_id = ? ORDER BY keyword_id",
            (_NOTE_ID,),
        )
        .fetchall()
    )

    assert target.keywords_match(note_id=_NOTE_ID, keywords=("KEEP", "new"))
    assert first_links == second_links
    assert len(second_links) == 2
    stale = (
        db.get_connection()
        .execute("SELECT keyword, deleted FROM keywords WHERE id = ?", (stale_id,))
        .fetchone()
    )
    assert tuple(stale) == ("stale", 0)


def test_target_keyword_sync_handles_hidden_deleted_links_exactly(
    target_harness,
) -> None:
    target, _service, _folders, db = target_harness
    target.create_note(
        note_id=_NOTE_ID,
        payload=_payload(keywords=("hidden-stale", "hidden-keep")),
    )
    connection = db.get_connection()
    keyword_rows = connection.execute(
        "SELECT id, keyword FROM keywords "
        "WHERE keyword IN ('hidden-stale', 'hidden-keep')"
    ).fetchall()
    keyword_ids = {row["keyword"]: row["id"] for row in keyword_rows}
    connection.execute(
        "UPDATE keywords SET deleted = 1, version = version + 1 WHERE id IN (?, ?)",
        (keyword_ids["hidden-stale"], keyword_ids["hidden-keep"]),
    )
    connection.commit()

    target.sync_keywords(note_id=_NOTE_ID, keywords=("HIDDEN-KEEP",))

    links = connection.execute(
        "SELECT keyword_id FROM note_keywords WHERE note_id = ?", (_NOTE_ID,)
    ).fetchall()
    assert [row[0] for row in links] == [keyword_ids["hidden-keep"]]
    keyword_state = connection.execute(
        "SELECT keyword, deleted, version FROM keywords "
        "WHERE id IN (?, ?) ORDER BY keyword",
        (keyword_ids["hidden-stale"], keyword_ids["hidden-keep"]),
    ).fetchall()
    assert [tuple(row) for row in keyword_state] == [
        ("HIDDEN-KEEP", 0, 3),
        ("hidden-stale", 1, 2),
    ]
    undelete_sync = connection.execute(
        "SELECT operation, client_id, version, payload FROM sync_log "
        "WHERE entity = 'keywords' AND entity_id = ? ORDER BY change_id DESC LIMIT 1",
        (str(keyword_ids["hidden-keep"]),),
    ).fetchone()
    assert undelete_sync is not None
    assert tuple(undelete_sync[:3]) == ("update", "target-user", 3)
    undelete_payload = json.loads(undelete_sync["payload"])
    assert set(undelete_payload) == {
        "id",
        "keyword",
        "created_at",
        "last_modified",
        "deleted",
        "client_id",
        "version",
    }
    assert undelete_payload["id"] == keyword_ids["hidden-keep"]
    assert undelete_payload["keyword"] == "HIDDEN-KEEP"
    assert undelete_payload["deleted"] == 0
    assert undelete_payload["client_id"] == "target-user"
    assert undelete_payload["version"] == 3
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM keywords_fts "
            "JOIN keywords ON keywords.id = keywords_fts.rowid "
            "WHERE keywords.id = ? AND keywords_fts MATCH ?",
            (keyword_ids["hidden-keep"], '"HIDDEN-KEEP"'),
        ).fetchone()[0]
        == 1
    )


def test_executor_creates_multi_payload_notes_with_exact_keywords_and_memberships(
    real_executor,
) -> None:
    executor, target, receipts, _service, folders, db = real_executor
    payloads = (
        _payload(title="First", content="First body", keywords=("Alpha", "beta")),
        _payload(title="Second", content="Second body", keywords=("Gamma",)),
    )
    item = _execution_item(
        item_id="multi-create",
        payloads=payloads,
        action=ImportAction.CREATE_NEW,
        memberships=(
            ProposedFolderMembership(0, ("Imported Root",)),
            ProposedFolderMembership(1, ("Imported Root", "Nested")),
        ),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(
            ("Imported Root",),
            ("Imported Root", "Nested"),
        ),
    )

    receipt = executor.execute(approved)

    assert (receipt.state, receipt.imported, receipt.failed, receipt.completed) == (
        ImportSessionState.COMPLETED,
        2,
        0,
        2,
    )
    root = folders.get_folder_by_path(("Imported Root",))
    nested = folders.get_folder_by_path(("Imported Root", "Nested"))
    assert root is not None and nested is not None
    assert root.folder_id == _expected_folder_id("/imported root")
    assert nested.folder_id == _expected_folder_id("/imported root/nested")
    for payload_index, payload in enumerate(payloads):
        note_id = _expected_note_id(item.item_id, payload_index)
        note = target.read_note(note_id=note_id)
        assert note is not None
        assert (note.title, note.content, set(note.keywords)) == (
            payload.title,
            payload.content,
            set(payload.keywords),
        )
    assert (
        _active_membership_count(
            db, folder_id=root.folder_id, note_id=_expected_note_id(item.item_id, 0)
        )
        == 1
    )
    assert (
        _active_membership_count(
            db, folder_id=nested.folder_id, note_id=_expected_note_id(item.item_id, 1)
        )
        == 1
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert all(
        effect.state is ImportEffectState.APPLIED
        for effect in (
            *durable.payload_effects,
            *durable.folder_effects,
            *durable.membership_effects,
        )
    )
    assert durable.items[0].outcome is ImportItemOutcome.IMPORTED


def test_executor_update_can_replace_without_adding_membership(real_executor) -> None:
    executor, target, _receipts, _service, folders, _db = real_executor
    existing = target.create_note(
        note_id="existing-replace",
        payload=_payload(content="Original body", keywords=("old",)),
    )
    replacement = _payload(content="Replacement body", keywords=("new", "exact"))
    item = _execution_item(
        item_id="replace-only",
        payloads=(replacement,),
        action=ImportAction.UPDATE_EXISTING,
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        replace_content=True,
    )

    receipt = executor.execute(_approved_execution_plan(item))

    assert (receipt.updated, receipt.failed) == (1, 0)
    updated = target.read_note(note_id=existing.note_id)
    assert updated is not None
    assert (updated.content, set(updated.keywords), updated.version) == (
        replacement.content,
        set(replacement.keywords),
        existing.version + 1,
    )
    assert folders.list_memberships(note_ids=(existing.note_id,)) == ()


def test_executor_update_can_add_membership_without_replacing_content(
    real_executor,
) -> None:
    executor, target, receipts, _service, folders, db = real_executor
    original_payload = _payload(content="Original body", keywords=("keep",))
    existing = target.create_note(
        note_id="existing-membership", payload=original_payload
    )
    item = _execution_item(
        item_id="membership-only",
        payloads=(_payload(content="Ignored replacement", keywords=("ignored",)),),
        action=ImportAction.UPDATE_EXISTING,
        memberships=(ProposedFolderMembership(0, ("Imported Root",)),),
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        add_membership=True,
    )

    receipt = executor.execute(
        _approved_execution_plan(
            item,
            proposed_folder_paths=(("Imported Root",),),
        )
    )

    assert (receipt.updated, receipt.failed) == (1, 0)
    unchanged = target.read_note(note_id=existing.note_id)
    assert unchanged is not None
    assert (unchanged.content, set(unchanged.keywords), unchanged.version) == (
        original_payload.content,
        set(original_payload.keywords),
        existing.version,
    )
    folder = folders.get_folder_by_path(("Imported Root",))
    assert folder is not None
    assert (
        _active_membership_count(
            db, folder_id=folder.folder_id, note_id=existing.note_id
        )
        == 1
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.payload_effects == ()
    assert durable.items[0].observed_version == existing.version


def test_executor_update_keeps_replace_effect_independent_from_folder_failure(
    real_executor,
) -> None:
    executor, target, receipts, _service, folders, _db = real_executor
    existing = target.create_note(
        note_id="existing-independent-update",
        payload=_payload(content="Original body", keywords=("old",)),
    )
    folders.create_folder(
        name="Imported Root", parent_id=None, folder_id="different-root-id"
    )
    replacement = _payload(content="Confirmed replacement", keywords=("new",))
    item = _execution_item(
        item_id="independent-update",
        payloads=(replacement,),
        action=ImportAction.UPDATE_EXISTING,
        memberships=(ProposedFolderMembership(0, ("Imported Root",)),),
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        replace_content=True,
        add_membership=True,
    )

    receipt = executor.execute(
        _approved_execution_plan(
            item,
            proposed_folder_paths=(("Imported Root",),),
        )
    )

    updated = target.read_note(note_id=existing.note_id)
    assert updated is not None
    assert (updated.content, set(updated.keywords), updated.version) == (
        replacement.content,
        set(replacement.keywords),
        existing.version + 1,
    )
    assert (receipt.updated, receipt.failed, receipt.reason_code) == (
        0,
        1,
        "folder_conflict",
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.payload_effects[0].state is ImportEffectState.APPLIED
    assert durable.folder_effects[0].state is ImportEffectState.FAILED
    assert durable.items[0].outcome is ImportItemOutcome.FAILED


def test_executor_membership_only_stale_version_durably_fails_membership_effect(
    real_executor,
) -> None:
    executor, target, receipts, _service, folders, _db = real_executor
    existing = target.create_note(note_id="existing-stale", payload=_payload())
    item = _execution_item(
        item_id="stale-membership",
        payloads=(_payload(),),
        action=ImportAction.UPDATE_EXISTING,
        memberships=(ProposedFolderMembership(0, ("Imported Root",)),),
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Imported Root",),),
    )
    target.replace_note(
        note_id=existing.note_id,
        expected_version=existing.version,
        payload=_payload(content="Changed after approval"),
    )

    receipt = executor.execute(approved)

    assert (
        receipt.state,
        receipt.updated,
        receipt.failed,
        receipt.reason_code,
    ) == (ImportSessionState.NEEDS_ATTENTION, 0, 1, "version_conflict")
    folder = folders.get_folder_by_path(("Imported Root",))
    assert folder is not None
    assert folders.list_memberships(note_ids=(existing.note_id,)) == ()
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.folder_effects[0].state is ImportEffectState.APPLIED
    assert durable.membership_effects[0].state is ImportEffectState.FAILED
    assert durable.membership_effects[0].reason_code == "version_conflict"
    assert durable.items[0].outcome is ImportItemOutcome.FAILED


def test_executor_all_skip_including_unsupported_and_failed_mutates_nothing(
    real_executor,
) -> None:
    executor, _target, receipts, _service, folders, db = real_executor
    items = tuple(
        _execution_item(
            item_id=f"skip-{classification.value}",
            payloads=(),
            action=ImportAction.SKIP,
            classification=classification,
        )
        for classification in (
            ImportClassification.UNSUPPORTED,
            ImportClassification.FAILED,
        )
    )

    receipt = executor.execute(
        _approved_execution_plan(
            *items,
            proposed_folder_paths=(("Unauthorized Root",),),
        )
    )

    assert (
        receipt.state,
        receipt.total,
        receipt.completed,
        receipt.skipped,
        receipt.failed,
    ) == (ImportSessionState.COMPLETED, 2, 2, 2, 0)
    assert folders.get_folder_by_path(("Unauthorized Root",)) is None
    assert db.get_connection().execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.folder_effects == ()
    assert durable.payload_effects == ()
    assert durable.membership_effects == ()
    assert all(item.outcome is ImportItemOutcome.SKIPPED for item in durable.items)


def test_executor_creates_only_authorized_folders_and_reuses_only_resolved_root(
    real_executor,
) -> None:
    executor, _target, _receipts, _service, folders, _db = real_executor
    existing_root = folders.create_folder(
        name="Imported Root", parent_id=None, folder_id="preexisting-root"
    )
    item = _execution_item(
        item_id="root-reuse",
        payloads=(_payload(),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Imported Root", "Used Child")),),
        add_membership=True,
    )

    receipt = executor.execute(
        _approved_execution_plan(
            item,
            proposed_folder_paths=(
                ("Imported Root",),
                ("Imported Root", "Used Child"),
                ("Imported Root", "Unused Child"),
            ),
            root_collision=RootCollisionState(
                proposed_label="Imported Root",
                collides=True,
                choice=RootCollisionChoice.USE_EXISTING,
            ),
        )
    )

    assert receipt.imported == 1
    assert folders.get_folder_by_path(("Imported Root",)) == existing_root
    used = folders.get_folder_by_path(("Imported Root", "Used Child"))
    assert used is not None
    assert used.folder_id == _expected_folder_id("/imported root/used child")
    assert folders.get_folder_by_path(("Imported Root", "Unused Child")) is None


def test_executor_folder_conflict_fails_dependent_work_without_creating_note(
    real_executor,
) -> None:
    executor, target, receipts, _service, folders, _db = real_executor
    folders.create_folder(
        name="Imported Root", parent_id=None, folder_id="different-root-id"
    )
    item = _execution_item(
        item_id="folder-conflict",
        payloads=(_payload(),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Imported Root",)),),
        add_membership=True,
    )

    receipt = executor.execute(
        _approved_execution_plan(
            item,
            proposed_folder_paths=(("Imported Root",),),
        )
    )

    assert (
        receipt.state,
        receipt.imported,
        receipt.failed,
        receipt.retryable,
        receipt.reason_code,
    ) == (ImportSessionState.NEEDS_ATTENTION, 0, 1, 0, "folder_conflict")
    assert target.read_note(note_id=_expected_note_id(item.item_id, 0)) is None
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.folder_effects[0].state is ImportEffectState.FAILED
    assert durable.folder_effects[0].reason_code == "folder_conflict"
    assert durable.payload_effects[0].state is ImportEffectState.PENDING
    assert durable.items[0].outcome is ImportItemOutcome.FAILED


def test_executor_reports_honest_mixed_counts_and_retryable_failures(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, _receipts, _service, _folders, _db = real_executor
    existing = target.create_note(note_id="existing-mixed", payload=_payload())
    imported = _execution_item(
        item_id="created",
        payloads=(_payload(title="Created"),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Imported Root",)),),
        add_membership=True,
    )
    updated = _execution_item(
        item_id="updated",
        payloads=(_payload(title="Updated", content="Updated body"),),
        action=ImportAction.UPDATE_EXISTING,
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        replace_content=True,
    )
    skipped = _execution_item(
        item_id="skipped",
        payloads=(),
        action=ImportAction.SKIP,
        classification=ImportClassification.UNSUPPORTED,
    )
    retry_payload = _payload(title="Retry target")
    failed = _execution_item(
        item_id="retryable",
        payloads=(retry_payload,),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Imported Root",)),),
        add_membership=True,
    )
    original_create = target.create_note

    def selectively_busy(*, note_id: str, payload: ParsedNotePayload):
        if payload is retry_payload:
            raise ImportTargetRetryableError
        return original_create(note_id=note_id, payload=payload)

    monkeypatch.setattr(target, "create_note", selectively_busy)

    receipt = executor.execute(
        _approved_execution_plan(
            imported,
            updated,
            skipped,
            failed,
            proposed_folder_paths=(("Imported Root",),),
        )
    )

    assert (
        receipt.state,
        receipt.total,
        receipt.completed,
        receipt.imported,
        receipt.updated,
        receipt.skipped,
        receipt.failed,
        receipt.retryable,
        receipt.reason_code,
    ) == (ImportSessionState.NEEDS_ATTENTION, 4, 4, 1, 1, 1, 1, 1, "database_busy")


def test_executor_persists_each_effect_before_advancing_and_aborts_fatal_faults(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    item = _execution_item(
        item_id="fatal-after-create",
        payloads=(_payload(),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Imported Root",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Imported Root",),),
    )

    def fatal_membership(*, folder_id: str, note_id: str) -> None:
        raise ImportTargetInternalError

    monkeypatch.setattr(target, "attach_membership", fatal_membership)

    with pytest.raises(ImportTargetInternalError):
        executor.execute(approved)

    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.state is ImportSessionState.RUNNING
    assert durable.folder_effects[0].state is ImportEffectState.APPLIED
    assert durable.payload_effects[0].state is ImportEffectState.APPLIED
    assert durable.membership_effects[0].state is ImportEffectState.PENDING
    assert durable.items[0].outcome is ImportItemOutcome.PENDING
    assert target.read_note(note_id=_expected_note_id(item.item_id, 0)) is not None


def test_executor_summarizes_retryability_across_all_membership_failures(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    item = _execution_item(
        item_id="mixed-membership-failures",
        payloads=(_payload(),),
        action=ImportAction.CREATE_NEW,
        memberships=(
            ProposedFolderMembership(0, ("Imported Root", "Permanent")),
            ProposedFolderMembership(0, ("Imported Root", "Retryable")),
        ),
        add_membership=True,
    )

    def mixed_failures(*, folder_id: str, note_id: str) -> None:
        del note_id
        if folder_id == _expected_folder_id("/imported root/permanent"):
            raise ImportTargetPermanentError
        raise ImportTargetRetryableError

    monkeypatch.setattr(target, "attach_membership", mixed_failures)

    receipt = executor.execute(
        _approved_execution_plan(
            item,
            proposed_folder_paths=(
                ("Imported Root",),
                ("Imported Root", "Permanent"),
                ("Imported Root", "Retryable"),
            ),
        )
    )

    assert (
        receipt.state,
        receipt.failed,
        receipt.retryable,
        receipt.reason_code,
    ) == (ImportSessionState.NEEDS_ATTENTION, 1, 1, "database_busy")
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert [effect.state for effect in durable.membership_effects] == [
        ImportEffectState.FAILED,
        ImportEffectState.FAILED,
    ]
    assert [effect.retryable for effect in durable.membership_effects] == [False, True]


def test_executor_summarizes_retryability_across_all_required_folder_failures(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    item = _execution_item(
        item_id="mixed-folder-failures",
        payloads=(_payload(),),
        action=ImportAction.CREATE_NEW,
        memberships=(
            ProposedFolderMembership(0, ("Imported Root", "Permanent")),
            ProposedFolderMembership(0, ("Imported Root", "Retryable")),
        ),
        add_membership=True,
    )
    original_ensure = target.ensure_folder

    def mixed_folder_failures(
        *, segments: tuple[str, ...], folder_id: str, allow_existing: bool
    ):
        if segments[-1] == "Permanent":
            raise ImportTargetPermanentError
        if segments[-1] == "Retryable":
            raise ImportTargetRetryableError
        return original_ensure(
            segments=segments,
            folder_id=folder_id,
            allow_existing=allow_existing,
        )

    monkeypatch.setattr(target, "ensure_folder", mixed_folder_failures)

    receipt = executor.execute(
        _approved_execution_plan(
            item,
            proposed_folder_paths=(
                ("Imported Root",),
                ("Imported Root", "Permanent"),
                ("Imported Root", "Retryable"),
            ),
        )
    )

    assert (
        receipt.state,
        receipt.failed,
        receipt.retryable,
        receipt.reason_code,
    ) == (ImportSessionState.NEEDS_ATTENTION, 1, 1, "database_busy")
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert [effect.state for effect in durable.folder_effects] == [
        ImportEffectState.APPLIED,
        ImportEffectState.FAILED,
        ImportEffectState.FAILED,
    ]
    assert [effect.retryable for effect in durable.folder_effects] == [
        False,
        False,
        True,
    ]
    assert durable.payload_effects[0].state is ImportEffectState.PENDING


def test_target_membership_attach_is_idempotent_and_requires_active_targets(
    target_harness,
) -> None:
    target, _service, folders, db = target_harness
    folder = target.ensure_folder(
        segments=("Imported",), folder_id=_FOLDER_ID, allow_existing=False
    )
    target.create_note(note_id=_NOTE_ID, payload=_payload())

    first = target.attach_membership(folder_id=folder.folder_id, note_id=_NOTE_ID)
    retry = target.attach_membership(folder_id=folder.folder_id, note_id=_NOTE_ID)

    assert retry == first
    assert (
        _active_membership_count(db, folder_id=folder.folder_id, note_id=_NOTE_ID) == 1
    )

    db.get_connection().execute(
        "UPDATE note_folders SET deleted = 1 WHERE id = ?", (folder.folder_id,)
    )
    db.get_connection().commit()
    with pytest.raises(ImportTargetPermanentError):
        target.attach_membership(folder_id=folder.folder_id, note_id=_NOTE_ID)

    other = folders.create_folder(name="Other", parent_id=None)
    db.get_connection().execute(
        "UPDATE notes SET deleted = 1 WHERE id = ?", (_NOTE_ID,)
    )
    db.get_connection().commit()
    with pytest.raises(ImportTargetPermanentError):
        target.attach_membership(folder_id=other.folder_id, note_id=_NOTE_ID)


@pytest.mark.parametrize(
    "hostile",
    [
        "database is locked at /private/secret.db for note-secret",
        "database table is locked at /private/secret.db for note-secret",
        "database is busy at /private/secret.db for note-secret",
    ],
)
def test_target_exception_translation_is_safe_and_does_not_chain_raw_errors(
    target_harness, monkeypatch: pytest.MonkeyPatch, hostile: str
) -> None:
    target, _service, _folders, _db = target_harness

    def locked_read(*_args, **_kwargs):
        try:
            raise sqlite3.OperationalError(hostile)
        except sqlite3.OperationalError as exc:
            raise CharactersRAGDBError(hostile) from exc

    monkeypatch.setattr(target, "_read_note", locked_read)

    with pytest.raises(ImportTargetRetryableError) as caught:
        target.read_note(note_id=_NOTE_ID)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert hostile not in str(caught.value)
    assert hostile not in repr(caught.value)
    assert hostile not in "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )


def test_target_internal_value_errors_are_fatal_and_baseexceptions_escape(
    target_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, _service, _folders, _db = target_harness

    def invalid_read(*_args, **_kwargs):
        raise ValueError("hostile /private/path note-secret")

    monkeypatch.setattr(target, "_read_note", invalid_read)
    with pytest.raises(ImportTargetInternalError) as caught:
        target.read_note(note_id=_NOTE_ID)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "hostile" not in repr(caught.value)

    def interrupted_read(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(target, "_read_note", interrupted_read)
    with pytest.raises(KeyboardInterrupt):
        target.read_note(note_id=_NOTE_ID)


@pytest.mark.parametrize(
    ("fault", "expected_type"),
    [
        (
            FolderValidationError("private folder validation detail"),
            ImportTargetPermanentError,
        ),
        (
            FolderCapabilityError(
                reason_code="private-reason",
                user_message="private folder capability detail",
            ),
            ImportTargetPermanentError,
        ),
        (
            CharactersRAGDBError("private database detail"),
            ImportTargetPermanentError,
        ),
        (sqlite3.OperationalError("private SQL detail"), ImportTargetPermanentError),
        (sqlite3.IntegrityError("private integrity detail"), ImportTargetConflictError),
        (FolderCollisionError("private collision detail"), ImportTargetConflictError),
        (FolderConflictError("private conflict detail"), ImportTargetConflictError),
    ],
)
def test_target_expected_faults_keep_their_item_level_translation(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    fault: Exception,
    expected_type: type[ImportTargetError],
) -> None:
    target, _service, _folders, _db = target_harness

    def expected_failure(*_args, **_kwargs):
        raise fault

    monkeypatch.setattr(target, "_read_note", expected_failure)

    with pytest.raises(expected_type) as caught:
        target.read_note(note_id=_NOTE_ID)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "private" not in str(caught.value)
    assert "private" not in repr(caught.value)
    assert "private" not in "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )


class _UnexpectedRuntimeFault(RuntimeError):
    pass


@pytest.mark.parametrize(
    "fault_type",
    [AssertionError, MemoryError, TypeError, ValueError, _UnexpectedRuntimeFault],
)
def test_target_unexpected_faults_abort_as_safe_internal_errors(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fault_type: type[Exception],
) -> None:
    target, _service, _folders, _db = target_harness
    private_detail = "PRIVATE-INTERNAL-DETAIL /private/source note-secret"

    def unexpected_failure(*_args, **_kwargs):
        raise fault_type(private_detail)

    monkeypatch.setattr(target, "_read_note", unexpected_failure)
    loguru_messages: list[str] = []
    sink_id = loguru_logger.add(lambda message: loguru_messages.append(str(message)))
    try:
        with pytest.raises(Exception) as caught:
            target.read_note(note_id=_NOTE_ID)
    finally:
        loguru_logger.remove(sink_id)

    assert type(caught.value) is ImportTargetInternalError
    assert not isinstance(caught.value, ImportTargetError)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.__suppress_context__ is True
    rendered_traceback = "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    rendered = (
        str(caught.value)
        + repr(caught.value)
        + rendered_traceback
        + "".join(loguru_messages)
        + caplog.text
    )
    assert private_detail not in rendered


def test_unexpected_fault_cannot_borrow_item_level_sqlite_classification(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _service, _folders, _db = target_harness
    private_detail = "PRIVATE-WRAPPED-INTERNAL /private/source note-secret"

    def unexpected_failure(*_args, **_kwargs):
        try:
            raise sqlite3.OperationalError(f"database is locked {private_detail}")
        except sqlite3.OperationalError as exc:
            raise _UnexpectedRuntimeFault(private_detail) from exc

    monkeypatch.setattr(target, "_read_note", unexpected_failure)

    with pytest.raises(ImportTargetInternalError) as caught:
        target.read_note(note_id=_NOTE_ID)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.__suppress_context__ is True
    rendered_traceback = "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    assert private_detail not in rendered_traceback


def test_unknown_fault_chain_is_not_inspected_for_sqlite_contention(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _service, _folders, _db = target_harness
    private_detail = "PRIVATE-HOSTILE-SQLITE /private/source note-secret"

    class HostileOperationalError(sqlite3.OperationalError):
        def __str__(self) -> str:
            raise ValueError(private_detail)

    def unexpected_failure(*_args, **_kwargs):
        try:
            raise HostileOperationalError
        except HostileOperationalError as exc:
            raise _UnexpectedRuntimeFault(private_detail) from exc

    monkeypatch.setattr(target, "_read_note", unexpected_failure)

    with pytest.raises(ImportTargetInternalError) as caught:
        target.read_note(note_id=_NOTE_ID)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert private_detail not in "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda target: target.read_note(note_id=""),
        lambda target: target.replace_note(
            note_id=_NOTE_ID,
            expected_version=0,
            payload=_payload(),
        ),
        lambda target: target.keywords_match(note_id=_NOTE_ID, keywords="invalid"),
        lambda target: target.ensure_folder(
            segments=("Private",),
            folder_id=_FOLDER_ID,
            allow_existing="invalid",  # type: ignore[arg-type]
        ),
    ],
)
def test_explicit_target_input_validation_remains_permanent_without_context(
    target_harness,
    operation,
) -> None:
    target, _service, _folders, _db = target_harness

    with pytest.raises(ImportTargetPermanentError) as caught:
        operation(target)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_constructor_explicit_validation_remains_permanent_without_context() -> None:
    with pytest.raises(ImportTargetPermanentError) as caught:
        LocalNoteImportTarget(
            db=object(),  # type: ignore[arg-type]
            folder_repository=object(),  # type: ignore[arg-type]
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_constructor_component_access_faults_are_safe_fatal_errors(
    target_harness,
) -> None:
    _target, _service, _folders, db = target_harness
    private_detail = "PRIVATE-CONSTRUCTOR-FAULT /private/database"

    class FailingRepository(LocalNoteFolderRepository):
        @property
        def db(self):
            raise ValueError(private_detail)

    repository = object.__new__(FailingRepository)

    with pytest.raises(ImportTargetInternalError) as caught:
        LocalNoteImportTarget(db=db, folder_repository=repository)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert private_detail not in str(caught.value)
    assert private_detail not in repr(caught.value)
    assert private_detail not in "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )


@pytest.mark.parametrize(
    ("helper_name", "operation"),
    [
        ("_keyword_rows", lambda target: target.read_note(note_id=_NOTE_ID)),
        (
            "_linked_keyword_rows",
            lambda target: target.sync_keywords(
                note_id=_NOTE_ID,
                keywords=("replacement",),
            ),
        ),
        (
            "_keyword_rows",
            lambda target: target.keywords_match(
                note_id=_NOTE_ID,
                keywords=("Project", "draft"),
            ),
        ),
    ],
)
def test_internal_db_row_type_errors_are_safe_fatal_errors(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    operation,
) -> None:
    target, _service, _folders, _db = target_harness
    target.create_note(note_id=_NOTE_ID, payload=_payload())
    monkeypatch.setattr(target, helper_name, lambda *_args, **_kwargs: [object()])

    with pytest.raises(ImportTargetInternalError) as caught:
        operation(target)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "row",
    [
        {"id": _NOTE_ID, "title": "Title", "version": 1},
        {"id": _NOTE_ID, "title": object(), "content": "Body", "version": 1},
        {"id": _NOTE_ID, "title": "Title", "content": "Body", "version": True},
    ],
)
def test_selected_note_row_contract_faults_are_safe_fatal_errors(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    row: dict[str, object],
) -> None:
    target, _service, _folders, db = target_harness

    class FakeCursor:
        def execute(self, *_args, **_kwargs):
            return self

        def fetchone(self):
            return row

    @contextmanager
    def fake_transaction():
        yield FakeCursor()

    monkeypatch.setattr(db, "transaction", fake_transaction)
    monkeypatch.setattr(target, "_keyword_rows", lambda *_args: [])

    with pytest.raises(ImportTargetInternalError) as caught:
        target.read_note(note_id=_NOTE_ID)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_selected_keyword_row_missing_column_is_a_safe_fatal_error(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _service, _folders, _db = target_harness
    target.create_note(note_id=_NOTE_ID, payload=_payload())
    monkeypatch.setattr(target, "_keyword_rows", lambda *_args: [{}])

    with pytest.raises(ImportTargetInternalError) as caught:
        target.read_note(note_id=_NOTE_ID)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"id": "not-an-integer", "keyword": "Project", "deleted": 0},
        {"id": 1, "keyword": object(), "deleted": 0},
        {"id": 1, "keyword": "Project", "deleted": 2},
    ],
)
def test_linked_keyword_row_contract_faults_are_safe_fatal_errors(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    row: dict[str, object],
) -> None:
    target, _service, _folders, _db = target_harness
    target.create_note(note_id=_NOTE_ID, payload=_payload())
    monkeypatch.setattr(target, "_linked_keyword_rows", lambda *_args: [row])

    with pytest.raises(ImportTargetInternalError) as caught:
        target.sync_keywords(note_id=_NOTE_ID, keywords=("replacement",))

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "rows",
    [
        (),
        [{}],
        [{"keyword": object()}],
    ],
)
def test_keyword_match_row_contract_faults_are_safe_fatal_errors(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    rows: object,
) -> None:
    target, _service, _folders, _db = target_harness
    target.create_note(note_id=_NOTE_ID, payload=_payload())
    monkeypatch.setattr(target, "_keyword_rows", lambda *_args: rows)

    with pytest.raises(ImportTargetInternalError) as caught:
        target.keywords_match(note_id=_NOTE_ID, keywords=("Project", "draft"))

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    ("selected_row", "lastrowid"),
    [
        (None, "not-an-integer"),
        ({"id": "not-an-integer", "version": 1, "deleted": 0}, None),
        ({"id": 1, "version": True, "deleted": 0}, None),
    ],
)
def test_keyword_ensure_contract_faults_are_safe_fatal_errors(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    selected_row: dict[str, object] | None,
    lastrowid: object,
) -> None:
    target, _service, _folders, db = target_harness

    class FakeResult:
        def __init__(self, *, row=None, inserted_id=None) -> None:
            self._row = row
            self.lastrowid = inserted_id

        def fetchone(self):
            return self._row

    class FakeCursor:
        def execute(self, query, *_args, **_kwargs):
            if "SELECT id, version, deleted FROM keywords" in query:
                return FakeResult(row=selected_row)
            if "INSERT INTO keywords" in query:
                return FakeResult(inserted_id=lastrowid)
            raise AssertionError("unexpected query")

    @contextmanager
    def fake_transaction():
        yield FakeCursor()

    monkeypatch.setattr(db, "transaction", fake_transaction)
    monkeypatch.setattr(target, "_active_note_exists", lambda *_args: True)
    monkeypatch.setattr(target, "_linked_keyword_rows", lambda *_args: [])

    with pytest.raises(ImportTargetInternalError) as caught:
        target.sync_keywords(note_id=_NOTE_ID, keywords=("replacement",))

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_create_note_postcondition_fault_is_safe_fatal_and_rolls_back(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _service, _folders, db = target_harness
    reads = iter((None, None))
    monkeypatch.setattr(target, "_read_note", lambda *_args: next(reads))

    with pytest.raises(ImportTargetInternalError) as caught:
        target.create_note(note_id=_NOTE_ID, payload=_payload())

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert (
        db.get_connection()
        .execute("SELECT 1 FROM notes WHERE id = ?", (_NOTE_ID,))
        .fetchone()
        is None
    )


def test_replace_note_postcondition_fault_is_safe_fatal_and_rolls_back(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _service, _folders, db = target_harness
    original = target.create_note(note_id=_NOTE_ID, payload=_payload())
    connection = db.get_connection()

    def snapshot_target_state() -> tuple[object, ...]:
        return (
            tuple(
                connection.execute(
                    "SELECT title, content, version FROM notes WHERE id = ?",
                    (_NOTE_ID,),
                ).fetchone()
            ),
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT k.keyword, k.deleted, k.version "
                    "FROM keywords AS k "
                    "JOIN note_keywords AS nk ON nk.keyword_id = k.id "
                    "WHERE nk.note_id = ? ORDER BY k.id",
                    (_NOTE_ID,),
                ).fetchall()
            ],
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT entity, entity_id, operation, version, payload "
                    "FROM sync_log ORDER BY change_id"
                ).fetchall()
            ],
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT rowid, title, content FROM notes_fts ORDER BY rowid"
                ).fetchall()
            ],
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT rowid, keyword FROM keywords_fts ORDER BY rowid"
                ).fetchall()
            ],
        )

    before = snapshot_target_state()
    private_detail = "PRIVATE-REPLACE-POSTCONDITION /private/note-secret"
    real_read = target._read_note
    read_count = 0

    def mismatching_postcondition_read(cursor, note_id):
        nonlocal read_count
        read_count += 1
        actual = real_read(cursor, note_id)
        if read_count == 2:
            assert actual is not None
            return replace(actual, content=private_detail)
        return actual

    monkeypatch.setattr(target, "_read_note", mismatching_postcondition_read)

    with pytest.raises(ImportTargetInternalError) as caught:
        target.replace_note(
            note_id=_NOTE_ID,
            expected_version=original.version,
            payload=_payload(
                title="Replacement",
                content="Replacement body",
                keywords=("replacement",),
            ),
        )

    assert not isinstance(caught.value, ImportTargetError)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert private_detail not in str(caught.value)
    assert private_detail not in repr(caught.value)
    assert private_detail not in "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    assert snapshot_target_state() == before


@pytest.mark.parametrize(
    "folder_id",
    ["invalid/path", ".leading", "éclair"],
)
def test_target_reuses_strict_folder_id_validation_before_repository_access(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    folder_id: str,
) -> None:
    target, _service, folders, _db = target_harness
    repository_accesses: list[str] = []

    def record_path_access(*_args, **_kwargs):
        repository_accesses.append("get_folder_by_path")

    def record_id_access(*_args, **_kwargs):
        repository_accesses.append("get_folder")

    original_create_folder = folders.create_folder

    def record_create(*args, **kwargs):
        repository_accesses.append("create_folder")
        return original_create_folder(*args, **kwargs)

    monkeypatch.setattr(folders, "get_folder_by_path", record_path_access)
    monkeypatch.setattr(folders, "get_folder", record_id_access)
    monkeypatch.setattr(folders, "create_folder", record_create)

    with pytest.raises(ImportTargetPermanentError) as caught:
        target.ensure_folder(
            segments=("Private Folder",),
            folder_id=folder_id,
            allow_existing=False,
        )

    assert repository_accesses == []
    assert folder_id not in repr(caught.value)


def test_target_construction_and_mutations_do_not_log_private_values(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _target, _service, folders, db = target_harness
    private_user_id = db.client_id
    private_db_path = db.db_path_str
    private_note_id = "private-note-id-never-log"
    private_title = "PRIVATE-TITLE-NEVER-LOG"
    private_content = "PRIVATE-CONTENT-NEVER-LOG"
    private_keyword = "PRIVATE-KEYWORD-NEVER-LOG"
    private_folder_id = "private-folder-id-never-log"
    private_source_path = "PRIVATE-SOURCE-PATH-NEVER-LOG"
    private_raw_error = (
        "database is locked PRIVATE-RAW-FAILURE-NEVER-LOG /private/source/path"
    )
    loguru_messages: list[str] = []
    sink_id = loguru_logger.add(lambda message: loguru_messages.append(str(message)))
    try:
        target = LocalNoteImportTarget(db=db, folder_repository=folders)
        folder = target.ensure_folder(
            segments=(private_source_path,),
            folder_id=private_folder_id,
            allow_existing=False,
        )
        target.create_note(
            note_id=private_note_id,
            payload=_payload(
                title=private_title,
                content=private_content,
                keywords=(private_keyword,),
            ),
        )
        target.read_note(note_id=private_note_id)
        target.keywords_match(note_id=private_note_id, keywords=(private_keyword,))
        target.attach_membership(
            folder_id=folder.folder_id,
            note_id=private_note_id,
        )
        target.replace_note(
            note_id=private_note_id,
            expected_version=1,
            payload=_payload(
                title="replacement-title-never-log",
                content="replacement-content-never-log",
                keywords=("replacement-keyword-never-log",),
            ),
        )
        target.sync_keywords(note_id=private_note_id, keywords=(private_keyword,))

        def failing_read(*_args, **_kwargs):
            try:
                raise sqlite3.OperationalError(private_raw_error)
            except sqlite3.OperationalError as exc:
                raise CharactersRAGDBError(private_raw_error) from exc

        monkeypatch.setattr(target, "_read_note", failing_read)
        with pytest.raises(ImportTargetRetryableError):
            target.read_note(note_id=private_note_id)
    finally:
        loguru_logger.remove(sink_id)

    rendered = "".join(loguru_messages) + caplog.text
    for private_value in (
        private_note_id,
        private_user_id,
        private_db_path,
        private_title,
        private_content,
        private_keyword,
        private_folder_id,
        private_source_path,
        "replacement-title-never-log",
        "replacement-content-never-log",
        "replacement-keyword-never-log",
        private_raw_error,
    ):
        assert private_value not in rendered


def test_target_injected_failures_do_not_log_raw_private_details(
    target_harness,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target, _service, _folders, _db = target_harness
    hostile = (
        "database is locked PRIVATE-RAW-FAILURE /private/source/path note-id-never-log"
    )

    def failing_read(*_args, **_kwargs):
        try:
            raise sqlite3.OperationalError(hostile)
        except sqlite3.OperationalError as exc:
            raise CharactersRAGDBError(hostile) from exc

    monkeypatch.setattr(target, "_read_note", failing_read)
    loguru_messages: list[str] = []
    sink_id = loguru_logger.add(lambda message: loguru_messages.append(str(message)))
    try:
        with pytest.raises(ImportTargetRetryableError):
            target.read_note(note_id="note-id-never-log")
    finally:
        loguru_logger.remove(sink_id)

    assert hostile not in "".join(loguru_messages) + caplog.text


def test_target_sql_matches_service_metadata_fts_and_sync_conventions(
    target_harness,
) -> None:
    target, service, _folders, db = target_harness
    service.add_note(
        "target-user", "Ordinary title", "Ordinary service body", note_id="ordinary"
    )
    target.create_note(
        note_id=_NOTE_ID,
        payload=_payload(
            title="Target title", content="Target searchable body", keywords=("Legacy",)
        ),
    )

    connection = db.get_connection()
    ordinary = connection.execute(
        "SELECT created_at, last_modified, client_id, version FROM notes WHERE id = ?",
        ("ordinary",),
    ).fetchone()
    created = connection.execute(
        "SELECT created_at, last_modified, client_id, version FROM notes WHERE id = ?",
        (_NOTE_ID,),
    ).fetchone()
    assert ordinary is not None and created is not None
    assert created[2:] == ordinary[2:] == ("target-user", 1)
    assert created[0] == created[1]
    assert created[0].tzinfo == UTC
    assert (
        connection.execute(
            """
        SELECT COUNT(*) FROM notes_fts
        JOIN notes ON notes.rowid = notes_fts.rowid
        WHERE notes.id = ? AND notes_fts MATCH ?
        """,
            (_NOTE_ID, "searchable"),
        ).fetchone()[0]
        == 1
    )
    assert (
        connection.execute(
            """
        SELECT COUNT(*) FROM keywords_fts
        JOIN keywords ON keywords.id = keywords_fts.rowid
        JOIN note_keywords ON note_keywords.keyword_id = keywords.id
        WHERE note_keywords.note_id = ? AND keywords_fts MATCH ?
        """,
            (_NOTE_ID, "Legacy"),
        ).fetchone()[0]
        == 1
    )

    target.replace_note(
        note_id=_NOTE_ID,
        expected_version=1,
        payload=_payload(
            title="Updated title",
            content="Updated current body",
            keywords=("Current",),
        ),
    )

    updated = connection.execute(
        "SELECT client_id, version FROM notes WHERE id = ?", (_NOTE_ID,)
    ).fetchone()
    assert tuple(updated) == ("target-user", 2)
    note_sync = connection.execute(
        "SELECT operation, timestamp, client_id, version, payload FROM sync_log "
        "WHERE entity = 'notes' AND entity_id = ? ORDER BY change_id",
        (_NOTE_ID,),
    ).fetchall()
    assert [(row["operation"], row["version"]) for row in note_sync] == [
        ("create", 1),
        ("update", 2),
    ]
    for row in note_sync:
        payload = json.loads(row["payload"])
        assert set(payload) == {
            "id",
            "title",
            "content",
            "created_at",
            "last_modified",
            "deleted",
            "client_id",
            "version",
        }
        assert payload["id"] == _NOTE_ID
        assert payload["deleted"] == 0
        assert payload["client_id"] == row["client_id"] == "target-user"
        assert payload["version"] == row["version"]
        assert payload["last_modified"] == row["timestamp"].isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
    link_sync = connection.execute(
        "SELECT entity_id, operation, client_id, version, payload FROM sync_log "
        "WHERE entity = 'note_keywords' AND entity_id LIKE ? ORDER BY change_id",
        (f"{_NOTE_ID}_%",),
    ).fetchall()
    assert [row["operation"] for row in link_sync] == ["create", "delete", "create"]
    for row in link_sync:
        payload = json.loads(row["payload"])
        assert row["client_id"] == "target-user"
        assert row["version"] == 1
        assert row["entity_id"] == f"{payload['note_id']}_{payload['keyword_id']}"
        assert payload["note_id"] == _NOTE_ID
        expected_keys = {"note_id", "keyword_id"}
        if row["operation"] == "create":
            expected_keys.add("created_at")
        assert set(payload) == expected_keys
    legacy = connection.execute(
        "SELECT deleted FROM keywords WHERE keyword = 'Legacy'"
    ).fetchone()
    assert legacy[0] == 0
    assert (
        connection.execute(
            """
        SELECT COUNT(*) FROM notes_fts
        JOIN notes ON notes.rowid = notes_fts.rowid
        WHERE notes.id = ? AND notes_fts MATCH ?
        """,
            (_NOTE_ID, "Target OR searchable"),
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            """
        SELECT COUNT(*) FROM notes_fts
        JOIN notes ON notes.rowid = notes_fts.rowid
        WHERE notes.id = ? AND notes_fts MATCH ?
        """,
            (_NOTE_ID, "Updated AND current"),
        ).fetchone()[0]
        == 1
    )
    assert (
        connection.execute(
            """
        SELECT COUNT(*) FROM keywords_fts
        JOIN keywords ON keywords.id = keywords_fts.rowid
        JOIN note_keywords ON note_keywords.keyword_id = keywords.id
        WHERE note_keywords.note_id = ? AND keywords_fts MATCH ?
        """,
            (_NOTE_ID, "Legacy"),
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            """
        SELECT COUNT(*) FROM keywords_fts
        JOIN keywords ON keywords.id = keywords_fts.rowid
        JOIN note_keywords ON note_keywords.keyword_id = keywords.id
        WHERE note_keywords.note_id = ? AND keywords_fts MATCH ?
        """,
            (_NOTE_ID, "Current"),
        ).fetchone()[0]
        == 1
    )
