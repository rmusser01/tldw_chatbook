"""Behavior tests for deterministic local Database Notes import target calls."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from datetime import UTC
from itertools import pairwise
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
    ImportExecutionProgress,
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
from tldw_chatbook.Notes.note_import_planner import (
    PriorImportObservation,
    _private_payload_fingerprint,
)
from tldw_chatbook.Notes.note_import_receipts import (
    ImportReceiptConflictError,
    NoteImportReceiptRepository,
)
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
    folder_sync_id = _db.get_connection().execute(
        "SELECT sync_id FROM note_folders WHERE id = ?", (_FOLDER_ID,)
    ).fetchone()[0]
    assert str(UUID(folder_sync_id)) == folder_sync_id
    assert UUID(folder_sync_id).version == 4


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
    portable_ids = [
        row[0]
        for row in db.get_connection().execute(
            "SELECT k.sync_id FROM keywords AS k "
            "JOIN note_keywords AS nk ON nk.keyword_id = k.id "
            "WHERE nk.note_id = ? ORDER BY k.id",
            (_NOTE_ID,),
        )
    ]
    assert len(portable_ids) == 2
    assert all(str(UUID(sync_id)) == sync_id for sync_id in portable_ids)
    assert all(UUID(sync_id).version == 4 for sync_id in portable_ids)
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


@pytest.mark.parametrize(
    ("batch_size", "item_count", "expected_running_completed"),
    [
        (1, 5, [0, 1, 2, 3, 4, 5]),
        (2, 5, [0, 2, 4, 5]),
        (100, 101, [0, 100, 101]),
    ],
)
def test_executor_processes_items_in_bounded_batches_with_monotonic_frozen_progress(
    target_harness,
    tmp_path: Path,
    batch_size: int,
    item_count: int,
    expected_running_completed: list[int],
) -> None:
    target, _service, _folders, _db = target_harness
    receipts = NoteImportReceiptRepository(tmp_path / f"progress-{batch_size}.sqlite3")
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=batch_size,
    )
    items = tuple(
        _execution_item(
            item_id=f"skip-{index}",
            payloads=(),
            action=ImportAction.SKIP,
            classification=ImportClassification.UNSUPPORTED,
        )
        for index in range(item_count)
    )
    progress: list[ImportExecutionProgress] = []
    plan = _execution_plan(*items)
    if item_count > plan.bounds.max_files:
        plan = replace(
            plan,
            bounds=replace(plan.bounds, max_files=item_count),
        )

    receipt = executor.execute(
        approve_note_import_plan(plan, approval_id=_EXECUTION_APPROVAL_ID),
        progress_callback=progress.append,
    )

    assert receipt.state is ImportSessionState.COMPLETED
    assert [
        update.completed
        for update in progress
        if update.state is ImportSessionState.RUNNING
    ] == expected_running_completed
    assert progress[-1].state is ImportSessionState.COMPLETED
    assert progress[-1].completed == item_count
    assert all(
        earlier.completed <= later.completed for earlier, later in pairwise(progress)
    )
    with pytest.raises(FrozenInstanceError):
        progress[-1].completed = 0  # type: ignore[misc]


def test_executor_cancellation_before_first_batch_mutates_no_target(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    item = _execution_item(
        item_id="cancel-before-first",
        payloads=(_payload(),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Cancelled Root",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Cancelled Root",),),
    )
    cancel = threading.Event()
    cancel.set()

    def forbidden_create(**_kwargs):
        raise AssertionError("cancelled execution reached the target")

    monkeypatch.setattr(target, "create_note", forbidden_create)
    progress: list[ImportExecutionProgress] = []

    receipt = executor.execute(
        approved,
        cancel_event=cancel,
        progress_callback=progress.append,
    )

    assert receipt.state is ImportSessionState.CANCELLED
    assert progress[-1].state is ImportSessionState.CANCELLED
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.state is ImportSessionState.CANCELLED
    assert durable.items[0].outcome is ImportItemOutcome.PENDING
    assert durable.payload_effects[0].state is ImportEffectState.PENDING
    assert durable.folder_effects[0].state is ImportEffectState.PENDING


def test_executor_cancels_only_between_batches_after_current_target_call_returns(
    target_harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _service, _folders, _db = target_harness
    receipts = NoteImportReceiptRepository(tmp_path / "between-batches.sqlite3")
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=1,
    )
    first = _execution_item(
        item_id="blocked-first",
        payloads=(_payload(title="First"),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Imported Root",)),),
        add_membership=True,
    )
    second = _execution_item(
        item_id="must-remain-pending",
        payloads=(_payload(title="Second"),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Imported Root",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        first,
        second,
        proposed_folder_paths=(("Imported Root",),),
    )
    entered = threading.Event()
    release = threading.Event()
    cancel = threading.Event()
    original_create = target.create_note
    calls: list[str] = []

    def blocking_create(*, note_id: str, payload: ParsedNotePayload):
        calls.append(payload.title)
        if payload.title == "First":
            entered.set()
            assert release.wait(timeout=5)
        return original_create(note_id=note_id, payload=payload)

    monkeypatch.setattr(target, "create_note", blocking_create)
    result: Queue[object] = Queue()
    worker = threading.Thread(
        target=lambda: result.put(executor.execute(approved, cancel_event=cancel)),
        daemon=True,
    )
    worker.start()
    assert entered.wait(timeout=5)
    cancel.set()
    time.sleep(0.05)
    assert worker.is_alive()
    release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    receipt = result.get_nowait()
    assert receipt.state is ImportSessionState.CANCELLED
    assert calls == ["First"]
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.items[0].outcome is ImportItemOutcome.IMPORTED
    assert durable.items[1].outcome is ImportItemOutcome.PENDING


def test_executor_cancels_between_folder_batches_after_current_target_returns(
    target_harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _service, folders, _db = target_harness
    receipts = NoteImportReceiptRepository(tmp_path / "folder-batch-cancel.sqlite3")
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=1,
    )
    items = tuple(
        _execution_item(
            item_id=item_id,
            payloads=(_payload(title=title),),
            action=ImportAction.CREATE_NEW,
            memberships=(ProposedFolderMembership(0, (folder_name,)),),
            add_membership=True,
        )
        for item_id, title, folder_name in (
            ("first-folder-item", "First", "A First Folder"),
            ("second-folder-item", "Second", "B Second Folder"),
        )
    )
    approved = _approved_execution_plan(
        *items,
        proposed_folder_paths=(("A First Folder",), ("B Second Folder",)),
    )
    started = threading.Event()
    release = threading.Event()
    cancel = threading.Event()
    original_ensure = target.ensure_folder
    calls: list[tuple[str, ...]] = []

    def blocking_ensure(*, segments, folder_id, allow_existing):
        copied = tuple(segments)
        calls.append(copied)
        if copied == ("A First Folder",):
            started.set()
            if not release.wait(5):
                raise AssertionError("folder target release timed out")
        return original_ensure(
            segments=copied,
            folder_id=folder_id,
            allow_existing=allow_existing,
        )

    monkeypatch.setattr(target, "ensure_folder", blocking_ensure)
    progress: list[ImportExecutionProgress] = []
    results: Queue[object] = Queue()

    def run() -> None:
        try:
            results.put(
                executor.execute(
                    approved,
                    cancel_event=cancel,
                    progress_callback=progress.append,
                )
            )
        except BaseException as error:  # noqa: BLE001 - cross-thread test capture
            results.put(error)

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(5)
    cancel.set()
    assert worker.is_alive()
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    result = results.get_nowait()
    if isinstance(result, BaseException):
        raise result

    assert result.state is ImportSessionState.CANCELLED
    assert calls == [("A First Folder",)]
    assert folders.get_folder_by_path(("A First Folder",)) is not None
    assert folders.get_folder_by_path(("B Second Folder",)) is None
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.state is ImportSessionState.CANCELLED
    assert [effect.state for effect in durable.folder_effects] == [
        ImportEffectState.APPLIED,
        ImportEffectState.PENDING,
    ]
    assert all(item.outcome is ImportItemOutcome.PENDING for item in durable.items)
    assert progress[-1].state is ImportSessionState.CANCELLED
    assert all(update.completed == 0 for update in progress)
    assert all(
        earlier.completed <= later.completed for earlier, later in pairwise(progress)
    )
    with pytest.raises(FrozenInstanceError):
        progress[-1].completed = 1  # type: ignore[misc]


@pytest.mark.asyncio
async def test_executor_async_offloads_the_whole_execution_and_keeps_loop_responsive(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, _receipts, _service, _folders, _db = real_executor
    item = _execution_item(
        item_id="async-blocked",
        payloads=(_payload(),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Async Root",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Async Root",),),
    )
    entered = threading.Event()
    release = threading.Event()
    original_create = target.create_note
    callback_threads: list[int] = []
    caller_thread = threading.get_ident()

    def blocking_create(*, note_id: str, payload: ParsedNotePayload):
        entered.set()
        assert release.wait(timeout=5)
        return original_create(note_id=note_id, payload=payload)

    monkeypatch.setattr(target, "create_note", blocking_create)
    heartbeat = 0

    async def beat() -> None:
        nonlocal heartbeat
        while not release.is_set():
            heartbeat += 1
            await asyncio.sleep(0)

    heartbeat_task = asyncio.create_task(beat())
    execution_task = asyncio.create_task(
        executor.execute_async(
            approved,
            progress_callback=lambda _progress: callback_threads.append(
                threading.get_ident()
            ),
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    await asyncio.sleep(0.02)
    observed_heartbeat = heartbeat
    release.set()
    receipt = await execution_task
    await heartbeat_task

    assert receipt.state is ImportSessionState.COMPLETED
    assert observed_heartbeat > 1
    assert callback_threads
    assert set(callback_threads) == {caller_thread}


@pytest.mark.asyncio
async def test_execute_async_file_backed_recursive_import_reopens_and_replays_once(
    tmp_path: Path,
) -> None:
    """Smoke the complete async boundary across both durable SQLite owners."""
    notes_path = tmp_path / "notes.sqlite3"
    receipt_path = tmp_path / "notes-sync-state.sqlite3"
    item = _execution_item(
        item_id="recursive-file-backed",
        payloads=(
            _payload(
                title="Nested note",
                content="Durable body",
                keywords=("Recursive", "Smoke"),
            ),
        ),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Selected Root", "Nested Folder")),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(
            ("Selected Root",),
            ("Selected Root", "Nested Folder"),
        ),
    )
    note_id = _expected_note_id(item.item_id, 0)

    first_db = CharactersRAGDB(notes_path, client_id="file-backed-smoke")
    try:
        first_folders = LocalNoteFolderRepository(first_db)
        first_executor = NoteImportExecutor(
            target=LocalNoteImportTarget(
                db=first_db,
                folder_repository=first_folders,
            ),
            receipt_repository=NoteImportReceiptRepository(receipt_path),
            batch_size=1,
        )

        first_receipt = await first_executor.execute_async(approved)

        assert first_receipt.state is ImportSessionState.COMPLETED
        assert first_receipt.imported == 1
    finally:
        first_db.close_connection()

    reopened_db = CharactersRAGDB(notes_path, client_id="file-backed-smoke")
    try:
        reopened_folders = LocalNoteFolderRepository(reopened_db)
        root = reopened_folders.get_folder_by_path(("Selected Root",))
        nested = reopened_folders.get_folder_by_path(("Selected Root", "Nested Folder"))
        assert root is not None
        assert nested is not None
        assert nested.parent_id == root.folder_id

        note_row = (
            reopened_db.get_connection()
            .execute(
                "SELECT title, content, version FROM notes "
                "WHERE id = ? AND deleted = 0",
                (note_id,),
            )
            .fetchone()
        )
        assert tuple(note_row) == ("Nested note", "Durable body", 1)
        assert (
            _active_membership_count(
                reopened_db,
                folder_id=nested.folder_id,
                note_id=note_id,
            )
            == 1
        )

        reopened_receipts = NoteImportReceiptRepository(receipt_path)
        durable = reopened_receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
        assert durable.state is ImportSessionState.COMPLETED
        assert durable.items[0].outcome is ImportItemOutcome.IMPORTED
        assert all(
            effect.state is ImportEffectState.APPLIED
            for effect in (
                *durable.folder_effects,
                *durable.payload_effects,
                *durable.membership_effects,
            )
        )

        reopened_executor = NoteImportExecutor(
            target=LocalNoteImportTarget(
                db=reopened_db,
                folder_repository=reopened_folders,
            ),
            receipt_repository=reopened_receipts,
            batch_size=1,
        )
        replay_receipt = await reopened_executor.execute_async(approved)

        assert replay_receipt == first_receipt
        assert (
            reopened_db.get_connection()
            .execute("SELECT COUNT(*) FROM notes WHERE id = ?", (note_id,))
            .fetchone()[0]
            == 1
        )
        assert (
            _active_membership_count(
                reopened_db,
                folder_id=nested.folder_id,
                note_id=note_id,
            )
            == 1
        )
    finally:
        reopened_db.close_connection()


@pytest.mark.parametrize("crash_boundary", ["folder", "payload", "membership", "item"])
def test_executor_reopens_and_reconciles_each_create_crash_window(
    target_harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_boundary: str,
) -> None:
    target, _service, folders, db = target_harness
    receipt_path = tmp_path / f"crash-{crash_boundary}.sqlite3"
    receipts = NoteImportReceiptRepository(receipt_path)
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=1,
    )
    payload = _payload(keywords=("Exact", "replay"))
    item = _execution_item(
        item_id=f"crash-{crash_boundary}",
        payloads=(payload,),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Crash Root",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Crash Root",),),
    )
    original_effects = receipts.transition_effects
    original_item = receipts.transition_item
    crashed = False

    def crash_before_effect_transition(approval_id, transitions):
        nonlocal crashed
        copied = tuple(transitions)
        if not crashed and copied[0].category.value == crash_boundary:
            crashed = True
            raise RuntimeError("simulated process interruption")
        return original_effects(approval_id, copied)

    def crash_before_item_transition(approval_id, item_id, outcome, **kwargs):
        nonlocal crashed
        if not crashed and crash_boundary == "item":
            crashed = True
            raise RuntimeError("simulated process interruption")
        return original_item(approval_id, item_id, outcome, **kwargs)

    monkeypatch.setattr(receipts, "transition_effects", crash_before_effect_transition)
    monkeypatch.setattr(receipts, "transition_item", crash_before_item_transition)

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        executor.execute(approved)
    assert crashed
    monkeypatch.undo()

    reopened_receipts = NoteImportReceiptRepository(receipt_path)
    resumed = NoteImportExecutor(
        target=target,
        receipt_repository=reopened_receipts,
        batch_size=1,
    ).execute(approved)

    assert resumed.state is ImportSessionState.COMPLETED
    assert (resumed.imported, resumed.failed) == (1, 0)
    note_id = _expected_note_id(item.item_id, 0)
    note = target.read_note(note_id=note_id)
    assert note is not None
    assert (note.version, set(note.keywords)) == (1, set(payload.keywords))
    folder = folders.get_folder_by_path(("Crash Root",))
    assert folder is not None
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM notes WHERE id = ? AND deleted = 0", (note_id,))
        .fetchone()[0]
        == 1
    )
    assert (
        _active_membership_count(db, folder_id=folder.folder_id, note_id=note_id) == 1
    )
    durable = reopened_receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.state is ImportSessionState.COMPLETED
    assert all(
        effect.state is ImportEffectState.APPLIED
        for effect in (
            *durable.folder_effects,
            *durable.payload_effects,
            *durable.membership_effects,
        )
    )


def test_executor_reopens_after_committed_create_item_finalization(
    target_harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _service, folders, db = target_harness
    receipt_path = tmp_path / "post-create-item-finalization.sqlite3"
    receipts = NoteImportReceiptRepository(receipt_path)
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=1,
    )
    payload = _payload(title="Post-finalized create", keywords=("finalized",))
    item = _execution_item(
        item_id="post-finalized-create",
        payloads=(payload,),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Post Finalized Create",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Post Finalized Create",),),
    )
    original_transition_item = receipts.transition_item
    crashed = False

    def crash_after_item_finalize(approval_id, item_id, outcome, **kwargs):
        nonlocal crashed
        result = original_transition_item(approval_id, item_id, outcome, **kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError("simulated post-create-finalization interruption")
        return result

    monkeypatch.setattr(receipts, "transition_item", crash_after_item_finalize)
    with pytest.raises(RuntimeError, match="post-create-finalization interruption"):
        executor.execute(approved)
    crashed_snapshot = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert crashed_snapshot.state is ImportSessionState.RUNNING
    assert crashed_snapshot.items[0].outcome is ImportItemOutcome.IMPORTED
    session_id = crashed_snapshot.session_id
    monkeypatch.undo()

    def forbidden_create(**_kwargs):
        raise AssertionError("finalized create was repeated")

    monkeypatch.setattr(target, "create_note", forbidden_create)
    reopened_receipts = NoteImportReceiptRepository(receipt_path)
    resumed = NoteImportExecutor(
        target=target,
        receipt_repository=reopened_receipts,
        batch_size=1,
    ).execute(approved)

    note_id = _expected_note_id(item.item_id, 0)
    note = target.read_note(note_id=note_id)
    folder = folders.get_folder_by_path(("Post Finalized Create",))
    assert resumed.state is ImportSessionState.COMPLETED
    assert (
        reopened_receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID).session_id
        == session_id
    )
    assert note is not None and (note.version, note.title) == (1, payload.title)
    assert folder is not None
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM notes WHERE id = ?", (note_id,))
        .fetchone()[0]
        == 1
    )
    assert (
        _active_membership_count(db, folder_id=folder.folder_id, note_id=note_id) == 1
    )


def test_executor_reopens_after_committed_update_item_finalization(
    target_harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _service, folders, db = target_harness
    receipt_path = tmp_path / "post-update-item-finalization.sqlite3"
    receipts = NoteImportReceiptRepository(receipt_path)
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=1,
    )
    existing = target.create_note(
        note_id="post-finalized-update-target",
        payload=_payload(content="Before", keywords=("old",)),
    )
    replacement = _payload(content="After", keywords=("new", "finalized"))
    item = _execution_item(
        item_id="post-finalized-update",
        payloads=(replacement,),
        action=ImportAction.UPDATE_EXISTING,
        memberships=(ProposedFolderMembership(0, ("Post Finalized Update",)),),
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        replace_content=True,
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Post Finalized Update",),),
    )
    original_transition_item = receipts.transition_item
    crashed = False

    def crash_after_item_finalize(approval_id, item_id, outcome, **kwargs):
        nonlocal crashed
        result = original_transition_item(approval_id, item_id, outcome, **kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError("simulated post-update-finalization interruption")
        return result

    monkeypatch.setattr(receipts, "transition_item", crash_after_item_finalize)
    with pytest.raises(RuntimeError, match="post-update-finalization interruption"):
        executor.execute(approved)
    crashed_snapshot = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert crashed_snapshot.state is ImportSessionState.RUNNING
    assert crashed_snapshot.items[0].outcome is ImportItemOutcome.UPDATED
    session_id = crashed_snapshot.session_id
    monkeypatch.undo()

    def forbidden_replace(**_kwargs):
        raise AssertionError("finalized optimistic update was repeated")

    monkeypatch.setattr(target, "replace_note", forbidden_replace)
    reopened_receipts = NoteImportReceiptRepository(receipt_path)
    resumed = NoteImportExecutor(
        target=target,
        receipt_repository=reopened_receipts,
        batch_size=1,
    ).execute(approved)

    updated = target.read_note(note_id=existing.note_id)
    folder = folders.get_folder_by_path(("Post Finalized Update",))
    assert resumed.state is ImportSessionState.COMPLETED
    assert (
        reopened_receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID).session_id
        == session_id
    )
    assert updated is not None
    assert (updated.version, updated.content, set(updated.keywords)) == (
        existing.version + 1,
        replacement.content,
        set(replacement.keywords),
    )
    assert folder is not None
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM notes WHERE id = ?", (existing.note_id,))
        .fetchone()[0]
        == 1
    )
    assert (
        _active_membership_count(
            db,
            folder_id=folder.folder_id,
            note_id=existing.note_id,
        )
        == 1
    )


def test_executor_reconciles_update_crash_without_repeating_optimistic_update(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    existing = target.create_note(
        note_id="crash-update-target",
        payload=_payload(content="Before", keywords=("old",)),
    )
    replacement = _payload(content="After", keywords=("new", "exact"))
    item = _execution_item(
        item_id="crash-update",
        payloads=(replacement,),
        action=ImportAction.UPDATE_EXISTING,
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        replace_content=True,
    )
    approved = _approved_execution_plan(item)
    original_effects = receipts.transition_effects
    crashed = False

    def crash_after_update(approval_id, transitions):
        nonlocal crashed
        copied = tuple(transitions)
        if not crashed and copied[0].category.value == "payload":
            crashed = True
            raise RuntimeError("simulated update interruption")
        return original_effects(approval_id, copied)

    monkeypatch.setattr(receipts, "transition_effects", crash_after_update)
    with pytest.raises(RuntimeError, match="simulated update interruption"):
        executor.execute(approved)
    assert target.read_note(note_id=existing.note_id).version == existing.version + 1
    monkeypatch.undo()

    receipt = executor.execute(approved)

    updated = target.read_note(note_id=existing.note_id)
    assert updated is not None
    assert receipt.state is ImportSessionState.COMPLETED
    assert (updated.version, updated.content, set(updated.keywords)) == (
        existing.version + 1,
        replacement.content,
        set(replacement.keywords),
    )


def test_executor_fresh_update_conflict_uses_version_conflict_reason(
    real_executor,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    existing = target.create_note(
        note_id="fresh-update-conflict",
        payload=_payload(content="Before"),
    )
    item = _execution_item(
        item_id="fresh-update-conflict",
        payloads=(_payload(content="Approved"),),
        action=ImportAction.UPDATE_EXISTING,
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        replace_content=True,
    )
    approved = _approved_execution_plan(item)
    target.replace_note(
        note_id=existing.note_id,
        expected_version=existing.version,
        payload=_payload(content="Concurrent edit"),
    )

    receipt = executor.execute(approved)

    assert (receipt.state, receipt.reason_code) == (
        ImportSessionState.NEEDS_ATTENTION,
        "version_conflict",
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.payload_effects[0].reason_code == "version_conflict"


def test_executor_interrupted_update_with_later_version_uses_note_conflict_reason(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    existing = target.create_note(
        note_id="recovery-update-conflict",
        payload=_payload(content="Before"),
    )
    replacement = _payload(content="Approved")
    item = _execution_item(
        item_id="recovery-update-conflict",
        payloads=(replacement,),
        action=ImportAction.UPDATE_EXISTING,
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        replace_content=True,
    )
    approved = _approved_execution_plan(item)
    original_effects = receipts.transition_effects
    crashed = False

    def crash_after_update(approval_id, transitions):
        nonlocal crashed
        copied = tuple(transitions)
        if not crashed and copied[0].category.value == "payload":
            crashed = True
            raise RuntimeError("simulated update interruption")
        return original_effects(approval_id, copied)

    monkeypatch.setattr(receipts, "transition_effects", crash_after_update)
    with pytest.raises(RuntimeError, match="simulated update interruption"):
        executor.execute(approved)
    monkeypatch.undo()
    target.replace_note(
        note_id=existing.note_id,
        expected_version=existing.version + 1,
        payload=_payload(content="Later edit"),
    )

    receipt = executor.retry_failed(approved)

    assert (receipt.state, receipt.reason_code) == (
        ImportSessionState.NEEDS_ATTENTION,
        "note_conflict",
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.payload_effects[0].reason_code == "note_conflict"


def test_executor_durably_fails_applied_update_that_diverges_before_item_finalize(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    existing = target.create_note(
        note_id="applied-update-diverged",
        payload=_payload(content="Before"),
    )
    item = _execution_item(
        item_id="applied-update-diverged",
        payloads=(_payload(content="Approved"),),
        action=ImportAction.UPDATE_EXISTING,
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        replace_content=True,
    )
    approved = _approved_execution_plan(item)
    original_transition_item = receipts.transition_item
    crashed = False

    def crash_before_item_finalize(approval_id, item_id, outcome, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("simulated item-finalize interruption")
        return original_transition_item(approval_id, item_id, outcome, **kwargs)

    monkeypatch.setattr(receipts, "transition_item", crash_before_item_finalize)
    with pytest.raises(RuntimeError, match="item-finalize interruption"):
        executor.execute(approved)
    monkeypatch.undo()
    session_id = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID).session_id
    target.replace_note(
        note_id=existing.note_id,
        expected_version=existing.version + 1,
        payload=_payload(content="Later edit"),
    )

    receipt = executor.retry_failed(approved)

    assert (receipt.state, receipt.reason_code) == (
        ImportSessionState.NEEDS_ATTENTION,
        "note_conflict",
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.payload_effects[0].state is ImportEffectState.APPLIED
    assert durable.payload_effects[0].reason_code == "note_conflict"
    assert durable.items[0].outcome is ImportItemOutcome.FAILED
    assert durable.items[0].reason_code == "note_conflict"
    assert durable.items[0].retryable is False
    assert receipts.prior_observations_for_plan(approved.plan) == ()

    def forbidden_read(**_kwargs):
        raise AssertionError("permanent reconciliation conflict was retried")

    monkeypatch.setattr(target, "read_note", forbidden_read)
    retried = executor.retry_failed(approved)
    retried_durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert retried.state is ImportSessionState.NEEDS_ATTENTION
    assert retried_durable.session_id == session_id
    assert retried_durable.payload_effects[0].state is ImportEffectState.APPLIED
    assert retried_durable.payload_effects[0].reason_code == "note_conflict"
    assert retried_durable.items[0].reason_code == "note_conflict"


def test_executor_applied_multi_create_divergence_preserves_per_payload_counts(
    target_harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _service, _folders, _db = target_harness
    receipts = NoteImportReceiptRepository(tmp_path / "multi-divergence.sqlite3")
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=1,
    )
    item = _execution_item(
        item_id="multi-divergence",
        payloads=(
            _payload(title="Diverges"),
            _payload(title="Remains exact"),
        ),
        action=ImportAction.CREATE_NEW,
        memberships=(
            ProposedFolderMembership(0, ("Multi Divergence",)),
            ProposedFolderMembership(1, ("Multi Divergence",)),
        ),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Multi Divergence",),),
    )
    original_transition_item = receipts.transition_item
    crashed = False

    def crash_before_item_finalize(approval_id, item_id, outcome, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("simulated multi item-finalize interruption")
        return original_transition_item(approval_id, item_id, outcome, **kwargs)

    monkeypatch.setattr(receipts, "transition_item", crash_before_item_finalize)
    with pytest.raises(RuntimeError, match="multi item-finalize interruption"):
        executor.execute(approved)
    monkeypatch.undo()
    divergent_note_id = _expected_note_id(item.item_id, 0)
    target.replace_note(
        note_id=divergent_note_id,
        expected_version=1,
        payload=_payload(title="Later divergent edit"),
    )

    receipt = executor.retry_failed(approved)

    assert (receipt.state, receipt.imported, receipt.failed, receipt.completed) == (
        ImportSessionState.NEEDS_ATTENTION,
        1,
        1,
        2,
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert [effect.state for effect in durable.payload_effects] == [
        ImportEffectState.APPLIED,
        ImportEffectState.APPLIED,
    ]
    assert [effect.reason_code for effect in durable.payload_effects] == [
        "note_conflict",
        None,
    ]
    assert durable.items[0].outcome is ImportItemOutcome.FAILED
    exact_note = target.read_note(note_id=_expected_note_id(item.item_id, 1))
    assert exact_note is not None
    assert (exact_note.version, exact_note.title) == (1, "Remains exact")


def test_executor_applied_update_divergence_fails_pending_membership(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    existing = target.create_note(
        note_id="update-pending-membership",
        payload=_payload(content="Before"),
    )
    item = _execution_item(
        item_id="update-pending-membership",
        payloads=(_payload(content="Approved"),),
        action=ImportAction.UPDATE_EXISTING,
        memberships=(ProposedFolderMembership(0, ("Pending Membership",)),),
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        replace_content=True,
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Pending Membership",),),
    )
    original_effects = receipts.transition_effects
    crashed = False

    def crash_after_payload_receipt(approval_id, transitions):
        nonlocal crashed
        copied = tuple(transitions)
        result = original_effects(approval_id, copied)
        if not crashed and copied[0].category.value == "payload":
            crashed = True
            raise RuntimeError("simulated post-payload-receipt interruption")
        return result

    monkeypatch.setattr(receipts, "transition_effects", crash_after_payload_receipt)
    with pytest.raises(RuntimeError, match="post-payload-receipt interruption"):
        executor.execute(approved)
    monkeypatch.undo()
    crashed_snapshot = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert crashed_snapshot.payload_effects[0].state is ImportEffectState.APPLIED
    assert crashed_snapshot.membership_effects[0].state is ImportEffectState.PENDING
    target.replace_note(
        note_id=existing.note_id,
        expected_version=existing.version + 1,
        payload=_payload(content="Later divergent edit"),
    )

    receipt = executor.retry_failed(approved)

    assert (receipt.state, receipt.updated, receipt.failed) == (
        ImportSessionState.NEEDS_ATTENTION,
        0,
        1,
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.payload_effects[0].state is ImportEffectState.APPLIED
    assert durable.payload_effects[0].reason_code == "note_conflict"
    assert durable.membership_effects[0].state is ImportEffectState.FAILED
    assert durable.membership_effects[0].reason_code == "note_conflict"
    assert durable.items[0].outcome is ImportItemOutcome.FAILED


def test_executor_retry_failed_reuses_exact_session_and_only_retries_retryable_work(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    retry_payload = _payload(title="Retryable")
    permanent_payload = _payload(title="Permanent")
    items = tuple(
        _execution_item(
            item_id=title.casefold(),
            payloads=(payload,),
            action=ImportAction.CREATE_NEW,
            memberships=(ProposedFolderMembership(0, ("Retry Root",)),),
            add_membership=True,
        )
        for title, payload in (
            ("Retryable", retry_payload),
            ("Permanent", permanent_payload),
        )
    )
    approved = _approved_execution_plan(
        *items,
        proposed_folder_paths=(("Retry Root",),),
    )
    original_create = target.create_note
    calls: list[str] = []

    def fail_once(*, note_id: str, payload: ParsedNotePayload):
        calls.append(payload.title)
        if payload.title == "Retryable":
            raise ImportTargetRetryableError
        raise ImportTargetPermanentError

    monkeypatch.setattr(target, "create_note", fail_once)
    failed = executor.execute(approved)
    assert (failed.state, failed.failed, failed.retryable) == (
        ImportSessionState.NEEDS_ATTENTION,
        2,
        1,
    )
    session_id = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID).session_id
    calls.clear()

    def retry_only(*, note_id: str, payload: ParsedNotePayload):
        calls.append(payload.title)
        return original_create(note_id=note_id, payload=payload)

    monkeypatch.setattr(target, "create_note", retry_only)
    retried = executor.retry_failed(approved)

    assert calls == ["Retryable"]
    assert (retried.state, retried.imported, retried.failed, retried.retryable) == (
        ImportSessionState.NEEDS_ATTENTION,
        1,
        1,
        0,
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.session_id == session_id
    assert [effect.reason_code for effect in durable.payload_effects] == [
        None,
        "target_invalid",
    ]


def test_executor_retry_failed_rejects_changed_plan_digest_before_target_call(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, _receipts, _service, _folders, _db = real_executor
    payload = _payload(title="Retry digest")
    item = _execution_item(
        item_id="retry-digest",
        payloads=(payload,),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Retry Root",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Retry Root",),),
    )
    monkeypatch.setattr(
        target,
        "create_note",
        lambda **_kwargs: (_ for _ in ()).throw(ImportTargetRetryableError()),
    )
    executor.execute(approved)
    changed_item = replace(item, payloads=(_payload(title="Changed authority"),))
    changed = _approved_execution_plan(
        changed_item,
        proposed_folder_paths=(("Retry Root",),),
    )
    target_calls = 0

    def forbidden_target(**_kwargs):
        nonlocal target_calls
        target_calls += 1
        raise AssertionError("digest mismatch reached the target")

    monkeypatch.setattr(target, "create_note", forbidden_target)

    with pytest.raises(ImportReceiptConflictError, match="authority"):
        executor.retry_failed(changed)
    assert target_calls == 0


@pytest.mark.parametrize(
    ("original_payload", "substituted_payload"),
    [
        (_payload(title="Caf\u00e9"), _payload(title="Cafe\u0301")),
        (_payload(content="Caf\u00e9"), _payload(content="Cafe\u0301")),
        (
            _payload(keywords=("Caf\u00e9",)),
            _payload(keywords=("Cafe\u0301",)),
        ),
        (
            replace(_payload(), template_name="Caf\u00e9"),
            replace(_payload(), template_name="Cafe\u0301"),
        ),
    ],
)
def test_executor_rejects_canonically_equivalent_text_substitution_before_target_call(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
    original_payload: ParsedNotePayload,
    substituted_payload: ParsedNotePayload,
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    original_item = _execution_item(
        item_id="unicode-authority",
        payloads=(original_payload,),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Unicode Root",)),),
        add_membership=True,
    )
    original = _approved_execution_plan(
        original_item,
        proposed_folder_paths=(("Unicode Root",),),
    )
    receipts.begin(original, batch_size=25)
    substituted = _approved_execution_plan(
        replace(original_item, payloads=(substituted_payload,)),
        proposed_folder_paths=(("Unicode Root",),),
    )
    target_calls = 0

    def forbidden_target(**_kwargs):
        nonlocal target_calls
        target_calls += 1
        raise AssertionError("text substitution reached the target")

    monkeypatch.setattr(target, "create_note", forbidden_target)

    with pytest.raises(ImportReceiptConflictError, match="authority") as caught:
        executor.execute(substituted)
    assert target_calls == 0
    assert target.read_note(note_id=_expected_note_id(original_item.item_id, 0)) is None
    assert "Caf\u00e9" not in str(caught.value)
    assert "Cafe\u0301" not in str(caught.value)
    assert "Caf\u00e9" not in repr(substituted)
    assert "Cafe\u0301" not in repr(substituted)


def test_executor_retry_failed_resumes_cancelled_pending_work_in_same_session(
    real_executor,
) -> None:
    executor, _target, receipts, _service, _folders, _db = real_executor
    item = _execution_item(
        item_id="cancel-retry",
        payloads=(_payload(),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Cancel Retry",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Cancel Retry",),),
    )
    cancel = threading.Event()
    cancel.set()
    cancelled = executor.execute(approved, cancel_event=cancel)
    session_id = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID).session_id

    retried = executor.retry_failed(approved)

    assert cancelled.state is ImportSessionState.CANCELLED
    assert retried.state is ImportSessionState.COMPLETED
    assert (
        receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID).session_id == session_id
    )


def test_executor_create_identity_collision_uses_note_conflict_reason(
    real_executor,
) -> None:
    executor, target, _receipts, _service, _folders, _db = real_executor
    item = _execution_item(
        item_id="identity-collision",
        payloads=(_payload(title="Approved"),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Collision Root",)),),
        add_membership=True,
    )
    target.create_note(
        note_id=_expected_note_id(item.item_id, 0),
        payload=_payload(title="Different occupant"),
    )

    receipt = executor.execute(
        _approved_execution_plan(
            item,
            proposed_folder_paths=(("Collision Root",),),
        )
    )

    assert (receipt.state, receipt.reason_code) == (
        ImportSessionState.NEEDS_ATTENTION,
        "note_conflict",
    )


def test_completed_single_payload_receipt_produces_private_exact_prior_observation(
    real_executor,
) -> None:
    executor, _target, receipts, _service, _folders, _db = real_executor
    payload = _payload(title="Prior private", content="Private prior body")
    item = _execution_item(
        item_id="prior-single",
        payloads=(payload,),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Prior Root",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Prior Root",),),
    )
    executor.execute(approved)

    observations = receipts.prior_observations_for_plan(approved.plan)

    assert len(observations) == 1
    observation = observations[0]
    assert isinstance(observation, PriorImportObservation)
    assert observation.display_path == item.source.display_path
    assert observation.match_kind is ImportMatchKind.EXACT
    assert observation.note_id == _expected_note_id(item.item_id, 0)
    assert observation.note_version == 1
    assert observation.payload_fingerprint == _private_payload_fingerprint((payload,))
    rendered = repr(receipts) + repr(observation) + repr(observations)
    for private_value in (
        str(item.source.source_path),
        item.source.display_path,
        payload.content,
        observation.note_id,
        observation.payload_fingerprint,
    ):
        assert private_value not in rendered


def test_prior_observations_select_the_latest_completed_matching_source(
    target_harness,
    tmp_path: Path,
) -> None:
    target, _service, _folders, _db = target_harness
    receipts = NoteImportReceiptRepository(tmp_path / "prior-latest.sqlite3")
    first_payload = _payload(title="Older receipt", content="Older body")
    second_payload = _payload(title="Latest receipt", content="Latest body")
    first_item = _execution_item(
        item_id="same-source",
        payloads=(first_payload,),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Prior Latest",)),),
        add_membership=True,
    )
    second_item = replace(first_item, payloads=(second_payload,))
    first_id = _EXECUTION_APPROVAL_ID
    second_id = "00000000-0000-4000-8000-000000000042"
    first_plan = _execution_plan(
        first_item,
        proposed_folder_paths=(("Prior Latest",),),
    )
    second_plan = _execution_plan(
        second_item,
        proposed_folder_paths=(("Prior Latest",),),
        root_collision=RootCollisionState(
            proposed_label="Prior Latest",
            collides=True,
            choice=RootCollisionChoice.USE_EXISTING,
        ),
    )
    first = approve_note_import_plan(first_plan, approval_id=first_id)
    second = approve_note_import_plan(second_plan, approval_id=second_id)
    NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=1,
    ).execute(first)
    NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=1,
    ).execute(second)

    observation = receipts.prior_observations_for_plan(second.plan)[0]

    assert observation.note_id == str(
        uuid5(UUID(second_id), f"note:{second_item.item_id}:0")
    )
    assert observation.payload_fingerprint == _private_payload_fingerprint(
        (second_payload,)
    )


def test_prior_observations_do_not_fall_back_past_latest_multi_payload_receipt(
    target_harness,
    tmp_path: Path,
) -> None:
    target, _service, _folders, _db = target_harness
    receipts = NoteImportReceiptRepository(tmp_path / "prior-no-fallback.sqlite3")
    first_item = _execution_item(
        item_id="same-source-single",
        payloads=(_payload(title="Older single"),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Prior No Fallback",)),),
        add_membership=True,
    )
    latest_item = replace(
        first_item,
        item_id="same-source-multi",
        payloads=(
            _payload(title="Latest first"),
            _payload(title="Latest second"),
        ),
        memberships=(
            ProposedFolderMembership(0, ("Prior No Fallback",)),
            ProposedFolderMembership(1, ("Prior No Fallback",)),
        ),
    )
    first = approve_note_import_plan(
        _execution_plan(
            first_item,
            proposed_folder_paths=(("Prior No Fallback",),),
        ),
        approval_id=_EXECUTION_APPROVAL_ID,
    )
    latest = approve_note_import_plan(
        _execution_plan(
            latest_item,
            proposed_folder_paths=(("Prior No Fallback",),),
            root_collision=RootCollisionState(
                proposed_label="Prior No Fallback",
                collides=True,
                choice=RootCollisionChoice.USE_EXISTING,
            ),
        ),
        approval_id="00000000-0000-4000-8000-000000000043",
    )
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=1,
    )
    executor.execute(first)
    executor.execute(latest)

    assert receipts.prior_observations_for_plan(first.plan) == ()


def test_prior_observations_reject_duplicate_items_in_latest_source_session(
    target_harness,
    tmp_path: Path,
) -> None:
    target, _service, _folders, _db = target_harness
    receipts = NoteImportReceiptRepository(tmp_path / "prior-duplicate-source.sqlite3")
    older_item = _execution_item(
        item_id="duplicate-source-older",
        payloads=(_payload(title="Older exact source"),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Duplicate Source",)),),
        add_membership=True,
    )
    older = approve_note_import_plan(
        _execution_plan(
            older_item,
            proposed_folder_paths=(("Duplicate Source",),),
        ),
        approval_id=_EXECUTION_APPROVAL_ID,
    )
    first_duplicate = replace(
        older_item,
        item_id="duplicate-source-first",
        payloads=(_payload(title="Newest first duplicate"),),
    )
    second_duplicate = replace(
        older_item,
        item_id="duplicate-source-second",
        payloads=(_payload(title="Newest second duplicate"),),
    )
    newest = approve_note_import_plan(
        _execution_plan(
            first_duplicate,
            second_duplicate,
            proposed_folder_paths=(("Duplicate Source",),),
            root_collision=RootCollisionState(
                proposed_label="Duplicate Source",
                collides=True,
                choice=RootCollisionChoice.USE_EXISTING,
            ),
        ),
        approval_id="00000000-0000-4000-8000-000000000044",
    )
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=2,
    )
    executor.execute(older)
    executor.execute(newest)

    observations = receipts.prior_observations_for_plan(older.plan)

    assert observations == ()
    private_values = (
        str(older_item.source.source_path),
        older_item.source.display_path,
        older_item.payloads[0].content,
    )
    rendered = repr(receipts) + repr(observations)
    assert all(value not in rendered for value in private_values)


@pytest.mark.parametrize("excluded_state", ["multi", "failed", "cancelled", "missing"])
def test_prior_observations_omit_unconfirmed_or_non_single_payload_sources(
    target_harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    excluded_state: str,
) -> None:
    target, _service, _folders, _db = target_harness
    receipts = NoteImportReceiptRepository(tmp_path / f"prior-{excluded_state}.sqlite3")
    executor = NoteImportExecutor(
        target=target,
        receipt_repository=receipts,
        batch_size=1,
    )
    payloads = (
        (_payload(title="First"), _payload(title="Second"))
        if excluded_state == "multi"
        else (_payload(),)
    )
    memberships = tuple(
        ProposedFolderMembership(index, ("Prior Excluded",))
        for index in range(len(payloads))
    )
    item = _execution_item(
        item_id=f"prior-{excluded_state}",
        payloads=payloads,
        action=ImportAction.CREATE_NEW,
        memberships=memberships,
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Prior Excluded",),),
    )
    if excluded_state == "failed":
        monkeypatch.setattr(
            target,
            "create_note",
            lambda **_kwargs: (_ for _ in ()).throw(ImportTargetPermanentError()),
        )
        executor.execute(approved)
    elif excluded_state == "cancelled":
        cancel = threading.Event()
        cancel.set()
        executor.execute(approved, cancel_event=cancel)
    elif excluded_state == "multi":
        executor.execute(approved)

    assert receipts.prior_observations_for_plan(approved.plan) == ()


@pytest.mark.parametrize("fatal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_executor_never_records_process_control_exceptions_as_item_failures(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
    fatal_type: type[BaseException],
) -> None:
    executor, target, receipts, _service, _folders, _db = real_executor
    item = _execution_item(
        item_id=f"fatal-{fatal_type.__name__.casefold()}",
        payloads=(_payload(),),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Fatal Root",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Fatal Root",),),
    )

    def raise_fatal(**_kwargs):
        raise fatal_type

    monkeypatch.setattr(target, "create_note", raise_fatal)
    with pytest.raises(fatal_type):
        executor.execute(approved)

    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.state is ImportSessionState.RUNNING
    assert durable.items[0].outcome is ImportItemOutcome.PENDING
    assert durable.payload_effects[0].state is ImportEffectState.PENDING


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


def test_executor_update_keeps_content_and_unblocked_membership_independent(
    real_executor,
) -> None:
    executor, target, receipts, _service, folders, db = real_executor
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
        memberships=(
            ProposedFolderMembership(0, ("Imported Root",)),
            ProposedFolderMembership(0, ("Good Root",)),
        ),
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
            proposed_folder_paths=(("Imported Root",), ("Good Root",)),
        )
    )

    updated = target.read_note(note_id=existing.note_id)
    assert updated is not None
    assert (updated.content, set(updated.keywords), updated.version) == (
        replacement.content,
        set(replacement.keywords),
        existing.version + 1,
    )
    good_root = folders.get_folder_by_path(("Good Root",))
    assert good_root is not None
    assert (
        _active_membership_count(
            db, folder_id=good_root.folder_id, note_id=existing.note_id
        )
        == 1
    )
    assert (receipt.updated, receipt.failed, receipt.reason_code) == (
        0,
        1,
        "folder_conflict",
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert durable.payload_effects[0].state is ImportEffectState.APPLIED
    assert [effect.state for effect in durable.folder_effects] == [
        ImportEffectState.FAILED,
        ImportEffectState.APPLIED,
    ]
    assert [effect.state for effect in durable.membership_effects] == [
        ImportEffectState.FAILED,
        ImportEffectState.APPLIED,
    ]
    assert durable.membership_effects[0].reason_code == "folder_conflict"
    assert durable.membership_effects[0].retryable is False
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


def test_executor_membership_attach_rechecks_note_version_atomically(
    real_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, target, receipts, _service, folders, db = real_executor
    existing = target.create_note(note_id="membership-race", payload=_payload())
    item = _execution_item(
        item_id="membership-race",
        payloads=(_payload(),),
        action=ImportAction.UPDATE_EXISTING,
        memberships=(ProposedFolderMembership(0, ("Race Root",)),),
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=existing.note_id,
            note_version=existing.version,
        ),
        add_membership=True,
    )
    approved = _approved_execution_plan(
        item,
        proposed_folder_paths=(("Race Root",),),
    )
    original_attach = target.attach_membership
    raced = False

    def racing_attach(*, folder_id: str, note_id: str, **kwargs) -> None:
        nonlocal raced
        if not raced:
            raced = True
            db.get_connection().execute(
                "UPDATE notes SET version = version + 1 WHERE id = ?",
                (note_id,),
            )
            db.get_connection().commit()
        original_attach(folder_id=folder_id, note_id=note_id, **kwargs)

    monkeypatch.setattr(target, "attach_membership", racing_attach)

    receipt = executor.execute(approved)

    assert raced
    assert (
        receipt.state,
        receipt.updated,
        receipt.failed,
        receipt.reason_code,
    ) == (ImportSessionState.NEEDS_ATTENTION, 0, 1, "version_conflict")
    folder = folders.get_folder_by_path(("Race Root",))
    assert folder is not None
    assert (
        _active_membership_count(
            db,
            folder_id=folder.folder_id,
            note_id=existing.note_id,
        )
        == 0
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
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
        memberships=(
            ProposedFolderMembership(0, ("Imported Root", "Inherited Child")),
            ProposedFolderMembership(0, ("Good Root",)),
        ),
        add_membership=True,
    )

    receipt = executor.execute(
        _approved_execution_plan(
            item,
            proposed_folder_paths=(
                ("Imported Root",),
                ("Imported Root", "Inherited Child"),
                ("Good Root",),
            ),
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
    assert [effect.state for effect in durable.folder_effects] == [
        ImportEffectState.FAILED,
        ImportEffectState.APPLIED,
        ImportEffectState.FAILED,
    ]
    assert [effect.reason_code for effect in durable.folder_effects] == [
        "folder_conflict",
        None,
        "folder_conflict",
    ]
    assert durable.payload_effects[0].state is ImportEffectState.FAILED
    assert durable.payload_effects[0].reason_code == "folder_conflict"
    assert durable.payload_effects[0].retryable is False
    assert [effect.state for effect in durable.membership_effects] == [
        ImportEffectState.FAILED,
        ImportEffectState.FAILED,
    ]
    assert [effect.reason_code for effect in durable.membership_effects] == [
        "folder_conflict",
        "folder_conflict",
    ]
    assert [effect.retryable for effect in durable.membership_effects] == [False, False]
    assert durable.items[0].outcome is ImportItemOutcome.FAILED


def test_executor_folder_blocked_create_payload_does_not_stop_unaffected_payload(
    real_executor,
) -> None:
    executor, target, receipts, _service, folders, db = real_executor
    folders.create_folder(
        name="Blocked Root", parent_id=None, folder_id="different-blocked-root-id"
    )
    item = _execution_item(
        item_id="mixed-folder-dependencies",
        payloads=(
            _payload(title="Blocked"),
            _payload(title="Unaffected"),
        ),
        action=ImportAction.CREATE_NEW,
        memberships=(
            ProposedFolderMembership(0, ("Blocked Root",)),
            ProposedFolderMembership(1, ("Good Root",)),
        ),
        add_membership=True,
    )

    receipt = executor.execute(
        _approved_execution_plan(
            item,
            proposed_folder_paths=(("Blocked Root",), ("Good Root",)),
        )
    )

    assert (
        receipt.state,
        receipt.completed,
        receipt.imported,
        receipt.failed,
        receipt.retryable,
        receipt.reason_code,
    ) == (ImportSessionState.NEEDS_ATTENTION, 2, 1, 1, 0, "folder_conflict")
    blocked_note_id = _expected_note_id(item.item_id, 0)
    unaffected_note_id = _expected_note_id(item.item_id, 1)
    assert target.read_note(note_id=blocked_note_id) is None
    unaffected = target.read_note(note_id=unaffected_note_id)
    assert unaffected is not None
    assert unaffected.title == "Unaffected"
    good_root = folders.get_folder_by_path(("Good Root",))
    assert good_root is not None
    assert (
        _active_membership_count(
            db, folder_id=good_root.folder_id, note_id=unaffected_note_id
        )
        == 1
    )
    durable = receipts.load_session_snapshot(_EXECUTION_APPROVAL_ID)
    assert [effect.state for effect in durable.folder_effects] == [
        ImportEffectState.FAILED,
        ImportEffectState.APPLIED,
    ]
    assert [effect.state for effect in durable.payload_effects] == [
        ImportEffectState.FAILED,
        ImportEffectState.APPLIED,
    ]
    assert [effect.state for effect in durable.membership_effects] == [
        ImportEffectState.FAILED,
        ImportEffectState.APPLIED,
    ]
    assert durable.payload_effects[0].reason_code == "folder_conflict"
    assert durable.membership_effects[0].reason_code == "folder_conflict"
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

    def fatal_membership(
        *, folder_id: str, note_id: str, expected_note_version: int
    ) -> None:
        del folder_id, note_id, expected_note_version
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

    def mixed_failures(
        *, folder_id: str, note_id: str, expected_note_version: int
    ) -> None:
        del note_id, expected_note_version
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
    assert durable.payload_effects[0].state is ImportEffectState.FAILED
    assert durable.payload_effects[0].reason_code == "database_busy"
    assert durable.payload_effects[0].retryable is True
    assert [effect.state for effect in durable.membership_effects] == [
        ImportEffectState.FAILED,
        ImportEffectState.FAILED,
    ]
    assert [effect.reason_code for effect in durable.membership_effects] == [
        "target_invalid",
        "database_busy",
    ]
    assert [effect.retryable for effect in durable.membership_effects] == [False, True]


def test_target_membership_attach_is_idempotent_and_requires_active_targets(
    target_harness,
) -> None:
    target, _service, folders, db = target_harness
    folder = target.ensure_folder(
        segments=("Imported",), folder_id=_FOLDER_ID, allow_existing=False
    )
    target.create_note(note_id=_NOTE_ID, payload=_payload())

    first = target.attach_membership(
        folder_id=folder.folder_id,
        note_id=_NOTE_ID,
        expected_note_version=1,
    )
    retry = target.attach_membership(
        folder_id=folder.folder_id,
        note_id=_NOTE_ID,
        expected_note_version=1,
    )

    assert retry == first
    assert (
        _active_membership_count(db, folder_id=folder.folder_id, note_id=_NOTE_ID) == 1
    )

    db.get_connection().execute(
        "UPDATE note_folders SET deleted = 1 WHERE id = ?", (folder.folder_id,)
    )
    db.get_connection().commit()
    with pytest.raises(ImportTargetPermanentError):
        target.attach_membership(
            folder_id=folder.folder_id,
            note_id=_NOTE_ID,
            expected_note_version=1,
        )

    other = folders.create_folder(name="Other", parent_id=None)
    db.get_connection().execute(
        "UPDATE notes SET deleted = 1 WHERE id = ?", (_NOTE_ID,)
    )
    db.get_connection().commit()
    with pytest.raises(ImportTargetPermanentError):
        target.attach_membership(
            folder_id=other.folder_id,
            note_id=_NOTE_ID,
            expected_note_version=1,
        )


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


_EXPECTED_FAULT_CANARY = "NOTE-IMPORT-EXPECTED-FAULT-CANARY"


@pytest.mark.parametrize(
    ("fault", "expected_type"),
    [
        (
            FolderValidationError(f"{_EXPECTED_FAULT_CANARY} folder validation detail"),
            ImportTargetPermanentError,
        ),
        (
            FolderCapabilityError(
                reason_code=f"{_EXPECTED_FAULT_CANARY}-reason",
                user_message=f"{_EXPECTED_FAULT_CANARY} folder capability detail",
            ),
            ImportTargetPermanentError,
        ),
        (
            CharactersRAGDBError(f"{_EXPECTED_FAULT_CANARY} database detail"),
            ImportTargetPermanentError,
        ),
        (
            sqlite3.OperationalError(f"{_EXPECTED_FAULT_CANARY} SQL detail"),
            ImportTargetPermanentError,
        ),
        (
            sqlite3.IntegrityError(f"{_EXPECTED_FAULT_CANARY} integrity detail"),
            ImportTargetConflictError,
        ),
        (
            FolderCollisionError(f"{_EXPECTED_FAULT_CANARY} collision detail"),
            ImportTargetConflictError,
        ),
        (
            FolderConflictError(f"{_EXPECTED_FAULT_CANARY} conflict detail"),
            ImportTargetConflictError,
        ),
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
    assert _EXPECTED_FAULT_CANARY not in str(caught.value)
    assert _EXPECTED_FAULT_CANARY not in repr(caught.value)
    # A traceback's source paths may legitimately include /private (macOS).
    assert _EXPECTED_FAULT_CANARY not in "".join(
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
            expected_note_version=1,
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

    # task-19564: the v45 retention triggers drop superseded `sync_log`
    # versions, so the `create` row no longer survives the replace -- a note's
    # full text is not kept in the log once a newer version exists. Its
    # payload shape is still asserted, by reading it while it is the frontier.
    note_sync_on_create = connection.execute(
        "SELECT operation, timestamp, client_id, version, payload FROM sync_log "
        "WHERE entity = 'notes' AND entity_id = ? ORDER BY change_id",
        (_NOTE_ID,),
    ).fetchall()
    assert [
        (row["operation"], row["version"]) for row in note_sync_on_create
    ] == [("create", 1)]

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
        ("update", 2),
    ]
    for row in [*note_sync_on_create, *note_sync]:
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
