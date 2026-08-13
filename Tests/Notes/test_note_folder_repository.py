"""Behavior tests for the local Database Note folder repository."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
import sqlite3
import uuid

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
import tldw_chatbook.Notes.note_folder_repository as folder_repository_module
from tldw_chatbook.Notes.note_folder_models import (
    FolderCollisionError,
    FolderConflictError,
    FolderValidationError,
)
from tldw_chatbook.Notes.note_folder_repository import (
    LocalNoteFolderRepository,
    _raise_mutation_operational_error,
)


@pytest.fixture
def repository(tmp_path) -> Iterator[LocalNoteFolderRepository]:
    """Return a repository backed by the real ChaChaNotes SQLite database."""
    db = CharactersRAGDB(tmp_path / "folders.db", client_id="folder-tests")
    yield LocalNoteFolderRepository(db)
    db.close_connection()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _attach_membership(
    repository: LocalNoteFolderRepository,
    *,
    folder_id: str,
    note_id: str,
    ownership: str = "manual",
    owner_id: str = "",
    owner_active: bool = True,
) -> str:
    membership_id = str(uuid.uuid4())
    now = _timestamp()
    with repository.db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO note_folder_memberships(
                id, folder_id, note_id, ownership, owner_id, owner_active,
                version, deleted, created_at, modified_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
            """,
            (
                membership_id,
                folder_id,
                note_id,
                ownership,
                owner_id,
                int(owner_active),
                now,
                now,
            ),
        )
    return membership_id


def _folder_rows(
    repository: LocalNoteFolderRepository,
) -> tuple[tuple[object, ...], ...]:
    rows = repository.db.get_connection().execute(
        "SELECT id, parent_id, name, normalized_name, path, normalized_path, "
        "version, deleted, modified_at FROM note_folders ORDER BY id"
    ).fetchall()
    return tuple(tuple(row) for row in rows)


def test_create_and_list_nested_folders(repository: LocalNoteFolderRepository) -> None:
    work = repository.create_folder(name="Work", parent_id=None)
    plans = repository.create_folder(name="Plans", parent_id=work.folder_id)

    page = repository.list_children(parent_id=work.folder_id, limit=50, offset=0)

    assert page.folders == (plans,)
    assert plans.path == "/Work/Plans"
    assert plans.normalized_path == "/work/plans"


def test_create_rejects_unicode_equivalent_active_path(
    repository: LocalNoteFolderRepository,
) -> None:
    repository.create_folder(name="Résumé", parent_id=None)

    with pytest.raises(FolderCollisionError):
        repository.create_folder(name="re\u0301sume\u0301", parent_id=None)


def test_create_does_not_misclassify_other_unique_failures(
    repository: LocalNoteFolderRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = repository.create_folder(name="Existing", parent_id=None)
    monkeypatch.setattr(
        "tldw_chatbook.Notes.note_folder_repository.uuid.uuid4",
        lambda: uuid.UUID(existing.folder_id),
    )

    with pytest.raises(FolderValidationError) as caught:
        repository.create_folder(name="Different", parent_id=None)

    assert not isinstance(caught.value, FolderCollisionError)


def test_create_rejects_missing_or_deleted_parent(
    repository: LocalNoteFolderRepository,
) -> None:
    deleted = repository.create_folder(name="Deleted", parent_id=None)
    repository.db.get_connection().execute(
        "UPDATE note_folders SET deleted = 1 WHERE id = ?", (deleted.folder_id,)
    )
    repository.db.get_connection().commit()

    with pytest.raises(FolderValidationError):
        repository.create_folder(name="Child", parent_id="missing")
    with pytest.raises(FolderValidationError):
        repository.create_folder(name="Child", parent_id=deleted.folder_id)


def test_get_folder_excludes_deleted_unless_requested(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Archive", parent_id=None)
    repository.db.get_connection().execute(
        "UPDATE note_folders SET deleted = 1 WHERE id = ?", (folder.folder_id,)
    )
    repository.db.get_connection().commit()

    assert repository.get_folder(folder.folder_id) is None
    expected = type(folder)(
        folder_id=folder.folder_id,
        parent_id=None,
        name="Archive",
        path="/Archive",
        normalized_path="/archive",
        version=1,
        deleted=True,
    )
    assert repository.get_folder(folder.folder_id, include_deleted=True) == expected
    assert repository.get_folder("missing", include_deleted=True) is None


def test_list_children_is_deterministic_and_pages_zero_exact_and_limit_plus_one(
    repository: LocalNoteFolderRepository,
) -> None:
    parent = repository.create_folder(name="Parent", parent_id=None)
    zulu = repository.create_folder(name="Zulu", parent_id=parent.folder_id)
    alpha = repository.create_folder(name="alpha", parent_id=parent.folder_id)
    bravo = repository.create_folder(name="Bravo", parent_id=parent.folder_id)

    first = repository.list_children(parent_id=parent.folder_id, limit=2, offset=0)
    exact_end = repository.list_children(parent_id=parent.folder_id, limit=2, offset=1)
    empty = repository.list_children(parent_id=parent.folder_id, limit=2, offset=3)

    assert first.folders == (alpha, bravo)
    assert first.total_folders == 3
    assert first.total_notes == 0
    assert first.next_offset == 2
    assert exact_end.folders == (bravo, zulu)
    assert exact_end.next_offset is None
    assert empty.folders == ()
    assert empty.total_folders == 3
    assert empty.next_offset is None


@pytest.mark.parametrize(
    ("limit", "offset"), [(0, 0), (501, 0), (1, -1), (True, 0), (1, False)]
)
def test_list_children_rejects_invalid_bounds(
    repository: LocalNoteFolderRepository, limit: int, offset: int
) -> None:
    with pytest.raises(FolderValidationError):
        repository.list_children(parent_id=None, limit=limit, offset=offset)


@pytest.mark.parametrize("note_limit", [0, 1001, True])
def test_load_tree_batch_rejects_invalid_note_limit(
    repository: LocalNoteFolderRepository, note_limit: int
) -> None:
    with pytest.raises(FolderValidationError):
        repository.load_tree_batch(expanded_folder_ids=(), note_limit=note_limit)


@pytest.mark.parametrize("read_method", ["list", "tree"])
def test_multi_statement_reads_use_one_owned_snapshot(
    repository: LocalNoteFolderRepository, read_method: str
) -> None:
    repository.create_folder(name="Root", parent_id=None)
    statements: list[str] = []
    connection = repository.db.get_connection()
    connection.set_trace_callback(statements.append)

    if read_method == "list":
        repository.list_children(parent_id=None, limit=50, offset=0)
    else:
        repository.load_tree_batch(expanded_folder_ids=(), note_limit=50)

    connection.set_trace_callback(None)
    transaction_statements = [
        statement.strip().upper()
        for statement in statements
        if statement.strip().upper() in {"BEGIN", "COMMIT"}
    ]
    assert transaction_statements == ["BEGIN", "COMMIT"]


def test_multi_statement_reads_do_not_finish_a_caller_owned_transaction(
    repository: LocalNoteFolderRepository,
) -> None:
    connection = repository.db.get_connection()
    connection.execute("BEGIN")

    repository.list_children(parent_id=None, limit=50, offset=0)
    repository.load_tree_batch(expanded_folder_ids=(), note_limit=50)

    assert connection.in_transaction
    connection.rollback()


def test_load_tree_batch_loads_roots_and_unfiled_notes(
    repository: LocalNoteFolderRepository,
) -> None:
    work = repository.create_folder(name="Work", parent_id=None)
    repository.create_folder(name="Personal", parent_id=None)
    unfiled = repository.db.add_note("Unfiled", "Body")
    inactive = repository.db.add_note("Inactive owner", "Body")
    filed = repository.db.add_note("Filed", "Body")
    assert unfiled is not None and inactive is not None and filed is not None
    _attach_membership(
        repository,
        folder_id=work.folder_id,
        note_id=inactive,
        ownership="managed",
        owner_id="restored-device",
        owner_active=False,
    )
    _attach_membership(repository, folder_id=work.folder_id, note_id=filed)

    page = repository.load_tree_batch(expanded_folder_ids=(), note_limit=50)

    assert [folder.name for folder in page.folders] == ["Personal", "Work"]
    assert page.memberships == ()
    assert [note["id"] for note in page.notes] == [inactive, unfiled]
    assert page.total_folders == 2
    assert page.total_notes == 2
    assert page.next_offset is None


def test_load_tree_batch_bulk_loads_expanded_folders_and_inactive_owner_rows(
    repository: LocalNoteFolderRepository,
) -> None:
    work = repository.create_folder(name="Work", parent_id=None)
    plans = repository.create_folder(name="Plans", parent_id=work.folder_id)
    later = repository.create_folder(name="Later", parent_id=plans.folder_id)
    work_note = repository.db.add_note("Work note", "W")
    plans_note = repository.db.add_note("Plans note", "P")
    assert work_note is not None and plans_note is not None
    active_id = _attach_membership(
        repository, folder_id=work.folder_id, note_id=work_note
    )
    inactive_id = _attach_membership(
        repository,
        folder_id=plans.folder_id,
        note_id=plans_note,
        ownership="managed",
        owner_id="missing-owner",
        owner_active=False,
    )
    statements: list[str] = []
    repository.db.get_connection().set_trace_callback(statements.append)

    page = repository.load_tree_batch(
        expanded_folder_ids=(plans.folder_id, work.folder_id, plans.folder_id),
        note_limit=50,
    )

    repository.db.get_connection().set_trace_callback(None)
    assert page.folders == (plans, later)
    assert [row.membership_id for row in page.memberships] == [active_id, inactive_id]
    assert [note["id"] for note in page.notes] == [plans_note, work_note]
    selects = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT")
    ]
    assert sum("FROM note_folders" in statement for statement in selects) == 1
    assert (
        sum("FROM note_folder_memberships" in statement for statement in selects) == 1
    )
    assert sum("FROM notes AS n" in statement for statement in selects) == 1


def test_load_tree_batch_limits_notes_and_reports_next_offset(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    for title in ("A", "B", "C"):
        note_id = repository.db.add_note(title, title)
        assert note_id is not None
        _attach_membership(repository, folder_id=folder.folder_id, note_id=note_id)

    page = repository.load_tree_batch(
        expanded_folder_ids=(folder.folder_id,), note_limit=2
    )

    assert len(page.notes) == 2
    assert page.total_notes == 3
    assert page.next_offset == 2


@pytest.mark.parametrize("name", ["", "..", "bad/name", "bad\\name", "\x00"])
def test_create_rejects_malformed_names(
    repository: LocalNoteFolderRepository, name: str
) -> None:
    with pytest.raises(FolderValidationError):
        repository.create_folder(name=name, parent_id=None)


def test_create_rejects_malformed_stored_parent_paths(
    repository: LocalNoteFolderRepository,
) -> None:
    parent = repository.create_folder(name="Parent", parent_id=None)
    repository.db.get_connection().execute(
        "UPDATE note_folders SET path = ?, normalized_path = ? WHERE id = ?",
        ("relative", "relative", parent.folder_id),
    )
    repository.db.get_connection().commit()

    with pytest.raises(FolderValidationError):
        repository.create_folder(name="Child", parent_id=parent.folder_id)


def test_rename_updates_complete_subtree_once_and_preserves_ids(
    repository: LocalNoteFolderRepository,
) -> None:
    work = repository.create_folder(name="Work", parent_id=None)
    plans = repository.create_folder(name="Plans", parent_id=work.folder_id)
    later = repository.create_folder(name="Later", parent_id=plans.folder_id)

    result = repository.rename_folder(
        work.folder_id, name="Projects", expected_version=work.version
    )

    renamed = repository.get_folder(work.folder_id)
    renamed_plans = repository.get_folder(plans.folder_id)
    renamed_later = repository.get_folder(later.folder_id)
    assert (
        renamed is not None
        and renamed_plans is not None
        and renamed_later is not None
    )
    assert result.folder == renamed
    assert result.affected_folder_ids == (
        work.folder_id,
        plans.folder_id,
        later.folder_id,
    )
    assert (
        renamed.folder_id,
        renamed.path,
        renamed.normalized_path,
        renamed.version,
    ) == (work.folder_id, "/Projects", "/projects", 2)
    assert (
        renamed_plans.folder_id,
        renamed_plans.path,
        renamed_plans.normalized_path,
        renamed_plans.version,
    ) == (plans.folder_id, "/Projects/Plans", "/projects/plans", 2)
    assert (
        renamed_later.path,
        renamed_later.normalized_path,
        renamed_later.version,
    ) == ("/Projects/Plans/Later", "/projects/plans/later", 2)
    modified_values = repository.db.get_connection().execute(
        "SELECT DISTINCT modified_at FROM note_folders WHERE id IN (?, ?, ?)",
        (work.folder_id, plans.folder_id, later.folder_id),
    ).fetchall()
    assert len(modified_values) == 1


def test_move_beneath_descendant_is_typed_and_atomic(
    repository: LocalNoteFolderRepository,
) -> None:
    work = repository.create_folder(name="Work", parent_id=None)
    plans = repository.create_folder(name="Plans", parent_id=work.folder_id)
    before = _folder_rows(repository)

    with pytest.raises(FolderValidationError):
        repository.move_folder(
            work.folder_id, parent_id=plans.folder_id, expected_version=work.version
        )

    assert _folder_rows(repository) == before


def test_move_destination_collision_is_typed_and_atomic(
    repository: LocalNoteFolderRepository,
) -> None:
    source = repository.create_folder(name="Source", parent_id=None)
    repository.create_folder(name="Child", parent_id=source.folder_id)
    destination = repository.create_folder(name="Destination", parent_id=None)
    repository.create_folder(name="Source", parent_id=destination.folder_id)
    before = _folder_rows(repository)
    statements: list[str] = []
    repository.db.get_connection().set_trace_callback(statements.append)

    with pytest.raises(FolderCollisionError):
        repository.move_folder(
            source.folder_id,
            parent_id=destination.folder_id,
            expected_version=source.version,
        )

    repository.db.get_connection().set_trace_callback(None)
    assert _folder_rows(repository) == before
    assert not any(
        statement.lstrip().upper().startswith("UPDATE NOTE_FOLDERS")
        for statement in statements
    )


def test_collision_preflight_uses_bounded_chunks_and_finds_a_later_collision(
    repository: LocalNoteFolderRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repository.create_folder(name="Root", parent_id=None)
    repository.create_folder(name="Alpha", parent_id=root.folder_id)
    repository.create_folder(name="Bravo", parent_id=root.folder_id)
    repository.create_folder(name="Zulu", parent_id=root.folder_id)
    now = _timestamp()
    with repository.db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO note_folders(
                id, parent_id, name, normalized_name, path,
                normalized_path, version, deleted, created_at, modified_at
            ) VALUES (?, NULL, ?, ?, ?, ?, 1, 0, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                "Zulu",
                "zulu",
                "/Renamed/Zulu",
                "/renamed/zulu",
                now,
                now,
            ),
        )
    before = _folder_rows(repository)
    monkeypatch.setattr(
        folder_repository_module,
        "_COLLISION_PREFLIGHT_CHUNK_SIZE",
        2,
        raising=False,
    )
    statements: list[str] = []
    connection = repository.db.get_connection()
    connection.set_trace_callback(statements.append)

    with pytest.raises(FolderCollisionError):
        repository.rename_folder(
            root.folder_id, name="Renamed", expected_version=root.version
        )

    connection.set_trace_callback(None)
    preflight_selects = [
        statement
        for statement in statements
        if "SELECT id, normalized_path FROM note_folders" in statement
        and "normalized_path IN" in statement
    ]
    assert len(preflight_selects) == 2
    assert _folder_rows(repository) == before


def test_stale_rename_is_typed_and_atomic(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Work", parent_id=None)
    before = _folder_rows(repository)

    with pytest.raises(FolderConflictError):
        repository.rename_folder(
            folder.folder_id, name="Projects", expected_version=folder.version + 1
        )

    assert _folder_rows(repository) == before


def test_move_rejects_missing_inactive_and_malformed_destination_parent(
    repository: LocalNoteFolderRepository,
) -> None:
    source = repository.create_folder(name="Source", parent_id=None)
    inactive = repository.create_folder(name="Inactive", parent_id=None)
    repository.soft_delete_folder(inactive.folder_id, expected_version=inactive.version)
    before = _folder_rows(repository)

    for parent_id in ("missing", inactive.folder_id, source.folder_id, ""):
        with pytest.raises(FolderValidationError):
            repository.move_folder(
                source.folder_id,
                parent_id=parent_id,
                expected_version=source.version,
            )

    assert _folder_rows(repository) == before


def test_soft_delete_and_restore_subtree_preserve_memberships_and_note_row(
    repository: LocalNoteFolderRepository,
) -> None:
    work = repository.create_folder(name="Work", parent_id=None)
    plans = repository.create_folder(name="Plans", parent_id=work.folder_id)
    note_id = repository.db.add_note("Plan", "Body")
    assert note_id is not None
    membership_id = _attach_membership(
        repository, folder_id=plans.folder_id, note_id=note_id
    )
    note_before = tuple(
        repository.db.get_connection()
        .execute("SELECT deleted, version FROM notes WHERE id = ?", (note_id,))
        .fetchone()
    )

    deleted = repository.soft_delete_folder(
        work.folder_id, expected_version=work.version
    )

    assert deleted.affected_folder_ids == (work.folder_id, plans.folder_id)
    assert repository.get_folder(work.folder_id) is None
    assert repository.get_folder(plans.folder_id) is None
    deleted_work = repository.get_folder(work.folder_id, include_deleted=True)
    deleted_plans = repository.get_folder(plans.folder_id, include_deleted=True)
    assert deleted_work is not None and deleted_plans is not None
    assert deleted_work.deleted and deleted_plans.deleted
    assert (deleted_work.version, deleted_plans.version) == (2, 2)
    assert repository.list_children(parent_id=None, limit=50, offset=0).folders == ()

    restored = repository.restore_folder(
        work.folder_id, expected_version=deleted_work.version
    )

    restored_work = repository.get_folder(work.folder_id)
    restored_plans = repository.get_folder(plans.folder_id)
    assert restored.affected_folder_ids == (work.folder_id, plans.folder_id)
    assert restored_work is not None and restored_plans is not None
    assert (restored_work.version, restored_plans.version) == (3, 3)
    membership = repository.db.get_connection().execute(
        "SELECT id, deleted, version FROM note_folder_memberships WHERE id = ?",
        (membership_id,),
    ).fetchone()
    assert tuple(membership) == (membership_id, 0, 1)
    note_after = tuple(
        repository.db.get_connection()
        .execute("SELECT deleted, version FROM notes WHERE id = ?", (note_id,))
        .fetchone()
    )
    assert note_after == note_before


def test_restore_collision_is_typed_and_atomic(
    repository: LocalNoteFolderRepository,
) -> None:
    archived = repository.create_folder(name="Work", parent_id=None)
    child = repository.create_folder(name="Plans", parent_id=archived.folder_id)
    repository.soft_delete_folder(archived.folder_id, expected_version=archived.version)
    active = repository.create_folder(name="work", parent_id=None)
    before = _folder_rows(repository)

    with pytest.raises(FolderCollisionError):
        repository.restore_folder(archived.folder_id, expected_version=2)

    assert _folder_rows(repository) == before
    assert repository.get_folder(active.folder_id) == active
    restored_child = repository.get_folder(child.folder_id, include_deleted=True)
    assert restored_child is not None and restored_child.deleted


def test_restore_rejects_inactive_external_parent_atomically(
    repository: LocalNoteFolderRepository,
) -> None:
    parent = repository.create_folder(name="Parent", parent_id=None)
    child = repository.create_folder(name="Child", parent_id=parent.folder_id)
    repository.soft_delete_folder(parent.folder_id, expected_version=parent.version)
    before = _folder_rows(repository)

    with pytest.raises(FolderValidationError):
        repository.restore_folder(child.folder_id, expected_version=2)

    assert _folder_rows(repository) == before


def test_restore_rejects_missing_external_parent_atomically(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Orphan", parent_id=None)
    repository.soft_delete_folder(folder.folder_id, expected_version=folder.version)
    connection = repository.db.get_connection()
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute(
        "UPDATE note_folders SET parent_id = ? WHERE id = ?",
        ("missing-parent", folder.folder_id),
    )
    connection.commit()
    connection.execute("PRAGMA foreign_keys = ON")
    before = _folder_rows(repository)

    with pytest.raises(FolderValidationError):
        repository.restore_folder(folder.folder_id, expected_version=2)

    assert _folder_rows(repository) == before


@pytest.mark.parametrize(
    ("operation", "folder_state"),
    [
        ("rename", "missing"),
        ("move", "missing"),
        ("delete", "missing"),
        ("restore", "missing"),
        ("restore", "active"),
    ],
)
def test_mutations_reject_missing_or_invalid_target_state(
    repository: LocalNoteFolderRepository, operation: str, folder_state: str
) -> None:
    active = repository.create_folder(name="Active", parent_id=None)
    folder_id = active.folder_id if folder_state == "active" else "missing"

    with pytest.raises(FolderValidationError):
        if operation == "rename":
            repository.rename_folder(folder_id, name="New", expected_version=1)
        elif operation == "move":
            repository.move_folder(folder_id, parent_id=None, expected_version=1)
        elif operation == "delete":
            repository.soft_delete_folder(folder_id, expected_version=1)
        else:
            repository.restore_folder(folder_id, expected_version=1)


@pytest.mark.parametrize("operation", ["rename", "move", "delete", "restore"])
def test_multirow_mutation_rolls_back_to_its_own_boundary_when_outer_catches(
    repository: LocalNoteFolderRepository, operation: str
) -> None:
    root = repository.create_folder(name="Root", parent_id=None)
    repository.create_folder(name="Child", parent_id=root.folder_id)
    destination = None
    expected_version = root.version
    if operation == "move":
        destination = repository.create_folder(name="Destination", parent_id=None)
    elif operation == "restore":
        repository.soft_delete_folder(
            root.folder_id, expected_version=expected_version
        )
        expected_version += 1

    connection = repository.db.get_connection()
    connection.execute(
        """
        CREATE TRIGGER fail_child_folder_update
        BEFORE UPDATE ON note_folders
        WHEN OLD.name = 'Child'
        BEGIN
          SELECT RAISE(ABORT, 'forced child folder failure');
        END
        """
    )
    connection.commit()
    before = _folder_rows(repository)

    with repository.db.transaction():
        with pytest.raises(FolderValidationError):
            if operation == "rename":
                repository.rename_folder(
                    root.folder_id,
                    name="Renamed",
                    expected_version=expected_version,
                )
            elif operation == "move":
                assert destination is not None
                repository.move_folder(
                    root.folder_id,
                    parent_id=destination.folder_id,
                    expected_version=expected_version,
                )
            elif operation == "delete":
                repository.soft_delete_folder(
                    root.folder_id, expected_version=expected_version
                )
            else:
                repository.restore_folder(
                    root.folder_id, expected_version=expected_version
                )

    assert _folder_rows(repository) == before


def test_restore_rebases_deleted_subtree_after_external_parent_rename(
    repository: LocalNoteFolderRepository,
) -> None:
    work = repository.create_folder(name="Work", parent_id=None)
    plans = repository.create_folder(name="Plans", parent_id=work.folder_id)
    later = repository.create_folder(name="Later", parent_id=plans.folder_id)
    repository.soft_delete_folder(plans.folder_id, expected_version=plans.version)
    repository.rename_folder(
        work.folder_id, name="Projects", expected_version=work.version
    )

    restored = repository.restore_folder(plans.folder_id, expected_version=2)

    restored_plans = repository.get_folder(plans.folder_id)
    restored_later = repository.get_folder(later.folder_id)
    assert restored.affected_folder_ids == (plans.folder_id, later.folder_id)
    assert restored_plans is not None and restored_later is not None
    assert (
        restored_plans.parent_id,
        restored_plans.path,
        restored_plans.normalized_path,
        restored_plans.version,
    ) == (work.folder_id, "/Projects/Plans", "/projects/plans", 3)
    assert (
        restored_later.parent_id,
        restored_later.path,
        restored_later.normalized_path,
        restored_later.version,
    ) == (
        plans.folder_id,
        "/Projects/Plans/Later",
        "/projects/plans/later",
        3,
    )
    connection = repository.db.get_connection()
    connection.execute(
        "UPDATE note_folders SET path = ?, normalized_path = ? WHERE id = ?",
        ("/Detached/Plans", "/detached/plans", plans.folder_id),
    )
    connection.execute(
        "UPDATE note_folders SET path = ?, normalized_path = ? WHERE id = ?",
        (
            "/Detached/Plans/Later",
            "/detached/plans/later",
            later.folder_id,
        ),
    )
    connection.commit()
    before_cycle_attempt = _folder_rows(repository)

    with pytest.raises(FolderValidationError):
        repository.move_folder(
            work.folder_id,
            parent_id=plans.folder_id,
            expected_version=2,
        )

    assert _folder_rows(repository) == before_cycle_attempt


def test_restore_only_revives_descendants_from_the_target_delete_operation(
    repository: LocalNoteFolderRepository,
) -> None:
    work = repository.create_folder(name="Work", parent_id=None)
    plans = repository.create_folder(name="Plans", parent_id=work.folder_id)
    current = repository.create_folder(name="Current", parent_id=work.folder_id)
    detail = repository.create_folder(name="Detail", parent_id=current.folder_id)
    repository.soft_delete_folder(plans.folder_id, expected_version=plans.version)
    repository.soft_delete_folder(work.folder_id, expected_version=work.version)

    result = repository.restore_folder(work.folder_id, expected_version=2)

    restored_work = repository.get_folder(work.folder_id)
    restored_current = repository.get_folder(current.folder_id)
    restored_detail = repository.get_folder(detail.folder_id)
    still_deleted_plans = repository.get_folder(
        plans.folder_id, include_deleted=True
    )
    assert result.affected_folder_ids == (
        work.folder_id,
        current.folder_id,
        detail.folder_id,
    )
    assert restored_work is not None
    assert restored_current is not None
    assert restored_detail is not None
    assert still_deleted_plans is not None and still_deleted_plans.deleted
    assert still_deleted_plans.version == 2


@pytest.mark.parametrize("operation", ["rename", "move", "delete", "restore"])
def test_multirow_mutation_treats_ignored_child_update_as_atomic_conflict(
    repository: LocalNoteFolderRepository, operation: str
) -> None:
    root = repository.create_folder(name="Root", parent_id=None)
    repository.create_folder(name="Child", parent_id=root.folder_id)
    destination = None
    expected_version = root.version
    if operation == "move":
        destination = repository.create_folder(name="Destination", parent_id=None)
    elif operation == "restore":
        repository.soft_delete_folder(root.folder_id, expected_version=root.version)
        expected_version = 2
    connection = repository.db.get_connection()
    connection.execute(
        """
        CREATE TRIGGER ignore_child_folder_update
        BEFORE UPDATE ON note_folders
        WHEN OLD.name = 'Child'
        BEGIN
          SELECT RAISE(IGNORE);
        END
        """
    )
    connection.commit()
    before = _folder_rows(repository)

    with pytest.raises(FolderConflictError):
        if operation == "rename":
            repository.rename_folder(
                root.folder_id, name="Renamed", expected_version=expected_version
            )
        elif operation == "move":
            assert destination is not None
            repository.move_folder(
                root.folder_id,
                parent_id=destination.folder_id,
                expected_version=expected_version,
            )
        elif operation == "delete":
            repository.soft_delete_folder(
                root.folder_id, expected_version=expected_version
            )
        else:
            repository.restore_folder(
                root.folder_id, expected_version=expected_version
            )

    assert _folder_rows(repository) == before


@pytest.mark.parametrize(
    "error_code",
    [
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
        getattr(sqlite3, "SQLITE_BUSY_SNAPSHOT", sqlite3.SQLITE_BUSY),
    ],
)
def test_mutation_database_contention_is_a_stable_typed_conflict(
    error_code: int,
) -> None:
    error = sqlite3.OperationalError("database-specific contention detail")
    error.sqlite_errorcode = error_code

    with pytest.raises(FolderConflictError) as caught:
        _raise_mutation_operational_error(error)

    assert str(caught.value) == "Folder changed during mutation."
