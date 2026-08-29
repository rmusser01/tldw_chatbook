"""Behavior tests for the local Database Note folder repository."""

from __future__ import annotations

import inspect
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

import tldw_chatbook.Notes.note_folder_repository as folder_repository_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    FolderCollisionError,
    FolderConflictError,
    FolderValidationError,
    NoteTreeMutationContext,
    NoteTreePathStep,
)
from tldw_chatbook.Notes.note_folder_repository import (
    LocalNoteFolderRepository,
    _raise_membership_integrity_error,
    _raise_mutation_operational_error,
    _raise_wrapped_repository_error,
)


class _FolderIdSubclass(str):
    pass


@pytest.fixture
def repository(tmp_path) -> Iterator[LocalNoteFolderRepository]:
    """Return a repository backed by the real ChaChaNotes SQLite database."""
    db = CharactersRAGDB(tmp_path / "folders.db", client_id="folder-tests")
    yield LocalNoteFolderRepository(db)
    db.close_connection()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace(
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
    membership_id: str | None = None,
) -> str:
    membership_id = membership_id or str(uuid.uuid4())
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


def _insert_note(
    repository: LocalNoteFolderRepository,
    *,
    note_id: str,
    title: str,
    content: str = "",
) -> None:
    with repository.db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes(id, title, content, client_id) VALUES (?, ?, ?, ?)",
            (note_id, title, content, "folder-paging-test"),
        )


def _folder_rows(
    repository: LocalNoteFolderRepository,
) -> tuple[tuple[object, ...], ...]:
    rows = repository.db.get_connection().execute(
        "SELECT id, parent_id, name, normalized_name, path, normalized_path, "
        "version, deleted, modified_at FROM note_folders ORDER BY id"
    ).fetchall()
    return tuple(tuple(row) for row in rows)


def _membership_rows(
    repository: LocalNoteFolderRepository,
) -> tuple[tuple[object, ...], ...]:
    rows = repository.db.get_connection().execute(
        "SELECT id, folder_id, note_id, ownership, owner_id, owner_active, "
        "version, deleted, modified_at FROM note_folder_memberships ORDER BY id"
    ).fetchall()
    return tuple(tuple(row) for row in rows)


def _note_row(repository: LocalNoteFolderRepository, note_id: str) -> tuple[object, ...]:
    row = repository.db.get_connection().execute(
        "SELECT id, title, content, deleted, version FROM notes WHERE id = ?",
        (note_id,),
    ).fetchone()
    assert row is not None
    return tuple(row)


def test_create_and_list_nested_folders(repository: LocalNoteFolderRepository) -> None:
    work = repository.create_folder(name="Work", parent_id=None)
    plans = repository.create_folder(name="Plans", parent_id=work.folder_id)

    page = repository.list_children(parent_id=work.folder_id, limit=50, offset=0)

    assert page.folders == (plans,)
    assert plans.path == "/Work/Plans"
    assert plans.normalized_path == "/work/plans"


@pytest.mark.parametrize(
    "folder_id",
    [
        "00000000-0000-5000-8000-000000000001",
        "a._:-Z9",
        "a" + ("b" * 255),
    ],
)
def test_create_folder_accepts_a_bounded_safe_deterministic_caller_id(
    repository: LocalNoteFolderRepository, folder_id: str
) -> None:
    folder = repository.create_folder(
        name="Imported", parent_id=None, folder_id=folder_id
    )

    assert folder.folder_id == folder_id


@pytest.mark.parametrize(
    "folder_id",
    [
        "",
        7,
        False,
        " leading",
        ".leading",
        "-leading",
        ":leading",
        "a b",
        "a/b",
        "a\\b",
        "a\x00b",
        "a\x01b",
        "a\tb",
        "éclair",
        "a" + ("b" * 256),
        _FolderIdSubclass("valid-subclass"),
    ],
)
def test_create_folder_rejects_malformed_caller_id_without_mutation(
    repository: LocalNoteFolderRepository, folder_id: object
) -> None:
    before = _folder_rows(repository)

    with pytest.raises(FolderValidationError) as caught:
        repository.create_folder(
            name="Private input",
            parent_id=None,
            folder_id=folder_id,  # type: ignore[arg-type]
        )

    assert _folder_rows(repository) == before
    assert "Private input" not in str(caught.value)


def test_create_folder_validates_caller_id_before_opening_a_transaction(
    repository: LocalNoteFolderRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_transaction():
        raise AssertionError("invalid caller ID reached the transaction boundary")

    monkeypatch.setattr(repository.db, "transaction", forbidden_transaction)

    with pytest.raises(FolderValidationError):
        repository.create_folder(
            name="Private input",
            parent_id=None,
            folder_id="invalid/path",
        )


def test_create_folder_without_caller_id_retains_uuid_behavior(
    repository: LocalNoteFolderRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = uuid.UUID("00000000-0000-4000-8000-000000000009")
    monkeypatch.setattr(folder_repository_module.uuid, "uuid4", lambda: generated)

    folder = repository.create_folder(name="Generated", parent_id=None)

    assert folder.folder_id == str(generated)


def test_get_folder_by_path_uses_normalized_exact_segments(
    repository: LocalNoteFolderRepository,
) -> None:
    cafe = repository.create_folder(name="Café", parent_id=None)
    ideas = repository.create_folder(name="Ideas", parent_id=cafe.folder_id)
    repository.create_folder(name="Ideas Archive", parent_id=cafe.folder_id)

    assert repository.get_folder_by_path(("Cafe\u0301", "ideas")) == ideas
    assert repository.get_folder_by_path(("Cafe\u0301",)) == cafe
    assert repository.get_folder_by_path(("Cafe\u0301", "idea")) is None
    assert repository.get_folder_by_path(("Cafe\u0301", "ideas", "later")) is None


def test_get_folder_by_path_excludes_deleted_folders(
    repository: LocalNoteFolderRepository,
) -> None:
    deleted = repository.create_folder(name="Deleted", parent_id=None)
    repository.db.get_connection().execute(
        "UPDATE note_folders SET deleted = 1 WHERE id = ?", (deleted.folder_id,)
    )
    repository.db.get_connection().commit()

    assert repository.get_folder_by_path(("deleted",)) is None


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


def test_commit_time_database_contention_is_a_stable_conflict(tmp_path) -> None:
    path = tmp_path / "commit-contention.db"
    db = CharactersRAGDB(path, client_id="commit-contention")
    repository = LocalNoteFolderRepository(db)
    connection = db.get_connection()
    assert connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0] == "delete"
    connection.execute("PRAGMA busy_timeout = 1")
    reader = sqlite3.connect(path)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM note_folders").fetchone()

        with pytest.raises(FolderConflictError, match="Folder changed during mutation"):
            repository.create_folder(name="Blocked", parent_id=None)
    finally:
        reader.rollback()
        reader.close()
        db.close_connection()

    reopened = CharactersRAGDB(path, client_id="contention-check")
    try:
        count = reopened.get_connection().execute(
            "SELECT COUNT(*) FROM note_folders WHERE name = ?", ("Blocked",)
        ).fetchone()[0]
        assert count == 0
    finally:
        reopened.close_connection()


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


def test_get_folder_uses_shared_transaction_wrapper(
    repository: LocalNoteFolderRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    original_transaction = repository.db.transaction
    transaction_calls = 0

    @contextmanager
    def recording_transaction() -> Iterator[sqlite3.Cursor]:
        nonlocal transaction_calls
        transaction_calls += 1
        with original_transaction() as cursor:
            yield cursor

    monkeypatch.setattr(repository.db, "transaction", recording_transaction)

    assert repository.get_folder(folder.folder_id) == folder
    assert transaction_calls == 1


@pytest.mark.parametrize("method_name", ["create_folder", "get_folder"])
def test_repository_methods_do_not_execute_sql_on_raw_connection(
    method_name: str,
) -> None:
    method = getattr(LocalNoteFolderRepository, method_name)

    assert ".get_connection()" not in inspect.getsource(method)


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


@pytest.mark.parametrize("read_method", ["list", "tree", "memberships"])
def test_multi_statement_reads_use_one_owned_snapshot(
    repository: LocalNoteFolderRepository, read_method: str
) -> None:
    repository.create_folder(name="Root", parent_id=None)
    statements: list[str] = []
    connection = repository.db.get_connection()
    connection.set_trace_callback(statements.append)

    if read_method == "list":
        repository.list_children(parent_id=None, limit=50, offset=0)
    elif read_method == "tree":
        repository.load_tree_batch(expanded_folder_ids=(), note_limit=50)
    else:
        repository.list_memberships(
            note_ids=tuple(f"note-{index:04d}" for index in range(801))
        )

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
    repository.list_memberships(
        note_ids=tuple(f"note-{index:04d}" for index in range(801))
    )

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


def test_load_tree_batch_can_skip_exhausted_note_query(
    repository: LocalNoteFolderRepository,
) -> None:
    repository.create_folder(name="Work", parent_id=None)
    note_id = repository.db.add_note("Unfiled", "Body")
    assert note_id is not None
    statements: list[str] = []
    connection = repository.db.get_connection()
    connection.set_trace_callback(statements.append)

    page = repository.load_tree_batch(
        expanded_folder_ids=(), note_limit=50, load_notes=False
    )

    connection.set_trace_callback(None)
    assert page.notes == ()
    assert page.total_notes == 0
    assert page.next_offset is None
    assert not any("AS CANDIDATE ORDER BY TITLE" in sql.upper() for sql in statements)


def test_root_batch_reports_managed_descendants_without_expanding_them(
    repository: LocalNoteFolderRepository,
) -> None:
    root = repository.create_folder(name="Work", parent_id=None)
    child = repository.create_folder(name="Project", parent_id=root.folder_id)
    note_id = repository.db.add_note("Plan", "Body")
    assert note_id is not None
    _attach_membership(
        repository,
        folder_id=child.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="sync-root",
    )

    page = repository.load_tree_batch(expanded_folder_ids=(), note_limit=50)

    assert page.managed_folder_ids == (root.folder_id,)
    assert page.inactive_managed_folder_ids == ()


def test_managed_folder_lookup_walks_from_memberships_to_ancestors(
    repository: LocalNoteFolderRepository,
) -> None:
    root = repository.create_folder(name="Work", parent_id=None)
    child = repository.create_folder(name="Project", parent_id=root.folder_id)
    note_id = repository.db.add_note("Plan", "Body")
    assert note_id is not None
    _attach_membership(
        repository,
        folder_id=child.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="sync-root",
    )
    statements: list[str] = []
    connection = repository.db.get_connection()
    connection.set_trace_callback(statements.append)

    repository.load_tree_batch(expanded_folder_ids=(), note_limit=50)

    connection.set_trace_callback(None)
    managed_query = next(
        statement for statement in statements if "managed_ancestors" in statement
    )
    assert "SELECT DISTINCT membership.folder_id" in managed_query
    assert "JOIN managed_ancestors" in managed_query
    assert "JOIN subtree" not in managed_query


def test_search_batch_loads_matching_note_placements_and_all_ancestors(
    repository: LocalNoteFolderRepository,
) -> None:
    root = repository.create_folder(name="Work", parent_id=None)
    child = repository.create_folder(name="Project", parent_id=root.folder_id)
    note_id = repository.db.add_note("Hidden garden plan", "Body")
    assert note_id is not None
    membership_id = _attach_membership(
        repository,
        folder_id=child.folder_id,
        note_id=note_id,
    )

    page = repository.load_tree_search(note_ids=(note_id,))

    assert {folder.folder_id for folder in page.folders} == {
        root.folder_id,
        child.folder_id,
    }
    assert [membership.membership_id for membership in page.memberships] == [
        membership_id
    ]
    assert [note["id"] for note in page.notes] == [note_id]


def test_search_batch_loads_collapsed_placements_matching_folder_path(
    repository: LocalNoteFolderRepository,
) -> None:
    root = repository.create_folder(name="Work", parent_id=None)
    child = repository.create_folder(name="Project", parent_id=root.folder_id)
    note_id = repository.db.add_note("Unrelated title", "No content match")
    assert note_id is not None
    membership_id = _attach_membership(
        repository,
        folder_id=child.folder_id,
        note_id=note_id,
    )

    page = repository.load_tree_search(note_ids=(), folder_query="work / project")

    assert {folder.folder_id for folder in page.folders} == {
        root.folder_id,
        child.folder_id,
    }
    assert [membership.membership_id for membership in page.memberships] == [
        membership_id
    ]
    assert [note["id"] for note in page.notes] == [note_id]


def test_managed_subtree_cannot_be_renamed_through_repository_boundary(
    repository: LocalNoteFolderRepository,
) -> None:
    root = repository.create_folder(name="Work", parent_id=None)
    child = repository.create_folder(name="Project", parent_id=root.folder_id)
    note_id = repository.db.add_note("Plan", "Body")
    assert note_id is not None
    _attach_membership(
        repository,
        folder_id=child.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="sync-root",
    )

    with pytest.raises(RuntimeError, match="managed by sync"):
        repository.rename_folder(
            root.folder_id,
            name="Renamed",
            expected_version=root.version,
        )

    assert repository.get_folder(root.folder_id) == root


def test_managed_subtree_rejects_create_move_and_delete_at_repository_boundary(
    repository: LocalNoteFolderRepository,
) -> None:
    managed_root = repository.create_folder(name="Managed", parent_id=None)
    managed_child = repository.create_folder(
        name="Project", parent_id=managed_root.folder_id
    )
    manual = repository.create_folder(name="Manual", parent_id=None)
    note_id = repository.db.add_note("Plan", "Body")
    assert note_id is not None
    _attach_membership(
        repository,
        folder_id=managed_child.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="sync-root",
    )

    with pytest.raises(RuntimeError, match="managed by sync"):
        repository.create_folder(name="Blocked", parent_id=managed_root.folder_id)
    with pytest.raises(RuntimeError, match="managed by sync"):
        repository.move_folder(
            manual.folder_id,
            parent_id=managed_root.folder_id,
            expected_version=manual.version,
        )
    with pytest.raises(RuntimeError, match="managed by sync"):
        repository.move_folder(
            managed_root.folder_id,
            parent_id=None,
            expected_version=managed_root.version,
        )
    with pytest.raises(RuntimeError, match="managed by sync"):
        repository.soft_delete_folder(
            managed_root.folder_id,
            expected_version=managed_root.version,
        )

    assert repository.get_folder(manual.folder_id) == manual
    assert repository.get_folder(managed_root.folder_id) == managed_root


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
        if statement.lstrip().upper().startswith(("SELECT", "WITH"))
    ]
    # One additional constant-shape recursive query carries authoritative
    # managed-folder ownership through collapsed and paginated branches.
    assert sum("FROM note_folders" in statement for statement in selects) == 2
    assert sum("FROM note_folder_memberships" in statement for statement in selects) == 2
    # The membership query repeats the bounded note-page CTE so it never binds
    # every returned note ID and remains compatible with SQLite's 999-variable cap.
    assert sum("FROM notes AS n" in statement for statement in selects) == 2
    assert 3 <= len(selects) <= 4


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


def test_load_tree_batch_returns_second_note_page_with_only_page_memberships(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    note_ids: list[str] = []
    membership_ids: list[str] = []
    for title in ("A", "B", "C", "D", "E"):
        note_id = repository.db.add_note(title, title)
        assert note_id is not None
        note_ids.append(note_id)
        membership_ids.append(
            _attach_membership(
                repository, folder_id=folder.folder_id, note_id=note_id
            )
        )

    page = repository.load_tree_batch(
        expanded_folder_ids=(folder.folder_id,),
        note_limit=2,
        note_offset=2,
    )

    assert [note["id"] for note in page.notes] == note_ids[2:4]
    assert {row.membership_id for row in page.memberships} == set(membership_ids[2:4])
    assert page.total_notes == 5
    assert page.next_offset == 4


def test_load_tree_batch_pages_folders_independently_from_notes(
    repository: LocalNoteFolderRepository,
) -> None:
    folders = tuple(
        repository.create_folder(name=name, parent_id=None)
        for name in ("A", "B", "C")
    )

    first = repository.load_tree_batch(
        expanded_folder_ids=(), note_limit=10, folder_limit=2
    )
    second = repository.load_tree_batch(
        expanded_folder_ids=(),
        note_limit=10,
        folder_limit=2,
        folder_offset=2,
    )

    assert first.folders == folders[:2]
    assert first.total_folders == 3
    assert first.next_folder_offset == 2
    assert second.folders == folders[2:]
    assert second.total_folders == 3
    assert second.next_folder_offset is None


def test_list_children_exposes_folder_cursor_without_breaking_legacy_cursor(
    repository: LocalNoteFolderRepository,
) -> None:
    for name in ("A", "B", "C"):
        repository.create_folder(name=name, parent_id=None)

    page = repository.list_children(parent_id=None, limit=2, offset=0)

    assert page.next_offset == 2
    assert page.next_folder_offset == 2


def test_child_folder_pages_are_exact_ordered_and_parent_scoped(
    repository: LocalNoteFolderRepository,
) -> None:
    parent = repository.create_folder(name="Parent", parent_id=None, folder_id="parent")
    for index in reversed(range(45)):
        if index < 44:
            repository.create_folder(
                name=f"Root {index:02d}",
                parent_id=None,
                folder_id=f"root-{index:02d}",
            )
        repository.create_folder(
            name=f"Child {index:02d}",
            parent_id=parent.folder_id,
            folder_id=f"child-{index:02d}",
        )

    expected_roots = ["parent", *(f"root-{index:02d}" for index in range(44))]
    expected_children = [f"child-{index:02d}" for index in range(45)]
    for parent_id, expected_ids in (
        (None, expected_roots),
        (parent.folder_id, expected_children),
    ):
        total = len(expected_ids)
        for offset in (0, 20, 40, 60):
            page = repository.page_child_folders(
                parent_id=parent_id, limit=20, offset=offset
            )

            assert [folder.folder_id for folder in page.folders] == expected_ids[
                offset : offset + 20
            ]
            assert len(page.folders) <= 20
            assert page.total_folders == total
            assert page.start_offset == offset
            assert page.previous_offset == (
                None if offset == 0 else min(max(0, offset - 20), max(0, total - 20))
            )
            assert page.next_offset == (
                offset + len(page.folders)
                if offset + len(page.folders) < total
                else None
            )


def test_note_placement_pages_keep_duplicates_shadow_descendants_and_order(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Wild%_", parent_id=None, folder_id="target")
    descendant = repository.create_folder(
        name="Descendant", parent_id=folder.folder_id, folder_id="descendant"
    )
    wildcard_peer = repository.create_folder(
        name="WildXX", parent_id=None, folder_id="wildcard-peer"
    )
    false_descendant = repository.create_folder(
        name="Elsewhere",
        parent_id=wildcard_peer.folder_id,
        folder_id="false-descendant",
    )
    expected: list[tuple[str, str, str]] = []
    for index in reversed(range(25)):
        note_id = f"filed-note-{index:02d}"
        title = f"Title {index:02d}" if index % 2 else f"title {index:02d}"
        _insert_note(repository, note_id=note_id, title=title)
        membership_id = _attach_membership(
            repository, folder_id=folder.folder_id, note_id=note_id
        )
        expected.append((title, note_id, membership_id))

    duplicate_note_id = "duplicate-note"
    _insert_note(repository, note_id=duplicate_note_id, title="Duplicate")
    duplicate_memberships = (
        _attach_membership(
            repository, folder_id=folder.folder_id, note_id=duplicate_note_id
        ),
        _attach_membership(
            repository,
            folder_id=folder.folder_id,
            note_id=duplicate_note_id,
            ownership="managed",
            owner_id="duplicate-owner",
        ),
    )
    expected.extend(
        ("Duplicate", duplicate_note_id, membership_id)
        for membership_id in duplicate_memberships
    )

    shadowed_note_id = "shadowed-note"
    _insert_note(repository, note_id=shadowed_note_id, title="Shadowed")
    _attach_membership(
        repository,
        folder_id=folder.folder_id,
        note_id=shadowed_note_id,
        ownership="managed",
        owner_id="shadow-owner",
    )
    _attach_membership(
        repository,
        folder_id=descendant.folder_id,
        note_id=shadowed_note_id,
        ownership="managed",
        owner_id="shadow-owner",
    )

    inactive_child_note_id = "inactive-child-note"
    _insert_note(repository, note_id=inactive_child_note_id, title="Inactive child")
    surviving_ancestor_id = _attach_membership(
        repository,
        folder_id=folder.folder_id,
        note_id=inactive_child_note_id,
        ownership="managed",
        owner_id="inactive-child-owner",
    )
    _attach_membership(
        repository,
        folder_id=descendant.folder_id,
        note_id=inactive_child_note_id,
        ownership="managed",
        owner_id="inactive-child-owner",
        owner_active=False,
    )
    expected.append(("Inactive child", inactive_child_note_id, surviving_ancestor_id))

    wildcard_note_id = "wildcard-note"
    _insert_note(repository, note_id=wildcard_note_id, title="Wildcard")
    wildcard_membership_id = _attach_membership(
        repository,
        folder_id=folder.folder_id,
        note_id=wildcard_note_id,
        ownership="managed",
        owner_id="wildcard-owner",
    )
    _attach_membership(
        repository,
        folder_id=false_descendant.folder_id,
        note_id=wildcard_note_id,
        ownership="managed",
        owner_id="wildcard-owner",
    )
    expected.append(("Wildcard", wildcard_note_id, wildcard_membership_id))
    expected.sort(key=lambda item: (item[0].lower(), item[1], item[2]))

    observed: list[tuple[str, str, str]] = []
    for offset in (0, 20, 40):
        page = repository.page_note_placements(
            parent_id=folder.folder_id, limit=20, offset=offset
        )
        observed.extend(
            (
                str(placement.note["title"]),
                str(placement.note["id"]),
                placement.membership.membership_id,
            )
            for placement in page.placements
            if placement.membership is not None
        )
        assert len(page.placements) <= 20
        assert page.total_placements == len(expected)
        assert page.start_offset == offset
        assert page.previous_offset == (
            None
            if offset == 0
            else min(max(0, offset - 20), max(0, len(expected) - 20))
        )
        assert page.next_offset == (
            offset + len(page.placements)
            if offset + len(page.placements) < len(expected)
            else None
        )
        assert all(
            placement.folder_id == folder.folder_id
            and placement.membership is not None
            and placement.membership.folder_id == folder.folder_id
            for placement in page.placements
        )

    assert observed == expected
    assert [row[2] for row in observed if row[1] == duplicate_note_id] == sorted(
        duplicate_memberships
    )


def test_unfiled_placement_page_is_exact_ordered_and_synthetic(
    repository: LocalNoteFolderRepository,
) -> None:
    filed_folder = repository.create_folder(name="Filed", parent_id=None)
    expected: list[tuple[str, str]] = []
    for index in reversed(range(45)):
        note_id = f"unfiled-note-{index:02d}"
        title = f"Note {index:02d}" if index % 2 else f"note {index:02d}"
        _insert_note(repository, note_id=note_id, title=title)
        expected.append((title, note_id))
    filed_note_id = "filed-only-note"
    _insert_note(repository, note_id=filed_note_id, title="Filed only")
    _attach_membership(
        repository, folder_id=filed_folder.folder_id, note_id=filed_note_id
    )
    expected.sort(key=lambda item: (item[0].lower(), item[1]))

    observed: list[tuple[str, str]] = []
    for offset in (0, 20, 40, 60):
        page = repository.page_note_placements(parent_id=None, limit=20, offset=offset)
        observed.extend(
            (str(placement.note["title"]), str(placement.note["id"]))
            for placement in page.placements
        )
        assert len(page.placements) <= 20
        assert page.total_placements == 45
        assert page.start_offset == offset
        assert page.previous_offset == (
            None if offset == 0 else min(max(0, offset - 20), max(0, 45 - 20))
        )
        assert page.next_offset == (
            offset + len(page.placements)
            if offset + len(page.placements) < 45
            else None
        )
        assert all(
            placement.folder_id is None and placement.membership is None
            for placement in page.placements
        )

    assert observed == expected


@pytest.mark.parametrize("row_count", [5, 45])
@pytest.mark.parametrize("method_name", ["folders", "placements"])
def test_exact_page_query_count_is_constant(
    repository: LocalNoteFolderRepository,
    row_count: int,
    method_name: str,
) -> None:
    parent = repository.create_folder(name="Parent", parent_id=None)
    for index in range(row_count):
        if method_name == "folders":
            repository.create_folder(
                name=f"Child {index:03d}", parent_id=parent.folder_id
            )
        else:
            note_id = f"query-note-{index:03d}"
            _insert_note(repository, note_id=note_id, title=f"Note {index:03d}")
            _attach_membership(repository, folder_id=parent.folder_id, note_id=note_id)
    statements: list[str] = []
    connection = repository.db.get_connection()
    connection.set_trace_callback(statements.append)

    if method_name == "folders":
        repository.page_child_folders(parent_id=parent.folder_id, limit=20, offset=0)
    else:
        repository.page_note_placements(parent_id=parent.folder_id, limit=20, offset=0)

    connection.set_trace_callback(None)
    reads = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith(("SELECT", "WITH"))
    ]
    assert len(reads) == 2


def test_note_placement_suppression_plan_searches_child_membership_by_note_id(
    repository: LocalNoteFolderRepository,
) -> None:
    parent = repository.create_folder(name="Parent", parent_id=None)
    child = repository.create_folder(name="Child", parent_id=parent.folder_id)
    note_id = repository.db.add_note("Plan", "Body")
    assert note_id is not None
    _attach_membership(
        repository,
        folder_id=parent.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="shared-owner",
    )
    _attach_membership(
        repository,
        folder_id=child.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="shared-owner",
    )
    connection = repository.db.get_connection()
    assert (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'"
        ).fetchone()
        is None
    )
    statements: list[str] = []
    connection.set_trace_callback(statements.append)

    repository.page_note_placements(parent_id=parent.folder_id, limit=20, offset=0)

    connection.set_trace_callback(None)
    suppression_statements = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("WITH EFFECTIVE_MEMBERSHIPS")
    ]
    assert len(suppression_statements) == 2
    for statement in suppression_statements:
        details = [
            str(row["detail"])
            for row in connection.execute(f"EXPLAIN QUERY PLAN {statement}")
        ]
        child_lookup = [detail for detail in details if "child_m" in detail]
        assert child_lookup == [
            "SEARCH child_m USING INDEX idx_note_folder_memberships_active_note "
            "(note_id=?)"
        ]


@pytest.mark.parametrize(
    ("parent_id", "limit", "offset"),
    [
        ("", 20, 0),
        (7, 20, 0),
        (False, 20, 0),
        (None, 0, 0),
        (None, 501, 0),
        (None, True, 0),
        (None, 20.5, 0),
        (None, 20, -1),
        (None, 20, False),
        (None, 20, 1.5),
    ],
)
@pytest.mark.parametrize("method_name", ["page_child_folders", "page_note_placements"])
def test_exact_page_methods_validate_before_sql(
    repository: LocalNoteFolderRepository,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    parent_id: object,
    limit: object,
    offset: object,
) -> None:
    def forbidden_transaction():
        raise AssertionError("invalid paging input reached SQL")

    monkeypatch.setattr(repository.db, "transaction", forbidden_transaction)

    with pytest.raises(FolderValidationError):
        getattr(repository, method_name)(
            parent_id=parent_id, limit=limit, offset=offset
        )


def test_load_tree_batch_preserves_totals_beyond_last_page(
    repository: LocalNoteFolderRepository,
) -> None:
    folders = tuple(
        repository.create_folder(name=name, parent_id=None)
        for name in ("A", "B", "C")
    )
    for title in ("One", "Two", "Three"):
        note_id = repository.db.add_note(title, title)
        assert note_id is not None
        _attach_membership(
            repository,
            folder_id=folders[0].folder_id,
            note_id=note_id,
        )

    page = repository.load_tree_batch(
        expanded_folder_ids=(folders[0].folder_id,),
        note_limit=2,
        note_offset=10,
        folder_limit=2,
        folder_offset=10,
    )

    assert page.folders == ()
    assert page.notes == ()
    assert page.memberships == ()
    assert page.total_folders == 0
    assert page.total_notes == 3
    assert page.next_offset is None
    assert page.next_folder_offset is None

    roots = repository.load_tree_batch(
        expanded_folder_ids=(),
        note_limit=2,
        folder_limit=2,
        folder_offset=10,
    )
    assert roots.folders == ()
    assert roots.total_folders == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"note_offset": -1},
        {"folder_limit": 0},
        {"folder_limit": 501},
        {"folder_offset": -1},
        {"membership_limit": 0},
        {"membership_limit": 1001},
        {"membership_offset": -1},
        {"expanded_folder_ids": tuple(f"folder-{index}" for index in range(101))},
    ],
)
def test_load_tree_batch_rejects_unbounded_or_invalid_page_inputs(
    repository: LocalNoteFolderRepository, kwargs: dict[str, object]
) -> None:
    arguments: dict[str, object] = {
        "expanded_folder_ids": (),
        "note_limit": 10,
        **kwargs,
    }
    with pytest.raises(FolderValidationError):
        repository.load_tree_batch(**arguments)


def test_load_tree_batch_bounds_expanded_id_input_consumption(
    repository: LocalNoteFolderRepository,
) -> None:
    consumed = 0

    def repeated_ids() -> Iterator[str]:
        nonlocal consumed
        for _ in range(10_000):
            consumed += 1
            yield "same-folder"

    with pytest.raises(FolderValidationError):
        repository.load_tree_batch(
            expanded_folder_ids=repeated_ids(),
            note_limit=10,
        )

    assert consumed == 101


def test_load_tree_batch_pages_high_cardinality_memberships(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    note_id = repository.db.add_note("Note", "Body")
    assert note_id is not None
    now = _timestamp()
    membership_ids = tuple(str(uuid.uuid4()) for _ in range(1001))
    with repository.db.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO note_folder_memberships("
            "id, folder_id, note_id, ownership, owner_id, owner_active, "
            "version, deleted, created_at, modified_at"
            ") VALUES (?, ?, ?, 'managed', ?, 1, 1, 0, ?, ?)",
            (
                (
                    membership_id,
                    folder.folder_id,
                    note_id,
                    f"owner-{index:04d}",
                    now,
                    now,
                )
                for index, membership_id in enumerate(membership_ids)
            ),
        )

    first = repository.load_tree_batch(
        expanded_folder_ids=(folder.folder_id,),
        note_limit=1,
        membership_limit=1000,
    )
    second = repository.load_tree_batch(
        expanded_folder_ids=(folder.folder_id,),
        note_limit=1,
        membership_limit=1000,
        membership_offset=1000,
    )

    assert len(first.memberships) == 1000
    assert first.total_memberships == 1001
    assert first.next_membership_offset == 1000
    assert len(second.memberships) == 1
    assert second.total_memberships == 1001
    assert second.next_membership_offset is None
    assert {row.membership_id for row in first.memberships + second.memberships} == set(
        membership_ids
    )


def test_load_tree_batch_stays_below_supported_sqlite_variable_limit(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    now = _timestamp()
    note_ids = tuple(str(uuid.uuid4()) for _ in range(1000))
    with repository.db.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO notes("
            "id, title, content, last_modified, client_id, version, deleted, created_at"
            ") VALUES (?, ?, 'Body', ?, 'folder-tests', 1, 0, ?)",
            (
                (note_id, f"Note {index:04d}", now, now)
                for index, note_id in enumerate(note_ids)
            ),
        )
        cursor.executemany(
            "INSERT INTO note_folder_memberships("
            "id, folder_id, note_id, ownership, owner_id, owner_active, "
            "version, deleted, created_at, modified_at"
            ") VALUES (?, ?, ?, 'manual', '', 1, 1, 0, ?, ?)",
            (
                (str(uuid.uuid4()), folder.folder_id, note_id, now, now)
                for note_id in note_ids
            ),
        )

    connection = repository.db.get_connection()
    previous_limit = connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        page = repository.load_tree_batch(
            expanded_folder_ids=(folder.folder_id,),
            note_limit=1000,
            membership_limit=1000,
        )
    finally:
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)

    assert len(page.notes) == 1000
    assert len(page.memberships) == 1000
    assert page.total_memberships == 1000
    assert page.next_membership_offset is None


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


def test_soft_delete_advances_colliding_zulu_tombstone_timestamp(
    repository: LocalNoteFolderRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = repository.create_folder(name="First", parent_id=None)
    second = repository.create_folder(name="Second", parent_id=None)
    fixed_timestamp = "2026-08-13T13:58:15.123Z"
    monkeypatch.setattr(
        folder_repository_module,
        "_utc_timestamp",
        lambda: fixed_timestamp,
    )

    repository.soft_delete_folder(first.folder_id, expected_version=first.version)
    repository.soft_delete_folder(second.folder_id, expected_version=second.version)

    rows = repository.db.get_connection().execute(
        "SELECT id, modified_at FROM note_folders WHERE id IN (?, ?) ORDER BY id",
        (first.folder_id, second.folder_id),
    ).fetchall()
    timestamps = {str(row["modified_at"]) for row in rows}
    assert timestamps == {
        fixed_timestamp,
        "2026-08-13T13:58:15.124Z",
    }


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

    with repository.db.transaction(), pytest.raises(FolderValidationError):
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


def test_wrapped_non_contention_database_error_is_preserved() -> None:
    wrapped = CharactersRAGDBError("non-contention failure")
    wrapped.__cause__ = sqlite3.OperationalError("disk I/O error")
    wrapped.__cause__.sqlite_errorcode = sqlite3.SQLITE_IOERR

    with pytest.raises(CharactersRAGDBError) as caught:
        _raise_wrapped_repository_error(wrapped)

    assert caught.value is wrapped


def test_managed_reconcile_never_removes_manual_membership(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    note_id = repository.db.add_note("Note", "Body")
    assert note_id is not None
    manual = repository.attach_manual(folder_id=folder.folder_id, note_id=note_id)

    repository.reconcile_managed(
        owner_id="root-a", desired=((folder.folder_id, note_id),)
    )
    repository.reconcile_managed(owner_id="root-a", desired=())

    active = repository.list_memberships(
        note_ids=(note_id,), include_inactive=True
    )
    assert active == (manual,)
    assert _note_row(repository, note_id)[3:] == (0, 1)


def test_removing_one_managed_owner_leaves_other_owner_and_note(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    note_id = repository.db.add_note("Note", "Body")
    assert note_id is not None
    repository.reconcile_managed(
        owner_id="root-a", desired=((folder.folder_id, note_id),)
    )
    repository.reconcile_managed(
        owner_id="root-b", desired=((folder.folder_id, note_id),)
    )
    note_before = _note_row(repository, note_id)

    assert repository.remove_owner_memberships(owner_id="root-a") == 1
    remaining = repository.list_memberships(
        note_ids=(note_id,), include_inactive=True
    )

    assert [(item.ownership, item.owner_id) for item in remaining] == [
        ("managed", "root-b")
    ]
    assert _note_row(repository, note_id) == note_before
    assert repository.remove_owner_memberships(owner_id="root-a") == 0


def test_attach_manual_is_idempotent_and_revives_only_latest_history(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    note_id = repository.db.add_note("Note", "Body")
    assert note_id is not None
    now = "2026-08-12T12:00:00.000Z"
    with repository.db.transaction() as cursor:
        for membership_id in ("manual-a", "manual-b"):
            cursor.execute(
                """
                INSERT INTO note_folder_memberships(
                    id, folder_id, note_id, ownership, owner_id, owner_active,
                    version, deleted, created_at, modified_at
                ) VALUES (?, ?, ?, 'manual', '', 1, 4, 1, ?, ?)
                """,
                (membership_id, folder.folder_id, note_id, now, now),
            )

    revived = repository.attach_manual(
        folder_id=folder.folder_id, note_id=note_id
    )
    repeated = repository.attach_manual(
        folder_id=folder.folder_id, note_id=note_id
    )

    assert revived.membership_id == "manual-b"
    assert revived.version == 5
    assert repeated == revived
    rows = repository.db.get_connection().execute(
        "SELECT id, version, deleted FROM note_folder_memberships ORDER BY id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("manual-a", 4, 1),
        ("manual-b", 5, 0),
    ]


def test_conflict_copy_exact_manual_membership_reads_active_then_latest_deleted(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Conflict copies", parent_id=None)
    note_id = repository.db.add_note("Original", "note side")
    assert note_id is not None
    active_id = _attach_membership(
        repository,
        folder_id=folder.folder_id,
        note_id=note_id,
    )
    _attach_membership(
        repository,
        folder_id=folder.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="root-1",
    )
    with repository.db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO note_folder_memberships("
            "id, folder_id, note_id, ownership, owner_id, owner_active, "
            "version, deleted, created_at, modified_at) "
            "VALUES ('manual-old', ?, ?, 'manual', '', 1, 3, 1, ?, ?)",
            (
                folder.folder_id,
                note_id,
                "2026-08-20T00:00:00.000Z",
                "2026-08-20T00:00:00.000Z",
            ),
        )

    active = repository.get_exact_manual_membership(
        folder_id=folder.folder_id,
        note_id=note_id,
        include_deleted=True,
    )

    assert active is not None
    assert active[0].membership_id == active_id
    assert active[1] is False
    assert repository.detach_manual(
        folder_id=folder.folder_id,
        note_id=note_id,
        expected_version=active[0].version,
    )
    assert (
        repository.get_exact_manual_membership(
            folder_id=folder.folder_id,
            note_id=note_id,
            include_deleted=False,
        )
        is None
    )
    deleted = repository.get_exact_manual_membership(
        folder_id=folder.folder_id,
        note_id=note_id,
        include_deleted=True,
    )
    assert deleted is not None
    assert deleted[0].membership_id == active_id
    assert deleted[0].version == active[0].version + 1
    assert deleted[1] is True


def test_conflict_copy_folder_ownership_detects_managed_candidate_and_ancestor(
    repository: LocalNoteFolderRepository,
) -> None:
    parent = repository.create_folder(name="Conflict copies", parent_id=None)
    child = repository.create_folder(
        name="My synced notes", parent_id=parent.folder_id
    )
    note_id = repository.db.add_note("Managed", "body")
    assert note_id is not None
    repository.reconcile_managed(
        owner_id="another-root",
        desired=((child.folder_id, note_id),),
    )

    assert repository.has_managed_folder_ownership(child.folder_id) is True
    assert repository.has_managed_folder_ownership(parent.folder_id) is True


def test_attach_manual_expected_note_version_guards_new_active_and_revived_rows(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Guarded", parent_id=None)
    note_id = repository.db.add_note("Note", "Body")
    assert note_id is not None
    repository.db.get_connection().execute(
        "UPDATE notes SET version = 2 WHERE id = ?",
        (note_id,),
    )
    repository.db.get_connection().commit()

    with pytest.raises(FolderConflictError):
        repository.attach_manual(
            folder_id=folder.folder_id,
            note_id=note_id,
            expected_note_version=1,
        )
    assert repository.list_memberships(note_ids=(note_id,)) == ()

    created = repository.attach_manual(
        folder_id=folder.folder_id,
        note_id=note_id,
        expected_note_version=2,
    )
    with pytest.raises(FolderConflictError):
        repository.attach_manual(
            folder_id=folder.folder_id,
            note_id=note_id,
            expected_note_version=1,
        )
    assert repository.list_memberships(note_ids=(note_id,)) == (created,)

    assert repository.detach_manual(
        folder_id=folder.folder_id,
        note_id=note_id,
        expected_version=created.version,
    )
    with pytest.raises(FolderConflictError):
        repository.attach_manual(
            folder_id=folder.folder_id,
            note_id=note_id,
            expected_note_version=1,
        )
    assert repository.list_memberships(note_ids=(note_id,)) == ()

    revived = repository.attach_manual(
        folder_id=folder.folder_id,
        note_id=note_id,
        expected_note_version=2,
    )
    assert revived.membership_id == created.membership_id
    assert revived.version == created.version + 2


def test_attach_manual_retries_a_generated_membership_id_collision(
    repository: LocalNoteFolderRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_folder = repository.create_folder(name="First", parent_id=None)
    second_folder = repository.create_folder(name="Second", parent_id=None)
    first_note = repository.db.add_note("First", "Body")
    second_note = repository.db.add_note("Second", "Body")
    assert first_note is not None and second_note is not None
    existing = repository.attach_manual(
        folder_id=first_folder.folder_id, note_id=first_note
    )
    replacement_id = uuid.uuid4()
    generated_ids = iter(
        (uuid.uuid4(), uuid.UUID(existing.membership_id), replacement_id)
    )
    monkeypatch.setattr(
        folder_repository_module.uuid, "uuid4", lambda: next(generated_ids)
    )

    attached = repository.attach_manual(
        folder_id=second_folder.folder_id, note_id=second_note
    )

    assert attached.membership_id == str(replacement_id)
    assert set(
        repository.list_memberships(note_ids=(first_note, second_note))
    ) == {existing, attached}


def test_membership_unique_classifier_distinguishes_owner_and_primary_key(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    first_note = repository.db.add_note("First", "Body")
    second_note = repository.db.add_note("Second", "Body")
    assert first_note is not None and second_note is not None
    existing = repository.attach_manual(folder_id=folder.folder_id, note_id=first_note)
    connection = repository.db.get_connection()
    now = _timestamp()

    with pytest.raises(sqlite3.IntegrityError) as owner_collision:
        connection.execute(
            """
            INSERT INTO note_folder_memberships(
                id, folder_id, note_id, ownership, owner_id,
                created_at, modified_at
            ) VALUES (?, ?, ?, 'manual', '', ?, ?)
            """,
            (str(uuid.uuid4()), folder.folder_id, first_note, now, now),
        )
    connection.rollback()
    with pytest.raises(FolderConflictError, match="Membership changed"):
        _raise_membership_integrity_error(owner_collision.value)

    with pytest.raises(sqlite3.IntegrityError) as id_collision:
        connection.execute(
            """
            INSERT INTO note_folder_memberships(
                id, folder_id, note_id, ownership, owner_id,
                created_at, modified_at
            ) VALUES (?, ?, ?, 'manual', '', ?, ?)
            """,
            (existing.membership_id, folder.folder_id, second_note, now, now),
        )
    connection.rollback()
    with pytest.raises(FolderValidationError, match="stored constraints"):
        _raise_membership_integrity_error(id_collision.value)


def test_detach_manual_succeeds_conflicts_when_stale_and_is_false_when_absent(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    note_id = repository.db.add_note("Note", "Body")
    assert note_id is not None
    membership = repository.attach_manual(
        folder_id=folder.folder_id, note_id=note_id
    )

    with pytest.raises(FolderConflictError):
        repository.detach_manual(
            folder_id=folder.folder_id,
            note_id=note_id,
            expected_version=membership.version + 1,
        )
    assert repository.list_memberships(note_ids=(note_id,)) == (membership,)

    assert repository.detach_manual(
        folder_id=folder.folder_id,
        note_id=note_id,
        expected_version=membership.version,
    )
    assert repository.detach_manual(
        folder_id=folder.folder_id,
        note_id=note_id,
        expected_version=membership.version + 1,
    ) is False


@pytest.mark.parametrize("invalid_kind", ["folder", "note"])
def test_reconcile_validates_all_desired_rows_before_writing(
    repository: LocalNoteFolderRepository, invalid_kind: str
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    note_id = repository.db.add_note("Note", "Body")
    assert note_id is not None
    repository.reconcile_managed(
        owner_id="root-a", desired=((folder.folder_id, note_id),)
    )
    before = _membership_rows(repository)
    desired = (
        (
            "missing-folder" if invalid_kind == "folder" else folder.folder_id,
            "missing-note" if invalid_kind == "note" else note_id,
        ),
    )

    with pytest.raises(FolderValidationError):
        repository.reconcile_managed(owner_id="root-a", desired=desired)

    assert _membership_rows(repository) == before


def test_reconcile_is_idempotent_owner_scoped_and_revives_deleted_rows(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    note_id = repository.db.add_note("Note", "Body")
    assert note_id is not None
    manual = repository.attach_manual(folder_id=folder.folder_id, note_id=note_id)
    other = repository.reconcile_managed(
        owner_id="root-b", desired=((folder.folder_id, note_id),)
    )[0]
    first = repository.reconcile_managed(
        owner_id="root-a", desired=((folder.folder_id, note_id),)
    )[0]
    rows_after_first = _membership_rows(repository)

    assert repository.reconcile_managed(
        owner_id="root-a", desired=((folder.folder_id, note_id),)
    ) == (first,)
    assert _membership_rows(repository) == rows_after_first

    repository.reconcile_managed(owner_id="root-a", desired=())
    repository.db.get_connection().execute(
        "UPDATE note_folder_memberships SET owner_active = 0 WHERE id = ?",
        (first.membership_id,),
    )
    repository.db.get_connection().commit()
    revived = repository.reconcile_managed(
        owner_id="root-a", desired=((folder.folder_id, note_id),)
    )[0]

    assert revived.membership_id == first.membership_id
    assert revived.version == first.version + 2
    assert revived.owner_active
    all_active = repository.list_memberships(
        note_ids=(note_id,), include_inactive=True
    )
    assert {item.membership_id for item in all_active} == {
        manual.membership_id,
        other.membership_id,
        revived.membership_id,
    }


def test_convert_owner_to_manual_reuses_active_and_revives_deleted_manual_rows(
    repository: LocalNoteFolderRepository,
) -> None:
    first_folder = repository.create_folder(name="First", parent_id=None)
    second_folder = repository.create_folder(name="Second", parent_id=None)
    first_note = repository.db.add_note("First", "Body")
    second_note = repository.db.add_note("Second", "Body")
    assert first_note is not None and second_note is not None
    active_manual = repository.attach_manual(
        folder_id=first_folder.folder_id, note_id=first_note
    )
    deleted_manual = repository.attach_manual(
        folder_id=second_folder.folder_id, note_id=second_note
    )
    assert repository.detach_manual(
        folder_id=second_folder.folder_id,
        note_id=second_note,
        expected_version=deleted_manual.version,
    )
    repository.reconcile_managed(
        owner_id="root-a",
        desired=(
            (first_folder.folder_id, first_note),
            (second_folder.folder_id, second_note),
        ),
    )
    other = repository.reconcile_managed(
        owner_id="root-b", desired=((first_folder.folder_id, first_note),)
    )[0]
    notes_before = (_note_row(repository, first_note), _note_row(repository, second_note))

    assert repository.convert_owner_to_manual(owner_id="root-a") == 2
    assert repository.convert_owner_to_manual(owner_id="root-a") == 0

    memberships = repository.list_memberships(
        note_ids=(first_note, second_note), include_inactive=True
    )
    manual_rows = [item for item in memberships if item.ownership == "manual"]
    assert {item.membership_id for item in manual_rows} == {
        active_manual.membership_id,
        deleted_manual.membership_id,
    }
    assert next(
        item for item in manual_rows if item.membership_id == active_manual.membership_id
    ).version == active_manual.version
    assert next(
        item for item in manual_rows if item.membership_id == deleted_manual.membership_id
    ).version == deleted_manual.version + 2
    assert [item for item in memberships if item.ownership == "managed"] == [other]
    assert (_note_row(repository, first_note), _note_row(repository, second_note)) == notes_before


def test_unknown_owner_convergence_and_restore_reviews_are_deterministic(
    repository: LocalNoteFolderRepository,
) -> None:
    alpha = repository.create_folder(name="Alpha", parent_id=None)
    beta = repository.create_folder(name="Beta", parent_id=None)
    first_note = repository.db.add_note("First", "Body")
    second_note = repository.db.add_note("Second", "Body")
    assert first_note is not None and second_note is not None
    root_a = repository.reconcile_managed(
        owner_id="root-a",
        desired=((alpha.folder_id, first_note), (beta.folder_id, first_note)),
    )
    root_b = repository.reconcile_managed(
        owner_id="root-b", desired=((beta.folder_id, second_note),)
    )[0]

    assert repository.mark_unknown_owners_inactive(
        active_owner_ids=("root-b", "root-b")
    ) == 2
    assert repository.mark_unknown_owners_inactive(active_owner_ids=("root-b",)) == 0
    assert repository.list_memberships(note_ids=(first_note, second_note)) == (root_b,)
    assert {
        item.membership_id
        for item in repository.list_memberships(
            note_ids=(first_note, second_note), include_inactive=True
        )
    } == {root_a[0].membership_id, root_a[1].membership_id, root_b.membership_id}

    reviews = repository.list_restore_reviews()
    assert len(reviews) == 1
    assert reviews[0].owner_id == "root-a"
    assert reviews[0].membership_ids == tuple(
        sorted(item.membership_id for item in root_a)
    )
    assert (reviews[0].note_count, reviews[0].folder_count) == (1, 2)

    assert repository.mark_unknown_owners_inactive(
        active_owner_ids=("root-a", "root-b")
    ) == 2
    assert repository.list_restore_reviews() == ()


def test_reconcile_rolls_back_to_owned_savepoint_when_later_update_fails(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    note_ids = [repository.db.add_note(title, "Body") for title in ("A", "B")]
    assert all(note_id is not None for note_id in note_ids)
    repository.reconcile_managed(
        owner_id="root-a",
        desired=tuple((folder.folder_id, str(note_id)) for note_id in note_ids),
    )
    fail_note_id = max(str(note_id) for note_id in note_ids)
    connection = repository.db.get_connection()
    connection.execute(
        f"""
        CREATE TRIGGER fail_later_membership_update
        BEFORE UPDATE ON note_folder_memberships
        WHEN OLD.note_id = '{fail_note_id}'
        BEGIN
          SELECT RAISE(ABORT, 'forced later membership failure');
        END
        """
    )
    connection.commit()
    before = _membership_rows(repository)

    with repository.db.transaction(), pytest.raises(FolderValidationError):
        repository.reconcile_managed(owner_id="root-a", desired=())

    assert _membership_rows(repository) == before


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("attach_manual", {"folder_id": "", "note_id": "note"}),
        ("attach_manual", {"folder_id": "folder", "note_id": ""}),
        ("list_memberships", {"note_ids": "note"}),
        ("reconcile_managed", {"owner_id": "", "desired": ()}),
        ("reconcile_managed", {"owner_id": "root", "desired": ("bad",)}),
        ("convert_owner_to_manual", {"owner_id": " "}),
        ("remove_owner_memberships", {"owner_id": ""}),
        ("mark_unknown_owners_inactive", {"active_owner_ids": "root"}),
    ],
)
def test_membership_methods_reject_invalid_inputs(
    repository: LocalNoteFolderRepository, method: str, kwargs: dict[str, object]
) -> None:
    with pytest.raises(FolderValidationError):
        getattr(repository, method)(**kwargs)


@pytest.mark.parametrize(
    ("note_count", "expected_next_offset"), [(0, None), (2, None), (3, 2)]
)
def test_load_tree_batch_pages_zero_exact_limit_and_limit_plus_one_notes(
    repository: LocalNoteFolderRepository,
    note_count: int,
    expected_next_offset: int | None,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    now = _timestamp()
    with repository.db.transaction() as cursor:
        for index in range(note_count):
            note_id = f"note-{index:03d}"
            cursor.execute(
                "INSERT INTO notes(id, title, content, client_id) VALUES (?, ?, '', ?)",
                (note_id, f"Note {index:03d}", "paging-test"),
            )
            cursor.execute(
                """
                INSERT INTO note_folder_memberships(
                    id, folder_id, note_id, ownership, owner_id,
                    created_at, modified_at
                ) VALUES (?, ?, ?, 'manual', '', ?, ?)
                """,
                (f"membership-{index:03d}", folder.folder_id, note_id, now, now),
            )

    page = repository.load_tree_batch(
        expanded_folder_ids=(folder.folder_id,), note_limit=2
    )

    assert len(page.notes) == min(note_count, 2)
    assert page.total_notes == note_count
    assert page.next_offset == expected_next_offset
    assert len(page.memberships) == min(note_count, 2)


@pytest.mark.parametrize("placement_count", [10, 500])
def test_load_tree_batch_query_count_is_constant_for_placements(
    repository: LocalNoteFolderRepository, placement_count: int
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    now = _timestamp()
    with repository.db.transaction() as cursor:
        for index in range(placement_count):
            note_id = f"note-{index:04d}"
            cursor.execute(
                "INSERT INTO notes(id, title, content, client_id) VALUES (?, ?, '', ?)",
                (note_id, f"Note {index:04d}", "query-shape-test"),
            )
            cursor.execute(
                """
                INSERT INTO note_folder_memberships(
                    id, folder_id, note_id, ownership, owner_id,
                    created_at, modified_at
                ) VALUES (?, ?, ?, 'manual', '', ?, ?)
                """,
                (f"membership-{index:04d}", folder.folder_id, note_id, now, now),
            )
    statements: list[str] = []
    connection = repository.db.get_connection()
    connection.set_trace_callback(statements.append)

    page = repository.load_tree_batch(
        expanded_folder_ids=(folder.folder_id,), note_limit=1000
    )

    connection.set_trace_callback(None)
    selects = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith(("SELECT", "WITH"))
    ]
    assert len(page.notes) == placement_count
    assert 3 <= len(selects) <= 4
    assert len(selects) <= 4


def test_list_memberships_chunks_large_unique_id_sets_without_per_note_queries(
    repository: LocalNoteFolderRepository,
) -> None:
    note_ids = tuple(f"note-{index:04d}" for index in range(801))
    statements: list[str] = []
    connection = repository.db.get_connection()
    connection.set_trace_callback(statements.append)

    assert repository.list_memberships(note_ids=reversed(note_ids)) == ()

    connection.set_trace_callback(None)
    selects = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith(("SELECT", "WITH"))
    ]
    assert len(selects) == 3
    assert all("note_id IN" in statement for statement in selects)


def test_tree_locator_folder_returns_deep_root_to_target_page_offsets(
    repository: LocalNoteFolderRepository,
) -> None:
    roots = tuple(
        repository.create_folder(
            name=f"Root {index:02d}", parent_id=None, folder_id=f"root-{index:02d}"
        )
        for index in range(25)
    )
    children = tuple(
        repository.create_folder(
            name=f"Child {index:02d}",
            parent_id=roots[-1].folder_id,
            folder_id=f"child-{index:02d}",
        )
        for index in range(25)
    )

    location = repository.locate_note_tree_folder(
        folder_id=children[-1].folder_id, page_size=20
    )

    assert location is not None
    assert location.placement_id == FolderPlacementId.folder(children[-1].folder_id)
    assert location.note_id is None
    assert location.membership_id is None
    assert location.placement_offset is None
    assert location.path == (
        NoteTreePathStep(
            folder_id=roots[-1].folder_id,
            parent_id=None,
            containing_offset=20,
        ),
        NoteTreePathStep(
            folder_id=children[-1].folder_id,
            parent_id=roots[-1].folder_id,
            containing_offset=20,
        ),
    )


def test_tree_locator_path_plan_searches_siblings_by_parent(
    repository: LocalNoteFolderRepository,
) -> None:
    root = repository.create_folder(
        name="Plan Root", parent_id=None, folder_id="plan-root"
    )
    child = repository.create_folder(
        name="Plan Child", parent_id=root.folder_id, folder_id="plan-child"
    )
    _insert_note(repository, note_id="plan-note", title="Plan Note")
    _attach_membership(
        repository,
        folder_id=child.folder_id,
        note_id="plan-note",
        membership_id="plan-membership",
    )
    connection = repository.db.get_connection()
    assert (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'"
        ).fetchone()
        is None
    )
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    try:
        folder_location = repository.locate_note_tree_folder(
            folder_id=child.folder_id, page_size=20
        )
        placement_location = repository.locate_note_tree_placement(
            note_id="plan-note", page_size=20
        )
    finally:
        connection.set_trace_callback(None)

    expected_path = (
        NoteTreePathStep(folder_id=root.folder_id, parent_id=None, containing_offset=0),
        NoteTreePathStep(
            folder_id=child.folder_id,
            parent_id=root.folder_id,
            containing_offset=0,
        ),
    )
    assert folder_location is not None
    assert folder_location.path == expected_path
    assert placement_location is not None
    assert placement_location.path == expected_path
    path_statements = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("WITH RECURSIVE PATH")
    ]
    assert len(path_statements) == 2
    for statement in path_statements:
        details = [
            str(row["detail"])
            for row in connection.execute(f"EXPLAIN QUERY PLAN {statement}")
        ]
        sibling_lookup = [detail for detail in details if "sibling" in detail]
        assert sibling_lookup == [
            "SEARCH sibling USING INDEX idx_note_folders_active_parent (parent_id=?)"
        ]
        assert not any("SCAN sibling" in detail for detail in details)


def test_tree_locator_folder_returns_none_for_inactive_target(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Removed", parent_id=None)
    with repository.db.transaction() as cursor:
        cursor.execute(
            "UPDATE note_folders SET deleted = 1 WHERE id = ?", (folder.folder_id,)
        )

    assert (
        repository.locate_note_tree_folder(folder_id=folder.folder_id, page_size=20)
        is None
    )


def test_tree_locator_placement_honors_preferences_canonical_order_and_offsets(
    repository: LocalNoteFolderRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha = repository.create_folder(name="Alpha", parent_id=None, folder_id="alpha")
    child = repository.create_folder(
        name="Child", parent_id=alpha.folder_id, folder_id="alpha-child"
    )
    beta = repository.create_folder(name="Beta", parent_id=None, folder_id="beta")
    for index in range(25):
        note_id = f"leading-{index:02d}"
        _insert_note(repository, note_id=note_id, title=f"Ahead {index:02d}")
        _attach_membership(
            repository,
            folder_id=child.folder_id,
            note_id=note_id,
            membership_id=f"leading-membership-{index:02d}",
        )
    note_id = "located-note"
    _insert_note(repository, note_id=note_id, title="Located")
    alpha_manual = _attach_membership(
        repository,
        folder_id=child.folder_id,
        note_id=note_id,
        membership_id="membership-alpha-manual",
    )
    _attach_membership(
        repository,
        folder_id=alpha.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="shadow-owner",
        membership_id="membership-shadowed-ancestor",
    )
    _attach_membership(
        repository,
        folder_id=child.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="shadow-owner",
        membership_id="membership-visible-child",
    )
    beta_first = _attach_membership(
        repository,
        folder_id=beta.folder_id,
        note_id=note_id,
        membership_id="membership-beta-a",
    )
    beta_exact = _attach_membership(
        repository,
        folder_id=beta.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="beta-exact-owner",
        membership_id="membership-beta-z",
    )

    original_transaction = repository.db.transaction
    transaction_count = 0

    def counted_transaction(*, immediate: bool = False):
        nonlocal transaction_count
        transaction_count += 1
        return original_transaction(immediate=immediate)

    monkeypatch.setattr(repository.db, "transaction", counted_transaction)

    exact = repository.locate_note_tree_placement(
        note_id=note_id,
        page_size=20,
        preferred_folder_id=alpha.folder_id,
        preferred_membership_id=beta_exact,
    )
    assert transaction_count == 1
    transaction_count = 0
    preferred_folder = repository.locate_note_tree_placement(
        note_id=note_id,
        page_size=20,
        preferred_folder_id=beta.folder_id,
        preferred_membership_id="missing-membership",
    )
    assert transaction_count == 1
    transaction_count = 0
    canonical = repository.locate_note_tree_placement(note_id=note_id, page_size=20)
    assert transaction_count == 1

    assert exact is not None and exact.membership_id == beta_exact
    assert exact.placement_id == FolderPlacementId.note(
        beta.folder_id, note_id, beta_exact
    )
    assert preferred_folder is not None
    assert preferred_folder.membership_id == beta_first
    assert canonical is not None
    assert canonical.membership_id == alpha_manual
    assert canonical.path == (
        NoteTreePathStep(alpha.folder_id, None, 0),
        NoteTreePathStep(child.folder_id, alpha.folder_id, 0),
    )
    assert canonical.placement_offset == 20


def test_tree_locator_placement_uses_unfiled_only_without_active_placement(
    repository: LocalNoteFolderRepository,
) -> None:
    folder = repository.create_folder(name="Folder", parent_id=None)
    unfiled_note = "unfiled-locator-note"
    inactive_note = "inactive-membership-note"
    deleted_note = "deleted-locator-note"
    for note_id in (unfiled_note, inactive_note, deleted_note):
        _insert_note(repository, note_id=note_id, title=note_id)
    _attach_membership(
        repository,
        folder_id=folder.folder_id,
        note_id=inactive_note,
        ownership="managed",
        owner_id="inactive-owner",
        owner_active=False,
    )
    with repository.db.transaction() as cursor:
        cursor.execute("UPDATE notes SET deleted = 1 WHERE id = ?", (deleted_note,))

    location = repository.locate_note_tree_placement(note_id=unfiled_note, page_size=20)
    inactive_location = repository.locate_note_tree_placement(
        note_id=inactive_note, page_size=20
    )

    assert location is not None
    assert location.placement_id == FolderPlacementId.unfiled(unfiled_note)
    assert location.membership_id is None
    assert location.path == ()
    assert location.placement_offset == 0
    assert inactive_location is not None
    assert inactive_location.membership_id is None
    assert (
        repository.locate_note_tree_placement(note_id=deleted_note, page_size=20)
        is None
    )
    assert (
        repository.locate_note_tree_placement(note_id="missing-note", page_size=20)
        is None
    )


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("locate_note_tree_folder", {"folder_id": "", "page_size": 20}),
        ("locate_note_tree_folder", {"folder_id": 7, "page_size": 20}),
        ("locate_note_tree_folder", {"folder_id": "folder", "page_size": 0}),
        ("locate_note_tree_folder", {"folder_id": "folder", "page_size": 501}),
        ("locate_note_tree_placement", {"note_id": "", "page_size": 20}),
        ("locate_note_tree_placement", {"note_id": "note", "page_size": True}),
        (
            "locate_note_tree_placement",
            {"note_id": "note", "page_size": 20, "preferred_folder_id": 7},
        ),
        (
            "locate_note_tree_placement",
            {"note_id": "note", "page_size": 20, "preferred_membership_id": ""},
        ),
    ],
)
def test_tree_locator_validates_all_inputs_before_sql(
    repository: LocalNoteFolderRepository,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    kwargs: dict[str, object],
) -> None:
    def forbidden_transaction():
        raise AssertionError("invalid locator input reached SQL")

    monkeypatch.setattr(repository.db, "transaction", forbidden_transaction)

    with pytest.raises(FolderValidationError):
        getattr(repository, method_name)(**kwargs)


def test_affected_parents_context_includes_subtrees_ancestors_and_all_placements(
    repository: LocalNoteFolderRepository,
) -> None:
    root = repository.create_folder(name="Root", parent_id=None, folder_id="ctx-root")
    parent = repository.create_folder(
        name="Parent", parent_id=root.folder_id, folder_id="ctx-parent"
    )
    target = repository.create_folder(
        name="Target", parent_id=parent.folder_id, folder_id="ctx-target"
    )
    child = repository.create_folder(
        name="Child", parent_id=target.folder_id, folder_id="ctx-child"
    )
    inactive = repository.create_folder(
        name="Inactive placement", parent_id=None, folder_id="ctx-inactive"
    )
    note_id = "affected-note"
    _insert_note(repository, note_id=note_id, title="Affected")
    _attach_membership(repository, folder_id=target.folder_id, note_id=note_id)
    _attach_membership(repository, folder_id=child.folder_id, note_id=note_id)
    _attach_membership(
        repository,
        folder_id=inactive.folder_id,
        note_id=note_id,
        ownership="managed",
        owner_id="inactive-owner",
        owner_active=False,
    )

    context = repository.load_note_tree_mutation_context(
        folder_ids=(target.folder_id, target.folder_id),
        note_ids=(note_id, note_id),
        include_folder_subtrees=True,
    )

    assert isinstance(context, NoteTreeMutationContext)
    assert context.folder_ids == (child.folder_id, target.folder_id)
    assert context.parent_ids == (parent.folder_id, target.folder_id)
    assert context.ancestor_ids == (
        parent.folder_id,
        root.folder_id,
        target.folder_id,
    )
    assert context.placement_parent_ids == (child.folder_id, target.folder_id)

    with repository.db.transaction() as cursor:
        cursor.execute("UPDATE notes SET deleted = 1 WHERE id = ?", (note_id,))
    deleted_note_context = repository.load_note_tree_mutation_context(
        note_ids=(note_id,)
    )
    assert deleted_note_context.placement_parent_ids == (
        child.folder_id,
        target.folder_id,
    )

    with repository.db.transaction() as cursor:
        cursor.execute(
            "UPDATE note_folders SET deleted = 1 WHERE id IN (?, ?)",
            (target.folder_id, child.folder_id),
        )
    deleted_context = repository.load_note_tree_mutation_context(
        folder_ids=(target.folder_id,), include_folder_subtrees=True
    )
    assert deleted_context.folder_ids == (child.folder_id, target.folder_id)
    assert deleted_context.parent_ids == (parent.folder_id, target.folder_id)
    assert deleted_context.ancestor_ids == (
        parent.folder_id,
        root.folder_id,
        target.folder_id,
    )


def test_affected_parents_context_omits_unfiled_sentinel_and_is_frozen(
    repository: LocalNoteFolderRepository,
) -> None:
    _insert_note(repository, note_id="unfiled-context-note", title="Unfiled")

    context = repository.load_note_tree_mutation_context(
        note_ids=(item for item in ("unfiled-context-note", "unfiled-context-note"))
    )

    assert context == NoteTreeMutationContext((), (), (), ())
    with pytest.raises(AttributeError):
        context.folder_ids = ("changed",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"folder_ids": "folder"},
        {"folder_ids": ("",)},
        {"note_ids": "note"},
        {"note_ids": (False,)},
        {"include_folder_subtrees": 1},
    ],
)
def test_affected_parents_validates_inputs_before_sql(
    repository: LocalNoteFolderRepository,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
) -> None:
    def forbidden_transaction():
        raise AssertionError("invalid mutation-context input reached SQL")

    monkeypatch.setattr(repository.db, "transaction", forbidden_transaction)

    with pytest.raises(FolderValidationError):
        repository.load_note_tree_mutation_context(**kwargs)


def test_search_note_placement_page_filters_suppresses_orders_and_pages_exactly(
    repository: LocalNoteFolderRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha = repository.create_folder(name="Alpha", parent_id=None, folder_id="alpha")
    project = repository.create_folder(
        name="Project", parent_id=alpha.folder_id, folder_id="alpha-project"
    )
    beta = repository.create_folder(name="Beta", parent_id=None, folder_id="beta")
    unrelated = repository.create_folder(
        name="Unrelated", parent_id=None, folder_id="unrelated"
    )

    _insert_note(repository, note_id="shadow-note", title="Alpha", content="project")
    _attach_membership(
        repository,
        folder_id=alpha.folder_id,
        note_id="shadow-note",
        ownership="managed",
        owner_id="shadow-owner",
        membership_id="shadow-ancestor",
    )
    _attach_membership(
        repository,
        folder_id=project.folder_id,
        note_id="shadow-note",
        ownership="managed",
        owner_id="shadow-owner",
        membership_id="shadow-descendant",
    )
    _insert_note(repository, note_id="duplicate-note", title="Beta", content="project")
    for membership_id in ("duplicate-a", "duplicate-b"):
        _attach_membership(
            repository,
            folder_id=project.folder_id,
            note_id="duplicate-note",
            ownership="manual" if membership_id == "duplicate-a" else "managed",
            owner_id="" if membership_id == "duplicate-a" else "duplicate-owner",
            membership_id=membership_id,
        )
    _insert_note(repository, note_id="breadcrumb-note", title="Zulu", content="other")
    _attach_membership(
        repository,
        folder_id=project.folder_id,
        note_id="breadcrumb-note",
        membership_id="breadcrumb-membership",
    )
    _insert_note(repository, note_id="beta-note", title="Project plan", content="other")
    _attach_membership(
        repository,
        folder_id=beta.folder_id,
        note_id="beta-note",
        membership_id="beta-membership",
    )
    _insert_note(repository, note_id="unfiled-note", title="Unfiled", content="project")
    _insert_note(
        repository, note_id="unrelated-note", title="No match", content="other"
    )
    _attach_membership(
        repository,
        folder_id=unrelated.folder_id,
        note_id="unrelated-note",
        membership_id="unrelated-membership",
    )

    original_transaction = repository.db.transaction
    transaction_count = 0

    def counted_transaction(*, immediate: bool = False):
        nonlocal transaction_count
        transaction_count += 1
        return original_transaction(immediate=immediate)

    monkeypatch.setattr(repository.db, "transaction", counted_transaction)

    first = repository.search_note_tree_placements(query="project", limit=3, offset=0)
    assert transaction_count == 1
    transaction_count = 0
    second = repository.search_note_tree_placements(query="project", limit=3, offset=3)
    assert transaction_count == 1
    transaction_count = 0
    empty = repository.search_note_tree_placements(query="project", limit=3, offset=20)
    assert transaction_count == 1

    assert [
        (
            placement.folder_id,
            placement.note["id"],
            placement.membership.membership_id if placement.membership else None,
        )
        for placement in first.placements
    ] == [
        (project.folder_id, "shadow-note", "shadow-descendant"),
        (project.folder_id, "duplicate-note", "duplicate-a"),
        (project.folder_id, "duplicate-note", "duplicate-b"),
    ]
    assert first.total_placements == 6
    assert first.start_offset == 0
    assert first.previous_offset is None
    assert first.next_offset == 3
    assert first.ancestor_folders == (alpha, project)

    assert [
        (
            placement.folder_id,
            placement.note["id"],
            placement.membership.membership_id if placement.membership else None,
        )
        for placement in second.placements
    ] == [
        (project.folder_id, "breadcrumb-note", "breadcrumb-membership"),
        (beta.folder_id, "beta-note", "beta-membership"),
        (None, "unfiled-note", None),
    ]
    assert second.total_placements == 6
    assert second.start_offset == 3
    assert second.previous_offset == 0
    assert second.next_offset is None
    assert second.ancestor_folders == (alpha, project, beta)

    assert empty.placements == ()
    assert empty.total_placements == 6
    assert empty.start_offset == 20
    assert empty.previous_offset == 3
    assert empty.next_offset is None
    assert empty.ancestor_folders == ()


@pytest.mark.parametrize(
    ("query", "limit", "offset"),
    [
        (7, 20, 0),
        ("bad\x00query", 20, 0),
        ("x" * 201, 20, 0),
        ("query", 0, 0),
        ("query", 501, 0),
        ("query", True, 0),
        ("query", 20, -1),
        ("query", 20, False),
    ],
)
def test_search_note_placement_page_validates_before_sql(
    repository: LocalNoteFolderRepository,
    monkeypatch: pytest.MonkeyPatch,
    query: object,
    limit: object,
    offset: object,
) -> None:
    def forbidden_transaction():
        raise AssertionError("invalid search-page input reached SQL")

    monkeypatch.setattr(repository.db, "transaction", forbidden_transaction)

    with pytest.raises(FolderValidationError):
        repository.search_note_tree_placements(query=query, limit=limit, offset=offset)
