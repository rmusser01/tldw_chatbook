# Local Database Note Folder Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the durable local schema, repository, and normalized service contract for hierarchical Database Note folders and ownership-aware memberships.

**Architecture:** ChaChaNotes v36 owns logical folders and memberships; a focused repository performs all folder SQL and a normalized Notes service routes scope-aware operations. The repository returns bounded bulk pages for the dependent navigator task without importing Textual. This plan changes no Notes UI, filesystem sync behavior, one-time import flow, Sync-v2 M1 payload, or remote folder API.

**Tech Stack:** Python 3.11+, SQLite/FTS5, frozen dataclasses, pytest/pytest-asyncio, Hypothesis

---

## Scope and governance

- Backlog task: `TASK-15705`
- Dependent navigator task: `TASK-15706`
- Approved design: `Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md`
- ADR required: yes
- ADR paths:
  - `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
  - `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
- Reason: this slice implements the accepted local folder schema, ownership,
  backup, normalized service, and paging boundaries.

## Explicit slice boundary

This plan delivers the local data/service foundation consumed by `TASK-15706`.
It does not change the Notes navigator, rename the Media entry, build **Add from
files…**, import directory content, create the device-private sync database, watch
files, migrate legacy sync roots, or add server folder endpoints.
`ScopeType.SERVER_NOTE` reports honest unsupported folder capabilities until the
server-contract slice lands. Existing Sync-v2 M1 `notes.note` payloads remain
unchanged.

## File responsibility map

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/DB/migrations/chachanotes_v35_to_v36_note_folders.sql` | Create folder and membership tables, indexes, constraints, and bump schema version atomically. |
| `tldw_chatbook/DB/ChaChaNotes_DB.py` | Register and verify the v36 migration only. |
| `tldw_chatbook/Notes/note_folder_models.py` | Frozen normalized folder, membership, page, capability, mutation, and restore-review values. |
| `tldw_chatbook/Notes/note_folder_repository.py` | Sole local SQL owner for hierarchy, optimistic subtree mutations, memberships, bounded pages, and restore review. |
| `tldw_chatbook/Notes/notes_scope_service.py` | Scope-aware typed facade that delegates local operations and reports unsupported scopes. |
| `tldw_chatbook/app.py` | Compose one repository into the existing `NotesScopeService`. |
| `Tests/DB/test_chachanotes_note_folders_migration.py` | Fresh/prior migration, constraints, rollback, and idempotence evidence. |
| `Tests/Notes/test_note_folder_models.py` | Name/path normalization and immutable-type evidence. |
| `Tests/Notes/test_note_folder_repository.py` | Mutation, collision, ownership, paging, query-bound, and restore-review evidence. |
| `Tests/Notes/test_notes_scope_service_folders.py` | Typed local routing and explicit remote capability behavior. |
| `Tests/Sync_Interop/test_notes_outbox_producer.py` | Contract ratchet that folder data does not enter Sync-v2 M1 note payloads. |
| `tldw_chatbook/UI/Tools_Settings_Window.py` | Inspect only: its existing ChaChaNotes backup remains the owner of the new logical tables. |

### Task 1: Add the ChaChaNotes v36 folder schema

**Files:**
- Create: `tldw_chatbook/DB/migrations/chachanotes_v35_to_v36_note_folders.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:166`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:4836-4883`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:5000-5045`
- Create: `Tests/DB/test_chachanotes_note_folders_migration.py`

- [ ] **Step 1: Write failing fresh-schema and v35 migration tests**

Create the test module with a `_schema_version` helper and these tests:

```python
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


EXPECTED_FOLDER_TABLES = {"note_folders", "note_folder_memberships"}


def _schema_version(db: CharactersRAGDB) -> int:
    row = db.get_connection().execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (db._SCHEMA_NAME,),
    ).fetchone()
    return int(row["version"])


def _table_names(db: CharactersRAGDB) -> set[str]:
    rows = db.get_connection().execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {str(row["name"]) for row in rows}


def _seed_v35(path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    with monkeypatch.context() as v35:
        v35.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 35)
        db = CharactersRAGDB(path, client_id="v35-seed")
        note_id = db.add_note("Existing", "Body")
        assert _schema_version(db) == 35
        db.close_connection()
    return str(note_id)


def test_fresh_database_has_v36_folder_schema(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="fresh")

    assert _schema_version(db) == 36
    assert EXPECTED_FOLDER_TABLES <= _table_names(db)


def test_v35_database_migrates_without_assigning_existing_notes(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "v35.db"
    note_id = _seed_v35(path, monkeypatch)

    migrated = CharactersRAGDB(path, client_id="v36-open")
    count = migrated.get_connection().execute(
        "SELECT COUNT(*) AS count FROM note_folder_memberships"
    ).fetchone()["count"]

    assert _schema_version(migrated) == 36
    assert migrated.get_note_by_id(note_id) is not None
    assert count == 0
```

- [ ] **Step 2: Run the tests and confirm the expected red state**

Run: `pytest Tests/DB/test_chachanotes_note_folders_migration.py -q`

Expected: FAIL because current schema version is 35 and the two tables do not
exist.

- [ ] **Step 3: Add the complete v35→v36 SQL migration**

Create `chachanotes_v35_to_v36_note_folders.sql` with:

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE note_folders(
  id              TEXT PRIMARY KEY,
  parent_id       TEXT REFERENCES note_folders(id),
  name            TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  path            TEXT NOT NULL,
  normalized_path TEXT NOT NULL,
  version         INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
  deleted         INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0, 1)),
  created_at      TEXT NOT NULL,
  modified_at     TEXT NOT NULL,
  CHECK(parent_id IS NULL OR parent_id <> id)
);

CREATE UNIQUE INDEX uq_note_folders_active_normalized_path
  ON note_folders(normalized_path) WHERE deleted = 0;
CREATE INDEX idx_note_folders_active_parent
  ON note_folders(parent_id, normalized_name) WHERE deleted = 0;

CREATE TABLE note_folder_memberships(
  id              TEXT PRIMARY KEY,
  folder_id       TEXT NOT NULL REFERENCES note_folders(id),
  note_id         TEXT NOT NULL REFERENCES notes(id),
  ownership       TEXT NOT NULL CHECK(ownership IN ('manual', 'managed')),
  owner_id        TEXT NOT NULL DEFAULT '',
  owner_active    INTEGER NOT NULL DEFAULT 1 CHECK(owner_active IN (0, 1)),
  version         INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
  deleted         INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0, 1)),
  created_at      TEXT NOT NULL,
  modified_at     TEXT NOT NULL,
  CHECK(
    (ownership = 'manual' AND owner_id = '' AND owner_active = 1) OR
    (ownership = 'managed' AND length(owner_id) > 0)
  )
);

CREATE UNIQUE INDEX uq_note_folder_memberships_active_owner
  ON note_folder_memberships(folder_id, note_id, ownership, owner_id)
  WHERE deleted = 0;
CREATE INDEX idx_note_folder_memberships_active_folder
  ON note_folder_memberships(folder_id, note_id) WHERE deleted = 0;
CREATE INDEX idx_note_folder_memberships_active_note
  ON note_folder_memberships(note_id, folder_id) WHERE deleted = 0;
CREATE INDEX idx_note_folder_memberships_restore_review
  ON note_folder_memberships(owner_active, owner_id)
  WHERE deleted = 0 AND ownership = 'managed';

UPDATE db_schema_version
   SET version = 36
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 35;
```

- [ ] **Step 4: Register the migration through the existing atomic pattern**

Set `_CURRENT_SCHEMA_VERSION = 36`. Add `_migrate_from_v35_to_v36` beside the v34
migration. It must:

1. reject every starting version except 35;
2. read the migration file as UTF-8;
3. execute each complete SQLite statement through the cursor returned by
   `self.transaction()`—never `Connection.executescript`;
4. reject trailing incomplete SQL;
5. verify the stored version is exactly 36 before returning; and
6. wrap I/O, SQLite, and schema errors as `SchemaError` naming V35→V36.

Register `35: self._migrate_from_v35_to_v36` in `migration_steps`.

- [ ] **Step 5: Add constraint, idempotence, and rollback tests**

Use direct parameterized inserts to prove the database rejects:

- two active folders with the same `normalized_path`;
- a folder whose parent is itself;
- a managed membership with an empty `owner_id`; and
- a manual membership with a non-empty `owner_id`.

Open an already-v36 database twice and assert the schema and row counts are
unchanged. Patch `Path.read_text` only for this migration path to return a script
that creates `note_folders` and then executes invalid SQL; assert initialization
raises `SchemaError`, the stored version remains 35, and neither new table remains.

- [ ] **Step 6: Run migration coverage**

Run:

```bash
pytest Tests/DB/test_chachanotes_note_folders_migration.py \
  Tests/DB/test_chachanotes_console_context_memory_migration.py \
  Tests/DB/test_chachanotes_world_book_regex_migration.py -q
```

Expected: PASS. Replace old hard-coded `35` assertions only when they mean “the
current version”; retain fixed numbers for deliberately seeded historical schemas.
Use `rg -n "== 35|version = 35|version: 35" Tests tldw_chatbook` to perform the
complete current-version census before running the broader DB suite.

- [ ] **Step 7: Commit the schema slice**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/DB/migrations/chachanotes_v35_to_v36_note_folders.sql \
  Tests/DB/test_chachanotes_note_folders_migration.py \
  Tests/DB/test_chachanotes_console_context_memory_migration.py \
  Tests/DB/test_chachanotes_world_book_regex_migration.py
git commit -m "feat(notes): add local folder schema"
```

### Task 2: Define normalized folder contracts

**Files:**
- Create: `tldw_chatbook/Notes/note_folder_models.py`
- Create: `Tests/Notes/test_note_folder_models.py`

- [ ] **Step 1: Write failing normalization tests**

```python
import pytest

from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    FolderValidationError,
    NormalizedFolderName,
    join_normalized_folder_path,
    normalize_folder_name,
)


def test_normalize_folder_name_preserves_display_and_keys_unicode() -> None:
    assert normalize_folder_name("  Résumé  ") == NormalizedFolderName(
        display="Résumé", key="résumé"
    )


@pytest.mark.parametrize("name", ["", "  ", ".", "..", "a/b", "a\\b", "a\x00b"])
def test_normalize_folder_name_rejects_invalid_segments(name: str) -> None:
    with pytest.raises(FolderValidationError):
        normalize_folder_name(name)


def test_normalized_path_has_one_leading_separator() -> None:
    assert join_normalized_folder_path("", "work") == "/work"
    assert join_normalized_folder_path("/work", "plans") == "/work/plans"


def test_note_placement_identity_includes_folder() -> None:
    assert FolderPlacementId.note("f-1", "n-1") != FolderPlacementId.note(
        "f-2", "n-1"
    )
```

- [ ] **Step 2: Run the tests and confirm the expected red state**

Run: `pytest Tests/Notes/test_note_folder_models.py -q`

Expected: FAIL because `note_folder_models` does not exist.

- [ ] **Step 3: Implement immutable models and normalization**

Create the following public contract, with Google-style docstrings on public APIs:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping
import unicodedata


FolderOwnership = Literal["manual", "managed"]
FolderCapabilityName = Literal[
    "list", "create", "rename", "move", "delete", "restore", "membership"
]


class FolderValidationError(ValueError):
    """Folder input cannot be represented safely."""


class FolderCollisionError(RuntimeError):
    """A requested active folder path already exists."""


class FolderConflictError(RuntimeError):
    """A folder or membership optimistic version is stale."""


class FolderCapabilityError(RuntimeError):
    def __init__(self, *, reason_code: str, user_message: str) -> None:
        super().__init__(user_message)
        self.reason_code = reason_code
        self.user_message = user_message


@dataclass(frozen=True)
class NormalizedFolderName:
    display: str
    key: str


@dataclass(frozen=True)
class NoteFolder:
    folder_id: str
    parent_id: str | None
    name: str
    path: str
    normalized_path: str
    version: int
    deleted: bool


@dataclass(frozen=True)
class NoteFolderMembership:
    membership_id: str
    folder_id: str
    note_id: str
    ownership: FolderOwnership
    owner_id: str
    owner_active: bool
    version: int


@dataclass(frozen=True)
class NoteFolderCapability:
    operation: FolderCapabilityName
    supported: bool
    reason_code: str = ""
    user_message: str = ""


@dataclass(frozen=True)
class NoteFolderPage:
    folders: tuple[NoteFolder, ...]
    memberships: tuple[NoteFolderMembership, ...]
    notes: tuple[Mapping[str, Any], ...]
    total_folders: int
    total_notes: int
    next_offset: int | None


@dataclass(frozen=True)
class FolderMutationResult:
    folder: NoteFolder
    affected_folder_ids: tuple[str, ...]


@dataclass(frozen=True)
class RestoredManagedMembershipReview:
    owner_id: str
    membership_ids: tuple[str, ...]
    note_count: int
    folder_count: int


class FolderPlacementId:
    @staticmethod
    def folder(folder_id: str) -> str:
        return f"folder:{folder_id}"

    @staticmethod
    def note(folder_id: str, note_id: str) -> str:
        return f"note:{folder_id}:{note_id}"

    @staticmethod
    def unfiled(note_id: str) -> str:
        return f"unfiled:{note_id}"


def normalize_folder_name(name: str) -> NormalizedFolderName:
    if not isinstance(name, str):
        raise FolderValidationError("Folder name must be text.")
    display = name.strip()
    if not display or display in {".", ".."}:
        raise FolderValidationError("Folder name cannot be empty, '.' or '..'.")
    if len(display) > 255:
        raise FolderValidationError("Folder name must be 255 characters or fewer.")
    if any(character in display for character in ("/", "\\", "\x00")):
        raise FolderValidationError("Folder name cannot contain a path separator or NUL.")
    key = unicodedata.normalize("NFKC", display).casefold()
    return NormalizedFolderName(display=display, key=key)


def join_normalized_folder_path(parent_path: str, child_key: str) -> str:
    parent = parent_path.rstrip("/")
    return f"{parent}/{child_key}" if parent else f"/{child_key}"
```

- [ ] **Step 4: Add property coverage**

Use Hypothesis text strategies excluding NUL and separators. Assert normalization
is idempotent, keys equal `NFKC(display).casefold()`, and path joining never creates
`//`, `/.`, or `/..`. Include composed/decomposed accents and German sharp-S fixed
fixtures.

- [ ] **Step 5: Run and commit the model slice**

Run: `pytest Tests/Notes/test_note_folder_models.py -q`

Expected: PASS.

```bash
git add tldw_chatbook/Notes/note_folder_models.py \
  Tests/Notes/test_note_folder_models.py
git commit -m "feat(notes): define folder contracts"
```

### Task 3: Implement folder hierarchy mutations

**Files:**
- Create: `tldw_chatbook/Notes/note_folder_repository.py`
- Create: `Tests/Notes/test_note_folder_repository.py`

- [ ] **Step 1: Write failing create, list, and collision tests**

Use a real `CharactersRAGDB` fixture and this concrete behavior:

```python
def test_create_and_list_nested_folders(repository) -> None:
    work = repository.create_folder(name="Work", parent_id=None)
    plans = repository.create_folder(name="Plans", parent_id=work.folder_id)

    page = repository.list_children(parent_id=work.folder_id, limit=50, offset=0)

    assert page.folders == (plans,)
    assert plans.path == "/Work/Plans"
    assert plans.normalized_path == "/work/plans"


def test_create_rejects_unicode_equivalent_active_path(repository) -> None:
    repository.create_folder(name="Résumé", parent_id=None)

    with pytest.raises(FolderCollisionError):
        repository.create_folder(name="re\u0301sume\u0301", parent_id=None)
```

- [ ] **Step 2: Run the tests and confirm the expected red state**

Run: `pytest Tests/Notes/test_note_folder_repository.py -q`

Expected: FAIL because `LocalNoteFolderRepository` does not exist.

- [ ] **Step 3: Implement repository construction, row mapping, creation, and reads**

Create `LocalNoteFolderRepository(db: CharactersRAGDB)`. Every public method uses
the injected DB and parameterized SQL; it never constructs another DB handle.
Implement these exact methods:

- `create_folder(*, name, parent_id) -> NoteFolder`
- `get_folder(folder_id, *, include_deleted=False) -> NoteFolder | None`
- `list_children(*, parent_id, limit, offset) -> NoteFolderPage`
- `load_tree_batch(*, expanded_folder_ids, note_limit) -> NoteFolderPage`

Creation normalizes the name, reads an active parent when supplied, constructs both
display and normalized paths, inserts a UUID and one UTC timestamp, then reads the
inserted row back in the same transaction. Convert the partial unique-index error
to `FolderCollisionError` without exposing raw SQL.

Validate `1 <= limit <= 500`, `offset >= 0`, and `1 <= note_limit <= 1000`.
`load_tree_batch` must issue one folder query, one membership query, and one joined
notes query for all requested folder IDs. An empty requested-ID set loads roots and
unfiled notes. The membership query returns active rows for both active and inactive
owners so restore-review placements remain projectable. Unfiled means an active note
for which no active, owner-active membership to an active folder exists; a note may
therefore be Unfiled while an inactive restored placement also remains visible to
the later navigator.

- [ ] **Step 4: Write failing optimistic subtree tests**

Add five concrete tests:

1. Rename `Work` to `Projects` and assert `Work/Plans` becomes
   `Projects/Plans`, both versions increment once, and IDs remain stable.
2. Try moving `Work` beneath `Plans`; assert `FolderValidationError` and unchanged
   rows.
3. Create a destination collision, attempt a move, and assert every path and
   version is unchanged.
4. Rename with a stale version and assert `FolderConflictError`.
5. Soft-delete and restore a subtree; assert memberships survive and no note row's
   `deleted` or `version` field changes.

- [ ] **Step 5: Implement optimistic rename, move, delete, and restore**

Add:

- `rename_folder(folder_id, *, name, expected_version) -> FolderMutationResult`
- `move_folder(folder_id, *, parent_id, expected_version) -> FolderMutationResult`
- `soft_delete_folder(folder_id, *, expected_version) -> FolderMutationResult`
- `restore_folder(folder_id, *, expected_version) -> FolderMutationResult`

For rename/move, read the target and entire active subtree ordered by path length.
Reject a stale target, self/descendant parent, missing parent, and every resulting
active-path collision before the first update. Compute each new display and
normalized path by replacing only the validated subtree prefix. Update parent-first,
increment every affected folder version once, and use one operation timestamp.

Soft-delete marks the complete subtree deleted without touching memberships or
notes and increments each affected folder version once. Restore validates that all
resulting active paths are free, then restores parent-first and increments each
affected folder version once. A parent outside the restored subtree must already be
active.

- [ ] **Step 6: Mutation-check the two destructive guards**

Temporarily remove the descendant-parent rejection and confirm the cycle test fails.
Restore it. Temporarily skip collision preflight and confirm the rollback test fails
or raises the database uniqueness error instead of the typed error. Restore it.

- [ ] **Step 7: Run and commit hierarchy mutations**

Run:

```bash
pytest Tests/Notes/test_note_folder_models.py \
  Tests/Notes/test_note_folder_repository.py -q -k "folder and not membership"
```

Expected: PASS.

```bash
git add tldw_chatbook/Notes/note_folder_repository.py \
  Tests/Notes/test_note_folder_repository.py
git commit -m "feat(notes): add folder hierarchy repository"
```

### Task 4: Add ownership-aware memberships, paging, and restore review

**Files:**
- Modify: `tldw_chatbook/Notes/note_folder_repository.py`
- Modify: `Tests/Notes/test_note_folder_repository.py`

- [ ] **Step 1: Write failing manual and managed membership tests**

Use three folders, two notes, and two owner IDs. Prove:

```python
def test_managed_reconcile_does_not_remove_manual_membership(repository, seeded) -> None:
    folder, note = seeded.folder, seeded.note
    manual = repository.attach_manual(folder_id=folder.folder_id, note_id=note)

    repository.reconcile_managed(
        owner_id="root-a", desired=((folder.folder_id, note),)
    )
    repository.reconcile_managed(owner_id="root-a", desired=())

    active = repository.list_memberships(note_ids=(note,), include_inactive=True)
    assert [item.membership_id for item in active] == [manual.membership_id]
    assert active[0].ownership == "manual"


def test_remove_one_owner_leaves_other_owner_and_note(repository, seeded) -> None:
    folder, note = seeded.folder, seeded.note
    repository.reconcile_managed(
        owner_id="root-a", desired=((folder.folder_id, note),)
    )
    repository.reconcile_managed(
        owner_id="root-b", desired=((folder.folder_id, note),)
    )

    assert repository.remove_owner_memberships(owner_id="root-a") == 1
    remaining = repository.list_memberships(note_ids=(note,), include_inactive=True)

    assert {item.owner_id for item in remaining} == {"root-b"}
    assert repository.db.get_note_by_id(note) is not None
```

Also test idempotent manual attach, stale manual detach, inactive-owner grouping,
manual conversion, removal, and conversion when an equivalent manual membership
already exists.

- [ ] **Step 2: Run the membership tests and confirm the expected red state**

Run: `pytest Tests/Notes/test_note_folder_repository.py -q -k membership`

Expected: FAIL because membership methods do not exist.

- [ ] **Step 3: Implement membership operations**

Add these exact methods:

- `attach_manual(*, folder_id, note_id) -> NoteFolderMembership`
- `detach_manual(*, folder_id, note_id, expected_version) -> bool`
- `list_memberships(*, note_ids, include_inactive=False) -> tuple[NoteFolderMembership, ...]`
- `reconcile_managed(*, owner_id, desired) -> tuple[NoteFolderMembership, ...]`
- `convert_owner_to_manual(*, owner_id) -> int`
- `remove_owner_memberships(*, owner_id) -> int`
- `mark_unknown_owners_inactive(*, active_owner_ids) -> int`
- `list_restore_reviews() -> tuple[RestoredManagedMembershipReview, ...]`

Attach validates an active folder and active note, revives an exact soft-deleted row
when present, choosing the most recently modified match and leaving older history
deleted, and otherwise inserts. `detach_manual` can target only ownership
`manual` with empty owner and must match `expected_version`.

`reconcile_managed` normalizes the desired pairs to a set, validates every folder
and note before changing rows, then diffs only memberships whose `owner_id` matches
the caller. Other managed owners and all manual rows are invisible to its removal
set. Revived and inserted rows become `owner_active = 1`.

Conversion inserts or revives manual rows first and soft-deletes that owner's
managed rows in the same transaction. Removal soft-deletes only that owner's rows.
Both operations return the number of managed rows changed and are idempotent.

- [ ] **Step 4: Pin bounded bulk reads and constant query shape**

Use `connection.set_trace_callback` around `load_tree_batch`. Compare a fixture with
10 note placements against one with 500. Assert the SELECT count is identical and
does not exceed four: folders, memberships, joined notes, and totals. Assert no SQL
statement contains one query per note ID.

Add paging tests for 0, exact-limit, and limit-plus-one children/notes. The page's
`next_offset` is `None` at the end and otherwise `offset + returned_count`.

- [ ] **Step 5: Add backup/restore ownership evidence**

Create folders, manual memberships, and a managed membership in a file-backed
ChaChaNotes database. Use `sqlite3.Connection.backup`—the database-level mechanism
used by Settings—to copy it to another path. Reopen through `CharactersRAGDB` and
assert logical rows survive.

Call `mark_unknown_owners_inactive(active_owner_ids=())`; assert
`list_restore_reviews()` groups the managed row under its owner, active page reads
exclude it, conversion makes its placement manual, and neither conversion nor
removal changes the note row. Inspect `SETTINGS_DATABASES` and assert no separate
sync-state database was added.

- [ ] **Step 6: Run and commit membership/recovery work**

Run:

```bash
pytest Tests/Notes/test_note_folder_repository.py \
  Tests/DB/test_chachanotes_note_folders_migration.py -q
```

Expected: PASS.

```bash
git add tldw_chatbook/Notes/note_folder_repository.py \
  Tests/Notes/test_note_folder_repository.py \
  Tests/DB/test_chachanotes_note_folders_migration.py
git commit -m "feat(notes): add owned folder memberships"
```

### Task 5: Route folders through the normalized Notes service

**Files:**
- Modify: `tldw_chatbook/Notes/notes_scope_service.py`
- Modify: `tldw_chatbook/app.py:5180-5192`
- Create: `Tests/Notes/test_notes_scope_service_folders.py`
- Modify: `Tests/Notes/test_notes_scope_service.py`
- Modify: `Tests/Sync_Interop/test_notes_outbox_producer.py`

- [ ] **Step 1: Write failing local routing and capability tests**

```python
@pytest.mark.asyncio
async def test_list_folder_children_routes_to_local_repository() -> None:
    repository = RecordingFolderRepository()
    service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
        folder_repository=repository,
    )

    await service.list_note_folder_children(
        scope=ScopeType.LOCAL_NOTE,
        parent_id=None,
        limit=50,
        offset=0,
        user_id="local-user",
    )

    assert repository.calls == [("list_children", None, 50, 0)]


def test_server_folder_capabilities_are_honestly_unsupported() -> None:
    service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
    )

    capabilities = service.note_folder_capabilities(scope=ScopeType.SERVER_NOTE)

    assert capabilities
    assert all(not item.supported for item in capabilities)
    assert {item.reason_code for item in capabilities} == {"server_contract_missing"}
```

The recording repository implements only the invoked method and appends
`("list_children", parent_id, limit, offset)` before returning an empty
`NoteFolderPage`.

- [ ] **Step 2: Run the tests and confirm the expected red state**

Run: `pytest Tests/Notes/test_notes_scope_service_folders.py -q`

Expected: FAIL because the constructor and folder methods do not exist.

- [ ] **Step 3: Implement the scope-aware facade**

Add `folder_repository: Any = None` to `NotesScopeService.__init__` and store it.
Add these public methods:

- `note_folder_capabilities`
- `list_note_folder_children`
- `load_note_folder_tree_batch`
- `create_note_folder`
- `rename_note_folder`
- `move_note_folder`
- `delete_note_folder`
- `restore_note_folder`
- `attach_note_to_folder`
- `detach_note_from_folder`
- `convert_note_folder_owner_to_manual`
- `remove_note_folder_owner_memberships`
- `list_note_folder_restore_reviews`

Local methods require `user_id`, enforce the corresponding existing local action
ID (`notes.list/create/update/delete.local`), require the injected repository, and
delegate without changing payload shapes. Folder reads use `notes.list.local`;
membership attach/detach and folder rename/move use `notes.update.local`.

Server methods return a capability row for each operation with
`reason_code="server_contract_missing"`; mutation attempts raise
`FolderCapabilityError` carrying the same reason and upgrade guidance. Workspace
folder operations use `reason_code="scope_not_supported"`. Never fall back to flat
note writes or raw SQL.

- [ ] **Step 4: Compose one repository in the app**

Import `LocalNoteFolderRepository` and construct it from the existing
`self.chachanotes_db` immediately before `NotesScopeService` construction:

```python
folder_repository = (
    LocalNoteFolderRepository(self.chachanotes_db)
    if self.chachanotes_db is not None
    else None
)
self.notes_scope_service = NotesScopeService(
    local_notes_service=self.notes_service,
    server_service=self.server_notes_workspace_service,
    policy_enforcer=self.service_policy_enforcer,
    sync_scope_service=getattr(self, "sync_scope_service", None),
    folder_repository=folder_repository,
)
```

Do not instantiate a repository or `CharactersRAGDB` per service call.

- [ ] **Step 5: Pin policy and scope behavior**

Add parameterized tests proving every local method enforces its stated action ID,
missing local repository raises `FolderCapabilityError(reason_code="local_store_missing")`,
server mutations make no server-service call, and workspace calls make no local or
server call.

- [ ] **Step 6: Pin the Sync-v2 M1 boundary**

Extend `test_notes_outbox_producer.py` so a normal local save still produces a
payload whose keys contain none of `folder`, `membership`, `owner_id`, or `binding`.
In the scope-folder tests, inject a recording Sync-v2 producer, perform create,
rename, membership attach, and delete-folder operations, and assert both its upsert
and delete lists remain empty.

- [ ] **Step 7: Run and commit the service slice**

Run:

```bash
pytest Tests/Notes/test_notes_scope_service_folders.py \
  Tests/Notes/test_notes_scope_service.py \
  Tests/Sync_Interop/test_notes_outbox_producer.py -q
```

Expected: PASS.

```bash
git add tldw_chatbook/Notes/notes_scope_service.py tldw_chatbook/app.py \
  Tests/Notes/test_notes_scope_service_folders.py \
  Tests/Notes/test_notes_scope_service.py \
  Tests/Sync_Interop/test_notes_outbox_producer.py
git commit -m "feat(notes): route local folder operations"
```

### Task 6: Verify performance, compatibility, and task closeout

**Files:**
- Modify: `Tests/Notes/test_note_folder_repository.py`
- Modify: `backlog/tasks/task-15705 - Add-local-Database-Note-folder-data-foundation.md`
- Modify only if an incident generalizes: `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Measure representative bulk behavior**

Populate a scratch database with 5,000 notes, 500 folders, and 10,000 memberships.
Measure root loading and one 500-note expanded batch with `time.perf_counter`, while
also recording SQL statement counts. Record the observed fixture size, elapsed
times, and SELECT counts in the Backlog Implementation Notes. Do not invent a
latency threshold after seeing the result; the required invariant is bounded pages
and constant query shape.

- [ ] **Step 2: Run static and focused verification**

```bash
python -m ruff check \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/Notes/note_folder_models.py \
  tldw_chatbook/Notes/note_folder_repository.py \
  tldw_chatbook/Notes/notes_scope_service.py \
  tldw_chatbook/app.py \
  Tests/DB/test_chachanotes_note_folders_migration.py \
  Tests/Notes/test_note_folder_models.py \
  Tests/Notes/test_note_folder_repository.py \
  Tests/Notes/test_notes_scope_service_folders.py
pytest Tests/DB/test_chachanotes_note_folders_migration.py \
  Tests/Notes/test_note_folder_models.py \
  Tests/Notes/test_note_folder_repository.py \
  Tests/Notes/test_notes_scope_service_folders.py \
  Tests/Notes/test_notes_scope_service.py \
  Tests/Sync_Interop/test_notes_outbox_producer.py -q
```

Expected: Ruff exits 0 and all focused tests pass.

- [ ] **Step 3: Run broader database and Notes regressions**

```bash
pytest Tests/DB/ Tests/Notes/ -q
```

Expected: PASS. If the repository baseline is red, rerun the exact failure against
the pre-change commit or an isolated clean worktree before classifying it as
unrelated; record command and counterfactual evidence.

- [ ] **Step 4: Self-review architectural boundaries**

Review the diff and prove:

- folder SQL exists only in the migration and `note_folder_repository.py`;
- no Textual module is imported by folder models or repository;
- no folder field appears in Sync-v2 M1 envelopes;
- no server mutation is attempted without the future capability;
- all subtree writes and managed reconciliation use one transaction;
- folder deletion never updates a note row;
- logical folders survive ChaChaNotes backup; and
- no device-private sync database or root path was introduced.

- [ ] **Step 5: Complete Backlog hygiene and commit closeout**

Check all seven acceptance criteria, add concise Implementation Notes describing
the approach, files, tests, measured query evidence, trade-offs, and ADR-059/073.
Set `TASK-15705` to Done only when every repository Definition of Done requirement
is satisfied. Add a lessons entry only if implementation produced a demonstrated,
generalizable incident.

```bash
git add "backlog/tasks/task-15705 - Add-local-Database-Note-folder-data-foundation.md" \
  backlog/docs/lessons-testing-evidence.md
git commit -m "docs(notes): close folder data foundation"
```

## Final implementation review checklist

- [ ] The v35→v36 migration is atomic, idempotent, and preserves existing notes
  without inventing memberships.
- [ ] Folder collisions and stale versions fail before partial subtree changes.
- [ ] Manual and owner-scoped managed memberships remain independently removable.
- [ ] Folder deletion, membership removal, and restore review never delete notes.
- [ ] Bulk reads are bounded and have constant query shape.
- [ ] Logical folders participate in ChaChaNotes backup; unmatched managed owners
  restore inactive for review.
- [ ] Server/workspace capability gaps are explicit and make no fallback mutation.
- [ ] No folder data appears in Sync-v2 M1 `notes.note` envelopes.
- [ ] Focused and broader verification pass, or unrelated baselines are documented
  with counterfactual evidence.
