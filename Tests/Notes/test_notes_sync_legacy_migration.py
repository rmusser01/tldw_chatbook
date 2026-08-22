from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest

from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
from tldw_chatbook.Notes.notes_sync_legacy import (
    LEGACY_MIGRATION_REPORT_LIMIT,
    LegacyMigrationReportEntry,
    LegacyNotesSyncMigrationError,
    LegacyNotesSyncMigrationResult,
    _bounded_report,
    authorize_legacy_candidate_activation,
    persist_legacy_notes_sync_migration,
    plan_legacy_notes_sync_migration,
    snapshot_legacy_notes_sync,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_reconciler import (
    BindingObservation,
    ReconciliationInput,
    ReconciliationPlan,
    plan_reconciliation,
)
from tldw_chatbook.Utils.sensitive_paths import resolve_sensitive_context


_LEGACY_SCHEMA = """
CREATE TABLE notes (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    file_path_on_disk TEXT,
    relative_file_path_on_disk TEXT,
    sync_root_folder TEXT,
    last_synced_disk_file_hash TEXT,
    last_synced_disk_file_mtime REAL,
    is_externally_synced INTEGER NOT NULL DEFAULT 0,
    sync_strategy TEXT,
    sync_excluded INTEGER NOT NULL DEFAULT 0,
    file_extension TEXT,
    deleted INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL DEFAULT ''
);
CREATE TABLE sync_sessions (
    session_id TEXT PRIMARY KEY,
    sync_root_folder TEXT NOT NULL,
    sync_direction TEXT NOT NULL,
    conflict_resolution TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    total_files INTEGER NOT NULL DEFAULT 0,
    processed_files INTEGER NOT NULL DEFAULT 0,
    conflicts_found INTEGER NOT NULL DEFAULT 0,
    errors_count INTEGER NOT NULL DEFAULT 0,
    client_id TEXT NOT NULL,
    summary TEXT
);
"""


def _legacy_connection(tmp_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(tmp_path / "legacy-notes.sqlite")
    connection.executescript(_LEGACY_SCHEMA)
    connection.commit()
    return connection


def _settings(root: Path | str | None = None) -> dict[str, object]:
    notes: dict[str, object] = {
        "auto_sync_enabled": True,
        "sync_on_close": True,
        "conflict_resolution": "disk_wins",
        "sync_direction": "disk_to_db",
    }
    if root is not None:
        notes["sync_directory"] = str(root)
    return {"notes": notes, "unrelated": {"preserved": True}}


def _add_note(
    connection: sqlite3.Connection,
    *,
    note_id: object,
    root: object,
    relative_path: object,
    file_path: object,
    version: object = 3,
    content_digest: object = "a" * 64,
    externally_synced: object = 1,
    strategy: object = "newer_wins",
) -> None:
    connection.execute(
        """
        INSERT INTO notes (
            id, version, file_path_on_disk, relative_file_path_on_disk,
            sync_root_folder, last_synced_disk_file_hash,
            last_synced_disk_file_mtime, is_externally_synced, sync_strategy,
            sync_excluded, file_extension, deleted, content
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '.md', 0, 'PRIVATE content')
        """,
        (
            note_id,
            version,
            os.fspath(file_path) if isinstance(file_path, os.PathLike) else file_path,
            relative_path,
            os.fspath(root) if isinstance(root, os.PathLike) else root,
            content_digest,
            1234.5,
            externally_synced,
            strategy,
        ),
    )
    connection.commit()


def _add_session(
    connection: sqlite3.Connection,
    *,
    session_id: object,
    root: object,
    direction: object = "db_to_disk",
    conflict: object = "db_wins",
) -> None:
    connection.execute(
        """
        INSERT INTO sync_sessions (
            session_id, sync_root_folder, sync_direction,
            conflict_resolution, started_at, status, client_id
        ) VALUES (?, ?, ?, ?, '2026-08-20T00:00:00Z', 'completed', 'legacy')
        """,
        (
            session_id,
            os.fspath(root) if isinstance(root, os.PathLike) else root,
            direction,
            conflict,
        ),
    )
    connection.commit()


def _snapshot_and_plan(
    connection: sqlite3.Connection,
    settings: Mapping[str, object],
    **kwargs: object,
):
    snapshot = snapshot_legacy_notes_sync(
        connection,
        settings,
        note_scope_id="local",
        **kwargs,
    )
    return snapshot, plan_legacy_notes_sync_migration(snapshot)


def _private_counts(store: NotesDeviceStateStore) -> dict[str, int]:
    queries = {
        "notes_sync_roots": "SELECT COUNT(*) FROM notes_sync_roots",
        "notes_sync_bindings": "SELECT COUNT(*) FROM notes_sync_bindings",
        "notes_sync_legacy_migrations": (
            "SELECT COUNT(*) FROM notes_sync_legacy_migrations"
        ),
        "notes_sync_operations": "SELECT COUNT(*) FROM notes_sync_operations",
        "notes_sync_recovery": "SELECT COUNT(*) FROM notes_sync_recovery",
        "notes_sync_store_settings": ("SELECT COUNT(*) FROM notes_sync_store_settings"),
        "import_sessions": "SELECT COUNT(*) FROM import_sessions",
    }
    with store.transaction() as connection:
        return {
            name: int(connection.execute(query).fetchone()[0])
            for name, query in queries.items()
        }


def test_multiple_config_row_and_session_roots_become_separate_paused_candidates(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    config_root = tmp_path / "config-root"
    row_root = tmp_path / "row-root"
    session_root = tmp_path / "session-root"
    for root in (config_root, row_root, session_root):
        root.mkdir()
    row_file = row_root / "item.md"
    row_file.write_bytes(b"row")
    _add_note(
        connection,
        note_id="note-row",
        root=row_root,
        relative_path="item.md",
        file_path=row_file,
    )
    _add_session(connection, session_id="session-only", root=session_root)

    _snapshot, plan = _snapshot_and_plan(connection, _settings(config_root))

    assert len(plan.roots) == 3
    assert {root.state for root in plan.roots} == {NotesSyncRootState.PAUSED}
    assert {root.direction for root in plan.roots} == {NotesSyncDirection.BIDIRECTIONAL}
    assert {root.last_status_code for root in plan.roots} == {
        "migration_review_required"
    }
    assert len(plan.bindings) == 1
    assert plan.bindings[0].state is NotesSyncBindingState.CANDIDATE
    assert {item.reason_code for item in plan.report} >= {
        "config_only_root",
        "row_only_root",
        "session_only_root",
        "legacy_policy_ignored",
    }
    assert plan.requires_fresh_dry_run is True
    assert plan.requires_explicit_activation is True
    connection.close()


def test_config_only_and_row_only_evidence_are_preserved_without_policy_inheritance(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    config_root = tmp_path / "config"
    row_root = tmp_path / "row"
    config_root.mkdir()
    row_root.mkdir()
    row_file = row_root / "row.md"
    row_file.write_text("row", encoding="utf-8")
    _add_note(
        connection,
        note_id="note-1",
        root=row_root,
        relative_path="row.md",
        file_path=row_file,
        strategy="disk_wins",
    )

    first_snapshot, first = _snapshot_and_plan(connection, _settings(config_root))
    changed_settings = _settings(config_root)
    changed_settings["notes"]["conflict_resolution"] = "db_wins"  # type: ignore[index]
    changed_settings["notes"]["sync_direction"] = "db_to_disk"  # type: ignore[index]
    second_snapshot, second = _snapshot_and_plan(connection, changed_settings)

    assert tuple(root.direction for root in first.roots) == tuple(
        root.direction for root in second.roots
    )
    assert first_snapshot.source_fingerprint != second_snapshot.source_fingerprint
    assert "disk_wins" not in repr(first)
    assert "db_wins" not in repr(second)
    assert "auto_sync" not in repr(first)
    connection.close()


def test_row_only_file_metadata_derives_a_safe_root_without_repairing_bad_paths(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "derived-root"
    nested = root / "folder"
    nested.mkdir(parents=True)
    note_file = nested / "note.md"
    note_file.write_bytes(b"note")
    _add_note(
        connection,
        note_id="row-only-note",
        root=None,
        relative_path="folder/note.md",
        file_path=note_file,
    )

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert len(plan.roots) == 1
    assert plan.roots[0].canonical_path == str(root.resolve())
    assert len(plan.bindings) == 1
    assert plan.bindings[0].normalized_relative_path == "folder/note.md"
    assert "row_only_root" in {item.reason_code for item in plan.report}
    connection.close()


def test_missing_and_overlapping_roots_are_reported_without_candidates(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    missing = tmp_path / "missing"
    _add_session(connection, session_id="parent", root=parent)
    _add_session(connection, session_id="child", root=child)

    _snapshot, plan = _snapshot_and_plan(connection, _settings(missing))

    assert plan.roots == ()
    assert plan.bindings == ()
    reasons = [item.reason_code for item in plan.report]
    assert "root_missing" in reasons
    assert reasons.count("root_overlap") == 2
    assert "deletion" not in " ".join(reasons)
    connection.close()


def test_duplicate_binding_paths_and_file_identities_are_not_arbitrarily_owned(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    same_path = root / "same.md"
    same_path.write_bytes(b"same")
    original = root / "original.md"
    alias = root / "alias.md"
    original.write_bytes(b"linked")
    os.link(original, alias)
    _add_note(
        connection,
        note_id="note-path-a",
        root=root,
        relative_path="same.md",
        file_path=same_path,
    )
    _add_note(
        connection,
        note_id="note-path-b",
        root=root,
        relative_path="same.md",
        file_path=same_path,
    )
    _add_note(
        connection,
        note_id="note-identity-a",
        root=root,
        relative_path="original.md",
        file_path=original,
    )
    _add_note(
        connection,
        note_id="note-identity-b",
        root=root,
        relative_path="alias.md",
        file_path=alias,
    )

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert len(plan.roots) == 1
    assert plan.bindings == ()
    assert {item.reason_code for item in plan.report} >= {
        "duplicate_binding_path",
        "duplicate_file_identity",
    }
    connection.close()


def test_distinct_rows_under_one_root_are_normal_evidence_not_duplicate_roots(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    for ordinal in range(2):
        note_file = root / f"note-{ordinal}.md"
        note_file.write_bytes(b"note")
        _add_note(
            connection,
            note_id=f"note-{ordinal}",
            root=root,
            relative_path=note_file.name,
            file_path=note_file,
        )

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert len(plan.roots) == 1
    assert len(plan.bindings) == 2
    assert "duplicate_root_evidence" not in {item.reason_code for item in plan.report}
    connection.close()


def test_out_of_root_and_unsafe_rows_are_skipped_without_deletion_inference(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_bytes(b"outside")
    target = root / "target.md"
    target.write_bytes(b"target")
    linked = root / "linked.md"
    os.link(target, linked)
    _add_note(
        connection,
        note_id="out-of-root",
        root=root,
        relative_path="inside.md",
        file_path=outside,
    )
    _add_note(
        connection,
        note_id="unsafe-link",
        root=root,
        relative_path="linked.md",
        file_path=linked,
    )

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert len(plan.roots) == 1
    assert plan.bindings == ()
    reasons = {item.reason_code for item in plan.report}
    assert "file_out_of_root" in reasons
    assert "unsafe_file_identity" in reasons
    assert not any("delet" in reason for reason in reasons)
    connection.close()


def test_unsafe_roots_and_invalid_values_fail_closed_with_bounded_reports(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    invalid_file = real_root / "invalid.md"
    invalid_file.write_bytes(b"invalid")
    _add_note(
        connection,
        note_id="bad/note/id",
        root=real_root,
        relative_path="../invalid.md",
        file_path=invalid_file,
        version=-1,
        content_digest="not-a-digest",
    )

    _snapshot, plan = _snapshot_and_plan(
        connection,
        {
            "notes": {
                "sync_directory": linked_root,
                "auto_sync_enabled": "yes",
                "sync_direction": [],
            }
        },
    )

    assert len(plan.roots) == 1
    assert plan.roots[0].state is NotesSyncRootState.PAUSED
    assert plan.bindings == ()
    assert {item.reason_code for item in plan.report} >= {
        "root_link_or_reparse",
        "invalid_note_evidence",
        "invalid_legacy_policy",
    }
    assert len(plan.report) <= 200
    assert "PRIVATE" not in repr(plan)
    assert str(real_root) not in repr(plan)
    connection.close()


def test_snapshot_is_read_only_and_does_not_retain_note_content(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "PRIVATE-root"
    root.mkdir()
    note_file = root / "note.md"
    note_file.write_bytes(b"PRIVATE file bytes")
    _add_note(
        connection,
        note_id="note-1",
        root=root,
        relative_path="note.md",
        file_path=note_file,
    )
    settings = _settings(root)
    before_dump = tuple(connection.iterdump())
    before_settings = repr(settings)
    before_file = note_file.read_bytes()
    before_mtime = note_file.stat().st_mtime_ns
    forbidden = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_DROP_TABLE,
    }
    connection.set_authorizer(
        lambda action, *_args: (
            sqlite3.SQLITE_DENY if action in forbidden else sqlite3.SQLITE_OK
        )
    )

    snapshot = snapshot_legacy_notes_sync(
        connection,
        settings,
        note_scope_id="local",
    )
    connection.set_authorizer(None)

    assert tuple(connection.iterdump()) == before_dump
    assert repr(settings) == before_settings
    assert note_file.read_bytes() == before_file
    assert note_file.stat().st_mtime_ns == before_mtime
    assert "PRIVATE file bytes" not in repr(snapshot)
    assert str(root) not in repr(snapshot)
    assert not hasattr(snapshot.roots[0], "raw_path")
    assert not hasattr(snapshot.notes[0], "raw_root")
    assert not hasattr(snapshot.notes[0], "raw_file_path")
    assert not hasattr(snapshot, "policy_values")
    connection.close()


def test_source_fingerprint_is_deterministic_and_changes_with_legacy_evidence(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    note_file = root / "note.md"
    note_file.write_bytes(b"note")
    _add_note(
        connection,
        note_id="note-1",
        root=root,
        relative_path="note.md",
        file_path=note_file,
    )

    first, first_plan = _snapshot_and_plan(connection, _settings(root))
    repeated, repeated_plan = _snapshot_and_plan(connection, _settings(root))
    connection.execute(
        "UPDATE notes SET last_synced_disk_file_mtime = 9999 WHERE id = 'note-1'"
    )
    connection.commit()
    changed, _changed_plan = _snapshot_and_plan(connection, _settings(root))

    assert first == repeated
    assert first_plan == repeated_plan
    assert first.source_fingerprint == repeated.source_fingerprint
    assert changed.source_fingerprint != first.source_fingerprint
    connection.close()


def test_report_is_bounded_and_marks_truncation(tmp_path: Path) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    for ordinal in range(LEGACY_MIGRATION_REPORT_LIMIT + 5):
        missing = root / f"missing-{ordinal}.md"
        _add_note(
            connection,
            note_id=f"note-{ordinal}",
            root=root,
            relative_path=missing.name,
            file_path=missing,
        )

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert len(plan.report) == LEGACY_MIGRATION_REPORT_LIMIT
    assert plan.report[-1].reason_code == "migration_report_truncated"
    assert plan.bindings == ()
    connection.close()


def test_report_marks_only_real_overflow_as_truncated() -> None:
    entries = tuple(
        LegacyMigrationReportEntry(
            "file_missing",
            binding_id=f"binding-{ordinal}",
        )
        for ordinal in range(LEGACY_MIGRATION_REPORT_LIMIT + 1)
    )

    exact = _bounded_report(entries[:LEGACY_MIGRATION_REPORT_LIMIT])
    overflow = _bounded_report(entries)

    assert exact == entries[:LEGACY_MIGRATION_REPORT_LIMIT]
    assert len(overflow) == LEGACY_MIGRATION_REPORT_LIMIT
    assert overflow[:-1] == entries[: LEGACY_MIGRATION_REPORT_LIMIT - 1]
    assert overflow[-1] == LegacyMigrationReportEntry("migration_report_truncated")


def test_migration_result_enforces_the_fixed_report_bound() -> None:
    entries = tuple(
        LegacyMigrationReportEntry(
            "file_missing",
            binding_id=f"binding-{ordinal}",
        )
        for ordinal in range(LEGACY_MIGRATION_REPORT_LIMIT + 1)
    )

    accepted = LegacyNotesSyncMigrationResult(
        False,
        0,
        0,
        entries[:LEGACY_MIGRATION_REPORT_LIMIT],
    )
    assert accepted.report == entries[:LEGACY_MIGRATION_REPORT_LIMIT]
    with pytest.raises(ValueError, match="report exceeds"):
        LegacyNotesSyncMigrationResult(False, 0, 0, entries)
    with pytest.raises(TypeError, match="report must be a tuple"):
        LegacyNotesSyncMigrationResult(False, 0, 0, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="report must be a tuple"):
        LegacyNotesSyncMigrationResult(False, 0, 0, (object(),))  # type: ignore[arg-type]


def test_persist_is_one_private_transaction_idempotent_and_inert(
    tmp_path: Path,
) -> None:
    legacy = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    note_file = root / "note.md"
    note_file.write_bytes(b"note")
    _add_note(
        legacy,
        note_id="note-1",
        root=root,
        relative_path="note.md",
        file_path=note_file,
    )
    snapshot, plan = _snapshot_and_plan(legacy, _settings(root))

    class CountingStore(NotesDeviceStateStore):
        transaction_count = 0

        @contextmanager
        def transaction(
            self, *, immediate: bool = False
        ) -> Iterator[sqlite3.Connection]:
            self.transaction_count += 1
            with super().transaction(immediate=immediate) as connection:
                yield connection

    store = CountingStore(tmp_path / "private-state.sqlite")
    first = persist_legacy_notes_sync_migration(store, plan)
    assert store.transaction_count == 1
    second = persist_legacy_notes_sync_migration(store, plan)
    assert store.transaction_count == 2

    assert first.already_migrated is False
    assert first.root_count == 1
    assert first.binding_count == 1
    assert second.already_migrated is True
    assert second.root_count == 0
    assert second.binding_count == 0
    assert first.report == plan.report
    assert second.report == plan.report
    assert len(repr(first)) < 160
    assert not any(item.reason_code in repr(first) for item in first.report)
    counts = _private_counts(store)
    assert counts == {
        "notes_sync_roots": 1,
        "notes_sync_bindings": 1,
        "notes_sync_legacy_migrations": 1,
        "notes_sync_operations": 0,
        "notes_sync_recovery": 0,
        "notes_sync_store_settings": 0,
        "import_sessions": 0,
    }
    with store.transaction() as connection:
        root_row = connection.execute(
            "SELECT state, logical_folder_id, cursor FROM notes_sync_roots"
        ).fetchone()
        binding_state = connection.execute(
            "SELECT state FROM notes_sync_bindings"
        ).fetchone()[0]
    assert root_row == ("paused", None, None)
    assert binding_state == "candidate"
    assert snapshot.source_fingerprint == plan.source_fingerprint
    assert note_file.read_bytes() == b"note"
    legacy.close()


def test_changed_source_fingerprint_reuses_existing_paused_candidates(
    tmp_path: Path,
) -> None:
    legacy = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    note_file = root / "note.md"
    note_file.write_bytes(b"note")
    _add_note(
        legacy,
        note_id="note-1",
        root=root,
        relative_path="note.md",
        file_path=note_file,
    )
    _first_snapshot, first_plan = _snapshot_and_plan(legacy, _settings(root))
    store = NotesDeviceStateStore(tmp_path / "private-state.sqlite")
    persist_legacy_notes_sync_migration(store, first_plan)
    legacy.execute(
        "UPDATE notes SET last_synced_disk_file_mtime = 9999 WHERE id = 'note-1'"
    )
    legacy.commit()
    _second_snapshot, second_plan = _snapshot_and_plan(legacy, _settings(root))

    result = persist_legacy_notes_sync_migration(store, second_plan)

    assert result.already_migrated is False
    assert result.root_count == 0
    assert result.binding_count == 0
    assert _private_counts(store)["notes_sync_roots"] == 1
    assert _private_counts(store)["notes_sync_bindings"] == 1
    assert _private_counts(store)["notes_sync_legacy_migrations"] == 2
    legacy.close()


def test_crash_between_private_writes_rolls_back_the_whole_migration(
    tmp_path: Path,
) -> None:
    legacy = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    note_file = root / "note.md"
    note_file.write_bytes(b"note")
    _add_note(
        legacy,
        note_id="note-1",
        root=root,
        relative_path="note.md",
        file_path=note_file,
    )
    _snapshot, plan = _snapshot_and_plan(legacy, _settings(root))

    class FailingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def execute(self, statement: str, parameters: tuple[object, ...] = ()):
            if "INSERT INTO notes_sync_bindings" in statement:
                raise RuntimeError("PRIVATE crash detail")
            return self._connection.execute(statement, parameters)

    class FailingStore(NotesDeviceStateStore):
        @contextmanager
        def transaction(self, *, immediate: bool = False):
            with super().transaction(immediate=immediate) as connection:
                yield FailingConnection(connection)

    store = FailingStore(tmp_path / "private-state.sqlite")
    with pytest.raises(LegacyNotesSyncMigrationError) as raised:
        persist_legacy_notes_sync_migration(store, plan)

    assert str(raised.value) == "legacy_migration_failed"
    assert raised.value.__cause__ is None
    assert "PRIVATE" not in repr(raised.value)
    assert _private_counts(store)["notes_sync_roots"] == 0
    assert _private_counts(store)["notes_sync_bindings"] == 0
    assert _private_counts(store)["notes_sync_legacy_migrations"] == 0
    legacy.close()


def test_fresh_task_19005_dry_run_and_explicit_activation_are_both_required(
    tmp_path: Path,
) -> None:
    legacy = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    _snapshot, migration = _snapshot_and_plan(legacy, _settings(root))
    root_id = migration.roots[0].root_id
    fresh = ReconciliationInput(
        root_id=root_id,
        direction=NotesSyncDirection.BIDIRECTIONAL,
        bindings=(),
        observation_generation=1,
        expected_generation=1,
    )
    reviewed = plan_reconciliation(fresh)

    with pytest.raises(ValueError, match="explicit_activation_required"):
        authorize_legacy_candidate_activation(
            root_id,
            dry_run=reviewed,
            fresh_observations=fresh,
            explicitly_approved=False,
        )

    stale = ReconciliationInput(
        root_id=root_id,
        direction=NotesSyncDirection.BIDIRECTIONAL,
        bindings=(),
        observation_generation=2,
        expected_generation=2,
    )
    with pytest.raises(ValueError, match="stale_review"):
        authorize_legacy_candidate_activation(
            root_id,
            dry_run=reviewed,
            fresh_observations=stale,
            explicitly_approved=True,
        )

    unavailable = ReconciliationInput(
        root_id=root_id,
        direction=NotesSyncDirection.BIDIRECTIONAL,
        bindings=(),
        observation_generation=1,
        expected_generation=1,
        root_available=False,
    )
    with pytest.raises(ValueError, match="complete_dry_run_required"):
        authorize_legacy_candidate_activation(
            root_id,
            dry_run=plan_reconciliation(unavailable),
            fresh_observations=unavailable,
            explicitly_approved=True,
        )

    authorization = authorize_legacy_candidate_activation(
        root_id,
        dry_run=reviewed,
        fresh_observations=fresh,
        explicitly_approved=True,
    )
    assert authorization.root_id == root_id
    assert authorization.observation_token == reviewed.observation_token
    assert authorization.direction is NotesSyncDirection.BIDIRECTIONAL
    assert "path" not in repr(authorization).lower()
    assert authorization.observation_token not in repr(authorization)
    legacy.close()


def test_activation_recomputes_the_exact_fresh_plan_body() -> None:
    fresh = ReconciliationInput(
        root_id="root-1",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        bindings=(
            BindingObservation(
                binding_id="binding-1",
                note_scope_id="local",
                note_id="note-1",
                baseline_file_digest="a" * 64,
                baseline_note_digest="b" * 64,
                baseline_identity_digest="c" * 64,
                baseline_relative_path="note.md",
                file_digest="d" * 64,
                note_digest="e" * 64,
                file_identity_digest="c" * 64,
                relative_path="note.md",
                note_version=1,
            ),
        ),
        observation_generation=1,
        expected_generation=1,
    )
    actual = plan_reconciliation(fresh)
    forged = ReconciliationPlan(
        root_id=fresh.root_id,
        observation_token=actual.observation_token,
        safe_actions=(),
        attention=(),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
    )

    with pytest.raises(ValueError, match="dry_run_plan_mismatch"):
        authorize_legacy_candidate_activation(
            fresh.root_id,
            dry_run=forged,
            fresh_observations=fresh,
            explicitly_approved=True,
        )


def test_folder_to_notes_activation_accepts_read_only_filesystem() -> None:
    fresh = ReconciliationInput(
        root_id="root-read-only",
        direction=NotesSyncDirection.FOLDER_TO_NOTES,
        bindings=(),
        observation_generation=1,
        expected_generation=1,
        write_capable=False,
    )
    reviewed = plan_reconciliation(fresh)

    authorization = authorize_legacy_candidate_activation(
        fresh.root_id,
        dry_run=reviewed,
        fresh_observations=fresh,
        explicitly_approved=True,
    )

    assert authorization.direction is NotesSyncDirection.FOLDER_TO_NOTES


def test_file_path_canonical_alias_is_recognized_as_root_descendant(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    note_file = root / "note.md"
    note_file.write_bytes(b"note")
    canonical = note_file.resolve()
    canonical_text = str(canonical)
    if not canonical_text.startswith("/private/var/"):
        pytest.skip("host has no /var to /private/var canonical alias")
    lexical_alias = Path(canonical_text.removeprefix("/private"))
    assert lexical_alias.samefile(canonical)
    _add_note(
        connection,
        note_id="note-1",
        root=root,
        relative_path="note.md",
        file_path=lexical_alias,
    )

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert len(plan.bindings) == 1
    assert "file_out_of_root" not in {item.reason_code for item in plan.report}
    connection.close()


def test_case_aliases_of_one_physical_root_collapse_to_one_candidate(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "MixedCase"
    root.mkdir()
    alias = tmp_path / "mixedcase"
    if not alias.exists():
        pytest.skip("filesystem is case-sensitive")
    _add_session(connection, session_id="first", root=root)
    _add_session(connection, session_id="second", root=alias)

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert len(plan.roots) == 1
    assert "root_overlap" not in {item.reason_code for item in plan.report}
    connection.close()


def test_case_alias_ancestor_descendant_roots_are_rejected_as_overlap(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    parent = tmp_path / "MixedCase"
    child = parent / "child"
    child.mkdir(parents=True)
    alias = tmp_path / "mixedcase"
    if not alias.exists():
        pytest.skip("filesystem is case-sensitive")
    _add_session(connection, session_id="parent", root=alias)
    _add_session(connection, session_id="child", root=child)

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert plan.roots == ()
    assert [item.reason_code for item in plan.report].count("root_overlap") == 2
    connection.close()


def test_existing_file_case_alias_is_recognized_by_filesystem_identity(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    note_file = root / "MixedCase.md"
    note_file.write_bytes(b"note")
    alias = root / "mixedcase.md"
    if not alias.exists():
        pytest.skip("filesystem is case-sensitive")
    _add_note(
        connection,
        note_id="note-1",
        root=root,
        relative_path=note_file.name,
        file_path=alias,
    )

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert len(plan.bindings) == 1
    assert "file_out_of_root" not in {item.reason_code for item in plan.report}
    connection.close()


def test_descendant_directory_symlink_is_unsafe_legacy_evidence(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "note.md"
    outside_file.write_bytes(b"outside")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    _add_note(
        connection,
        note_id="note-1",
        root=root,
        relative_path="linked/note.md",
        file_path=root / "linked" / "note.md",
    )

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert plan.bindings == ()
    assert "unsafe_file_identity" in {item.reason_code for item in plan.report}
    connection.close()


def test_actual_application_private_root_is_rejected_by_default(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    private_root = resolve_sensitive_context().user_data_dir
    assert private_root.is_dir()

    _snapshot, plan = _snapshot_and_plan(connection, _settings(private_root))

    assert plan.roots == ()
    assert "private_path_overlap" in {item.reason_code for item in plan.report}
    connection.close()


def test_actual_application_private_root_case_alias_is_rejected_by_identity(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    private_root = resolve_sensitive_context().user_data_dir
    assert private_root is not None and private_root.is_dir()
    alias = private_root.with_name(private_root.name.swapcase())
    if not alias.exists() or not alias.samefile(private_root):
        pytest.skip("filesystem is case-sensitive")

    _snapshot, plan = _snapshot_and_plan(connection, _settings(alias))

    assert plan.roots == ()
    assert "private_path_overlap" in {item.reason_code for item in plan.report}
    connection.close()


def test_private_identity_comparison_failure_rejects_without_disclosure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "PRIVATE-root"
    root.mkdir()
    monkeypatch.setattr(
        "tldw_chatbook.Notes.notes_sync_legacy._filesystem_paths_overlap",
        lambda *_args: (_ for _ in ()).throw(OSError("PRIVATE comparison detail")),
    )

    _snapshot, plan = _snapshot_and_plan(connection, _settings(root))

    assert plan.roots == ()
    assert "comparison_root_unavailable" in {item.reason_code for item in plan.report}
    assert "PRIVATE" not in repr(plan)
    connection.close()


def test_source_fingerprint_covers_file_presence_and_identity(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    note_file = root / "note.md"
    _add_note(
        connection,
        note_id="note-1",
        root=root,
        relative_path="note.md",
        file_path=note_file,
    )

    missing = snapshot_legacy_notes_sync(
        connection,
        {"notes": {}},
        note_scope_id="local",
    )
    note_file.write_bytes(b"now present")
    present = snapshot_legacy_notes_sync(
        connection,
        {"notes": {}},
        note_scope_id="local",
    )

    assert missing.source_fingerprint != present.source_fingerprint
    connection.close()


def test_missing_file_is_report_only_without_fabricated_binding(
    tmp_path: Path,
) -> None:
    connection = _legacy_connection(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    missing = root / "missing.md"
    _add_note(
        connection,
        note_id="note-1",
        root=root,
        relative_path="missing.md",
        file_path=missing,
    )

    _snapshot, plan = _snapshot_and_plan(connection, {"notes": {}})

    assert plan.bindings == ()
    assert "file_missing" in {item.reason_code for item in plan.report}
    connection.close()
