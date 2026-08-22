"""Read-only legacy Notes sync source and canonical digest contracts."""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier, local
from typing import Any

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.Notes import notes_sync_legacy_migration as legacy
from tldw_chatbook.Notes import notes_sync_state as sync_state
from tldw_chatbook.Notes.notes_sync_state import (
    MAX_SYNC_BINDINGS,
    MAX_SYNC_ROOTS,
    MigrationState,
    NotesSyncStateRepository,
    SyncStateCapacityError,
)
from tldw_chatbook.Notes.notes_sync_state_schema import notes_sync_state_transaction


_NOTE_FIELDS = (
    "id",
    "file_path_on_disk",
    "relative_file_path_on_disk",
    "sync_root_folder",
    "last_synced_disk_file_hash",
    "last_synced_disk_file_mtime",
    "is_externally_synced",
    "sync_strategy",
    "sync_excluded",
    "file_extension",
    "version",
    "deleted",
)
_CONFLICT_FIELDS = (
    "id",
    "session_id",
    "note_id",
    "file_path",
    "conflict_type",
    "db_content_hash",
    "disk_content_hash",
    "db_modified_time",
    "disk_modified_time",
    "resolution",
    "resolved_at",
)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(tmp_path / "notes.sqlite3", client_id="legacy-source")


def _add_note(db: CharactersRAGDB, note_id: str, **sync_values: object) -> None:
    assert db.add_note(note_id, "body", note_id=note_id) == note_id
    if not sync_values:
        return
    assignments = ", ".join(f"{name} = ?" for name in sync_values)
    with db.transaction() as connection:
        connection.execute(
            f"UPDATE notes SET {assignments} WHERE id = ?",  # noqa: S608 - fixed test keys
            (*sync_values.values(), note_id),
        )


def _add_conflict(
    db: CharactersRAGDB,
    *,
    conflict_id: int,
    resolution: str | None,
    disk_modified_time: float = 2.5,
) -> None:
    session_id = f"session-{conflict_id}"
    with db.transaction() as connection:
        connection.execute(
            """INSERT INTO sync_sessions (
                   id, session_id, sync_root_folder, sync_direction,
                   conflict_resolution, status, client_id
               ) VALUES (?, ?, ?, 'bidirectional', 'ask', 'completed', 'test')""",
            (conflict_id, session_id, f"root-{conflict_id}"),
        )
        connection.execute(
            """INSERT INTO sync_conflicts (
                   id, session_id, note_id, file_path, conflict_type,
                   db_content_hash, disk_content_hash, db_modified_time,
                   disk_modified_time, resolution, resolved_at
               ) VALUES (?, ?, NULL, ?, 'both_changed', 'db-hash', 'disk-hash',
                         '2026-08-22T00:00:00Z', ?, ?, NULL)""",
            (
                conflict_id,
                session_id,
                f"candidate-{conflict_id}.md",
                disk_modified_time,
                resolution,
            ),
        )


def _note_row(note_id: str, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": note_id,
        "file_path_on_disk": None,
        "relative_file_path_on_disk": None,
        "sync_root_folder": None,
        "last_synced_disk_file_hash": None,
        "last_synced_disk_file_mtime": None,
        "is_externally_synced": 0,
        "sync_strategy": None,
        "sync_excluded": 0,
        "file_extension": ".md",
        "version": 1,
        "deleted": 0,
    }
    row.update(changes)
    return row


def _conflict_row(conflict_id: int, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": conflict_id,
        "session_id": f"session-{conflict_id}",
        "note_id": None,
        "file_path": f"candidate-{conflict_id}.md",
        "conflict_type": "both_changed",
        "db_content_hash": None,
        "disk_content_hash": None,
        "db_modified_time": None,
        "disk_modified_time": None,
        "resolution": None,
        "resolved_at": None,
    }
    row.update(changes)
    return row


def test_legacy_source_reader_pins_every_note_predicate_term_and_exact_order(
    tmp_path: Path,
) -> None:
    db = _new_db(tmp_path)
    cases = (
        ("h-strategy", {"sync_strategy": "bidirectional"}),
        ("a-external", {"is_externally_synced": 1}),
        ("g-mtime", {"last_synced_disk_file_mtime": 1.25}),
        ("b-excluded", {"sync_excluded": 1}),
        ("f-hash", {"last_synced_disk_file_hash": "hash"}),
        ("d-relative", {"relative_file_path_on_disk": "relative.md"}),
        ("e-root", {"sync_root_folder": "legacy-root"}),
        ("c-file", {"file_path_on_disk": "legacy-file.md"}),
    )
    for note_id, values in cases:
        _add_note(db, note_id, **values)
    _add_note(db, "i-soft-deleted", sync_strategy="disk_to_db", deleted=1)
    _add_note(db, "z-unrelated")

    notes, conflicts = db.read_legacy_notes_sync_source_rows()

    assert isinstance(notes, tuple)
    assert isinstance(conflicts, tuple)
    assert [row["id"] for row in notes] == sorted(note_id for note_id, _ in cases) + [
        "i-soft-deleted"
    ]
    assert all(type(row) is dict for row in notes)
    assert all(tuple(row) == _NOTE_FIELDS for row in notes)
    assert conflicts == ()
    assert "content" not in notes[0]
    db.close()


def test_legacy_source_reader_pins_unresolved_conflict_predicate_and_order(
    tmp_path: Path,
) -> None:
    db = _new_db(tmp_path)
    for conflict_id, resolution in (
        (5, "merge"),
        (4, "skip"),
        (3, "use_disk"),
        (2, None),
        (1, "use_db"),
    ):
        _add_conflict(db, conflict_id=conflict_id, resolution=resolution)

    notes, conflicts = db.read_legacy_notes_sync_source_rows()

    assert notes == ()
    assert [row["id"] for row in conflicts] == [2, 4]
    assert all(tuple(row) == _CONFLICT_FIELDS for row in conflicts)
    db.close()


def test_legacy_source_reader_rejects_an_ambient_transaction_snapshot(
    tmp_path: Path,
) -> None:
    first = _new_db(tmp_path)
    second = CharactersRAGDB(first.db_path, client_id="legacy-source-second")
    _add_note(first, "before", sync_strategy="bidirectional")

    with first.transaction():
        with pytest.raises(CharactersRAGDBError, match="independent transaction"):
            first.read_legacy_notes_sync_source_rows()
        _add_note(second, "after", sync_strategy="bidirectional")

    assert [row["id"] for row in first.read_legacy_notes_sync_source_rows()[0]] == [
        "after",
        "before",
    ]
    first.close()
    second.close()


def test_consecutive_fresh_source_reads_observe_an_intervening_connection_commit(
    tmp_path: Path,
) -> None:
    first = _new_db(tmp_path)
    second = CharactersRAGDB(first.db_path, client_id="legacy-source-second")
    _add_note(first, "before", sync_strategy="bidirectional")

    before = first.read_legacy_notes_sync_source_rows()
    _add_note(second, "after", sync_strategy="bidirectional")
    after = first.read_legacy_notes_sync_source_rows()

    assert [row["id"] for row in before[0]] == ["before"]
    assert [row["id"] for row in after[0]] == ["after", "before"]
    first.close()
    second.close()


def test_unrelated_notes_and_resolved_conflicts_do_not_change_source_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    db = _new_db(tmp_path)
    _add_note(db, "relevant", sync_strategy="bidirectional")
    _add_conflict(db, conflict_id=1, resolution=None)
    before = legacy.capture_legacy_source(db)

    _add_note(db, "unrelated")
    for conflict_id, resolution in ((2, "use_db"), (3, "use_disk"), (4, "merge")):
        _add_conflict(db, conflict_id=conflict_id, resolution=resolution)
    after = legacy.capture_legacy_source(db)

    assert after.digest == before.digest
    assert after.source == before.source
    db.close()


def test_canonical_source_digest_is_stable_across_mapping_and_input_order() -> None:
    config = {
        "sync_directory": "Cafe\u0301/秘密",
        "sync_direction": "bidirectional",
        "sync_conflict_resolution": "newer_wins",
    }
    notes = [
        _note_row("z", sync_strategy="disk_to_db"),
        dict(reversed(tuple(_note_row("a", sync_strategy="db_to_disk").items()))),
    ]
    conflicts = [_conflict_row(9), dict(reversed(tuple(_conflict_row(2).items())))]

    first = legacy._source_revision(config, notes, conflicts)
    second = legacy._source_revision(
        dict(reversed(tuple(config.items()))),
        list(reversed(notes)),
        list(reversed(conflicts)),
    )

    assert first == second
    assert first["notes"][0]["id"] == "a"
    assert first["conflicts"][0]["id"] == 2
    assert legacy._canonical_digest(first) == legacy._canonical_digest(second)
    assert legacy._canonical_digest(first) == _digest(first)


def test_canonical_digest_preserves_unicode_and_json_scalar_types() -> None:
    assert legacy._canonical_digest("é") != legacy._canonical_digest("e\u0301")
    values = (None, False, 0, "0")
    assert len({legacy._canonical_digest(value) for value in values}) == len(values)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (None, None),
        (1.5, float.hex(1.5)),
        (-0.0, float.hex(-0.0)),
        (math.inf, "invalid_non_finite_real"),
        (-math.inf, "invalid_non_finite_real"),
        (math.nan, "invalid_non_finite_real"),
    ),
)
def test_real_value_uses_exact_finite_hex_and_nonfinite_marker(
    value: float | None,
    expected: str | None,
) -> None:
    assert legacy._real_value(value) == expected


def test_real_value_conversion_error_has_no_private_exception_chain() -> None:
    secret = "private-real-conversion-sentinel"

    class BrokenReal:
        def __float__(self) -> float:
            raise ValueError(secret)

    with pytest.raises(legacy.LegacyNotesSyncSourceError) as raised:
        legacy._real_value(BrokenReal())

    error: BaseException | None = raised.value
    while error is not None:
        assert secret not in str(error)
        error = error.__cause__ or error.__context__


def test_source_revision_encodes_real_fields_in_their_exact_canonical_shape() -> None:
    source = legacy._source_revision(
        {
            "sync_directory": "root",
            "sync_direction": "bidirectional",
            "sync_conflict_resolution": "newer_wins",
        },
        (
            _note_row("finite", last_synced_disk_file_mtime=1.5),
            _note_row("nonfinite", last_synced_disk_file_mtime=math.inf),
        ),
        (_conflict_row(1, disk_modified_time=-0.0),),
    )

    assert source["notes"][0]["last_synced_disk_file_mtime"] == float.hex(1.5)
    assert (
        source["notes"][1]["last_synced_disk_file_mtime"] == "invalid_non_finite_real"
    )
    assert source["conflicts"][0]["disk_modified_time"] == float.hex(-0.0)


def test_capture_source_uses_exact_missing_config_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "empty.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    db = _new_db(tmp_path)

    snapshot = legacy.capture_legacy_source(db)

    assert snapshot.source["config"] == {
        "sync_conflict_resolution": "newer_wins",
        "sync_direction": "bidirectional",
        "sync_directory": "~/Documents/Notes",
    }
    assert snapshot.digest == _digest(snapshot.source)
    assert "~/Documents/Notes" not in repr(snapshot)
    db.close()


def test_captured_source_projection_cannot_mutate_the_digest_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "empty.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    db = _new_db(tmp_path)
    snapshot = legacy.capture_legacy_source(db)

    exposed = snapshot.source
    exposed["config"]["sync_directory"] = "mutated"

    assert snapshot.source["config"]["sync_directory"] == "~/Documents/Notes"
    assert snapshot.digest == _digest(snapshot.source)
    db.close()


def test_snapshot_defensively_owns_constructor_source_and_immutable_storage() -> None:
    source = legacy._source_revision(
        {
            "sync_directory": "private/root",
            "sync_direction": "bidirectional",
            "sync_conflict_resolution": "newer_wins",
        },
        (),
        (),
    )
    snapshot = legacy.LegacyNotesSyncSourceSnapshot(source, _digest(source))

    source["config"]["sync_directory"] = "mutated"

    assert snapshot.source["config"]["sync_directory"] == "private/root"
    assert snapshot.digest == _digest(snapshot.source)
    with pytest.raises(FrozenInstanceError):
        snapshot._canonical_source = "mutated"  # type: ignore[misc]


def test_capture_source_prefers_new_conflict_key_and_preserves_exact_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[notes]
sync_directory = 0
sync_direction = false
conflict_resolution = "legacy"
sync_conflict_resolution = "new"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    db = _new_db(tmp_path)

    snapshot = legacy.capture_legacy_source(db)

    assert snapshot.source["config"] == {
        "sync_conflict_resolution": "new",
        "sync_direction": False,
        "sync_directory": 0,
    }
    db.close()


def test_capture_source_falls_back_to_legacy_conflict_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[notes]\nconflict_resolution = "legacy-only"\n', encoding="utf-8"
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    db = _new_db(tmp_path)

    snapshot = legacy.capture_legacy_source(db)

    assert snapshot.source["config"]["sync_conflict_resolution"] == "legacy-only"
    db.close()


@pytest.mark.parametrize(
    "key", ("sync_directory", "sync_direction", "sync_conflict_resolution")
)
@pytest.mark.parametrize("value", ("[]", "{}"))
def test_capture_source_rejects_non_scalar_config_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(f"[notes]\n{key} = {value}\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    db = _new_db(tmp_path)

    with pytest.raises(legacy.LegacyNotesSyncSourceError, match="invalid_config_type"):
        legacy.capture_legacy_source(db)

    db.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("disk_to_db", ("folder_to_notes", None)),
        ("folder_to_notes", ("folder_to_notes", None)),
        ("db_to_disk", ("notes_to_folder", None)),
        ("notes_to_folder", ("notes_to_folder", None)),
        ("bidirectional", ("bidirectional", None)),
        (None, ("unspecified", "legacy_direction_invalid")),
        (False, ("unspecified", "legacy_direction_invalid")),
        ("unknown", ("unspecified", "legacy_direction_invalid")),
    ),
)
def test_legacy_direction_aliases_are_exact(
    value: object, expected: tuple[str, str | None]
) -> None:
    assert legacy.map_legacy_direction(value) == expected


def test_all_locator_shapes_match_independent_canonical_oracle() -> None:
    raw_value = "../秘密/e\u0301"
    raw_digest = legacy.legacy_value_digest(raw_value)
    assert raw_digest == _digest(
        {"type": "tldw_notes_sync_legacy_value", "value": raw_value, "version": 1}
    )

    root_digest = legacy.legacy_root_locator_digest(raw_value)
    assert root_digest == _digest(
        {
            "lexical_root_path": raw_value,
            "type": "tldw_notes_sync_legacy_root_locator",
            "version": 1,
        }
    )

    binding_digest = legacy.legacy_binding_locator_digest(
        "note-é", "nested/../e\u0301.md", root_digest
    )
    assert binding_digest == _digest(
        {
            "lexical_relative_path": "nested/../e\u0301.md",
            "note_id": "note-é",
            "root_locator_digest": root_digest,
            "type": "tldw_notes_sync_legacy_binding_locator",
            "version": 1,
        }
    )

    for item_kind, primary_key in (
        ("root", root_digest),
        ("binding", binding_digest),
        ("legacy_conflict", 7),
    ):
        assert legacy.legacy_item_locator_digest(item_kind, primary_key) == _digest(
            {
                "item_kind": item_kind,
                "legacy_primary_key": primary_key,
                "type": "tldw_notes_sync_legacy_item_locator",
                "version": 1,
            }
        )


@pytest.mark.parametrize(
    ("item_kind", "primary_key"),
    (
        ("root", "raw/private/path"),
        ("root", {"field": "wrong", "value_digest": "a" * 64}),
        ("binding", {"note_id": "note", "relative_value": "raw/path"}),
        ("legacy_conflict", True),
    ),
)
def test_item_locator_rejects_noncanonical_primary_key_shapes(
    item_kind: str, primary_key: object
) -> None:
    with pytest.raises(legacy.LegacyNotesSyncSourceError, match="invalid_item_locator"):
        legacy.legacy_item_locator_digest(item_kind, primary_key)


@pytest.mark.parametrize("value", (None, False, 0, "", "nul\x00path", "x" * 32_769))
def test_malformed_root_paths_get_deterministic_non_null_rejected_locator(
    value: object,
) -> None:
    digest = legacy.rejected_root_item_locator_digest(value)
    expected_primary_key = {
        "field": "notes.sync_directory",
        "value_digest": _digest(
            {"type": "tldw_notes_sync_legacy_value", "value": value, "version": 1}
        ),
    }
    assert digest == _digest(
        {
            "item_kind": "root",
            "legacy_primary_key": expected_primary_key,
            "type": "tldw_notes_sync_legacy_item_locator",
            "version": 1,
        }
    )
    assert digest
    if type(value) is str and value:
        assert value not in digest


@pytest.mark.parametrize("relative", (None, False, 0, "", "nul\x00path", "x" * 32_769))
def test_malformed_binding_paths_get_deterministic_non_null_rejected_locator(
    relative: object,
) -> None:
    root_value = "../private-root"
    digest = legacy.rejected_binding_item_locator_digest(
        "note-private", relative, root_value
    )
    expected_primary_key = {
        "note_id": "note-private",
        "relative_value_digest": legacy.legacy_value_digest(relative),
        "root_value_digest": legacy.legacy_value_digest(root_value),
    }
    assert digest == legacy.legacy_item_locator_digest("binding", expected_primary_key)
    assert digest
    if type(relative) is str and relative:
        assert relative not in digest
    assert root_value not in digest


def test_capture_source_uses_real_io_but_never_candidate_filesystem_operands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = "../never-touch-root-秘密"
    candidate_relative = "nested/never-touch-file.md"
    candidate_conflict = "../never-touch-conflict.md"
    candidates = (candidate_root, candidate_relative, candidate_conflict)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[notes]\nsync_directory = "{candidate_root}"\n', encoding="utf-8"
    )
    config_path.chmod(0o600)
    db = _new_db(tmp_path)
    _add_note(
        db,
        "candidate-note",
        relative_file_path_on_disk=candidate_relative,
        sync_root_folder=candidate_root,
    )
    _add_conflict(db, conflict_id=1, resolution=None)
    with db.transaction() as connection:
        connection.execute(
            "UPDATE sync_conflicts SET file_path = ? WHERE id = 1",
            (candidate_conflict,),
        )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    before_config = config_path.read_bytes()
    before_config_mode = config_path.stat().st_mode
    before_notes = db.read_legacy_notes_sync_source_rows()
    with db.transaction() as connection:
        before_database_rows = (
            tuple(tuple(row) for row in connection.execute("SELECT * FROM notes")),
            tuple(
                tuple(row) for row in connection.execute("SELECT * FROM sync_sessions")
            ),
            tuple(
                tuple(row) for row in connection.execute("SELECT * FROM sync_conflicts")
            ),
        )
    candidate_operands = tuple(str(tmp_path / value) for value in candidates)
    assert all(not os.path.lexists(value) for value in candidate_operands)
    candidate_accesses: list[str] = []

    def reject_candidate(operand: object) -> None:
        text = str(operand)
        if any(candidate in text for candidate in candidates):
            candidate_accesses.append(text)
            raise AssertionError(
                f"candidate filesystem access: {type(operand).__name__}"
            )

    original_open = builtins.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any):
        reject_candidate(file)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    for method_name in ("resolve", "absolute", "stat", "lstat", "open", "iterdir"):
        original = getattr(Path, method_name)

        def guarded_path_method(
            self: Path,
            *args: Any,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            reject_candidate(self)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded_path_method)

    snapshot = legacy.capture_legacy_source(db)

    assert snapshot.source["config"]["sync_directory"] == candidate_root
    assert snapshot.source["notes"][0]["relative_file_path_on_disk"] == (
        candidate_relative
    )
    assert snapshot.source["conflicts"][0]["file_path"] == candidate_conflict
    assert candidate_accesses == []
    assert config_path.read_bytes() == before_config
    assert config_path.stat().st_mode == before_config_mode
    assert db.read_legacy_notes_sync_source_rows() == before_notes
    with db.transaction() as connection:
        after_database_rows = (
            tuple(tuple(row) for row in connection.execute("SELECT * FROM notes")),
            tuple(
                tuple(row) for row in connection.execute("SELECT * FROM sync_sessions")
            ),
            tuple(
                tuple(row) for row in connection.execute("SELECT * FROM sync_conflicts")
            ),
        )
    assert after_database_rows == before_database_rows
    assert all(not os.path.lexists(value) for value in candidate_operands)
    db.close()


def _snapshot(
    *,
    directory: object = "legacy/config-root",
    direction: object = "bidirectional",
    notes: tuple[dict[str, object], ...] = (),
    conflicts: tuple[dict[str, object], ...] = (),
) -> legacy.LegacyNotesSyncSourceSnapshot:
    source = legacy._source_revision(
        {
            "sync_conflict_resolution": "newer_wins",
            "sync_direction": direction,
            "sync_directory": directory,
        },
        notes,
        conflicts,
    )
    return legacy.LegacyNotesSyncSourceSnapshot(source, _digest(source))


def _migration_repository(tmp_path: Path) -> NotesSyncStateRepository:
    return NotesSyncStateRepository(tmp_path / "migration-state.sqlite3")


def test_migration_persists_multiple_lexical_roots_exact_directions_and_siblings(
    tmp_path: Path,
) -> None:
    repository = _migration_repository(tmp_path)
    snapshot = _snapshot(
        direction="disk_to_db",
        notes=(
            _note_row(
                "note-a",
                sync_root_folder="legacy/root-a",
                relative_file_path_on_disk="a.md",
            ),
            _note_row(
                "note-b",
                sync_root_folder="legacy/root-b",
                relative_file_path_on_disk="b.md",
            ),
            _note_row(
                "note-bad",
                sync_root_folder="legacy/root-b",
                relative_file_path_on_disk="",
            ),
            _note_row(
                "note-root-bad",
                sync_root_folder="bad\x00root",
                relative_file_path_on_disk="otherwise-valid.md",
            ),
        ),
        conflicts=(_conflict_row(7, note_id="note-a"),),
    )

    run = repository.record_legacy_generation(snapshot)
    roots = repository.list_roots()
    bindings = tuple(
        binding
        for root in roots
        for binding in repository.list_bindings(root_id=root.root_id)
    )
    items = repository.list_migration_items(run.migration_id)

    assert run.state is MigrationState.PENDING_RECHECK
    assert {root.lexical_root_path for root in roots} == {
        "legacy/config-root",
        "legacy/root-a",
        "legacy/root-b",
    }
    assert {root.direction for root in roots} == {"folder_to_notes"}
    assert {root.state.value for root in roots} == {"candidate"}
    assert all(root.needs_rescan for root in roots)
    assert {binding.note_id for binding in bindings} == {"note-a", "note-b"}
    assert {binding.state.value for binding in bindings} == {"candidate"}
    assert all(binding.needs_rescan for binding in bindings)
    assert any(
        item.item_kind == "binding"
        and item.outcome == "rejected"
        and item.reason_code == "legacy_relative_path_invalid"
        for item in items
    )
    assert any(
        item.item_kind == "root"
        and item.outcome == "rejected"
        and item.reason_code == "legacy_root_path_invalid"
        for item in items
    )
    assert any(
        item.item_kind == "legacy_conflict"
        and item.outcome == "needs_rescan"
        and item.reason_code == "legacy_conflict"
        for item in items
    )
    with notes_sync_state_transaction(
        tmp_path / "migration-state.sqlite3"
    ) as connection:
        destination_sql = " ".join(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL"
            )
        ).lower()
    assert "watcher" not in destination_sql
    assert "activation" not in destination_sql
    assert "content_hash" not in destination_sql
    assert "conflict_content" not in destination_sql


def test_migration_maps_invalid_direction_to_exact_review_state(tmp_path: Path) -> None:
    repository = _migration_repository(tmp_path)

    run = repository.record_legacy_generation(_snapshot(direction=False))
    roots = repository.list_roots()
    items = repository.list_migration_items(run.migration_id)

    assert len(roots) == 1
    assert roots[0].direction == "unspecified"
    assert roots[0].needs_rescan is True
    assert roots[0].reason_code == "legacy_direction_invalid"
    assert any(
        item.item_kind == "root"
        and item.outcome == "needs_rescan"
        and item.reason_code == "legacy_direction_invalid"
        for item in items
    )


def test_duplicate_note_equivalence_class_and_existing_owner_choose_no_winner(
    tmp_path: Path,
) -> None:
    repository = _migration_repository(tmp_path)
    existing_root = repository.create_candidate_root(
        "manual/root", "Manual", "bidirectional"
    )
    repository.create_provisional_binding(existing_root.root_id, "claimed", "manual.md")
    snapshot = _snapshot(
        notes=(
            _note_row(
                "duplicate",
                sync_root_folder="legacy/a",
                relative_file_path_on_disk="one.md",
            ),
            _note_row(
                "duplicate",
                sync_root_folder="legacy/b",
                relative_file_path_on_disk="two.md",
            ),
            _note_row(
                "claimed",
                sync_root_folder="legacy/a",
                relative_file_path_on_disk="claimed.md",
            ),
        )
    )

    run = repository.record_legacy_generation(snapshot)
    items = repository.list_migration_items(run.migration_id)
    migrated = tuple(
        binding
        for root in repository.list_roots()
        for binding in repository.list_bindings(root_id=root.root_id)
        if binding.source_migration_id == run.migration_id
    )

    assert migrated == ()
    assert sum(item.reason_code == "duplicate_note_claim" for item in items) == 2
    assert sum(item.reason_code == "note_already_owned" for item in items) == 1


def test_duplicate_preflight_preserves_only_an_exact_existing_locator_match(
    tmp_path: Path,
) -> None:
    repository = _migration_repository(tmp_path)
    first = repository.record_legacy_generation(
        _snapshot(
            notes=(
                _note_row(
                    "note",
                    sync_root_folder="legacy/a",
                    relative_file_path_on_disk="one.md",
                ),
            )
        )
    )
    original_binding = next(
        binding
        for root in repository.list_roots()
        for binding in repository.list_bindings(root_id=root.root_id)
        if binding.note_id == "note"
    )
    changed = _snapshot(
        direction="disk_to_db",
        notes=(
            _note_row(
                "note",
                sync_root_folder="legacy/a",
                relative_file_path_on_disk="one.md",
            ),
            _note_row(
                "note",
                sync_root_folder="legacy/b",
                relative_file_path_on_disk="two.md",
            ),
        ),
    )

    second = repository.record_legacy_generation(changed)
    binding_after = repository.get_binding(original_binding.binding_id)
    items = repository.list_migration_items(second.migration_id)

    assert binding_after.source_migration_id == second.migration_id
    assert binding_after.row_version == original_binding.row_version + 1
    assert (
        sum(item.outcome == "matched" for item in items if item.item_kind == "binding")
        == 1
    )
    assert sum(item.reason_code == "duplicate_note_claim" for item in items) == 1
    assert first.migration_id != second.migration_id


def test_exact_digest_and_pending_replay_never_rewrite_candidates_or_items(
    tmp_path: Path,
) -> None:
    repository = _migration_repository(tmp_path)
    snapshot = _snapshot(
        notes=(
            _note_row(
                "note",
                sync_root_folder="legacy/root",
                relative_file_path_on_disk="note.md",
            ),
        )
    )
    first = repository.record_legacy_generation(snapshot)
    root_before = next(
        root
        for root in repository.list_roots()
        if root.source_migration_id == first.migration_id
    )
    items_before = repository.list_migration_items(first.migration_id)

    second = repository.record_legacy_generation(snapshot)
    root_after = repository.get_root(root_before.root_id)

    assert second == first
    assert root_after == root_before
    assert repository.list_migration_items(first.migration_id) == items_before


def test_pending_crash_replay_performs_only_fresh_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _migration_repository(tmp_path)
    snapshot = _snapshot(
        notes=(
            _note_row(
                "note",
                sync_root_folder="legacy/root",
                relative_file_path_on_disk="note.md",
            ),
        )
    )
    pending = repository.record_legacy_generation(snapshot)
    roots_before = repository.list_roots()
    items_before = repository.list_migration_items(pending.migration_id)
    captures = iter((snapshot, snapshot))
    monkeypatch.setattr(legacy, "capture_legacy_source", lambda _db: next(captures))

    terminal = legacy.migrate_legacy_notes_sync_state(repository, object())

    assert terminal.state is MigrationState.MATCHED_RECHECK
    assert repository.list_roots() == roots_before
    assert repository.list_migration_items(pending.migration_id) == items_before


def test_changed_digest_updates_only_exact_migration_owned_candidates(
    tmp_path: Path,
) -> None:
    repository = _migration_repository(tmp_path)
    original = _snapshot(
        notes=(
            _note_row(
                "candidate",
                sync_root_folder="legacy/root",
                relative_file_path_on_disk="candidate.md",
            ),
            _note_row(
                "reviewed",
                sync_root_folder="legacy/reviewed",
                relative_file_path_on_disk="reviewed.md",
            ),
        )
    )
    first = repository.record_legacy_generation(original)
    roots = {root.lexical_root_path: root for root in repository.list_roots()}
    reviewed_binding = next(
        binding
        for binding in repository.list_bindings(
            root_id=roots["legacy/reviewed"].root_id
        )
        if binding.note_id == "reviewed"
    )
    repository.pause_root(
        roots["legacy/reviewed"].root_id,
        roots["legacy/reviewed"].row_version,
        "reviewed",
    )
    repository.mark_binding_needs_attention(
        reviewed_binding.binding_id,
        reviewed_binding.row_version,
        "reviewed",
    )
    changed = _snapshot(
        direction="db_to_disk",
        notes=tuple(original.source["notes"]),  # type: ignore[arg-type]
    )

    second = repository.record_legacy_generation(changed)
    updated_candidate = repository.get_root(roots["legacy/root"].root_id)
    untouched_reviewed = repository.get_root(roots["legacy/reviewed"].root_id)
    untouched_binding = repository.get_binding(reviewed_binding.binding_id)

    assert second.migration_id != first.migration_id
    assert updated_candidate.direction == "notes_to_folder"
    assert updated_candidate.source_migration_id == second.migration_id
    assert untouched_reviewed.state.value == "paused"
    assert untouched_reviewed.source_migration_id == first.migration_id
    assert untouched_binding.state.value == "needs_attention"
    assert untouched_binding.source_migration_id == first.migration_id


def test_changed_digest_never_reopens_disconnected_migration_rows(
    tmp_path: Path,
) -> None:
    repository = _migration_repository(tmp_path)
    original = _snapshot(
        notes=(
            _note_row(
                "note",
                sync_root_folder="legacy/root",
                relative_file_path_on_disk="note.md",
            ),
        )
    )
    first = repository.record_legacy_generation(original)
    migrated_root = next(
        root
        for root in repository.list_roots()
        if root.lexical_root_path == "legacy/root"
    )
    migrated_binding = repository.list_bindings(root_id=migrated_root.root_id)[0]
    repository.disconnect_root(migrated_root.root_id, migrated_root.row_version)
    changed = _snapshot(
        direction="db_to_disk",
        notes=tuple(original.source["notes"]),  # type: ignore[arg-type]
    )

    second = repository.record_legacy_generation(changed)
    root_after = repository.get_root(migrated_root.root_id)
    binding_after = repository.get_binding(migrated_binding.binding_id)
    items = repository.list_migration_items(second.migration_id)

    assert root_after.state.value == "disconnected"
    assert binding_after.state.value == "disconnected"
    assert root_after.source_migration_id == first.migration_id
    assert binding_after.source_migration_id == first.migration_id
    assert any(item.reason_code == "candidate_not_mutable" for item in items)
    assert any(item.reason_code == "root_claim_unavailable" for item in items)


@pytest.mark.parametrize("maximum_version_row", ("root", "binding"))
def test_migration_rejects_unadvanceable_candidate_before_any_generation_write(
    tmp_path: Path,
    maximum_version_row: str,
) -> None:
    repository = _migration_repository(tmp_path)
    original = _snapshot(
        notes=(
            _note_row(
                "note",
                sync_root_folder="legacy/root",
                relative_file_path_on_disk="note.md",
            ),
        )
    )
    first = repository.record_legacy_generation(original)
    root = next(
        root
        for root in repository.list_roots()
        if root.lexical_root_path == "legacy/root"
    )
    binding = repository.list_bindings(root_id=root.root_id)[0]
    database = tmp_path / "migration-state.sqlite3"
    with notes_sync_state_transaction(database, immediate=True) as connection:
        table = "sync_roots" if maximum_version_row == "root" else "sync_bindings"
        identifier = (
            root.root_id if maximum_version_row == "root" else binding.binding_id
        )
        id_column = "root_id" if maximum_version_row == "root" else "binding_id"
        connection.execute(
            f"UPDATE {table} SET row_version = ? WHERE {id_column} = ?",  # noqa: S608
            ((2**63) - 1, identifier),
        )
        before_rows = {
            name: tuple(connection.execute(f"SELECT * FROM {name}"))  # noqa: S608
            for name in (
                "sync_migration_runs",
                "sync_roots",
                "sync_bindings",
                "sync_migration_items",
            )
        }
    changed = _snapshot(
        direction="disk_to_db",
        notes=tuple(original.source["notes"]),  # type: ignore[arg-type]
    )

    with pytest.raises(sync_state.NotesSyncStateError, match="cannot be advanced"):
        repository.record_legacy_generation(changed)

    with notes_sync_state_transaction(database) as connection:
        after_rows = {
            name: tuple(connection.execute(f"SELECT * FROM {name}"))  # noqa: S608
            for name in before_rows
        }
    assert after_rows == before_rows
    assert len(after_rows["sync_migration_runs"]) == 1
    assert after_rows["sync_migration_runs"][0][0] == first.migration_id


@pytest.mark.parametrize(
    ("root_count", "binding_count", "limit_name"),
    (
        (MAX_SYNC_ROOTS, 0, "root"),
        (0, MAX_SYNC_BINDINGS, "binding"),
    ),
)
def test_capacity_preflight_aborts_before_any_migration_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_count: int,
    binding_count: int,
    limit_name: str,
) -> None:
    repository = _migration_repository(tmp_path)
    database = tmp_path / "migration-state.sqlite3"
    with notes_sync_state_transaction(database):
        pass
    destination_writes: list[str] = []
    original_transaction = sync_state._repository_transaction

    @contextmanager
    def traced_transaction(database_path: Path, *, immediate: bool = False):
        with original_transaction(database_path, immediate=immediate) as connection:
            if immediate:
                connection.set_trace_callback(destination_writes.append)
            yield connection

    monkeypatch.setattr(sync_state, "_repository_transaction", traced_transaction)
    monkeypatch.setattr(
        "tldw_chatbook.Notes.notes_sync_state.MAX_SYNC_ROOTS", root_count or 64
    )
    monkeypatch.setattr(
        "tldw_chatbook.Notes.notes_sync_state.MAX_SYNC_BINDINGS",
        binding_count or 100_000,
    )
    if root_count:
        monkeypatch.setattr("tldw_chatbook.Notes.notes_sync_state.MAX_SYNC_ROOTS", 0)
    if binding_count:
        monkeypatch.setattr("tldw_chatbook.Notes.notes_sync_state.MAX_SYNC_BINDINGS", 0)
    snapshot = _snapshot(
        notes=(
            _note_row(
                "note",
                sync_root_folder="legacy/root",
                relative_file_path_on_disk="note.md",
            ),
        )
    )

    with pytest.raises(SyncStateCapacityError, match=limit_name):
        repository.record_legacy_generation(snapshot)

    with notes_sync_state_transaction(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM sync_migration_runs"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT count(*) FROM sync_migration_items"
        ).fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM sync_roots").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM sync_bindings").fetchone() == (
            0,
        )
    assert not any(
        statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))
        for statement in destination_writes
    )


def test_migration_item_combinations_are_enforced_and_counts_are_derived(
    tmp_path: Path,
) -> None:
    repository = _migration_repository(tmp_path)
    run = repository.record_legacy_generation(
        _snapshot(
            notes=(
                _note_row(
                    "note",
                    sync_root_folder="legacy/root",
                    relative_file_path_on_disk="note.md",
                ),
            )
        )
    )
    root = next(
        root
        for root in repository.list_roots()
        if root.lexical_root_path == "legacy/root"
    )
    binding = repository.list_bindings(root_id=root.root_id)[0]
    counts_before = repository.migration_item_counts(run.migration_id)
    database = tmp_path / "migration-state.sqlite3"
    invalid_combinations = (
        ("root", "created", None, None, None),
        ("root", "rejected", root.root_id, None, "invalid_path"),
        ("binding", "created", root.root_id, None, None),
        ("legacy_conflict", "created", None, None, None),
        (
            "legacy_conflict",
            "needs_rescan",
            root.root_id,
            binding.binding_id,
            "legacy_conflict",
        ),
    )
    for index, (kind, outcome, root_id, binding_id, reason) in enumerate(
        invalid_combinations
    ):
        with pytest.raises(sqlite3.IntegrityError):
            with notes_sync_state_transaction(database, immediate=True) as connection:
                connection.execute(
                    """INSERT INTO sync_migration_items (
                           migration_id, item_kind, source_locator_digest, outcome,
                           root_id, binding_id, reason_code, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                    (
                        run.migration_id,
                        kind,
                        f"{index + 10:064x}",
                        outcome,
                        root_id,
                        binding_id,
                        reason,
                    ),
                )
    with notes_sync_state_transaction(database, immediate=True) as connection:
        connection.execute(
            "DELETE FROM sync_migration_items WHERE migration_id = ?",
            (run.migration_id,),
        )
    assert counts_before
    assert repository.migration_item_counts(run.migration_id) == ()


def test_migrate_orchestration_records_matched_and_drifted_fresh_rechecks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_a = _snapshot(directory="legacy/a")
    snapshot_b = _snapshot(directory="legacy/b")
    captures = iter((snapshot_a, snapshot_a, snapshot_a, snapshot_b))
    monkeypatch.setattr(legacy, "capture_legacy_source", lambda _db: next(captures))
    first_repository = NotesSyncStateRepository(tmp_path / "matched.sqlite3")
    second_repository = NotesSyncStateRepository(tmp_path / "drifted.sqlite3")

    matched = legacy.migrate_legacy_notes_sync_state(first_repository, object())
    drifted = legacy.migrate_legacy_notes_sync_state(second_repository, object())

    assert matched.state is MigrationState.MATCHED_RECHECK
    assert drifted.state is MigrationState.DRIFTED
    assert matched.source_revision_after == snapshot_a.digest
    assert drifted.source_revision_after == snapshot_b.digest
    assert all(root.needs_rescan for root in first_repository.list_roots())
    assert all(root.needs_rescan for root in second_repository.list_roots())


def test_two_connections_finalizing_different_rechecks_return_one_immutable_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "migration-state.sqlite3"
    snapshot = _snapshot()
    first = NotesSyncStateRepository(database)
    second = NotesSyncStateRepository(database)
    first.record_legacy_generation(snapshot)
    barrier = Barrier(2)
    thread_state = local()
    original_require = sync_state._require_migration_run

    def controlled_capture(snapshots: tuple[object, object]):
        capture_index = getattr(thread_state, "capture_index", 0)
        thread_state.capture_index = capture_index + 1
        return snapshots[capture_index]

    monkeypatch.setattr(legacy, "capture_legacy_source", controlled_capture)

    def controlled_require(connection: sqlite3.Connection, migration_id: str):
        run = original_require(connection, migration_id)
        if not getattr(thread_state, "observed_pending", False):
            thread_state.observed_pending = True
            assert run.state is MigrationState.PENDING_RECHECK
            barrier.wait()
        return run

    monkeypatch.setattr(
        "tldw_chatbook.Notes.notes_sync_state._require_migration_run",
        controlled_require,
    )

    def migrate(
        repository: NotesSyncStateRepository,
        snapshots: tuple[object, object],
    ):
        return legacy.migrate_legacy_notes_sync_state(repository, snapshots)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda pair: migrate(*pair),
                (
                    (first, (snapshot, snapshot)),
                    (second, (snapshot, _snapshot(directory="legacy/drifted"))),
                ),
            )
        )

    assert results[0] == results[1]
    assert results[0].state in {
        MigrationState.MATCHED_RECHECK,
        MigrationState.DRIFTED,
    }
    assert results[0].source_revision_after in {
        snapshot.digest,
        _snapshot(directory="legacy/drifted").digest,
    }


def test_terminal_migration_replay_skips_fresh_source_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _migration_repository(tmp_path)
    snapshot = _snapshot()
    terminal = repository.finalize_legacy_generation(
        repository.record_legacy_generation(snapshot).migration_id,
        source_revision_after=snapshot.digest,
    )
    captures = iter((snapshot,))
    monkeypatch.setattr(legacy, "capture_legacy_source", lambda _db: next(captures))

    replay = legacy.migrate_legacy_notes_sync_state(repository, object())

    assert replay == terminal


def test_sync_foundation_non_goals_and_legacy_owner_are_source_ratchets() -> None:
    project_root = Path(__file__).parents[2]
    notes_root = project_root / "tldw_chatbook/Notes"
    sync_engine = notes_root / "sync_engine.py"
    assert hashlib.sha256(sync_engine.read_bytes()).hexdigest() == (
        "96d8223654da8669bd5f5115fc43928c22f3c1e5c551f4e73c206b5cd0b9dd17"
    )

    foundation_paths = (
        notes_root / "notes_sync_state_schema.py",
        notes_root / "notes_sync_state.py",
        notes_root / "notes_sync_legacy_migration.py",
    )
    prohibited = (
        "activate",
        "activation",
        "watcher",
        "reconcile",
        "resolver",
        "journal",
        "tldw_chatbook.ui",
        "server",
        "sync_v2",
        "sync-v2",
        "backup_",
        "portable export",
    )
    for source_path in foundation_paths:
        source = source_path.read_text(encoding="utf-8").lower()
        assert all(term not in source for term in prohibited), source_path.name
        assert "loguru" not in source
        assert "import logging" not in source
        assert "logger." not in source

    invocation_marker = "migrate_legacy_notes_sync_state("
    startup_callers = []
    migration_module = notes_root / "notes_sync_legacy_migration.py"
    for source_path in (project_root / "tldw_chatbook").rglob("*.py"):
        if source_path == migration_module:
            continue
        source = source_path.read_text(encoding="utf-8")
        if invocation_marker in source:
            startup_callers.append(source_path.relative_to(project_root).as_posix())
    assert startup_callers == []


def test_migration_privacy_redacts_hash_inputs_models_and_aggregates(
    tmp_path: Path,
) -> None:
    private_path = "/private/alice/notes/quarterly.md"
    private_note_id = "private-note-identity"
    private_content_hash = "private-content-hash-input"
    snapshot = _snapshot(
        directory=private_path,
        notes=(
            _note_row(
                private_note_id,
                file_path_on_disk=private_path,
                sync_root_folder=private_path,
                last_synced_disk_file_hash=private_content_hash,
                is_externally_synced=1,
            ),
        ),
        conflicts=(
            _conflict_row(
                1,
                note_id=private_note_id,
                file_path=private_path,
                db_content_hash=private_content_hash,
                disk_content_hash=private_content_hash,
            ),
        ),
    )
    repository = _migration_repository(tmp_path)
    run = repository.record_legacy_generation(snapshot)
    items = repository.list_migration_items(run.migration_id)
    aggregates = repository.migration_item_counts(run.migration_id)

    rendered = repr((snapshot, run, items, aggregates))
    for private in (
        private_path,
        private_note_id,
        private_content_hash,
        snapshot.digest,
        run.migration_id,
    ):
        assert private not in rendered
