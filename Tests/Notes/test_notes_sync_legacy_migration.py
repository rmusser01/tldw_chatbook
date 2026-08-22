"""Read-only legacy Notes sync source and canonical digest contracts."""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes import notes_sync_legacy_migration as legacy


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
