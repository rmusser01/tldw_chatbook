"""Trajectory import: read shared traces into read-only snapshots (task-16320).

Round-trip tests build a REAL export file (real ``CharactersRAGDB`` temp
DB via the task-16813 test fixtures, real writer) and feed it to the
import seam, proving the ADR-067 consumer contract: the shared validator
is the only validation, mapping covers compaction + variants + usage +
redaction, and the pure module never references the app DB.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tldw_chatbook.Chat import trajectory_import
from tldw_chatbook.Chat.trajectory_export import (
    TrajectoryExportError,
    build_trajectory_export,
    write_trajectory_export,
)
from tldw_chatbook.Chat.trajectory_import import (
    TrajectoryImportError,
    load_trajectory_snapshot,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.Chat.test_trajectory_export import (
    LONG_TOOL_RESULT,
    _seed_conversation,
    _seed_library_activity,
)


@dataclass(frozen=True)
class VariantSetLike:
    """Duck-typed ``ConsoleVariantSet`` stand-in (same mirror as export tests)."""

    turn_id: str
    variants: tuple[str, ...]
    selected_index: int = 0


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(tmp_path / "test.db", client_id="test")


def _trace_file(tmp_path: Path, database: CharactersRAGDB, **export_kwargs) -> Path:
    """Build + write a real export file for one seeded conversation."""
    conv = _seed_conversation(database)
    payload = build_trajectory_export(database, conv, **export_kwargs)
    return write_trajectory_export(tmp_path / "shared-trace.json", payload)


# ---------------------------------------------------------------------------
# Happy path: file -> snapshot through the real mapping
# ---------------------------------------------------------------------------


def test_import_file_renders_records_compaction_and_variants(db, tmp_path) -> None:
    trace = _trace_file(
        tmp_path,
        db,
        variant_sets=(VariantSetLike("t2", ("first draft", "winning draft"), 1),),
    )

    snapshot = load_trajectory_snapshot(trace)

    kinds = [record.kind for turn in snapshot.turns for record in turn.records]
    assert kinds == [
        "user",
        "assistant",
        "tool_call",
        "tool_result",
        "user",
        "assistant",
        "compaction",
    ]
    # Usage travels through usage_json -> ProviderUsage.
    assistant_records = [
        record
        for turn in snapshot.turns
        for record in turn.records
        if record.kind == "assistant"
    ]
    assert assistant_records[-1].usage is not None
    assert assistant_records[-1].usage.output == 4
    # Variant sets travel: superseded content attaches to turn t2's assistant.
    assert "first draft" in assistant_records[-1].variants
    # ISO-string timestamps in the file parse into timing fields.
    assert assistant_records[-1].step_started_at is None  # a2 row has no timing
    first_user = snapshot.turns[0].records[0]
    assert first_user.step_started_at == 1_755_165_600.0


def test_redacted_payloads_render_with_redaction_marker(db, tmp_path) -> None:
    trace = _trace_file(tmp_path, db)  # default export = redacted

    snapshot = load_trajectory_snapshot(trace)

    tool_call = next(
        record
        for turn in snapshot.turns
        for record in turn.records
        if record.kind == "tool_call"
    )
    assert tool_call.payload is not None
    assert tool_call.payload.get("redacted") is True
    assert tool_call.payload.get("result_preview")
    # The full un-redacted payload never entered the snapshot.
    assert LONG_TOOL_RESULT not in json.dumps(tool_call.payload)


def test_import_accepts_path_str_and_parsed_dict(db, tmp_path) -> None:
    trace = _trace_file(tmp_path, db)
    from_path = load_trajectory_snapshot(trace)
    from_str = load_trajectory_snapshot(str(trace))
    document = json.loads(trace.read_text(encoding="utf-8"))
    from_dict = load_trajectory_snapshot(document)

    assert from_path.turns == from_str.turns == from_dict.turns
    assert from_path.turns  # snapshots are non-empty


def test_import_legacy_conversation_without_sidecar(db, tmp_path) -> None:
    conv = _seed_conversation(db, with_sidecar=False)
    payload = build_trajectory_export(db, conv)
    trace = write_trajectory_export(tmp_path / "legacy.json", payload)

    snapshot = load_trajectory_snapshot(trace)

    kinds = [record.kind for turn in snapshot.turns for record in turn.records]
    assert kinds == ["user", "assistant", "user", "assistant", "compaction"]


def test_imported_library_activity_remains_inert_sidecar_data(db) -> None:
    conversation_id = _seed_conversation(db)
    _seed_library_activity(db, conversation_id)
    document = build_trajectory_export(
        db, conversation_id, include_payloads=True
    )
    assert any(
        row["event_kind"] == "library_activity"
        for row in document["trajectory_rows"]
    )

    snapshot = load_trajectory_snapshot(document)
    kinds = [record.kind for turn in snapshot.turns for record in turn.records]

    assert "library_activity" not in kinds
    assert "note-secret-id" not in repr(snapshot)


# ---------------------------------------------------------------------------
# Malformed input -> actionable TrajectoryImportError
# ---------------------------------------------------------------------------


def test_not_json_file_is_rejected_with_actionable_message(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(TrajectoryImportError, match="bad.json.*not valid JSON"):
        load_trajectory_snapshot(bad)


def test_missing_file_is_rejected_with_named_path(tmp_path) -> None:
    with pytest.raises(TrajectoryImportError, match="Cannot read.*nope.json"):
        load_trajectory_snapshot(tmp_path / "nope.json")


def test_wrong_format_marker_is_rejected_via_shared_validator(db) -> None:
    payload = build_trajectory_export(db, _seed_conversation(db))
    payload["format"] = "something-else"
    with pytest.raises(TrajectoryImportError, match="'format'"):
        load_trajectory_snapshot(payload)
    # The shared-validator lineage is preserved (ADR-067 seam).
    with pytest.raises(TrajectoryExportError):
        load_trajectory_snapshot(payload)


def test_unsupported_version_is_rejected(db) -> None:
    payload = build_trajectory_export(db, _seed_conversation(db))
    payload["version"] = 2
    with pytest.raises(TrajectoryImportError, match="version 2"):
        load_trajectory_snapshot(payload)


def test_missing_section_is_rejected(db) -> None:
    payload = build_trajectory_export(db, _seed_conversation(db))
    del payload["messages"]
    with pytest.raises(TrajectoryImportError, match="'messages'"):
        load_trajectory_snapshot(payload)


def test_import_rejects_reserved_thinking_via_shared_validator(db) -> None:
    payload = build_trajectory_export(db, _seed_conversation(db))
    payload["messages"][0]["thinking_blocks_json"] = "IMPORT-THINKING-CANARY"

    with pytest.raises(
        TrajectoryImportError, match="reserved thinking field"
    ) as caught:
        load_trajectory_snapshot(payload)

    assert "IMPORT-THINKING-CANARY" not in str(caught.value)


def test_json_array_top_level_is_rejected(tmp_path) -> None:
    bad = tmp_path / "array.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(TrajectoryImportError, match="JSON object"):
        load_trajectory_snapshot(bad)


# ---------------------------------------------------------------------------
# Read-only proof: no DB references in the pure module (structural)
# ---------------------------------------------------------------------------


def test_pure_module_never_imports_or_names_the_db() -> None:
    """Structural no-DB-write proof: the import module cannot touch the DB.

    It holds no DB imports and never names the DB module, so it cannot
    open, read, or write the application database. (The behavioral
    counterpart -- DB row counts unchanged across a UI import -- lives in
    ``Tests/UI/test_trajectory_import_ui.py``.)
    """
    source = Path(trajectory_import.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("tldw_chatbook.DB"), (
                f"trajectory_import must not import the DB layer (found {name!r})"
            )
    assert "ChaChaNotes_DB" not in source
    assert "import sqlite3" not in source
