"""Trajectory export: format, redaction, writer, validator (task-16813).

Round-trip tests prove the ADR-067 contract: a real ``CharactersRAGDB``
(temp file, real schema-v38 sidecar + real auxiliary-attempt rows) is
exported, validated, written/read through the atomic writer, and the
file's data re-renders through the REAL projection
(``derive_trajectory``) -- the file carries everything the view needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_context_repository import (
    AuxiliaryAttemptStart,
    AuxiliaryAttemptStatus,
    ConsoleContextRepository,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.Chat.trajectory_export import (
    PREVIEW_MAX_CHARS,
    TrajectoryExportError,
    build_trajectory_export,
    validate_trajectory_export,
    write_trajectory_export,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, TrajectoryRowWrite

LONG_TOOL_RESULT = "R" * 300  # longer than the preview cap

_TOOL_PAYLOAD = json.dumps(
    {
        "name": "fs_read",
        "args": {"path": "/tmp/report.txt"},
        "result": LONG_TOOL_RESULT,
    }
)

_T0 = "2026-08-14T12:00:00"  # deterministic message timestamps


@dataclass(frozen=True)
class VariantSetLike:
    """Duck-typed ``ConsoleVariantSet`` stand-in (same mirror as UI tests)."""

    turn_id: str
    variants: tuple[str, ...]
    selected_index: int = 0


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(tmp_path / "test.db", client_id="test")


def _seed_conversation(database: CharactersRAGDB, *, with_sidecar: bool = True) -> str:
    """Seed one conversation: 2 turns, tool rows, usage, compaction attempt.

    Ledger order mirrors ``Tests/UI/test_trajectory_screen.py`` fixtures:
    u1, a1, tool_call, tool_result (turn t1); u2, a2 (turn t2).
    """
    conv = database.add_conversation({"title": "trajectory export conv"})
    u1 = database.add_message(
        {
            "conversation_id": conv,
            "sender": "user",
            "content": "hello trajectory world",
            "timestamp": _T0,
        }
    )
    a1 = database.add_message(
        {
            "conversation_id": conv,
            "sender": "assistant",
            "content": "checking that for you",
            "parent_message_id": u1,
            "timestamp": "2026-08-14T12:00:02",
        }
    )
    u2 = database.add_message(
        {
            "conversation_id": conv,
            "sender": "user",
            "content": "second question about zebras",
            "parent_message_id": a1,
            "timestamp": "2026-08-14T12:01:00",
        }
    )
    a2 = database.add_message(
        {
            "conversation_id": conv,
            "sender": "assistant",
            "content": "zebras have stripes",
            "parent_message_id": u2,
            "timestamp": "2026-08-14T12:01:05",
            "usage_json": json.dumps(
                {
                    "uncached_input": 10,
                    "output": 4,
                    "provider": "test-provider",
                    "model": "test-model",
                }
            ),
        }
    )
    database.set_conversation_active_leaf(conv, a2)

    if with_sidecar:
        database.upsert_trajectory_rows(
            [
                TrajectoryRowWrite(
                    message_id=u1,
                    conversation_id=conv,
                    turn_id="t1",
                    seq=1,
                    event_kind="user",
                    step_started_at=1_755_165_600.0,
                ),
                TrajectoryRowWrite(
                    message_id=a1,
                    conversation_id=conv,
                    turn_id="t1",
                    seq=2,
                    event_kind="assistant",
                    step_started_at=1_755_165_600.0,
                    first_token_at=1_755_165_602.0,
                    completed_at=1_755_165_605.0,
                    model="test-model",
                    provider="test-provider",
                ),
                TrajectoryRowWrite(
                    message_id=a1,
                    conversation_id=conv,
                    turn_id="t1",
                    seq=3,
                    event_kind="tool_call",
                    payload_json=_TOOL_PAYLOAD,
                ),
                TrajectoryRowWrite(
                    message_id=a1,
                    conversation_id=conv,
                    turn_id="t1",
                    seq=4,
                    event_kind="tool_result",
                    payload_json=_TOOL_PAYLOAD,
                ),
                TrajectoryRowWrite(
                    message_id=u2,
                    conversation_id=conv,
                    turn_id="t2",
                    seq=5,
                    event_kind="user",
                    step_started_at=1_755_165_660.0,
                ),
                TrajectoryRowWrite(
                    message_id=a2,
                    conversation_id=conv,
                    turn_id="t2",
                    seq=6,
                    event_kind="assistant",
                    model="test-model",
                    provider="test-provider",
                ),
            ]
        )

    repository = ConsoleContextRepository(database)
    repository.start_auxiliary_attempt(
        AuxiliaryAttemptStart(
            operation_id="op-compaction-1",
            conversation_id=conv,
            purpose="conversation_compaction",
            provider="test-provider",
            model="test-model",
            requested_output_cap=100,
            estimated_input_tokens=50,
            started_at="2026-08-14T12:05:00Z",
        )
    )
    repository.finish_auxiliary_attempt(
        "op-compaction-1",
        status=AuxiliaryAttemptStatus.SUCCEEDED,
        finished_at="2026-08-14T12:05:03Z",
        elapsed_ms=3_000,
        usage=ProviderUsage(output=2),
    )
    return conv


def _snapshot_from_export(payload: dict):
    """Re-render through the REAL projection using only export-file data."""
    return derive_trajectory(
        payload["messages"],
        {
            message["id"]: ProviderUsage.from_json(message["usage_json"])
            for message in payload["messages"]
        },
        payload["trajectory_rows"],
        payload.get("variants") or (),
        payload["compaction_records"],
        payload.get("active_leaf_message_id"),
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_file_renders_through_projection(db, tmp_path) -> None:
    conv = _seed_conversation(db)
    payload = build_trajectory_export(
        db,
        conv,
        variant_sets=(VariantSetLike("t2", ("first draft", "winning draft"), 1),),
    )
    normalized = validate_trajectory_export(payload)
    path = write_trajectory_export(tmp_path / "trace.json", normalized)
    # What an importer sees: read the file back, validate, re-render.
    on_disk = validate_trajectory_export(json.loads(path.read_text(encoding="utf-8")))

    assert on_disk["format"] == "tldw-trajectory"
    assert on_disk["version"] == 1
    assert on_disk["conversation"]["id"] == conv
    assert on_disk["conversation"]["title"] == "trajectory export conv"
    assert on_disk["redacted"] is True
    assert [m["id"] for m in on_disk["messages"]] == [
        m["id"] for m in payload["messages"]
    ]
    assert len(on_disk["trajectory_rows"]) == 6
    assert on_disk["compaction_records"][0]["purpose"] == "conversation_compaction"
    assert on_disk["variants"][0]["selected_index"] == 1

    snapshot = _snapshot_from_export(on_disk)
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
    # Usage travels: the assistant record carries the seeded tokens.
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
    # Redacted tool payload still renders a preview.
    tool_call = next(r for r in snapshot.turns[0].records if r.kind == "tool_call")
    assert tool_call.payload is not None
    assert tool_call.payload.get("redacted") is True


def test_export_omits_image_blobs_and_message_keys(db) -> None:
    conv = _seed_conversation(db)
    payload = build_trajectory_export(db, conv)
    assert set(payload["messages"][0]) == {
        "id",
        "sender",
        "content",
        "timestamp",
        "parent_message_id",
        "usage_json",
        "assistant_generation_state",
    }
    assert "image_data" not in payload["messages"][0]


def test_unknown_conversation_raises(db) -> None:
    with pytest.raises(TrajectoryExportError, match="not found"):
        build_trajectory_export(db, "no-such-conversation")


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redaction_default_replaces_tool_payloads(db) -> None:
    conv = _seed_conversation(db)
    payload = build_trajectory_export(db, conv)

    assert payload["redacted"] is True
    tool_rows = [
        r
        for r in payload["trajectory_rows"]
        if r["event_kind"] in ("tool_call", "tool_result")
    ]
    assert len(tool_rows) == 2
    for row in tool_rows:
        stub = json.loads(row["payload_json"])
        assert stub == {
            "name": "fs_read",
            "result_preview": LONG_TOOL_RESULT[:PREVIEW_MAX_CHARS],
            "args_preview": '{"path": "/tmp/report.txt"}',
            "redacted": True,
        }
        assert len(stub["result_preview"]) <= PREVIEW_MAX_CHARS
        assert "\n" not in stub["result_preview"]
        # The full payload must not leak anywhere in the document.
        assert LONG_TOOL_RESULT not in json.dumps(payload)
    # Non-tool rows are untouched.
    user_row = next(r for r in payload["trajectory_rows"] if r["event_kind"] == "user")
    assert user_row["payload_json"] is None


def test_include_payloads_opt_in_keeps_verbatim(db) -> None:
    conv = _seed_conversation(db)
    payload = build_trajectory_export(db, conv, include_payloads=True)

    assert payload["redacted"] is False
    tool_row = next(
        r for r in payload["trajectory_rows"] if r["event_kind"] == "tool_call"
    )
    assert tool_row["payload_json"] == _TOOL_PAYLOAD


# ---------------------------------------------------------------------------
# Legacy (no sidecar rows)
# ---------------------------------------------------------------------------


def test_legacy_conversation_without_sidecar_exports(db) -> None:
    conv = _seed_conversation(db, with_sidecar=False)
    payload = build_trajectory_export(db, conv)
    normalized = validate_trajectory_export(payload)

    assert normalized["trajectory_rows"] == []
    assert len(normalized["messages"]) == 4
    snapshot = _snapshot_from_export(normalized)
    kinds = [record.kind for turn in snapshot.turns for record in turn.records]
    assert kinds == ["user", "assistant", "user", "assistant", "compaction"]


# ---------------------------------------------------------------------------
# Validator rejections
# ---------------------------------------------------------------------------


def _valid_payload(db) -> dict:
    return build_trajectory_export(db, _seed_conversation(db))


def test_rejects_wrong_format_marker(db) -> None:
    payload = _valid_payload(db)
    payload["format"] = "something-else"
    with pytest.raises(TrajectoryExportError, match="'format'"):
        validate_trajectory_export(payload)


def test_rejects_higher_version(db) -> None:
    payload = _valid_payload(db)
    payload["version"] = 2
    with pytest.raises(TrajectoryExportError, match="version 2"):
        validate_trajectory_export(payload)


def test_rejects_missing_messages_section(db) -> None:
    payload = _valid_payload(db)
    del payload["messages"]
    with pytest.raises(TrajectoryExportError, match="'messages'"):
        validate_trajectory_export(payload)


def test_rejects_missing_trajectory_rows_section(db) -> None:
    payload = _valid_payload(db)
    del payload["trajectory_rows"]
    with pytest.raises(TrajectoryExportError, match="'trajectory_rows'"):
        validate_trajectory_export(payload)


def test_rejects_message_missing_id(db) -> None:
    payload = _valid_payload(db)
    del payload["messages"][1]["id"]
    with pytest.raises(TrajectoryExportError, match=r"messages\[1\]\.id"):
        validate_trajectory_export(payload)


def test_validator_normalizes_optional_sections(db) -> None:
    payload = _valid_payload(db)
    for key in ("compaction_records", "variants", "active_leaf_message_id"):
        payload.pop(key, None)
    normalized = validate_trajectory_export(payload)
    assert normalized["compaction_records"] == []
    assert normalized["variants"] == []
    assert normalized["active_leaf_message_id"] is None


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def test_write_is_atomic_and_overwrites(tmp_path) -> None:
    payload = _valid_payload(CharactersRAGDB(tmp_path / "test.db", client_id="test"))
    target = tmp_path / "trace.json"

    first = write_trajectory_export(target, payload)
    assert first == target
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob("*.tmp")) == []  # no temp leftovers

    payload["redacted"] = False
    write_trajectory_export(str(target), payload)  # str path + overwrite
    assert json.loads(target.read_text(encoding="utf-8"))["redacted"] is False
    # Only the target (plus the DB's own files) remains -- no temp artifacts.
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
