"""Bounded Library preparation sidecar contracts (Task 12)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.Chat.library_preparation import (
    LIBRARY_PREPARATION_EVENT_KIND,
    LIBRARY_PREPARATION_MAX_BYTES,
    LIBRARY_PREPARATION_RESULT_COUNT_MAX,
    LibraryPreparationContribution,
    LibraryPreparationEvent,
    LibraryPreparationValidationError,
    decode_library_preparation_event,
    encode_library_preparation_event,
    library_preparation_event_for_outcome,
    project_library_preparation,
)
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.Chat.trajectory_export import build_trajectory_export
from tldw_chatbook.Chat.trajectory_import import load_trajectory_snapshot
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, TrajectoryRowWrite
from Tests.Chat.test_trajectory_export import _seed_conversation


def _event(
    *,
    outcome: str = "zero_matches",
    attempt_id: str = "attempt-opaque-1",
    result_count: int = 0,
    source_types: tuple[str, ...] = ("notes", "media", "conversations"),
) -> LibraryPreparationEvent:
    return LibraryPreparationEvent(
        version=1,
        outcome=outcome,  # type: ignore[arg-type]
        attempt_id=attempt_id,
        result_count=result_count,
        source_types=source_types,  # type: ignore[arg-type]
    )


class _RecordingWriter:
    def __init__(self, sequence: int = 17) -> None:
        self.sequence = sequence
        self.operations: list[tuple[object, ...]] = []

    def next_trajectory_sequence(self) -> int:
        self.operations.append(("next_trajectory_sequence",))
        return self.sequence

    def execute(self, statement: str, parameters: tuple[object, ...], /) -> None:
        self.operations.append(("execute", statement, parameters))

    def executemany(self, statement, parameter_rows, /) -> None:
        raise AssertionError("one preparation event must use one execute")


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(tmp_path / "test.db", client_id="preparation-test")


def _insert_preparation_row(
    database: CharactersRAGDB,
    conversation_id: str,
    *,
    payload_json: str,
    seq: int = 7,
) -> str:
    messages = database.get_messages_for_conversation(
        conversation_id,
        limit=100,
        include_image_data=False,
    )
    user_id = next(str(row["id"]) for row in messages if row["sender"] == "user")
    database.upsert_trajectory_rows(
        [
            TrajectoryRowWrite(
                message_id=user_id,
                conversation_id=conversation_id,
                turn_id=user_id,
                seq=seq,
                event_kind=LIBRARY_PREPARATION_EVENT_KIND,
                payload_json=payload_json,
            )
        ]
    )
    return user_id


def test_event_is_immutable_and_round_trips_exact_canonical_payload() -> None:
    event = _event(outcome="bypassed", result_count=4)

    with pytest.raises(FrozenInstanceError):
        event.result_count = 5  # type: ignore[misc]

    encoded = encode_library_preparation_event(event)
    assert encoded == (
        '{"version":1,"outcome":"bypassed","attempt_id":"attempt-opaque-1",'
        '"result_count":4,"source_types":["notes","media","conversations"]}'
    )
    assert decode_library_preparation_event(encoded) == event
    assert len(encoded.encode("utf-8")) <= LIBRARY_PREPARATION_MAX_BYTES


@pytest.mark.parametrize("outcome", ["zero_matches", "bypassed"])
def test_only_the_two_disclosure_outcomes_create_events(outcome: str) -> None:
    event = library_preparation_event_for_outcome(
        outcome,
        attempt_id="attempt-1",
        result_count=0,
        source_types=("notes", "media", "conversations"),
    )

    assert event is not None
    assert event.outcome == outcome


@pytest.mark.parametrize(
    "outcome",
    [
        None,
        "cancelled",
        "retrieval_failure",
        "persistence_failure",
        "destination_changed",
    ],
)
def test_cancelled_and_failure_outcomes_create_no_durable_event(
    outcome: str | None,
) -> None:
    assert (
        library_preparation_event_for_outcome(
            outcome,
            attempt_id="attempt-1",
            result_count=0,
            source_types=("notes", "media", "conversations"),
        )
        is None
    )


def test_unknown_outcome_does_not_silently_widen_the_event_vocabulary() -> None:
    with pytest.raises(LibraryPreparationValidationError):
        library_preparation_event_for_outcome(
            "retrieved_sources",
            attempt_id="attempt-1",
            result_count=2,
            source_types=("notes",),
        )


@pytest.mark.parametrize(
    "event",
    [
        LibraryPreparationEvent(2, "zero_matches", "attempt", 0, ("notes",)),
        LibraryPreparationEvent(1, "success", "attempt", 0, ("notes",)),
        LibraryPreparationEvent(1, "zero_matches", "", 0, ("notes",)),
        LibraryPreparationEvent(1, "zero_matches", "has whitespace", 0, ("notes",)),
        LibraryPreparationEvent(1, "zero_matches", "attempt", -1, ("notes",)),
        LibraryPreparationEvent(
            1,
            "zero_matches",
            "attempt",
            LIBRARY_PREPARATION_RESULT_COUNT_MAX + 1,
            ("notes",),
        ),
        LibraryPreparationEvent(1, "zero_matches", "attempt", 0, ("files",)),
        LibraryPreparationEvent(
            1,
            "zero_matches",
            "attempt",
            0,
            ("notes", "notes"),
        ),
        LibraryPreparationEvent(
            1,
            "zero_matches",
            "attempt",
            0,
            ("media", "notes"),
        ),
    ],
)
def test_encoder_rejects_invalid_or_unbounded_events(
    event: LibraryPreparationEvent,
) -> None:
    with pytest.raises(LibraryPreparationValidationError):
        encode_library_preparation_event(event)


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "\ud800",
        "[]",
        '{"version":1,"outcome":[],"attempt_id":"a",'
        '"result_count":0,"source_types":["notes"]}',
        '{"version":NaN,"outcome":"zero_matches","attempt_id":"a",'
        '"result_count":0,"source_types":["notes"]}',
        '{"version":1,"outcome":"zero_matches","attempt_id":"a",'
        '"result_count":0,"source_types":["notes"],"query":"secret"}',
        '{"version":1,"outcome":"zero_matches","attempt_id":"a",'
        '"result_count":0,"source_types":["notes"],"source_id":"note-1"}',
        '{"version":1,"version":1,"outcome":"zero_matches",'
        '"attempt_id":"a","result_count":0,"source_types":["notes"]}',
    ],
)
def test_decoder_rejects_malformed_duplicate_or_unknown_fields(payload: str) -> None:
    with pytest.raises(LibraryPreparationValidationError):
        decode_library_preparation_event(payload)


def test_payload_cannot_retain_queries_sources_destinations_or_credentials() -> None:
    encoded = encode_library_preparation_event(
        _event(attempt_id="opaque-123", source_types=("notes", "media"))
    )
    decoded = json.loads(encoded)

    assert tuple(decoded) == (
        "version",
        "outcome",
        "attempt_id",
        "result_count",
        "source_types",
    )
    forbidden = (
        "query",
        "source_id",
        "title",
        "body",
        "path",
        "exception",
        "endpoint",
        "provider",
        "model",
        "credential",
    )
    assert all(term not in encoded.lower() for term in forbidden)


def test_contribution_owns_user_and_uses_one_sequence_before_exact_insert() -> None:
    writer = _RecordingWriter(sequence=17)
    contribution = LibraryPreparationContribution(_event())

    contribution.write(
        writer=writer,
        conversation_id="conversation-1",
        message_ids={"user": "user-1", "assistant": "assistant-1"},
    )

    payload = encode_library_preparation_event(_event())
    assert contribution.owner_message_key == "user"
    assert writer.operations == [
        ("next_trajectory_sequence",),
        (
            "execute",
            "INSERT INTO message_trajectory_metadata("
            "message_id, conversation_id, turn_id, seq, event_kind, payload_json"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                "user-1",
                "conversation-1",
                "user-1",
                17,
                "library_preparation",
                payload,
            ),
        ),
    ]


def test_contribution_fails_before_sequence_allocation_without_user_owner() -> None:
    writer = _RecordingWriter()

    with pytest.raises(ValueError, match="(?i)user"):
        LibraryPreparationContribution(_event()).write(
            writer=writer,
            conversation_id="conversation-1",
            message_ids={"assistant": "assistant-1"},
        )

    assert writer.operations == []


def test_contribution_rejects_a_runtime_owner_override() -> None:
    writer = _RecordingWriter()
    contribution = LibraryPreparationContribution(
        _event(),
        owner_message_key="assistant",  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="(?i)user"):
        contribution.write(
            writer=writer,
            conversation_id="conversation-1",
            message_ids={"user": "user-1", "assistant": "assistant-1"},
        )

    assert writer.operations == []


def test_projection_is_active_turn_filtered_bounded_and_input_order_independent() -> (
    None
):
    first = {
        "message_id": "turn-1",
        "conversation_id": "conversation-1",
        "turn_id": "turn-1",
        "seq": 8,
        "event_kind": "library_preparation",
        "payload_json": encode_library_preparation_event(_event()),
    }
    second = {
        "message_id": "turn-2",
        "conversation_id": "conversation-1",
        "turn_id": "turn-2",
        "seq": 3,
        "event_kind": "library_preparation",
        "payload_json": encode_library_preparation_event(
            _event(
                outcome="bypassed",
                attempt_id="attempt-2",
                result_count=2,
                source_types=("notes", "media"),
            )
        ),
    }
    inactive = dict(second, message_id="turn-3", turn_id="turn-3", seq=1)

    expected = project_library_preparation(
        [second, first, inactive],
        ["turn-1", "turn-2"],
    )
    reverse = project_library_preparation(
        [inactive, first, second],
        ["turn-1", "turn-2"],
    )

    assert expected == reverse
    assert [(view.turn_id, view.outcome) for view in expected] == [
        ("turn-1", "zero_matches"),
        ("turn-2", "bypassed"),
    ]
    assert expected[1].result_count == 2
    assert expected[1].source_types == ("notes", "media")


def test_projection_omits_malformed_wrong_owner_and_duplicate_rows_fail_closed() -> (
    None
):
    valid_payload = encode_library_preparation_event(_event())
    rows = [
        {
            "message_id": "turn-bad-json",
            "turn_id": "turn-bad-json",
            "seq": 1,
            "event_kind": "library_preparation",
            "payload_json": '{"query":"secret"}',
        },
        {
            "message_id": "assistant-1",
            "turn_id": "turn-wrong-owner",
            "seq": 2,
            "event_kind": "library_preparation",
            "payload_json": valid_payload,
        },
        {
            "message_id": "turn-duplicate",
            "turn_id": "turn-duplicate",
            "seq": 3,
            "event_kind": "library_preparation",
            "payload_json": valid_payload,
        },
        {
            "message_id": "turn-duplicate",
            "turn_id": "turn-duplicate",
            "seq": 4,
            "event_kind": "library_preparation",
            "payload_json": valid_payload,
        },
        {
            "message_id": "turn-other-kind",
            "turn_id": "turn-other-kind",
            "seq": 5,
            "event_kind": "library_activity",
            "payload_json": valid_payload,
        },
    ]

    assert (
        project_library_preparation(
            rows,
            [
                "turn-bad-json",
                "turn-wrong-owner",
                "turn-duplicate",
                "turn-other-kind",
            ],
        )
        == ()
    )


def test_generic_trajectory_sidecar_cannot_displace_or_duplicate_user_anchor() -> None:
    messages = [
        {
            "id": "user-1",
            "sender": "user",
            "content": "hello",
            "timestamp": "2026-08-22T10:00:00Z",
            "parent_message_id": None,
            "deleted": 0,
        }
    ]
    user_anchor = {
        "message_id": "user-1",
        "conversation_id": "conversation-1",
        "turn_id": "user-1",
        "seq": 1,
        "event_kind": "user",
        "step_started_at": 11.0,
        "payload_json": None,
    }
    sidecar = {
        "message_id": "user-1",
        "conversation_id": "conversation-1",
        "turn_id": "user-1",
        "seq": 2,
        "event_kind": "library_preparation",
        "step_started_at": 999.0,
        "payload_json": encode_library_preparation_event(_event()),
    }

    snapshot = derive_trajectory(messages, {}, [user_anchor, sidecar], (), ())

    assert len(snapshot.turns) == 1
    assert len(snapshot.turns[0].records) == 1
    record = snapshot.turns[0].records[0]
    assert record.kind == "user"
    assert record.step_started_at == 11.0


def test_default_and_full_export_use_the_same_canonical_bounded_payload(db) -> None:
    conversation_id = _seed_conversation(db)
    canonical = encode_library_preparation_event(
        _event(outcome="bypassed", result_count=3)
    )
    _insert_preparation_row(db, conversation_id, payload_json=canonical)

    default = build_trajectory_export(db, conversation_id)
    full = build_trajectory_export(db, conversation_id, include_payloads=True)
    default_row = next(
        row
        for row in default["trajectory_rows"]
        if row["event_kind"] == "library_preparation"
    )
    full_row = next(
        row
        for row in full["trajectory_rows"]
        if row["event_kind"] == "library_preparation"
    )

    assert default_row["payload_json"] == canonical
    assert full_row["payload_json"] == canonical
    assert len(default_row["payload_json"].encode("utf-8")) <= 1024


def test_export_fails_closed_in_both_modes_for_a_malformed_sidecar(db) -> None:
    conversation_id = _seed_conversation(db)
    _insert_preparation_row(
        db,
        conversation_id,
        payload_json=json.dumps(
            {
                "version": 1,
                "outcome": "zero_matches",
                "attempt_id": "attempt-1",
                "result_count": 0,
                "source_types": ["notes"],
                "query": "private query",
                "source_id": "private-source",
            }
        ),
    )

    default = build_trajectory_export(db, conversation_id)
    full = build_trajectory_export(db, conversation_id, include_payloads=True)

    for payload in (default, full):
        row = next(
            item
            for item in payload["trajectory_rows"]
            if item["event_kind"] == "library_preparation"
        )
        assert row["payload_json"] is None
        serialized = json.dumps(row)
        assert "private query" not in serialized
        assert "private-source" not in serialized


def test_imported_sidecar_and_unknown_fields_are_inert(db) -> None:
    conversation_id = _seed_conversation(db)
    payload = build_trajectory_export(db, conversation_id)
    user_row = next(row for row in payload["messages"] if row["sender"] == "user")
    payload["trajectory_rows"].append(
        {
            "message_id": user_row["id"],
            "conversation_id": conversation_id,
            "turn_id": user_row["id"],
            "seq": 99,
            "event_kind": "library_preparation",
            "step_started_at": 999.0,
            "first_token_at": None,
            "completed_at": None,
            "model": None,
            "provider": None,
            "payload_json": json.dumps(
                {
                    "version": 1,
                    "outcome": "bypassed",
                    "attempt_id": "attempt-imported",
                    "result_count": 0,
                    "source_types": ["notes"],
                    "activate_context": True,
                    "action": "dispatch",
                }
            ),
        }
    )

    snapshot = load_trajectory_snapshot(payload)
    kinds = [record.kind for turn in snapshot.turns for record in turn.records]

    assert "library_preparation" not in kinds
    assert kinds == [
        "user",
        "assistant",
        "tool_call",
        "tool_result",
        "user",
        "assistant",
        "compaction",
    ]
    imported_user = next(
        record
        for turn in snapshot.turns
        for record in turn.records
        if record.message_id == user_row["id"]
    )
    assert imported_user.step_started_at != 999.0
    assert (
        project_library_preparation(
            payload["trajectory_rows"],
            [str(user_row["id"])],
        )
        == ()
    )


# --- TASK-33621.51: what a sent turn's Automatic Library search did ---------
#
# The v1 record held only "zero_matches" and a reasonless "bypassed", so a
# grounded turn had no record at all and a bypassed one lost why it was
# bypassed. Each test below fails on the pre-fix vocabulary.

_ALL_SOURCES = ("notes", "media", "conversations")


def test_a_grounded_send_records_how_many_sources_it_sent() -> None:
    event = library_preparation_event_for_outcome(
        "evidence_sent",
        attempt_id="attempt-1",
        result_count=3,
        source_types=_ALL_SOURCES,
    )

    assert event is not None
    encoded = encode_library_preparation_event(event)
    assert decode_library_preparation_event(encoded) == event
    assert json.loads(encoded)["outcome"] == "evidence_sent"
    assert json.loads(encoded)["result_count"] == 3


def test_a_grounded_send_cannot_record_zero_sources() -> None:
    with pytest.raises(LibraryPreparationValidationError):
        library_preparation_event_for_outcome(
            "evidence_sent",
            attempt_id="attempt-1",
            result_count=0,
            source_types=_ALL_SOURCES,
        )


@pytest.mark.parametrize("reason", ["timed_out", "stopped", "failed"])
def test_a_bypassed_send_keeps_why_it_was_bypassed(reason: str) -> None:
    event = library_preparation_event_for_outcome(
        "bypassed",
        attempt_id="attempt-1",
        result_count=0,
        source_types=_ALL_SOURCES,
        bypass_reason=reason,
    )

    assert event is not None
    encoded = encode_library_preparation_event(event)
    assert json.loads(encoded)["bypass_reason"] == reason
    decoded = decode_library_preparation_event(encoded)
    assert decoded == event
    assert decoded.bypass_reason == reason


def test_a_record_without_a_reason_keeps_its_exact_v1_bytes() -> None:
    """Rows written before the reason existed still decode, unchanged."""
    legacy = (
        '{"version":1,"outcome":"bypassed","attempt_id":"attempt-1",'
        '"result_count":0,"source_types":["notes"]}'
    )

    decoded = decode_library_preparation_event(legacy)

    assert decoded.bypass_reason is None
    assert encode_library_preparation_event(decoded) == legacy


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("zero_matches", "timed_out"),
        ("evidence_sent", "failed"),
        ("bypassed", "the query was secret"),
        ("bypassed", ""),
    ],
)
def test_a_reason_belongs_only_to_a_bypass_and_its_closed_vocabulary(
    outcome: str, reason: str
) -> None:
    with pytest.raises(LibraryPreparationValidationError):
        library_preparation_event_for_outcome(
            outcome,
            attempt_id="attempt-1",
            result_count=1 if outcome == "evidence_sent" else 0,
            source_types=_ALL_SOURCES,
            bypass_reason=reason,
        )


@pytest.mark.parametrize(
    "payload",
    [
        '{"version":1,"outcome":"zero_matches","attempt_id":"a",'
        '"result_count":0,"source_types":["notes"],"bypass_reason":"failed"}',
        '{"version":1,"outcome":"bypassed","attempt_id":"a",'
        '"result_count":0,"source_types":["notes"],"bypass_reason":null}',
        '{"version":1,"outcome":"bypassed","attempt_id":"a",'
        '"result_count":0,"source_types":["notes"],"bypass_reason":"query"}',
        '{"version":1,"outcome":"evidence_sent","attempt_id":"a",'
        '"result_count":0,"source_types":["notes"]}',
    ],
)
def test_the_decoder_rejects_reasons_and_counts_outside_the_contract(
    payload: str,
) -> None:
    with pytest.raises(LibraryPreparationValidationError):
        decode_library_preparation_event(payload)


def test_projection_carries_each_turns_count_and_reason() -> None:
    rows = [
        {
            "message_id": turn_id,
            "turn_id": turn_id,
            "seq": seq,
            "event_kind": "library_preparation",
            "payload_json": encode_library_preparation_event(event),
        }
        for seq, (turn_id, event) in enumerate(
            (
                (
                    "turn-grounded",
                    library_preparation_event_for_outcome(
                        "evidence_sent",
                        attempt_id="a1",
                        result_count=4,
                        source_types=_ALL_SOURCES,
                    ),
                ),
                (
                    "turn-bypassed",
                    library_preparation_event_for_outcome(
                        "bypassed",
                        attempt_id="a2",
                        result_count=0,
                        source_types=_ALL_SOURCES,
                        bypass_reason="stopped",
                    ),
                ),
            ),
            start=1,
        )
    ]

    views = project_library_preparation(rows, ["turn-grounded", "turn-bypassed"])

    assert [(v.turn_id, v.outcome, v.result_count, v.bypass_reason) for v in views] == [
        ("turn-grounded", "evidence_sent", 4, None),
        ("turn-bypassed", "bypassed", 0, "stopped"),
    ]


@pytest.mark.parametrize(
    ("outcome", "count", "reason"),
    [("evidence_sent", 2, None), ("bypassed", 0, "timed_out")],
)
def test_export_keeps_the_new_records_instead_of_dropping_them(
    db, outcome: str, count: int, reason: str | None
) -> None:
    conversation_id = _seed_conversation(db)
    canonical = encode_library_preparation_event(
        library_preparation_event_for_outcome(
            outcome,
            attempt_id="attempt-1",
            result_count=count,
            source_types=_ALL_SOURCES,
            bypass_reason=reason,
        )
    )
    _insert_preparation_row(db, conversation_id, payload_json=canonical)

    for include_payloads in (False, True):
        exported = build_trajectory_export(
            db, conversation_id, include_payloads=include_payloads
        )
        row = next(
            item
            for item in exported["trajectory_rows"]
            if item["event_kind"] == "library_preparation"
        )
        assert row["payload_json"] == canonical
