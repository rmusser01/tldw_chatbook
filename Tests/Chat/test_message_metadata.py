"""Structured per-message metadata (task-2364).

The field exists so machine consumers stop inferring facts from UI copy:
the reseed builder, exports and summaries read `interrupted` instead of
matching the visible "⏹ interrupted" marker, and an empty realtime user
row records WHY it is empty instead of stranding silently.
"""

import json

import pytest

from tldw_chatbook.Chat.message_metadata import (
    TRANSCRIPT_STATUSES,
    MessageMetadata,
)


def test_defaults_are_the_no_metadata_state():
    metadata = MessageMetadata()

    assert metadata.engine == ""
    assert metadata.provider == ""
    assert metadata.model == ""
    assert metadata.interrupted is False
    assert metadata.transcript_status == ""
    assert metadata.is_empty is True


def test_any_populated_field_makes_it_non_empty():
    assert MessageMetadata(engine="realtime").is_empty is False
    assert MessageMetadata(interrupted=True).is_empty is False
    assert MessageMetadata(transcript_status="pending").is_empty is False


def test_instances_are_frozen_and_reject_invented_keys():
    """A free-form dict lets every caller invent its own vocabulary; the
    whole point of the dataclass is that they cannot."""
    metadata = MessageMetadata(engine="realtime")

    with pytest.raises(Exception):
        metadata.engine = "pipeline"  # type: ignore[misc]
    with pytest.raises(TypeError):
        MessageMetadata(engine="realtime", latency_ms=12)  # type: ignore[call-arg]


def test_unknown_transcript_status_is_refused_at_construction():
    """Closed vocabulary: a typo must fail where it is written, not months
    later when a reader silently never matches it."""
    with pytest.raises(ValueError):
        MessageMetadata(transcript_status="done")

    for status in TRANSCRIPT_STATUSES:
        assert MessageMetadata(transcript_status=status).transcript_status == status


def test_json_round_trip_preserves_every_field():
    metadata = MessageMetadata(
        engine="realtime",
        provider="openai",
        model="gpt-realtime",
        interrupted=True,
        transcript_status="final",
    )

    raw = metadata.to_json()
    assert json.loads(raw)["engine"] == "realtime"
    assert MessageMetadata.from_json(raw) == metadata


def test_from_json_returns_none_for_missing_or_corrupt_payloads():
    """Legacy rows (written before the column existed) and any garbage that
    reaches the column must degrade to "no metadata known", never raise --
    this runs on the resume path."""
    assert MessageMetadata.from_json(None) is None
    assert MessageMetadata.from_json("") is None
    assert MessageMetadata.from_json("not json") is None
    assert MessageMetadata.from_json("[1, 2, 3]") is None


def test_from_json_drops_unknown_keys_and_unknown_statuses():
    """A row written by a newer build (or hand-edited) must not crash a
    resume, and an unrecognised status is reported as "unknown", not
    smuggled through as if this build understood it."""
    restored = MessageMetadata.from_json(
        json.dumps(
            {
                "engine": "realtime",
                "interrupted": 1,
                "transcript_status": "teleported",
                "latency_ms": 12,
            }
        )
    )

    assert restored == MessageMetadata(engine="realtime", interrupted=True)
    assert restored.transcript_status == ""


@pytest.mark.parametrize(
    "stored, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("True", True),
        ("false", False),
        ("False", False),
        ("1", True),
        ("0", False),
        ("garbage", False),
        ("", False),
        (1, True),
        (0, False),
        (None, False),
        ([], False),
    ],
)
def test_interrupted_survives_every_shape_a_payload_can_carry_it_in(stored, expected):
    """Qodo Q1: `bool("false")` is True. A JSON payload that spells the flag
    as a STRING -- a hand-edited row, a foreign writer, an older/newer
    serializer -- would restore a NOT-interrupted reply as interrupted,
    silently inverting a durable fact on resume and in exports."""
    restored = MessageMetadata.from_json(json.dumps({"interrupted": stored}))

    assert restored is not None
    assert restored.interrupted is expected
