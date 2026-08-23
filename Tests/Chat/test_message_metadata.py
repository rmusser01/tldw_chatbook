"""Structured per-message metadata (task-2364).

The field exists so machine consumers stop inferring facts from UI copy:
the reseed builder, exports and summaries read `interrupted` instead of
matching the visible "⏹ interrupted" marker, and an empty realtime user
row records WHY it is empty instead of stranding silently.
"""

import json

import pytest

from tldw_chatbook.Chat.message_metadata import (
    CHARACTER_EMOTE_FALLBACK_REASONS,
    MESSAGE_ORIGIN_AGENT_WAKE,
    MESSAGE_ORIGINS,
    TEMPLATE_KINDS,
    TRANSCRIPT_STATUSES,
    CharacterEmoteEventMetadata,
    CharacterEmoteMetadata,
    MessageMetadata,
)


def test_defaults_are_the_no_metadata_state():
    metadata = MessageMetadata()

    assert metadata.engine == ""
    assert metadata.provider == ""
    assert metadata.model == ""
    assert metadata.interrupted is False
    assert metadata.transcript_status == ""
    assert metadata.template_kind == ""
    assert metadata.template_source == ""
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


def test_character_greeting_provenance_round_trips():
    metadata = MessageMetadata(
        template_kind="character_greeting",
        template_source="Hello {{user}}.",
    )

    assert MessageMetadata.from_json(metadata.to_json()) == metadata


def test_unknown_template_kind_degrades_and_drops_source():
    restored = MessageMetadata.from_json(
        json.dumps({"template_kind": "future_kind", "template_source": "secret"})
    )

    assert restored is not None
    assert restored.template_kind == ""
    assert restored.template_source == ""


def test_template_source_requires_the_closed_kind():
    with pytest.raises(ValueError, match="template_source"):
        MessageMetadata(template_source="Hello {{user}}")


def test_template_kind_is_refused_outside_closed_vocabulary():
    with pytest.raises(ValueError, match="template_kind"):
        MessageMetadata(template_kind="future_kind", template_source="Hello {{user}}")

    for kind in TEMPLATE_KINDS:
        if kind:
            assert (
                MessageMetadata(
                    template_kind=kind, template_source="Hello"
                ).template_kind
                == kind
            )
        else:
            assert MessageMetadata(template_kind=kind).template_kind == kind


def test_template_kind_requires_a_nonblank_source():
    with pytest.raises(ValueError, match="template_source"):
        MessageMetadata(template_kind="character_greeting")


def test_from_json_returns_none_for_missing_or_corrupt_payloads():
    """Legacy rows (written before the column existed) and any garbage that
    reaches the column must degrade to "no metadata known", never raise --
    this runs on the resume path."""
    assert MessageMetadata.from_json(None) is None
    assert MessageMetadata.from_json("") is None
    assert MessageMetadata.from_json("not json") is None
    assert MessageMetadata.from_json("[1, 2, 3]") is None


@pytest.mark.parametrize("template_source", [None, "", "   "])
def test_from_json_drops_incomplete_character_greeting_provenance(
    template_source,
):
    """A damaged provenance pair must not prevent conversation resume."""
    restored = MessageMetadata.from_json(
        json.dumps(
            {
                "engine": "pipeline",
                "template_kind": "character_greeting",
                "template_source": template_source,
            }
        )
    )

    assert restored == MessageMetadata(engine="pipeline")


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


# ---------------------------------------------------------------------------
# PR3a-2 Task 5: the machine-origin marking on auto-wake notice rows.
# ---------------------------------------------------------------------------


def test_agent_wake_origin_round_trips():
    """The wake notice's not-user-input marking is a durable machine fact:
    it must survive persistence and resume exactly, not live only in the
    row's visible copy (the pre-task-2364 failure mode this module exists
    to prevent)."""
    metadata = MessageMetadata(origin=MESSAGE_ORIGIN_AGENT_WAKE)

    assert metadata.is_empty is False
    raw = metadata.to_json()
    assert json.loads(raw)["origin"] == "agent_wake"
    assert MessageMetadata.from_json(raw) == metadata


def test_unknown_origin_is_refused_at_construction():
    """Closed vocabulary, same rule as transcript_status: a typo'd origin
    fails where it is written, never silently un-matching every reader."""
    with pytest.raises(ValueError):
        MessageMetadata(origin="agent-wake")

    for origin in MESSAGE_ORIGINS:
        assert MessageMetadata(origin=origin).origin == origin


def test_from_json_degrades_an_unrecognised_origin_to_blank():
    """A payload written by a NEWER build's wider vocabulary must load (a
    resume can never fail over metadata) but must not pass the unknown
    value through as if this build understood it."""
    restored = MessageMetadata.from_json(
        json.dumps({"engine": "realtime", "origin": "from_the_future"})
    )

    assert restored == MessageMetadata(engine="realtime")
    assert restored.origin == ""


def _emote_metadata() -> CharacterEmoteMetadata:
    return CharacterEmoteMetadata(
        mood_label="sad",
        mood_confidence=None,
        mood_topic=None,
        emote_events=(
            CharacterEmoteEventMetadata("smug", 0),
            CharacterEmoteEventMetadata("sad", 8),
        ),
        sanitized_utf16_length=12,
        actor_kind="character",
        actor_id=7,
        pack_id=11,
        pack_version_id=13,
        expression_key="sad",
        expression_id=17,
        asset_id=19,
        fallback_reason="",
    )


def test_character_emote_metadata_round_trips_as_bounded_scalars() -> None:
    metadata = MessageMetadata(character_emote=_emote_metadata())

    payload = json.loads(metadata.to_json())

    assert payload["character_emote"]["emote_events"] == [
        {"at_char": 0, "state": "smug"},
        {"at_char": 8, "state": "sad"},
    ]
    assert MessageMetadata.from_json(metadata.to_json()) == metadata


@pytest.mark.parametrize(
    "events",
    [
        tuple(CharacterEmoteEventMetadata(f"state-{index}", index) for index in range(6)),
        (CharacterEmoteEventMetadata("sad", 9), CharacterEmoteEventMetadata("happy", 8)),
        (CharacterEmoteEventMetadata("sad", 13),),
    ],
)
def test_character_emote_event_bounds_are_strict(events) -> None:
    with pytest.raises(ValueError):
        CharacterEmoteMetadata(
            mood_label=events[-1].state,
            emote_events=events,
            sanitized_utf16_length=12,
        )


def test_character_emote_final_explicit_state_must_match_mood() -> None:
    with pytest.raises(ValueError, match="mood_label"):
        CharacterEmoteMetadata(
            mood_label="happy",
            emote_events=(CharacterEmoteEventMetadata("sad", 0),),
            sanitized_utf16_length=3,
        )


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "7"])
def test_character_emote_local_identities_are_positive_integers(value) -> None:
    with pytest.raises(ValueError):
        CharacterEmoteMetadata(
            mood_label="neutral",
            sanitized_utf16_length=0,
            actor_kind="character",
            actor_id=value,
        )


def test_character_emote_fallback_vocabulary_is_closed() -> None:
    with pytest.raises(ValueError, match="fallback_reason"):
        CharacterEmoteMetadata(
            mood_label="neutral",
            sanitized_utf16_length=0,
            fallback_reason="/private/path/to/asset.png",
        )

    for reason in CHARACTER_EMOTE_FALLBACK_REASONS:
        assert CharacterEmoteMetadata(
            mood_label="neutral",
            sanitized_utf16_length=0,
            fallback_reason=reason,
        ).fallback_reason == reason


@pytest.mark.parametrize(
    "bad_emote",
    [
        [],
        {"mood_label": "../../bad", "sanitized_utf16_length": 0},
        {"mood_label": "sad", "sanitized_utf16_length": True},
        {
            "mood_label": "sad",
            "sanitized_utf16_length": 2,
            "emote_events": [{"state": "sad", "at_char": 3}],
        },
        {
            "mood_label": "sad",
            "sanitized_utf16_length": 2,
            "assistant_text": "Emote: sad",
        },
        {
            "mood_label": "sad",
            "sanitized_utf16_length": 2,
            "actor_id": "server-character-id",
        },
    ],
)
def test_malformed_character_emote_load_drops_only_nested_record(bad_emote) -> None:
    restored = MessageMetadata.from_json(
        json.dumps({"engine": "pipeline", "character_emote": bad_emote})
    )

    assert restored == MessageMetadata(engine="pipeline")


def test_character_emote_payload_has_no_content_or_path_fields() -> None:
    payload = json.dumps(json.loads(MessageMetadata(character_emote=_emote_metadata()).to_json()))

    for forbidden in (
        "assistant_text",
        "directive",
        "prompt",
        "provider_payload",
        "storage_relpath",
        "manual_override",
        "server_id",
    ):
        assert forbidden not in payload
