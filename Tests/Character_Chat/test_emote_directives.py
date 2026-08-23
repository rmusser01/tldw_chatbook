"""Pinned server-compatible character emote directive contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.Character_Chat.emote_directives import (
    EMOTE_EVENT_LIMIT,
    STREAM_PREFIX_BUFFER_LIMIT,
    CharacterEmoteEvent,
    CharacterEmoteStreamParser,
    normalize_character_emote_state,
    parse_character_emote_directives,
)

pytestmark = pytest.mark.unit

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "character_emote_directives.json"
)
_FROZEN_VECTORS = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("vector", _FROZEN_VECTORS, ids=lambda item: item["name"])
def test_one_shot_parser_matches_frozen_cross_language_vectors(vector: dict) -> None:
    """One-shot Python behavior stays pinned to the server/WebUI corpus."""

    parsed = parse_character_emote_directives(vector["input"])

    assert parsed.clean_text == vector["clean_text"]
    assert [
        {"state": event.state, "at_char": event.at_char}
        for event in parsed.events
    ] == vector["events"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Thinking Hard ", "thinking-hard"),
        ("../../bad", None),
        ("a" * 40, "a" * 40),
        ("a" * 41, None),
        (None, None),
    ],
)
def test_state_normalization_is_exact(raw: object, expected: str | None) -> None:
    assert normalize_character_emote_state(raw) == expected


def test_event_contract_and_cap_are_frozen() -> None:
    assert EMOTE_EVENT_LIMIT == 5
    assert CharacterEmoteEvent(state="smug", at_char=0) == CharacterEmoteEvent(
        state="smug",
        at_char=0,
    )


def _stream_chunks(chunks: list[str]) -> tuple[str, list[dict[str, int | str]]]:
    parser = CharacterEmoteStreamParser()
    clean_parts: list[str] = []
    events: list[dict[str, int | str]] = []
    for chunk in chunks:
        result = parser.push(chunk)
        clean_parts.append(result.visible_text)
        events.extend(
            {"state": event.state, "at_char": event.at_char}
            for event in result.events
        )
        assert parser.pending_char_count <= STREAM_PREFIX_BUFFER_LIMIT
    flushed = parser.flush()
    clean_parts.append(flushed.visible_text)
    events.extend(
        {"state": event.state, "at_char": event.at_char}
        for event in flushed.events
    )
    assert parser.flush().visible_text == ""
    assert parser.flush().events == ()
    return "".join(clean_parts), events


@pytest.mark.parametrize("vector", _FROZEN_VECTORS, ids=lambda item: item["name"])
def test_stream_parser_matches_vectors_across_adversarial_chunks(vector: dict) -> None:
    text = vector["input"]
    chunkings = [
        list(text),
        [text[index : index + size] for index, size in _chunk_slices(text)],
        [text],
    ]
    for split_at in range(len(text) + 1):
        chunkings.append([text[:split_at], text[split_at:]])

    for chunks in chunkings:
        clean_text, events = _stream_chunks(chunks)
        assert clean_text == vector["clean_text"]
        assert events == vector["events"]


def _chunk_slices(text: str):
    sizes = (1, 2, 5, 3)
    offset = 0
    index = 0
    while offset < len(text):
        size = sizes[index % len(sizes)]
        yield offset, size
        offset += size
        index += 1


def test_stream_releases_long_ordinary_prose_before_newline() -> None:
    parser = CharacterEmoteStreamParser()
    prose = "This is ordinary prose. " * 5_000

    result = parser.push(prose)

    assert result.visible_text == prose
    assert result.events == ()
    assert parser.pending_char_count == 0


def test_stream_discards_overlong_directive_without_unbounded_buffering() -> None:
    parser = CharacterEmoteStreamParser()
    invalid = "Emote: " + ("x" * 100_000)

    result = parser.push(invalid)

    assert result.visible_text == ""
    assert result.events == ()
    assert parser.pending_char_count <= STREAM_PREFIX_BUFFER_LIMIT
    assert parser.push("\nVisible").visible_text == "Visible"


def test_stream_emits_multiple_same_chunk_events_in_order() -> None:
    parser = CharacterEmoteStreamParser()

    result = parser.push("Emote: smug\nA\nEmote: sad\nB")
    flushed = parser.flush()

    assert result.visible_text == "A\nB"
    assert result.events == (
        CharacterEmoteEvent("smug", 0),
        CharacterEmoteEvent("sad", 2),
    )
    assert flushed.visible_text == ""


@pytest.mark.parametrize("candidate", ["E", "Em", "Emote:", "Emote: smug"])
def test_cancel_discards_incomplete_control_candidates(candidate: str) -> None:
    parser = CharacterEmoteStreamParser()

    assert parser.push("Visible\n" + candidate).visible_text == "Visible\n"
    canceled = parser.cancel()

    assert canceled.visible_text == ""
    assert canceled.events == ()
    assert parser.pending_char_count == 0
    assert parser.flush().visible_text == ""


def test_flush_accepts_unterminated_directive_that_cancel_discards() -> None:
    parser = CharacterEmoteStreamParser()

    assert parser.push("Emote: surprised").events == ()

    assert parser.flush().events == (CharacterEmoteEvent("surprised", 0),)
