"""Pinned server-compatible character emote directive contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tldw_chatbook.Character_Chat.emote_directives as emote_directives_module
from tldw_chatbook.Character_Chat.emote_directives import (
    EMOTE_EVENT_LIMIT,
    STREAM_PREFIX_BUFFER_LIMIT,
    CharacterEmoteEvent,
    CharacterEmoteStreamParser,
    append_character_emote_prompt_instruction,
    normalize_character_emote_state,
    parse_character_emote_directives,
    project_character_emote_assets,
    project_character_emote_states,
    utf16_length,
)

pytestmark = pytest.mark.unit

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "character_emote_directives.json"
)
_FROZEN_VECTORS = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def test_state_normalizer_docstring_documents_public_contract() -> None:
    docstring = normalize_character_emote_state.__doc__ or ""

    assert "\n    Args:" in docstring
    assert "\n    Returns:" in docstring


@pytest.mark.parametrize("vector", _FROZEN_VECTORS, ids=lambda item: item["name"])
def test_one_shot_parser_matches_frozen_cross_language_vectors(vector: dict) -> None:
    """One-shot Python behavior stays pinned to the server/WebUI corpus."""

    parsed = parse_character_emote_directives(vector["input"])

    assert parsed.clean_text == vector["clean_text"]
    assert [
        {"state": event.state, "at_char": event.at_char} for event in parsed.events
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
            {"state": event.state, "at_char": event.at_char} for event in result.events
        )
        assert parser.pending_char_count <= STREAM_PREFIX_BUFFER_LIMIT
    flushed = parser.flush()
    clean_parts.append(flushed.visible_text)
    events.extend(
        {"state": event.state, "at_char": event.at_char} for event in flushed.events
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


def _drive_stream(
    parser: CharacterEmoteStreamParser,
    chunks: list[str],
) -> tuple[str, tuple[CharacterEmoteEvent, ...], int]:
    visible: list[str] = []
    events: list[CharacterEmoteEvent] = []
    for chunk in chunks:
        result = parser.push(chunk)
        visible.append(result.visible_text)
        events.extend(result.events)
    flushed = parser.flush()
    visible.append(flushed.visible_text)
    events.extend(flushed.events)
    return "".join(visible), tuple(events), parser._clean_length


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param(
            ["Emo", "te: smug\nHello\nWorld"],
            id="directive-split-across-chunks-mid-prefix",
        ),
        pytest.param(
            ["Before\nEmote", ": sad\nAfter"],
            id="chunk-ends-mid-emote-marker",
        ),
        pytest.param(
            ["Text\n``", "`\ncode\nEmote: hidden\n``", "`\nAfter"],
            id="chunk-ends-mid-fence-marker",
        ),
        pytest.param(
            ["Emote: a\nEmote: b\n", "Emote: c\nVisible tail"],
            id="back-to-back-directives",
        ),
        pytest.param(
            ["Hi \U0001f600 there\nEmo", "te: happy\nTail \U0001f389 end"],
            id="astral-utf16-offsets",
        ),
        pytest.param(
            list("Emote: smug\nA\n```\nEmote: x\n```\nEmote: sad\nB"),
            id="one-char-chunk-stream",
        ),
    ],
)
def test_run_publishing_is_equivalent_to_char_by_char(chunks: list[str]) -> None:
    """TASK-22227: run publishing must match per-character publishing exactly.

    Same visible text, same events (states AND UTF-16 offsets), and the same
    internal clean-length accumulator, for the given chunking versus the
    degenerate one-character stream versus the one-shot parser.
    """

    text = "".join(chunks)
    run_visible, run_events, run_clean = _drive_stream(
        CharacterEmoteStreamParser(), chunks
    )
    char_visible, char_events, char_clean = _drive_stream(
        CharacterEmoteStreamParser(), list(text)
    )
    oneshot = parse_character_emote_directives(text)

    assert run_visible == char_visible == oneshot.clean_text
    assert run_events == char_events == oneshot.events
    assert run_clean == char_clean == utf16_length(run_visible)


def test_stream_publishes_runs_not_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-22227: visible text is published in newline/chunk-bounded runs.

    The per-character implementation paid one ``utf16_length`` encode per
    visible character (~16k for a 16k-char reply); runs pay at most one per
    chunk continuation plus a small constant per line.
    """

    calls = {"count": 0}
    real_utf16_length = emote_directives_module.utf16_length

    def counting_utf16_length(value: str) -> int:
        calls["count"] += 1
        return real_utf16_length(value)

    monkeypatch.setattr(
        emote_directives_module, "utf16_length", counting_utf16_length
    )

    paragraph = (
        "The rain had not stopped since morning, and the streets shone like "
        "polished slate under the gas lamps as she counted the doorways.\n"
    )
    parts: list[str] = [paragraph]
    for state in ("thinking", "surprised", "happy", "sad"):
        parts.extend([f"Emote: {state}\n", paragraph * 31])
    reply = "".join(parts)
    assert len(reply) > 16_000
    chunks = [reply[index : index + 64] for index in range(0, len(reply), 64)]

    visible, events, _clean = _drive_stream(CharacterEmoteStreamParser(), chunks)
    stream_publish_calls = calls["count"]

    oneshot = parse_character_emote_directives(reply)
    assert visible == oneshot.clean_text
    assert events == oneshot.events
    assert stream_publish_calls <= len(chunks) + 2 * reply.count("\n") + 8


def test_prompt_projection_uses_only_round_tripping_canonical_keys() -> None:
    assets = [
        {"expression_key": "neutral", "display_label": "Wrong label"},
        {"expression_key": "custom:quiet_focus", "display_label": "Ignore me"},
        {"expression_key": "joy", "display_label": "Alias"},
        {"expression_key": "custom:joy", "display_label": "Alias collision"},
        {"expression_key": "custom:thinking-hard"},
        {"expression_key": "../../bad"},
        {"expression_key": "happy"},
        {"expression_key": "neutral"},
        {"display_label": "label-only-must-not-project"},
    ]

    assert project_character_emote_states(assets) == (
        "neutral",
        "quiet_focus",
        "happy",
    )


def test_asset_projection_maps_states_to_first_round_tripping_sources() -> None:
    """TASK-22227: the slug->source map mirrors the states projection exactly."""

    assets = [
        {"expression_key": "neutral", "id": 1},
        {"expression_key": "custom:quiet_focus", "id": 2},
        {"expression_key": "joy", "id": 3},
        {"expression_key": "custom:joy", "id": 4},
        {"expression_key": "../../bad", "id": 5},
        {"expression_key": "happy", "id": 6},
        {"expression_key": "neutral", "id": 7},
        {"display_label": "label-only-must-not-project", "id": 8},
    ]

    projected = project_character_emote_assets(assets)

    assert tuple(projected) == project_character_emote_states(assets)
    assert [(state, source["id"]) for state, source in projected.items()] == [
        ("neutral", 1),
        ("quiet_focus", 2),
        ("happy", 6),
    ]


def test_asset_projection_normalizes_each_asset_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-22227: projection is O(assets) -- two normalize calls per asset."""

    calls = {"normalize_state": 0, "normalize_key": 0}
    real_normalize_state = emote_directives_module.normalize_character_emote_state
    real_normalize_key = emote_directives_module.normalize_expression_key

    def counting_state(value: object) -> str | None:
        calls["normalize_state"] += 1
        return real_normalize_state(value)

    def counting_key(value: str) -> str | None:
        calls["normalize_key"] += 1
        return real_normalize_key(value)

    monkeypatch.setattr(
        emote_directives_module, "normalize_character_emote_state", counting_state
    )
    monkeypatch.setattr(
        emote_directives_module, "normalize_expression_key", counting_key
    )

    assets = [
        {"expression_key": f"custom:state_{index:02d}", "id": index + 1}
        for index in range(40)
    ]
    projected = project_character_emote_assets(assets)

    assert len(projected) == 40
    assert calls["normalize_state"] <= len(assets)
    assert calls["normalize_key"] <= len(assets)


def test_prompt_projection_keeps_first_asset_order_and_caps_instruction() -> None:
    assets = [{"expression_key": f"custom:state_{index}"} for index in range(27)]
    states = project_character_emote_states(assets)

    assert states == tuple(f"state_{index}" for index in range(27))
    instruction = append_character_emote_prompt_instruction(" Base prompt ", states)
    visible = ", ".join(f"state_{index}" for index in range(25))
    assert instruction == (
        "Base prompt\n\n"
        "When the character expression should change, emit a standalone line exactly "
        "like `Emote: <state>`. Prefer these available states: "
        f"{visible} (+2 more). Do not emit an emote after every sentence."
    )


def test_prompt_instruction_handles_empty_base_and_inventory() -> None:
    assert append_character_emote_prompt_instruction("", ()) == (
        "When the character expression should change, emit a standalone line exactly "
        "like `Emote: <state>`. Do not emit an emote after every sentence."
    )
