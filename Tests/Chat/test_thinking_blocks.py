"""Canonical Console thinking-envelope contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
    ThinkingEnvelopeValidationError,
    MAX_THINKING_BLOCKS,
    MAX_THINKING_BLOCK_ID_CHARS,
    MAX_THINKING_ENVELOPE_BYTES,
    MAX_THINKING_PROVENANCE_CHARS,
    MAX_THINKING_TEXT_BYTES,
    dump_thinking_blocks_json,
    normalize_thinking_history_policy,
    parse_thinking_blocks_json,
    read_thinking_blocks_json,
)


def test_thinking_envelope_round_trips_displayable_and_proprietary() -> None:
    envelope = ThinkingEnvelope(
        blocks=(
            DisplayableThinkingBlock(
                block_id="round-0",
                round_ordinal=0,
                provider="llama_cpp",
                model="qwen3",
                protocol="openai_chat",
                source_format="start_anchored_think",
                status="complete",
                text="  deliberate\nreasoning  ",
            ),
            ProprietaryThinkingBlock(
                block_id="round-1",
                round_ordinal=1,
                provider="moonshot",
                model="kimi-k3",
                protocol="chat_completions",
                source_format="reasoning_content",
                status="complete",
            ),
        )
    )

    raw = dump_thinking_blocks_json(envelope)

    assert parse_thinking_blocks_json(raw) == envelope
    assert "deliberate" not in repr(envelope)


@pytest.mark.parametrize("value", [None, ""])
def test_nullable_or_missing_history_policy_resolves_to_auto(value: object) -> None:
    assert normalize_thinking_history_policy(value) == "auto"


def _displayable(**changes: object) -> dict[str, object]:
    block: dict[str, object] = {
        "block_id": "block-0",
        "round_ordinal": 0,
        "provider": "llama_cpp",
        "model": "qwen3",
        "protocol": "openai_chat",
        "source_format": "start_anchored_think",
        "status": "complete",
        "visibility": "displayable",
        "text": "visible thinking",
    }
    block.update(changes)
    return block


def _envelope(*blocks: dict[str, object], version: object = 1) -> dict[str, object]:
    return {"version": version, "blocks": list(blocks)}


@pytest.mark.parametrize("value", [123, True, ["not", "an", "envelope"]])
def test_parse_rejects_non_string_json_input(value: object) -> None:
    with pytest.raises(ThinkingEnvelopeValidationError):
        parse_thinking_blocks_json(value)


def test_parse_rejects_boolean_ordinal() -> None:
    with pytest.raises(ThinkingEnvelopeValidationError, match="round_ordinal"):
        parse_thinking_blocks_json(
            json.dumps(_envelope(_displayable(round_ordinal=True)))
        )


@pytest.mark.parametrize(
    "value",
    [
        _envelope(_displayable(extra="no")),
        {"version": 1, "blocks": [], "extra": "no"},
        _envelope(_displayable(visibility="proprietary", text="must not exist")),
        _envelope(
            {key: value for key, value in _displayable().items() if key != "text"}
        ),
    ],
)
def test_parse_rejects_unknown_or_visibility_invalid_keys(
    value: dict[str, object],
) -> None:
    with pytest.raises(ThinkingEnvelopeValidationError):
        parse_thinking_blocks_json(json.dumps(value))


@pytest.mark.parametrize(
    "block",
    [
        _displayable(block_id=""),
        _displayable(text=""),
        _displayable(text="x" * (MAX_THINKING_TEXT_BYTES + 1)),
        _displayable(text="é" * (MAX_THINKING_TEXT_BYTES // 2 + 1)),
    ],
)
def test_parse_rejects_invalid_or_oversized_displayable_text(
    block: dict[str, object],
) -> None:
    with pytest.raises(ThinkingEnvelopeValidationError):
        parse_thinking_blocks_json(json.dumps(_envelope(block)))


def test_parse_rejects_duplicate_ids_and_non_monotonic_ordinals() -> None:
    duplicate = _envelope(_displayable(), _displayable(round_ordinal=1))
    unordered = _envelope(
        _displayable(round_ordinal=1), _displayable(block_id="block-1", round_ordinal=0)
    )

    for value in (duplicate, unordered):
        with pytest.raises(ThinkingEnvelopeValidationError):
            parse_thinking_blocks_json(json.dumps(value))


def test_parse_enforces_block_and_envelope_bounds() -> None:
    blocks = [
        _displayable(block_id=f"block-{index}", round_ordinal=index)
        for index in range(MAX_THINKING_BLOCKS + 1)
    ]
    oversized_envelope = _envelope(
        *[
            _displayable(
                block_id=f"large-{index}",
                round_ordinal=index,
                text="x" * MAX_THINKING_TEXT_BYTES,
            )
            for index in range(5)
        ]
    )

    with pytest.raises(ThinkingEnvelopeValidationError):
        parse_thinking_blocks_json(json.dumps(_envelope(*blocks)))
    assert (
        len(json.dumps(oversized_envelope).encode("utf-8"))
        > MAX_THINKING_ENVELOPE_BYTES
    )
    with pytest.raises(ThinkingEnvelopeValidationError):
        parse_thinking_blocks_json(json.dumps(oversized_envelope))


def test_character_boundaries_allow_non_ascii_identifier_and_provenance() -> None:
    block = _displayable(
        block_id="é" * MAX_THINKING_BLOCK_ID_CHARS,
        provider="é" * MAX_THINKING_PROVENANCE_CHARS,
        model="é" * MAX_THINKING_PROVENANCE_CHARS,
        protocol="é" * MAX_THINKING_PROVENANCE_CHARS,
        source_format="é" * MAX_THINKING_PROVENANCE_CHARS,
    )

    assert parse_thinking_blocks_json(json.dumps(_envelope(block))) == ThinkingEnvelope(
        blocks=(
            DisplayableThinkingBlock(
                block_id="é" * MAX_THINKING_BLOCK_ID_CHARS,
                round_ordinal=0,
                provider="é" * MAX_THINKING_PROVENANCE_CHARS,
                model="é" * MAX_THINKING_PROVENANCE_CHARS,
                protocol="é" * MAX_THINKING_PROVENANCE_CHARS,
                source_format="é" * MAX_THINKING_PROVENANCE_CHARS,
                status="complete",
                text="visible thinking",
            ),
        )
    )


def test_parse_accepts_oversized_raw_json_when_canonical_envelope_is_bounded() -> None:
    raw = "{\n" + (" " * MAX_THINKING_ENVELOPE_BYTES) + '"version":1,"blocks":[]}'

    assert len(raw.encode("utf-8")) > MAX_THINKING_ENVELOPE_BYTES
    assert parse_thinking_blocks_json(raw) == ThinkingEnvelope(blocks=())


def test_unknown_durable_version_is_preserved_while_direct_parse_rejects_it() -> None:
    raw = '{"version":2,"blocks":[]}'

    with pytest.raises(ThinkingEnvelopeValidationError):
        parse_thinking_blocks_json(raw)

    result = read_thinking_blocks_json(raw)
    assert result.envelope is None
    assert result.opaque_json == raw
    assert result.warning is not None
    assert not result.generation_actions_enabled


def test_oversized_unknown_durable_version_is_not_retained_opaquely() -> None:
    raw = json.dumps({"version": 2, "blocks": ["x" * MAX_THINKING_ENVELOPE_BYTES]})

    result = read_thinking_blocks_json(raw)

    assert result.envelope is None
    assert result.opaque_json is None
    assert result.warning is not None
    assert "envelope size" in result.warning
    assert result.generation_actions_enabled


def test_malformed_supported_durable_data_is_discarded_without_content() -> None:
    result = read_thinking_blocks_json('{"version":1,"blocks":[{"text":"secret"}]}')

    assert result.envelope is None
    assert result.opaque_json is None
    assert result.warning is not None
    assert "allowed keys" in result.warning
    assert "secret" not in result.warning
    assert result.generation_actions_enabled


def test_block_types_are_immutable_and_proprietary_data_is_structurally_absent() -> (
    None
):
    displayable = DisplayableThinkingBlock(
        block_id="block-0",
        round_ordinal=0,
        provider="llama_cpp",
        model="qwen3",
        protocol="openai_chat",
        source_format="start_anchored_think",
        status="complete",
        text="private to repr",
    )
    proprietary = ProprietaryThinkingBlock(
        block_id="block-1",
        round_ordinal=1,
        provider="moonshot",
        model="kimi-k3",
        protocol="chat_completions",
        source_format="reasoning_content",
        status="complete",
    )

    with pytest.raises(FrozenInstanceError):
        displayable.text = "changed"  # type: ignore[misc]
    assert not hasattr(proprietary, "text")
    assert "private to repr" not in repr(displayable)
