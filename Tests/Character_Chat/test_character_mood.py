"""Pinned WebUI-compatible character mood fallback contracts."""

from __future__ import annotations

import pytest

from tldw_chatbook.Character_Chat.character_mood import (
    CHARACTER_MOOD_LABELS,
    CharacterMoodDetection,
    detect_character_mood,
)

pytestmark = pytest.mark.unit


def test_empty_input_has_pinned_neutral_result() -> None:
    assert detect_character_mood(assistant_text="") == CharacterMoodDetection(
        label="neutral",
        confidence=0.4,
        topic=None,
    )


def test_upstream_excited_vector_matches_exact_result() -> None:
    detected = detect_character_mood(
        assistant_text="Wow, this is amazing! Let's go!",
        user_text="Can you celebrate this win with me?",
    )

    assert detected == CharacterMoodDetection(
        label="excited",
        confidence=0.98,
        topic="celebrate",
    )


def test_upstream_neutral_vector_matches_exact_result() -> None:
    assert detect_character_mood(
        assistant_text="Here is the summary of the API response payload."
    ) == CharacterMoodDetection(
        label="neutral",
        confidence=0.5,
        topic="here",
    )


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("I am happy and glad.", "happy"),
        ("Amazing!", "excited"),
        ("I am sad and I am sorry.", "sad"),
        ("This is angry, furious rage.", "angry"),
        ("Maybe we should think?", "thinking"),
        ("I am confused and not sure.", "confused"),
        ("Wow??", "surprised"),
    ],
)
def test_each_pinned_mood_label_is_reachable(text: str, label: str) -> None:
    detected = detect_character_mood(assistant_text=text)

    assert detected.label == label
    assert 0.35 <= detected.confidence <= 0.98


def test_preceding_user_text_participates_in_scoring() -> None:
    detected = detect_character_mood(
        assistant_text="I hear you.",
        user_text="I am furious and angry about this.",
    )

    assert detected.label == "angry"


def test_score_ties_keep_pinned_label_order() -> None:
    detected = detect_character_mood(assistant_text="happy angry")

    assert CHARACTER_MOOD_LABELS == (
        "neutral",
        "happy",
        "excited",
        "sad",
        "angry",
        "thinking",
        "confused",
        "surprised",
    )
    assert detected.label == "happy"
    assert detected.confidence == 0.54


@pytest.mark.parametrize(
    ("text", "label", "confidence"),
    [
        ("?", "neutral", 0.48),
        ("!!", "excited", 0.66),
    ],
)
def test_assistant_punctuation_uses_fractional_scores(
    text: str,
    label: str,
    confidence: float,
) -> None:
    detected = detect_character_mood(assistant_text=text)

    assert detected.label == label
    assert detected.confidence == confidence


def test_topic_uses_user_first_winner_and_is_bounded() -> None:
    assert detect_character_mood(
        assistant_text="beta",
        user_text="alpha beta alpha",
    ).topic == "alpha"
    assert detect_character_mood(assistant_text="x" * 45).topic == "x" * 40
