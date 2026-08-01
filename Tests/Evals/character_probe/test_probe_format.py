import pytest

from tldw_chatbook.Evals.character_probe.models import Probe, ProbeSet
from tldw_chatbook.Evals.character_probe.probe_format import (
    format_probe_text,
    parse_probe_text,
)


def test_single_probe_single_turn():
    assert parse_probe_text("What do you think about lying?") == ProbeSet(
        probes=(Probe(turns=("What do you think about lying?",)),)
    )


def test_turns_split_on_the_turn_delimiter():
    text = "What do you think about lying?\n---\nAnd if it protected someone?"
    assert parse_probe_text(text) == ProbeSet(
        probes=(
            Probe(
                turns=(
                    "What do you think about lying?",
                    "And if it protected someone?",
                )
            ),
        )
    )


def test_probes_split_on_the_probe_delimiter():
    text = "First probe\n===\nSecond probe"
    parsed = parse_probe_text(text)
    assert len(parsed.probes) == 2
    assert parsed.probes[0].turns == ("First probe",)
    assert parsed.probes[1].turns == ("Second probe",)


def test_a_turn_may_span_multiple_paragraphs():
    """The whole point of the delimiter format: complex prompts are the subject."""
    text = "Describe your earliest memory.\n\nTake your time, and include what you could smell."
    parsed = parse_probe_text(text)
    assert parsed.probes[0].turns == (
        "Describe your earliest memory.\n\nTake your time, and include what you could smell.",
    )


def test_interior_whitespace_is_preserved_exactly():
    text = "Line one\n    indented line\nLine three"
    assert parsed_turn(text) == "Line one\n    indented line\nLine three"


def parsed_turn(text: str) -> str:
    return parse_probe_text(text).probes[0].turns[0]


def test_blank_lines_around_a_turn_are_stripped():
    text = "\n\nWhat is your name?\n\n\n---\n\nAnd your age?\n"
    parsed = parse_probe_text(text)
    assert parsed.probes[0].turns == ("What is your name?", "And your age?")


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="no probes"):
        parse_probe_text("   \n\n  ")


def test_a_probe_with_no_turns_is_rejected():
    with pytest.raises(ValueError, match="probe 2"):
        parse_probe_text("Real probe\n===\n   \n===\nAnother")


def test_round_trip_through_format_and_parse():
    original = ProbeSet(
        probes=(
            Probe(turns=("One\n\nwith a paragraph", "Two")),
            Probe(turns=("Three",)),
        )
    )
    assert parse_probe_text(format_probe_text(original)) == original
