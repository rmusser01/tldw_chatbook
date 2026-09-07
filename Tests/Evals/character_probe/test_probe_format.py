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


def test_whitespace_only_lines_are_stripped():
    """Trailing spaces on an empty line should not corrupt the turn."""
    text = "  \nHello\n---\nSecond"
    parsed = parse_probe_text(text)
    assert parsed.probes[0].turns == ("Hello", "Second")


def test_stray_turn_delimiter_raises_error():
    """A duplicated or stray turn delimiter in the middle should raise."""
    text = "First\n---\n---\nSecond"
    with pytest.raises(ValueError, match="stray or duplicated"):
        parse_probe_text(text)


def test_probe_rejects_empty_turn():
    """Attempting to create a probe with an empty turn should raise."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        Probe(turns=("",))


def test_probe_rejects_whitespace_only_turn():
    """Attempting to create a probe with a whitespace-only turn should raise."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        Probe(turns=("   ",))


def test_turn_containing_bare_delimiter_does_not_round_trip():
    """Document the lossy behavior: delimiters within turn text are not escaped."""
    # This turn contains a bare --- line
    original_turn = "before\n---\nafter"
    probe = Probe(turns=(original_turn,))
    formatted = format_probe_text(ProbeSet(probes=(probe,)))
    parsed = parse_probe_text(formatted)
    # The turn is split into two at the --- delimiter
    assert len(parsed.probes[0].turns) == 2
    assert parsed.probes[0].turns == ("before", "after")


def test_a_whitespace_padded_delimiter_still_delimits():
    """Deliberate leniency, pinned so it cannot drift either way.

    A strict match would make an INVISIBLE trailing space silently fail to
    delimit, merging two turns into one prompt that then runs and produces
    plausible-looking results -- likely (editors add trailing whitespace
    constantly) and very hard to see. The lenient match's cost is that an
    indented literal --- inside a turn is eaten, which is less likely and
    fails visibly. See the module docstring for the full ruling.
    """
    parsed = parse_probe_text("First turn\n  ---  \nSecond turn")
    assert parsed.probes[0].turns == ("First turn", "Second turn")


def test_a_whitespace_padded_probe_delimiter_still_delimits():
    parsed = parse_probe_text("Probe one\n\t===\nProbe two")
    assert [p.turns for p in parsed.probes] == [("Probe one",), ("Probe two",)]


def test_an_indented_delimiter_inside_a_turn_is_eaten():
    """The documented cost of the leniency above -- pinned so the limitation
    is deliberate rather than a surprise, matching how the bare-delimiter
    round-trip loss is already pinned."""
    parsed = parse_probe_text("before\n    ---\nafter")
    assert parsed.probes[0].turns == ("before", "after")
