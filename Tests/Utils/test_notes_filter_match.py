"""The Library Notes list filter's matching rule, one row per case (TASK-34000.12).

Review finding N-09 (notes-library-ux-review-2026-10-02): the list filter
bound the whole input as ONE quoted FTS5 phrase, so ``meet`` missed
"meeting", ``Lisb`` missed "Lisbon", ``size chunk`` missed "chunk size" and
a note's keywords were never consulted. ``build_notes_filter_match`` is the
one rule that replaced it: every whitespace-delimited word is a quoted
literal, the words are AND-ed, the LAST word is a prefix, and the input is
a phrase only when the user wrapped it in a matching pair of double quotes.

Every string the rule produces is built by ``quote_fts5_token`` /
``quote_fts5_phrase`` (TASK-19558: one escape, parameters only), which is
what the rows marked "FTS operators are never honoured" pin. This file runs
in the PR Fast Lane, which installs no Hypothesis; the generated-input
properties live in ``test_notes_filter_match_properties.py``.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Utils.fts5_match_forms import (
    NOTES_FILTER_MAX_TOKENS,
    NOTES_FILTER_MIN_PREFIX_CHARS,
    NotesFilterMatch,
    build_notes_filter_match,
    fts5_query_is_searchable,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "query",
    ["", "   ", ".,;", "\x00", None, 42, '"', '""', "“”", '"   "', "\"\x00\""],
    ids=["empty", "blanks", "punctuation", "nul", "none", "int", "one-quote",
         "two-quotes", "curly-pair", "quoted-blanks", "quoted-nul"],
)
def test_an_unsearchable_input_builds_no_expression(query) -> None:
    """None exactly where the primitive's searchability check says None.

    The quoted cases are the phrase-mode half of the same rule: a pair of
    quotes around nothing searchable is not a phrase to look for.
    """
    assert build_notes_filter_match(query) is None
    assert not fts5_query_is_searchable(query)


@pytest.mark.parametrize(
    ("query", "expressions", "phrase"),
    [
        # the last word is a prefix: "meet" reaches "meeting"
        ("meet", ('"meet"*',), False),
        ("meeting", ('"meeting"*',), False),
        # case is left to unicode61 (the tokenizer folds it)
        ("Lisb", ('"Lisb"*',), False),
        ("lisb", ('"lisb"*',), False),
        ("LISB", ('"LISB"*',), False),
        # any order: every word AND-ed, only the last one starred
        ("size chunk", ('"size"', '"chunk"*'), False),
        ("chunk size", ('"chunk"', '"size"*'), False),
        ("  size   chunk  ", ('"size"', '"chunk"*'), False),
        # quoted input is a phrase, and only quoted input is
        ('"chunk size"', ('"chunk size"',), True),
        ("“chunk size”", ('"chunk size"',), True),
        ('"size chunk"', ('"size chunk"',), True),
        ('"  chunk size  "', ('"chunk size"',), True),
        # an unbalanced or interior quote is ordinary text, doubled by the
        # primitive; a one-character last word is matched whole
        ('"x', ('"""x"',), False),
        ('x"', ('"x"""',), False),
        ('"chunk size”', ('"""chunk"', '"size”"*'), False),
        ('O"Brien', ('"O""Brien"*',), False),
        ('say "hi" now', ('"say"', '"""hi"""', '"now"*'), False),
        ('"a ""b"" c"', ('"a """"b"""" c"',), True),
        # keyword shapes: one token, one quoted prefix phrase
        ("zebra", ('"zebra"*',), False),
        ("homelab", ('"homelab"*',), False),
        ("71d30fb1", ('"71d30fb1"*',), False),
        ("conversation:71d", ('"conversation:71d"*',), False),
        ("zebra nothing", ('"zebra"', '"nothing"*'), False),
        ("zebra meeting", ('"zebra"', '"meeting"*'), False),
        # diacritics and CJK are the tokenizer's business, not the rule's
        ("cafe", ('"cafe"*',), False),
        ("café", ('"café"*',), False),
        ("会議", ('"会議"*',), False),
        # a one-character last word is whole: "a" never fans out
        ("a", ('"a"',), False),
        ("chunk a", ('"chunk"', '"a"'), False),
        # a last word whose final RUN is one character is whole too
        ("obsidian-3", ('"obsidian-3"',), False),
        # FTS operators are never honoured (TASK-19558): literal text
        ("chunk*", ('"chunk*"*',), False),
        ("size OR chunk", ('"size"', '"OR"', '"chunk"*'), False),
        ("title:chunk", ('"title:chunk"*',), False),
        ("NEAR(size chunk)", ('"NEAR(size"', '"chunk)"*'), False),
        ("-size", ('"-size"*',), False),
        # pure-punctuation words are dropped before the last word is chosen
        ("size !!!", ('"size"*',), False),
    ],
)
def test_the_matching_rule_as_a_table(query, expressions, phrase) -> None:
    match = build_notes_filter_match(query)

    assert match == NotesFilterMatch(
        expressions=expressions, phrase=phrase, fts_expression=" ".join(expressions)
    )


def test_word_mode_is_capped_at_the_token_limit_and_stars_the_last_kept_word() -> None:
    words = [f"w{index:02d}" for index in range(40)]

    match = build_notes_filter_match(" ".join(words))

    assert match is not None and not match.phrase
    assert len(match.expressions) == NOTES_FILTER_MAX_TOKENS == 32
    assert match.expressions[:-1] == tuple(f'"{word}"' for word in words[:31])
    assert match.expressions[-1] == '"w31"*'
    assert "w32" not in match.fts_expression


def test_one_very_long_word_is_one_bounded_prefix() -> None:
    match = build_notes_filter_match("x" * 200)

    assert match is not None
    assert match.expressions == (f'"{"x" * 200}"*',)


def test_the_prefix_threshold_is_the_published_constant() -> None:
    assert NOTES_FILTER_MIN_PREFIX_CHARS == 2
    shortest_starred = "y" * NOTES_FILTER_MIN_PREFIX_CHARS
    assert build_notes_filter_match(shortest_starred).expressions == (
        f'"{shortest_starred}"*',
    )
    assert build_notes_filter_match("y").expressions == ('"y"',)


def test_the_result_is_frozen() -> None:
    match = build_notes_filter_match("meet")
    assert match is not None
    with pytest.raises((AttributeError, TypeError)):
        match.phrase = True  # type: ignore[misc]
