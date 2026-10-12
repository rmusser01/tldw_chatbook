"""Qodo #8 (PR #2921): the bounded text model for user-typed Dreams phrases
(a topic or the region in Settings, a goal in the goals modal).

That text is embedded in the query-synthesis prompt and, on the
deterministic fallback, in an outbound search query, so it is validated at
the entry point rather than persisted after a bare ``strip()``.

Review round 3, item 8 reshaped the rule: it is about what the text IS (one
visible line, bounded, no hidden characters), not about what it mentions.
The script-token check is gone -- nothing renders this text as HTML or
script, and it refused ordinary programming topics.

Round 4 (C4, I3) stopped it refusing text people type -- subdivision
flags, private-use and not-yet-assigned code points, visible format
characters, Thai typed with zero width spaces -- made it remove harmless
invisible characters instead of refusing them, required something
readable (punctuation alone is not a topic), and made each refusal name
the one thing that was wrong.
"""

from __future__ import annotations

import unicodedata

import pytest

from tldw_chatbook.Utils import input_validation
from tldw_chatbook.Utils.input_validation import (
    DREAMS_PROFILE_TEXT_MAX_LENGTH,
    validate_dreams_profile_text,
)

_FAMILY = "family \U0001f468‍\U0001f469‍\U0001f467"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("visit Japan", "visit Japan"),
        ("  jazz guitar  ", "jazz guitar"),
        ("\u65e5\u672c\u306e\u796d\u308a\u30002027", "\u65e5\u672c\u306e\u796d\u308a 2027"),
        ("x" * DREAMS_PROFILE_TEXT_MAX_LENGTH, "x" * DREAMS_PROFILE_TEXT_MAX_LENGTH),
        # Angle brackets, Rich-style brackets and braces are ordinary text.
        ("C++ <templates> and a<b comparisons", "C++ <templates> and a<b comparisons"),
        ("<b>bold</b> moves", "<b>bold</b> moves"),
        ("[b]bold[/b] moves", "[b]bold[/b] moves"),
        ("flights to {region}", "flights to {region}"),
        # No script rule (round 3, item 8): a phrase ABOUT scripting is a
        # perfectly good interest, and nothing renders it as script.
        ("JavaScript: The Good Parts", "JavaScript: The Good Parts"),
        ("why onclick= handlers are bad", "why onclick= handlers are bad"),
        ("<script>alert(1)</script> explained", "<script>alert(1)</script> explained"),
        # Invisible characters that real text needs stay allowed: the
        # joiner inside an emoji sequence, the non-joiner Persian uses,
        # the directional MARKS, and variation selectors.
        (_FAMILY, _FAMILY),
        ("می‌خواهم", "می‌خواهم"),
        ("שלום‏ 2027", "שלום‏ 2027"),
        ("abc‎def", "abc‎def"),
        ("abc؜def", "abc؜def"),
        ("❤️ music", "❤️ music"),
        ("\U0001f1ef\U0001f1f5 travel", "\U0001f1ef\U0001f1f5 travel"),
        # Combining marks: Arabic with harakat, Hebrew with niqqud.
        ("مُحَمَّد", "مُحَمَّد"),
        ("שָׁלוֹם", "שָׁלוֹם"),
        # A no-break space INSIDE text is a space -- stored as one (N5).
        ("jazz\u00a0guitar", "jazz guitar"),
    ],
)
def test_accepts_plain_text_and_returns_it_trimmed(value, expected):
    # The stored form is NFC, which may reorder stacked combining marks
    # (harakat, niqqud) without changing the text.
    assert validate_dreams_profile_text(value) == unicodedata.normalize(
        "NFC", expected)


_TOO_LONG, _ONE_LINE, _NOTHING, _DIRECTION, _HIDDEN, _NOT_TEXT = (
    "longer than", "single line", "nothing readable", "text-direction",
    "invisible formatting", "not valid text")

_ENGLAND = "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
_SCOTLAND = "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f"
_WALES = "\U0001f3f4\U000e0067\U000e0062\U000e0077\U000e006c\U000e0073\U000e007f"


@pytest.mark.parametrize(
    "value, reason",
    [
        ("x" * (DREAMS_PROFILE_TEXT_MAX_LENGTH + 1), _TOO_LONG),
        ("", _NOTHING),
        ("   ", _NOTHING),
        (None, _NOTHING),
        (42, _NOTHING),
        ("two\nlines", _ONE_LINE),
        ("tab\tinside", _ONE_LINE),
        ("nul\x00byte", _ONE_LINE),
        ("esc\x1bape", _ONE_LINE),
        # Unicode's own line and paragraph separators.
        ("two\u2028lines", _ONE_LINE),
        ("two\u2029paragraphs", _ONE_LINE),
        ("next\u0085line", _ONE_LINE),
        # Bidirectional embedding/override/isolate controls make text
        # DISPLAY in another order than it is stored and sent.
        ("invoice‮gnp.exe", _DIRECTION),
        ("embed‪ left", _DIRECTION),
        ("embed‫ right", _DIRECTION),
        ("pop‬ x", _DIRECTION),
        ("override‭ ltr", _DIRECTION),
        ("isolate⁦ ltr", _DIRECTION),
        ("isolate⁧ rtl", _DIRECTION),
        ("isolate⁨ first-strong", _DIRECTION),
        ("pop⁩ isolate", _DIRECTION),
        # Nothing a person can read (round 4, I3), whatever its category:
        ("​", _NOTHING),             # zero width space (removed)
        ("⁠", _NOTHING),             # word joiner (removed)
        ("﻿", _NOTHING),             # byte order mark (removed)
        ("￼", _NOTHING),             # object replacement (removed)
        ("‌", _NOTHING),             # an allowed joiner, alone
        ("ㅤ", _NOTHING),             # Hangul filler (a "letter")
        ("⠀", _NOTHING),             # braille blank (a "symbol")
        ("\U0001d159", _NOTHING),         # musical null notehead (a "symbol")
        (" ", _NOTHING),             # no-break space only
        ("́̂", _NOTHING),       # combining marks on nothing
        (".", _NOTHING),                  # punctuation is not a topic
        ("???", _NOTHING),
        ("— —", _NOTHING),
        # Invisible characters that mean something else than what shows:
        ("a⁪b", _HIDDEN),            # deprecated format character
        ("a￹b", _HIDDEN),            # interlinear annotation anchor
        ("a\U000e0041b", _HIDDEN),        # a tag character outside a flag
        ("\U0001f3f4\U000e0067\U000e0062 football", _HIDDEN),  # flag, no end
        ("\U000e0067\U000e0062\U000e007f football", _HIDDEN),  # tags, no flag
        # Not text at all:
        ("a￿b", _NOT_TEXT),          # noncharacter
        ("a﷐b", _NOT_TEXT),          # noncharacter
        ("a\U0001fffeb", _NOT_TEXT),      # noncharacter (plane 1)
        ("a\ud800b", _NOT_TEXT),          # lone surrogate (cannot be stored)
    ],
)
def test_a_refusal_names_what_was_wrong(value, reason):
    with pytest.raises(ValueError) as excinfo:
        validate_dreams_profile_text(value)
    # The message is fixed copy for a notice: it names the one thing that
    # was wrong (round 4, C4), never echoes the input, and names no rule
    # that does not exist.
    message = str(excinfo.value)
    assert reason in message
    assert [other in message for other in (
        _TOO_LONG, _ONE_LINE, _NOTHING, _DIRECTION, _HIDDEN, _NOT_TEXT)
    ].count(True) == 1
    assert "markup" not in message and "script" not in message
    assert "x" * 20 not in message
    if reason == _TOO_LONG:
        assert str(DREAMS_PROFILE_TEXT_MAX_LENGTH) in message


# --- Round 4, C4: the rule refused text people type ------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        # Subdivision flags are tag SEQUENCES: the tag characters are what
        # makes England's flag England's.
        (f"{_ENGLAND} football", f"{_ENGLAND} football"),
        (_SCOTLAND, _SCOTLAND),
        (f"rugby {_WALES}", f"rugby {_WALES}"),
        # Private use: the Apple logo, a font's icons.
        (" events", " events"),
        ("", ""),
        # "Unassigned" is only what THIS interpreter's Unicode tables do
        # not know yet: a Unicode 16 emoji, CJK Extension I. The answer
        # must not change with the Python version.
        ("\U0001fae9 mood", "\U0001fae9 mood"),
        ("\U0001fae9", "\U0001fae9"),
        ("\U0002ebf0 kanji", "\U0002ebf0 kanji"),
        ("a͸b", "a͸b"),
        # Format characters that are VISIBLE: Arabic number sign, end of
        # ayah, Syriac abbreviation mark.
        ("؀١٢٣", "؀١٢٣"),
        ("۝١", "۝١"),
        ("܏ܐܒ", "܏ܐܒ"),
        # Mongolian vowel separator: shaping needs it.
        ("ᠮᠣᠩᠭᠣᠯ᠎ᠠ",
         "ᠮᠣᠩᠭᠣᠯ᠎ᠠ"),
        # A symbol or an emoji alone is a topic.
        ("❤", "❤"),
        ("\U0001f3b8", "\U0001f3b8"),
        ("€", "€"),
    ],
)
def test_accepts_text_people_actually_type(value, expected):
    assert validate_dreams_profile_text(value) == unicodedata.normalize(
        "NFC", expected)


@pytest.mark.parametrize(
    "value, stored",
    [
        # Invisible characters that only hint at breaks or come along with
        # a paste are REMOVED, not refused: Thai and Khmer typed with zero
        # width spaces between words, a soft hyphen, a leading byte order
        # mark, a word joiner, an object placeholder.
        ("ภาษา​ไทย",
         "ภาษาไทย"),
        ("jazz­guitar", "jazzguitar"),
        ("﻿jazz guitar", "jazz guitar"),
        ("jazz⁠guitar", "jazzguitar"),
        ("jazz￼ guitar", "jazz guitar"),
        # A blank that takes up room is a space (round 4, I3): braille's
        # own space, the Hangul filler.
        ("jazz⠀guitar", "jazz guitar"),
        ("⠓⠑⠇⠇⠕⠀⠺⠕⠗⠇⠙",
         "⠓⠑⠇⠇⠕ ⠺⠕⠗⠇⠙"),
        ("jazzㅤguitar", "jazz guitar"),
        ("  jazz \u2800 guitar  ", "jazz guitar"),
    ],
)
def test_harmless_invisible_characters_are_removed_not_refused(value, stored):
    assert validate_dreams_profile_text(value) == stored


def test_a_letter_that_normalisation_splits_still_counts_once():
    """101 Devanagari nukta letters are 101 characters to the person who
    typed them; NFC writes each as two code points, and the rule then said
    "at most 200" to someone who had typed 101."""
    nukta = "क़"
    assert len(unicodedata.normalize("NFC", nukta)) == 2, "fixture"

    stored = validate_dreams_profile_text(nukta * 101)

    assert stored == unicodedata.normalize("NFC", nukta * 101)
    assert validate_dreams_profile_text(
        nukta * DREAMS_PROFILE_TEXT_MAX_LENGTH)
    with pytest.raises(ValueError, match="longer than"):
        validate_dreams_profile_text(nukta * (DREAMS_PROFILE_TEXT_MAX_LENGTH + 1))


def test_the_answer_does_not_depend_on_this_interpreters_unicode_tables(
        monkeypatch):
    """A code point this Python calls unassigned (Cn) and one a newer
    Python calls a symbol (So) must get the same answer."""
    real = unicodedata.category

    class _OlderTables:
        normalize = staticmethod(unicodedata.normalize)

        @staticmethod
        def category(char):
            return "Cn" if char == "\U0001f3b8" else real(char)

    assert validate_dreams_profile_text("\U0001f3b8") == "\U0001f3b8"
    monkeypatch.setattr(input_validation, "unicodedata", _OlderTables)
    assert validate_dreams_profile_text("\U0001f3b8") == "\U0001f3b8"


def test_composed_and_decomposed_spellings_are_stored_as_one():
    """The stored form is NFC, so "café" typed with a combining accent and
    with the precomposed letter is ONE topic for the (facet, text) key."""
    composed = "café culture"
    decomposed = "café culture"
    assert decomposed != composed

    assert validate_dreams_profile_text(decomposed) == composed
    assert validate_dreams_profile_text(composed) == composed


def test_the_limit_counts_code_points_of_the_stored_form():
    """200 code points that are not combining marks: an astral emoji is
    one, and an accent never counts apart from its letter."""
    assert len(validate_dreams_profile_text("\U0001f600" * 200)) == 200
    decomposed = "é" * DREAMS_PROFILE_TEXT_MAX_LENGTH  # 400 as typed
    assert validate_dreams_profile_text(decomposed) == (
        "é" * DREAMS_PROFILE_TEXT_MAX_LENGTH)
    with pytest.raises(ValueError):
        validate_dreams_profile_text("é" * (DREAMS_PROFILE_TEXT_MAX_LENGTH + 1))


def test_an_oversized_paste_is_refused_before_any_per_character_work(
        monkeypatch):
    """A 5,000,000-character paste used to be scanned character by
    character (574 ms on the UI thread) before the length refused it."""

    class _NoUnicodeWork:
        @staticmethod
        def category(_char):
            raise AssertionError("scanned an over-long text")

        normalize = bidirectional = category

    monkeypatch.setattr(input_validation, "unicodedata", _NoUnicodeWork)

    with pytest.raises(ValueError):
        validate_dreams_profile_text("x" * 5_000_000)


def test_the_shared_text_helper_is_not_part_of_this_rule(monkeypatch):
    """The script-token patterns live in the app-wide ``validate_text_input``;
    this validator no longer consults it (and does not change it)."""

    def _must_not_run(*args, **kwargs):
        raise AssertionError("the shared helper was consulted")

    monkeypatch.setattr(input_validation, "validate_text_input", _must_not_run)

    assert validate_dreams_profile_text("javascript: basics") == (
        "javascript: basics")
    assert unicodedata.normalize("NFC", "x") == "x"  # the real module is intact


# === Review of #2921, round 5, F3: the stored form must pass its own rule ==========
#
# Round 4 measured ``min(len(typed), len(stored))``. On a re-read the typed
# length is gone: 101 nukta letters were accepted and stored as 202 code
# points, and validating THAT raised "longer than 200 characters" -- so a
# region saved in Settings was ignored by ``dreams_region()``, and a derived
# keyword passed the first check of the signal refresh and was dropped by
# the second.

from hypothesis import given, settings as hyp_settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_NUKTA = "\u0958"  # one letter; NFC writes it as a base letter and a mark


def test_text_that_normalisation_lengthens_is_accepted_again_as_stored():
    stored = validate_dreams_profile_text(_NUKTA * 101)
    assert len(stored) == 202, "fixture: NFC lengthened it past 200"

    assert validate_dreams_profile_text(stored) == stored


def test_the_measure_counts_letters_not_the_marks_on_them():
    """At most 200 code points that are not combining marks -- which NFC
    cannot change -- and at most 600 in all."""
    limit = DREAMS_PROFILE_TEXT_MAX_LENGTH
    assert len(validate_dreams_profile_text(_NUKTA * limit)) == 2 * limit
    with pytest.raises(ValueError, match="longer than"):
        validate_dreams_profile_text(_NUKTA * (limit + 1))
    # Two marks on every letter ("q" composes with neither): 600 code
    # points, 200 letters.
    stacked = "q\u0331\u0325" * limit
    assert len(unicodedata.normalize("NFC", stacked)) == 3 * limit, "fixture"
    assert validate_dreams_profile_text(stacked) == unicodedata.normalize(
        "NFC", stacked)
    # Three marks on every letter is 800: past the bound on the whole.
    with pytest.raises(ValueError, match="longer than"):
        validate_dreams_profile_text("q\u0331\u0325\u0323" * limit)
    # Marks cannot be used to smuggle length in on a few letters.
    with pytest.raises(ValueError, match="longer than"):
        validate_dreams_profile_text("q" + "\u0331" * 700)


def test_a_region_that_normalisation_lengthens_is_used_not_ignored(monkeypatch):
    """The reviewer's probe: a 179-code-point phrase with a nukta letter in
    every word is saved as 209, and ``dreams_region()`` then returned ""."""
    from tldw_chatbook.Dreams import settings as dreams_settings

    typed = " ".join([_NUKTA + "\u093e\u0928\u0942\u0928"] * 30)[:179]
    saved = validate_dreams_profile_text(typed)
    assert len(typed) == 179 and len(saved) > 200, "fixture"
    monkeypatch.setattr(
        dreams_settings, "get_cli_setting",
        lambda section, key, default=None: saved if key == "region" else default)

    assert dreams_settings.dreams_region() == saved


#: Characters that exercise every way the rule rewrites or measures text:
#: marks that compose and marks that do not, letters NFC splits, Hangul
#: jamo NFC joins, the characters it removes, the blanks it writes as a
#: space, every kind of space, joiners, emoji with their selectors.
_F3_ALPHABET = (
    "abz AZ09 .,-'\"()[]"
    "\u0301\u0308\u0323\u093c\u0958\u0959\u095a\u00e9e\u0915"
    "\u1100\u1161\u11a8\uac00"
    "\u200b\u00ad\u2060\ufeff\ufffc\u2062"
    "\u2800\u3164\uffa0\u00a0\u3000\u2003"
    "\u200c\u200d\u200e\ufe0f\u20e3#\u203c\u2764\U0001f3b8\U0001f1ef\U0001f1f5"
    "\u0e20\u0e32\u65e5\u672c"
)


#: Text near the limit made of what NFC lengthens (nukta letters), what
#: it shortens (a letter and its accent, Hangul jamo) and what the rule
#: removes or rewrites -- where the measure decides.
_F3_NEAR_THE_LIMIT = st.builds(
    lambda letters, count, filler: letters * count + filler,
    st.sampled_from(["\u0958", "\u0959\u0301", "e\u0301", "\u1100\u1161",
                     "a\u0323\u0331", "\u0958\u200b", "\u0958\u2800"]),
    st.integers(90, 210),
    st.text(alphabet="a \u0301\u0958", max_size=12),
)


@hyp_settings(max_examples=1500, deadline=None)
@given(st.one_of(
    st.text(alphabet=st.sampled_from(_F3_ALPHABET), max_size=260),
    _F3_NEAR_THE_LIMIT))
def test_validating_the_stored_form_again_changes_nothing(typed):
    try:
        stored = validate_dreams_profile_text(typed)
    except ValueError:
        return  # refused input stores nothing

    assert validate_dreams_profile_text(stored) == stored


# --- Review round 5, N6: the bound is on what is left, not on what is removed -----


def test_text_that_is_short_once_invisible_characters_are_removed_is_kept():
    """A zero width character x800 in front of "x" was refused as "longer
    than 200 characters": the pre-check ran before the removal."""
    assert validate_dreams_profile_text("\u200b" * 800 + "x") == "x"
    # 206 characters as pasted, 166 once the soft hyphens are gone.
    assert validate_dreams_profile_text(
        ("jazz\u00ad" * 40) + "guitar") == "jazz" * 40 + "guitar"


@pytest.mark.parametrize(
    "value, stored",
    [
        # Pasted maths carries the invisible operators; like the soft
        # hyphen they are removed, not refused.
        ("a\u2062b", "ab"),
        ("f\u2061(x)", "f(x)"),
        ("x\u2063y", "xy"),
        ("2\u20643", "23"),
    ],
)
def test_invisible_maths_operators_are_removed_like_the_soft_hyphen(
        value, stored):
    assert validate_dreams_profile_text(value) == stored


def test_a_huge_paste_is_still_refused_before_any_unicode_work(monkeypatch):
    """The cheap bound stays, on the raw text: 5,000,000 characters -- of
    letters, or of characters that would be removed -- never reach the
    per-character work."""

    class _NoUnicodeWork:
        @staticmethod
        def category(_char):
            raise AssertionError("scanned an over-long text")

        normalize = bidirectional = category

    monkeypatch.setattr(input_validation, "unicodedata", _NoUnicodeWork)

    for paste in ("x" * 5_000_000, "\u200b" * 5_000_000 + "x"):
        with pytest.raises(ValueError, match="longer than"):
            validate_dreams_profile_text(paste)


# --- Review round 5, N5: one spacing, one topic -----------------------------------
#
# The removals and the blank-to-space rewrite could leave runs of spaces:
# "rust <ZWSP> tui" was stored as "rust  tui", and that and "rust tui" are
# two topics under the exact (facet, text) key.


@pytest.mark.parametrize(
    "value",
    [
        "rust tui", "rust  tui", "rust \u200b tui", "rust\u00a0tui",
        "rust\u3000tui", "rust \u2800 tui", "rust\u2003\u2003tui",
        " rust\u202ftui ", "rust\u205f \u00a0 tui",
    ],
)
def test_every_spacing_of_a_phrase_is_stored_as_one(value):
    assert validate_dreams_profile_text(value) == "rust tui"


@pytest.mark.parametrize(
    "value",
    ["rust\ttui", "rust\ntui", "rust\u2028tui", "rust\x1ctui"],
)
def test_collapsing_spaces_does_not_let_line_breaks_through(value):
    with pytest.raises(ValueError, match="single line"):
        validate_dreams_profile_text(value)


def test_the_mongolian_vowel_separator_is_not_a_space():
    """It is a format character shaping needs, not whitespace: kept."""
    mongolian = "\u182e\u1823\u1829\u182d\u1823\u182f\u180e\u1820"
    assert validate_dreams_profile_text(mongolian) == mongolian


# --- Review round 5, N4: an emoji the user can type is readable ------------------
#
# The readable test counted letters, numbers, other symbols (So) and
# currency signs. Emoji written as punctuation or a maths sign with the
# emoji presentation selector -- the keycaps, the double exclamation mark,
# the left-right arrow -- were refused as "nothing readable".


@pytest.mark.parametrize(
    "value",
    [
        "#\ufe0f\u20e3",          # keycap number sign
        "*\ufe0f\u20e3",          # keycap asterisk
        "#\u20e3",                # keycap without the selector
        "\u203c\ufe0f",           # double exclamation mark
        "\u2049\ufe0f",           # exclamation question mark
        "\u2194\ufe0f",           # left-right arrow
        "\u203c\ufe0f news",
    ],
)
def test_an_emoji_made_of_punctuation_or_a_sign_is_readable(value):
    assert validate_dreams_profile_text(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "+", "\u221e", "\u2192", "^", "~",  # maths signs and modifiers alone
        "\u203c",                           # punctuation without the selector
        "\ufe0f", "\u20e3", "\ufe0f\u20e3",  # the selectors on nothing
        "+ - \u00d7",
    ],
)
def test_maths_signs_punctuation_and_bare_selectors_are_not(value):
    with pytest.raises(ValueError, match="nothing readable"):
        validate_dreams_profile_text(value)


@pytest.mark.parametrize("value", ["\u2665", "\u20ac", "\u00a9", "\U0001f3b8"])
def test_a_symbol_alone_is_readable(value):
    assert validate_dreams_profile_text(value) == value


# --- Review round 6, G2: the emoji rule must not accept invisible text -----------
#
# Round 5 accepted ANY character in front of the emoji selector or the
# keycap, so invisible characters with a selector after them were a topic.
# An emoji written with a selector is now one Unicode actually defines: a
# keycap is 0-9, # or * (with or without U+FE0F) and U+20E3; an emoji
# presentation sequence of punctuation or a maths sign exists only for the
# few listed below. "+\ufe0f" is not an emoji, so it is what "+" is: a
# maths sign, which alone is not a topic.


@pytest.mark.parametrize(
    "value",
    [
        "\u200d\ufe0f",          # zero width joiner + VS16
        "\u200e\ufe0f",          # left-to-right mark + VS16
        "\u061c\u20e3",          # Arabic letter mark + keycap
        "\u180e\ufe0f",          # Mongolian vowel separator + VS16
        "\u115f\ufe0f",          # Hangul choseong filler + VS16
        "\u3164\ufe0f",          # Hangul filler + VS16
        "\U0001d159\ufe0f",      # musical null notehead + VS16
        "\u0301\ufe0f",          # combining acute + VS16
        "\u034f\ufe0f",          # combining grapheme joiner + VS16
        "+\ufe0f", ".\ufe0f", "-\u20e3", "+\u20e3", "!\ufe0f", "~\ufe0f",
        "\u2192\ufe0f",          # rightwards arrow: no emoji sequence
    ],
)
def test_a_selector_or_keycap_on_a_character_that_is_no_emoji_is_not_readable(
        value):
    with pytest.raises(ValueError, match="nothing readable"):
        validate_dreams_profile_text(value)


@pytest.mark.parametrize(
    "value",
    [
        "0\u20e3", "7\ufe0f\u20e3", "#\ufe0f\u20e3", "*\u20e3",
        "\u203c\ufe0f", "\u2049\ufe0f", "\u2194\ufe0f", "\u25fb\ufe0f",
        "\u25fe\ufe0f", "\u2934\ufe0f", "\u2935\ufe0f", "\u3030\ufe0f",
        "\u303d\ufe0f",
    ],
)
def test_every_emoji_unicode_writes_with_a_selector_on_a_sign_is_readable(
        value):
    assert validate_dreams_profile_text(value) == value
