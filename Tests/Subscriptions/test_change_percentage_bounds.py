"""`ContentExtractor.calculate_change_percentage`: sane values, bounded cost.

TASK-16839, from the task-15764 review (finding 5). The old implementation ran a
character-level ``difflib.SequenceMatcher.ratio()`` with ``autojunk=True`` over
the two full raw texts, which has two entangled failure regimes:

* **Degenerate values** — for large Latin pages every character clears
  autojunk's 1%-popularity bar, the matcher junks its whole alphabet, and a
  5%-edited page reported ~100% changed. The percentage was meaningless exactly
  where it was cheap.
* **Quadratic cost** — for a large character repertoire (CJK/unicode-heavy)
  nothing is junked and ``ratio()`` goes quadratic: measured 4x per doubling,
  ~1 s at 160 K chars, extrapolating to ~7 minutes at the 10 MB
  ``MAX_FETCH_BYTES_PAGE`` cap, on a GIL-holding worker thread.

The shipped fix computes the ratio over ``_segment_for_diff`` segments -- the
same sentence/line-sized basis the stored diff body, ``diff_summary`` and
``added_and_removed_text`` already use -- so the percentage now means "fraction
of the page's segment content that appeared or disappeared" and every consumer
(the ``change_threshold`` comparison, the withheld disposition, the reader's
"N% changed" headline) reads one coherent story. The ratio is an O(n)
order-insensitive multiset comparison at EVERY size (see
``_segment_change_ratio``): the fix round retired an initial order-sensitive
``SequenceMatcher`` tier after the independent review reproduced a sign-flip
cliff at its 4,000-segment boundary for reorder-shaped edits.

The born-red pins below were each run against the implementation they indict
and failed for the advertised reason (degenerate value; blown wall-clock
bound; the reorder tier-cliff).
"""

import random
import time

import pytest

from tldw_chatbook.Subscriptions.monitoring_engine import (
    ContentExtractor,
    _segment_for_diff,
)

# --- deterministic content builders ------------------------------------------

# Only very common characters (lowercase letters, space, period): at >=~20 KB
# every one of them exceeds autojunk's 1%-of-sequence popularity bar, which is
# precisely the shape that degenerated the old character-level matcher.
_COMMON_WORDS = (
    "the report section remains unchanged in this revision and the terms "
    "stay exactly as they were with no further notice given here"
).split()


def _latin_page(n_sentences: int) -> str:
    """One long extracted-text line of `n_sentences` common-word sentences."""
    return " ".join(
        " ".join(_COMMON_WORDS[(i + j) % len(_COMMON_WORDS)] for j in range(9)) + "."
        for i in range(n_sentences)
    )


def _edit_fraction(text: str, fraction: float) -> str:
    """Rewrite every ``1/fraction``-th sentence entirely (scattered edits)."""
    sentences = text.split(". ")
    step = max(1, round(1 / fraction))
    for i in range(0, len(sentences), step):
        sentences[i] = "completely rewritten clause about another topic entirely"
    return ". ".join(sentences)


def _cjk_page(n_sentences: int, chars_per_sentence: int = 40) -> str:
    """Spaceless CJK-shaped prose: sentences end with 。and no whitespace.

    Seeded-random draws over a 6,000-codepoint repertoire keep every character
    rare AND aperiodic -- the regime where autojunk junked nothing and the old
    matcher went cleanly quadratic. (A deterministic cycling generator was
    tried first and did NOT reproduce the quadratic regime at the pre-fix
    HEAD: periodic text lets the matcher find giant matches cheaply.)
    """
    rng = random.Random(1234)
    out = []
    for _ in range(n_sentences):
        out.append(
            "".join(
                chr(0x4E00 + rng.randrange(6000)) for _ in range(chars_per_sentence - 1)
            )
        )
    return "。".join(out) + "。"


# --- value sanity (the degeneracy pin) ---------------------------------------


def test_five_percent_edited_large_latin_page_reports_a_small_percentage():
    """Born-red degeneracy pin (task-16839 AC#1).

    At the pre-fix HEAD this exact input returned 0.471 ("47% changed") for a
    page with 5% of its sentences rewritten -- and took ~39 s doing it: Latin
    text this size junks nearly every character it is made of, so the matcher
    had almost nothing left to align (the task-15764 review measured a full
    1.0 on its shape). The segment-level basis reports the honest magnitude:
    ~5% of the page's sentences changed.
    """
    old = _latin_page(2400)  # ~128 KB of extracted text
    new = _edit_fraction(old, 0.05)
    assert len(old) > 100_000, "the pin is about LARGE pages; keep it large"

    pct = ContentExtractor.calculate_change_percentage(old, new)

    assert pct == pytest.approx(0.05, abs=0.02), (
        "a 5%-edited large Latin page must report ~5% changed, not the "
        f"autojunk-degenerate ~100% (got {pct!r})"
    )


def test_identical_and_disjoint_and_empty_extremes():
    page = _latin_page(300)
    assert ContentExtractor.calculate_change_percentage(page, page) == 0.0
    assert ContentExtractor.calculate_change_percentage("", "") == 0.0
    assert ContentExtractor.calculate_change_percentage(page, "") == 1.0
    assert ContentExtractor.calculate_change_percentage("", page) == 1.0

    disjoint = " ".join(
        f"totally different words about frogs and rivers number {i}."
        for i in range(300)
    )
    pct = ContentExtractor.calculate_change_percentage(page, disjoint)
    assert pct == pytest.approx(1.0, abs=0.01), (
        f"two pages sharing no sentences must report ~100% changed (got {pct!r})"
    )


def _pure_reorder_pct(n_per_side: int) -> float:
    """Change percentage for a page whose segments were shuffled, nothing else."""
    sentences = [
        f"unique sentence number {i} holding its own distinct words."
        for i in range(n_per_side)
    ]
    old = " ".join(sentences)
    assert len(_segment_for_diff(old)) == n_per_side, (
        "the probe is only meaningful if each sentence is its own segment"
    )
    shuffled = sentences[:]
    random.Random(99).shuffle(shuffled)
    new = " ".join(shuffled)
    return ContentExtractor.calculate_change_percentage(old, new)


def test_pure_reorder_reports_zero_change_with_no_size_cliff():
    """Born-red fix-round pin (task-16839 review, Finding 1).

    At commit 6d22de89f a purely reordered page -- same segments, shuffled,
    zero content change -- reported pct=0.9925 at 4,000 total segments (the
    order-sensitive SequenceMatcher alignment tier) and pct=0.0000 at 4,002
    (the order-insensitive multiset tier): one extra sentence per side flipped
    "99% changed" to "0% changed" for functionally the same event. The fix
    retired the alignment tier; the multiset ratio is the sole mechanism at
    every size, so a pure reorder reports 0.0 everywhere -- the documented
    semantic decision (see ``_segment_change_ratio``): a segment that merely
    moved is not content change, and the diff body still shows the moves.
    """
    below = _pure_reorder_pct(2000)  # 4,000 total segments: the retired tier boundary
    above = _pure_reorder_pct(2001)  # 4,002 total segments: just past it
    assert abs(below - above) < 0.01, (
        f"tier cliff: a pure reorder reports {below:.4f} at 4,000 total "
        f"segments but {above:.4f} at 4,002 -- the two mechanisms disagree "
        f"at their boundary"
    )
    for n_per_side in (12, 300, 2000, 2001, 6000):
        pct = _pure_reorder_pct(n_per_side)
        assert pct == 0.0, (
            f"a purely reordered page must report 0.0 at any size "
            f"(got {pct!r} at {n_per_side} segments/side)"
        )


def test_small_page_one_sentence_edit_is_one_segments_worth():
    """The percentage is segment-granular: 1 edited sentence of 11 is ~9%."""
    old = " ".join(
        f"clause {i} of the terms stays exactly as written." for i in range(10)
    )
    old += " the price is 42 credits per widget."
    new = old.replace(
        "the price is 42 credits per widget.",
        "pricing moved to a metered plan billed hourly instead.",
    )
    pct = ContentExtractor.calculate_change_percentage(old, new)
    assert pct == pytest.approx(1 - (2 * 10 / 22), abs=1e-9)


def test_very_large_page_keeps_sane_values():
    """A ~640 KB / 12,000-segment page reports the same magnitudes as a small
    one -- the multiset ratio is one mechanism at every size, so there is no
    tier boundary for the value to jump at (fix round, review Finding 1)."""
    old = _latin_page(12_000)
    assert len(_segment_for_diff(old)) == 12_000
    new = _edit_fraction(old, 0.05)
    pct = ContentExtractor.calculate_change_percentage(old, new)
    assert pct == pytest.approx(0.05, abs=0.02)
    assert ContentExtractor.calculate_change_percentage(old, old) == 0.0


def test_cjk_prose_segments_on_sentence_enders_and_reports_small_edits():
    """CJK prose (no whitespace, 。sentence enders) gets sentence segments.

    Before task-16839 ``_SENTENCE_BOUNDARY`` required trailing whitespace, so
    a whole CJK page was ONE unit fixed-sliced into 110-char segments whose
    boundaries all shift under any insertion -- a one-sentence edit would have
    re-sliced (and so "changed") half the page.
    """
    old = _cjk_page(400)
    segments = _segment_for_diff(old)
    assert len(segments) == 400, "each 。-ended sentence must be its own segment"

    sentences = old.split("。")
    sentences[200] = "完全不同的新句子在这里出现了"
    new = "。".join(sentences)
    pct = ContentExtractor.calculate_change_percentage(old, new)
    assert pct == pytest.approx(2 / 800, abs=1e-9), (
        f"one edited sentence of 400 must report ~0.25% (got {pct!r})"
    )


# --- bounded cost (the quadratic-regime pin) ---------------------------------

# These are wall-clock regression bounds, deliberately LOOSE (an order of
# magnitude over the measured times) so machine noise never flakes them: the
# regression they guard is seconds-to-minutes, not milliseconds.


def _timed_percentage(old: str, new: str) -> tuple[float, float]:
    start = time.perf_counter()
    pct = ContentExtractor.calculate_change_percentage(old, new)
    return pct, time.perf_counter() - start


def test_large_repertoire_page_is_not_quadratic():
    """Born-red cost pin (task-16839 AC#2), loose wall-clock bound.

    ~640 K chars over a 6,000-codepoint random repertoire: nothing is junked,
    and the pre-fix character matcher was quadratic here (4x per doubling --
    tens of seconds at this size, ~7 minutes at the 10 MB cap); the segment
    basis is linear. The 2 s bound is an order of magnitude over the post-fix
    measurement.
    """
    old = _cjk_page(16_000)  # 16000 sentences x 40 chars = 640K chars
    sentences = old.split("。")
    sentences[8000] = "改" * 39
    new = "。".join(sentences)

    pct, elapsed = _timed_percentage(old, new)

    assert elapsed < 2.0, f"took {elapsed:.1f}s -- the quadratic regime is back"
    assert 0.0 < pct < 0.01, f"one edited sentence of 16000 (got {pct!r})"


def test_ten_megabyte_fetch_cap_is_bounded_latin_and_spaceless():
    """AC#2: bounded at the 10 MB ``MAX_FETCH_BYTES_PAGE`` cap, for the
    many-segments shape (Latin), the CJK-prose shape (~10 MB utf-8), and the
    single-giant-unit shape (no sentence enders at all). At the pre-fix HEAD
    the Latin shape alone was minutes. Measured (2026-08-16, this change):
    all comfortably sub-second; 5 s is the loose regression bound."""
    old = _latin_page(200_000)  # ~10.7 MB of text, ~200K segments
    assert len(old) >= 10_000_000
    new = _edit_fraction(old, 0.05)
    pct, elapsed = _timed_percentage(old, new)
    assert elapsed < 5.0, f"10MB Latin took {elapsed:.1f}s"
    assert pct == pytest.approx(0.05, abs=0.02)

    old_cjk = _cjk_page(85_000)  # 3.4M chars, ~10 MB utf-8
    cjk_sentences = old_cjk.split("。")
    del cjk_sentences[40_000:42_000]  # drop ~2.4% of the page
    new_cjk = "。".join(cjk_sentences)
    pct, elapsed = _timed_percentage(old_cjk, new_cjk)
    assert elapsed < 5.0, f"10MB CJK prose took {elapsed:.1f}s"
    assert pct == pytest.approx(2000 / 169_000, abs=0.005)

    # No sentence enders anywhere: one giant unbreakable unit per side, the
    # shape that made `textwrap.wrap` quadratic before this task fixed-sliced
    # it. Boundaries shift under the deletion, so the value is coarse here
    # (documented); the assertions are the bound and value validity.
    blob_old = _cjk_page(85_000).replace("。", "")
    blob_new = blob_old[:1_000_000] + blob_old[1_100_000:]
    pct, elapsed = _timed_percentage(blob_old, blob_new)
    assert elapsed < 5.0, f"10MB spaceless blob took {elapsed:.1f}s"
    assert 0.0 < pct <= 1.0


def test_repetitive_page_is_neither_junked_nor_quadratic():
    """A page of few distinct segments must not resurrect either regime.

    Segment-level autojunk would junk a segment repeated >1% of a >=200-segment
    page (the old defect one level up), and an unjunked SequenceMatcher
    alignment over heavy repetition is the quadratic shape -- the multiset
    ratio has neither mechanism: counting multiplicity is immune to
    repetition, and its cost is O(n) by construction. Exact small value,
    bounded time.
    """
    old = " ".join(f"entry number {i % 7} is listed here." for i in range(4000))
    new_sentences = old.split(". ")
    new_sentences[2000] = "one genuinely new entry appears"
    new = ". ".join(new_sentences)

    pct, elapsed = _timed_percentage(old, new)

    assert elapsed < 2.0, f"repetitive page took {elapsed:.1f}s"
    assert 0.0 < pct < 0.01, (
        f"one edited segment of 4000 must report ~0.025%, not junk-degenerate "
        f"~100% (got {pct!r})"
    )
