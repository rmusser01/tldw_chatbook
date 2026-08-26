"""ChapterDetector must emit ONE chapter per titled heading (task-16850).

Pre-fix, `ChapterDetector.detect_chapters` appended an empty-content
"placeholder" Chapter for every titled heading match AND appended the body
accumulated after that heading as a SEPARATE chapter when the next heading
arrived; only the final placeholder was ever back-filled (by the
"don't forget the last chapter" branch). Net effect, probed by the
task-15773 review on a 13-header book: 25 chapters alternating
`content_len=0` placeholders with body rows — roughly 2x the true count.
Since task-15773 made the chapter table truthful, users saw the doubled
rows, and the editor preview landed on the empty placeholder first.

These tests were born red at `c8b951616` (13-header book detected 25
chapters there) and pin the fixed scan: on a new heading, the previous
heading's chapter closes WITH the body accumulated since it; at EOF the
last one closes the same way.

The edge shapes deliberately KEEP their pre-fix behavior (task-16850 AC #2),
verified by probing the detector at `c8b951616` before restructuring:

- preamble text before the first heading -> an untitled `number=0` /
  "Chapter 0" row (it was already emitted that way by the old
  "save previous chapter" branch);
- headerless document -> ONE chapter `number=1` / "Chapter 1" holding
  everything (the old EOF branch's numbering, NOT the "Full Content"
  fallback);
- whitespace-only input -> the "Full Content" fallback chapter;
- a heading with no body before the next heading (or EOF) -> its chapter is
  kept, empty, so the user still sees every heading they pasted (pre-fix
  those placeholders were empty too — consumers already tolerate
  `content == ""`).
"""

from __future__ import annotations

from tldw_chatbook.TTS.audiobook_generator import ChapterDetector


def _thirteen_header_book() -> tuple[str, list[str]]:
    """The 15773-review probe shape: 13 'Chapter N' headings, each with a body.

    Returns the text and the per-chapter expected body strings.
    """
    parts: list[str] = []
    bodies: list[str] = []
    for n in range(1, 14):
        body = f"Body of chapter {n}, where things happen."
        parts.append(f"Chapter {n}")
        parts.append(body)
        parts.append("")
        bodies.append(body)
    return "\n".join(parts), bodies


class TestOneChapterPerHeading:
    """AC #1: N titled headings -> exactly N chapters, title AND body."""

    def test_thirteen_headers_detect_exactly_thirteen_chapters(self):
        text, _ = _thirteen_header_book()
        chapters = ChapterDetector.detect_chapters(text)
        assert len(chapters) == 13, (
            f"expected 13 chapters for 13 headings, got {len(chapters)}: "
            f"{[(c.title, len(c.content)) for c in chapters]}"
        )

    def test_every_chapter_carries_its_own_heading_and_body(self):
        """Not just non-empty: the body must be the one following ITS heading.

        Pre-fix, titled rows had content_len=0 and the prose sat in separate
        rows — so this asserts the pairing, not merely fullness.
        """
        text, bodies = _thirteen_header_book()
        chapters = ChapterDetector.detect_chapters(text)
        for i, (chapter, body) in enumerate(zip(chapters, bodies), start=1):
            assert chapter.number == i
            assert chapter.title == f"Chapter {i}"
            assert chapter.content.strip() == body, (
                f"chapter {i} ({chapter.title!r}) should own the body that "
                f"follows its heading; got {chapter.content!r}"
            )

    def test_no_zero_content_chapter_for_a_heading_that_has_body_text(self):
        """AC #2, first half — the pre-fix alternating-placeholder shape."""
        text, _ = _thirteen_header_book()
        chapters = ChapterDetector.detect_chapters(text)
        empties = [c.title for c in chapters if not c.content.strip()]
        assert empties == [], f"zero-content chapters emitted: {empties}"

    def test_positions_span_from_heading_line_to_body_end(self):
        text = "Chapter 1\nBody one.\nChapter 2\nBody two."
        chapters = ChapterDetector.detect_chapters(text)
        assert [(c.start_position, c.end_position) for c in chapters] == [
            (0, 1),
            (2, 3),
        ]


class TestEdgeShapesKeepCurrentBehavior:
    """AC #2, second half — pinned against the pre-fix probe at c8b951616."""

    def test_preamble_before_the_first_heading_stays_an_untitled_chapter_zero(self):
        text = "Preamble text before any heading.\nChapter 1\nBody one."
        chapters = ChapterDetector.detect_chapters(text)
        assert [(c.number, c.title, c.content) for c in chapters] == [
            (0, "Chapter 0", "Preamble text before any heading."),
            (1, "Chapter 1", "Body one."),
        ]

    def test_headerless_document_stays_one_chapter_one(self):
        """Pre-fix, a headerless document came out of the EOF branch as a
        single `number=1` / "Chapter 1" chapter (not the "Full Content"
        fallback, which only fires on whitespace-only input)."""
        text = "Just some prose.\nMore prose here."
        chapters = ChapterDetector.detect_chapters(text)
        assert len(chapters) == 1
        assert chapters[0].number == 1
        assert chapters[0].title == "Chapter 1"
        assert chapters[0].content == "Just some prose.\nMore prose here."

    def test_whitespace_only_input_stays_the_full_content_fallback(self):
        text = "   \n\n  "
        chapters = ChapterDetector.detect_chapters(text)
        assert len(chapters) == 1
        assert chapters[0].title == "Full Content"
        assert chapters[0].content == text

    def test_back_to_back_headings_keep_an_empty_chapter_for_the_bodiless_one(self):
        """A heading with no body before the next heading still gets its own
        (empty) chapter — dropping it would silently eat a title the user
        pasted. Pre-fix these placeholders were empty too, so every consumer
        already tolerates `content == ""`; what changes is that a heading
        WITH a body can no longer be empty."""
        text = "Chapter 1\nChapter 2\nBody two."
        chapters = ChapterDetector.detect_chapters(text)
        assert [(c.number, c.title, c.content) for c in chapters] == [
            (1, "Chapter 1", ""),
            (2, "Chapter 2", "Body two."),
        ]

    def test_trailing_bodiless_heading_at_eof_keeps_its_empty_chapter(self):
        text = "Chapter 1\nBody one.\nChapter 2"
        chapters = ChapterDetector.detect_chapters(text)
        assert [(c.number, c.title, c.content) for c in chapters] == [
            (1, "Chapter 1", "Body one."),
            (2, "Chapter 2", ""),
        ]
