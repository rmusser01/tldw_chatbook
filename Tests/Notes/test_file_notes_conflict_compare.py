"""Pure tests for bounded File Notes conflict comparisons."""

from __future__ import annotations

from tldw_chatbook.Notes.file_notes_conflict_compare import (
    CONFLICT_DIFF_MAX_OUTPUT_CHARS,
    ConflictSide,
    build_conflict_comparison,
)


def test_comparison_labels_base_draft_and_disk_changes() -> None:
    comparison = build_conflict_comparison(
        ConflictSide.from_text("Base", "one\ntwo\n"),
        ConflictSide.from_text("Draft", "one\ndraft\n"),
        ConflictSide.from_text("Disk", "one\ndisk\n"),
    )

    assert comparison.output_elided is False
    assert "Base → Draft" in comparison.diff_text
    assert "--- Base" in comparison.diff_text
    assert "+++ Draft" in comparison.diff_text
    assert "-two" in comparison.diff_text
    assert "+draft" in comparison.diff_text
    assert "Base → Disk" in comparison.diff_text
    assert "+disk" in comparison.diff_text
    assert [side.label for side in comparison.sides] == ["Base", "Draft", "Disk"]
    assert all(len(side.sha256) == 64 for side in comparison.sides)


def test_comparison_reports_missing_disk_without_inventing_content() -> None:
    comparison = build_conflict_comparison(
        ConflictSide.from_text("Base", "base"),
        ConflictSide.from_text("Draft", "draft"),
        ConflictSide.absent("Disk"),
    )

    assert comparison.sides[2].state == "absent"
    assert comparison.sides[2].summary == "Disk · absent"
    assert "Base → Disk" in comparison.diff_text
    assert "Disk is absent; no textual diff is available." in comparison.diff_text


def test_comparison_bounds_oversized_input_and_output() -> None:
    oversized = "line\n" * 10_001
    input_elided = build_conflict_comparison(
        ConflictSide.from_text("Base", oversized),
        ConflictSide.from_text("Draft", oversized + "draft\n"),
        ConflictSide.from_text("Disk", oversized + "disk\n"),
    )
    assert input_elided.output_elided is True
    assert "Diff omitted because one or more sides exceed" in input_elided.diff_text
    assert "SHA-256" in input_elided.summary_text

    base = "".join(f"before-{index}\n" for index in range(7_000))
    draft = "".join(f"draft-{index}\n" for index in range(7_000))
    disk = "".join(f"disk-{index}\n" for index in range(7_000))
    output_elided = build_conflict_comparison(
        ConflictSide.from_text("Base", base),
        ConflictSide.from_text("Draft", draft),
        ConflictSide.from_text("Disk", disk),
    )
    assert output_elided.output_elided is True
    assert len(output_elided.diff_text) <= CONFLICT_DIFF_MAX_OUTPUT_CHARS
    assert "comparison output elided" in output_elided.diff_text
