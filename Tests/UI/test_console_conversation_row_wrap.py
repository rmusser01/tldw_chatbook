"""Unit tests for the Console rail title wrap/truncate helpers."""

from types import SimpleNamespace

from rich.cells import cell_len

from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
    truncate_console_row_cells,
    wrap_console_conversation_title,
)


def _relabel_decision(measured, *, current, measured_before):
    """Evaluate the width-relabel decision without building a widget."""
    stub = SimpleNamespace(
        _row_content_width=current,
        _row_width_measured=measured_before,
    )
    return ConsoleWorkspaceContextTray._should_relabel_at_width(stub, measured)


def test_first_measurement_adopts_real_width_over_fallback() -> None:
    # Even a one-cell difference from the pre-measurement fallback must
    # relabel the very first time, so rows leave the fallback budget behind.
    assert _relabel_decision(21, current=20, measured_before=False) is True


def test_first_measurement_noop_when_width_already_matches() -> None:
    assert _relabel_decision(20, current=20, measured_before=False) is False


def test_one_cell_scrollbar_toggle_is_ignored_after_first_measure() -> None:
    # A collapse removes rows -> the 1-cell rail scrollbar disappears ->
    # width shifts by one. That must NOT trigger a recompose.
    assert _relabel_decision(24, current=23, measured_before=True) is False
    assert _relabel_decision(22, current=23, measured_before=True) is False


def test_unchanged_width_is_a_noop_after_first_measure() -> None:
    assert _relabel_decision(23, current=23, measured_before=True) is False


def test_multi_cell_resize_relabels_after_first_measure() -> None:
    # Both boundary directions at exactly the threshold, plus larger jumps.
    assert _relabel_decision(25, current=23, measured_before=True) is True
    assert _relabel_decision(21, current=23, measured_before=True) is True
    assert _relabel_decision(40, current=23, measured_before=True) is True
    assert _relabel_decision(10, current=23, measured_before=True) is True


def test_short_title_is_single_line() -> None:
    assert wrap_console_conversation_title("Quick test", 30) == ("Quick test",)


def test_exact_fit_is_single_line_without_ellipsis() -> None:
    assert wrap_console_conversation_title("0123456789", 10) == ("0123456789",)


def test_long_title_wraps_at_word_boundary() -> None:
    lines = wrap_console_conversation_title("Debugging the RAG splat", 20)
    assert lines == ("Debugging the RAG", "splat")


def test_overflowing_title_ellipsizes_second_line() -> None:
    lines = wrap_console_conversation_title(
        "Debugging the RAG splat bug in retrieval", 20
    )
    assert lines == ("Debugging the RAG", "splat bug in retrie…")
    assert all(cell_len(line) <= 20 for line in lines)


def test_cut_landing_on_word_boundary_keeps_full_head() -> None:
    # "aaaa bbbb " is exactly 10 cells; the whole first two words must
    # survive on line 1 rather than breaking back to "aaaa".
    assert wrap_console_conversation_title("aaaa bbbb cccc", 10) == (
        "aaaa bbbb",
        "cccc",
    )


def test_spaceless_token_hard_breaks() -> None:
    lines = wrap_console_conversation_title("A" * 50, 20)
    assert lines == ("A" * 20, "A" * 19 + "…")


def test_budget_floor_clamps_to_ten_cells() -> None:
    lines = wrap_console_conversation_title("aaaa bbbb cccc", 3)
    assert lines == ("aaaa bbbb", "cccc")
    assert all(cell_len(line) <= 10 for line in lines)


def test_wide_characters_measure_in_cells() -> None:
    lines = wrap_console_conversation_title("日" * 8, 10)
    assert lines == ("日" * 5, "日" * 3)
    assert all(cell_len(line) <= 10 for line in lines)


def test_blank_title_falls_back_to_untitled() -> None:
    assert wrap_console_conversation_title("   ", 30) == ("Untitled conversation",)


def test_truncate_returns_short_text_unchanged() -> None:
    assert truncate_console_row_cells("saved - 2d", 20) == "saved - 2d"


def test_truncate_ellipsizes_long_text() -> None:
    result = truncate_console_row_cells("Workspace A - saved chat - 2d", 20)
    assert result == "Workspace A - saved…"
    assert cell_len(result) <= 20


def test_truncate_is_cell_aware_for_wide_characters() -> None:
    result = truncate_console_row_cells("日" * 12, 10)
    assert cell_len(result) <= 10
    assert result.endswith("…")
