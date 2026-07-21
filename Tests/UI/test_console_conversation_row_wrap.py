"""Unit tests for the Console rail title wrap/truncate helpers."""

from rich.cells import cell_len

from tldw_chatbook.Widgets.Console.console_workspace_context import (
    truncate_console_row_cells,
    wrap_console_conversation_title,
)


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
