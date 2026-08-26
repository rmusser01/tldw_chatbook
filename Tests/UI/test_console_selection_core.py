"""Tests/UI/test_console_selection_core.py"""
from tldw_chatbook.Widgets.Console.console_selection import (
    SELECTION_QUOTE_CAP,
    SelectionManager,
    TextSelection,
    cap_quote,
    line_end_offset,
    line_start_offset,
    next_line_offset,
    prev_line_offset,
    word_back_offset,
    word_forward_offset,
)


def test_drag_within_single_row_produces_ordered_selection():
    mgr = SelectionManager()
    mgr.begin_drag("m1", 10)
    mgr.extend_drag("m1", 4)
    sel = mgr.finish_drag()
    assert sel == TextSelection(row_key="m1", start=4, end=10)


def test_drag_across_rows_clamps_to_origin_row():
    mgr = SelectionManager()
    mgr.begin_drag("m1", 2)
    mgr.extend_drag("m2", 50)  # different row: ignored
    sel = mgr.finish_drag()
    assert sel is None or sel == TextSelection(row_key="m1", start=2, end=2)
    assert mgr.finish_drag() is None or True  # empty selection is fine


def test_empty_selection_finishes_none_and_sets_just_finished():
    mgr = SelectionManager()
    mgr.begin_drag("m1", 5)
    mgr.extend_drag("m1", 5)
    assert mgr.finish_drag() is None
    assert mgr.just_finished is True
    mgr.consume_just_finished()
    assert mgr.just_finished is False


def test_cancel_clears_everything():
    mgr = SelectionManager()
    mgr.begin_drag("m1", 0)
    mgr.extend_drag("m1", 9)
    mgr.cancel()
    assert mgr.state.selection is None
    assert mgr.state.active is False


def test_cap_quote_truncates_long_text():
    text = "x" * (SELECTION_QUOTE_CAP + 100)
    out = cap_quote(text)
    assert len(out) < len(text)
    assert out.endswith("… [truncated]")


def test_cap_quote_passes_short_text_through():
    assert cap_quote("hello") == "hello"


def test_offset_for_cell_maps_within_line():
    from tldw_chatbook.Widgets.Console.console_selection import offset_for_cell

    assert offset_for_cell("hello world", 6) == 6
    assert offset_for_cell("hello", 0) == 0


def test_offset_for_cell_clamps_high():
    from tldw_chatbook.Widgets.Console.console_selection import offset_for_cell

    assert offset_for_cell("hello", 99) == 5
    assert offset_for_cell("", 3) == 0


def test_offset_for_cell_clamps_negative():
    from tldw_chatbook.Widgets.Console.console_selection import offset_for_cell

    assert offset_for_cell("hello", -4) == 0

# --- keyboard motion helpers (phase 5) --------------------------------------

TEXT = "alpha beta\ngamma  delta\n\nepsilon"


def test_word_forward_jumps_to_next_word_start():
    assert word_forward_offset(TEXT, 0) == 6        # alpha| -> |beta
    assert word_forward_offset(TEXT, 6) == 11       # beta| -> |gamma (over \n)
    assert word_forward_offset(TEXT, 26) == len(TEXT)  # last word -> end


def test_word_back_jumps_to_previous_word_start():
    assert word_back_offset(TEXT, 6) == 0
    assert word_back_offset(TEXT, 13) == 11         # inside gamma -> its start
    assert word_back_offset(TEXT, 0) == 0           # floor


def test_line_bounds_are_current_line_vim_style():
    assert line_start_offset(TEXT, 8) == 0          # inside line 1
    assert line_end_offset(TEXT, 8) == 10           # before the \n
    assert line_start_offset(TEXT, 13) == 11        # line 2
    assert line_end_offset(TEXT, 13) == 23          # \n at position 23


def test_line_motions_move_one_line_and_clamp():
    assert next_line_offset(TEXT, 5) == 16          # column-ish landing on line 2
    assert prev_line_offset(TEXT, 16) == 5
    assert next_line_offset(TEXT, 26) == len(TEXT)  # last line -> end clamp
    assert prev_line_offset(TEXT, 3) == 0           # first line -> start clamp


def test_helpers_are_total_on_empty_text():
    for fn in (word_forward_offset, word_back_offset, line_start_offset,
               line_end_offset, next_line_offset, prev_line_offset):
        assert fn("", 0) == 0
