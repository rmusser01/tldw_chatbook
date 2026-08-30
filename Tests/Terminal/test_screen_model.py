"""Behavioral tests for safe persistent-terminal screen projections."""

from __future__ import annotations

from collections import deque
from collections.abc import MutableMapping, MutableSequence, MutableSet
from dataclasses import FrozenInstanceError
import logging
import unicodedata

import pyte
import pytest

from tldw_chatbook.Terminal.contracts import TerminalReason
from tldw_chatbook.Terminal.screen_model import TerminalScreenModel


def _feed_in_seven_byte_chunks(model: TerminalScreenModel, value: bytes) -> None:
    for offset in range(0, len(value), 7):
        model.feed(value[offset : offset + 7])


def _projected_text(model: TerminalScreenModel) -> str:
    snapshot = model.snapshot()
    return "\n".join(line.text for line in (*snapshot.scrollback, *snapshot.lines))


def test_incremental_utf8_preserves_split_scalars_and_replaces_invalid_bytes() -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    model.feed(b"A\xe2")
    assert model.visible_text() == "A"
    model.feed(b"\x82\xacB\xffC")

    assert model.visible_text() == "A\u20acB\ufffdC"


def test_finish_replaces_an_incomplete_utf8_scalar() -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    model.feed(b"A\xe2\x82")
    model.finish()

    assert model.visible_text() == "A\ufffd"


def test_ascii_wide_combining_and_joiner_graphemes_use_safe_cells() -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    _feed_in_seven_byte_chunks(model, "A界e\u0301👩\u200d💻".encode())
    snapshot = model.snapshot()

    assert snapshot.lines[0].text == "A界é👩\u200d💻"
    # Cursor coordinates are 1-based; the six occupied columns leave the
    # insertion cursor at column seven.
    assert snapshot.cursor_column == 7
    assert snapshot.lines[0].column_width == 6


def test_variation_selector_width_change_preserves_following_cell() -> None:
    model = TerminalScreenModel(columns=10, rows=2)

    model.feed("❤️X".encode())
    line = model.snapshot().lines[0]

    assert line.text == "❤️X"
    assert line.column_width == 3
    assert [(cell.text, cell.width) for run in line.runs for cell in run.cells] == [
        ("❤️", 2),
        ("X", 1),
    ]


def test_wide_character_at_right_edge_wraps_without_exceeding_viewport() -> None:
    model = TerminalScreenModel(columns=10, rows=2)

    model.feed("123456789界".encode())
    snapshot = model.snapshot()

    assert snapshot.lines[0].text == "123456789"
    assert snapshot.lines[0].column_width == 9
    assert snapshot.lines[1].text == "界"
    assert snapshot.lines[1].column_width == 2
    assert all(line.column_width <= 10 for line in snapshot.lines)


def test_writing_into_wide_continuation_clears_both_original_halves() -> None:
    model = TerminalScreenModel(columns=10, rows=2)

    model.feed("界".encode() + b"\x1b[DX")
    line = model.snapshot().lines[0]

    assert line.text == " X"
    assert line.column_width == 2


def test_variation_selector_width_change_at_right_edge_wraps_atomically() -> None:
    model = TerminalScreenModel(columns=10, rows=2)

    model.feed("123456789❤️X".encode())
    snapshot = model.snapshot()

    assert snapshot.lines[0].text == "123456789"
    assert snapshot.lines[1].text == "❤️X"
    assert snapshot.lines[1].column_width == 3
    assert all(line.column_width <= 10 for line in snapshot.lines)


def test_cell_scalar_overflow_is_replaced_and_counted_without_content() -> None:
    model = TerminalScreenModel(columns=10, rows=2)

    model.feed(("a" + "\u0301" * 40).encode())
    snapshot = model.snapshot()

    assert snapshot.lines[0].text == "\ufffd"
    assert snapshot.cell_overflow_count == 1
    assert all(
        len(cell.text) <= 32 and len(cell.text.encode()) <= 256
        for run in snapshot.lines[0].runs
        for cell in run.cells
    )


def test_cell_overflow_marker_moves_with_inserted_cells() -> None:
    model = TerminalScreenModel(columns=10, rows=2)
    model.feed(("Xa" + "\u0301" * 40).encode())

    model.feed(b"\r\x1b[@\x1b[3G")
    model.feed("\u0301".encode())

    assert model.snapshot().lines[0].text == " X\u0301�"


def test_cursor_savepoint_stack_evicts_the_oldest_after_sixteen() -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    for column in range(17):
        model.feed(f"\x1b[{column + 1}G\x1b7".encode())
    snapshot = model.snapshot()

    assert snapshot.cursor_savepoints == 16
    model.feed(b"\x1b8")
    assert model.snapshot().cursor_column == 17


def test_style_runs_are_safe_immutable_values() -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    model.feed(b"\x1b[1;3;4;31;44mX\x1b[0mY")
    snapshot = model.snapshot()
    first, second = snapshot.lines[0].runs

    assert first.text == "X"
    assert first.style.fg == "red"
    assert first.style.bg == "blue"
    assert first.style.bold is True
    assert first.style.italics is True
    assert first.style.underscore is True
    assert second.text == "Y"
    with pytest.raises(FrozenInstanceError):
        first.style.fg = "green"  # type: ignore[misc]


def test_trailing_styled_spaces_remain_visible_safe_cells() -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    model.feed(b"\x1b[44m   \x1b[0m")
    line = model.snapshot().lines[0]

    assert line.text == "   "
    assert line.column_width == 3
    assert len(line.runs) == 1
    assert line.runs[0].style.bg == "blue"


def test_alternate_screen_never_enters_normal_scrollback() -> None:
    model = TerminalScreenModel(columns=20, rows=2)
    model.feed(b"primary-1\r\nprimary-2")
    primary = model.visible_text()
    scrollback_before = model.snapshot().scrollback

    model.feed(b"\x1b[?1049halt-1\r\nalt-2\r\nalt-3")
    assert model.snapshot().in_alternate is True
    assert model.snapshot().scrollback == scrollback_before
    model.feed(b"\x1b[?1049l")

    assert model.snapshot().in_alternate is False
    assert model.snapshot().scrollback == scrollback_before
    assert model.visible_text() == primary


def test_resize_updates_both_screens_and_preserves_each_cursor() -> None:
    model = TerminalScreenModel(columns=10, rows=3)
    model.feed(b"primary\x1b[2;4H\x1b[?1049halt\x1b[3;5H")

    model.resize(columns=12, rows=4)

    alternate = model.snapshot()
    assert alternate.in_alternate is True
    assert (alternate.cursor_row, alternate.cursor_column) == (3, 5)
    assert len(alternate.lines) == 4
    model.feed(b"\x1b[?1049l")
    primary = model.snapshot()
    assert (primary.cursor_row, primary.cursor_column) == (2, 4)
    assert len(primary.lines) == 4


def test_reset_from_alternate_screen_restores_coherent_primary_state() -> None:
    model = TerminalScreenModel(columns=20, rows=2)
    model.feed(b"primary\x1b[?1049halt\x1bc")

    snapshot = model.snapshot()

    assert snapshot.in_alternate is False
    assert model.visible_text() == ""
    assert (snapshot.cursor_row, snapshot.cursor_column) == (1, 1)


def test_scrollback_accounting_is_text_plus_runs_plus_line_overhead() -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    model.feed(b"\x1b[31mA\x1b[0mB\r\nline-2\r\nline-3")
    retained = model.snapshot().scrollback[0]

    assert retained.text == "AB"
    assert len(retained.runs) == 2
    assert retained.accounted_bytes == 2 + 2 * 32 + 16
    assert model.snapshot().scrollback_bytes == retained.accounted_bytes


def test_scrollback_evicts_oldest_lines_at_the_line_limit() -> None:
    model = TerminalScreenModel(
        columns=20,
        rows=2,
        scrollback_line_limit=2,
        scrollback_byte_limit=4 * 1024 * 1024,
    )

    # Five logical lines in a two-row viewport produce three scrolled lines,
    # so this actually crosses the two-line limit.
    model.feed(b"A\r\nB\r\nC\r\nD\r\nE")

    assert [line.text for line in model.snapshot().scrollback] == ["B", "C"]


def test_scrollback_evicts_oldest_lines_at_the_byte_limit() -> None:
    model = TerminalScreenModel(
        columns=20,
        rows=2,
        scrollback_line_limit=5_000,
        scrollback_byte_limit=98,
    )

    # Each one-character/default-style line accounts for 49 bytes. Three
    # scrolled lines force oldest-first eviction at the 98-byte limit.
    model.feed(b"A\r\nB\r\nC\r\nD\r\nE")
    snapshot = model.snapshot()

    assert [line.text for line in snapshot.scrollback] == ["B", "C"]
    assert snapshot.scrollback_bytes == 98


@pytest.mark.parametrize(
    "control",
    [
        b"\x1b]0;HOST_TITLE\x07",
        b"\x1b]1;HOST_ICON\x1b\\",
        b"\x1b]8;;https://secret.invalid\x07link\x1b]8;;\x07",
        b"\x1b]9;NOTIFICATION_SECRET\x07",
        b"\x1b]52;c;CLIPBOARD_SECRET\x07",
        b"\x1bPDEVICE_SECRET\x1b\\",
    ],
)
def test_host_affecting_and_string_controls_never_reach_safe_cells(
    control: bytes,
) -> None:
    model = TerminalScreenModel(columns=80, rows=3)

    model.feed(b"before" + control + b"after")
    projected = _projected_text(model)

    expected = (
        "beforelinkafter\n\n"
        if b"https://secret.invalid" in control
        else "beforeafter\n\n"
    )
    assert projected == expected
    assert "SECRET" not in repr(model.snapshot())
    assert "secret.invalid" not in repr(model.snapshot())
    assert model.pending_replies() == ()


def test_allowlisted_device_queries_create_only_bounded_code_owned_replies() -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    model.feed(b"X\x1b[5n\x1b[6n\x1b[c")

    assert model.pending_replies() == (b"\x1b[0n", b"\x1b[1;2R", b"\x1b[?6c")
    assert all(len(reply) <= 256 for reply in model.pending_replies())
    assert model.take_pending_replies() == (
        b"\x1b[0n",
        b"\x1b[1;2R",
        b"\x1b[?6c",
    )
    assert model.pending_replies() == ()


def test_unsupported_csi_shapes_do_not_fail_the_screen_model() -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    model.feed(b"before\x1b[1;2A\x1b[?5n\x1b[>1mafter")

    assert model.visible_text() == "beforeafter"
    assert model.snapshot().failure_reason is None


QUALIFICATION_CORPUS = (
    (
        "parser-powershell-cmd-fixtures/cmd",
        b"Microsoft Windows [Version 10.0.17763]\r\nC:\\>dir\r\n",
    ),
    (
        "parser-powershell-cmd-fixtures/powershell",
        b"\x1b[93mPS C:\\>\x1b[0m Get-Command python\r\n",
    ),
    (
        "parser-full-screen-programs/editor",
        b"\x1b[?1049h\x1b[Heditor fixture\x1b[?1049l",
    ),
    (
        "parser-full-screen-programs/pager",
        b"\x1b[2J\x1b[H\x1b[7mpager fixture\x1b[0m\r\n:",
    ),
    (
        "parser-full-screen-programs/monitor",
        b"\x1b[H\x1b[7mprocess monitor fixture\x1b[0m",
    ),
    ("parser-unicode-cells", "A界e\u0301🙂".encode()),
    ("parser-alternate-screen", b"primary\x1b[?1049halt\x1b[?1049l"),
    ("parser-bracketed-paste", b"\x1b[?2004htext\x1b[?2004l"),
    ("parser-terminal-queries", b"\x1b[5n\x1b[6n\x1b[c"),
    (
        "parser-malformed-controls",
        b"\x1b[999999999999;::::m\xff\xfeplain\x1b]bad\x07",
    ),
    ("parser-incomplete-sequence-bounds/incomplete-utf8", b"\xe2\x82"),
    ("parser-incomplete-sequence-bounds/non-csi", b"\x1b" + b"x" * 16),
    ("parser-incomplete-sequence-bounds/csi", b"\x1b[" + b"1" * 257),
    (
        "parser-incomplete-sequence-bounds/parameters",
        b"\x1b[" + b"1;" * 33 + b"m",
    ),
    ("parser-incomplete-sequence-bounds/value", b"\x1b[10000m"),
    (
        "parser-incomplete-sequence-bounds/intermediates",
        b"\x1b[" + b" " * 17 + b"m",
    ),
    ("parser-incomplete-sequence-bounds/osc", b"\x1b]" + b"x" * 4096),
    ("parser-incomplete-sequence-bounds/dcs", b"\x1bP" + b"x" * 4096),
    ("parser-incomplete-sequence-bounds/apc", b"\x1b_" + b"x" * 4096),
    ("parser-incomplete-sequence-bounds/pm", b"\x1b^" + b"x" * 4096),
)


@pytest.mark.parametrize(("row_id", "fixture"), QUALIFICATION_CORPUS)
def test_qualification_corpus_projects_only_safe_content(
    row_id: str, fixture: bytes
) -> None:
    model = TerminalScreenModel(columns=120, rows=40)

    model.feed(b"before")
    _feed_in_seven_byte_chunks(model, fixture)
    model.feed(b"\x18after")
    model.finish()
    projected = _projected_text(model)

    assert model.snapshot().failure_reason is None, row_id
    # Full-screen fixtures may legitimately clear or overwrite the earlier
    # marker. The post-CAN marker proves recovery and keeps this assertion
    # discriminating against a neutral/no-op implementation.
    assert "after" in projected, row_id
    assert "\x1b" not in projected, row_id
    assert not any(
        unicodedata.category(character) in {"Cc", "Cs"}
        or 0x80 <= ord(character) <= 0x9F
        for character in projected
        if character != "\n"
    ), row_id


def test_parser_failure_stops_projection_with_a_content_free_reason(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    def fail_without_exposing_payload(_: str) -> None:
        raise RuntimeError("PARSER_SECRET")

    monkeypatch.setattr(model._stream, "feed", fail_without_exposing_payload)
    with caplog.at_level(logging.DEBUG):
        model.feed(b"OUTPUT_SECRET")
        model.feed(b"later")

    snapshot = model.snapshot()
    assert snapshot.failure_reason is TerminalReason.TERMINAL_PROTOCOL_FAILED
    assert snapshot.lines[0].text == ""
    assert "SECRET" not in repr(snapshot)
    assert "SECRET" not in caplog.text


def test_snapshot_is_immutable_and_contains_no_parser_owned_collections() -> None:
    model = TerminalScreenModel(columns=20, rows=2)
    model.feed(b"safe")
    snapshot = model.snapshot()

    assert isinstance(snapshot.lines, tuple)
    assert isinstance(snapshot.lines[0].runs, tuple)
    assert isinstance(snapshot.lines[0].runs[0].cells, tuple)
    with pytest.raises(FrozenInstanceError):
        snapshot.cursor_column = 99  # type: ignore[misc]


def test_every_mutable_parser_and_screen_collection_is_classified_and_bounded() -> None:
    model = TerminalScreenModel(columns=20, rows=2)
    model.feed(b"\x1b[1;2Hsafe")
    mutable_types = (MutableMapping, MutableSequence, MutableSet, deque, bytearray)

    def mutable_attributes(value: object) -> set[str]:
        return {
            name
            for name, member in vars(value).items()
            if isinstance(member, mutable_types)
        }

    assert mutable_attributes(model) == {"_scrollback", "_pending_replies"}
    assert mutable_attributes(model._gate) == {"_buffer"}
    assert mutable_attributes(model._stream) == set()
    assert mutable_attributes(model._screens) == set()

    parser_locals = model._stream._parser.gi_frame.f_locals
    parser_collections = {
        name: member
        for name, member in parser_locals.items()
        if isinstance(member, mutable_types)
    }
    assert set(parser_collections) == {
        "basic",
        "OSC_TERMINATORS",
        "basic_dispatch",
        "sharp_dispatch",
        "escape_dispatch",
        "csi_dispatch",
        "params",
    }
    assert parser_collections["basic"] is pyte.Stream.basic
    assert parser_collections["OSC_TERMINATORS"] == {"\x1b\\", "\x07", "\x9c"}
    assert set(parser_collections["basic_dispatch"]) == set(pyte.Stream.basic)
    assert set(parser_collections["sharp_dispatch"]) == set(pyte.Stream.sharp)
    assert set(parser_collections["escape_dispatch"]) == set(pyte.Stream.escape)
    assert set(parser_collections["csi_dispatch"]) == set(pyte.Stream.csi)
    assert len(parser_collections["params"]) <= 32

    for screen in (model._screens.primary, model._screens.alternate):
        assert mutable_attributes(screen) == {
            "savepoints",
            "buffer",
            "dirty",
            "mode",
            "tabstops",
        }
        assert len(screen.savepoints) <= 16
        assert len(screen.buffer) <= screen.lines
        assert all(len(line) <= screen.columns for line in screen.buffer.values())
        assert screen.dirty <= set(range(screen.lines))
        assert len(screen.mode) <= 10
        assert len(screen.tabstops) <= screen.columns


@pytest.mark.parametrize(
    "arguments",
    [
        {"columns": 4, "rows": 2},
        {"columns": 301, "rows": 2},
        {"columns": 20, "rows": 1},
        {"columns": 20, "rows": 121},
        {"columns": 20, "rows": 2, "scrollback_line_limit": 5_001},
        {
            "columns": 20,
            "rows": 2,
            "scrollback_byte_limit": 4 * 1024 * 1024 + 1,
        },
    ],
)
def test_screen_model_rejects_bounds_outside_the_terminal_contract(
    arguments: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        TerminalScreenModel(**arguments)


@pytest.mark.parametrize(
    ("columns", "rows"),
    [(4, 2), (301, 2), (20, 1), (20, 121)],
)
def test_resize_rejects_bounds_outside_the_terminal_contract(
    columns: int, rows: int
) -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    with pytest.raises(ValueError):
        model.resize(columns=columns, rows=rows)


def test_pending_reply_queue_is_capped_at_the_aggregate_reply_bound() -> None:
    model = TerminalScreenModel(columns=20, rows=2)

    model.feed(b"\x1b[5n" * 1_100)

    replies = model.pending_replies()
    assert sum(map(len, replies)) == 4 * 1024
    assert all(reply == b"\x1b[0n" for reply in replies)
