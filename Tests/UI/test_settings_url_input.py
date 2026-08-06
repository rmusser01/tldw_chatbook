"""Pinning tests for SettingsURLInput's zero-width-break display math.

task-1375 audit result: the widget keeps ``value`` raw and only rewrites the
RENDERED text (a zero-width break after ``http``/``https`` schemes so
textual-web browsers do not autolink endpoints). Because the break is a
zero-cell character, all editing operations (cursor left/right, Home/End,
selection, backspace/delete) run on the raw value and are correct by
construction; the only raw->display mapping lives in
``_textual_web_safe_url_display_index`` and the two stylize calls in
``render_line``. These tests pin that behavior so a regression presents as a
failed test instead of "settings is broken".
"""

import pytest
from rich.cells import cell_len
from textual.app import App
from textual.strip import Strip
from textual.widgets._input import Selection

from tldw_chatbook.UI.Screens.settings_screen import (
    TEXTUAL_WEB_URL_AUTOLINK_BREAK as BREAK,
    SettingsURLInput,
    _textual_web_safe_url_display,
    _textual_web_safe_url_display_index,
)


# ---------------------------------------------------------------------------
# Pure display-mapping helpers
# ---------------------------------------------------------------------------


def test_url_display_inserts_break_only_after_schemes():
    assert _textual_web_safe_url_display("https://api.example.com") == (
        f"https{BREAK}://api.example.com"
    )
    assert _textual_web_safe_url_display("http://localhost:8000/v1") == (
        f"http{BREAK}://localhost:8000/v1"
    )
    # No scheme -> untouched.
    assert _textual_web_safe_url_display("localhost:8000") == "localhost:8000"
    # Case-insensitive scheme, original casing preserved.
    assert _textual_web_safe_url_display("HTTPS://EXAMPLE.COM") == (
        f"HTTPS{BREAK}://EXAMPLE.COM"
    )
    # The stored value is recoverable: the break is the only addition.
    endpoint = "http://localhost:8000/v1/chat/completions"
    assert _textual_web_safe_url_display(endpoint).replace(BREAK, "") == endpoint


def test_url_display_handles_multiple_schemes():
    value = "http://a -> https://b"
    display = _textual_web_safe_url_display(value)
    assert display == f"http{BREAK}://a -> https{BREAK}://b"
    assert display.replace(BREAK, "") == value


def test_display_index_maps_every_raw_index_to_matching_boundary():
    """For every raw cursor index, the mapped display index must split the
    display string at the same text boundary as the raw index splits value."""
    for value in (
        "https://api.example.com/v1",
        "http://a https://b",
        "no scheme here",
    ):
        display = _textual_web_safe_url_display(value)
        for index in range(len(value) + 1):
            display_index = _textual_web_safe_url_display_index(value, index)
            assert display[:display_index].replace(BREAK, "") == value[:index]
            # The break belongs to the preceding scheme text: a raw index at
            # the insertion boundary maps AFTER the inserted break.
            assert 0 <= display_index <= len(display)


def test_display_index_counts_each_inserted_break_once():
    value = "http://a https://b"
    first_break = len("http")
    second_break = value.index("https") + len("https")
    assert _textual_web_safe_url_display_index(value, first_break - 1) == (
        first_break - 1
    )
    assert _textual_web_safe_url_display_index(value, first_break) == (
        first_break + 1
    )
    assert _textual_web_safe_url_display_index(value, second_break) == (
        second_break + 2
    )
    assert _textual_web_safe_url_display_index(value, len(value)) == (
        len(value) + 2
    )


def test_zero_width_break_occupies_zero_cells():
    """The scroll/cursor cell math (Input._cursor_offset, Strip.crop) is only
    correct because the break is zero cells wide; pin that invariant."""
    assert cell_len(BREAK) == 0
    assert cell_len(f"https{BREAK}://x") == cell_len("https://x")


# ---------------------------------------------------------------------------
# Widget-level rendering / editing behavior
# ---------------------------------------------------------------------------

VALUE = "https://api.example.com/v1"


class _URLInputTestApp(App[None]):
    def compose(self):
        yield SettingsURLInput(value=VALUE, id="url")


@pytest.fixture
async def app_and_pilot():
    app = _URLInputTestApp()
    async with app.run_test(size=(60, 5)) as pilot:
        yield app, pilot


def _styled_cells(strip: Strip) -> list[tuple[int, str]]:
    """Cells whose style differs from the trailing padding's base style."""
    cells: list[tuple[str, str]] = []
    for segment in strip:
        for char in segment.text:
            cells.append((char, repr(segment.style)))
    base_style = cells[-1][1]
    return [(i, char) for i, (char, style) in enumerate(cells) if style != base_style]


def _freeze_cursor_blink(widget: SettingsURLInput) -> None:
    widget.cursor_blink = False
    widget._cursor_visible = True


@pytest.mark.asyncio
async def test_rendered_text_matches_display_without_visual_change(app_and_pilot):
    app, pilot = app_and_pilot
    widget = app.query_one("#url", SettingsURLInput)
    await pilot.pause()

    rendered = widget.render_line(0).text.rstrip()
    assert rendered == f"https{BREAK}://api.example.com/v1"
    # No visual change in normal terminal rendering: stripping the zero-width
    # break reproduces the raw value, and the cell width is unchanged.
    assert rendered.replace(BREAK, "") == VALUE
    assert cell_len(rendered) == cell_len(VALUE)
    assert widget.value == VALUE  # stored value stays raw


@pytest.mark.asyncio
async def test_cursor_style_lands_on_correct_cell_around_break(app_and_pilot):
    app, pilot = app_and_pilot
    widget = app.query_one("#url", SettingsURLInput)
    widget.focus()
    _freeze_cursor_blink(widget)
    await pilot.pause()

    display = _textual_web_safe_url_display(VALUE)

    # raw 4 (http|s) -> display 4, the cursor block sits on 's'.
    widget.selection = Selection.cursor(4)
    await pilot.pause()
    widget._cursor_visible = True
    assert _styled_cells(widget.render_line(0)) == [(4, display[4])]
    assert display[4] == "s"

    # raw 5 (https|:) -> display 6, skipping the break: cursor sits on ':'.
    widget.selection = Selection.cursor(5)
    await pilot.pause()
    widget._cursor_visible = True
    assert _styled_cells(widget.render_line(0)) == [(6, display[6])]
    assert display[6] == ":"

    # raw 6 (:|/) -> display 7, cursor sits on '/'.
    widget.selection = Selection.cursor(6)
    await pilot.pause()
    widget._cursor_visible = True
    assert _styled_cells(widget.render_line(0)) == [(7, display[7])]
    assert display[7] == "/"

    # End of input -> cursor on the padded cell appended after the display
    # text (display index len(display)).
    widget.selection = Selection.cursor(len(VALUE))
    await pilot.pause()
    widget._cursor_visible = True
    styled = _styled_cells(widget.render_line(0))
    assert styled == [(len(display), " ")]


@pytest.mark.asyncio
async def test_selection_style_spans_break_correctly(app_and_pilot):
    app, pilot = app_and_pilot
    widget = app.query_one("#url", SettingsURLInput)
    widget.focus()
    _freeze_cursor_blink(widget)
    await pilot.pause()

    # raw (3, 8) = "ps://": display span covers 'p','s', the break, ':','/','/'
    # plus the cursor cell at display 9 ('a').
    widget.selection = Selection(3, 8)
    await pilot.pause()
    widget._cursor_visible = True
    styled = _styled_cells(widget.render_line(0))
    assert styled == [
        (3, "p"),
        (4, "s"),
        (5, BREAK),
        (6, ":"),
        (7, "/"),
        (8, "/"),
        (9, "a"),
    ]

    # raw (0, 5) = "https": display end maps past the break, so the highlight
    # stops before ':' while the cursor rests on it (display 6).
    widget.selection = Selection(0, 5)
    await pilot.pause()
    widget._cursor_visible = True
    styled = _styled_cells(widget.render_line(0))
    assert (6, ":") in styled
    assert (7, "/") not in styled
    # Zero-width break inside the highlight is invisible but harmless.
    assert (5, BREAK) in styled


@pytest.mark.asyncio
async def test_deletion_at_and_around_break_positions(app_and_pilot):
    app, pilot = app_and_pilot
    widget = app.query_one("#url", SettingsURLInput)
    widget.focus()
    await pilot.pause()

    # Backspace at the break boundary (raw 5) deletes 's' from the RAW value.
    widget.selection = Selection.cursor(5)
    await pilot.pause()
    await pilot.press("backspace")
    assert widget.value == "http" + "://api.example.com/v1"
    assert BREAK not in widget.value
    assert widget.cursor_position == 4

    # Delete just before the boundary (raw 4) also removes 's'.
    widget.value = VALUE
    widget.selection = Selection.cursor(4)
    await pilot.pause()
    await pilot.press("delete")
    assert widget.value == "http" + "://api.example.com/v1"
    assert widget.cursor_position == 4

    # Select-all across the break and delete clears everything.
    widget.value = VALUE
    widget.selection = Selection(0, len(VALUE))
    await pilot.pause()
    await pilot.press("backspace")
    assert widget.value == ""


@pytest.mark.asyncio
async def test_cursor_movement_and_home_end_ignore_break(app_and_pilot):
    app, pilot = app_and_pilot
    widget = app.query_one("#url", SettingsURLInput)
    widget.focus()
    await pilot.pause()

    await pilot.press("home")
    assert widget.cursor_position == 0
    await pilot.press("end")
    assert widget.cursor_position == len(VALUE)

    # Arrow keys step through RAW positions: the break never swallows a step.
    widget.selection = Selection.cursor(4)
    await pilot.pause()
    await pilot.press("right")
    assert widget.cursor_position == 5
    await pilot.press("right")
    assert widget.cursor_position == 6
    await pilot.press("left")
    assert widget.cursor_position == 5


@pytest.mark.asyncio
async def test_scrolled_render_keeps_cursor_cell_consistent(app_and_pilot):
    """With the input scrolled, the cropped strip still shows the cursor on
    the correct character (cell math is unaffected by the zero-cell break)."""
    app, pilot = app_and_pilot
    widget = app.query_one("#url", SettingsURLInput)
    widget.focus()
    _freeze_cursor_blink(widget)
    widget.styles.width = 12
    await pilot.pause()

    # Cursor mid-URL: strip is cropped to the scroll offset; the styled cell
    # must be the raw character under the cursor.
    widget.selection = Selection.cursor(10)
    await pilot.pause()
    widget._cursor_visible = True
    scroll_x, _ = widget.scroll_offset
    strip = widget.render_line(0)
    styled = _styled_cells(strip)
    assert len(styled) == 1
    styled_index, styled_char = styled[0]
    assert styled_char == VALUE[10]
    assert styled_index == widget._display_index(10) - scroll_x

    # Cursor at end: the padded cursor cell stays visible in the crop.
    widget.selection = Selection.cursor(len(VALUE))
    await pilot.pause()
    widget._cursor_visible = True
    styled = _styled_cells(widget.render_line(0))
    assert len(styled) == 1
    assert styled[0][1] == " "


@pytest.mark.asyncio
async def test_password_mode_bypasses_display_rewrite(app_and_pilot):
    app, pilot = app_and_pilot
    widget = app.query_one("#url", SettingsURLInput)
    widget.password = True
    await pilot.pause()

    assert widget._display_index(5) == 5
    assert str(widget._value) == "•" * len(VALUE)
