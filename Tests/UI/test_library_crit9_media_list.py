"""Media list fixes from critique #9 (tasks 32210, 32213, 32227).

Built on the media host from ``test_library_media_render_fixes`` (green on
dev, unlike the 19k-line shell file) so these run against the REAL screen
path with the production stylesheet — the three defects here are all
paint/layout ones that a headless ``query_one`` never sees.
"""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Library.library_shell_state import LIBRARY_CHOICE_ACTIVE_MARKER
from tldw_chatbook.Widgets.Library.library_choice_strip import (
    LIBRARY_CHOICE_CURSOR,
    LibraryChoiceOptionList,
)
from tldw_chatbook.Widgets.Library.library_media_canvas import (
    LIBRARY_MEDIA_REVIEW_EMPTY_TOOLTIP,
)

from Tests.UI.test_library_media_render_fixes import _host, _painted
from Tests.UI.test_library_media_side_by_side import _open_media_list
from Tests.UI.test_library_shell import (
    _wait_for_condition,
    _wait_for_selector,
)


def _assert_painted_cursor_row(host, choices, *, expected_row: int) -> None:
    """The `█` reaches the screen on exactly one row — the highlighted one.

    Both halves in the PAINT, not only in the prompts (32210 AC#2): a
    prompt-level check cannot see an option row clipped away or a second
    row picking the glyph up from somewhere else.
    """
    lines = _painted(host, choices.region).splitlines()
    marked = [index for index, line in enumerate(lines) if "█" in line]
    assert marked == [expected_row], lines


@pytest.mark.asyncio
async def test_the_type_chooser_marks_its_cursor_with_the_house_bar():
    """task-32210: the chooser cursor is the list's ``█`` bar, not a tint.

    Before this the only cue was the OptionList's
    ``option-list--option-highlighted`` background swap, measured live at
    1.09:1 (critique #9 register row 6) — colour-only, and invisible in a
    plain-text capture of a keyboard interaction the footer commits the
    user to.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-type-filter", Button).press()
        choices = await _wait_for_selector(
            screen, pilot, "#library-media-type-choices"
        )
        choices.highlighted = 1
        await pilot.pause()
        prompts = [
            str(choices.get_option_at_index(index).prompt)
            for index in range(choices.option_count)
        ]
        assert prompts[1].startswith("█ "), prompts
        assert [p for p in prompts if p.startswith("█ ")] == [prompts[1]], prompts
        _assert_painted_cursor_row(host, choices, expected_row=1)


@pytest.mark.asyncio
async def test_the_sort_chooser_marks_its_cursor_with_the_house_bar():
    """Same cue on the sort chooser — one grammar across both choosers."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-sort", Button).press()
        choices = await _wait_for_selector(
            screen, pilot, "#library-media-sort-choices"
        )
        choices.highlighted = 2
        await pilot.pause()
        prompts = [
            str(choices.get_option_at_index(index).prompt)
            for index in range(choices.option_count)
        ]
        assert prompts[2].startswith("█ "), prompts
        assert [p for p in prompts if p.startswith("█ ")] == [prompts[2]], prompts
        _assert_painted_cursor_row(host, choices, expected_row=2)


async def _apply_media_type(screen, pilot, media_type: str):
    """Pick a type through the real chooser, exactly as a user would."""
    screen.query_one("#library-media-type-filter", Button).press()
    chooser = await _wait_for_selector(screen, pilot, "#library-media-type-choices")
    chooser.highlighted = next(
        index
        for index, option in enumerate(chooser.options)
        if getattr(option, "choice_value", None) == media_type
    )
    chooser.action_select()
    await _wait_for_condition(
        pilot,
        lambda: screen._media_state.type_filter == media_type,
        message=f"the {media_type} type never applied",
    )
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_zero_result_filter_keeps_the_type_facet_and_names_it():
    """task-32213: a filter miss must not swallow the toolbar.

    Critique #9 register row 10: with ``type: video`` and a query matching
    nothing the canvas returned right after the miss sentence, so the type
    that PRODUCED the empty page could neither be seen nor reset from the
    canvas — against the guide's own rule that a filtered empty page keeps
    its submitted facets visible until the user chooses the reset.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_type(screen, pilot, "video")
        filter_box = screen.query_one("#library-media-filter", Input)
        filter_box.value = "zzzznomatch"
        screen.set_focus(filter_box)
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: not screen.query("#library-media-row-0"),
            message="the filter miss never emptied the list",
        )
        assert screen.query("#library-media-type-filter"), (
            "the type facet must survive a 0-result page"
        )
        assert screen.query("#library-media-sort")
        assert screen.query("#library-media-trash-open")
        status = screen.query_one("#library-media-status", Static)
        assert str(status.content) == (
            "No media of type 'video' matched “zzzznomatch” "
            "in titles, content or keywords."
        ), str(status.content)
        assert screen.query("#library-media-empty-clear-type"), (
            "a type filter needs its reset"
        )
        assert not screen.query("#library-media-empty-import"), (
            "a filter miss never suggests Import (task-31224)"
        )
        # One `#library-media-status`, not two: the fall-through status line
        # below the toolbar is suppressed while this branch owns the copy.
        assert len(screen.query("#library-media-status")) == 1
        # task-32213 review, finding 3: keeping the toolbar means
        # `#library-media-type-filter` is now always an enabled fallback on
        # this page, which would have landed entry focus on the facet the
        # user did NOT type into. With a query in force the filter Input
        # leads instead -- the place to retype.
        assert (
            screen._library_media_empty_list_fallback_target()
            is screen.query_one("#library-media-filter", Input)
        )
        # Finding 4: no list-wide action may claim it can act on a page
        # with nothing on it. "Review these" carries the same "○" marker
        # and reason as "Select"; "Export…" is deliberately still live
        # (its scope is the type, not the query).
        review = screen.query_one("#library-media-review", Button)
        assert review.disabled is True
        assert str(review.label).startswith("○ ")
        assert str(review.tooltip) == LIBRARY_MEDIA_REVIEW_EMPTY_TOOLTIP


async def _enter_select_mode(screen, pilot):
    """Enter Select mode the way the footer advertises it."""
    await pilot.press("s")
    await _wait_for_condition(
        pilot,
        lambda: screen._media_state.select_mode is True,
        message="Select mode never armed",
    )
    await pilot.pause()


@pytest.mark.parametrize("size", ((100, 30), (60, 24)))
@pytest.mark.asyncio
async def test_the_shared_count_margin_never_clips_the_trash_heading(size):
    """task-32227 review, finding 5: the margin rides a heading too.

    ``library-toolbar-count`` is not only on counters — the Trash canvas
    puts it on its TITLE ``Static``, the last child of a ``height: 1``,
    ``overflow: hidden`` row whose budget task-28015 already measured to
    the cell (it painted "Local Trash · 1 i" at ~38 columns before that
    fix). A right margin spends one of those columns, so the heading's
    paint is pinned at both narrow widths.

    Through the REAL screen (`_host`), not the trash canvas harness:
    ``.library-toolbar-count`` lives in the Library SCREEN sheet, which
    `ConsolidatedCSSApp` does not register — under that harness the title's
    computed margin is 0 and this would measure nothing.
    """
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-trash-open", Button).press()
        title = await _wait_for_selector(
            screen, pilot, "#library-media-trash-title"
        )
        await pilot.pause()
        assert title.styles.margin.right == 1, title.styles.margin
        # The HEADING row, not the title's own region: the row is
        # ``overflow: hidden``, so it is the row's paint that loses the
        # tail when the budget runs out (the title's region happily
        # reports a width the compositor then clips away).
        heading = screen.query_one("#library-media-trash-heading")
        painted = _painted(host, heading.region)
        assert str(title.renderable) in painted, (painted, title.renderable)


@pytest.mark.parametrize("size", ((235, 52), (100, 30)))
@pytest.mark.asyncio
async def test_the_select_count_never_touches_the_first_action(size):
    """task-32227: the count and the next action shared a cell.

    Critique #9 register row 26: the select strip painted
    ``2 selected┃ Select all`` — the counter's last cell and the focused
    button's heavy border touched, so the two read as one control. Fixed on
    the shared ``.library-toolbar-count`` class, which every canvas's
    counter already carries, so this holds at both widths.
    """
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _enter_select_mode(screen, pilot)
        count = screen.query_one("#library-media-selected-count", Static)
        first = screen.query_one("#library-media-select-all", Button)
        assert first.region.x - count.region.right >= 1, (
            count.region,
            first.region,
        )


@pytest.mark.asyncio
async def test_the_chooser_cursor_keeps_the_active_marker_and_the_pick_payload():
    """``█`` rides in front of ``✓``; ``choice_value`` survives the rewrite."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-type-filter", Button).press()
        choices = await _wait_for_selector(
            screen, pilot, "#library-media-type-choices"
        )
        await pilot.pause()
        # "All types" is the active option AND the one the screen highlights
        # on open, so it carries both marks.
        assert str(choices.get_option_at_index(0).prompt) == "█ ✓ All types"
        assert [
            getattr(choices.get_option_at_index(index), "choice_value", "missing")
            for index in range(choices.option_count)
        ] == [None, "audio", "video"]


class _ChoiceListApp(App):
    """The bare widget, no screen: LibraryChoiceOptionList on its own."""

    def __init__(self, labels, active_index):
        super().__init__()
        self._labels = labels
        self._active_index = active_index

    def compose(self):
        options = []
        for index, label in enumerate(self._labels):
            option = Option(
                f"{LIBRARY_CHOICE_ACTIVE_MARKER} {label}"
                if index == self._active_index
                else label,
                id=f"choice-{index}",
            )
            option.choice_value = label
            options.append(option)
        chooser = LibraryChoiceOptionList(
            *options, id="probe-choices", compact=True, markup=False
        )
        chooser.highlighted = self._active_index
        yield chooser


def _probe_prompts(app):
    chooser = app.query_one("#probe-choices", LibraryChoiceOptionList)
    return chooser, [
        str(chooser.get_option_at_index(index).prompt)
        for index in range(chooser.option_count)
    ]


@pytest.mark.asyncio
async def test_the_cursor_never_eats_a_label_that_starts_with_the_bar():
    """Qodo #4: a stored media type may legitimately start with ``█ ``.

    The first implementation recovered each option's base text with
    ``removeprefix(LIBRARY_CHOICE_CURSOR)``, which cannot tell its own
    cursor from the same two characters in the label — so a type named
    ``█ redacted`` lost them the moment the chooser painted, and the
    visible option stopped agreeing with the ``choice_value`` payload
    behind it. The facet path accepts any non-empty string (a type
    literally named "All" is already a pinned case), so this is real data.
    """
    labels = ["All types", "█ redacted", "video"]
    app = _ChoiceListApp(labels, active_index=0)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        chooser, prompts = _probe_prompts(app)
        assert prompts == ["█ ✓ All types", "█ redacted", "video"], prompts

        # Onto the bar-named option: it gains a cursor, keeps its name.
        chooser.highlighted = 1
        await pilot.pause()
        _, prompts = _probe_prompts(app)
        assert prompts == ["✓ All types", "█ █ redacted", "video"], prompts

        # And away again: the name is restored whole, not one prefix short.
        chooser.highlighted = 2
        await pilot.pause()
        _, prompts = _probe_prompts(app)
        assert prompts == ["✓ All types", "█ redacted", "█ video"], prompts
        assert [
            getattr(chooser.get_option_at_index(index), "choice_value", "missing")
            for index in range(chooser.option_count)
        ] == labels


@pytest.mark.asyncio
async def test_the_cursor_moves_one_option_at_a_time_off_screen():
    """Qodo #3: the class's own contract, without a screen around it.

    Exactly one option carries the cursor at any time, it rides in front
    of the ``✓`` active marker, and every option's ``choice_value``
    survives the in-place prompt rewrite.
    """
    app = _ChoiceListApp(["All types", "audio", "video"], active_index=1)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        chooser, prompts = _probe_prompts(app)
        assert prompts == ["All types", "█ ✓ audio", "video"], prompts

        for index, expected in enumerate(
            (
                ["█ All types", "✓ audio", "video"],
                ["All types", "█ ✓ audio", "video"],
                ["All types", "✓ audio", "█ video"],
            )
        ):
            chooser.highlighted = index
            await pilot.pause()
            _, prompts = _probe_prompts(app)
            assert prompts == expected, (index, prompts)
            assert sum(p.startswith("█ ") for p in prompts) == 1, prompts
