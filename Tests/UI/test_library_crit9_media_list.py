"""Media list fixes from critique #9 (tasks 32210, 32213, 32227).

Built on the media host from ``test_library_media_render_fixes`` (green on
dev, unlike the 19k-line shell file) so these run against the REAL screen
path with the production stylesheet — the three defects here are all
paint/layout ones that a headless ``query_one`` never sees.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, OptionList, Static

from Tests.UI.test_library_media_render_fixes import _host, _painted
from Tests.UI.test_library_media_side_by_side import _open_media_list
from Tests.UI.test_library_shell import (
    _wait_for_condition,
    _wait_for_selector,
)


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
        painted = _painted(host, choices.region)
        assert "█" in painted, painted


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
        painted = _painted(host, choices.region)
        assert "█" in painted, painted


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
