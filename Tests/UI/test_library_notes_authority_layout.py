"""The Notes next-step guidance must survive actual compact pane widths."""

from pathlib import Path
from typing import ClassVar

import pytest
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)


class _AuthorityHarness(LibraryHarness):
    CSS_PATH: ClassVar[list[Path]] = list(APP_STYLESHEETS)


def _painted_text(screen, widget):
    region = widget.content_region.intersection(screen.region)
    return " ".join(
        strip.crop(region.x, region.right).text.strip()
        for strip in screen._compositor.render_strips()[region.y : region.bottom]
    ).strip()


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("empty", [True, False])
@private_profile_test
async def test_notes_next_step_is_fully_painted_across_pane_resize(
    request, theme, empty
):
    # A fixed two-row cap loses "files." at 80 columns even though the
    # Static's renderable still contains it. Assert the visible compositor.
    app = _build_test_app()
    _seed_conversations(app, [], notes=[] if empty else _two_notes())
    host = _AuthorityHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        host.theme = theme
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row("browse-notes")
        await _wait_for_selector(screen, pilot, "#library-notes-authority")

        for size in (
            (80, 24),
            (60, 24),
            (64, 24),
            (100, 30),
            (120, 36),
            (170, 48),
            (80, 24),
        ):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            authority = screen.query_one("#library-notes-authority", Static)
            expected = "Ready · Next: Create a note or add from files."
            # The narrow stage may omit the noun already in the source strip.
            # Both presentations must paint the entire next action.
            assert (
                _painted_text(screen, authority).removeprefix("Library notes · ")
                == expected
            ), (
                size,
                authority.region,
            )
            for selector in ("#library-notes-new", "#library-notes-add-from-files"):
                action = screen.query_one(selector, Button)
                assert action in screen._compositor.visible_widgets
                assert str(action.label).strip() in _painted_text(screen, action)

            field = screen.query_one("#library-notes-filter")
            field.focus()
            await pilot.pause()
            assert screen.focused is field
            assert field in screen._compositor.visible_widgets
