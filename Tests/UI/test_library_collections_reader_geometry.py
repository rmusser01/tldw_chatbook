"""Exact adaptive geometry for the Collections capture reader."""

from __future__ import annotations

import pytest
from textual import on
from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Utils.adaptive_reader_state import (
    PANE_GRIP_WIDTH,
    AdaptiveReaderLayoutPreferences,
    AdaptiveReaderLayoutProfile,
    resolve_adaptive_reader_layout,
)
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    AdaptiveReaderShellResized,
    LibraryAdaptiveReaderShell,
)


PROFILE = AdaptiveReaderLayoutProfile(work_min_width=48, work_comfort_width=56)
PREFERENCES = AdaptiveReaderLayoutPreferences()


@pytest.mark.parametrize(
    ("width", "expected"),
    (
        (160, (30, 40, 80)),
        (120, (0, 56, 54)),
        (100, (0, 42, 48)),
        (80, (0, 0, 70)),
    ),
)
def test_collections_profile_has_pinned_pure_geometry(width, expected) -> None:
    layout = resolve_adaptive_reader_layout(width, PREFERENCES, PROFILE)

    assert (layout.library_width, layout.items_width, layout.reader_width) == expected


class _GeometryApp(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.layout = resolve_adaptive_reader_layout(0, PREFERENCES, PROFILE)

    def compose(self):
        yield LibraryAdaptiveReaderShell(
            library=Static("Library"),
            items=Static("Items"),
            work=Static("Work"),
            layout=self.layout,
            id_prefix="library-collections",
            library_label="Library",
            items_label="Items",
            id="library-collections-reader-shell",
        )

    @on(AdaptiveReaderShellResized)
    def resolve_settled_layout(self, event: AdaptiveReaderShellResized) -> None:
        event.stop()
        shell = self.query_one("#library-collections-reader-shell", LibraryAdaptiveReaderShell)
        self.layout = resolve_adaptive_reader_layout(
            shell.content_size.width,
            PREFERENCES,
            PROFILE,
            previous=self.layout if self.layout.reader_width else None,
        )
        shell.sync_layout(self.layout)


@pytest.mark.parametrize("size", ((160, 50), (120, 35), (100, 30), (80, 24)))
async def test_mounted_collections_geometry_matches_one_settled_resolution(size) -> None:
    app = _GeometryApp()

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        shell = app.query_one("#library-collections-reader-shell", LibraryAdaptiveReaderShell)
        measured_width = shell.content_size.width
        expected = resolve_adaptive_reader_layout(measured_width, PREFERENCES, PROFILE)

        assert shell.effective_layout == expected
        assert shell.library.region.width == expected.library_width
        assert shell.items.region.width == expected.items_width
        assert shell.work.region.width == expected.reader_width
        assert shell.library_grip.region.width == PANE_GRIP_WIDTH
        assert shell.items_grip.region.width == PANE_GRIP_WIDTH
        assert shell.work.is_mounted and shell.work.display
        assert sum(child.region.width for child in shell.children) == measured_width
