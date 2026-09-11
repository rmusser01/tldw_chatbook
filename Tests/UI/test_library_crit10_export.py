"""Library critique #10: the Export canvas states its consequence, and every
blocked control carries its reason on the next line.

task-32353 AC#2 -- the canvas asked for a destination and a name and then
wrote a bundle nobody had seen the contents of. A consequence line (item
count, fidelity, estimated size) and a contents list now sit directly above
the button.

task-32362 -- "Export bundle (.zip)" carried no inline reason ("No
destination chosen" sat three rows up), and the Reader's "○ Find" had one
only on hover. Both now render the reason on the line below the control, in
the ``library-media-action-reason`` grammar task-31981 established.

Widget-render and geometry assertions use a real Textual ``Pilot``
(``App.run_test()``) against the consolidated widget CSS the real app loads;
the DB read that feeds the consequence line is exercised against a real
in-memory ``MediaDatabase`` in ``Tests/Library/test_library_export_scope.py``.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_export_state import (
    EXPORT_BUTTON_NO_DESTINATION_TOOLTIP,
    build_library_export_form_state,
)
from tldw_chatbook.Library.library_media_viewer_state import (
    build_library_media_viewer_state,
)
from tldw_chatbook.Widgets.Library.library_export_canvas import LibraryExportCanvas
from tldw_chatbook.Widgets.Library.library_media_viewer import LibraryMediaViewer


def _state(**overrides):
    base = dict(
        scope=ExportScope(kind="media", ids=("1", "2")),
        counts={"media": 2, "conversations": 0, "notes": 0, "prompts": 0},
        name="Library export 2026-09-11",
        description="",
        media_quality="original",
        destination="/tmp/out.zip",
        titles=("Attention Is All You Need", "Deep Residual Learning"),
        approx_bytes=4096,
    )
    base.update(overrides)
    return build_library_export_form_state(**base)


class _ExportHost(ConsolidatedCSSApp):
    """Minimal host: one export canvas, nothing else."""

    def __init__(self, state):
        super().__init__()
        self._state = state

    def compose(self):
        yield LibraryExportCanvas(self._state, id="library-export-canvas")


# --- task-32353 AC#2: the bundle is stated before it is written -------------


@pytest.mark.asyncio
async def test_the_export_canvas_says_what_the_bundle_will_contain():
    app = _ExportHost(_state())
    async with app.run_test(size=(235, 52)) as pilot:
        line = pilot.app.query_one("#library-export-consequence-line", Static)
        assert (
            str(line.renderable) == "Bundle: 2 media items · full files · about 4 KB before compression"
        ), str(line.renderable)
        assert line.display is True
        contents = pilot.app.query_one("#library-export-contents", Static)
        assert "Attention Is All You Need" in str(contents.renderable)
        assert "Deep Residual Learning" in str(contents.renderable)
        assert contents.display is True


@pytest.mark.asyncio
async def test_the_consequence_line_names_the_chosen_fidelity():
    """The fidelity is the knob's consequence, not its label -- a lossy
    bundle says so where the user is about to press."""
    app = _ExportHost(_state(media_quality="thumbnail"))
    async with app.run_test(size=(235, 52)) as pilot:
        line = pilot.app.query_one("#library-export-consequence-line", Static)
        assert "previews only" in str(line.renderable)


def test_an_unsizeable_scope_says_so_instead_of_guessing():
    """task-32353 AC#2: ``None`` bytes is honest copy, never a zero."""
    state = _state(
        scope=ExportScope(kind="everything"),
        counts={"media": 11, "conversations": 6, "notes": 7, "prompts": 5},
        titles=(),
        approx_bytes=None,
    )
    assert state.consequence_line == (
        "Bundle: 29 items · full files · size known once it runs"
    )
    assert state.contents_lines == ()


def test_a_long_contents_list_is_capped_and_says_how_many_it_hid():
    """The query fetches 21 titles at most, so the remainder is derived from
    the counts -- the only place the true total lives (review Low 3)."""
    state = _state(
        scope=ExportScope(kind="media"),
        counts={"media": 250, "conversations": 0, "notes": 0, "prompts": 0},
        titles=tuple(f"Item {n}" for n in range(21)),
    )
    assert len(state.contents_lines) == 21
    assert state.contents_lines[:20] == tuple(f"Item {n}" for n in range(20))
    assert state.contents_lines[-1] == "+ 230 more"


def test_a_contents_list_that_exactly_fills_the_cap_hides_nothing():
    state = _state(
        scope=ExportScope(kind="media"),
        counts={"media": 20, "conversations": 0, "notes": 0, "prompts": 0},
        titles=tuple(f"Item {n}" for n in range(20)),
    )
    assert len(state.contents_lines) == 20
    assert not any(line.startswith("+ ") for line in state.contents_lines)


def test_the_consequence_line_is_empty_until_the_counts_land():
    state = _state(counts=None)
    assert state.counts_loading is True
    assert state.consequence_line == ""


@pytest.mark.asyncio
async def test_both_lines_stay_mounted_while_counting_so_the_patcher_finds_them():
    """The counts-landing patcher updates in place, never by recompose --
    so both lines must already exist, display-toggled off."""
    app = _ExportHost(_state(counts=None))
    async with app.run_test(size=(235, 52)) as pilot:
        assert (
            pilot.app.query_one("#library-export-consequence-line", Static).display
            is False
        )
        assert pilot.app.query_one("#library-export-contents", Static).display is False


# --- task-32362: a blocked control's reason sits beside it ------------------


@pytest.mark.asyncio
async def test_the_blocked_export_button_carries_its_reason_on_the_next_line():
    app = _ExportHost(_state(destination=""))
    async with app.run_test(size=(235, 52)) as pilot:
        button = pilot.app.query_one("#library-export-submit", Button)
        reason = pilot.app.query_one("#library-export-submit-reason", Static)
        assert button.disabled
        assert str(button.label).startswith("○ ")
        # One source, not a second sentence: the inline line IS the tooltip.
        assert str(reason.renderable) == button.tooltip
        assert str(reason.renderable) == EXPORT_BUTTON_NO_DESTINATION_TOOLTIP
        assert reason.display is True
        assert reason.region.y == button.region.y + button.region.height


@pytest.mark.asyncio
async def test_a_ready_export_button_shows_no_reason_line():
    app = _ExportHost(_state())
    async with app.run_test(size=(235, 52)) as pilot:
        button = pilot.app.query_one("#library-export-submit", Button)
        assert button.disabled is False
        assert (
            pilot.app.query_one("#library-export-submit-reason", Static).display
            is False
        )


def test_the_inline_reason_and_the_tooltip_cannot_drift():
    """Every blocked predicate the tooltip knows, the inline line knows."""
    from tldw_chatbook.Library.library_export_state import export_button_tooltip

    for state in (
        _state(destination=""),
        _state(counts={"media": 0, "conversations": 0, "notes": 0, "prompts": 0}),
        _state(counts=None),
        _state(running=True),
    ):
        assert state.submit_blocked_reason == export_button_tooltip(state)
    assert _state().submit_blocked_reason == ""


# --- task-32362: the Reader's "○ Find" ---------------------------------------


class _ReaderHost(ConsolidatedCSSApp):
    def __init__(self, viewer: LibraryMediaViewer):
        super().__init__()
        self._viewer = viewer

    def compose(self):
        yield self._viewer


def _reader(**kwargs) -> LibraryMediaViewer:
    """A Reader on a real item with content but no analysis yet."""
    return LibraryMediaViewer(
        build_library_media_viewer_state(
            {
                "id": 1,
                "title": "Attention Is All You Need",
                "type": "article",
                "content": "body text",
                "analysis_content": "",
            }
        ),
        id="library-media-viewer",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_the_blocked_find_carries_its_reason_under_the_toolbar():
    viewer = _reader(reader_mode="analysis")
    app = _ReaderHost(viewer)
    async with app.run_test(size=(235, 52)) as pilot:
        find = pilot.app.query_one("#library-media-reader-find", Button)
        reason = pilot.app.query_one("#library-media-reader-find-reason", Static)
        assert find.disabled
        assert str(find.label).startswith("○ ")
        assert str(reason.renderable) == find.tooltip
        # The reason is the toolbar's immediate next sibling and paints
        # below the button. Asserted as DOM adjacency rather than an exact
        # row delta (the Export twin's shape): this host mounts the viewer
        # without its usual container, so the ds-toolbar Horizontal takes
        # more rows here than in the app. Live at 235x52 the reason renders
        # on the row directly under the toolbar -- capture 04.
        toolbar = pilot.app.query_one("#library-media-reader-primary-toolbar")
        siblings = list(toolbar.parent.children)
        assert siblings[siblings.index(toolbar) + 1] is reason
        assert reason.region.y >= find.region.y + find.region.height


@pytest.mark.asyncio
async def test_an_available_find_mounts_no_reason_line():
    app = _ReaderHost(_reader(reader_mode="read"))
    async with app.run_test(size=(235, 52)) as pilot:
        assert pilot.app.query_one("#library-media-reader-find", Button).disabled is False
        assert not pilot.app.query("#library-media-reader-find-reason")
