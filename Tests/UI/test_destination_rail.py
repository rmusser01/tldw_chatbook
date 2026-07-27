"""Shared destination rail widgets: the Chat-free base behind ConsoleRailHandle."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_rail_state import CONSOLE_RAIL_INSPECTOR_LABEL
from tldw_chatbook.Widgets.Console.console_rail_handle import ConsoleRailHandle
from tldw_chatbook.Widgets.destination_rail import (
    RAIL_SECTION_TOGGLE_PREFIX,
    DestinationRailHandle,
    DestinationRailSectionHeader,
)


class _HandleHarness(App[None]):
    def __init__(self, handle: DestinationRailHandle) -> None:
        super().__init__()
        self._handle = handle

    def compose(self) -> ComposeResult:
        yield self._handle


@pytest.mark.asyncio
async def test_base_handle_renders_label_and_badge_verbatim():
    """The base applies no vocabulary of its own -- Console's lives in its subclass."""
    handle = DestinationRailHandle(
        label="Catalog",
        badge="3 servers",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#lab-rail-open", Button).label) == "Catalog"
        assert str(app.query_one("#lab-rail-badge", Static).renderable) == "3 servers"


@pytest.mark.asyncio
async def test_base_handle_default_tooltip_names_the_rail():
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail-open", Button).tooltip == "Open Catalog rail"


@pytest.mark.asyncio
async def test_base_handle_accepts_an_explicit_tooltip():
    handle = DestinationRailHandle(
        label="Whatever",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="right",
        open_tooltip="Open Inspector rail",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail-open", Button).tooltip == "Open Inspector rail"


@pytest.mark.asyncio
async def test_base_handle_keeps_the_existing_css_class_names():
    """Class names are deliberately unchanged so the CSS bundle sees no diff."""
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert "console-rail-handle" in handle.classes
        assert "console-rail-handle-left" in handle.classes


@pytest.mark.asyncio
async def test_base_handle_omits_the_badge_when_empty():
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert not app.query("#lab-rail-badge")


def test_console_reexports_the_shared_disclosure_glyphs():
    """One definition, re-exported -- not two copies kept in step.

    These used to be declared in both modules with an equality assertion
    here holding them together: a lockstep between two files enforced from
    a third, invisible to anything static. ADR-034 gave ownership to
    `destination_rail`, which is what renders them, and made
    `console_glyphs` re-export.

    Asserting identity rather than equality is the point: `is` can only
    hold if there is a single definition, so this fails if anyone
    re-declares the literal instead of importing it. Equality would pass
    for a re-introduced duplicate.
    """
    from tldw_chatbook.Chat import console_glyphs
    from tldw_chatbook.Widgets.destination_rail import (
        GLYPH_COLLAPSED,
        GLYPH_EXPANDED,
    )

    assert console_glyphs.GLYPH_EXPANDED is GLYPH_EXPANDED
    assert console_glyphs.GLYPH_COLLAPSED is GLYPH_COLLAPSED


def _console_handle(**overrides) -> ConsoleRailHandle:
    kwargs = dict(
        label="Context",
        badge="",
        button_id="console-rail-open",
        badge_id="console-rail-badge",
        side="left",
    )
    kwargs.update(overrides)
    return ConsoleRailHandle(**kwargs)


@pytest.mark.asyncio
async def test_console_handle_is_a_destination_rail_handle():
    assert issubclass(ConsoleRailHandle, DestinationRailHandle)


@pytest.mark.asyncio
@pytest.mark.parametrize(("side", "expected"), [
    ("left", "Open Context rail"),
    ("right", "Open Inspector rail"),
])
async def test_console_handle_keeps_its_fixed_tooltips(side, expected):
    """Console's tooltips are fixed strings, not derived from the label."""
    app = _HandleHarness(_console_handle(side=side, label="Anything"))
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert app.query_one("#console-rail-open", Button).tooltip == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("badge", "expected"), [
    ("1 approval", "1 appr"),
    ("3 approvals", "3 appr"),
    ("artifact", "art"),
    ("something else", "something else"),
])
async def test_console_handle_abbreviates_right_side_badges(badge, expected):
    app = _HandleHarness(_console_handle(side="right", badge=badge))
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#console-rail-badge", Static).renderable) == expected


@pytest.mark.asyncio
async def test_console_handle_renames_the_inspector_label_on_the_right():
    app = _HandleHarness(
        _console_handle(side="right", label=CONSOLE_RAIL_INSPECTOR_LABEL)
    )
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#console-rail-open", Button).label) == "Inspector"


@pytest.mark.asyncio
async def test_console_handle_leaves_left_side_text_alone():
    app = _HandleHarness(_console_handle(side="left", badge="1 approval"))
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#console-rail-badge", Static).renderable) == "1 approval"


@pytest.mark.asyncio
async def test_derived_tooltip_follows_a_label_change_from_sync_state():
    """A tooltip derived from the label must not go stale when the label changes.

    ``sync_state`` updates ``label`` and recomposes, so a tooltip captured
    once at construction would keep naming the old rail. Console is immune
    because it supplies fixed tooltip strings, but any destination relying
    on the derived default would show the wrong rail name after a sync.
    """
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail-open", Button).tooltip == "Open Catalog rail"

        handle.sync_state("Sources", "")
        await pilot.pause()
        await pilot.pause()

        assert app.query_one("#lab-rail-open", Button).label.__str__() == "Sources"
        assert app.query_one("#lab-rail-open", Button).tooltip == "Open Sources rail"


@pytest.mark.asyncio
async def test_explicit_tooltip_is_not_rewritten_by_sync_state():
    """An explicitly supplied tooltip stays fixed -- that is Console's contract."""
    handle = DestinationRailHandle(
        label="Context",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
        open_tooltip="Open Context rail",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        handle.sync_state("Something Else", "")
        await pilot.pause()
        await pilot.pause()

        assert app.query_one("#lab-rail-open", Button).tooltip == "Open Context rail"
