"""Shared destination rail widgets: the Chat-free base behind ConsoleRailHandle."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

import tldw_chatbook
from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_RAIL_CONTEXT_LABEL,
    CONSOLE_RAIL_INSPECTOR_LABEL,
)
from tldw_chatbook.UI.Console_Modules.frame import frame_console_region
from tldw_chatbook.Widgets.Console.console_rail_handle import ConsoleRailHandle
from tldw_chatbook.Widgets.destination_rail import (
    RAIL_SECTION_TOGGLE_PREFIX,
    DestinationRailHandle,
    DestinationRailSectionHeader,
)


class _SectionHeaderHarness(App[None]):
    """Minimal host: one section header, over a body Static the header's
    own owning screen would normally show/hide -- mirrors how every real
    consumer (Console/Home/Library rails) wires the toggle Button's
    ``Pressed`` message to its own open/closed state."""

    def __init__(self) -> None:
        super().__init__()
        self.pressed_ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield DestinationRailSectionHeader(
            "Details", section_id="lab-details", open=True
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id:
            self.pressed_ids.append(event.button.id)


class _HandleHarness(App[None]):
    def __init__(self, handle: DestinationRailHandle) -> None:
        super().__init__()
        self._handle = handle

    def compose(self) -> ComposeResult:
        yield self._handle


class _StyledHandleHarness(_HandleHarness):
    """Handle harness using the same generated stylesheet as production."""

    CSS_PATH = str(
        Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
    )


def _assert_region_contains(container, child) -> None:
    assert container.x <= child.x
    assert container.y <= child.y
    assert child.right <= container.right
    assert child.bottom <= container.bottom


def _assert_solid_frame(handle: DestinationRailHandle) -> None:
    border = handle.styles.border
    assert border.top[0] == "solid"
    assert border.right[0] == "solid"
    assert border.bottom[0] == "solid"
    assert border.left[0] == "solid"


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


@pytest.mark.asyncio
async def test_clicking_the_section_title_posts_the_same_pressed_message_as_the_toggle():
    """task-2859 item 5: only the ``▸`` chip used to respond to a click --
    clicking the "Details" LABEL itself did nothing. The title now presses
    the toggle Button on click, posting the identical ``Button.Pressed``
    the owning screen's handler already expects (no new message type, no
    new wiring needed at any of the three consumers)."""
    app = _SectionHeaderHarness()
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        title = app.query_one("#console-rail-section-title-lab-details", Static)
        await pilot.click(title)
        await pilot.pause()

        assert app.pressed_ids == [f"{RAIL_SECTION_TOGGLE_PREFIX}lab-details"]


@pytest.mark.asyncio
async def test_clicking_the_toggle_chip_itself_does_not_double_fire():
    """The title's click handler must not ALSO fire when the toggle chip
    itself is clicked directly -- Button already stops its own Click event
    (``Button._on_click`` calls ``event.stop()``), so this pins that the
    header's own handler never sees it and presses the toggle a second
    time."""
    app = _SectionHeaderHarness()
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        toggle = app.query_one(f"#{RAIL_SECTION_TOGGLE_PREFIX}lab-details", Button)
        await pilot.click(toggle)
        await pilot.pause()

        assert app.pressed_ids == [f"{RAIL_SECTION_TOGGLE_PREFIX}lab-details"]


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
async def test_console_handles_share_full_height_solid_frame_with_real_css():
    measured: dict[str, tuple[int, int, int, object, object]] = {}

    for side in ("left", "right"):
        handle = _console_handle(side=side)
        app = _StyledHandleHarness(frame_console_region(handle))
        async with app.run_test(size=(40, 20)) as pilot:
            await pilot.pause()
            button = app.query_one("#console-rail-open", Button)

            assert handle.region.height == 20
            assert handle.styles.background.a > 0
            _assert_solid_frame(handle)
            assert button.styles.content_align_horizontal == "center"
            assert button.styles.content_align_vertical == "middle"
            _assert_region_contains(handle.content_region, button.region)
            if side == "right":
                assert handle.has_class("console-inspector-rail-handle")
            measured[side] = (
                handle.region.height,
                handle.region.width,
                handle.content_region.width,
                handle.styles.background,
                handle.styles.border,
            )

    left = measured["left"]
    right = measured["right"]
    assert left[0] == right[0] == 20
    assert left[1] == 13
    assert right[1] == 11
    assert right[2] == 9
    assert right[3] == left[3]
    assert right[4] == left[4]


@pytest.mark.asyncio
async def test_unbadged_console_inspector_button_fills_framed_content_height():
    handle = _console_handle(side="right")
    app = _StyledHandleHarness(frame_console_region(handle))

    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        button = app.query_one("#console-rail-open", Button)

        assert not app.query("#console-rail-badge")
        assert button.region.height == handle.content_region.height
        _assert_region_contains(handle.content_region, button.region)


@pytest.mark.asyncio
async def test_badged_console_inspector_reserves_exact_contained_badge_row():
    handle = _console_handle(side="right", badge="3 approvals")
    app = _StyledHandleHarness(frame_console_region(handle))

    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        button = app.query_one("#console-rail-open", Button)
        badge = app.query_one("#console-rail-badge", Static)

        assert str(badge.renderable) == "3 appr"
        assert button.region.bottom <= badge.region.y
        assert (
            button.region.height == handle.content_region.height - badge.region.height
        )
        _assert_region_contains(handle.content_region, button.region)
        _assert_region_contains(handle.content_region, badge.region)


@pytest.mark.asyncio
async def test_shared_right_destination_handle_keeps_compact_transparent_chrome():
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="right",
    )
    app = _StyledHandleHarness(handle)

    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        border = handle.styles.border

        assert handle.region.height <= 6
        assert handle.styles.background.a == 0
        assert border.top[0] in {"", "none"}
        assert border.right[0] in {"", "none"}
        assert border.bottom[0] in {"", "none"}
        assert border.left[0] in {"", "none"}


@pytest.mark.asyncio
async def test_console_handle_is_a_destination_rail_handle():
    assert issubclass(ConsoleRailHandle, DestinationRailHandle)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side", "expected"),
    [
        ("left", "Open Context rail"),
        ("right", "Open Inspector rail"),
    ],
)
async def test_console_handle_keeps_its_fixed_tooltips(side, expected):
    """Console's tooltips are fixed strings, not derived from the label."""
    app = _HandleHarness(_console_handle(side=side, label="Anything"))
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert app.query_one("#console-rail-open", Button).tooltip == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("badge", "expected"),
    [
        ("1 approval", "1 appr"),
        ("3 approvals", "3 appr"),
        ("artifact", "art"),
        ("something else", "something else"),
    ],
)
async def test_console_handle_abbreviates_right_side_badges(badge, expected):
    app = _HandleHarness(_console_handle(side="right", badge=badge))
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#console-rail-badge", Static).renderable) == expected


@pytest.mark.asyncio
async def test_console_handle_uses_inward_inspector_label_on_the_right() -> None:
    handle = _console_handle(side="right", label=CONSOLE_RAIL_INSPECTOR_LABEL)
    app = _StyledHandleHarness(frame_console_region(handle))
    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        button = app.query_one("#console-rail-open", Button)

        assert str(button.label) == "<-Inspect"
        assert button.tooltip == "Open Inspector rail"
        assert handle.region.width == 11
        assert handle.content_region.width == 9
        assert handle.content_region.contains_region(button.region)


@pytest.mark.asyncio
async def test_console_handle_uses_inward_context_label_on_the_left() -> None:
    handle = _console_handle(side="left", label=CONSOLE_RAIL_CONTEXT_LABEL)
    app = _StyledHandleHarness(frame_console_region(handle))

    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        button = app.query_one("#console-rail-open", Button)

        assert str(button.label) == "Context->"
        assert button.tooltip == "Open Context rail"
        assert handle.region.width == 13
        assert handle.content_region.width == 11
        assert handle.content_region.contains_region(button.region)


@pytest.mark.asyncio
async def test_console_handle_leaves_left_side_text_alone():
    app = _HandleHarness(_console_handle(side="left", badge="1 approval"))
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert (
            str(app.query_one("#console-rail-badge", Static).renderable) == "1 approval"
        )


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
