"""Geometry and interaction contracts for the shared Console bounded body."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)


LOCAL_HINT = "▼ more — scroll"


async def _settle(pilot: Pilot[None]) -> None:
    """Allow post-refresh reconciliation and its guarded layout pass to finish."""

    for _ in range(3):
        await pilot.pause()


def _lines(count: int) -> str:
    return "\n".join(f"line {index}" for index in range(count))


class _Harness(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    ConsoleBoundedSection {
        height: auto;
        min-height: 0;
    }

    .console-bounded-section-viewport {
        height: auto;
        min-height: 0;
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-size: 1 1;
    }

    .console-bounded-section-hint {
        display: none;
        height: 1;
        min-height: 1;
    }
    """

    def __init__(self, section: ConsoleBoundedSection) -> None:
        super().__init__()
        self.section = section

    def compose(self) -> ComposeResult:
        yield self.section


class _OuterScrollHarness(_Harness):
    CSS = (
        _Harness.CSS
        + """
        #outer {
            height: 6;
            overflow-y: auto;
        }
        .outer-filler {
            height: 12;
        }
        """
    )

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="outer"):
            yield self.section
            yield Static("after", classes="outer-filler")


class _FocusHarness(_Harness):
    def compose(self) -> ComposeResult:
        yield self.section
        yield Button("Outside", id="outside")


def _section(
    count: int,
    *,
    allocation: int | None = None,
    on_focus_recovery: Callable[[], None] | None = None,
) -> tuple[ConsoleBoundedSection, Static]:
    content = Static(_lines(count), id="content")
    content.display = count > 0
    return (
        ConsoleBoundedSection(
            content,
            section_id="run",
            allocation=allocation,
            on_focus_recovery=on_focus_recovery,
        ),
        content,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line_count", "viewport_height", "section_height", "has_overflow"),
    [
        (0, 0, 0, False),
        (1, 1, 1, False),
        (20, 20, 20, False),
        (21, 20, 21, True),
    ],
)
async def test_physical_content_line_boundaries(
    line_count: int,
    viewport_height: int,
    section_height: int,
    has_overflow: bool,
) -> None:
    section, _content = _section(line_count)
    app = _Harness(section)

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        viewport = app.query_one(
            "#console-bounded-section-run-viewport", VerticalScroll
        )
        hint = app.query_one("#console-bounded-section-run-hint", Static)

        assert section.id == "console-bounded-section-run"
        assert section.desired_content_lines == line_count
        assert viewport.content_region.height == viewport_height
        assert section.region.height == section_height
        assert (viewport.max_scroll_y > 0) is has_overflow
        assert viewport.can_focus is has_overflow
        assert hint.can_focus is False
        assert hint.display is has_overflow
        assert hint.region.height == int(has_overflow)
        assert str(hint.render()) == (LOCAL_HINT if has_overflow else "")


@pytest.mark.asyncio
async def test_overflow_hint_slot_stays_mounted_and_stable_at_scroll_end() -> None:
    section, _content = _section(21)
    app = _Harness(section)

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        viewport = section.viewport
        hint = section.hint
        assert hint in tuple(section.children)
        slot_region = hint.region
        assert str(hint.render()) == LOCAL_HINT

        viewport.scroll_end(animate=False)
        await pilot.pause()
        assert viewport.scroll_y == viewport.max_scroll_y
        assert hint.display is True
        assert hint.region == slot_region
        assert str(hint.render()) == ""

        viewport.scroll_up(animate=False)
        await pilot.pause()
        assert viewport.scroll_y < viewport.max_scroll_y
        assert hint.region == slot_region
        assert str(hint.render()) == LOCAL_HINT


@pytest.mark.asyncio
async def test_no_overflow_removes_hint_layout_and_viewport_focus_stop() -> None:
    section, _content = _section(20)
    app = _Harness(section)

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        assert section.hint in tuple(section.children)
        assert section.hint.display is False
        assert section.hint.region.height == 0
        assert section.viewport.can_focus is False

        focusable = [
            widget
            for widget in section.query(Widget)
            if widget.can_focus and widget.display
        ]
        assert section.viewport not in focusable


@pytest.mark.asyncio
async def test_zero_allocation_hides_non_empty_body_and_hint() -> None:
    section, _content = _section(21, allocation=0)
    app = _Harness(section)

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        assert section.desired_content_lines == 21
        assert section.viewport.content_region.height == 0
        assert section.region.height == 0
        assert section.viewport.can_focus is False
        assert section.hint.display is False
        assert section.hint.region.height == 0


@pytest.mark.asyncio
async def test_focused_descendant_is_fully_revealed() -> None:
    buttons = [Button(f"Row {index}", id=f"row-{index}") for index in range(12)]
    section = ConsoleBoundedSection(*buttons, section_id="tools", allocation=5)
    app = _Harness(section)

    async with app.run_test(size=(60, 20)) as pilot:
        await _settle(pilot)
        target = app.query_one("#row-11", Button)
        target.focus()
        await pilot.pause()

        assert section.viewport.scroll_y > 0
        assert section.viewport.content_region.contains_region(target.region)


@pytest.mark.asyncio
async def test_native_keyboard_scroll_actions_cover_lines_pages_and_ends() -> None:
    section, _content = _section(40, allocation=5)
    app = _Harness(section)

    async with app.run_test(size=(60, 20)) as pilot:
        await _settle(pilot)
        viewport = section.viewport
        viewport.focus()

        await pilot.press("down")
        await pilot.pause()
        assert viewport.scroll_y == 1

        await pilot.press("pagedown")
        await pilot.pause()
        assert viewport.scroll_y > 1

        await pilot.press("end")
        await pilot.pause()
        assert viewport.scroll_y == viewport.max_scroll_y
        assert str(section.hint.render()) == ""

        await pilot.press("up")
        await pilot.pause()
        assert viewport.scroll_y < viewport.max_scroll_y

        before_page_up = viewport.scroll_y
        await pilot.press("pageup")
        await pilot.pause()
        assert viewport.scroll_y < before_page_up

        await pilot.press("home")
        await pilot.pause()
        assert viewport.scroll_y == 0


@pytest.mark.asyncio
async def test_content_and_allocation_changes_clamp_and_reconcile_state() -> None:
    recovered: list[None] = []
    button = Button("focused", id="focused")
    tail = Static(_lines(29), id="tail")
    section = ConsoleBoundedSection(
        button,
        tail,
        section_id="source-readiness",
        allocation=10,
        on_focus_recovery=lambda: recovered.append(None),
    )
    app = _Harness(section)

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        button.focus()
        await pilot.pause()
        section.viewport.scroll_end(animate=False)
        await pilot.pause()
        assert section.viewport.scroll_y > 0

        section.set_allocation(5)
        await pilot.pause()
        assert section.viewport.content_region.height == 5
        assert section.viewport.scroll_y <= section.viewport.max_scroll_y
        assert section.viewport.can_focus is True
        assert section.hint.display is True

        await button.remove()
        tail.update(_lines(3))
        await section.recompose()
        await _settle(pilot)
        assert recovered == [None]
        assert section.desired_content_lines == 3
        assert section.viewport.scroll_y == 0
        assert section.viewport.can_focus is False
        assert section.hint.display is False

        section.set_allocation(2)
        await pilot.pause()
        assert section.viewport.content_region.height == 2
        assert section.viewport.can_focus is True
        assert section.hint.display is True

        section.set_allocation(8)
        await pilot.pause()
        assert section.viewport.content_region.height == 3
        assert section.viewport.scroll_y == 0
        assert section.viewport.can_focus is False
        assert section.hint.display is False


@pytest.mark.asyncio
async def test_viewport_focus_recovers_once_when_overflow_disappears() -> None:
    recovered: list[None] = []
    content = Static(_lines(21), id="shrinking-content")
    app: _FocusHarness

    def recover_focus() -> None:
        recovered.append(None)
        app.query_one("#outside", Button).focus()

    section = ConsoleBoundedSection(
        content,
        section_id="focus-shrink",
        on_focus_recovery=recover_focus,
    )
    app = _FocusHarness(section)

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        section.viewport.focus()
        await pilot.pause()
        assert app.focused is section.viewport
        assert section.viewport.can_focus is True

        content.update(_lines(3))
        section.request_reconcile()
        await _settle(pilot)
        assert recovered == [None]
        assert section.viewport.can_focus is False
        assert app.focused is app.query_one("#outside", Button)

        section.request_reconcile()
        await _settle(pilot)
        assert recovered == [None]


@pytest.mark.asyncio
async def test_removed_former_descendant_does_not_steal_valid_outside_focus() -> None:
    recovered: list[None] = []
    focused_row = Button("Inside", id="inside")
    section = ConsoleBoundedSection(
        focused_row,
        Static(_lines(20), id="inside-tail"),
        section_id="outside-focus",
        allocation=5,
        on_focus_recovery=lambda: recovered.append(None),
    )
    app = _FocusHarness(section)

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        focused_row.focus()
        await pilot.pause()
        assert app.focused is focused_row

        outside = app.query_one("#outside", Button)
        outside.focus()
        await pilot.pause()
        assert app.focused is outside

        await focused_row.remove()
        await section.recompose()
        await _settle(pilot)
        assert recovered == []
        assert app.focused is outside


def _wheel_down(widget: Widget) -> events.MouseScrollDown:
    return events.MouseScrollDown(widget, 0, 0, 0, 1, 0, False, False, False)


@pytest.mark.asyncio
async def test_pointer_scroll_bubbles_to_outer_owner_at_local_boundaries() -> None:
    section, _content = _section(12, allocation=3)
    app = _OuterScrollHarness(section)

    async with app.run_test(size=(60, 12)) as pilot:
        await _settle(pilot)
        outer = app.query_one("#outer", VerticalScroll)
        viewport = section.viewport

        viewport.scroll_end(animate=False)
        await pilot.pause()
        assert viewport.scroll_y == viewport.max_scroll_y
        assert outer.scroll_y == 0

        viewport.post_message(_wheel_down(viewport))
        await pilot.pause()
        assert viewport.scroll_y == viewport.max_scroll_y
        assert outer.scroll_y > 0


@pytest.mark.asyncio
async def test_non_scrollable_body_does_not_consume_pointer_scroll() -> None:
    section, _content = _section(1)
    app = _OuterScrollHarness(section)

    async with app.run_test(size=(60, 12)) as pilot:
        await _settle(pilot)
        outer = app.query_one("#outer", VerticalScroll)
        section.viewport.post_message(_wheel_down(section.viewport))
        await pilot.pause()
        assert section.viewport.scroll_y == 0
        assert outer.scroll_y > 0


@pytest.mark.asyncio
async def test_in_place_sync_and_same_instance_hide_show_preserve_offset() -> None:
    section, content = _section(30, allocation=5)
    app = _Harness(section)

    async with app.run_test(size=(60, 20)) as pilot:
        await _settle(pilot)
        section.viewport.scroll_to(y=8, animate=False)
        await pilot.pause()
        assert section.viewport.scroll_y == 8

        content.update(_lines(30))
        section.request_reconcile()
        await pilot.pause()
        assert section.viewport.scroll_y == 8

        section.display = False
        section.request_reconcile()
        await _settle(pilot)
        assert section.viewport.scroll_y == 8

        section.display = True
        section.request_reconcile()
        await _settle(pilot)
        assert section.viewport.scroll_y == 8


@pytest.mark.asyncio
async def test_width_shrink_and_growth_remeasure_wrapped_physical_rows() -> None:
    content = Static("word " * 100, id="wrapped")
    section = ConsoleBoundedSection(content, section_id="wrapped")
    app = _Harness(section)

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        wide_demand = section.desired_content_lines
        assert 0 < wide_demand <= 20
        assert section.viewport.can_focus is False

        await pilot.resize_terminal(20, 30)
        section.request_reconcile()
        await _settle(pilot)
        assert section.desired_content_lines > 20
        assert section.viewport.content_region.height == 20
        assert section.viewport.can_focus is True
        assert section.hint.display is True

        section.viewport.scroll_end(animate=False)
        await pilot.pause()
        assert section.viewport.scroll_y > 0

        await pilot.resize_terminal(60, 30)
        section.request_reconcile()
        await _settle(pilot)
        assert section.desired_content_lines == wide_demand
        assert section.viewport.scroll_y == 0
        assert section.viewport.can_focus is False
        assert section.hint.display is False


@pytest.mark.asyncio
async def test_same_instance_recompose_retains_content_and_preserves_offset() -> None:
    section, content = _section(30, allocation=5)
    app = _Harness(section)

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        section.viewport.scroll_to(y=8, animate=False)
        await pilot.pause()
        assert section.viewport.scroll_y == 8

        root_id = section.id
        viewport_id = section.viewport.id
        hint_id = section.hint.id
        await section.recompose()
        await _settle(pilot)

        assert section.id == root_id
        assert section.viewport.id == viewport_id
        assert section.hint.id == hint_id
        assert app.query_one("#content", Static) is content
        assert len(app.query("#content")) == 1
        assert section.desired_content_lines == 30
        assert section.viewport.content_region.height == 5
        assert section.viewport.scroll_y == 8
        assert section.viewport.can_focus is True
        assert section.hint.display is True
        assert str(section.hint.render()) == LOCAL_HINT

        content.update(_lines(3))
        await section.recompose()
        await _settle(pilot)
        assert app.query_one("#content", Static) is content
        assert len(app.query("#content")) == 1
        assert section.desired_content_lines == 3
        assert section.viewport.content_region.height == 3
        assert section.viewport.scroll_y == 0
        assert section.viewport.can_focus is False
        assert section.hint.display is False

        section.set_allocation(2)
        await section.recompose()
        await _settle(pilot)
        assert section.desired_content_lines == 3
        assert section.viewport.content_region.height == 2
        assert section.viewport.scroll_y == 0
        assert section.viewport.can_focus is True
        assert section.hint.display is True

        section.set_allocation(8)
        await section.recompose()
        await _settle(pilot)
        assert section.viewport.content_region.height == 3
        assert section.viewport.can_focus is False
        assert section.hint.display is False


class _CountingSection(ConsoleBoundedSection):
    reconcile_passes = 0

    def _reconcile(self) -> None:
        self.reconcile_passes += 1
        super()._reconcile()


@pytest.mark.asyncio
async def test_reconcile_requests_coalesce_and_equality_guards_do_not_loop() -> None:
    section = _CountingSection(Static(_lines(21)), section_id="approvals")
    app = _Harness(section)

    async with app.run_test(size=(60, 30)) as pilot:
        await _settle(pilot)
        section.reconcile_passes = 0

        section.request_reconcile()
        section.request_reconcile()
        section.request_reconcile()
        await pilot.pause()
        assert section.reconcile_passes == 1

        await pilot.pause()
        await pilot.pause()
        assert section.reconcile_passes == 1
