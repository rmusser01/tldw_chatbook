"""Focused Task-7 behavior for Inspector outer scrolling and navigation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual import events
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleStagedContextState,
)
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector
from tldw_chatbook.Widgets.Console.console_staged_context import (
    ConsoleStagedContextTray,
)

from Tests.UI.test_console_right_rail import (
    _wait_for_right_rail_condition,
    make_console_pilot,
)


INSPECTOR_OUTER_HINT = "▼ more sections — scroll"
INSPECTOR_OUTER_HINT_ID = "console-inspector-outer-scroll-hint"
STAGED_ONE_ROW_TEST_CSS = """
#console-right-rail {
    height: 19;
    min-height: 19;
    max-height: 19;
}
#console-inspector-rail-body {
    height: 1fr;
    min-height: 0;
}
ConsoleBoundedSection {
    height: auto;
    min-height: 0;
}
.console-staged-context-header {
    height: auto;
    min-height: 1;
}
.console-staged-source-row {
    height: auto;
    min-height: 1;
    max-height: 1;
    margin: 0;
}
"""
STAGED_OVERFLOW_TEST_CSS = """
#console-right-rail {
    height: 100%;
    min-height: 20;
}
#console-inspector-rail-body {
    height: 1fr;
    min-height: 0;
}
ConsoleBoundedSection {
    height: auto;
    min-height: 0;
}
.console-staged-context-header {
    height: auto;
    min-height: 1;
}
.console-staged-source-row {
    height: 2;
    min-height: 2;
    max-height: 2;
    margin: 0;
}
"""
RUN_OWNER_TEST_CSS = """
#console-right-rail {
    height: 100%;
    min-height: 20;
}
#console-inspector-rail-body {
    height: 1fr;
    min-height: 0;
}
#console-run-inspector,
#console-run-inspector-state {
    height: auto;
    min-height: 0;
}
.console-bounded-section-viewport {
    height: auto;
    min-height: 0;
}
ConsoleBoundedSection {
    height: auto;
    min-height: 0;
}
"""


def _inside(widget, owner) -> bool:
    return widget is owner or owner in widget.ancestors


async def _open_inspector(pilot):
    if not pilot.app.screen.query_one("#console-right-rail").display:
        await pilot.click("#console-inspector-rail-open")
    await pilot.pause()
    pilot.app.screen._stop_console_transcript_sync_timer()
    await pilot.pause()
    return pilot.app.screen.query_one("#console-right-rail")


async def _overflow(section: ConsoleBoundedSection, rows: int = 21) -> None:
    await section.viewport.remove_children()
    await section.viewport.mount(Static("\n".join(f"row {row}" for row in range(rows))))
    section.request_reconcile()


def _wheel_down(widget: Widget) -> events.MouseScrollDown:
    return events.MouseScrollDown(widget, 0, 0, 0, 1, 0, False, False, False)


def _fully_inside_outer(header: Widget, outer: Widget) -> bool:
    return (
        header.region.y >= outer.content_region.y
        and header.region.bottom <= outer.content_region.bottom
    )


def _external_boundary_header(section: ConsoleBoundedSection) -> Widget:
    siblings = list(section.parent.children)
    return siblings[siblings.index(section) - 1]


def _assert_recovery_references_are_current(rail) -> None:
    """Assert recovery state contains mounted controls from current boundaries only."""

    current_sections = {
        section.section_id: section for section, _header in rail._mounted_boundaries()
    }
    for section_id, (target, controls) in rail._section_focus_history.items():
        section = current_sections[section_id]
        assert target.is_attached
        assert target.is_mounted
        assert target is section.viewport or section.viewport in target.ancestors
        assert controls
        assert all(control.is_attached and control.is_mounted for control in controls)
        assert all(section.viewport in control.ancestors for control in controls)
    assert not rail._pending_focus_recoveries


def _staged_state(row_count: int) -> ConsoleStagedContextState:
    return ConsoleStagedContextState(
        heading="Sources",
        summary="",
        rows=tuple(
            ConsoleDisplayRow(f"Source {index}", f"value {index}")
            for index in range(row_count)
        ),
        source_count=row_count,
    )


@pytest.mark.asyncio
async def test_outer_hint_is_pinned_last_child_with_exact_nonfocusable_copy():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        header, project, summary, body, hint = rail.children

        assert header.id == "console-inspector-rail-header"
        assert project.id == "console-project-instruction-status"
        assert summary.id == "console-send-authority-summary"
        assert body.id == "console-inspector-rail-body"
        assert hint.id == INSPECTOR_OUTER_HINT_ID
        assert str(hint.renderable) in ("", INSPECTOR_OUTER_HINT)
        assert hint.can_focus is False


@pytest.mark.asyncio
async def test_terminal_resize_drives_counterfactual_hint_without_feedback_loop():
    async with make_console_pilot(size=(160, 80)) as pilot:
        rail = await _open_inspector(pilot)
        body = rail.query_one("#console-inspector-rail-body")
        hint = rail.query_one(f"#{INSPECTOR_OUTER_HINT_ID}", Static)
        await body.remove_children()
        child = Static("fixed demand", id="outer-fixed-child")
        child.styles.height = 30
        await body.mount(child)
        await pilot.resize_terminal(160, 79)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                not rail._outer_reconcile_scheduled
                and hint.display is False
                and body.content_region.height >= 30
            ),
            description="fixed demand fitting counterfactual viewport",
        )
        fitting_terminal_height = pilot.app.size.height - (
            body.content_region.height - 30
        )
        await pilot.resize_terminal(160, fitting_terminal_height)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                not rail._outer_reconcile_scheduled
                and hint.display is False
                and body.content_region.height == 30
            ),
            description="fixed demand exactly fitting counterfactual viewport",
        )
        settled_count = rail._outer_owner_reconcile_count
        await pilot.pause()
        await pilot.pause()
        assert rail._outer_owner_reconcile_count == settled_count

        await pilot.resize_terminal(160, fitting_terminal_height - 1)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                not rail._outer_reconcile_scheduled
                and hint.display is True
                and body.content_region.height == 28
            ),
            description="terminal shrink reserving one outer hint row",
        )
        assert str(hint.renderable) == INSPECTOR_OUTER_HINT
        assert hint.region.height == 1

        body.scroll_end(animate=False, immediate=True)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: body.scroll_y == body.max_scroll_y and str(hint.renderable) == "",
            description="outer hint blanking at the terminal-resize scroll end",
        )

        await pilot.resize_terminal(160, fitting_terminal_height)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                not rail._outer_reconcile_scheduled
                and hint.display is False
                and body.content_region.height >= 30
            ),
            description="terminal growth removing the counterfactual slot",
        )
        assert body.scroll_y == 0


@pytest.mark.asyncio
async def test_virtual_size_only_body_update_invalidates_owner_and_settles_cue(
    monkeypatch,
):
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        body = rail.query_one("#console-inspector-rail-body")
        hint = rail.query_one(f"#{INSPECTOR_OUTER_HINT_ID}", Static)
        await body.remove_children()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                body.content_region.height > 0 and not rail._outer_reconcile_scheduled
            ),
            description="empty outer owner geometry",
        )

        virtual_only_updates = []
        owner_invalidations = []
        original_size_updated = body._size_updated
        original_owner_callback = body._on_geometry_changed

        def observe_size_updated(size, virtual_size, container_size, layout=True):
            previous_size = body.size
            previous_virtual_size = body.virtual_size
            updated = original_size_updated(
                size,
                virtual_size,
                container_size,
                layout,
            )
            if previous_size == size and previous_virtual_size != virtual_size:
                virtual_only_updates.append((previous_virtual_size, virtual_size))
            return updated

        def observe_owner_invalidation():
            owner_invalidations.append(body.virtual_size)
            original_owner_callback()

        monkeypatch.setattr(body, "_size_updated", observe_size_updated)
        monkeypatch.setattr(body, "_on_geometry_changed", observe_owner_invalidation)
        child = Static("virtual owner", id="outer-virtual-size-child")
        child.styles.height = body.content_region.height + 2
        await body.mount(child)

        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                bool(virtual_only_updates)
                and bool(owner_invalidations)
                and hint.display is True
                and not rail._outer_reconcile_scheduled
            ),
            description="virtual-size invalidation settling the outer cue",
        )
        assert body.virtual_size.height > body.content_region.height


@pytest.mark.asyncio
async def test_outer_hint_slot_stays_but_copy_blanks_at_scroll_end():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        hint = rail.query_one(f"#{INSPECTOR_OUTER_HINT_ID}", Static)
        body = rail.query_one("#console-inspector-rail-body")

        await _wait_for_right_rail_condition(
            pilot,
            lambda: hint.display and body.max_scroll_y > 0,
            description="overflowing Inspector body",
        )
        assert str(hint.renderable) == INSPECTOR_OUTER_HINT

        body.scroll_end(animate=False, immediate=True)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: body.scroll_y == body.max_scroll_y and str(hint.renderable) == "",
            description="blank outer hint copy at scroll end",
        )
        assert hint.display is True
        assert hint.region.height == 1


@pytest.mark.asyncio
async def test_staged_owner_sync_drives_ten_eleven_ten_cue_and_clamp():
    async with make_console_pilot(size=(160, 80), css=STAGED_ONE_ROW_TEST_CSS) as pilot:
        rail = await _open_inspector(pilot)
        outer = rail.query_one("#console-inspector-rail-body")
        hint = rail.query_one(f"#{INSPECTOR_OUTER_HINT_ID}", Static)
        tray = rail.query_one("#console-staged-context-tray", ConsoleStagedContextTray)
        for child in tuple(outer.children):
            if child is not tray:
                await child.remove()

        tray.sync_state(_staged_state(10))
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                tray.query_one(
                    "#console-bounded-section-sources", ConsoleBoundedSection
                ).desired_content_lines
                == 10
            ),
            description="ten-source local owner reconciliation",
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                tray.region.height == outer.content_region.height
                and not rail._outer_reconcile_scheduled
                and hint.display is False
            ),
            description="ten-source counterfactual fit",
        )
        ten_row_demand = tray.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        ).desired_content_lines
        assert tuple(child.region.height for child in rail.children) == (1, 1, 6, 11, 0)

        tray.sync_state(_staged_state(11))
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                tray.query_one(
                    "#console-bounded-section-sources", ConsoleBoundedSection
                ).desired_content_lines
                > ten_row_demand
            ),
            description="eleven-source local owner reconciliation",
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: hint.display is True and not rail._outer_reconcile_scheduled,
            description="eleven-source outer owner overflow after local",
        )
        overflow_count = rail._outer_owner_reconcile_count
        await pilot.pause()
        assert hint.display is True
        assert outer.virtual_size.height > outer.content_region.height
        assert tuple(child.region.height for child in rail.children) == (1, 1, 6, 10, 1)
        assert rail._outer_owner_reconcile_count == overflow_count
        outer.scroll_end(animate=False, immediate=True)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: outer.scroll_y == outer.max_scroll_y,
            description="outer offset before owner shrink",
        )

        tray.sync_state(_staged_state(10))
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                tray.query_one(
                    "#console-bounded-section-sources", ConsoleBoundedSection
                ).desired_content_lines
                == ten_row_demand
                and hint.display is False
                and outer.scroll_y == 0
                and not rail._outer_reconcile_scheduled
            ),
            description="ten-source owner shrink removing slot and clamping",
        )
        assert tuple(child.region.height for child in rail.children) == (1, 1, 6, 11, 0)


@pytest.mark.asyncio
async def test_scroll_owner_cue_preserves_bold_and_clears_without_stale_underline():
    css = """
    .console-rail-section-title { text-style: bold; }
    .console-rail-collapse-button:focus { text-style: bold underline; }
    """
    async with make_console_pilot(size=(160, 45), css=css) as pilot:
        rail = await _open_inspector(pilot)
        outer = rail.query_one("#console-inspector-rail-body")
        collapse = rail.query_one("#console-inspector-rail-collapse", Button)
        sources = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        source_title = rail.query_one("#console-staged-context-title", Static)
        await _overflow(sources)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: sources.viewport.can_focus,
            description="overflowing source owner for focus styling",
        )

        assert source_title.get_visual_style().bold
        assert not source_title.get_visual_style().underline
        sources.viewport.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: source_title.get_visual_style().underline,
            description="active local header underline",
        )
        assert source_title.get_visual_style().bold

        collapse.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                collapse.get_visual_style().bold
                and collapse.get_visual_style().underline
                and not source_title.get_visual_style().underline
            ),
            description="completed collapse focus repaint",
        )
        assert not source_title.get_visual_style().underline

        outer.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: collapse.get_visual_style().underline,
            description="outer owner title underline",
        )
        outside = pilot.app.screen.query_one("#console-native-composer")
        outside.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: not collapse.get_visual_style().underline,
            description="declarative collapse style restored on focus leave",
        )
        assert not source_title.get_visual_style().underline


@pytest.mark.asyncio
async def test_n_and_p_are_rail_local_no_wrap_and_editable_input_keeps_printable_key():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        body = rail.query_one("#console-inspector-rail-body")
        sources = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await _overflow(sources)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: sources.viewport.can_focus,
            description="overflowing Sources viewport",
        )

        body.can_focus = True
        body.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is body,
            description="focused Inspector outer body",
        )
        collapse = rail.query_one("#console-inspector-rail-collapse", Button)
        assert "underline" in str(collapse.styles.text_style)
        await pilot.press("n")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is sources.viewport,
            description="next navigation to Sources",
        )
        external_header = sources.parent.children[0]
        assert "underline" in str(external_header.styles.text_style)

        await pilot.press("p")
        await pilot.pause()
        assert pilot.app.focused is sources.viewport

        editor = Input(id="inspector-editable")
        await sources.viewport.mount(editor)
        sources.request_reconcile()
        editor.focus()
        await pilot.press("n")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: editor.value == "n",
            description="editable Inspector input receiving n",
        )
        assert pilot.app.focused is editor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,boundary_selector", (("p", "sources"), ("n", "live-work"))
)
async def test_no_wrap_navigation_consumes_bubbled_key_before_screen_barge_in(
    key: str,
    boundary_selector: str,
):
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        section = rail.query_one(
            f"#console-bounded-section-{boundary_selector}", ConsoleBoundedSection
        )
        await _overflow(section)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: section.viewport.can_focus,
            description=f"overflowing {boundary_selector} no-wrap anchor",
        )
        section.viewport.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is section.viewport,
            description=f"focused {boundary_selector} no-wrap anchor",
        )

        barge_keys: list[str] = []
        pilot.app.screen._console_hands_free = SimpleNamespace(
            tick_timer=None,
            controller=SimpleNamespace(
                on_composer_key=lambda: barge_keys.append(key),
            ),
        )
        bubbled_key = events.Key(key, key)
        assert section.viewport.post_message(bubbled_key)
        await pilot.pause()
        await pilot.pause()

        assert pilot.app.focused is section.viewport
        assert barge_keys == []
        assert bubbled_key._stop_propagation is True
        assert bubbled_key._no_default_action is True


@pytest.mark.asyncio
async def test_navigation_keys_remain_unconsumed_outside_inspector_and_in_editable_input():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        section = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        editor = Input(id="inspector-navigation-consumption-editor")
        await section.viewport.mount(editor)
        editor.focus()

        editable_key = events.Key("n", "n")
        rail.on_key(editable_key)
        assert editable_key._stop_propagation is False
        assert editable_key._no_default_action is False

        outside = pilot.app.screen.query_one("#console-native-composer")
        outside.focus()
        outside_key = events.Key("p", "p")
        rail.on_key(outside_key)
        assert outside_key._stop_propagation is False
        assert outside_key._no_default_action is False


@pytest.mark.asyncio
async def test_navigation_from_scope_uses_first_following_boundary():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        sources = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await _overflow(sources)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: sources.viewport.can_focus,
            description="overflowing preceding Sources boundary",
        )
        scope = rail.query_one("#console-retrieval-scope-row")
        scope.can_focus = True
        scope.focus()
        await pilot.press("p")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is sources.viewport,
            description="Scope navigating to the preceding Sources boundary",
        )

        scope.focus()
        await pilot.press("n")

        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is rail.query_one("#console-inspector-rail-body"),
            description="Scope navigating exactly to outer body for inert Run",
        )

        assert not list(rail.query("#console-inspector-run-status-summary"))
        assert rail.query_one("#console-inspector-run-recipe")


@pytest.mark.asyncio
async def test_navigation_focuses_overflow_viewport_contains_header_and_preserves_state():
    async with make_console_pilot(
        size=(160, 30), css=STAGED_OVERFLOW_TEST_CSS
    ) as pilot:
        rail = await _open_inspector(pilot)
        outer = rail.query_one("#console-inspector-rail-body")
        tray = rail.query_one("#console-staged-context-tray", ConsoleStagedContextTray)
        tray.sync_state(_staged_state(12))
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                bool(
                    list(
                        tray.query(
                            "#console-bounded-section-sources",
                        )
                    )
                )
                and tray.query_one(
                    "#console-bounded-section-sources", ConsoleBoundedSection
                ).viewport.can_focus
            ),
            description="real staged-source owner overflow",
        )
        sources = tray.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        sources.viewport.scroll_to(y=2, animate=False, immediate=True)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: sources.viewport.scroll_y == 2,
            description="target local offset before navigation",
        )
        tray_display = tray.display
        section_display = sources.display

        outer.focus()
        await pilot.press("n")
        header = sources.parent.children[0]
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is sources.viewport,
            description="overflowing target focuses its viewport exactly",
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: _fully_inside_outer(header, outer),
            description="navigated external header fully contained",
        )
        assert sources.viewport.scroll_y == 2
        assert tray.display is tray_display
        assert sources.display is section_display


@pytest.mark.asyncio
async def test_nested_selected_boundary_uses_actual_sibling_header_coordinates():
    async with make_console_pilot(size=(160, 30)) as pilot:
        rail = await _open_inspector(pilot)
        outer = rail.query_one("#console-inspector-rail-body")
        selected = rail.query_one(
            "#console-bounded-section-selected-conversation",
            ConsoleBoundedSection,
        )
        run_wrapper = selected.parent.parent
        offset_spacer = Static("", id="nested-run-offset-spacer")
        offset_spacer.styles.height = 20
        await outer.mount(offset_spacer, before=run_wrapper)
        selected.set_allocation(1)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: selected.viewport.can_focus,
            description="nested Selected boundary overflow",
        )
        boundaries = rail._mounted_boundaries()
        selected_index = next(
            index
            for index, (section, _header) in enumerate(boundaries)
            if section is selected
        )
        actual_header = _external_boundary_header(selected)
        assert actual_header is boundaries[selected_index][1]
        assert actual_header is not selected.parent.children[0]

        previous_header = boundaries[selected_index - 1][1]
        previous_header.can_focus = True
        previous_header.focus()
        await pilot.press("n")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is selected.viewport,
            description="nested Selected target focuses its viewport exactly",
        )
        await pilot.pause()
        await pilot.pause()

        assert pilot.app.focused is selected.viewport
        assert _fully_inside_outer(actual_header, outer)


@pytest.mark.asyncio
async def test_newer_navigation_prevents_stale_delayed_header_reveal(
    monkeypatch: pytest.MonkeyPatch,
):
    async with make_console_pilot(size=(160, 30)) as pilot:
        rail = await _open_inspector(pilot)
        outer = rail.query_one("#console-inspector-rail-body")
        selected = rail.query_one(
            "#console-bounded-section-selected-conversation",
            ConsoleBoundedSection,
        )
        boundaries = rail._mounted_boundaries()
        selected_index = next(
            index
            for index, (section, _header) in enumerate(boundaries)
            if section is selected
        )
        successor = boundaries[selected_index + 1][0]
        run_wrapper = selected.parent.parent
        offset_spacer = Static("", id="stale-run-offset-spacer")
        offset_spacer.styles.height = 20
        await outer.mount(offset_spacer, before=run_wrapper)
        successor_header = _external_boundary_header(successor)
        boundary_gap = Static("", id="stale-boundary-gap")
        boundary_gap.styles.height = 8
        await successor.parent.mount(boundary_gap, before=successor_header)
        selected.set_allocation(1)
        await _overflow(successor)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: selected.viewport.can_focus and successor.viewport.can_focus,
            description="two overflowing targets for stale reveal guard",
        )

        previous_header = boundaries[selected_index - 1][1]
        previous_header.can_focus = True
        previous_header.focus()
        await pilot.pause()

        held_reveals = []
        original_call_after_refresh = rail.call_after_refresh

        def hold_boundary_reveal(callback, *args, **kwargs):
            if getattr(callback, "__name__", None) == "_reveal_boundary_header":
                held_reveals.append((callback, args, kwargs))
                return True
            return original_call_after_refresh(callback, *args, **kwargs)

        monkeypatch.setattr(rail, "call_after_refresh", hold_boundary_reveal)
        starting_generation = rail._navigation_generation
        assert rail.post_message(events.Key("n", "n"))
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                rail._navigation_generation == starting_generation + 1
                and pilot.app.focused is selected.viewport
                and len(held_reveals) == 1
            ),
            description="first navigation held before its reveal drains",
        )
        assert rail.post_message(events.Key("n", "n"))
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                rail._navigation_generation == starting_generation + 2
                and pilot.app.focused is successor.viewport
                and len(held_reveals) == 2
            ),
            description="second navigation overtaking the held first reveal",
        )

        stale_reveal = held_reveals[0]
        latest_reveal = held_reveals[1]
        monkeypatch.setattr(rail, "call_after_refresh", original_call_after_refresh)
        latest_reveal[0](*latest_reveal[1], **latest_reveal[2])
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                pilot.app.focused is successor.viewport
                and _fully_inside_outer(successor_header, outer)
            ),
            description="latest nested reveal settling before stale callback",
        )
        settled_scroll_y = outer.scroll_y
        await pilot.pause()
        await pilot.pause()

        scroll_calls = []

        def record_scroll_to(*args, **kwargs):
            scroll_calls.append((args, kwargs))

        settled_state = (
            pilot.app.focused,
            outer.scroll_y,
            rail._navigation_generation,
        )
        with monkeypatch.context() as scroll_probe:
            scroll_probe.setattr(outer, "scroll_to", record_scroll_to)
            stale_reveal[0](*stale_reveal[1], **stale_reveal[2])

            assert scroll_calls == []
            assert (
                pilot.app.focused,
                outer.scroll_y,
                rail._navigation_generation,
            ) == settled_state

            captured_follow_ons = []

            def capture_follow_on(callback, *args, **kwargs):
                captured_follow_ons.append((callback, args, kwargs))
                return True

            with monkeypatch.context() as counterfactual:
                counterfactual.setattr(
                    rail,
                    "_header_reveal_is_current",
                    lambda *_args, **_kwargs: True,
                )
                counterfactual.setattr(
                    rail,
                    "call_after_refresh",
                    capture_follow_on,
                )
                stale_reveal[0](*stale_reveal[1], **stale_reveal[2])

                assert len(scroll_calls) == 1
                assert len(captured_follow_ons) == 1
                assert (
                    getattr(captured_follow_ons[0][0], "__name__", None)
                    == "_reveal_boundary_header"
                )
                assert captured_follow_ons[0][1][-1] == 1

            assert (
                pilot.app.focused,
                outer.scroll_y,
                rail._navigation_generation,
            ) == settled_state

        assert pilot.app.focused is successor.viewport
        assert outer.scroll_y == settled_scroll_y
        assert _fully_inside_outer(successor_header, outer)


@pytest.mark.asyncio
async def test_navigation_focuses_first_enabled_visible_control_in_nonoverflow_target():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        inspector = rail.query_one("#console-run-inspector-state", ConsoleRunInspector)
        actions = tuple(
            replace(action, enabled=True, disabled_reason="")
            if action.widget_id == "console-inspector-save-chatbook"
            else action
            for action in inspector.state.actions
        )
        rows = tuple(
            row
            for row in inspector.state.rows
            if row.label
            not in {
                "Selected conversation",
                "Conversation source",
                "Workspace",
                "Resume state",
                "Prefill (next send only)",
                "Prefill (pinned)",
                "Session provider",
                "Session model",
                "Session endpoint",
                "Session sampling",
                "Session persona",
                "Selected message",
                "Message actions",
                "Keyboard",
                "Variants",
                "Excerpt",
                "Delete confirmation",
            }
        )
        inspector.sync_state(
            replace(
                inspector.state,
                rows=rows,
                actions=actions,
                dictionary_rows=(),
                dictionary_actions=(),
                world_book_rows=(),
                world_book_actions=(),
            )
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                bool(list(rail.query("#console-inspector-save-chatbook")))
                and not rail.query_one(
                    "#console-inspector-save-chatbook", Button
                ).disabled
                and rail.query_one("#console-inspector-save-chatbook", Button).display
            ),
            description="real enabled Artifacts control",
        )
        artifacts = rail.query_one("#console-inspector-save-chatbook", Button)
        artifacts.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is artifacts,
            description="enabled Artifacts control focus anchor",
        )
        # TASK-24704 (Qodo #6): `Changes` now renders whenever its one action
        # is disabled WITH a reason -- the normal state when change tracking
        # is off -- because that is the only way TASK-24606's disabled-action
        # explanation is reachable for an owner that has no rows of its own.
        # Several sections between Artifacts and Settings are all-`Static`
        # and so have no focusable control; `_focus_boundary` parks focus on
        # the outer scroller and reveals their header, exactly as `Run` and
        # `Source Readiness` already did.
        #
        # The contract is therefore "n reaches Settings and never gets
        # stuck", not "Settings is the next n". Asserting a press COUNT would
        # re-break every time a section is added or removed; asserting
        # progress catches the real defect, which was that `n` looped on the
        # first boundary forever (measured: six presses, all landing on
        # `#console-inspector-rail-body`).
        settings_control = rail.query_one("#console-settings-open", Button)
        visited: list[object] = []
        for _ in range(12):
            if pilot.app.focused is settings_control:
                break
            await pilot.press("n")
            await pilot.pause(0.1)
            visited.append(pilot.app.focused)
        assert pilot.app.focused is settings_control, (
            "n never reached the Settings control; visited "
            f"{[getattr(w, 'id', None) for w in visited]}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_mode", "expected_selector"),
    (
        ("header", "#console-inspector-approvals-heading"),
        ("body", f"#{CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID}"),
        ("none", f"#{CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID}"),
    ),
)
async def test_run_boundary_focus_never_leaks_to_sibling_group_control(
    target_mode: str,
    expected_selector: str,
):
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        inspector = rail.query_one("#console-run-inspector-state", ConsoleRunInspector)
        actions = tuple(
            replace(
                action,
                enabled=(
                    action.widget_id == CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID
                    or (
                        target_mode in {"header", "body"}
                        and action.widget_id == CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID
                    )
                ),
                disabled_reason=(
                    ""
                    if action.widget_id == CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID
                    or target_mode in {"header", "body"}
                    else "unavailable"
                ),
            )
            for action in inspector.state.actions
            if action.widget_id
            in {
                CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
                CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
            }
        )
        assert {action.widget_id for action in actions} == {
            CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
            CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
        }
        inspector.sync_state(
            replace(
                inspector.state,
                rows=(
                    ConsoleDisplayRow("Tools", "ready"),
                    ConsoleDisplayRow("Approvals", "none"),
                ),
                actions=actions,
                dictionary_rows=(),
                dictionary_actions=(),
                world_book_rows=(),
                world_book_actions=(),
            )
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                bool(list(rail.query("#console-bounded-section-approvals")))
                and bool(list(rail.query(f"#{CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID}")))
                and not rail.query_one(
                    f"#{CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID}", Button
                ).disabled
            ),
            description="real sibling Run Inspector boundaries",
        )
        approvals = rail.query_one(
            "#console-bounded-section-approvals", ConsoleBoundedSection
        )
        approvals.set_allocation(20)
        approvals_header = rail.query_one("#console-inspector-approvals-heading")
        artifacts = rail.query_one(f"#{CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID}", Button)
        if target_mode == "header":
            approvals_header.can_focus = True
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                not approvals.viewport.can_focus
                and not artifacts.disabled
                and artifacts.display
            ),
            description="nonoverflow target beside enabled sibling action",
        )

        tools_header = rail.query_one("#console-inspector-tools-heading")
        tools_header.can_focus = True
        tools_header.focus()
        await pilot.press("n")
        expected = rail.query_one(expected_selector)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is expected,
            description=f"{target_mode} target-local focus priority",
        )

        if target_mode != "none":
            assert pilot.app.focused is not artifacts
        if target_mode != "none":
            await pilot.press("p")
            await _wait_for_right_rail_condition(
                pilot,
                lambda: pilot.app.focused is tools_header,
                description="previous Run sibling without skip or wrap",
            )
            await pilot.press("n")
            await _wait_for_right_rail_condition(
                pilot,
                lambda: pilot.app.focused is expected,
                description="next Run sibling without skip or wrap",
            )


@pytest.mark.asyncio
async def test_interior_direct_boundaries_n_then_p_do_not_wrap():
    async with make_console_pilot(
        size=(160, 45), css=STAGED_OVERFLOW_TEST_CSS
    ) as pilot:
        rail = await _open_inspector(pilot)
        tray = rail.query_one("#console-staged-context-tray", ConsoleStagedContextTray)
        tray.sync_state(_staged_state(12))
        run = rail.query_one("#console-bounded-section-run", ConsoleBoundedSection)
        run.set_allocation(1)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                tray.query_one(
                    "#console-bounded-section-sources", ConsoleBoundedSection
                ).viewport.can_focus
                and run.viewport.can_focus
            ),
            description="two real interior direct boundaries overflow",
        )
        sources = tray.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        sources.viewport.focus()
        await pilot.press("n")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is run.viewport,
            description="interior direct boundary next",
        )
        assert _fully_inside_outer(
            run.parent.children[0], rail.query_one("#console-inspector-rail-body")
        )

        await pilot.press("p")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is sources.viewport,
            description="interior direct boundary previous",
        )


@pytest.mark.asyncio
async def test_other_nonboundary_anchor_and_outer_body_previous_target():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        sources = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        live_work = rail.query_one(
            "#console-bounded-section-live-work", ConsoleBoundedSection
        )
        await _overflow(sources)
        await _overflow(live_work)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: sources.viewport.can_focus and live_work.viewport.can_focus,
            description="outer-anchor boundary targets",
        )
        project = rail.query_one("#console-project-instruction-status-button", Button)
        project.focus()
        await pilot.press("p")
        await pilot.pause()
        assert pilot.app.focused is project
        await pilot.press("n")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is sources.viewport,
            description="non-boundary project row navigating forward",
        )

        outer = rail.query_one("#console-inspector-rail-body")
        outer.focus()
        await pilot.press("p")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is live_work.viewport,
            description="outer body navigating to last boundary",
        )


@pytest.mark.asyncio
async def test_tab_and_shift_tab_preserve_header_viewport_body_order():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        section = rail.query_one(
            "#console-bounded-section-session-settings", ConsoleBoundedSection
        )
        body_button = Button("body action", id="settings-body-tab-target")
        await section.viewport.remove_children()
        await section.viewport.mount(
            Static("\n".join(f"row {row}" for row in range(21))),
            body_button,
        )
        section.request_reconcile()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: section.viewport.can_focus,
            description="overflowing settings viewport in focus order",
        )
        header_button = rail.query_one("#console-settings-open", Button)
        header_button.focus()
        await pilot.press("tab")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is section.viewport,
            description="Tab from section header to viewport",
        )
        await pilot.press("tab")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is body_button,
            description="Tab from viewport to body control",
        )
        await pilot.press("shift+tab")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is section.viewport,
            description="Shift+Tab from body control to viewport",
        )
        await pilot.press("shift+tab")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is header_button,
            description="Shift+Tab from viewport to section header",
        )


@pytest.mark.asyncio
async def test_collapse_anchor_targets_first_or_last_without_wrapping():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        collapse = rail.query_one("#console-inspector-rail-collapse", Button)
        sources = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        live_work = rail.query_one(
            "#console-bounded-section-live-work", ConsoleBoundedSection
        )
        await _overflow(sources)
        await _overflow(live_work)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: sources.viewport.can_focus and live_work.viewport.can_focus,
            description="first and last overflowing Inspector boundaries",
        )

        collapse.focus()
        await pilot.press("n")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is sources.viewport,
            description="collapse anchor navigating to first boundary",
        )
        await pilot.press("p")
        await pilot.pause()
        assert pilot.app.focused is sources.viewport

        collapse.focus()
        await pilot.press("p")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is live_work.viewport,
            description="collapse anchor navigating to last boundary",
        )
        await pilot.press("n")
        await pilot.pause()
        assert pilot.app.focused is live_work.viewport


@pytest.mark.asyncio
async def test_disappearing_target_recovers_next_then_previous_and_keeps_outside_focus():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        section = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await section.viewport.remove_children()
        first = Button("first", id="recovery-first")
        middle = Button("middle", id="recovery-middle")
        last = Button("last", id="recovery-last")
        await section.viewport.mount(first, middle, last)
        section.request_reconcile()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: not section._reconcile_scheduled,
            description="recovery controls settled",
        )

        middle.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is middle,
            description="middle recovery target focused",
        )
        await middle.remove()
        section.request_reconcile()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is last,
            description="next recovery target",
        )

        await last.remove()
        section.request_reconcile()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is first,
            description="previous recovery target",
        )

        outside = pilot.app.screen.query_one("#console-native-composer")
        outside.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is outside,
            description="valid outside focus",
        )
        await first.remove()
        section.request_reconcile()
        await pilot.pause()
        assert pilot.app.focused is outside


@pytest.mark.asyncio
async def test_focus_leave_clears_recovery_incident_before_mutation_and_reentry():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        section = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await section.viewport.remove_children()
        former = Button("former action", id="leave-recovery-former")
        await section.viewport.mount(former)
        section.request_reconcile()
        former.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                rail._section_focus_history.get(section.section_id, (None, ()))[0]
                is former
            ),
            description="focused action captured by Inspector recovery",
        )

        outside = pilot.app.screen.query_one("#console-native-composer")
        outside.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: not rail.inspector_active(),
            description="intentional Inspector focus leave",
        )
        await pilot.pause()

        assert rail._section_focus_history == {}
        assert not rail._pending_focus_recoveries

        await former.remove()
        replacement = Button("replacement action", id="leave-recovery-replacement")
        await section.viewport.mount(replacement)
        section.request_reconcile()
        collapse = rail.query_one("#console-inspector-rail-collapse", Button)
        collapse.focus()
        await pilot.pause()
        await pilot.pause()

        assert pilot.app.focused is collapse
        _assert_recovery_references_are_current(rail)


@pytest.mark.asyncio
async def test_inactive_section_history_cannot_recover_after_focus_moves_elsewhere():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        sources = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        settings = rail.query_one(
            "#console-bounded-section-session-settings", ConsoleBoundedSection
        )
        await sources.viewport.remove_children()
        source_action = Button("source action", id="inactive-history-source")
        await sources.viewport.mount(source_action)
        sources.request_reconcile()
        await settings.viewport.remove_children()
        settings_action = Button("settings action", id="active-history-settings")
        await settings.viewport.mount(settings_action)
        settings.request_reconcile()

        source_action.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is source_action,
            description="first section recovery history",
        )
        settings_action.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is settings_action,
            description="focus moved to a different Inspector section",
        )

        assert set(rail._section_focus_history) == {settings.section_id}

        await source_action.remove()
        sources.request_reconcile()
        await pilot.pause()
        await pilot.pause()

        assert pilot.app.focused is settings_action
        _assert_recovery_references_are_current(rail)


@pytest.mark.asyncio
async def test_empty_owner_is_settled_absence_with_one_outer_fallback():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        sources = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await sources.viewport.remove_children()
        source_action = Button("source action", id="empty-owner-source-action")
        await sources.viewport.mount(source_action)
        sources.request_reconcile()
        source_action.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is source_action,
            description="empty-owner focused action",
        )

        owner = sources.parent
        await owner.remove_children()
        rail.request_outer_reconcile()
        outer = rail.query_one("#console-inspector-rail-body")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                pilot.app.focused is outer
                and not rail._pending_focus_recoveries
                and not rail._outer_reconcile_scheduled
            ),
            description="empty owner settled as absent boundary",
        )

        assert tuple(owner.children) == ()
        assert sources.section_id not in rail._section_focus_history


@pytest.mark.asyncio
async def test_persistent_malformed_owner_drains_recovery_with_outer_fallback():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        sources = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await sources.viewport.remove_children()
        source_action = Button("source action", id="malformed-owner-source-action")
        await sources.viewport.mount(source_action)
        sources.request_reconcile()
        source_action.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is source_action,
            description="malformed-owner focused action",
        )

        owner = sources.parent
        header = _external_boundary_header(sources)
        malformed = ConsoleBoundedSection(
            Static("persistent malformed owner"),
            section_id="persistent-malformed-owner",
        )
        await owner.mount(malformed, before=header)
        await sources.remove()
        await header.remove()
        rail.request_outer_reconcile()
        outer = rail.query_one("#console-inspector-rail-body")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                pilot.app.focused is outer
                and not rail._pending_focus_recoveries
                and not rail._outer_reconcile_scheduled
            ),
            description="persistent malformed owner recovery drained",
            attempts=10,
        )

        assert tuple(owner.children) == (malformed,)
        assert sources.section_id not in rail._section_focus_history


@pytest.mark.asyncio
async def test_structural_run_recompose_restores_same_stable_action_id():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        inspector = rail.query_one("#console-run-inspector-state", ConsoleRunInspector)
        initial_actions = tuple(
            replace(
                action,
                enabled=action.widget_id == CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
                disabled_reason=(
                    ""
                    if action.widget_id == CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID
                    else "unavailable"
                ),
            )
            for action in inspector.state.actions
        )
        assert {action.widget_id for action in initial_actions} >= {
            CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
            CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
        }
        initial_state = replace(inspector.state, actions=initial_actions)
        before_initial_recompose = inspector.recompose_count
        previous_mount = rail.query_one(
            f"#{CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID}", Button
        )
        inspector.sync_state(initial_state)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                inspector.recompose_count > before_initial_recompose
                and bool(list(rail.query(f"#{CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID}")))
                and rail.query_one(f"#{CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID}", Button)
                is not previous_mount
            ),
            description="enabled approval action before structural recompose",
        )
        previous = rail.query_one(f"#{CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID}", Button)
        previous.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is previous,
            description="approval action focused before structural recompose",
        )

        replacement_actions = tuple(
            replace(
                action,
                enabled=action.widget_id
                in {
                    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
                    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
                },
                disabled_reason=(
                    ""
                    if action.widget_id
                    in {
                        CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
                        CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
                    }
                    else "unavailable"
                ),
            )
            for action in initial_actions
        )
        before_recompose = inspector.recompose_count
        inspector.sync_state(replace(initial_state, actions=replacement_actions))
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                inspector.recompose_count > before_recompose
                and bool(list(rail.query(f"#{CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID}")))
                and rail.query_one(f"#{CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID}", Button)
                is not previous
            ),
            description="replacement approval action mounted after structural sync",
        )
        current = rail.query_one(f"#{CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID}", Button)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is current,
            description="same stable action id focus restored after structural sync",
        )

        assert pilot.app.focused is not rail.query_one("#console-retrieval-scope-row")
        history_target, _controls = rail._section_focus_history["approvals"]
        assert history_target is current
        _assert_recovery_references_are_current(rail)


@pytest.mark.asyncio
async def test_structural_control_and_boundary_removal_recovers_without_stale_reentry():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        inspector = rail.query_one("#console-run-inspector-state", ConsoleRunInspector)
        first = ConsoleInspectorAction("recompose-action-first", "First", True)
        middle = ConsoleInspectorAction("recompose-action-middle", "Middle", True)
        inserted = ConsoleInspectorAction("recompose-action-inserted", "Inserted", True)
        last = ConsoleInspectorAction("recompose-action-last", "Last", True)
        initial_state = replace(
            inspector.state,
            dictionary_rows=(),
            dictionary_actions=(first, middle, last),
        )
        inspector.sync_state(initial_state)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: bool(list(rail.query("#recompose-action-middle"))),
            description="three structural recovery controls mounted",
        )
        old_middle = rail.query_one("#recompose-action-middle", Button)
        old_middle.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is old_middle,
            description="middle structural recovery control focused",
        )

        inspector.sync_state(
            replace(initial_state, dictionary_actions=(first, inserted, last))
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                not list(rail.query("#recompose-action-middle"))
                and bool(list(rail.query("#recompose-action-inserted")))
                and bool(list(rail.query("#recompose-action-last")))
            ),
            description="focused middle control structurally removed",
        )
        current_inserted = rail.query_one("#recompose-action-inserted", Button)
        current_last = rail.query_one("#recompose-action-last", Button)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is current_inserted,
            description="new current-DOM control at the removed positional anchor",
        )
        assert pilot.app.focused is not current_last
        _assert_recovery_references_are_current(rail)

        inspector.sync_state(
            replace(initial_state, dictionary_actions=(), dictionary_rows=())
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: not list(rail.query("#console-bounded-section-chat-dictionaries")),
            description="focused structural boundary removed",
        )
        outer = rail.query_one("#console-inspector-rail-body")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is outer,
            description="removed boundary recovery to outer body",
        )

        collapse = rail.query_one("#console-inspector-rail-collapse", Button)
        collapse.focus()
        await pilot.pause()
        await pilot.pause()
        assert pilot.app.focused is collapse
        assert "chat-dictionaries" not in rail._section_focus_history
        _assert_recovery_references_are_current(rail)


@pytest.mark.asyncio
async def test_recovery_uses_external_header_control():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        settings = rail.query_one(
            "#console-bounded-section-session-settings", ConsoleBoundedSection
        )
        await settings.viewport.remove_children()
        settings_target = Button("body", id="settings-recovery-target")
        await settings.viewport.mount(settings_target)
        settings.request_reconcile()
        settings_target.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is settings_target,
            description="settings recovery target focus",
        )
        await settings_target.remove()
        settings.request_reconcile()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: getattr(pilot.app.focused, "id", None) == "console-settings-open",
            description="external section-header recovery",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_outer", (False, True))
async def test_recovery_falls_back_to_outer_body_then_collapse(disable_outer):
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        sources = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await sources.viewport.remove_children()
        source_target = Button("body", id="source-recovery-target")
        await sources.viewport.mount(source_target)
        sources.request_reconcile()
        source_target.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is source_target,
            description="source recovery target focus",
        )
        if disable_outer:
            rail.query_one("#console-inspector-rail-body").can_focus = False
        await source_target.remove()
        sources.request_reconcile()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                getattr(pilot.app.focused, "id", None)
                == (
                    "console-inspector-rail-collapse"
                    if disable_outer
                    else "console-inspector-rail-body"
                )
            ),
            description="Inspector fallback recovery",
        )


@pytest.mark.asyncio
async def test_pointer_wheel_hands_from_local_boundary_to_outer_body():
    async with make_console_pilot(size=(160, 30)) as pilot:
        rail = await _open_inspector(pilot)
        outer = rail.query_one("#console-inspector-rail-body")
        section = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await _overflow(section, rows=30)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: section.viewport.max_scroll_y > 0 and outer.max_scroll_y > 0,
            description="nested Inspector scroll owners",
        )
        section.viewport.scroll_end(animate=False, immediate=True)
        outer.scroll_home(animate=False, immediate=True)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                section.viewport.scroll_y == section.viewport.max_scroll_y
                and outer.scroll_y == 0
            ),
            description="local bottom and outer top",
        )
        section.viewport.post_message(_wheel_down(section.viewport))
        await _wait_for_right_rail_condition(
            pilot,
            lambda: outer.scroll_y > 0,
            description="pointer boundary handoff to Inspector outer body",
        )


@pytest.mark.asyncio
async def test_local_offset_survives_collapse_reopen_without_extra_writes(
    monkeypatch,
):
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        rail = await _open_inspector(pilot)
        section = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await _overflow(section, rows=30)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: section.viewport.max_scroll_y >= 5,
            description="scrollable continuity section",
        )
        section.viewport.scroll_to(y=5, animate=False, immediate=True)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: section.viewport.scroll_y == 5,
            description="stored local offset",
        )

        writes = []
        original = screen._set_console_rail_preference

        def observe_preferences(**changes):
            writes.append(changes)
            return original(**changes)

        monkeypatch.setattr(screen, "_set_console_rail_preference", observe_preferences)
        await pilot.click("#console-inspector-rail-collapse")
        await pilot.click("#console-inspector-rail-open")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: rail.display and section.viewport.scroll_y == 5,
            description="same-mounted rail offset after reopen",
        )
        assert len(writes) == 2
        await pilot.pause()
        assert len(writes) == 2


@pytest.mark.asyncio
async def test_run_sync_preserves_in_place_offsets_then_structural_shrink_clamps():
    async with make_console_pilot(size=(160, 60), css=RUN_OWNER_TEST_CSS) as pilot:
        rail = await _open_inspector(pilot)
        outer = rail.query_one("#console-inspector-rail-body")
        hint = rail.query_one(f"#{INSPECTOR_OUTER_HINT_ID}", Static)
        inspector = rail.query_one("#console-run-inspector-state", ConsoleRunInspector)
        run_wrapper = inspector.parent
        for child in tuple(outer.children):
            if child is not run_wrapper:
                await child.remove()
        for child in tuple(run_wrapper.children):
            if child is not inspector:
                await child.remove()
        await pilot.resize_terminal(160, 59)
        await pilot.resize_terminal(160, 60)
        large_rows = tuple(
            ConsoleDisplayRow(f"Dictionary {index}", f"value {index}")
            for index in range(25)
        )
        base_rows = tuple(
            row
            for row in inspector.state.rows
            if row.label
            not in {
                "Selected conversation",
                "Conversation source",
                "Workspace",
                "Resume state",
                "Prefill (next send only)",
                "Prefill (pinned)",
                "Session provider",
                "Session model",
                "Session endpoint",
                "Session sampling",
                "Session persona",
                "Selected message",
                "Message actions",
                "Keyboard",
                "Variants",
                "Excerpt",
                "Delete confirmation",
            }
        )
        large_state = replace(
            inspector.state,
            rows=base_rows,
            dictionary_rows=large_rows,
            dictionary_actions=(),
            world_book_rows=(),
            world_book_actions=(),
        )
        before_growth_recompose = inspector.recompose_count
        inspector.sync_state(large_state)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                inspector.recompose_count > before_growth_recompose
                and "value 0"
                in str(
                    rail.query_one(
                        "#console-inspector-dictionaries-row-0", Static
                    ).renderable
                )
            ),
            description="real structural Run owner growth committed",
        )
        section = rail.query_one(
            "#console-bounded-section-chat-dictionaries", ConsoleBoundedSection
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: section.desired_content_lines >= 25,
            description="Run dynamic rows measured through local owner",
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                section.viewport.max_scroll_y >= 4 and not section._reconcile_scheduled
            ),
            description="Run local owner geometry before outer",
        )
        desired = max(
            child.virtual_region_with_margin.bottom
            for child in outer.children
            if child.display
        )
        without_hint = outer.content_region.height + (
            hint.region.height if hint.display else 0
        )
        spacer_height = max(0, without_hint - desired + 2)
        if spacer_height:
            spacer = Static("", id="run-owner-fixed-spacer")
            spacer.styles.height = spacer_height
            await outer.mount(spacer)
            await pilot.resize_terminal(160, 59)
            await pilot.resize_terminal(160, 60)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                outer.max_scroll_y >= 2
                and hint.display is True
                and not rail._outer_reconcile_scheduled
            ),
            description="Run outer owner geometry after local",
        )
        section.viewport.scroll_to(y=4, animate=False, immediate=True)
        outer.scroll_to(y=2, animate=False, immediate=True)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: section.viewport.scroll_y == 4 and outer.scroll_y == 2,
            description="local and outer offsets before in-place sync",
        )

        updated_state = replace(
            large_state,
            dictionary_rows=tuple(
                replace(row, value=f"updated {index}")
                for index, row in enumerate(large_rows)
            ),
        )
        before_recompose = inspector.recompose_count
        inspector.sync_state(updated_state)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                "updated 0"
                in str(
                    rail.query_one(
                        "#console-inspector-dictionaries-row-0", Static
                    ).renderable
                )
                and not section._reconcile_scheduled
                and not rail._outer_reconcile_scheduled
            ),
            description="same-key Run in-place owner sync",
        )
        assert inspector.recompose_count == before_recompose
        assert (
            rail.query_one(
                "#console-bounded-section-chat-dictionaries", ConsoleBoundedSection
            )
            is section
        )
        assert section.viewport.scroll_y == 4
        assert outer.scroll_y == 2

        inspector.sync_state(replace(updated_state, dictionary_rows=large_rows[:2]))
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                rail.query_one(
                    "#console-bounded-section-chat-dictionaries",
                    ConsoleBoundedSection,
                )
                is not section
            ),
            description="real structural Run shrink replacement",
        )
        shrunk = rail.query_one(
            "#console-bounded-section-chat-dictionaries", ConsoleBoundedSection
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                shrunk.desired_content_lines == 2
                and shrunk.viewport.scroll_y == 0
                and not shrunk._reconcile_scheduled
            ),
            description="real structural Run local shrink clamp",
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                outer.scroll_y <= outer.max_scroll_y
                and hint.display is False
                and not rail._outer_reconcile_scheduled
            ),
            description="real structural Run outer shrink clamp and cue update",
        )


@pytest.mark.asyncio
async def test_responsive_hide_and_reveal_hands_off_focus_without_losing_offset():
    async with make_console_pilot(size=(128, 40)) as pilot:
        rail = await _open_inspector(pilot)
        section = rail.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        await _overflow(section, rows=30)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: section.viewport.max_scroll_y >= 3 and section.viewport.can_focus,
            description="responsive overflowing local section",
        )
        section.viewport.scroll_to(y=3, animate=False, immediate=True)
        section.viewport.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                pilot.app.focused is section.viewport and section.viewport.scroll_y == 3
            ),
            description="focused responsive local offset",
        )

        await pilot.resize_terminal(129, 40)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                getattr(pilot.app.focused, "id", None) == "console-inspector-rail-open"
                and rail.display is False
            ),
            description="responsive Inspector reveal-control handoff",
        )
        assert section.viewport.scroll_y == 3

        await pilot.resize_terminal(128, 40)
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                getattr(pilot.app.focused, "id", None)
                == "console-inspector-rail-collapse"
                and rail.display is True
            ),
            description="responsive Inspector collapse-control handoff",
        )
        assert section.viewport.scroll_y == 3


@pytest.mark.asyncio
async def test_footer_and_f1_help_read_live_inspector_focus(monkeypatch):
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        rail = await _open_inspector(pilot)
        registered = []
        monkeypatch.setattr(
            screen,
            "register_footer_shortcuts",
            lambda *, source, shortcuts: registered.append((source, shortcuts)),
        )

        outside = screen.query_one("#console-native-composer")
        outside.focus()
        screen._register_console_footer_shortcuts()
        assert ("n/p", "Sections") not in registered[-1][1]

        collapse = rail.query_one("#console-inspector-rail-collapse", Button)
        collapse.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: registered and ("n/p", "Sections") in registered[-1][1],
            description="Inspector footer hint on focus enter",
        )

        await screen.action_show_workbench_help()
        await pilot.pause()
        panel = pilot.app.screen
        assert isinstance(panel, WorkbenchHelpPanel)
        assert "n / p: next / previous section" in panel.state.render_text()
        await pilot.press("escape")
        await pilot.pause()

        screen = pilot.app.screen
        screen.query_one("#console-native-composer").focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: screen._console_inspector_active() is False,
            description="focus leaving Inspector before live F1 invocation",
        )
        await _wait_for_right_rail_condition(
            pilot,
            lambda: registered and ("n/p", "Sections") not in registered[-1][1],
            description="Inspector footer hint removed on focus leave",
        )
        await screen.action_show_workbench_help()
        await pilot.pause()
        panel = pilot.app.screen
        assert isinstance(panel, WorkbenchHelpPanel)
        assert "n / p: next / previous section" not in panel.state.render_text()


@pytest.mark.asyncio
async def test_boundary_anchor_does_not_survive_focus_leaving_the_scroller():
    """TASK-24704 (Qodo #5, second round): the anchor must not go stale.

    `n`/`p` parks focus on the outer scroller for a section with no focusable
    control, and remembers which boundary that was so the next press can
    continue rather than resetting to the first. That memory is only valid
    while the scroller still holds focus BECAUSE navigation put it there --
    Tab away and back, and the scroller is an ordinary Tab stop again, which
    must restore its documented "outside the list, wrap to the far end"
    meaning.
    """
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        outer = rail.query_one("#console-inspector-rail-body")
        collapse = rail.query_one("#console-inspector-rail-collapse", Button)

        # Anchor on a real control so `n` reaches the rail's key handler, then
        # navigate until an all-`Static` boundary parks focus on the scroller.
        project = rail.query_one("#console-project-instruction-status-button", Button)
        project.focus()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is project,
            description="navigation anchor before parking",
        )
        for _ in range(12):
            await pilot.press("n")
            await pilot.pause(0.1)
            if pilot.app.focused is outer:
                break
        assert pilot.app.focused is outer, "never parked on the outer scroller"
        assert rail._last_boundary_index is not None, "anchor was not recorded"

        # Ordinary focus movement away ends the navigation state...
        collapse.focus()
        await pilot.pause()
        assert rail._last_boundary_index is None, (
            "the anchor survived focus leaving the scroller, so the next n/p "
            "would continue from stale section history"
        )

        # ...and coming back does not resurrect it.
        outer.focus()
        await pilot.pause()
        assert rail._last_boundary_index is None
