"""Focused Task-7 behavior for Inspector outer scrolling and navigation."""

from __future__ import annotations

import pytest
from textual import events
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)

from Tests.UI.test_console_right_rail import (
    _wait_for_right_rail_condition,
    make_console_pilot,
)


INSPECTOR_OUTER_HINT = "▼ more sections — scroll"
INSPECTOR_OUTER_HINT_ID = "console-inspector-outer-scroll-hint"


def _inside(widget, owner) -> bool:
    return widget is owner or owner in widget.ancestors


async def _open_inspector(pilot):
    if not pilot.app.screen.query_one("#console-right-rail").display:
        await pilot.click("#console-inspector-rail-open")
    await pilot.pause()
    return pilot.app.screen.query_one("#console-right-rail")


async def _overflow(section: ConsoleBoundedSection, rows: int = 21) -> None:
    await section.viewport.remove_children()
    await section.viewport.mount(Static("\n".join(f"row {row}" for row in range(rows))))
    section.request_reconcile()


def _wheel_down(widget: Widget) -> events.MouseScrollDown:
    return events.MouseScrollDown(widget, 0, 0, 0, 1, 0, False, False, False)


@pytest.mark.asyncio
async def test_outer_hint_is_pinned_third_child_with_exact_nonfocusable_copy():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        header, body, hint = rail.children

        assert header.id is None
        assert body.id == "console-inspector-rail-body"
        assert hint.id == INSPECTOR_OUTER_HINT_ID
        assert str(hint.renderable) in ("", INSPECTOR_OUTER_HINT)
        assert hint.can_focus is False


@pytest.mark.asyncio
async def test_outer_hint_uses_counterfactual_ten_eleven_ten_transition():
    async with make_console_pilot(size=(160, 45)) as pilot:
        rail = await _open_inspector(pilot)
        body = rail.query_one("#console-inspector-rail-body")
        hint = rail.query_one(f"#{INSPECTOR_OUTER_HINT_ID}", Static)
        await body.remove_children()
        # The rail's one-cell border leaves 11 content rows: one header and
        # the exact ten-row counterfactual body viewport exercised below.
        rail.styles.height = 12
        child = Static("ten", id="outer-fixed-child")
        child.styles.height = 10
        await body.mount(child)
        rail.request_outer_reconcile()

        await _wait_for_right_rail_condition(
            pilot,
            lambda: not rail._outer_reconcile_scheduled and hint.display is False,
            description="ten rows without an outer hint slot",
        )

        child.styles.height = 11
        child.refresh(layout=True)
        rail.request_outer_reconcile()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: not rail._outer_reconcile_scheduled and hint.display is True,
            description="eleven rows with an outer hint slot",
        )
        assert str(hint.renderable) == INSPECTOR_OUTER_HINT
        assert hint.region.height == 1

        child.styles.height = 10
        child.refresh(layout=True)
        rail.request_outer_reconcile()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: not rail._outer_reconcile_scheduled and hint.display is False,
            description="ten rows after shrink without a sticky hint slot",
        )
        assert body.scroll_y == 0

        child.styles.height = 11
        child.refresh(layout=True)
        rail.styles.height = 12
        rail.request_outer_reconcile()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: hint.display is True and not rail._outer_reconcile_scheduled,
            description="fixed child overflowing after terminal shrink",
        )
        rail.styles.height = 13
        await _wait_for_right_rail_condition(
            pilot,
            lambda: hint.display is False and not rail._outer_reconcile_scheduled,
            description="fixed child fitting after terminal growth",
        )


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
            lambda: (
                _inside(
                    pilot.app.focused,
                    rail.query_one(
                        "#console-bounded-section-run", ConsoleBoundedSection
                    ),
                )
                or pilot.app.focused.id == "console-inspector-rail-body"
            ),
            description="Scope navigating to the following Run boundary",
        )

        run_status = rail.query_one("#console-inspector-run-status-summary")
        run_status.can_focus = True
        run_status.focus()
        await pilot.press("p")
        await _wait_for_right_rail_condition(
            pilot,
            lambda: pilot.app.focused is sources.viewport,
            description="run-status compact row navigating backward",
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
async def test_local_offset_survives_collapse_reopen_then_clamps_without_extra_writes(
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

        content = section.viewport.children[0]
        content.update("one\ntwo\nthree")
        content.refresh(layout=True)
        section.request_reconcile()
        rail.request_outer_reconcile()
        await _wait_for_right_rail_condition(
            pilot,
            lambda: (
                section.viewport.scroll_y == 0
                and not section._reconcile_scheduled
                and not rail._outer_reconcile_scheduled
            ),
            description="clamped local offset after shrink",
        )
        assert len(writes) == 2


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
