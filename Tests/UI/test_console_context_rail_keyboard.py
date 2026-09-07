"""Keyboard contract for the Context rail (TASK-23198).

A 2026-08-29 UX audit reported that Tab never leaves the Context rail and
called it a WCAG 2.1.2 (No Keyboard Trap) failure. That conclusion was
WRONG, and the first two tests record why so it is not "re-found" later.

Tab is scoped to the focused Console region ON PURPOSE
(``ChatScreen.action_focus_next``, TASK-2154.11): unscoped, a Tab tour of
the Console crossed all fifteen app-navigation buttons mid-session. WCAG
2.1.2 does not require Tab specifically -- it requires that focus can be
moved away using the keyboard, and that when that needs more than Tab, the
method is advised. F6 moves focus out in a single press, and the persistent
footer advertises "F6 next pane" at all times. The criterion is met.

What the audit was right about is the third test: the rail had no bindings
of its own, so a user who wanted every section shut had to click seven
disclosure toggles.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_left_rail import make_console_pilot


def _rail_nodes(screen):
    rail = screen.query_one("#console-left-rail")
    return rail, set(rail.query("*").nodes) | {rail}


async def _focus_inside_rail(pilot):
    screen = pilot.app.screen
    rail, nodes = _rail_nodes(screen)
    target = next(widget for widget in screen.focus_chain if widget in nodes)
    target.focus()
    await pilot.pause(0.2)
    return rail, nodes


@pytest.mark.asyncio
async def test_focus_can_leave_the_context_rail_with_the_keyboard_alone() -> None:
    """WCAG 2.1.2: there must be a keyboard way out, and there is."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        _rail, nodes = await _focus_inside_rail(pilot)

        await pilot.press("f6")
        await pilot.pause(0.25)

        focused = pilot.app.focused
        assert focused is not None, "F6 left focus nowhere"
        assert focused not in nodes, (
            "F6 did not move focus out of the Context rail; with Tab scoped to "
            "the region this is the only keyboard way out"
        )


@pytest.mark.asyncio
async def test_the_way_out_is_advertised_on_screen() -> None:
    """WCAG 2.1.2's advisory clause: if it is not Tab, say so, always."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        painted = " ".join(
            str(getattr(widget, "renderable", ""))
            for source in (pilot.app.screen, pilot.app)
            for widget in source.query("*")
            if widget.display
        )
        assert "F6" in painted, (
            "the escape key is not advertised anywhere on screen, which is what "
            "would make the scoped Tab an actual keyboard trap"
        )
        assert "next pane" in painted.lower()


@pytest.mark.asyncio
async def test_the_rail_can_collapse_and_expand_every_section_from_the_keyboard() -> (
    None
):
    """Seven disclosure clicks is not a keyboard affordance."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        rail, _nodes = await _focus_inside_rail(pilot)

        section_ids = (
            "workspace",
            "conversations",
            "model",
            "agent",
            "details",
            "character",
        )

        def open_flags() -> dict[str, bool]:
            return {
                section_id: bool(
                    screen.query_one(
                        f"#console-rail-section-header-{section_id}"
                    ).open
                )
                for section_id in section_ids
            }

        rail.action_collapse_all_sections()
        await pilot.pause(0.5)
        assert not any(open_flags().values()), (
            f"collapse-all left sections open: {open_flags()}"
        )

        rail.action_expand_all_sections()
        await pilot.pause(0.5)
        assert all(open_flags().values()), (
            f"expand-all left sections closed: {open_flags()}"
        )


@pytest.mark.asyncio
async def test_the_rail_declares_its_own_bindings() -> None:
    """The audit's S4-A: the rail declared no BINDINGS at all."""
    from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail

    actions = {
        binding.action
        for binding in ConsoleLeftRail.BINDINGS
        if hasattr(binding, "action")
    }
    assert "collapse_all_sections" in actions
    assert "expand_all_sections" in actions


@pytest.mark.asyncio
async def test_the_collapse_all_key_actually_fires_from_inside_the_rail() -> None:
    """Calling the action proves the action; only a key press proves the key."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        await _focus_inside_rail(pilot)

        section_ids = (
            "workspace",
            "conversations",
            "model",
            "agent",
            "details",
            "character",
        )

        def any_open() -> bool:
            return any(
                screen.query_one(f"#console-rail-section-header-{s}").open
                for s in section_ids
            )

        assert any_open(), "nothing was open to collapse"

        await pilot.press("ctrl+shift+left")
        await pilot.pause(0.6)
        assert not any_open(), (
            "ctrl+shift+left did not reach the rail; the binding is declared "
            "but not actually reachable from a focused rail control"
        )

        await pilot.press("ctrl+shift+right")
        await pilot.pause(0.6)
        assert all(
            screen.query_one(f"#console-rail-section-header-{s}").open
            for s in section_ids
        ), "ctrl+shift+right did not reopen every section"
