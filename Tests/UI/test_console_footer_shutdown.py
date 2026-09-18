"""Late Console setup/footer refreshes survive Textual's empty shutdown stack."""

import pytest
from textual.app import ScreenStackError
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_right_rail import make_console_pilot


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("callback", ["rail", "inspector", "setup"])
async def test_footer_focus_checks_survive_empty_stack_without_losing_live_hints(
    callback, request
):
    """Exercise the real footer path after the last screen has been popped."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        screen._set_console_rail_preference(right_open=True)
        await pilot.pause()
        collapse = screen.query_one("#console-inspector-rail-collapse", Button)
        screen.set_focus(collapse)
        assert pilot.app.focused is collapse
        assert screen._console_rail_focus_active() is True
        assert screen._console_inspector_active() is True
        screen._register_console_footer_shortcuts()
        rail_hints = dict(screen._footer_shortcut_registration[1])
        assert rail_hints["Esc"] == "composer · F6 panes"
        assert rail_hints["n/p"] == "Sections"

        # No await while the stack is empty: reproduce the precise callback
        # window deterministically, then let the normal harness teardown run.
        stack = pilot.app._screen_stack
        saved = list(stack)
        stack.clear()
        try:
            with pytest.raises(ScreenStackError):
                _ = pilot.app.focused
            if callback == "rail":
                assert screen._console_rail_focus_active() is False
            elif callback == "inspector":
                assert screen._console_inspector_active() is False
            else:
                screen._apply_console_setup_block(False)
                hints = dict(screen._footer_shortcut_registration[1])
                assert hints.get("Esc") != "composer · F6 panes"
                assert "n/p" not in hints
        finally:
            stack.extend(saved)

        assert pilot.app.focused is collapse
        assert screen._console_rail_focus_active() is True
        assert screen._console_inspector_active() is True
        screen._register_console_footer_shortcuts()
        assert dict(screen._footer_shortcut_registration[1]) == rail_hints
        composer = screen.query_one("#console-native-composer")
        screen.set_focus(composer)
        assert pilot.app.focused is composer
        assert screen._console_rail_focus_active() is False
        assert screen._console_inspector_active() is False
        screen._register_console_footer_shortcuts()
        hints = dict(screen._footer_shortcut_registration[1])
        assert hints.get("Esc") != "composer · F6 panes"
        assert "n/p" not in hints
