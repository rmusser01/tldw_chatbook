"""Visible hands-free toggle (task-18911 fix 2).

The Switch in the Console status header is the soft-keyboard-only
user's entry/exit for hands-free mode -- the mode was previously
keybinding-only (ctrl+shift+h / escape).
"""

import pytest
from textual.widgets import Switch

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Chat.console_hands_free import HandsFreeController


@pytest.mark.asyncio
async def test_switch_renders_and_toggles_hands_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This is a UI lifecycle test, not an STT integration test. Entering the
    # real loop opens the microphone and can initialize optional native ML
    # backends, which makes the test machine-dependent.
    def enter_without_capture(
        controller: HandsFreeController,
        capture_live: bool,
    ) -> None:
        del capture_live
        controller._transition("listening")

    monkeypatch.setattr(HandsFreeController, "enter", enter_without_capture)
    app = _build_test_app(configured_default="chat")
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(200):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "ChatScreen":
                break
        await pilot.pause(0.5)
        s = app.screen
        sw = s.query_one("#console-hands-free-switch", Switch)
        assert sw.value is False, "switch starts off when hands-free inactive"

        sw.focus()
        await pilot.press("enter")
        await pilot.pause(0.4)
        # hands-free session should now be active (or at minimum the switch
        # reflects the requested state)
        assert sw.value is True, "switch flips on after activation"
        active = s._console_hands_free is not None
        assert active is True, "hands-free session started via switch"

        # exit via the switch again
        sw.focus()
        await pilot.press("enter")
        await pilot.pause(0.4)
        assert s._console_hands_free is None, "hands-free session ended via switch"
