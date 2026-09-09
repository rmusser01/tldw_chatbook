"""Both Buddy management controls use the same app-owned opener."""

import pytest
from textual.widgets import Button

from Tests.UI.test_persona_buddy_widget import (
    _BuddyApp,
    _compositor_text,
    _FakeController,
    _wait_until,
)
from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    build_composer_menu_entries,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
    PersonaBuddyWidget,
)


def test_composer_menu_has_a_named_buddy_action():
    choices = [
        entry for entry in build_composer_menu_entries() if entry.action_id == "buddy"
    ]
    assert len(choices) == 1
    assert choices[0].label == "Buddy…"
    assert choices[0].enabled


@pytest.mark.asyncio
async def test_floating_settings_opens_the_shared_management_flow(monkeypatch):
    from tldw_chatbook.UI.Navigation import buddy_management

    opened = []
    monkeypatch.setattr(buddy_management, "open_buddy_management", opened.append)
    app = _BuddyApp(_FakeController())
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "BUDDY-A" in _compositor_text(app.screen))
        assert buddy.query("#persona-buddy-settings"), (
            "Floating Buddy settings control is missing"
        )
        button = buddy.query_one("#persona-buddy-settings", Button)
        assert button.tooltip == "Buddy & Persona Management"
        await pilot.pause()
        assert button.display and button.region.width >= 3
        assert await pilot.click(button, offset=(1, 0)), (
            button.region,
            _compositor_text(app.screen),
        )
        assert opened == [app]


@pytest.mark.asyncio
async def test_body_click_and_enter_open_interaction_without_geometry_write(
    monkeypatch,
):
    from tldw_chatbook.UI.Navigation import buddy_management

    opened = []
    monkeypatch.setattr(buddy_management, "open_buddy_interaction", opened.append)
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "BUDDY-A" in _compositor_text(app.screen))
        await pilot.pause()
        writes = []
        monkeypatch.setattr(buddy, "_schedule_geometry_persist", writes.append)
        await pilot.click(buddy.query_one("#persona-buddy-frame"), offset=(6, 3))
        assert opened == [app]
        await pilot.press("enter")
        assert opened == [app, app]
        assert writes == []
