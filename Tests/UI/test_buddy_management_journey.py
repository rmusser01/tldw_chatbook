"""Fresh-profile Console menu, independent artwork, and off-screen Buddy journey."""

import asyncio

import pytest
from textual.widgets import Button, Select, Switch

from Tests.UI.test_console_navigation_decisions import _until
from Tests.UI.test_console_screen_reuse import (
    _boot_settled,
    _press_until_screen,
    _scratch_env,
)


@pytest.mark.asyncio
@pytest.mark.ui
async def test_fresh_install_selects_independent_buddy_and_opens_pinned_chat_from_home(
    monkeypatch, tmp_path
):
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.buddy_management import (
        get_buddy_management,
        open_buddy_interaction,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
        BuddyManagementModal,
    )

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        controller = console._ensure_console_chat_controller()
        target = controller.store.ensure_session()
        persona_before = await asyncio.to_thread(
            app.local_character_persona_service.list_persona_profiles
        )

        console.query_one("#console-composer-menu", Button).press()
        await _until(lambda: type(app.screen).__name__ == "ConsoleComposerMenuModal")
        app.screen.query_one("#console-composer-menu-buddy", Button).press()
        await _until(lambda: isinstance(app.screen, BuddyManagementModal))
        # Screen-stack publication precedes nested Select label composition.
        await pilot.pause()
        manager = get_buddy_management(app)
        buddies = await asyncio.to_thread(manager.library.list_buddies)
        assert buddies, (
            "Fresh installation must offer built-in independent Buddy artwork"
        )
        modal = app.screen
        modal.query_one("#buddy-artwork", Select).value = buddies[0].id
        modal.query_one("#buddy-follow", Select).value = f"conversation:{target.id}"
        modal.query_one("#buddy-motion", Select).value = "static"
        modal.query_one("#buddy-enabled", Switch).value = True
        modal.query_one("#buddy-apply", Button).press()
        await _until(
            lambda: (
                app.app_config.get("buddy_interaction", {}).get("target_id")
                == target.id
            )
        )
        assert (
            manager.controller.current_preferences().selection.buddy_id == buddies[0].id
        )
        assert manager.preferences.animated is False
        assert (
            await asyncio.to_thread(
                app.local_character_persona_service.list_persona_profiles
            )
            == persona_before
        )
        assert manager.library.get_graph(buddies[0].id).identity.persona_id is None

        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        home = app.screen
        open_buddy_interaction(app)
        await _until(lambda: type(app.screen).__name__ == "BuddyConversationModal")
        await pilot.pause()
        assert app.screen.binding.target_id == target.id
        assert controller.store.active_session_id == target.id
        app.screen.query_one("#buddy-close", Button).press()
        await _until(lambda: app.screen is home)

        manager.request_open()
        await _until(lambda: isinstance(app.screen, BuddyManagementModal))
        await pilot.pause()
        before_cancel = manager.controller.current_preferences()
        app.screen.query_one("#buddy-enabled", Switch).value = False
        await pilot.press("escape")
        assert app.screen is home
        assert manager.controller.current_preferences() == before_cancel
