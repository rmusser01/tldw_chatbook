"""Archive entry points preserve the switcher's layout and activation boundary."""

from unittest.mock import Mock

import pytest
from textual.widgets import Button

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_conversation_activation import ConsoleActivationPhase
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)


@pytest.mark.parametrize(
    "phase",
    [ConsoleActivationPhase.OPENING_CANCELLABLE, ConsoleActivationPhase.COMMITTING],
)
def test_archive_search_cannot_interrupt_character_activation(phase):
    modal = ConsoleSessionSwitcherModal(on_full_search=Mock())
    modal._activation_phase = phase
    event = Button.Pressed(Button("Archive", id="console-switcher-archive"))
    # The busy guard must run before reading or dismissing a modal.
    modal.open_full_search(event)
    modal._on_full_search.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(52, 20), (100, 30)])
async def test_archive_routes_and_cancel_fit_switcher_viewport(size):
    app = ConsolidatedCSSApp()
    async with app.run_test(size=size) as pilot:
        modal = ConsoleSessionSwitcherModal(on_full_search=Mock())
        await app.push_screen(modal)
        await pilot.pause()
        for selector in (
            "#console-switcher-full-search",
            "#console-switcher-archive",
            "#console-switcher-cancel",
        ):
            button = modal.query_one(selector, Button)
            assert button.region.height == 1
            assert button.region.bottom <= size[1]
            assert button.region.right <= size[0]
            assert button.visible
