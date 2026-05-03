import pytest
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.Constants import TAB_STUDY
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.shell_destinations import get_shell_destination
from tldw_chatbook.app import TabNavigationProvider


def test_library_destination_keeps_workspaces_visible():
    library = get_shell_destination("library")

    assert "Workspaces" in library.purpose
    assert "Workspaces" in library.tooltip


def test_study_modules_remain_discoverable_as_legacy_direct_route():
    help_text = TabNavigationProvider.TAB_HELP_TEXT[TAB_STUDY].lower()

    assert "flashcards" in help_text
    assert "quizzes" in help_text


@pytest.mark.asyncio
async def test_navigation_exposes_explicit_overflow_hint():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.1)
        overflow = app.query_one("#nav-overflow-hint", Static)

    assert "More" in str(overflow.renderable)
    assert "Ctrl+P" in str(overflow.renderable)
