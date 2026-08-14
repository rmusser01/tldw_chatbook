import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.Constants import TAB_STUDY
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.shell_destinations import get_shell_destination
from tldw_chatbook.app import TabNavigationProvider


def test_library_destination_keeps_workspaces_visible():
    library = get_shell_destination("library")

    assert "Workspaces" in library.purpose
    assert "Workspaces" in library.tooltip


def test_study_modules_remain_discoverable_as_legacy_direct_route():
    help_text = TabNavigationProvider._shell_help_text(TAB_STUDY).lower()

    assert "flashcards" in help_text
    assert "quizzes" in help_text


@pytest.mark.asyncio
async def test_navigation_exposes_explicit_overflow_hint():
    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.1)
        overflow = app.query_one("#nav-overflow-hint")

        # Assert inside the running app: widget state (display in particular)
        # is not guaranteed to survive app shutdown.
        assert "More" in str(overflow.label)
        assert overflow.display is True
        assert overflow.tooltip == "All destinations"
