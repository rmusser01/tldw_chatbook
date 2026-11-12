"""All help surfaces open one report without navigating or bootstrapping Console."""

import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import Input, TextArea

from Tests.UI.test_bug_report_dialog import ReportHarness, answer
from tldw_chatbook.app import TldwCli
from tldw_chatbook.app_command_providers import QuickActionsProvider
from tldw_chatbook.UI.Logs_Window import LogsWindow
from tldw_chatbook.UI.bug_report_capture import capture_bug_report_context
from tldw_chatbook.Widgets.bug_report_dialog import BugReportDialog

pytestmark = pytest.mark.bootstrap_profile


class EntryHarness(ReportHarness):
    COMMANDS = {QuickActionsProvider}
    action_report_bug = TldwCli.action_report_bug

    def compose(self):
        yield Input("underlying unsaved text", id="draft")
        # In the app the destination shell bounds this window. Leave room for
        # the harness's draft input so the action bar remains inside the screen.
        window = LogsWindow(self)
        window.styles.height = "1fr"
        yield window


async def test_palette_search_and_discovery_expose_report_action():
    app = EntryHarness()
    async with app.run_test() as pilot:
        provider = QuickActionsProvider(app.screen)
        found = [hit async for hit in provider.search("report a bug")]
        discovered = [hit async for hit in provider.discover()]
        assert any("Report a bug" in str(hit.text) for hit in found)
        assert any("Report a bug" in str(hit.text) for hit in discovered)
        await pilot.press("ctrl+p")
        await pilot.press(*"report a bug")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, BugReportDialog)
        assert app.screen._collector.context.origin_route == "logs"
        assert app.query_one("#draft", Input).value == "underlying unsaved text"
        assert app.console_runtime is None


async def test_logs_action_preserves_filter_and_duplicate_invocation_reuses_report():
    app = EntryHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        window = app.query_one(LogsWindow)
        window.query_one("#logs-filter-text", Input).value = "needle"
        await pilot.pause()
        button = window.query_one("#report-bug-button")
        center = (button.region.width // 2, button.region.height // 2)
        assert await pilot.click(button, offset=center)
        # The shared action opens the dialog after the next screen refresh.
        async with asyncio.timeout(10):
            while not isinstance(app.screen, BugReportDialog):
                await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, BugReportDialog)
        dialog.query_one("#bug-report-description", TextArea).load_text(
            "keep this report"
        )
        app.action_report_bug()
        app.action_report_bug()
        await pilot.pause()
        assert app.screen is dialog
        assert (
            sum(isinstance(screen, BugReportDialog) for screen in app.screen_stack) == 1
        )
        assert window.query_one("#logs-filter-text", Input).value == "needle"
        await pilot.press("escape")
        await answer(pilot, "#confirm-button")
        assert app._bug_report_dialog is None


def test_shared_trace_does_not_select_unrelated_local_console():
    from Tests.Utils.test_bug_report_capture import _app

    app = _app(True)
    app.screen = SimpleNamespace(_shared_trace=True)
    context = capture_bug_report_context(app)
    assert context.conversation_id is None and context.conversation_label is None


async def test_about_action_is_on_canonical_settings_surface():
    from Tests.UI.test_destination_shells import DestinationHarness, _build_test_app
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    calls = []
    owner = _build_test_app()
    owner.action_report_bug = lambda: calls.append("report")
    class AboutHarness(DestinationHarness):
        def action_report_bug(self):
            owner.action_report_bug()

    app = AboutHarness(owner, "settings")
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await app.workers.wait_for_complete()
        screen.active_category = SettingsCategoryId.ABOUT.value
        await pilot.pause()
        button = screen.query_one("#settings-about-report-bug")
        button.focus()
        await pilot.press("enter")
        assert calls == ["report"]
