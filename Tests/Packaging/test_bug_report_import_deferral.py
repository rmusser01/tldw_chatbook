"""Reporting adds no Console/trajectory import to ordinary startup."""

from Tests.Packaging.test_exchange_export_trajectory_deferral import (
    _run_isolated_python,
)


def test_report_imports_and_no_console_open_keep_heavy_readers_deferred(tmp_path):
    result = _run_isolated_python(
        tmp_path,
        """
import asyncio
import sys
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Logs_Window import LogsWindow
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
for name in ("tldw_chatbook.Utils.bug_report_capture", "tldw_chatbook.Widgets.bug_report_dialog"):
    assert name not in sys.modules, name
before = set(sys.modules)
from textual.app import App
from tldw_chatbook.UI.Navigation.screen_state_store import RuntimeIdentity
class Host(App):
    action_report_bug = TldwCli.action_report_bug
    console_runtime = None
    current_tab = "logs"
    def _current_runtime_identity(self):
        return RuntimeIdentity("local")
async def check():
    host = Host()
    async with host.run_test() as pilot:
        host.action_report_bug()
        async with asyncio.timeout(10):
            while type(host.screen).__name__ != "BugReportDialog" or not host.screen.query("#bug-report-description"):
                await pilot.pause(0.02)
        await host.workers.wait_for_complete()
        assert host.console_runtime is None
        assert type(host.screen).__name__ == "BugReportDialog"
asyncio.run(check())
for name in ("tldw_chatbook.Chat.console_trace_native_reader", "tldw_chatbook.Chat.trajectory_export", "tldw_chatbook.Chat.trajectory_import", "torch", "transformers"):
    assert name not in set(sys.modules) - before, name
""",
    )
    assert result.returncode == 0, result.stderr[-6000:]
