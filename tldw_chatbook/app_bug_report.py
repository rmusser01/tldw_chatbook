"""App-owned state and lazy entry point for copyable bug reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Logging_Config import LogsBufferHandler
    from tldw_chatbook.UI.bug_report_capture import BugReportCollector
    from tldw_chatbook.Widgets.bug_report_dialog import BugReportDialog


class BugReportActions:
    """Keep one report dialog and its outstanding sources per app."""

    _persistent_log_handler: LogsBufferHandler | None = None
    _bug_report_collectors: set[BugReportCollector] | None = None
    _bug_report_dialog: BugReportDialog | None = None
    _bug_report_open_pending = False

    def action_report_bug(self) -> None:
        """Open one memory-only report after the command palette has dismissed."""
        current = self._bug_report_dialog
        if current is not None and current in self.screen_stack:
            if self.screen is current:
                current.request_description_focus()
            return
        if self._bug_report_open_pending:
            return
        self._bug_report_open_pending = True

        def open_report() -> None:
            self._bug_report_open_pending = False
            if getattr(self, "_shutting_down", False):
                return
            from tldw_chatbook.UI.bug_report_capture import capture_bug_report_context
            from tldw_chatbook.Widgets.bug_report_dialog import BugReportDialog

            dialog = BugReportDialog(self, capture_bug_report_context(self))
            self._bug_report_dialog = dialog

            def closed(_result: None) -> None:
                if self._bug_report_dialog is dialog:
                    self._bug_report_dialog = None

            self.push_screen(dialog, closed)

        self.call_after_refresh(open_report)
