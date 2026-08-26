"""Console-native settings summary widget."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsSummaryState
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)


CONSOLE_SETTINGS_BUTTON_HORIZONTAL_PADDING = 2
CONSOLE_SETTINGS_BUTTON_MIN_WIDTH = 9
CONSOLE_SETTINGS_BUTTON_MAX_WIDTH = 14
CONSOLE_SETTINGS_ROW_HEIGHT = 1


class ConsoleSettingsSummary(RecomposeCaptureGuard, Vertical):
    """Render compact Console session settings rows."""

    _BODY_ROWS = (
        ("console-settings-provider-row", "provider_row"),
        ("console-settings-model-row", "model_row"),
        ("console-settings-context-row", "context_row"),
        ("console-settings-endpoint-row", "endpoint_row"),
        ("console-settings-credential-row", "credential_row"),
        ("console-settings-transport-row", "transport_row"),
        ("console-settings-sampling-row", "sampling_row"),
        ("console-settings-identity-row", "identity_row"),
    )

    def __init__(
        self,
        state: ConsoleSettingsSummaryState,
        *,
        on_reconcile: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self._on_reconcile = on_reconcile
        self.add_class("console-settings-summary")
        self.styles.height = "auto"
        self.styles.min_height = 0

    def sync_state(self, state: ConsoleSettingsSummaryState) -> None:
        """Refresh the summary from a new state snapshot.

        No-ops when ``state`` equals the currently-applied snapshot
        (task-280: the Console 0.2s tick calls this unconditionally).

        Args:
            state: Latest settings-summary display state.

        Returns:
            None.
        """
        if state == self.state:
            return
        self.state = state
        try:
            for widget_id, field_name in self._BODY_ROWS:
                row = self.query_one(f"#{widget_id}", Static)
                value = self._row_text(getattr(state, field_name))
                row.update(value)
                row.display = bool(value)
            button = self.query_one("#console-settings-open", Button)
        except NoMatches:
            self.refresh(recompose=True)
            self.call_after_refresh(self._request_section_reconcile)
            return

        button.label = state.action_label
        button.tooltip = state.action_tooltip
        self._apply_button_sizing(button)
        self.call_after_refresh(self._request_section_reconcile)

    def _apply_button_sizing(self, button: Button) -> None:
        button_width = min(
            max(
                len(self.state.action_label)
                + CONSOLE_SETTINGS_BUTTON_HORIZONTAL_PADDING,
                CONSOLE_SETTINGS_BUTTON_MIN_WIDTH,
            ),
            CONSOLE_SETTINGS_BUTTON_MAX_WIDTH,
        )
        button.styles.width = button_width
        button.styles.min_width = button_width
        button.styles.max_width = button_width
        button.styles.height = CONSOLE_SETTINGS_ROW_HEIGHT
        button.styles.min_height = CONSOLE_SETTINGS_ROW_HEIGHT
        button.styles.max_height = CONSOLE_SETTINGS_ROW_HEIGHT
        button.styles.margin = 0

    @staticmethod
    def _row_text(value: str | None) -> str:
        """Return a Textual-safe settings row label."""
        return value or ""

    def compose(self) -> ComposeResult:
        header = Horizontal(
            id="console-settings-header", classes="console-settings-header"
        )
        header.styles.height = CONSOLE_SETTINGS_ROW_HEIGHT
        header.styles.min_height = CONSOLE_SETTINGS_ROW_HEIGHT
        header.styles.max_height = CONSOLE_SETTINGS_ROW_HEIGHT
        with header:
            title = Static(
                "Session Settings",
                id="console-settings-title",
                classes="destination-section console-settings-title",
            )
            title.styles.width = "1fr"
            title.styles.min_width = 0
            title.styles.height = CONSOLE_SETTINGS_ROW_HEIGHT
            title.styles.min_height = CONSOLE_SETTINGS_ROW_HEIGHT
            title.styles.max_height = CONSOLE_SETTINGS_ROW_HEIGHT
            yield title

            button = Button(
                self.state.action_label,
                id="console-settings-open",
                tooltip=self.state.action_tooltip,
                compact=True,
            )
            self._apply_button_sizing(button)
            yield button
        rows = []
        for widget_id, field_name in self._BODY_ROWS:
            value = self._row_text(getattr(self.state, field_name))
            row = Static(
                value,
                id=widget_id,
                classes="console-settings-row",
                markup=False,
            )
            row.display = bool(value)
            rows.append(row)
        yield ConsoleBoundedSection(*rows, section_id="session-settings")

    def _request_section_reconcile(self) -> None:
        """Settle local demand before invalidating the Inspector owner."""

        sections = list(self.query(ConsoleBoundedSection))
        if sections:
            sections[0].request_reconcile()
        if self._on_reconcile is not None:
            self._on_reconcile()
