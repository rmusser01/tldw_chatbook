"""Presentation-only widgets for one Console Assistant turn."""

from __future__ import annotations

from collections.abc import Iterable
from time import monotonic

from textual import events
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityStatus,
    RawCliPresentation,
)


_RAW_CLI_ELAPSED_STATES = frozenset({"running", "stopping"})
_RAW_CLI_ELAPSED_TICK_SECONDS = 0.1


def raw_cli_status_copy(
    presentation: RawCliPresentation,
    *,
    now: float | None = None,
) -> str:
    """Return explicit, terminal-safe lifecycle copy for a raw command row."""
    labels = {
        "starting": "Starting",
        "running": "Running",
        "stopping": "Stopping…",
        "exited": (
            "Exited"
            if presentation.exit_code is None
            else f"Exited {presentation.exit_code}"
        ),
        "timed_out": "Timed out",
        "cancelled": "Stopped",
        "cleanup_unproven": "Cleanup unproven",
        "failed": "Failed",
    }
    elapsed = presentation.elapsed_seconds
    if (
        presentation.lifecycle_state in _RAW_CLI_ELAPSED_STATES
        and presentation.started_at_monotonic is not None
    ):
        elapsed = max(
            elapsed,
            (monotonic() if now is None else now)
            - presentation.started_at_monotonic,
        )
    copy = f"{labels[presentation.lifecycle_state]} · {elapsed:.1f}s"
    if (
        presentation.cleanup_proven is False
        and presentation.lifecycle_state not in _RAW_CLI_ELAPSED_STATES
        and presentation.lifecycle_state != "cleanup_unproven"
    ):
        copy += " · Cleanup unproven"
    return copy


class ConsoleActivityActivated(Message):
    """Request selection and, when available, disclosure-state toggling."""

    def __init__(self, activity_message_id: str, *, toggle_requested: bool) -> None:
        self.message_id = activity_message_id
        self.toggle_requested = toggle_requested
        super().__init__()

    @property
    def activity_message_id(self) -> str:
        """Backward-compatible alias for the canonical original message ID."""
        return self.message_id


class ConsoleActivityHeader(Horizontal):
    """Focusable literal-text header for one Assistant activity marker."""

    can_focus = True

    def __init__(
        self,
        activity_message_id: str,
        label: str,
        status: ConsoleActivityStatus,
        *,
        expanded: bool = False,
        expandable: bool = False,
        selected: bool = False,
        raw_cli_presentation: RawCliPresentation | None = None,
    ) -> None:
        self.activity_message_id = activity_message_id
        self.label = label
        self.status = status
        self.expanded = expanded
        self.expandable = expandable
        self.selected = selected
        self.raw_cli_presentation = raw_cli_presentation
        self._raw_cli_elapsed_timer: Timer | None = None
        self.label_widget = Static(
            self._label_content(),
            id=f"console-activity-label-{activity_message_id}",
            classes="console-activity-label",
            markup=False,
        )
        self.status_widget = Static(
            self._status_content(),
            id=f"console-activity-status-{activity_message_id}",
            classes="console-activity-status",
            markup=False,
        )
        super().__init__(
            self.label_widget,
            self.status_widget,
            id=f"console-activity-header-{activity_message_id}",
            classes="console-activity-header",
        )
        self._sync_classes()

    def _label_content(self) -> Content:
        """Build the flexible literal label without interpreting its text."""
        chevron = ""
        if self.expandable:
            chevron = "▾ " if self.expanded else "▸ "
        return Content(f"{chevron}{self.label}")

    def _status_content(self) -> Content:
        """Build the fixed terminal-status copy kept separate from the label."""
        if self.raw_cli_presentation is not None:
            return Content(f"· {raw_cli_status_copy(self.raw_cli_presentation)}")
        return Content(f"· {self.status}")

    @property
    def renderable(self) -> Content:
        """Retain the former combined-text inspection seam for callers/tests."""
        return Content(
            f"{self._label_content().plain} {self._status_content().plain}"
        )

    def on_mount(self) -> None:
        """Own the elapsed repaint cadence for this command activity row."""
        self._sync_raw_cli_timer()

    def _tick_raw_cli_elapsed(self) -> None:
        if self.raw_cli_presentation is not None:
            self.status_widget.update(self._status_content())

    def _sync_raw_cli_timer(self) -> None:
        timer = self._raw_cli_elapsed_timer
        active = (
            self.raw_cli_presentation is not None
            and self.raw_cli_presentation.lifecycle_state in _RAW_CLI_ELAPSED_STATES
            and self.raw_cli_presentation.started_at_monotonic is not None
        )
        if active and timer is None:
            self._raw_cli_elapsed_timer = self.set_interval(
                _RAW_CLI_ELAPSED_TICK_SECONDS,
                self._tick_raw_cli_elapsed,
            )
        elif active:
            timer.resume()
        elif timer is not None:
            timer.stop()
            self._raw_cli_elapsed_timer = None

    def _sync_classes(self) -> None:
        self.set_class(self.selected, "console-activity-header-selected")
        self.set_class(self.expanded, "console-activity-header-expanded")
        self.set_class(self.expandable, "console-activity-header-expandable")
        for status in (
            "success",
            "blocked",
            "failed",
            "done",
            "live",
            "stopped",
            "unavailable",
        ):
            self.status_widget.set_class(
                self.status == status,
                f"console-activity-status-{status}",
            )

    def sync_header(
        self,
        label: str,
        status: ConsoleActivityStatus,
        *,
        expanded: bool,
        expandable: bool,
        selected: bool,
        raw_cli_presentation: RawCliPresentation | None = None,
    ) -> None:
        """Project transcript-owned disclosure state onto this header."""
        self.label = label
        self.status = status
        self.expanded = expanded
        self.expandable = expandable
        self.selected = selected
        self.raw_cli_presentation = raw_cli_presentation
        self._sync_classes()
        self.label_widget.update(self._label_content())
        self.status_widget.update(self._status_content())
        self._sync_raw_cli_timer()

    def _activate(self) -> None:
        self.post_message(
            ConsoleActivityActivated(
                self.activity_message_id,
                toggle_requested=self.expandable,
            )
        )

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.focus()
        self._activate()

    def on_key(self, event: events.Key) -> None:
        if event.key not in {"enter", "space"}:
            return
        event.stop()
        event.prevent_default()
        self._activate()


class ConsoleActivityDisclosure(Vertical):
    """Externally controlled disclosure for one structured activity marker."""

    def __init__(
        self,
        activity_message_id: str,
        label: str,
        status: ConsoleActivityStatus,
        *,
        expanded: bool = False,
        selected: bool = False,
        action_widgets: Iterable[Widget] = (),
        detail_widgets: Iterable[Widget] = (),
        detail_available: bool | None = None,
        raw_cli_presentation: RawCliPresentation | None = None,
    ) -> None:
        self.activity_message_id = activity_message_id
        self.label = label
        self.status = status
        self.expanded = expanded
        self.selected = selected
        self.raw_cli_presentation = raw_cli_presentation
        action_children = tuple(action_widgets)
        detail_children = tuple(detail_widgets)
        self._has_actions = bool(action_children)
        self.detail_available = (
            bool(detail_children) if detail_available is None else detail_available
        )
        self._has_detail = bool(detail_children)
        self.header = ConsoleActivityHeader(
            activity_message_id,
            label,
            status,
            expanded=expanded,
            expandable=self.detail_available,
            selected=selected,
            raw_cli_presentation=raw_cli_presentation,
        )
        self.action_stack = Vertical(
            *action_children,
            id=f"console-activity-actions-{activity_message_id}",
            classes="console-activity-action-stack",
        )
        self.detail_stack = Vertical(
            *detail_children,
            id=f"console-activity-detail-{activity_message_id}",
            classes="console-activity-detail-stack",
        )
        super().__init__(
            self.header,
            self.action_stack,
            self.detail_stack,
            id=f"console-activity-disclosure-{activity_message_id}",
            classes="console-activity-disclosure",
        )
        self._sync_visibility()

    def _sync_visibility(self) -> None:
        self.set_class(self.selected, "console-activity-disclosure-selected")
        self.set_class(self.expanded, "console-activity-disclosure-expanded")
        self.action_stack.display = self.selected and self._has_actions
        self.detail_stack.display = self.expanded and self._has_detail

    async def replace_detail_widgets(self, detail_widgets: Iterable[Widget]) -> None:
        """Replace lazy detail children without replacing the disclosure."""
        replacements = tuple(detail_widgets)
        if self.detail_stack.children:
            await self.detail_stack.remove_children()
        if replacements:
            await self.detail_stack.mount(*replacements)
        self._has_detail = bool(replacements)
        self.header.sync_header(
            self.label,
            self.status,
            expanded=self.expanded,
            expandable=self.detail_available,
            selected=self.selected,
        )
        self._sync_visibility()

    def sync_state(self, *, expanded: bool, selected: bool) -> None:
        """Apply transcript-owned selection and expansion state in place."""
        self.expanded = expanded
        self.selected = selected
        self.header.sync_header(
            self.label,
            self.status,
            expanded=expanded,
            expandable=self.detail_available,
            selected=selected,
            raw_cli_presentation=self.raw_cli_presentation,
        )
        self._sync_visibility()

    def sync_activity(
        self,
        label: str,
        status: ConsoleActivityStatus,
        *,
        expanded: bool,
        selected: bool,
        raw_cli_presentation: RawCliPresentation | None = None,
    ) -> None:
        """Apply new structured copy and transcript-owned state in place."""
        self.label = label
        self.status = status
        self.expanded = expanded
        self.selected = selected
        self.raw_cli_presentation = raw_cli_presentation
        self.header.sync_header(
            label,
            status,
            expanded=expanded,
            expandable=self.detail_available,
            selected=selected,
            raw_cli_presentation=raw_cli_presentation,
        )
        self._sync_visibility()


class ConsoleAssistantTurnWidget(Vertical):
    """Stable Assistant header/activity/answer/adjunct presentation shell."""

    def __init__(
        self,
        assistant_message_id: str,
        header_widget: Widget,
        activity_widgets: Iterable[Widget],
        answer_widget: Widget,
        adjunct_widgets: Iterable[Widget] = (),
    ) -> None:
        self.assistant_message_id = assistant_message_id
        self.header_widget = header_widget
        self.answer_widget = answer_widget
        self.activity_stack = Vertical(
            *tuple(activity_widgets),
            id=f"console-assistant-activities-{assistant_message_id}",
            classes="console-assistant-activity-stack",
        )
        self.adjunct_stack = Vertical(
            *tuple(adjunct_widgets),
            id=f"console-assistant-adjuncts-{assistant_message_id}",
            classes="console-assistant-adjunct-stack",
        )
        super().__init__(
            header_widget,
            self.activity_stack,
            answer_widget,
            self.adjunct_stack,
            id=f"console-assistant-turn-{assistant_message_id}",
            classes="console-assistant-turn",
        )

    async def replace_activity_widgets(
        self, activity_widgets: Iterable[Widget]
    ) -> None:
        """Replace only mounted activity children, retaining turn identity."""
        replacements = tuple(activity_widgets)
        if self.activity_stack.children:
            await self.activity_stack.remove_children()
        if replacements:
            await self.activity_stack.mount(*replacements)
