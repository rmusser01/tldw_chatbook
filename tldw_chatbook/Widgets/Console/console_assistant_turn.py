"""Presentation-only widgets for one Console Assistant turn."""

from __future__ import annotations

from collections.abc import Iterable

from textual import events
from textual.containers import Vertical
from textual.content import Content
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import ConsoleActivityStatus


class ConsoleActivityActivated(Message):
    """Request selection and, when available, disclosure-state toggling."""

    def __init__(self, activity_message_id: str, *, toggle_requested: bool) -> None:
        self.activity_message_id = activity_message_id
        self.toggle_requested = toggle_requested
        super().__init__()


class ConsoleActivityHeader(Static):
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
    ) -> None:
        self.activity_message_id = activity_message_id
        self.label = label
        self.status = status
        self.expanded = expanded
        self.expandable = expandable
        self.selected = selected
        super().__init__(
            self._content(),
            id=f"console-activity-header-{activity_message_id}",
            classes="console-activity-header",
            markup=False,
        )
        self._sync_classes()

    def _content(self) -> Content:
        """Build semantic copy without interpreting the structured label."""
        chevron = ""
        if self.expandable:
            chevron = "▾ " if self.expanded else "▸ "
        return Content(f"{chevron}{self.label} · {self.status}")

    def _sync_classes(self) -> None:
        self.set_class(self.selected, "console-activity-header-selected")
        self.set_class(self.expanded, "console-activity-header-expanded")
        self.set_class(self.expandable, "console-activity-header-expandable")

    def sync_header(
        self,
        label: str,
        status: ConsoleActivityStatus,
        *,
        expanded: bool,
        expandable: bool,
        selected: bool,
    ) -> None:
        """Project transcript-owned disclosure state onto this header."""
        self.label = label
        self.status = status
        self.expanded = expanded
        self.expandable = expandable
        self.selected = selected
        self._sync_classes()
        self.update(self._content())

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
    ) -> None:
        self.activity_message_id = activity_message_id
        self.label = label
        self.status = status
        self.expanded = expanded
        self.selected = selected
        action_children = tuple(action_widgets)
        detail_children = tuple(detail_widgets)
        self._has_actions = bool(action_children)
        self._has_detail = bool(detail_children)
        self.header = ConsoleActivityHeader(
            activity_message_id,
            label,
            status,
            expanded=expanded,
            expandable=self._has_detail,
            selected=selected,
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

    def sync_state(self, *, expanded: bool, selected: bool) -> None:
        """Apply transcript-owned selection and expansion state in place."""
        self.expanded = expanded
        self.selected = selected
        self.header.sync_header(
            self.label,
            self.status,
            expanded=expanded,
            expandable=self._has_detail,
            selected=selected,
        )
        self._sync_visibility()

    def sync_activity(
        self,
        label: str,
        status: ConsoleActivityStatus,
        *,
        expanded: bool,
        selected: bool,
    ) -> None:
        """Apply new structured copy and transcript-owned state in place."""
        self.label = label
        self.status = status
        self.expanded = expanded
        self.selected = selected
        self.header.sync_header(
            label,
            status,
            expanded=expanded,
            expandable=self._has_detail,
            selected=selected,
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
