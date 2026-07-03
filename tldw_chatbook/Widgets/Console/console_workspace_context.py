"""Console-native workspace context tray."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserGroup,
    ConsoleConversationBrowserRow,
    ConsoleConversationBrowserSection,
    ConsoleConversationBrowserState,
)
from tldw_chatbook.Workspaces.display_state import (
    CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT,
    ConsoleWorkspaceContextState,
    ConsoleWorkspaceConversationSectionState,
)


_STATUS_LABELS = {
    "workspace-thread": "workspace",
    "workspace": "workspace",
    "active": "active",
    "open": "open",
}
_STATUS_DETAIL_LABELS = {
    "workspace-thread": "saved workspace",
    "workspace": "saved workspace",
    "active": "active session",
    "open": "open session",
}
_MAX_CONVERSATION_ROW_TITLE = 20
_CONVERSATION_BROWSER_HEADER_HEIGHT = 1
_CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT = 1


class ConsoleWorkspaceStatusPair(Horizontal):
    """Render workspace authority metadata as a structured status row.

    Attributes:
        label: User-facing row label.
        value: User-facing row value.
        label_id: Textual widget id for the label cell.
        value_id: Textual widget id for the value cell.
    """

    def __init__(
        self,
        label: str,
        value: str,
        *,
        label_id: str,
        value_id: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the label/value status row.

        Args:
            label: User-facing row label.
            value: User-facing row value.
            label_id: Textual widget id for the label cell.
            value_id: Textual widget id for the value cell.
            **kwargs: Additional keyword arguments passed to ``Horizontal``.
        """
        super().__init__(classes="console-workspace-status-pair", **kwargs)
        self.label = label
        self.value = value
        self.label_id = label_id
        self.value_id = value_id
        self.styles.height = "auto"
        self.styles.min_height = 1

    def compose(self) -> ComposeResult:
        """Render the row as queryable Textual widgets.

        Returns:
            ComposeResult containing the label and value widgets.
        """
        label_widget = Static(
            self.label,
            id=self.label_id,
            classes="console-workspace-status-label",
            markup=False,
        )
        label_widget.styles.width = 10
        label_widget.styles.min_width = 10
        yield label_widget

        value_widget = Static(
            self.value,
            id=self.value_id,
            classes="console-workspace-status-value",
            markup=False,
        )
        value_widget.styles.width = "1fr"
        value_widget.styles.min_width = 0
        yield value_widget


class ConsoleWorkspaceContextTray(Vertical):
    """Render workspace selection, conversation scope, and recovery copy."""

    def __init__(
        self,
        state: ConsoleWorkspaceContextState,
        *,
        show_heading: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.show_heading = show_heading
        self.styles.height = "auto"
        self.styles.min_height = 0

    def on_mount(self) -> None:
        """Fit the tray to content after Textual has measured child widgets.

        Returns:
            None.
        """

        self.call_after_refresh(self._fit_height_to_content)

    def on_resize(self, event: Any) -> None:
        """Refit wrapped status rows when the rail width changes.

        Args:
            event: Textual resize event emitted for this tray.

        Returns:
            None.
        """

        self.call_after_refresh(self._fit_height_to_content)

    def sync_state(self, state: ConsoleWorkspaceContextState) -> None:
        """Refresh the mounted workspace context tray from new display state.

        Args:
            state: Latest workspace context display state.

        Returns:
            None.
        """
        self.state = state
        self.styles.min_height = 0
        scroll_parent = self._nearest_scroll_parent()
        parent_scroll_y = getattr(scroll_parent, "scroll_y", None)
        restore_scroll_y = int(parent_scroll_y) if parent_scroll_y is not None else None
        self.refresh(recompose=True)
        if self.is_mounted:
            self._schedule_recomposed_content_fit(restore_scroll_y=restore_scroll_y)

    def _nearest_scroll_parent(self) -> Any | None:
        """Return the nearest ancestor that owns vertical scrolling.

        Returns:
            Scrollable ancestor widget, or ``None`` when unavailable.
        """

        for ancestor in self.ancestors:
            if getattr(ancestor, "id", None) == "console-left-rail-body":
                return ancestor
        for ancestor in self.ancestors:
            if (
                getattr(ancestor, "max_scroll_y", 0) > 0
                and callable(getattr(ancestor, "scroll_to", None))
            ):
                return ancestor
        return None

    def _schedule_recomposed_content_fit(
        self,
        *,
        restore_scroll_y: int | None = None,
    ) -> None:
        """Schedule bounded fit passes after recomposing tray content.

        Args:
            restore_scroll_y: Parent rail scroll offset to restore after fitting.

        Returns:
            None.
        """

        def fit_and_restore_scroll() -> None:
            self._fit_height_to_content()
            self._restore_parent_scroll(restore_scroll_y)

        # Recompose settles child layout over more than one message turn in
        # scrolled rails; a fixed follow-up pass avoids a layout feedback loop.
        self.call_later(fit_and_restore_scroll)
        self.call_later(lambda: self.call_later(fit_and_restore_scroll))
        if restore_scroll_y is not None:
            self.set_timer(
                0.01,
                lambda: self._restore_parent_scroll(restore_scroll_y),
                name="console-workspace-context-scroll-restore",
            )

    def _restore_parent_scroll(self, scroll_y: int | None) -> None:
        """Restore the parent rail scroll position after a deferred fit pass.

        Args:
            scroll_y: Parent rail scroll offset to restore.

        Returns:
            None.
        """

        if scroll_y is None:
            return
        scroll_parent = self._nearest_scroll_parent()
        scroll_to = getattr(scroll_parent, "scroll_to", None)
        if callable(scroll_to):
            scroll_to(y=max(0, scroll_y), animate=False)

    def _fit_height_to_content(self) -> None:
        """Expose the full tray content height to the parent scroll container.

        Returns:
            None.
        """

        region = getattr(self, "region", None)
        if region is None or region.height <= 0:
            return

        content_top = 0
        content_bottom = 1
        for child in self.children:
            if not getattr(child, "display", True):
                continue
            child_virtual = getattr(child, "virtual_region", None)
            if child_virtual is None or child_virtual.height <= 0:
                continue
            content_bottom = max(
                content_bottom,
                child_virtual.y + child_virtual.height,
            )

        target_height = max(1, content_bottom - content_top)
        # When the tray is nested inside a collapsible left-rail section body,
        # size to natural content only. Stretching to the shared scroll parent
        # would push sibling sections (Context/Model/Details) out of the rail
        # viewport.
        if not self._is_inside_rail_section_body():
            scroll_parent = self._nearest_scroll_parent()
            parent_region = getattr(scroll_parent, "region", None)
            if parent_region is not None and parent_region.height > 0:
                target_height = max(target_height, int(parent_region.height))

        if int(region.height) != target_height:
            self.styles.height = target_height

    def _is_inside_rail_section_body(self) -> bool:
        """Return whether this tray is nested in a collapsible rail section body.

        Returns:
            True when an ancestor carries the ``console-rail-section-body``
            class, meaning the tray shares the left-rail scroll container with
            sibling sections and must not stretch to fill it.
        """

        for ancestor in self.ancestors:
            has_class = getattr(ancestor, "has_class", None)
            if callable(has_class) and has_class("console-rail-section-body"):
                return True
        return False

    @staticmethod
    def _static(text: str, *, id: str, classes: str = "") -> Static:
        """Create a plain Static widget with markup disabled.

        Args:
            text: Text to display.
            id: Textual widget id.
            classes: Optional CSS classes.

        Returns:
            Static widget configured for plain text.
        """
        return Static(str(text), id=id, classes=classes, markup=False)

    def _conversation_section(self) -> ConsoleWorkspaceConversationSectionState:
        """Return section state, adapting legacy row-only snapshots.

        Returns:
            Conversation section state for the currently mounted tray state.
        """
        section = self.state.conversation_section
        if section is not None:
            return section
        selected = next(
            (row for row in self.state.conversation_rows if row.selected),
            None,
        )
        selected_summary = (
            (
                f"{self._conversation_title(selected.title)} - "
                f"{self._conversation_detail_status(selected.status) or 'conversation'}"
            )
            if selected is not None
            else "No active conversation."
        )
        return ConsoleWorkspaceConversationSectionState(
            workspace_id="",
            collapsed=False,
            query="",
            selected_summary=selected_summary,
            rows=self.state.conversation_rows,
            workspace_total_count=len(self.state.conversation_rows),
            result_total_count=None,
            status_copy="",
            empty_copy=self.state.conversation_empty_copy,
            search_enabled=False,
            new_conversation_enabled=self.state.new_conversation_enabled,
        )

    @staticmethod
    def _conversation_count_title(
        section: ConsoleWorkspaceConversationSectionState,
    ) -> str:
        """Return the Conversations heading with a stable workspace count.

        Args:
            section: Conversation section state to summarize.

        Returns:
            Heading text containing the workspace conversation count.
        """
        count = section.workspace_total_count
        if count is None:
            count = len(section.rows)
        return f"Conversations ({count})"

    @staticmethod
    def _conversation_button(
        text: str,
        *,
        id: str,
        conversation_id: str,
        tooltip_label: str | None = None,
        selected: bool = False,
    ) -> Button:
        button = Button(
            Text(str(text)),
            id=id,
            classes="console-workspace-conversation-row",
            compact=True,
        )
        button.conversation_id = conversation_id
        button.tooltip = f"Switch to {tooltip_label or text.lstrip('> ').strip()}"
        button.set_class(selected, "console-workspace-conversation-row-selected")
        button.styles.height = 2
        button.styles.min_height = 2
        return button

    @staticmethod
    def _legacy_conversation_list_height(
        section: ConsoleWorkspaceConversationSectionState,
    ) -> int:
        """Return the full content height for legacy conversation rows.

        Args:
            section: Legacy conversation section state.

        Returns:
            Height needed to render every row without internal scrolling.
        """

        if not section.rows:
            return _CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT
        return max(
            _CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT,
            len(section.rows) * CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT,
        )

    @staticmethod
    def _conversation_browser_list_height(
        browser: ConsoleConversationBrowserState,
    ) -> int:
        """Return the full content height for the grouped browser rows.

        Args:
            browser: Grouped conversation browser state.

        Returns:
            Height needed to render every visible browser row.
        """

        height = 0
        for section in browser.sections:
            height += _CONVERSATION_BROWSER_HEADER_HEIGHT
            if section.collapsed:
                continue
            if section.groups:
                for group in section.groups:
                    height += _CONVERSATION_BROWSER_HEADER_HEIGHT
                    if group.collapsed:
                        continue
                    if group.rows:
                        height += (
                            len(group.rows)
                            * CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT
                        )
                    elif group.empty_copy:
                        height += _CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT
                continue
            if section.rows:
                height += (
                    len(section.rows) * CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT
                )
            elif section.empty_copy:
                height += _CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT
        return max(_CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT, height)

    def compose(self) -> ComposeResult:
        if self.show_heading:
            yield self._static(
                self.state.heading,
                id="console-workspace-context-title",
                classes="destination-section",
            )
        yield self._static(
            self._workspace_selector_label(),
            id="console-active-workspace",
            classes="console-workspace-status-row console-workspace-selector-row",
        )
        if self.state.change_workspace_enabled:
            yield Button(
                "Change workspace",
                id="console-change-workspace",
                classes="console-workspace-action",
                compact=True,
            )
            if self.state.change_workspace_recovery:
                yield self._static(
                    self.state.change_workspace_recovery,
                    id="console-change-workspace-recovery",
                    classes="console-workspace-recovery",
                )
        if self.state.recovery_copy:
            yield self._static(
                self.state.recovery_copy,
                id="console-workspace-recovery",
                classes="console-workspace-recovery",
            )

        browser = self.state.conversation_browser
        if browser is not None:
            yield from self._compose_conversation_browser(browser)
        else:
            yield from self._compose_legacy_conversation_section()

    def _compose_legacy_conversation_section(self) -> ComposeResult:
        """Render the transitional active-workspace conversation section."""

        section = self._conversation_section()
        section_controls_enabled = self.state.conversation_section is not None
        with Horizontal(
            id="console-workspace-conversations-header",
            classes="console-workspace-conversations-header",
        ):
            title = self._static(
                self._conversation_count_title(section),
                id="console-workspace-conversations-title",
                classes="destination-section",
            )
            title.styles.width = "1fr"
            yield title
            toggle_label = "+" if section.collapsed else "-"
            toggle = Button(
                toggle_label,
                id="console-workspace-conversations-toggle",
                classes=(
                    "console-workspace-action "
                    "console-workspace-conversations-toggle"
                ),
                compact=True,
                disabled=not section_controls_enabled,
            )
            toggle.tooltip = (
                "Expand Conversations"
                if section.collapsed
                else "Collapse Conversations"
            )
            toggle.styles.width = 3
            toggle.styles.min_width = 3
            yield toggle
        yield self._static(
            section.selected_summary or "No active conversation.",
            id="console-workspace-selected-conversation",
            classes="console-workspace-selected-conversation",
        )
        if not section.collapsed:
            with Horizontal(
                id="console-workspace-conversation-search-row",
                classes="console-workspace-conversation-search-row",
            ):
                search_input = Input(
                    value=section.query,
                    placeholder="Search workspace conversations",
                    id="console-workspace-conversation-search",
                    classes="console-workspace-conversation-search",
                    disabled=not section.search_enabled,
                )
                search_input.styles.width = "1fr"
                yield search_input
                clear_button = Button(
                    "Clear",
                    id="console-workspace-conversation-search-clear",
                    classes=(
                        "console-workspace-action "
                        "console-workspace-conversation-search-clear"
                    ),
                    compact=True,
                    disabled=(
                        not section.search_enabled
                        or not bool(str(section.query or "").strip())
                    ),
                )
                clear_button.tooltip = "Clear conversation search"
                yield clear_button
            if section.status_copy:
                yield self._static(
                    section.status_copy,
                    id="console-workspace-conversation-search-status",
                    classes="console-workspace-empty-copy",
                )
            if section.error_copy:
                yield self._static(
                    section.error_copy,
                    id="console-workspace-conversation-search-error",
                    classes="console-workspace-recovery",
                )
            conversation_list = Vertical(id="console-workspace-conversations")
            conversation_list.styles.height = self._legacy_conversation_list_height(
                section
            )
            conversation_list.styles.min_height = 0
            with conversation_list:
                if section.rows:
                    for index, row in enumerate(section.rows):
                        marker = "> " if row.selected else "  "
                        title = self._conversation_title(row.title)
                        visible_title = self._conversation_visible_title(title)
                        status = self._conversation_status(row.status)
                        detail = self._conversation_detail_status(row.status)
                        status_suffix = f" [{status}]" if status else ""
                        secondary = detail or "conversation"
                        yield self._conversation_button(
                            f"{marker}{visible_title}\n  {secondary}",
                            id=f"console-workspace-conversation-{index}",
                            conversation_id=row.conversation_id,
                            tooltip_label=f"{title}{status_suffix}",
                            selected=row.selected,
                        )
                else:
                    yield self._static(
                        section.empty_copy or self.state.conversation_empty_copy,
                        id="console-workspace-empty-conversations",
                        classes="console-workspace-empty-copy",
                    )
            if section.new_conversation_enabled:
                yield Button(
                    "New conversation",
                    id="console-new-workspace-conversation",
                    classes="console-workspace-action",
                    compact=True,
                )
                if self.state.new_conversation_recovery:
                    yield self._static(
                        self.state.new_conversation_recovery,
                        id="console-new-workspace-conversation-recovery",
                        classes="console-workspace-recovery",
                    )

    def _compose_conversation_browser(
        self,
        browser: ConsoleConversationBrowserState,
    ) -> ComposeResult:
        """Render the grouped all-workspaces conversation browser."""

        with Horizontal(
            id="console-workspace-conversations-header",
            classes="console-workspace-conversations-header",
        ):
            title = self._static(
                "Conversations",
                id="console-workspace-conversations-title",
                classes="destination-section",
            )
            title.styles.width = "1fr"
            yield title
        yield self._static(
            browser.selected_summary or "No active conversation.",
            id="console-workspace-selected-conversation",
            classes="console-workspace-selected-conversation",
        )
        with Horizontal(
            id="console-workspace-conversation-search-row",
            classes="console-workspace-conversation-search-row",
        ):
            search_input = Input(
                value=browser.query,
                placeholder="Search conversations",
                id="console-workspace-conversation-search",
                classes="console-workspace-conversation-search",
            )
            search_input.styles.width = "1fr"
            yield search_input
            clear_button = Button(
                "Clear",
                id="console-workspace-conversation-search-clear",
                classes=(
                    "console-workspace-action "
                    "console-workspace-conversation-search-clear"
                ),
                compact=True,
                disabled=not bool(str(browser.query or "").strip()),
            )
            clear_button.tooltip = "Clear conversation search"
            yield clear_button
        if browser.status_copy:
            yield self._static(
                browser.status_copy,
                id="console-workspace-conversation-search-status",
                classes="console-workspace-empty-copy",
            )
        if browser.error_copy:
            yield self._static(
                browser.error_copy,
                id="console-workspace-conversation-search-error",
                classes="console-workspace-recovery",
            )
        if not browser.marks_available:
            yield self._static(
                "Local stars unavailable",
                id="console-conversation-browser-marks-unavailable",
                classes="console-workspace-recovery",
            )

        row_index = 0
        conversation_list = Vertical(id="console-workspace-conversations")
        conversation_list.styles.height = self._conversation_browser_list_height(browser)
        conversation_list.styles.min_height = 0
        with conversation_list:
            for section in browser.sections:
                yield from self._compose_conversation_browser_section_header(section)
                if section.collapsed:
                    continue
                if section.groups:
                    for group_index, group in enumerate(section.groups):
                        yield from self._compose_conversation_browser_group_header(
                            group,
                            group_index,
                        )
                        if group.collapsed:
                            continue
                        if group.rows:
                            for row in group.rows:
                                yield from self._compose_conversation_browser_row(
                                    row,
                                    row_index,
                                    marks_available=browser.marks_available,
                                )
                                row_index += 1
                        elif group.empty_copy:
                            yield self._static(
                                group.empty_copy,
                                id=(
                                    "console-conversation-browser-group-empty-"
                                    f"{group_index}"
                                ),
                                classes="console-workspace-empty-copy",
                            )
                    if not section.groups and section.empty_copy:
                        yield self._static(
                            section.empty_copy,
                            id=(
                                "console-conversation-browser-"
                                f"{section.section_id}-empty"
                            ),
                            classes="console-workspace-empty-copy",
                        )
                    continue
                if section.rows:
                    for row in section.rows:
                        yield from self._compose_conversation_browser_row(
                            row,
                            row_index,
                            marks_available=browser.marks_available,
                        )
                        row_index += 1
                elif section.empty_copy:
                    yield self._static(
                        section.empty_copy,
                        id=f"console-conversation-browser-{section.section_id}-empty",
                        classes="console-workspace-empty-copy",
                    )

    def _compose_conversation_browser_section_header(
        self,
        section: ConsoleConversationBrowserSection,
    ) -> ComposeResult:
        """Render one grouped browser section header."""

        with Horizontal(classes="console-conversation-browser-section-header"):
            title = self._static(
                section.label,
                id=f"console-conversation-browser-{section.section_id}-title",
                classes="console-conversation-browser-section-title",
            )
            yield title
            toggle = Button(
                "+" if section.collapsed else "-",
                id=f"console-conversation-browser-section-toggle-{section.section_id}",
                classes=(
                    "console-workspace-action "
                    "console-workspace-conversations-toggle"
                ),
                compact=True,
            )
            toggle.group_id = f"section:{section.section_id}"
            toggle.tooltip = (
                f"Expand {section.label}"
                if section.collapsed
                else f"Collapse {section.label}"
            )
            yield toggle

    def _compose_conversation_browser_group_header(
        self,
        group: ConsoleConversationBrowserGroup,
        index: int,
    ) -> ComposeResult:
        """Render one workspace group header."""

        with Horizontal(classes="console-conversation-browser-group-header"):
            title = self._static(
                group.label,
                id=f"console-conversation-browser-group-title-{index}",
                classes="console-conversation-browser-group-title",
            )
            yield title
            toggle = Button(
                "+" if group.collapsed else "-",
                id=f"console-conversation-browser-group-toggle-{index}",
                classes=(
                    "console-workspace-action "
                    "console-workspace-conversations-toggle"
                ),
                compact=True,
            )
            toggle.group_id = group.group_id
            toggle.tooltip = (
                f"Expand {group.label}"
                if group.collapsed
                else f"Collapse {group.label}"
            )
            yield toggle

    def _compose_conversation_browser_row(
        self,
        row: ConsoleConversationBrowserRow,
        index: int,
        *,
        marks_available: bool,
    ) -> ComposeResult:
        """Render one grouped browser row plus its local star control."""

        with Horizontal(classes="console-conversation-browser-row-line"):
            title = self._conversation_title(row.title)
            visible_title = self._conversation_visible_title(title)
            status = self._conversation_status(row.status)
            detail = self._conversation_detail_status(row.status)
            secondary_parts = [
                part
                for part in (row.workspace_label, detail, row.updated_label)
                if str(part or "").strip()
            ]
            secondary = " - ".join(secondary_parts) or "conversation"
            marker = "> " if row.selected else "  "
            status_suffix = f" [{status}]" if status else ""
            row_button = Button(
                Text(f"{marker}{visible_title}\n  {secondary}"),
                id=f"console-workspace-conversation-{index}",
                classes="console-workspace-conversation-row",
                compact=True,
            )
            row_button.tooltip = f"Switch to {title}{status_suffix}"
            row_button.row_key = row.row_key
            row_button.conversation_id = row.conversation_id or row.row_key
            row_button.native_session_id = row.native_session_id
            row_button.scope_type = row.scope_type
            row_button.workspace_id = row.workspace_id
            row_button.set_class(row.selected, "console-workspace-conversation-row-selected")
            row_button.styles.height = 2
            row_button.styles.min_height = 2
            row_button.styles.width = "1fr"
            row_button.styles.min_width = 0
            yield row_button

            star_disabled = not marks_available or not row.star_enabled
            star_button = Button(
                "*" if row.starred else ".",
                id=f"console-conversation-star-{index}",
                classes="console-workspace-action console-conversation-star",
                compact=True,
                disabled=star_disabled,
            )
            if not marks_available:
                star_button.tooltip = "Local stars unavailable"
            elif not row.star_enabled:
                star_button.tooltip = "Send or save this conversation before starring."
            else:
                star_button.tooltip = (
                    "Unstar conversation" if row.starred else "Star conversation"
                )
            star_button.row_key = row.row_key
            star_button.conversation_id = row.conversation_id
            star_button.starred = row.starred
            yield star_button

    def _workspace_selector_label(self) -> str:
        """Return the visible active-workspace selector affordance."""
        workspace_label = self.state.workspace_label
        if workspace_label.startswith("Workspace: "):
            workspace_label = workspace_label.removeprefix("Workspace: ").strip()
        return workspace_label

    @staticmethod
    def _conversation_title(title: str) -> str:
        """Return a readable conversation label."""
        return str(title).strip() or "Untitled conversation"

    @staticmethod
    def _conversation_visible_title(title: str) -> str:
        """Return a rail-safe visible title that does not clip in narrow panes."""
        readable = ConsoleWorkspaceContextTray._conversation_title(title)
        if len(readable) <= _MAX_CONVERSATION_ROW_TITLE:
            return readable
        return f"{readable[: _MAX_CONVERSATION_ROW_TITLE - 3].rstrip()}..."

    @staticmethod
    def _conversation_status(status: str) -> str:
        """Return a short user-facing conversation status badge."""
        normalized = str(status or "").strip().lower()
        if not normalized:
            return ""
        return _STATUS_LABELS.get(normalized, normalized.replace("-", " "))

    @staticmethod
    def _conversation_detail_status(status: str) -> str:
        """Return second-line row metadata for row disambiguation."""
        normalized = str(status or "").strip().lower()
        if not normalized:
            return ""
        return _STATUS_DETAIL_LABELS.get(normalized, normalized.replace("-", " "))
