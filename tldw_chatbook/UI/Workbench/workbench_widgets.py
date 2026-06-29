"""Shared Textual Workbench widgets."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.UI.Workbench.workbench_state import (
    RecoveryState,
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchPaneState,
    WorkbenchState,
    WorkbenchStatus,
)


def _safe_id(value: str) -> str:
    """Return a Textual-safe ID segment for state-owned identifiers."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip())
    normalized = normalized.strip("-")
    return normalized or "item"


def _status_label(status: WorkbenchStatus) -> str:
    """Return readable status copy."""
    return status.replace("_", " ").title()


def _record_mount_churn(
    widget: Widget,
    scope: str,
    *,
    mounted: int = 0,
    removed: int = 0,
) -> None:
    """Best-effort mount churn hook used by responsiveness instrumentation."""
    try:
        monitor = getattr(widget.app, "ui_responsiveness_monitor", None)
        if monitor is not None:
            monitor.record_mounts(scope, mounted=mounted, removed=removed)
    except Exception:
        return


def _sync_status_classes(widget: Widget, status: WorkbenchStatus) -> None:
    """Apply one status class while clearing the rest."""
    for candidate in (
        "ready",
        "running",
        "blocked",
        "error",
        "paused",
        "empty",
        "loading",
    ):
        widget.set_class(candidate == status, f"status-{candidate}")


def _sync_density_classes(widget: Widget, density: str) -> None:
    """Apply one density class while clearing the rest."""
    widget.set_class(density == "normal", "density-normal")
    widget.set_class(density == "compact", "density-compact")


class WorkbenchActionRequested(Message):
    """Posted when a Workbench command button is pressed."""

    def __init__(self, action_id: str) -> None:
        super().__init__()
        self.action_id = action_id


class DestinationHeader(Vertical):
    """Stable destination title, subtitle, and status header."""

    def __init__(
        self,
        state: WorkbenchHeaderState,
        **kwargs: Any,
    ) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(
            classes=f"workbench-header ds-destination-header {classes}".strip(),
            **kwargs,
        )
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(
            self.state.title,
            id="workbench-header-title",
            classes="workbench-header-title",
        )
        yield Static(
            self.state.subtitle,
            id="workbench-header-subtitle",
            classes="workbench-header-subtitle",
        )
        yield Static(
            _status_label(self.state.status),
            id="workbench-header-status",
            classes="workbench-header-status ds-status-badge",
        )

    def on_mount(self) -> None:
        self.sync_state(self.state)

    def sync_state(self, state: WorkbenchHeaderState) -> None:
        """Refresh header copy and state classes without remounting."""
        self.state = state
        self.query_one("#workbench-header-title", Static).update(state.title)
        self.query_one("#workbench-header-subtitle", Static).update(state.subtitle)
        self.query_one("#workbench-header-status", Static).update(
            _status_label(state.status)
        )
        self.set_class(not state.subtitle, "has-empty-subtitle")
        _sync_status_classes(self, state.status)
        _sync_density_classes(self, state.density)


class CommandStrip(Horizontal):
    """Stable strip for visible Workbench actions."""

    def __init__(
        self,
        actions: Iterable[WorkbenchAction] = (),
        **kwargs: Any,
    ) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(
            classes=f"workbench-command-strip ds-toolbar {classes}".strip(),
            **kwargs,
        )
        self.actions = tuple(actions)
        self._button_ids_by_action_id: dict[str, str] = {}

    @staticmethod
    def _button_id(action: WorkbenchAction) -> str:
        return f"workbench-action-{_safe_id(action.id)}"

    def _build_button(self, action: WorkbenchAction) -> Button:
        button = Button(
            action.label,
            id=self._button_id(action),
            classes=action.css_classes,
            disabled=action.disabled,
            compact=True,
        )
        button.tooltip = action.tooltip
        setattr(button, "_workbench_action_id", action.id)
        self._button_ids_by_action_id[action.id] = button.id or ""
        return button

    def _sync_button(self, button: Button, action: WorkbenchAction) -> None:
        button.label = action.label
        button.disabled = action.disabled
        button.tooltip = action.tooltip
        setattr(button, "_workbench_action_id", action.id)
        button.set_class(action.primary, "is-primary")
        button.set_class(action.disabled, "is-disabled")
        button.set_class(True, "workbench-action")

    def compose(self) -> ComposeResult:
        for action in self.actions:
            yield self._build_button(action)

    def sync_actions(self, actions: Iterable[WorkbenchAction]) -> None:
        """Refresh action buttons while preserving the strip widget."""
        actions = tuple(actions)
        self.actions = actions
        actions_by_id = {action.id: action for action in actions}
        desired_button_ids = {
            action.id: self._button_id(action)
            for action in actions
        }
        mounted = 0
        removed = 0

        for child in list(self.children):
            action_id = getattr(child, "_workbench_action_id", "")
            if not action_id or action_id not in actions_by_id:
                child.remove()
                removed += 1
                continue
            if isinstance(child, Button):
                self._sync_button(child, actions_by_id[action_id])

        existing_action_ids = {
            getattr(child, "_workbench_action_id", "")
            for child in self.children
        }
        for action in actions:
            if action.id in existing_action_ids:
                continue
            self.mount(self._build_button(action))
            mounted += 1

        self._button_ids_by_action_id = {
            action_id: button_id
            for action_id, button_id in desired_button_ids.items()
        }
        if mounted or removed:
            _record_mount_churn(
                self,
                "workbench-actions",
                mounted=mounted,
                removed=removed,
            )

    @on(Button.Pressed, ".workbench-action")
    def on_workbench_button_pressed(self, event: Button.Pressed) -> None:
        """Route action button presses as stable typed Workbench messages."""
        action_id = getattr(event.button, "_workbench_action_id", "")
        if not action_id:
            return
        event.stop()
        self.post_message(WorkbenchActionRequested(action_id))


class ModeStrip(Horizontal):
    """Stable strip for Workbench mode chips."""

    def __init__(
        self,
        modes: Iterable[WorkbenchMode] = (),
        **kwargs: Any,
    ) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(
            classes=f"workbench-mode-strip {classes}".strip(),
            **kwargs,
        )
        self.modes = tuple(modes)

    @staticmethod
    def _mode_id(mode: WorkbenchMode) -> str:
        return f"workbench-mode-{_safe_id(mode.id)}"

    def _build_mode(self, mode: WorkbenchMode) -> Static:
        label = Static(
            mode.label,
            id=self._mode_id(mode),
            classes=mode.css_classes,
        )
        setattr(label, "_workbench_mode_id", mode.id)
        return label

    def _sync_mode_label(self, label: Static, mode: WorkbenchMode) -> None:
        label.update(mode.label)
        setattr(label, "_workbench_mode_id", mode.id)
        label.set_class(True, "workbench-mode")
        label.set_class(mode.active, "is-active")
        _sync_status_classes(label, mode.status)

    def compose(self) -> ComposeResult:
        for mode in self.modes:
            yield self._build_mode(mode)

    def sync_modes(self, modes: Iterable[WorkbenchMode]) -> None:
        """Refresh mode labels without remounting unchanged mode IDs."""
        modes = tuple(modes)
        self.modes = modes
        modes_by_id = {mode.id: mode for mode in modes}
        mounted = 0
        removed = 0

        for child in list(self.children):
            mode_id = getattr(child, "_workbench_mode_id", "")
            if not mode_id or mode_id not in modes_by_id:
                child.remove()
                removed += 1
                continue
            if isinstance(child, Static):
                self._sync_mode_label(child, modes_by_id[mode_id])

        existing_mode_ids = {
            getattr(child, "_workbench_mode_id", "")
            for child in self.children
        }
        for mode in modes:
            if mode.id in existing_mode_ids:
                continue
            self.mount(self._build_mode(mode))
            mounted += 1

        if mounted or removed:
            _record_mount_churn(
                self,
                "workbench-modes",
                mounted=mounted,
                removed=removed,
            )


class RecoveryCallout(Vertical):
    """Visible recovery copy with an optional action."""

    def __init__(
        self,
        state: RecoveryState | None = None,
        **kwargs: Any,
    ) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(
            classes=(
                f"workbench-recovery workbench-recovery-callout "
                f"ds-recovery-callout {classes}"
            ).strip(),
            **kwargs,
        )
        self.state = state
        self._plain_text = ""

    @property
    def renderable(self) -> Text:
        """Plain recovery text for tests and accessibility-oriented checks."""
        return Text(self._plain_text)

    def compose(self) -> ComposeResult:
        yield Static("", id="workbench-recovery-title", classes="workbench-recovery-title")
        yield Static("", id="workbench-recovery-body", classes="workbench-recovery-body")
        yield Button(
            "",
            id="workbench-recovery-action",
            classes="workbench-recovery-action workbench-action",
            compact=True,
        )

    def on_mount(self) -> None:
        self.sync_state(self.state)

    def sync_state(self, state: RecoveryState | None) -> None:
        """Refresh recovery copy, visibility, and action state."""
        self.state = state
        visible = bool(state and state.visible)
        self.set_class(not visible, "is-hidden")
        self.display = visible
        title = state.title if state else ""
        body = state.body if state else ""
        self._plain_text = "\n".join(part for part in (title, body) if part)
        self.query_one("#workbench-recovery-title", Static).update(title)
        self.query_one("#workbench-recovery-body", Static).update(body)

        action_button = self.query_one("#workbench-recovery-action", Button)
        action = state.action if state else None
        action_button.display = bool(action)
        action_button.disabled = bool(action.disabled) if action else True
        action_button.label = action.label if action else ""
        action_button.tooltip = action.tooltip if action else ""
        setattr(action_button, "_workbench_action_id", action.id if action else "")
        action_button.set_class(bool(action and action.primary), "is-primary")
        action_button.set_class(bool(action and action.disabled), "is-disabled")
        action_button.set_class(True, "workbench-action")

    @on(Button.Pressed, "#workbench-recovery-action")
    def on_recovery_action_pressed(self, event: Button.Pressed) -> None:
        """Emit the recovery action ID when the visible action is pressed."""
        action_id = getattr(event.button, "_workbench_action_id", "")
        if not action_id:
            return
        event.stop()
        self.post_message(WorkbenchActionRequested(action_id))


class WorkbenchPane(Vertical):
    """Stable titled pane frame for Workbench destinations."""

    def __init__(
        self,
        state: WorkbenchPaneState,
        content: Widget | Iterable[Widget] | None = None,
        **kwargs: Any,
    ) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(
            classes=f"workbench-pane ds-panel {classes}".strip(),
            **kwargs,
        )
        self.state = state
        self.content = content
        setattr(self, "_workbench_pane_id", state.id)

    def compose(self) -> ComposeResult:
        yield Static("", id=f"{self.id}-title" if self.id else None, classes="workbench-pane-title")
        yield Static("", id=f"{self.id}-status" if self.id else None, classes="workbench-pane-status ds-status-badge")
        content = self.content
        if isinstance(content, Widget):
            yield content
        elif content is not None:
            yield from content

    def on_mount(self) -> None:
        self.sync_state(self.state)

    def sync_state(self, state: WorkbenchPaneState) -> None:
        """Refresh pane copy and collapsed/status classes."""
        self.state = state
        setattr(self, "_workbench_pane_id", state.id)
        title = self.query_one(".workbench-pane-title", Static)
        status = self.query_one(".workbench-pane-status", Static)
        title.update(state.title)
        status.update(_status_label(state.status))
        self.set_class(state.collapsed, "is-collapsed")
        _sync_status_classes(self, state.status)


class StateBlock(Vertical):
    """Small stable block for route, density, and status instrumentation copy."""

    def __init__(
        self,
        state: WorkbenchState,
        **kwargs: Any,
    ) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(
            classes=f"workbench-state-block {classes}".strip(),
            **kwargs,
        )
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static("", id="workbench-state-route", classes="workbench-state-route")
        yield Static("", id="workbench-state-density", classes="workbench-state-density")

    def on_mount(self) -> None:
        self.sync_state(self.state)

    def sync_state(self, state: WorkbenchState) -> None:
        """Refresh state-block copy and classes."""
        self.state = state
        self.query_one("#workbench-state-route", Static).update(state.route_id)
        self.query_one("#workbench-state-density", Static).update(state.density)
        _sync_density_classes(self, state.density)


class WorkbenchFrame(Vertical):
    """Shared stable frame for Workbench destinations."""

    def __init__(
        self,
        state: WorkbenchState,
        **kwargs: Any,
    ) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(
            classes=f"workbench-frame {classes}".strip(),
            **kwargs,
        )
        self.state = state
        self._route_class: str | None = None

    def compose(self) -> ComposeResult:
        yield DestinationHeader(self.state.header, id="workbench-header")
        yield ModeStrip(self.state.modes, id="workbench-mode-strip")
        yield CommandStrip(self.state.actions, id="workbench-command-strip")
        yield RecoveryCallout(self.state.recovery, id="workbench-recovery")
        yield Vertical(id="workbench-pane-region", classes="workbench-pane-region")
        yield StateBlock(self.state, id="workbench-state-block")

    def on_mount(self) -> None:
        self.sync_state(self.state)

    def _sync_panes(self, panes: tuple[WorkbenchPaneState, ...]) -> None:
        region = self.query_one("#workbench-pane-region", Vertical)
        panes_by_id = {pane.id: pane for pane in panes}
        mounted = 0
        removed = 0

        for child in list(region.children):
            pane_id = getattr(child, "_workbench_pane_id", "")
            if not pane_id or pane_id not in panes_by_id:
                child.remove()
                removed += 1
                continue
            if isinstance(child, WorkbenchPane):
                child.sync_state(panes_by_id[pane_id])

        existing_pane_ids = {
            getattr(child, "_workbench_pane_id", "")
            for child in region.children
        }
        for pane in panes:
            if pane.id in existing_pane_ids:
                continue
            region.mount(
                WorkbenchPane(
                    pane,
                    id=f"workbench-pane-{_safe_id(pane.id)}",
                )
            )
            mounted += 1

        if mounted or removed:
            _record_mount_churn(
                self,
                "workbench-panes",
                mounted=mounted,
                removed=removed,
            )

    def sync_state(self, state: WorkbenchState) -> None:
        """Refresh the Workbench frame from a new immutable state snapshot."""
        self.state = state
        self.query_one("#workbench-header", DestinationHeader).sync_state(state.header)
        self.query_one("#workbench-mode-strip", ModeStrip).sync_modes(state.modes)
        self.query_one("#workbench-command-strip", CommandStrip).sync_actions(
            state.actions
        )
        self.query_one("#workbench-recovery", RecoveryCallout).sync_state(
            state.recovery
        )
        self._sync_panes(state.panes)
        self.query_one("#workbench-state-block", StateBlock).sync_state(state)
        _sync_density_classes(self, state.density)
        next_route_class = f"route-{_safe_id(state.route_id)}" if state.route_id else None
        if self._route_class and self._route_class != next_route_class:
            self.remove_class(self._route_class)
        if next_route_class:
            self.add_class(next_route_class)
        self._route_class = next_route_class
        _sync_status_classes(self, state.header.status)

    def get_direct_child_ids(self) -> tuple[str | None, ...]:
        """Return direct child IDs for stability tests and diagnostics."""
        return tuple(child.id for child in self.children)
