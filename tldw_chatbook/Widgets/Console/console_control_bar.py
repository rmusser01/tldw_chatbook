"""Console-native provider/model/source readiness controls."""

from __future__ import annotations

from typing import Any, Callable

from textual.app import ComposeResult
from textual import on
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchAction
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar


CONSOLE_CONTROL_BAR_HEIGHT = 2
TOP_ACTION_IDS = {
    "new-tab",
    "settings",
    "attach-context",
    "run-library-rag",
    "save-chatbook",
    "help",
}
CONSOLE_CONTROL_ACTION_WIDGET_IDS = {
    "new-tab": "console-control-new-tab",
    "settings": "console-control-settings",
    "attach-context": "console-control-attach-context",
    "run-library-rag": "console-control-run-library-rag",
    "save-chatbook": "console-control-save-chatbook",
    "help": "console-control-help",
}
FALLBACK_ACTIONS = (
    WorkbenchAction(
        id="settings",
        label="Settings",
        tooltip="Configure provider, model, tools, and generation",
    ),
    WorkbenchAction(
        id="attach-context",
        label="Attach",
        tooltip="Stage Library or workspace context",
    ),
    WorkbenchAction(
        id="run-library-rag",
        label="Library RAG",
        tooltip="Search Library evidence before sending",
    ),
    WorkbenchAction(
        id="help",
        label="Help",
        tooltip="Show visible Console actions and shortcuts",
    ),
)


def _summary_line(state: ConsoleControlState) -> str:
    return " | ".join(
        (
            state.provider_label,
            state.model_label,
            state.persona_label,
            state.rag_label,
            state.sources_label,
            state.tools_label,
            state.approvals_label,
        )
    )


class ConsoleControlBar(Vertical):
    """Visible Console control strip outside the transcript region.

    The widget renders Console-owned provider, model, persona, RAG, source,
    tools, and approval labels plus the compact provider/model controls. It
    exposes `sync_state()` so `ChatScreen` can refresh labels after the user
    changes provider/model state through existing sidebar or compact controls.
    """

    def __init__(
        self,
        state: ConsoleControlState,
        app_instance: Any,
        *,
        actions: tuple[WorkbenchAction, ...] = (),
        on_sidebar_toggle_requested: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Console control bar.

        Args:
            state: Display-state snapshot for Console readiness labels.
            app_instance: Main application instance used by `CompactModelBar`.
            actions: Current Workbench action state for top Console actions.
            on_sidebar_toggle_requested: Optional callback for routing compact
                sidebar-toggle requests to the embedded chat settings sidebar.
            **kwargs: Additional Textual widget arguments.
        """
        super().__init__(**kwargs)
        self.state = state
        self.app_instance = app_instance
        self.actions = tuple(actions)
        self.on_sidebar_toggle_requested = on_sidebar_toggle_requested
        self.styles.height = CONSOLE_CONTROL_BAR_HEIGHT
        self.styles.min_height = CONSOLE_CONTROL_BAR_HEIGHT
        self.styles.max_height = CONSOLE_CONTROL_BAR_HEIGHT

    @staticmethod
    def _compatibility_layout_widget(widget: Any) -> Any:
        """Keep pre-Workbench selectors mounted until parity tests migrate.

        Guarded by Console decomposition regressions that still query the
        legacy control-bar child IDs while visible state moves to ModeStrip.
        """
        widget.styles.display = "none"
        widget.styles.width = 0
        widget.styles.min_width = 0
        widget.styles.height = 0
        widget.styles.min_height = 0
        return widget

    def sync_state(
        self,
        state: ConsoleControlState,
        actions: tuple[WorkbenchAction, ...] | None = None,
    ) -> None:
        """Refresh visible label widgets from a new display-state snapshot.

        Args:
            state: Updated display-state snapshot to render.
            actions: Optional Workbench actions to sync into the top strip.
        """
        if actions is None and state == self.state:
            return
        if actions is not None and state == self.state and tuple(actions) == self.actions:
            return
        self.state = state
        if actions is not None:
            self.sync_actions(actions)
        label_values = {
            "#console-control-status-line": _summary_line(state),
            "#console-provider-label": state.provider_label,
            "#console-model-label": state.model_label,
            "#console-persona-label": state.persona_label,
            "#console-rag-label": state.rag_label,
            "#console-sources-label": state.sources_label,
            "#console-tools-label": state.tools_label,
            "#console-approvals-label": state.approvals_label,
            "#console-provider-chip": state.provider_label,
            "#console-model-chip": state.model_label,
            "#console-persona-chip": state.persona_label,
            "#console-rag-chip": state.rag_label,
            "#console-sources-chip": state.sources_label,
            "#console-tools-chip": state.tools_label,
            "#console-approvals-chip": state.approvals_label,
        }
        for selector, label in label_values.items():
            try:
                self.query_one(selector, Static).update(label)
            except NoMatches:
                continue
        chip_emphasis = {
            "#console-sources-chip": state.sources_active,
            "#console-tools-chip": state.tools_active,
            "#console-approvals-chip": state.approvals_active,
        }
        for selector, active in chip_emphasis.items():
            try:
                chip = self.query_one(selector, Static)
            except NoMatches:
                continue
            chip.set_class(not active, "console-chip-dim")
            chip.set_class(active, "console-chip-alert")

    @staticmethod
    def _chip(label: str, *, id: str, emphasis: bool | None = None) -> Static:
        classes = "console-control-chip"
        if emphasis is False:
            classes += " console-chip-dim"
        elif emphasis is True:
            classes += " console-chip-alert"
        return Static(label, id=id, classes=classes)

    @staticmethod
    def _action_widget_id(action: WorkbenchAction) -> str:
        return CONSOLE_CONTROL_ACTION_WIDGET_IDS.get(
            action.id,
            f"console-control-{action.id}",
        )

    def _visible_actions(self) -> tuple[WorkbenchAction, ...]:
        actions = self.actions if self.actions else FALLBACK_ACTIONS
        return tuple(action for action in actions if action.id in TOP_ACTION_IDS)

    @classmethod
    def _action(cls, action: WorkbenchAction) -> Button:
        button = Button(
            action.label,
            id=cls._action_widget_id(action),
            classes=f"console-control-action {action.css_classes}".strip(),
            disabled=action.disabled,
            compact=True,
            tooltip=action.tooltip or None,
        )
        setattr(button, "_workbench_action_id", action.id)
        return button

    @staticmethod
    def _sync_action_button(button: Button, action: WorkbenchAction) -> None:
        button.label = action.label
        button.disabled = action.disabled
        button.tooltip = action.tooltip or None
        setattr(button, "_workbench_action_id", action.id)
        button.set_class(True, "console-control-action")
        button.set_class(True, "workbench-action")
        button.set_class(action.primary, "is-primary")
        button.set_class(action.disabled, "is-disabled")

    def sync_actions(self, actions: tuple[WorkbenchAction, ...]) -> None:
        """Refresh top Console actions from current Workbench state."""
        actions = tuple(actions)
        if actions == self.actions:
            return
        self.actions = actions
        try:
            row = self.query_one("#console-control-action-row", Horizontal)
        except NoMatches:
            return

        visible_actions = self._visible_actions()
        desired_ids = tuple(action.id for action in visible_actions)
        current_ids = tuple(
            getattr(child, "_workbench_action_id", "")
            for child in row.children
            if getattr(child, "_workbench_action_id", "")
        )
        if current_ids != desired_ids:
            for child in list(row.children):
                if getattr(child, "_workbench_action_id", ""):
                    child.remove()
            new_actions = tuple(self._action(action) for action in visible_actions)
            if new_actions:
                row.mount(*new_actions)
            return

        actions_by_id = {action.id: action for action in visible_actions}
        for child in row.children:
            action_id = getattr(child, "_workbench_action_id", "")
            action = actions_by_id.get(action_id)
            if action is not None and isinstance(child, Button):
                self._sync_action_button(child, action)

    def compose(self) -> ComposeResult:
        with Horizontal(id="console-control-chip-row", classes="console-control-chip-row"):
            yield self._chip(self.state.provider_label, id="console-provider-chip")
            yield self._chip(self.state.model_label, id="console-model-chip")
            yield self._chip(self.state.persona_label, id="console-persona-chip")
            yield self._chip(self.state.rag_label, id="console-rag-chip")
            yield self._chip(
                self.state.sources_label,
                id="console-sources-chip",
                emphasis=self.state.sources_active,
            )
            yield self._chip(
                self.state.tools_label,
                id="console-tools-chip",
                emphasis=self.state.tools_active,
            )
            yield self._chip(
                self.state.approvals_label,
                id="console-approvals-chip",
                emphasis=self.state.approvals_active,
            )
        with Horizontal(id="console-control-action-row", classes="console-control-action-row"):
            for action in self._visible_actions():
                yield self._action(action)
        yield self._compatibility_layout_widget(Static(
            _summary_line(self.state),
            id="console-control-status-line",
            classes="console-control-summary-line",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.provider_label,
            id="console-provider-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.model_label,
            id="console-model-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.persona_label,
            id="console-persona-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.rag_label,
            id="console-rag-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.sources_label,
            id="console-sources-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.tools_label,
            id="console-tools-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(Static(
            self.state.approvals_label,
            id="console-approvals-label",
            classes="console-control-label console-hidden-control",
        ))
        yield self._compatibility_layout_widget(CompactModelBar(
            self.app_instance,
            on_sidebar_toggle_requested=self.on_sidebar_toggle_requested,
            id="console-compact-model-bar",
            classes="console-compact-model-bar console-hidden-control",
        ))

    @on(Button.Pressed, ".console-control-action")
    def on_console_control_action_pressed(self, event: Button.Pressed) -> None:
        """Route compact visible control actions through the Workbench seam."""
        action_id = getattr(event.button, "_workbench_action_id", "")
        if not action_id:
            return
        event.stop()
        self.post_message(WorkbenchActionRequested(action_id))
