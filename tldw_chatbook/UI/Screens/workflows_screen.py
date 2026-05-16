"""Workflows destination shell for reusable agent procedures."""

from loguru import logger as _logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Rule, Static

from ...Home.dashboard_state import (
    APPROVAL_RUN_STATUS,
    FAILED_RUN_STATUS,
    PAUSED_RUN_STATUS,
    RUNNING_RUN_STATUS,
    categorize_run_status,
)
from ...Widgets.destination_workbench import DestinationModeStrip
from ..Navigation.base_app_screen import BaseAppScreen
from .destination_recovery import DestinationRecoveryState


logger = _logger.bind(module="WorkflowsScreen")

WORKFLOWS_EMPTY_CONSOLE_RECOVERY = DestinationRecoveryState(
    status_label="Select an active run",
    unavailable_what="Console launch for Workflows",
    why="no active workflow run is available",
    next_action="Start or select a workflow run before opening it in Console.",
    recovery_action="Workflows",
    authority_owner="local workflow data",
    stable_selector="workflows-console-unavailable",
    disabled_tooltip="Start or select a workflow run before opening it in Console.",
)


class WorkflowsScreen(BaseAppScreen):
    """Reusable procedures, recipes, dry-runs, and outputs."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "workflows", **kwargs)
        self._current_console_follow_item = None
        self._latest_console_follow_item_id = None
        self._latest_console_context_loaded = False

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_latest_console_context()

    @work(exclusive=True, thread=True)
    def _refresh_latest_console_context(self) -> None:
        latest_console_item = self._latest_console_follow_item_from_adapter()
        self.app.call_from_thread(self._apply_latest_console_context, latest_console_item)

    def _apply_latest_console_context(self, latest_console_item) -> None:
        self._current_console_follow_item = latest_console_item
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        self._latest_console_context_loaded = True
        if self.is_mounted:
            self.refresh(recompose=True)

    def _latest_console_follow_item(self):
        return self._latest_console_follow_item_from_adapter()

    def _latest_console_follow_item_from_adapter(self):
        adapter = getattr(self.app_instance, "home_active_work_adapter", None)
        build_dashboard_input = getattr(adapter, "build_dashboard_input", None)
        if not callable(build_dashboard_input):
            return None
        try:
            providers = getattr(self.app_instance, "providers_models", {}) or {}
            has_recent_work = bool(getattr(self.app_instance, "_screen_states", {}))
            dashboard_input = build_dashboard_input(
                providers_models=providers,
                has_recent_work=has_recent_work,
            )
        except Exception:
            logger.warning(
                "Failed to load Workflows Console launch item from Home active-work adapter.",
                exc_info=True,
            )
            return None
        for item in tuple(getattr(dashboard_input, "active_work_items", ()) or ()):
            source = str(getattr(item, "source", "") or "").strip().lower()
            if (
                source == "workflows"
                and bool(getattr(item, "console_available", False))
                and getattr(item, "item_id", None)
            ):
                return item
        return None

    @staticmethod
    def _column_divider(identifier: str) -> Rule:
        divider = Rule(orientation="vertical", id=identifier)
        divider.add_class("destination-pane-divider")
        return divider

    @staticmethod
    def _status_text(value, *, fallback: str = "unknown") -> str:
        status = str(value or "").strip()
        return status or fallback

    def _inspector_state_summary(self, latest_console_item) -> str:
        if not self._latest_console_context_loaded:
            return "State: loading"
        if latest_console_item is not None:
            return f"State: {self._status_text(getattr(latest_console_item, 'status', None))}"
        return "State: blocked"

    def _approval_summary(self, latest_console_item) -> str:
        if not self._latest_console_context_loaded:
            return "Approvals: loading"
        if latest_console_item is None:
            return "Approvals: no active run"

        status_category = categorize_run_status(getattr(latest_console_item, "status", None))
        if status_category == APPROVAL_RUN_STATUS:
            return "Approvals: pending"
        return "Approvals: none pending"

    def _run_control_summary(self, latest_console_item) -> str:
        if not self._latest_console_context_loaded:
            return "Run control: loading"
        if latest_console_item is None:
            return "Run control: no active run selected"

        status_category = categorize_run_status(getattr(latest_console_item, "status", None))
        if status_category == FAILED_RUN_STATUS:
            return "Run control: retry available"
        if status_category == RUNNING_RUN_STATUS:
            return "Run control: pause available"
        if status_category == PAUSED_RUN_STATUS:
            return "Run control: resume available"
        if status_category == APPROVAL_RUN_STATUS:
            return "Run control: approval required"
        return "Run control: inspect status before acting"

    def _next_action_summary(self, latest_console_item) -> str:
        if latest_console_item is None:
            return "Next action: start or select a workflow run"

        status_category = categorize_run_status(getattr(latest_console_item, "status", None))
        if status_category == FAILED_RUN_STATUS:
            return "Next action: retry or open in Console"
        if status_category == PAUSED_RUN_STATUS:
            return "Next action: resume or open in Console"
        if status_category == APPROVAL_RUN_STATUS:
            return "Next action: review approval before Console follow"
        return "Next action: open in Console"

    def _action_state_label(self, latest_console_item) -> str:
        if latest_console_item is None:
            return "Recovery controls require an active workflow run"

        status_category = categorize_run_status(getattr(latest_console_item, "status", None))
        if status_category == FAILED_RUN_STATUS:
            return "Retry controls are not wired yet"
        if status_category == RUNNING_RUN_STATUS:
            return "Pause controls are not wired yet"
        if status_category == PAUSED_RUN_STATUS:
            return "Resume controls are not wired yet"
        if status_category == APPROVAL_RUN_STATUS:
            return "Approval review controls are not wired yet"
        return "Run controls depend on selected workflow state"

    def compose_content(self) -> ComposeResult:
        latest_console_item = self._current_console_follow_item
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        with Vertical(id="workflows-shell"):
            yield Static(
                "Workflows | Procedures, runs, dry-runs, approvals | Local | Console handoff",
                id="workflows-title",
                classes="ds-destination-header",
            )
            with DestinationModeStrip(id="workflows-mode-strip", classes="destination-filter-strip"):
                yield Static(
                    "Modes: Recipes Inputs Steps Dry run Approvals Outputs",
                    id="workflows-mode-label",
                    classes="destination-section",
                )
            with Horizontal(id="workflows-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="workflows-list-pane", classes="destination-workbench-pane"):
                    yield Static(
                        "Column 1: Procedure Library",
                        classes="destination-pane-title workflows-column-title",
                    )
                    yield Static("Recipes 0", classes="destination-section")
                    yield Static("Inputs 0", classes="destination-section")
                    yield Static("Steps 0", classes="destination-section")
                    yield Static("Dry Run 0", classes="destination-section")
                    yield Static("Approvals 0", classes="destination-section")
                    yield Static("Outputs 0", classes="destination-section")
                    yield Static("No workflow runs are active.", id="workflows-queue-empty")
                yield self._column_divider("workflows-list-detail-divider")
                with Vertical(id="workflows-detail-pane", classes="destination-workbench-pane"):
                    yield Static(
                        "Column 2: Run Detail / Output",
                        classes="destination-pane-title workflows-column-title",
                    )
                    if not self._latest_console_context_loaded:
                        yield Static(
                            "Loading workflow and Console launch context...",
                            id="workflows-loading-state",
                        )
                    elif latest_console_item is not None:
                        title = str(getattr(latest_console_item, "title", None) or "Untitled")
                        status = str(getattr(latest_console_item, "status", None) or "unknown")
                        yield Static("Console launch available", classes="destination-section")
                        yield Static(
                            f"Status: {escape_markup(status)}",
                            id="workflows-run-status",
                        )
                        yield Static(
                            Text.from_markup(
                                "Console can launch active workflow run: "
                                f"{escape_markup(title)} ({escape_markup(status)})."
                            ),
                            id="workflows-console-available",
                        )
                    else:
                        yield Static("Console launch unavailable", classes="destination-section")
                        yield Static("No active workflow run selected", id="workflows-empty-state")
                        yield Static("Select a workflow run or start a procedure to enable controls.")
                        yield Static(
                            WORKFLOWS_EMPTY_CONSOLE_RECOVERY.visible_copy,
                            id=WORKFLOWS_EMPTY_CONSOLE_RECOVERY.stable_selector,
                        )
                yield self._column_divider("workflows-detail-inspector-divider")
                with Vertical(id="workflows-inspector-pane", classes="destination-workbench-pane ds-inspector"):
                    yield Static(
                        "Column 3: Run Inspector",
                        classes="destination-pane-title workflows-column-title",
                    )
                    yield Static(
                        self._inspector_state_summary(latest_console_item),
                        id="workflows-state-summary",
                    )
                    yield Static("Inputs: required before run", id="workflows-inputs-summary")
                    yield Static(self._approval_summary(latest_console_item), id="workflows-approval-summary")
                    yield Static(
                        self._run_control_summary(latest_console_item),
                        id="workflows-run-control-summary",
                    )
                    if latest_console_item is not None:
                        yield Static("Console: ready", id="workflows-console-state")
                        yield Static(self._next_action_summary(latest_console_item), id="workflows-next-action")
                    else:
                        yield Static("Console: blocked", id="workflows-console-state")
                        yield Static(
                            self._next_action_summary(latest_console_item),
                            id="workflows-next-action",
                        )
                    yield Static(
                        self._action_state_label(latest_console_item),
                        id="workflows-action-state-label",
                        classes="destination-section",
                    )
                    yield Button(
                        "Retry run",
                        id="workflows-retry-run",
                        disabled=True,
                        tooltip="Retry this workflow run from Workflows when run-control services are available.",
                    )
                    yield Button(
                        "Pause run",
                        id="workflows-pause-run",
                        disabled=True,
                        tooltip="Pause this workflow run from Workflows when run-control services are available.",
                    )
                    yield Button(
                        "Review approval",
                        id="workflows-review-approval",
                        disabled=True,
                        tooltip="Review this workflow approval from Workflows when approval services are available.",
                    )
                    if not self._latest_console_context_loaded:
                        yield Button(
                            "Console launch unavailable",
                            id="workflows-launch-in-console",
                            disabled=True,
                            tooltip="Launch workflow context after Workflows finishes loading.",
                        )
                    elif latest_console_item is not None:
                        title = str(getattr(latest_console_item, "title", None) or "Untitled")
                        yield Button(
                            Text.from_markup(f"Launch {escape_markup(title)} in Console"),
                            id="workflows-launch-in-console",
                            tooltip="Open the active workflow run in Console.",
                        )
                    else:
                        yield Button(
                            "Console launch unavailable",
                            id="workflows-launch-in-console",
                            disabled=True,
                            tooltip=WORKFLOWS_EMPTY_CONSOLE_RECOVERY.disabled_tooltip,
                        )

    @on(Button.Pressed, "#workflows-launch-in-console")
    def launch_latest_workflow_run_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        target_id = self._latest_console_follow_item_id
        if not target_id:
            self.app_instance.notify(
                WORKFLOWS_EMPTY_CONSOLE_RECOVERY.disabled_tooltip,
                severity="warning",
            )
            return

        open_active_item_in_console = getattr(self.app_instance, "open_active_home_item_in_console", None)
        if not callable(open_active_item_in_console):
            self.app_instance.notify(
                "Console launch is unavailable for Workflows in this runtime.",
                severity="warning",
            )
            return

        open_active_item_in_console(
            target_id=target_id,
            target_route="chat",
        )
