"""Workflows destination shell for reusable agent procedures."""

from loguru import logger as _logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

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

    def compose_content(self) -> ComposeResult:
        latest_console_item = self._current_console_follow_item
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        with Vertical(id="workflows-shell"):
            yield Static("Workflows", id="workflows-title", classes="ds-destination-header")
            yield Static(
                "Workflows own what procedure runs.",
                id="workflows-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="workflows-sections", classes="ds-panel"):
                yield Static("Recipes", classes="destination-section")
                yield Static("Inputs", classes="destination-section")
                yield Static("Steps", classes="destination-section")
                yield Static("Dry Run", classes="destination-section")
                yield Static("Approvals", classes="destination-section")
                yield Static("Outputs", classes="destination-section")
                if latest_console_item is not None:
                    title = str(getattr(latest_console_item, "title", None) or "Untitled")
                    status = str(getattr(latest_console_item, "status", None) or "unknown")
                    yield Static("Console launch available", classes="destination-section")
                    yield Static(
                        Text.from_markup(
                            "Console can launch active workflow run: "
                            f"{escape_markup(title)} ({escape_markup(status)})."
                        ),
                        id="workflows-console-available",
                    )
                    yield Button(
                        Text.from_markup(f"Launch {escape_markup(title)} in Console"),
                        id="workflows-launch-in-console",
                        tooltip="Open the active workflow run in Console.",
                    )
                else:
                    yield Static("Console launch unavailable", classes="destination-section")
                    yield Static(
                        WORKFLOWS_EMPTY_CONSOLE_RECOVERY.visible_copy,
                        id=WORKFLOWS_EMPTY_CONSOLE_RECOVERY.stable_selector,
                    )
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
