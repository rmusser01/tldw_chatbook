"""Schedules destination shell for run timing and recovery."""

from collections.abc import Mapping
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Rule, Static

from ...Widgets.destination_workbench import DestinationModeStrip
from ..Navigation.base_app_screen import BaseAppScreen
from .destination_recovery import DestinationRecoveryState


logger = logger.bind(module="SchedulesScreen")

SCHEDULES_EMPTY_CONSOLE_RECOVERY = DestinationRecoveryState(
    status_label="Select an active run",
    unavailable_what="Console follow for Schedules",
    why="no active schedule run or reading digest output is available",
    next_action="Start or select a schedule run before opening it in Console.",
    recovery_action="Schedules",
    authority_owner="local schedule data",
    stable_selector="schedules-console-unavailable",
    disabled_tooltip="Start or select a schedule run before opening it in Console.",
)


class SchedulesScreen(BaseAppScreen):
    """When jobs, watchlists, and workflows run."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "schedules", **kwargs)
        self._current_console_follow_item = None
        self._latest_console_follow_item_id = None
        self._latest_console_launch_kwargs: dict[str, Any] | None = None
        self._latest_console_context_loaded = False

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_latest_console_context()

    @work(exclusive=True, thread=True)
    def _refresh_latest_console_context(self) -> None:
        latest_console_item = self._latest_console_follow_item_from_adapter()
        latest_console_launch = None
        if latest_console_item is None:
            latest_console_launch = self._latest_reading_digest_console_launch()
        self.app.call_from_thread(
            self._apply_latest_console_context,
            latest_console_item,
            latest_console_launch,
        )

    def _apply_latest_console_context(self, latest_console_item, latest_console_launch) -> None:
        self._current_console_follow_item = latest_console_item
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        self._latest_console_launch_kwargs = latest_console_launch
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
                "Failed to load Schedules Console follow item from Home active-work adapter.",
                exc_info=True,
            )
            return None
        for item in tuple(getattr(dashboard_input, "active_work_items", ()) or ()):
            if (
                getattr(item, "source", None) == "Schedules"
                and bool(getattr(item, "console_available", False))
                and getattr(item, "item_id", None)
            ):
                return item
        return None

    def _latest_reading_digest_console_launch(self) -> dict[str, Any] | None:
        service = getattr(self.app_instance, "local_media_reading_service", None)
        list_outputs = getattr(service, "list_reading_digest_outputs", None)
        if not callable(list_outputs):
            return None
        try:
            output_listing = list_outputs(schedule_id=None, limit=1, offset=0)
        except Exception:
            logger.warning(
                "Failed to load Schedules Console launch context from local reading digest outputs.",
                exc_info=True,
            )
            return None
        items = output_listing.get("items") if isinstance(output_listing, Mapping) else None
        latest_output = next(iter(tuple(items or ())), None)
        if not isinstance(latest_output, Mapping):
            return None

        output_id = latest_output.get("output_id") or latest_output.get("id")
        if output_id in (None, ""):
            return None

        metadata = latest_output.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        schedule_name = str(
            metadata.get("schedule_name")
            or latest_output.get("schedule_name")
            or latest_output.get("schedule_id")
            or ""
        ).strip()
        title = str(latest_output.get("title") or schedule_name or "Reading digest output").strip()
        item_count = metadata.get("item_count", latest_output.get("item_count"))
        payload = {
            "target_id": f"local:reading_digest_output:{output_id}",
            "output_id": output_id,
            "schedule_id": latest_output.get("schedule_id"),
            "schedule_name": schedule_name or None,
            "download_url": latest_output.get("download_url") or latest_output.get("storage_path"),
            "created_at": latest_output.get("created_at"),
            "item_count": item_count,
        }
        return {
            "source": "schedules",
            "title": title,
            "payload": payload,
            "status": "ready",
            "recovery": "Review this reading digest output from Schedules or return to Library.",
            "action_label": "Open schedule output",
        }

    @staticmethod
    def _column_divider(identifier: str) -> Rule:
        divider = Rule(orientation="vertical", id=identifier)
        divider.add_class("destination-pane-divider")
        return divider

    @staticmethod
    def _status_text(value: Any, *, fallback: str = "unknown") -> str:
        status = str(value or "").strip()
        return status or fallback

    def _inspector_state_summary(self, latest_console_item) -> str:
        if not self._latest_console_context_loaded:
            return "State: loading"
        if latest_console_item is not None:
            return f"State: {self._status_text(getattr(latest_console_item, 'status', None))}"
        if self._latest_console_launch_kwargs is not None:
            return "State: digest output available"
        return "State: blocked"

    def _retry_backoff_summary(self, latest_console_item) -> str:
        if not self._latest_console_context_loaded:
            return "Retry/backoff: loading"
        if latest_console_item is None:
            if self._latest_console_launch_kwargs is not None:
                return "Retry/backoff: not applicable to digest output"
            return "Retry/backoff: no active run selected"

        status = self._status_text(getattr(latest_console_item, "status", None)).lower()
        if status in {"failed", "error", "errored", "cancelled", "canceled"}:
            return "Retry/backoff: retry available from Schedules"
        if status in {"queued", "pending", "scheduled", "running", "active"}:
            return "Retry/backoff: not retrying"
        if status == "paused":
            return "Retry/backoff: paused"
        return "Retry/backoff: status unknown"

    def _run_control_summary(self, latest_console_item) -> str:
        if not self._latest_console_context_loaded:
            return "Run control: loading"
        if latest_console_item is None:
            if self._latest_console_launch_kwargs is not None:
                return "Run control: digest output is read-only"
            return "Run control: no active run selected"

        status = self._status_text(getattr(latest_console_item, "status", None)).lower()
        if status in {"failed", "error", "errored", "cancelled", "canceled"}:
            return "Run control: retry available"
        if status in {"running", "active", "queued", "pending", "scheduled"}:
            return "Run control: pause available"
        if status == "paused":
            return "Run control: resume available"
        if status in {"approval_required", "pending_approval"}:
            return "Run control: approval required"
        return "Run control: inspect status before acting"

    def _next_action_summary(self, latest_console_item) -> str:
        if latest_console_item is None:
            if self._latest_console_launch_kwargs is not None:
                return "Next action: open digest output in Console"
            return "Next action: start or select a schedule run"

        status = self._status_text(getattr(latest_console_item, "status", None)).lower()
        if status in {"failed", "error", "errored", "cancelled", "canceled"}:
            return "Next action: retry or open in Console"
        if status == "paused":
            return "Next action: resume or open in Console"
        if status in {"approval_required", "pending_approval"}:
            return "Next action: review approval before Console follow"
        return "Next action: open in Console"

    def _action_state_label(self, latest_console_item) -> str:
        if latest_console_item is None:
            if self._latest_console_launch_kwargs is not None:
                return "Digest output is read-only"
            return "Recovery controls require an active schedule run"

        status = self._status_text(getattr(latest_console_item, "status", None)).lower()
        if status in {"failed", "error", "errored", "cancelled", "canceled"}:
            return "Retry controls are not wired yet"
        if status in {"running", "active", "queued", "pending", "scheduled"}:
            return "Pause controls are not wired yet"
        if status == "paused":
            return "Resume controls are not wired yet"
        if status in {"approval_required", "pending_approval"}:
            return "Approval review controls are not wired yet"
        return "Run controls depend on selected schedule state"

    def compose_content(self) -> ComposeResult:
        latest_console_item = self._current_console_follow_item
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        with Vertical(id="schedules-shell"):
            yield Static(
                "Schedules | Jobs, digests, timers, retries | Local | Console handoff",
                id="schedules-title",
                classes="ds-destination-header",
            )
            with DestinationModeStrip(id="schedules-filter-strip", classes="destination-filter-strip"):
                yield Static(
                    "Filters: Next run Paused Failed Retry History",
                    id="schedules-filter-label",
                    classes="destination-section",
                )
            with Horizontal(id="schedules-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="schedules-list-pane", classes="destination-workbench-pane"):
                    yield Static("Column 1: Schedule Queue", classes="destination-pane-title schedules-column-title")
                    yield Static("Next Run 0", classes="destination-section")
                    yield Static("Paused 0", classes="destination-section")
                    yield Static("Failed 0", classes="destination-section")
                    yield Static("Retry 0", classes="destination-section")
                    yield Static("History 0", id="schedules-history-row", classes="destination-section")
                    yield Static("No scheduled runs are active.", id="schedules-queue-empty")
                yield self._column_divider("schedules-list-detail-divider")
                with Vertical(id="schedules-detail-pane", classes="destination-workbench-pane"):
                    yield Static("Column 2: Run Detail / Output", classes="destination-pane-title schedules-column-title")
                    if not self._latest_console_context_loaded:
                        yield Static(
                            "Loading schedule and Console follow context...",
                            id="schedules-loading-state",
                        )
                    elif latest_console_item is not None:
                        title = str(getattr(latest_console_item, "title", None) or "Untitled")
                        status = str(getattr(latest_console_item, "status", None) or "unknown")
                        yield Static("Console launch available", classes="destination-section")
                        yield Static(
                            f"Status: {escape_markup(status)}",
                            id="schedules-run-status",
                        )
                        yield Static(
                            Text.from_markup(
                                "Console can follow active schedule run: "
                                f"{escape_markup(title)} ({escape_markup(status)})."
                            ),
                            id="schedules-console-available",
                        )
                    elif self._latest_console_launch_kwargs is not None:
                        title = str(self._latest_console_launch_kwargs["title"])
                        yield Static("Console launch available", classes="destination-section")
                        yield Static(
                            Text.from_markup(
                                "Console can launch latest reading digest output: "
                                f"{escape_markup(title)}."
                            ),
                            id="schedules-console-available",
                        )
                    else:
                        yield Static("No active schedule run selected", id="schedules-empty-state")
                        yield Static("Select a run from the queue or create a scheduled job to enable controls.")
                        yield Static("Console recovery unavailable", classes="destination-section")
                        yield Static(
                            SCHEDULES_EMPTY_CONSOLE_RECOVERY.visible_copy,
                            id=SCHEDULES_EMPTY_CONSOLE_RECOVERY.stable_selector,
                        )
                yield self._column_divider("schedules-detail-inspector-divider")
                with Vertical(id="schedules-inspector-pane", classes="destination-workbench-pane ds-inspector"):
                    yield Static("Column 3: Status Inspector", classes="destination-pane-title schedules-column-title")
                    yield Static(
                        self._inspector_state_summary(latest_console_item),
                        id="schedules-state-summary",
                    )
                    yield Static(
                        self._retry_backoff_summary(latest_console_item),
                        id="schedules-retry-summary",
                    )
                    yield Static(
                        self._run_control_summary(latest_console_item),
                        id="schedules-run-control-summary",
                    )
                    if latest_console_item is not None or self._latest_console_launch_kwargs is not None:
                        yield Static("Console: ready", id="schedules-console-state")
                        yield Static(self._next_action_summary(latest_console_item), id="schedules-next-action")
                    else:
                        yield Static("Console: blocked", id="schedules-console-state")
                        yield Static(
                            self._next_action_summary(latest_console_item),
                            id="schedules-next-action",
                        )
                    yield Static(
                        self._action_state_label(latest_console_item),
                        id="schedules-action-state-label",
                        classes="destination-section",
                    )
                    yield Button(
                        "Retry run",
                        id="schedules-retry-run",
                        disabled=True,
                        tooltip="Retry this schedule run from Schedules when run-control services are available.",
                    )
                    yield Button(
                        "Pause run",
                        id="schedules-pause-run",
                        disabled=True,
                        tooltip="Pause this schedule run from Schedules when run-control services are available.",
                    )
                    yield Button(
                        "Review approval",
                        id="schedules-review-approval",
                        disabled=True,
                        tooltip="Review this schedule approval from Schedules when approval services are available.",
                    )
                    if not self._latest_console_context_loaded:
                        yield Button(
                            "Console recovery unavailable",
                            id="schedules-follow-in-console",
                            disabled=True,
                            tooltip="Stage schedule context after Schedules finishes loading.",
                        )
                    elif latest_console_item is not None:
                        title = str(getattr(latest_console_item, "title", None) or "Untitled")
                        yield Button(
                            Text.from_markup(f"Follow {escape_markup(title)} in Console"),
                            id="schedules-follow-in-console",
                            tooltip="Open the active schedule run in Console.",
                        )
                    elif self._latest_console_launch_kwargs is not None:
                        title = str(self._latest_console_launch_kwargs["title"])
                        yield Button(
                            Text.from_markup(f"Launch {escape_markup(title)} in Console"),
                            id="schedules-follow-in-console",
                            tooltip="Open the latest local reading digest output in Console.",
                        )
                    else:
                        yield Button(
                            "Console recovery unavailable",
                            id="schedules-follow-in-console",
                            disabled=True,
                            tooltip=SCHEDULES_EMPTY_CONSOLE_RECOVERY.disabled_tooltip,
                        )

    @on(Button.Pressed, "#schedules-follow-in-console")
    def follow_latest_schedule_run_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        target_id = self._latest_console_follow_item_id
        if target_id:
            open_active_item_in_console = getattr(self.app_instance, "open_active_home_item_in_console", None)
            if not callable(open_active_item_in_console):
                self.app_instance.notify(
                    "Console follow is unavailable for Schedules in this runtime.",
                    severity="warning",
                )
                return
            open_active_item_in_console(
                target_id=target_id,
                target_route="chat",
            )
            return

        launch_kwargs = self._latest_console_launch_kwargs
        if launch_kwargs is not None:
            open_in_console = getattr(self.app_instance, "open_console_for_live_work", None)
            if not callable(open_in_console):
                self.app_instance.notify(
                    "Console launch is unavailable for Schedules in this runtime.",
                    severity="warning",
                )
                return
            open_in_console(**launch_kwargs)
            return

        self.app_instance.notify(
            SCHEDULES_EMPTY_CONSOLE_RECOVERY.disabled_tooltip,
            severity="warning",
        )
