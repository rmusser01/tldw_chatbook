"""Schedules destination shell for run timing and recovery."""

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen


logger = logger.bind(module="SchedulesScreen")


class SchedulesScreen(BaseAppScreen):
    """When jobs, watchlists, and workflows run."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "schedules", **kwargs)
        self._current_console_follow_item = None
        self._latest_console_follow_item_id = None

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_latest_console_follow_item()

    @work(exclusive=True, thread=True)
    def _refresh_latest_console_follow_item(self) -> None:
        latest_console_item = self._latest_console_follow_item_from_adapter()
        self.app.call_from_thread(
            self._apply_latest_console_follow_item,
            latest_console_item,
        )

    def _apply_latest_console_follow_item(self, latest_console_item) -> None:
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

    def compose_content(self) -> ComposeResult:
        latest_console_item = self._current_console_follow_item
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        with Vertical(id="schedules-shell"):
            yield Static("Schedules", id="schedules-title", classes="ds-destination-header")
            yield Static(
                "Schedules own when things run.",
                id="schedules-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="schedules-sections", classes="ds-panel"):
                yield Static("Next Run", classes="destination-section")
                yield Static("No scheduler data is available yet.", id="schedules-empty-state")
                yield Static("Paused", classes="destination-section")
                yield Static("Failed", classes="destination-section")
                yield Static("Retry", classes="destination-section")
                if latest_console_item is not None:
                    title = str(getattr(latest_console_item, "title", None) or "Untitled")
                    status = str(getattr(latest_console_item, "status", None) or "unknown")
                    yield Static(
                        Text.from_markup(
                            "Console can follow active schedule run: "
                            f"{escape_markup(title)} ({escape_markup(status)})."
                        ),
                        id="schedules-console-available",
                    )
                    yield Button(
                        Text.from_markup(f"Follow {escape_markup(title)} in Console"),
                        id="schedules-follow-in-console",
                        tooltip="Open the active schedule run in Console.",
                    )
                else:
                    yield Static("Console recovery unavailable", classes="destination-section")
                    yield Static(
                        "No active schedule run is available for Console follow.",
                        id="schedules-console-unavailable",
                    )
                    yield Button(
                        "Console recovery unavailable",
                        id="schedules-follow-in-console",
                        disabled=True,
                        tooltip="Unavailable until Schedules has an active run with Console context.",
                    )

    @on(Button.Pressed, "#schedules-follow-in-console")
    def follow_latest_schedule_run_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        target_id = self._latest_console_follow_item_id
        if not target_id:
            self.app_instance.notify(
                "No active schedule run is available for Console follow.",
                severity="warning",
            )
            return
        open_in_console = getattr(self.app_instance, "open_active_home_item_in_console", None)
        if not callable(open_in_console):
            self.app_instance.notify(
                "Console follow is unavailable for Schedules in this runtime.",
                severity="warning",
            )
            return
        open_in_console(
            target_id=target_id,
            target_route="chat",
        )
