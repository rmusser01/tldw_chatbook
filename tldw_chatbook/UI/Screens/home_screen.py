"""Home dashboard screen for the master shell."""

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Home.dashboard_state import (
    HomeDashboard,
    HomeDashboardInput,
    summarize_home_dashboard,
)

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen


HOME_CONTROL_METHODS = {
    "home-approve": "approve_active_home_item",
    "home-reject": "reject_active_home_item",
    "home-pause": "pause_active_home_item",
    "home-resume": "resume_active_home_item",
    "home-retry": "retry_active_home_item",
    "home-open-details": "open_active_home_item_details",
    "home-open-in-console": "open_active_home_item_in_console",
    "home-open-chatbook-details": "open_active_home_item_details",
    "home-open-chatbook-in-console": "open_active_home_item_in_console",
}

HOME_CONTROL_METHODS_WITH_TARGET_ROUTE = {
    "home-open-details",
    "home-open-in-console",
    "home-open-chatbook-details",
    "home-open-chatbook-in-console",
}


class HomeScreen(BaseAppScreen):
    """Dashboard, notifications, readiness, and next-best action surface."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "home", **kwargs)
        self._current_dashboard: HomeDashboard | None = None

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_home_chatbook_artifact_snapshot()

    @work(exclusive=True, thread=True)
    def _refresh_home_chatbook_artifact_snapshot(self) -> None:
        adapter = getattr(self.app_instance, "home_active_work_adapter", None)
        refresh_snapshot = getattr(adapter, "refresh_chatbook_artifact_snapshot", None)
        if not callable(refresh_snapshot):
            return
        refresh_snapshot()
        self.app.call_from_thread(self._refresh_after_chatbook_artifact_snapshot)

    def _refresh_after_chatbook_artifact_snapshot(self) -> None:
        if self.is_mounted:
            self.refresh(recompose=True)

    def _build_dashboard_input(self) -> HomeDashboardInput:
        test_override = getattr(self.app_instance, "_home_dashboard_test_input", None)
        if test_override is not None:
            return test_override

        providers = getattr(self.app_instance, "providers_models", {}) or {}
        has_recent_work = bool(getattr(self.app_instance, "_screen_states", {}))
        return self.app_instance.home_active_work_adapter.build_dashboard_input(
            providers_models=providers,
            has_recent_work=has_recent_work,
        )

    def compose_content(self) -> ComposeResult:
        """Compose the Home dashboard route."""
        dashboard = summarize_home_dashboard(self._build_dashboard_input())
        self._current_dashboard = dashboard

        with Vertical(id="home-dashboard"):
            yield Static("Home", id="home-title", classes="ds-destination-header")
            yield Static(
                "Dashboard, notifications, status, active work, and next actions.",
                id="home-purpose",
                classes="destination-purpose",
            )
            for section in dashboard.sections:
                section_id = section.section_id.replace("_", "-")
                yield Static(section.title, id=f"home-{section_id}", classes="ds-panel")
                yield Static("\n".join(section.lines), id=f"home-{section_id}-body")

            yield Button(dashboard.next_action.label, id="home-primary-action")

            for control in dashboard.controls:
                yield Button(control.label, id=control.control_id, classes="ds-toolbar")

    @on(Button.Pressed)
    def handle_home_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        dashboard = self._current_dashboard
        if not button_id or dashboard is None:
            return

        if button_id == "home-primary-action":
            prepare = getattr(self.app_instance, "prepare_home_primary_action", None)
            if callable(prepare):
                prepare(dashboard.next_action)
            self.post_message(NavigateToScreen(dashboard.next_action.target_route))
            return

        control = next((item for item in dashboard.controls if item.control_id == button_id), None)
        if control is None:
            return

        method_name = HOME_CONTROL_METHODS.get(control.control_id)
        method = getattr(self.app_instance, method_name, None) if method_name else None
        if callable(method):
            kwargs = {}
            if control.target_id is not None:
                kwargs["target_id"] = control.target_id
            if control.control_id in HOME_CONTROL_METHODS_WITH_TARGET_ROUTE:
                kwargs["target_route"] = control.target_route
            if kwargs:
                method(**kwargs)
            else:
                method()
        else:
            self.app_instance.notify(
                f"{control.label} is not connected yet.",
                severity="warning",
            )
