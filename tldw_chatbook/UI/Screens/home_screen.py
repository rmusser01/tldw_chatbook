"""Home dashboard screen for the master shell."""

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Home.dashboard_state import (
    HomeDashboard,
    HomeDashboardInput,
    choose_home_selected_item,
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
        self._current_dashboard_input: HomeDashboardInput | None = None

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
        dashboard_input = self._build_dashboard_input()
        dashboard = summarize_home_dashboard(dashboard_input)
        self._current_dashboard = dashboard
        self._current_dashboard_input = dashboard_input

        sections = {section.section_id: section for section in dashboard.sections}

        def section_text(section_id: str) -> str:
            section = sections.get(section_id)
            return "\n".join(section.lines) if section is not None else ""

        selected_item = self._selected_home_item(dashboard_input)
        selected_item_copy = (
            f"{selected_item.title}\n"
            f"Source: {selected_item.source}\n"
            f"Status: {selected_item.status}\n"
            f"Target: {selected_item.detail_route}"
            if selected_item is not None
            else "No active work selected."
        )
        next_action_copy = f"{dashboard.next_action.label}\n{dashboard.next_action.reason}"

        with Vertical(id="home-dashboard"):
            yield Static("Home", id="home-title", classes="ds-destination-header")
            yield Static(
                "Dashboard, notifications, status, active work, and next actions.",
                id="home-purpose",
                classes="destination-purpose",
            )
            yield Static(
                (
                    "Home | Status, notifications, active work | "
                    f"{'Ready' if dashboard_input.model_ready else 'Blocked'} | Local"
                ),
                id="home-status-row",
                classes="destination-status-row",
            )
            yield Static("Status", id="home-status", classes="ds-panel")
            yield Static(section_text("status"), id="home-status-body")
            yield Static(
                "Scope: All modules | Filter: Needs attention / Running / Recent",
                id="home-scope-filter-row",
                classes="ds-panel",
            )
            with Horizontal(id="home-dashboard-grid", classes="ds-panel"):
                with Vertical(id="home-attention-queue", classes="home-dashboard-region"):
                    yield Static("Attention Queue", id="home-attention", classes="ds-panel")
                    yield Button(
                        dashboard.next_action.label,
                        id="home-primary-action",
                        classes="ds-toolbar",
                    )
                    yield Static(section_text("attention"), id="home-attention-body")
                with Vertical(id="home-active-work-region", classes="home-dashboard-region"):
                    yield Static("Active Work", id="home-active-work", classes="ds-panel")
                    for control in dashboard.controls:
                        yield Button(
                            control.label,
                            id=control.control_id,
                            classes="ds-toolbar",
                        )
                    yield Static(section_text("active_work"), id="home-active-work-body")
                with Vertical(id="home-inspector", classes="home-dashboard-region"):
                    yield Static(
                        "Selected item",
                        id="home-selected-item-title",
                        classes="destination-section",
                    )
                    yield Static(selected_item_copy, id="home-selected-item-body")
            with Vertical(id="home-next-actions-region", classes="ds-panel"):
                yield Static("Next Best Action", id="home-next-best-action", classes="ds-panel")
                yield Static(next_action_copy, id="home-next-best-action-body")
            with Vertical(id="home-recent-work-region", classes="ds-panel"):
                yield Static("Recent Work", id="home-recent-work", classes="ds-panel")
                yield Static(section_text("recent_work"), id="home-recent-work-body")

    def _selected_home_item(self, dashboard_input: HomeDashboardInput):
        return choose_home_selected_item(dashboard_input)

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
