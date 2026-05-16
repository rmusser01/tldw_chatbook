"""Home dashboard screen for the master shell."""

from collections.abc import Callable

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Constants import TAB_LLM, get_tab_display_label
from tldw_chatbook.Home.dashboard_state import (
    HomeDashboard,
    HomeDashboardInput,
    choose_home_selected_item,
    summarize_home_dashboard,
)

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from ..Navigation.shell_destinations import get_shell_destination, resolve_shell_route


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

HOME_ROUTE_LABEL_OVERRIDES = {
    "llm": get_tab_display_label(TAB_LLM),
    TAB_LLM: get_tab_display_label(TAB_LLM),
    "search-rag": "Search/RAG",
}


def _home_route_label(route: str) -> str:
    route = route.strip()
    if route in HOME_ROUTE_LABEL_OVERRIDES:
        return HOME_ROUTE_LABEL_OVERRIDES[route]

    resolved = resolve_shell_route(route)
    try:
        return get_shell_destination(resolved.destination_id).accessible_label
    except KeyError:
        return route.replace("_", " ").replace("-", " ").title()


def _home_runtime_status_label(state: HomeDashboardInput) -> str:
    source = str(state.runtime_source or "local").strip().lower()
    if source != "server":
        return "Local"
    server_label = str(state.server_label or "").strip()
    return f"Server: {server_label}" if server_label else "Server"


class HomeActionButton(Button):
    """Home button that emits press events even when app chrome hides layout."""

    def __init__(self, *args, fallback_press: Callable[[], None] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fallback_press = fallback_press

    def press(self):
        if not self.display and self._fallback_press is not None:
            self._fallback_press()
            return self
        return super().press()


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
        selected_item_title = "Selected item" if selected_item is not None else "Selected action"
        selected_item_copy = (
            f"{selected_item.title}\n"
            f"Source: {selected_item.source}\n"
            f"Status: {selected_item.status}\n"
            f"Target: {selected_item.detail_route}"
            if selected_item is not None
            else (
                f"{dashboard.next_action.label}\n"
                f"{dashboard.next_action.reason}\n"
                f"Destination: {_home_route_label(dashboard.next_action.target_route)}\n"
                f"Enter: Open {_home_route_label(dashboard.next_action.target_route)}\n"
                "Ctrl+P: Search commands"
            )
        )
        next_action_copy = section_text("next_best_action")

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
                    f"{'Ready' if dashboard_input.model_ready else 'Blocked'} | "
                    f"{_home_runtime_status_label(dashboard_input)}"
                ),
                id="home-status-row",
                classes="destination-status-row",
            )
            yield Static(section_text("status"), id="home-status", classes="destination-status-row")
            yield Static(
                "Scope: All modules | Filter: Needs attention / Running / Recent",
                id="home-scope-filter-row",
                classes="ds-panel",
            )
            yield Static(
                "Keys: Enter open selected | Tab switch pane | Ctrl+P command palette",
                id="home-action-hints",
                classes="destination-status-row",
            )
            with Horizontal(id="home-dashboard-grid", classes="ds-panel destination-workbench"):
                with Vertical(
                    id="home-attention-queue",
                    classes="home-dashboard-region home-narrow-pane destination-workbench-pane",
                ):
                    yield Static("Attention Queue", id="home-attention", classes="destination-section")
                    yield HomeActionButton(
                        dashboard.next_action.label,
                        id="home-primary-action",
                        classes="ds-toolbar",
                        fallback_press=self._activate_home_primary_action,
                    )
                    yield Static(section_text("attention"), id="home-attention-body")
                yield Static("", id="home-attention-active-divider", classes="home-pane-divider")
                with Vertical(
                    id="home-active-work-region",
                    classes=(
                        "home-dashboard-region home-wide-pane "
                        "home-system-status-region destination-workbench-pane"
                    ),
                ):
                    yield Static("System Status", id="home-system-status", classes="destination-section")
                    for control in dashboard.controls:
                        yield HomeActionButton(
                            control.label,
                            id=control.control_id,
                            classes="ds-toolbar",
                            fallback_press=lambda control_id=control.control_id: (
                                self._activate_home_control(control_id)
                            ),
                        )
                    yield Static(section_text("system_status"), id="home-system-status-body")
                    yield Static(section_text("active_work"), id="home-active-work-body")
                yield Static("", id="home-active-inspector-divider", classes="home-pane-divider")
                with Vertical(
                    id="home-inspector",
                    classes="home-dashboard-region destination-workbench-pane ds-inspector",
                ):
                    yield Static(
                        selected_item_title,
                        id="home-selected-item-title",
                        classes="destination-section",
                    )
                    yield Static(selected_item_copy, id="home-selected-item-body")
            with Horizontal(id="home-followup-row"):
                with Vertical(
                    id="home-next-actions-region",
                    classes="home-followup-region destination-workbench-pane",
                ):
                    yield Static(
                        "Next Best Action",
                        id="home-next-best-action",
                        classes="ds-panel destination-section",
                    )
                    yield Static(next_action_copy, id="home-next-best-action-body")
                yield Static("", id="home-followup-divider", classes="home-pane-divider")
                with Vertical(
                    id="home-recent-work-region",
                    classes="home-followup-region destination-workbench-pane",
                ):
                    yield Static(
                        "Recent Work",
                        id="home-recent-work",
                        classes="ds-panel destination-section",
                    )
                    yield Static(section_text("recent_work"), id="home-recent-work-body")

    def _selected_home_item(self, dashboard_input: HomeDashboardInput):
        return choose_home_selected_item(dashboard_input)

    @on(Button.Pressed)
    def handle_home_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if not button_id:
            return

        if button_id == "home-primary-action":
            self._activate_home_primary_action()
            return

        self._activate_home_control(button_id)

    def _activate_home_primary_action(self) -> None:
        dashboard = self._current_dashboard
        if dashboard is None:
            return
        prepare = getattr(self.app_instance, "prepare_home_primary_action", None)
        if callable(prepare):
            prepare(dashboard.next_action)
        self.post_message(NavigateToScreen(dashboard.next_action.target_route))

    def _activate_home_control(self, button_id: str) -> None:
        dashboard = self._current_dashboard
        if dashboard is None:
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
