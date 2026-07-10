"""Home dashboard screen for the master shell."""

from collections.abc import Callable
from dataclasses import replace

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Constants import TAB_LLM, get_tab_display_label
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.Home.dashboard_state import (
    HomeDashboard,
    HomeDashboardInput,
    HomeTriageState,
    build_home_triage_state,
    choose_home_selected_item,
    summarize_home_dashboard,
)
from tldw_chatbook.Home.home_rail_state import (
    HOME_RAIL_SECTION_IDS,
    HomeRailPreferences,
    coerce_home_rail_preferences,
    serialize_home_rail_preferences,
)
from tldw_chatbook.Widgets.Console.console_rail_section import (
    CONSOLE_RAIL_SECTION_TOGGLE_PREFIX,
    ConsoleRailSectionHeader,
)
from tldw_chatbook.Widgets.Home.home_canvas import HomeCanvas
from tldw_chatbook.Widgets.Home.home_rail import HOME_RAIL_ROW_PREFIX, HomeRail

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from ..Navigation.shell_destinations import get_shell_destination, resolve_shell_route
from .settings_config_models import SettingsCategoryId


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
    "home-review-flashcards": "open_home_flashcards_review",
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


def _home_primary_action_context(action: object) -> dict[str, object]:
    if getattr(action, "action_id", None) == "fix_model_setup":
        return {"category": SettingsCategoryId.PROVIDERS_MODELS.value}
    return {}


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
        self._home_selected_row_id: str = ""

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_home_chatbook_artifact_snapshot()

    @work(exclusive=True, thread=True)
    def _refresh_home_chatbook_artifact_snapshot(self) -> None:
        adapter = getattr(self.app_instance, "home_active_work_adapter", None)
        refresh_flashcards_due = getattr(adapter, "refresh_flashcards_due_snapshot", None)
        if callable(refresh_flashcards_due):
            chachanotes_db = getattr(self.app_instance, "chachanotes_db", None)
            if getattr(chachanotes_db, "is_memory_db", False):
                # SQLite ``:memory:`` connections are thread-local -- the
                # flashcards-due provider ultimately queries ChaChaNotes
                # directly, and only the thread that created the DB has the
                # migrated schema. Running the refresh on THIS worker thread
                # would open a brand-new, unmigrated in-memory connection, so
                # hop back onto the UI thread for the in-memory case.
                # File-backed DBs keep the off-thread call -- that's the
                # whole point of this worker.
                self.app.call_from_thread(refresh_flashcards_due)
            else:
                refresh_flashcards_due()
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
        dashboard_input = self.app_instance.home_active_work_adapter.build_dashboard_input(
            providers_models=providers,
            has_recent_work=has_recent_work,
        )
        manager = getattr(self.app_instance, "acp_runtime_process_manager", None)
        snapshot = getattr(manager, "snapshot", None)
        if callable(snapshot):
            raw_snapshot = snapshot()
            if isinstance(raw_snapshot, dict):
                dashboard_input = replace(
                    dashboard_input,
                    acp_ready=str(raw_snapshot.get("status") or "") == "running",
                )
        return dashboard_input

    def compose_content(self) -> ComposeResult:
        """Compose the Home triage route: header, rail, focus canvas."""
        dashboard_input = self._build_dashboard_input()
        triage = build_home_triage_state(
            dashboard_input,
            selected_row_id=self._home_selected_row_id,
        )
        # Keep the legacy dashboard object for the unchanged control dispatch.
        self._current_dashboard = summarize_home_dashboard(dashboard_input)
        self._current_dashboard_input = dashboard_input
        self._home_selected_row_id = triage.selected_row_id

        yield Static(
            triage.header_line,
            id="home-header-line",
            classes="destination-status-row",
        )
        triage_grid = Horizontal(
            id="home-triage-grid", classes="ds-panel destination-workbench"
        )
        triage_grid.styles.height = "1fr"
        triage_grid.styles.min_height = 12
        with triage_grid:
            rail = HomeRail(
                triage,
                self._home_rail_preferences(),
                id="home-rail",
                classes="destination-workbench-pane",
            )
            rail.styles.height = "100%"
            yield rail
            canvas = HomeCanvas(
                triage.canvas,
                action_button_factory=self._home_action_button,
                id="home-canvas",
                classes="destination-workbench-pane",
            )
            canvas.styles.height = "100%"
            yield canvas

    def _home_action_button(self, label: str, control_id: str) -> HomeActionButton:
        """Build a canvas action button with the fallback-press wiring."""
        if control_id == "home-primary-action":
            return HomeActionButton(
                label,
                id="home-primary-action",
                classes="home-canvas-action",
                fallback_press=self._activate_home_primary_action,
            )
        return HomeActionButton(
            label,
            id=control_id,
            classes="home-canvas-action",
            fallback_press=lambda control_id=control_id: (
                self._activate_home_control(control_id)
            ),
        )

    def _home_rail_preferences(self) -> HomeRailPreferences:
        """Read persisted Home rail section preferences."""
        app_config = getattr(self.app_instance, "app_config", None)
        raw = None
        if isinstance(app_config, dict):
            home_config = app_config.get("home")
            if isinstance(home_config, dict):
                rail_state = home_config.get("rail_state")
                if isinstance(rail_state, dict):
                    raw = rail_state.get("sections")
        return coerce_home_rail_preferences(raw)

    def _set_home_rail_section(self, section_id: str, open_state: bool) -> None:
        """Persist one section preference and sync the rail body/header."""
        if section_id not in HOME_RAIL_SECTION_IDS:
            return
        from dataclasses import replace as dataclass_replace

        preferences = dataclass_replace(
            self._home_rail_preferences(), **{f"{section_id}_open": open_state}
        )
        serialized = serialize_home_rail_preferences(preferences)
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            home_config = app_config.get("home")
            if not isinstance(home_config, dict):
                home_config = {}
                app_config["home"] = home_config
            rail_state = home_config.get("rail_state")
            if not isinstance(rail_state, dict):
                rail_state = {}
                home_config["rail_state"] = rail_state
            rail_state["sections"] = serialized
        self._save_home_rail_preferences(serialized)
        try:
            body = self.query_one(f"#home-rail-section-body-{section_id}")
            header = self.query_one(
                f"#home-rail-section-header-{section_id}", ConsoleRailSectionHeader
            )
        except Exception:
            return
        body.styles.display = "block" if open_state else "none"
        header.sync_open(open_state)

    @work(thread=True)
    def _save_home_rail_preferences(self, serialized: dict[str, bool]) -> None:
        """Persist Home rail preferences without blocking the UI thread."""
        try:
            save_setting_to_cli_config("home.rail_state", "sections", serialized)
        except Exception:
            pass

    def _sync_home_triage(self) -> None:
        """Rebuild triage state and refresh rail + canvas in place."""
        dashboard_input = self._build_dashboard_input()
        triage = build_home_triage_state(
            dashboard_input,
            selected_row_id=self._home_selected_row_id,
        )
        self._current_dashboard = summarize_home_dashboard(dashboard_input)
        self._current_dashboard_input = dashboard_input
        self._home_selected_row_id = triage.selected_row_id
        try:
            self.query_one("#home-rail", HomeRail).sync_state(
                triage, self._home_rail_preferences()
            )
            self.query_one("#home-canvas", HomeCanvas).sync_state(triage.canvas)
            self.query_one("#home-header-line", Static).update(triage.header_line)
        except Exception:
            pass

    def _selected_home_item(self, dashboard_input: HomeDashboardInput):
        return choose_home_selected_item(dashboard_input)

    @on(Button.Pressed)
    def handle_home_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if not button_id:
            return

        if button_id.startswith(HOME_RAIL_ROW_PREFIX):
            event.stop()
            row_id = str(getattr(event.button, "row_id", "") or "")
            if row_id:
                self._home_selected_row_id = row_id
                self._sync_home_triage()
            return

        if button_id.startswith(f"{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}home-"):
            event.stop()
            section_id = button_id.removeprefix(
                f"{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}home-"
            )
            currently_open = bool(
                getattr(self._home_rail_preferences(), f"{section_id}_open", True)
            )
            self._set_home_rail_section(section_id, not currently_open)
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
        self.post_message(
            NavigateToScreen(
                dashboard.next_action.target_route,
                screen_context=_home_primary_action_context(dashboard.next_action),
            )
        )

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
