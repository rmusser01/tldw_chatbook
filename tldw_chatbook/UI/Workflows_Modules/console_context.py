"""Secondary current-dev Console handoff; never recomposes the editor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.UI.Navigation.screen_state_store import RuntimeIdentity
from tldw_chatbook.UI.Workflows_Modules.library import compact_button

if TYPE_CHECKING:
    from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem


class WorkflowConsoleContext(Vertical):
    """Inspect existing Home active work; this region creates no workflow runs."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(id="workflows-console-context", **kwargs)
        self.app_instance = app_instance
        self.item: HomeActiveWorkItem | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="workflow-console-actions"):
            yield compact_button(
                "Open in Console", "workflows-launch-in-console", disabled=True
            )
            yield Static(
                "Console context: loading…", id="workflows-run-status", markup=False
            )
            yield Static(
                "No active workflow run",
                id="workflows-console-unavailable",
                markup=False,
            )

    def on_mount(self) -> None:
        self.refresh_context()

    def latest_item(self, has_recent_work: bool) -> HomeActiveWorkItem | None:
        adapter = getattr(self.app_instance, "home_active_work_adapter", None)
        build = getattr(adapter, "build_dashboard_input", None)
        if not callable(build):
            return None
        try:
            data = build(
                providers_models=getattr(self.app_instance, "providers_models", {})
                or {},
                has_recent_work=has_recent_work,
            )
        except Exception:  # noqa: BLE001 -- optional existing Console context must not break authoring
            logger.warning("Workflows Console context is unavailable")
            return None
        for item in tuple(getattr(data, "active_work_items", ()) or ()):
            if (
                str(getattr(item, "source", "") or "").strip().lower() == "workflows"
                and bool(getattr(item, "console_available", False))
                and getattr(item, "item_id", None)
            ):
                return item
        return None

    def refresh_context(self) -> None:
        # ScreenStateStore belongs to the app thread. Only the existing
        # dashboard adapter's potentially slow read belongs in the worker.
        state = getattr(
            getattr(self.app_instance, "runtime_policy", None), "state", None
        )
        store = getattr(self.app_instance, "screen_state_store", None)
        recent = bool(
            state and store and store.has_snapshots(RuntimeIdentity.from_state(state))
        )
        self._refresh_from_adapter(recent)

    @work(exclusive=True, group="workflows-console-context", thread=True)
    def _refresh_from_adapter(self, has_recent_work: bool) -> None:
        item = self.latest_item(has_recent_work)
        self.app.call_from_thread(self.apply_context, item)

    def apply_context(self, item: HomeActiveWorkItem | None) -> None:
        """Update only stable labels/buttons; draft and focus remain attached."""
        self.item = item
        if not self.is_mounted:
            return
        status = "No active workflow run"
        if item:
            status = f"{getattr(item, 'title', 'Untitled')} · {getattr(item, 'status', 'unknown')}"
        status_widget = self.query_one("#workflows-run-status", Static)
        status_widget.display = item is not None
        status_widget.update(status)
        self.query_one("#workflows-console-unavailable").display = item is None
        button = self.query_one("#workflows-launch-in-console", Button)
        button.disabled = item is None
        button.tooltip = (
            "Open the selected existing run in Console."
            if item
            else "Start or select a workflow run before opening it in Console."
        )

    @on(Button.Pressed, "#workflows-launch-in-console")
    def launch(self, event: Button.Pressed) -> None:
        event.stop()
        target = getattr(self.item, "item_id", None)
        if not target:
            self.app_instance.notify(
                "Start or select a workflow run before opening it in Console.",
                severity="warning",
            )
            return
        open_item = getattr(self.app_instance, "open_active_home_item_in_console", None)
        if not callable(open_item):
            self.app_instance.notify(
                "Console launch is unavailable for Workflows in this runtime.",
                severity="warning",
            )
            return
        open_item(target_id=target, target_route="chat")
