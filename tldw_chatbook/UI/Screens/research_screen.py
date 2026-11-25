"""Research Sessions screen implementation."""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.widgets import Button, Select

from ...Constants import TAB_RESEARCH_WORKSPACE
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from ..Navigation.pending_handoff_store import HandoffChannel
from ..Research_Window import ResearchWindow


class ResearchScreen(BaseAppScreen):
    """Screen wrapper for Research Sessions functionality."""

    CSS_PATH = None
    BINDINGS = []

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "research", **kwargs)
        self.research_window: ResearchWindow | None = None
        self._pending_restore_state: Dict[str, Any] | None = None
        self._return_target = None

    def compose_content(self) -> ComposeResult:
        """Compose the screen's content: the ResearchWindow plus any pending
        state restoration."""
        from ..Research_Workspace_Modules.mode_bar import ResearchModeStrip

        yield ResearchModeStrip(active_route="research", id="research-mode-strip")
        self.research_window = ResearchWindow(
            self.app_instance,
            id="research-window",
            classes="window",
        )
        if self._pending_restore_state is not None:
            self.research_window.restore_state(self._pending_restore_state)
            self._pending_restore_state = None
        yield self.research_window

    def on_mount(self) -> None:
        self.call_after_refresh(self._start_run_handoff)

    def on_screen_resume(self) -> None:
        self.call_after_refresh(self._start_run_handoff)

    def _start_run_handoff(self) -> None:
        store = getattr(self.app_instance, "pending_handoffs", None)
        if store is None or not self.is_attached:
            return
        if claim := store.claim(HandoffChannel.RESEARCH_RUN_TARGET):
            self.run_worker(
                self._open_run_handoff(store, claim),
                group="research-run-handoff",
                exit_on_error=False,
            )

    async def _open_run_handoff(self, store, claim) -> None:
        from ..Navigation.research_run_handoffs import (
            research_run_target_current,
            workspace_target_from_run,
        )

        target = claim.value
        window = self.research_window
        app = self.app_instance

        def current() -> bool:
            return (
                self.is_attached
                and self.app.is_running
                and self.app.screen is self
                and store.is_current_claim(claim)
                and research_run_target_current(app, target.origin)
            )

        try:
            if window is None or not current():
                return
            source = target.origin.data_source.value
            chooser = window.query_one("#research-source-select", Select)
            with chooser.prevent(Select.Changed):
                chooser.value = source
            await window.switch_source(source)
            if not current() or window.current_source != source:
                return
            revision = window._selection_revision
            run = await window.controller.get_run(source, target.run_id)
            if not current() or revision != window._selection_revision:
                return
            if workspace_target_from_run(app, run) != target:
                self.notify(
                    "This run belongs to a different workspace launch.",
                    severity="warning",
                )
                return
            window.select_run(run)
            store.acknowledge_current(claim)
            if source == "local" and run.get("status") == "draft":
                window._start_local_engine(target.run_id)
        except Exception as exc:
            logger.warning("Research run handoff read failed ({})", type(exc).__name__)
            if current():
                self.notify(
                    "The selected run could not open. Refresh Runs and retry.",
                    severity="warning",
                )
        finally:
            if store.is_current_claim(claim):
                store.release(claim)

    @on(ResearchWindow.SelectionChanged)
    def _run_selection_changed(self) -> None:
        from ..Navigation.research_run_handoffs import workspace_target_from_run

        window = self.research_window
        if window is None or not self.is_attached or not self.app.is_running:
            return
        run = window.selected_run
        self._set_return_target(None)
        if (
            window._record_get(run, "status") != "completed"
            and window.current_bundle is None
        ):
            return
        target = workspace_target_from_run(self.app_instance, run)
        if (
            target is not None
            and target.origin.data_source.value == window.current_source
        ):
            self._set_return_target(target)
        elif window.current_source == "server" and (
            run_id := window._record_get(run, "id")
        ):
            self.run_worker(
                self._load_return_origin(window, run_id, window._selection_revision),
                group="research-return-origin",
                exclusive=True,
                exit_on_error=False,
            )

    async def _load_return_origin(self, window, run_id: str, revision: int) -> None:
        from ..Navigation.research_run_handoffs import (
            active_server_identity,
            workspace_target_from_run,
        )

        identity = active_server_identity(self.app_instance)
        if identity is None:
            return
        try:
            run = await window.controller.get_run("server", run_id)
        except Exception as exc:
            logger.debug("Research return origin unavailable ({})", type(exc).__name__)
            return
        if (
            self.is_attached
            and self.app.is_running
            and self.app.screen is self
            and window.current_source == "server"
            and window._selection_revision == revision
            and active_server_identity(self.app_instance) == identity
            and window._record_get(run, "id") == run_id
        ):
            self._set_return_target(workspace_target_from_run(self.app_instance, run))

    def _set_return_target(self, target) -> None:
        self._return_target = target
        self.query_one("#research-return-to-workspace", Button).display = (
            target is not None
        )

    @on(Button.Pressed, "#research-return-to-workspace")
    def _return_to_workspace(self) -> None:
        from ..Navigation.research_run_handoffs import research_run_target_current

        app = self.app_instance
        target = self._return_target
        window = self.research_window
        if (
            target is None
            or window is None
            or window._record_get(window.selected_run, "id") != target.run_id
            or window.current_source != target.origin.data_source.value
        ):
            return
        if not research_run_target_current(app, target.origin):
            self._set_return_target(None)
            self.notify(
                "Open this run with its original Server profile and account.",
                severity="warning",
            )
            return
        if app._stage_handoff(
            HandoffChannel.RESEARCH_WORKSPACE_RETURN,
            target,
            recovery="The run could not return to its workspace.",
        ):
            app.post_message(NavigateToScreen(TAB_RESEARCH_WORKSPACE))

    def save_state(self) -> Dict[str, Any]:
        """Persist the hosted window's state for navigation restore.

        Returns:
            The window's save-state dict, or the pending restore state when
            the window has not been composed yet.
        """
        try:
            window = self.query_one("#research-window", ResearchWindow)
        except Exception:
            window = self.research_window
        if window is None:
            return self._pending_restore_state or {"source": "local"}
        return window.save_state()

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore the hosted window's state after navigation.

        Args:
            state: The state dict previously returned by ``save_state``;
                deferred until compose when the window is not mounted yet.
        """
        try:
            window = self.query_one("#research-window", ResearchWindow)
        except Exception:
            window = self.research_window
        if window is None:
            logger.debug(
                "Deferring ResearchScreen restore_state until window is composed"
            )
            self._pending_restore_state = dict(state or {})
            return
        window.restore_state(state)
