"""ACP destination shell for agent sessions and runtimes."""

from typing import Any

from rich.markup import escape as escape_markup
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Rule, Static

from ...ACP_Interop.runtime_process import ACPRuntimeProcessResult
from ...ACP_Interop.runtime_session import ACPRuntimeSessionState
from ...Widgets.destination_workbench import DestinationModeStrip
from ..Navigation.base_app_screen import BaseAppScreen
from .destination_recovery import DestinationRecoveryState


ACP_RUNTIME_NOT_CONFIGURED = DestinationRecoveryState(
    status_label="Runtime not configured",
    unavailable_what="ACP agent launch",
    why="no ACP-compatible runtime is configured",
    next_action="Configure ACP runtime setup in ACP before launch.",
    recovery_action="ACP",
    authority_owner="ACP runtime",
    stable_selector="acp-empty-state",
    disabled_tooltip="Configure an ACP-compatible runtime in ACP before launching an ACP agent.",
)

ACP_CONSOLE_FOLLOW_UNAVAILABLE = DestinationRecoveryState(
    status_label="Runtime not configured",
    unavailable_what="Console follow for ACP sessions",
    why="ACP session payloads require a configured ACP runtime",
    next_action="Configure an ACP runtime and start a session before following it in Console.",
    recovery_action="ACP",
    authority_owner="ACP runtime",
    stable_selector="acp-console-unavailable",
    disabled_tooltip="Configure an ACP runtime and start a session before following it in Console.",
)

ACP_SESSION_FOLLOW_UNAVAILABLE = DestinationRecoveryState(
    status_label="No ACP session payload",
    unavailable_what="Console follow for ACP sessions",
    why="no ACP session payload is available",
    next_action="Start or resume an ACP session in ACP before following it in Console.",
    recovery_action="ACP",
    authority_owner="ACP runtime",
    stable_selector="acp-console-unavailable",
    disabled_tooltip="Start or resume an ACP session in ACP before following it in Console.",
)


class ACPScreen(BaseAppScreen):
    """Agent Client Protocol agents, sessions, runtimes, diffs, and terminals."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "acp", **kwargs)

    @staticmethod
    def _display_status(value: str) -> str:
        status = (value or "unknown").replace("_", " ").strip()
        return status[:1].upper() + status[1:] if status else "Unknown"

    @staticmethod
    def _column_divider(identifier: str) -> Rule:
        divider = Rule(orientation="vertical", id=identifier)
        divider.add_class("destination-pane-divider")
        return divider

    def _runtime_session_state(self) -> ACPRuntimeSessionState:
        provider = getattr(self.app_instance, "get_acp_runtime_session_state", None)
        if callable(provider):
            raw_state = provider()
        else:
            raw_state = getattr(self.app_instance, "acp_runtime_session_state", None)
            if raw_state is None:
                manager = getattr(self.app_instance, "acp_runtime_process_manager", None)
                snapshot = getattr(manager, "snapshot", None)
                raw_state = snapshot() if callable(snapshot) else None
        return ACPRuntimeSessionState.from_any(raw_state)

    def _runtime_process_snapshot(self) -> dict:
        explicit_state = ACPRuntimeSessionState.from_any(
            getattr(self.app_instance, "acp_runtime_session_state", None)
        )
        manager = getattr(self.app_instance, "acp_runtime_process_manager", None)
        snapshot = getattr(manager, "snapshot", None)
        if callable(snapshot):
            raw_snapshot = snapshot()
            if isinstance(raw_snapshot, dict):
                manager_status = str(raw_snapshot.get("status") or "not_configured")
                if explicit_state.runtime_configured and manager_status == "not_configured":
                    return {
                        "status": "running"
                        if explicit_state.has_console_session_payload
                        else "configured",
                        "launch_available": False,
                        "stop_available": False,
                        "recovery": "ACP runtime session state is available.",
                        "command_display": explicit_state.runtime_display_name,
                    }
                return dict(raw_snapshot)
        state = explicit_state if explicit_state.runtime_configured else self._runtime_session_state()
        return {
            "status": "running" if state.has_console_session_payload else "configured"
            if state.runtime_configured
            else "not_configured",
            "launch_available": False,
            "stop_available": False,
            "recovery": "ACP runtime process manager is unavailable.",
            "command_display": "not configured",
        }

    def compose_content(self) -> ComposeResult:
        process_snapshot = self._runtime_process_snapshot()
        state = self._runtime_session_state()
        runtime_configured = state.runtime_configured
        console_launch = state.to_console_live_work_launch()
        has_session_payload = console_launch is not None
        process_status = str(process_snapshot.get("status") or "not_configured")
        launch_available = bool(process_snapshot.get("launch_available"))
        stop_available = bool(process_snapshot.get("stop_available"))
        process_recovery = str(process_snapshot.get("recovery") or "")
        runtime_display_name = escape_markup(state.runtime_display_name)
        session_display_name = escape_markup(state.session_display_name)
        display_process_status = self._display_status(process_status)
        console_status = "Console-ready" if has_session_payload else "Console blocked"
        state_summary = f"State: {display_process_status} · {console_status}"
        title_state = (
            "Runtime ready"
            if runtime_configured and process_status != "running"
            else "Runtime running"
            if process_status == "running"
            else "Runtime needed"
        )
        runtime_line = (
            f"  Runtime configured: {runtime_display_name}"
            if runtime_configured
            else "  Runtime blocked"
        )
        session_line = (
            f"  Session: {session_display_name}"
            if runtime_configured
            else "  No sessions"
        )
        list_row_label = session_display_name if has_session_payload else runtime_display_name
        list_row_badges = (
            f"({escape_markup(str(state.session_status or process_status))}) (console-ready)"
            if has_session_payload
            else f"({escape_markup(process_status)}) (console-blocked)"
            if runtime_configured
            else "(blocked) (setup-needed)"
        )
        session_detail_title = (
            "Active Session"
            if has_session_payload
            else "Runtime Ready"
            if runtime_configured
            else "Runtime Setup"
        )
        payload = state.session_payload
        process_id = payload.get("pid")
        started_at = payload.get("started_at")
        console_recovery = (
            ACP_SESSION_FOLLOW_UNAVAILABLE
            if runtime_configured
            else ACP_CONSOLE_FOLLOW_UNAVAILABLE
        )
        follow_label = (
            "Follow ACP Session in Console"
            if has_session_payload
            else "Console follow unavailable"
        )
        follow_disabled_reason = (
            "Console follow ready: session payload available"
            if has_session_payload
            else "Console follow disabled: no ACP session payload"
            if runtime_configured
            else "Console follow disabled: no session"
        )

        with Vertical(id="acp-shell"):
            yield Static(
                f"ACP | Agent protocol sessions and runtimes | {title_state} | Local/Remote",
                id="acp-title",
                classes="ds-destination-header",
            )
            yield Static(
                "Agent Client Protocol interoperability for agent sessions, runtimes, diffs, and terminals.",
                id="acp-purpose",
                classes="destination-purpose",
            )
            with DestinationModeStrip(id="acp-mode-strip", classes="destination-mode-strip"):
                yield Static(
                    "View: Agents / Sessions / Runtimes / Compatibility | Scope: Ready + Blocked",
                    id="acp-mode-label",
                    classes="destination-section",
                )
            with Horizontal(id="acp-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="acp-list-pane", classes="destination-workbench-pane acp-framed-pane"):
                    yield Static(
                        "Agents / Sessions",
                        classes="destination-section acp-column-title",
                    )
                    yield Static(
                        f"> {list_row_label} {list_row_badges}",
                        id="acp-session-list-row",
                        classes="acp-selected-session-row",
                    )
                    yield Static(f"  Runtime: {runtime_display_name}", id="acp-runtime-display")
                    yield Static(session_line, id="acp-session-status")
                    yield Static(runtime_line, id="acp-runtime-status")
                    yield Static(
                        "  Diffs: not supported by current runtime payload",
                        id="acp-diffs-unavailable",
                    )
                    yield Static(
                        "  Terminal: no terminal stream attached",
                        id="acp-terminal-unavailable",
                    )
                yield self._column_divider("acp-list-detail-divider")
                with Vertical(id="acp-detail-pane", classes="destination-workbench-pane acp-framed-pane"):
                    yield Static(
                        session_detail_title,
                        classes="destination-section acp-column-title",
                    )
                    if runtime_configured:
                        yield Static(state_summary, id="acp-state-summary")
                        yield Static("Runtime configured", id="acp-runtime-ready-state")
                        yield Static(f"Runtime: {runtime_display_name}", id="acp-runtime-summary")
                        yield Static(f"Session: {session_display_name}", id="acp-session-summary")
                        if has_session_payload:
                            yield Static(
                                f"Session ready: {session_display_name}",
                                id="acp-session-ready",
                            )
                            yield Static(
                                "Console follow ready: session payload available",
                                id="acp-console-ready",
                            )
                            if process_id is not None:
                                yield Static(
                                    f"Process: pid {escape_markup(str(process_id))}",
                                    id="acp-process-id",
                                )
                            if started_at:
                                yield Static(
                                    f"Started: {escape_markup(str(started_at))}",
                                    id="acp-started-at",
                                )
                            yield Static(
                                f"Handoff ID: {escape_markup(state.session_id)}",
                                id="acp-console-target",
                            )
                        else:
                            yield Static(
                                ACP_SESSION_FOLLOW_UNAVAILABLE.visible_copy,
                                id=ACP_SESSION_FOLLOW_UNAVAILABLE.stable_selector,
                            )
                    else:
                        yield Static(
                            ACP_RUNTIME_NOT_CONFIGURED.visible_copy,
                            id=ACP_RUNTIME_NOT_CONFIGURED.stable_selector,
                        )
                        yield Static(
                            "Setup steps:\n"
                            "1. Add an ACP-compatible runtime.\n"
                            "2. Start or resume an ACP session.\n"
                            "3. Follow live work in Console.",
                            id="acp-runtime-setup-steps",
                        )
                        yield Static(
                            ACP_CONSOLE_FOLLOW_UNAVAILABLE.visible_copy,
                            id=ACP_CONSOLE_FOLLOW_UNAVAILABLE.stable_selector,
                        )
                yield self._column_divider("acp-detail-inspector-divider")
                with Vertical(
                    id="acp-inspector-pane",
                    classes="destination-workbench-pane ds-inspector acp-framed-pane",
                ):
                    yield Static(
                        "Compatibility / Actions",
                        classes="destination-section acp-column-title",
                    )
                    version = state.runtime_version or "n/a"
                    yield Static(
                        "Compatibility",
                        id="acp-compatibility-group-title",
                        classes="destination-section acp-inspector-group-title",
                    )
                    yield Static(f"ACP version: {version}", id="acp-version-status")
                    yield Static("Runtime owner: ACP", id="acp-runtime-owner")
                    yield Static(
                        f"Runtime state: {escape_markup(process_status)}",
                        id="acp-runtime-process-state",
                    )
                    yield Static(
                        f"Runtime recovery: {escape_markup(process_recovery)}",
                        id="acp-runtime-process-recovery",
                    )
                    launch_reason = (
                        "Launch ready: runtime command configured"
                        if launch_available
                        else "Launch disabled: runtime missing"
                        if not runtime_configured
                        else f"Launch disabled: {process_recovery}"
                    )
                    yield Static(
                        "Actions",
                        classes="destination-section acp-inspector-group-title",
                    )
                    if has_session_payload:
                        yield Static(
                            "Primary action",
                            id="acp-primary-action-group-title",
                            classes="destination-section acp-inspector-group-title",
                        )
                    yield Static(follow_disabled_reason, id="acp-follow-disabled-reason")
                    yield Button(
                        follow_label,
                        id="acp-follow-in-console",
                        disabled=not has_session_payload,
                        tooltip=(
                            "Open this ACP session in Console."
                            if has_session_payload
                            else console_recovery.disabled_tooltip
                        ),
                    )
                    yield Static(
                        "Runtime controls",
                        id="acp-runtime-controls-title",
                        classes="destination-section acp-inspector-group-title",
                    )
                    if process_status == "running":
                        yield Static(
                            "Launch/restart hidden: runtime is already running.",
                            id="acp-launch-disabled-reason",
                        )
                    else:
                        yield Static(launch_reason, id="acp-launch-disabled-reason")
                        yield Button(
                            "Launch ACP Agent",
                            id="acp-launch-agent",
                            disabled=not launch_available,
                            tooltip=(
                                "Start the configured ACP runtime."
                                if launch_available
                                else ACP_RUNTIME_NOT_CONFIGURED.disabled_tooltip
                                if not runtime_configured
                                else process_recovery
                            ),
                        )
                        yield Button(
                            "Restart Runtime",
                            id="acp-restart-runtime",
                            disabled=not launch_available,
                            tooltip=(
                                "Restart the configured ACP runtime."
                                if launch_available
                                else "ACP runtime is not ready to restart."
                            ),
                        )
                    yield Button(
                        "Stop Runtime",
                        id="acp-stop-runtime",
                        disabled=not stop_available,
                        tooltip=(
                            "Stop the running ACP runtime."
                            if stop_available
                            else "No running ACP runtime to stop."
                        ),
                    )

    @on(Button.Pressed, "#acp-launch-agent")
    @on(Button.Pressed, "#acp-restart-runtime")
    def launch_acp_runtime(self, event: Button.Pressed) -> None:
        event.stop()
        self._launch_acp_runtime_worker("ACP agent session")

    @work(exclusive=True, thread=True)
    def _launch_acp_runtime_worker(self, title: str) -> None:
        manager = getattr(self.app_instance, "acp_runtime_process_manager", None)
        launcher = getattr(manager, "start_session", None)
        if not callable(launcher):
            self.app.call_from_thread(
                self.notify,
                "ACP runtime launch is unavailable.",
                severity="warning",
            )
            return
        result = launcher(title=title)
        self.app.call_from_thread(
            self._apply_acp_runtime_result,
            result,
            "ACP runtime started. Console follow is ready.",
        )

    def _apply_acp_runtime_result(
        self,
        result: ACPRuntimeProcessResult | Any,
        success_message: str | None = None,
    ) -> None:
        session_state = getattr(result, "session_state", None)
        if session_state is not None:
            self.app_instance.acp_runtime_session_state = session_state
        status = str(getattr(result, "status", "unknown"))
        recovery = (
            success_message
            if status == "running" and success_message
            else str(getattr(result, "recovery", "ACP runtime state updated."))
        )
        severity = "information" if status == "running" else "warning"
        self.notify(recovery, severity=severity)
        self.refresh(recompose=True)

    @on(Button.Pressed, "#acp-stop-runtime")
    def stop_acp_runtime(self, event: Button.Pressed) -> None:
        event.stop()
        self._stop_acp_runtime_worker()

    @work(exclusive=True, thread=True)
    def _stop_acp_runtime_worker(self) -> None:
        manager = getattr(self.app_instance, "acp_runtime_process_manager", None)
        stopper = getattr(manager, "stop", None)
        if not callable(stopper):
            self.app.call_from_thread(
                self.notify,
                "ACP runtime stop is unavailable.",
                severity="warning",
            )
            return
        result = stopper()
        self.app.call_from_thread(self._apply_acp_runtime_result, result, None)

    @on(Button.Pressed, "#acp-follow-in-console")
    def follow_acp_session_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        launch = self._runtime_session_state().to_console_live_work_launch()
        if launch is None:
            self.notify(
                ACP_SESSION_FOLLOW_UNAVAILABLE.disabled_tooltip,
                severity="warning",
            )
            return
        opener = getattr(self.app_instance, "open_console_for_live_work", None)
        if not callable(opener):
            self.notify("Console live-work handoff is unavailable.", severity="warning")
            return
        opener(**launch.to_pending_payload())
