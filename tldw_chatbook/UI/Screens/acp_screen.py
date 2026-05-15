"""ACP destination shell for agent sessions and runtimes."""

from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Rule, Static

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
    def _column_divider(identifier: str) -> Rule:
        divider = Rule(orientation="vertical", id=identifier)
        divider.add_class("destination-pane-divider")
        return divider

    def _runtime_session_state(self) -> ACPRuntimeSessionState:
        provider = getattr(self.app_instance, "get_acp_runtime_session_state", None)
        raw_state = provider() if callable(provider) else getattr(
            self.app_instance,
            "acp_runtime_session_state",
            None,
        )
        return ACPRuntimeSessionState.from_any(raw_state)

    def compose_content(self) -> ComposeResult:
        state = self._runtime_session_state()
        runtime_configured = state.runtime_configured
        console_launch = state.to_console_live_work_launch()
        has_session_payload = console_launch is not None
        runtime_display_name = escape_markup(state.runtime_display_name)
        session_display_name = escape_markup(state.session_display_name)
        title_state = "Runtime ready" if runtime_configured else "Runtime needed"
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
                    "Modes: Agents Sessions Runtimes Compatibility | Filter: Ready Blocked",
                    id="acp-mode-label",
                    classes="destination-section",
                )
            with Horizontal(id="acp-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="acp-list-pane", classes="destination-workbench-pane"):
                    yield Static(
                        "Column 1: Agents / Sessions",
                        classes="destination-section acp-column-title",
                    )
                    yield Static(f"> {runtime_display_name}", id="acp-runtime-display")
                    yield Static(session_line, id="acp-session-status")
                    yield Static(runtime_line, id="acp-runtime-status")
                    yield Static("  Diffs unavailable", id="acp-diffs-unavailable")
                    yield Static("  Terminal unavailable", id="acp-terminal-unavailable")
                yield self._column_divider("acp-list-detail-divider")
                with Vertical(id="acp-detail-pane", classes="destination-workbench-pane"):
                    yield Static(
                        "Column 2: Session Detail / Runtime Setup",
                        classes="destination-section acp-column-title",
                    )
                    if runtime_configured:
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
                with Vertical(id="acp-inspector-pane", classes="destination-workbench-pane ds-inspector"):
                    yield Static(
                        "Column 3: Compatibility / Actions",
                        classes="destination-section acp-column-title",
                    )
                    version = state.runtime_version or "n/a"
                    yield Static(f"ACP version: {version}", id="acp-version-status")
                    yield Static("Runtime owner: ACP", id="acp-runtime-owner")
                    launch_reason = (
                        "Launch disabled: session launch contract pending"
                        if runtime_configured
                        else "Launch disabled: runtime missing"
                    )
                    yield Static(launch_reason, id="acp-launch-disabled-reason")
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
                    yield Button(
                        "Launch ACP Agent",
                        id="acp-launch-agent",
                        disabled=True,
                        tooltip=(
                            "ACP session launch is not wired yet."
                            if runtime_configured
                            else ACP_RUNTIME_NOT_CONFIGURED.disabled_tooltip
                        ),
                    )

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
