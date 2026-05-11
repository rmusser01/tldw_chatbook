"""ACP destination shell for agent sessions and runtimes."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Rule, Static

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


class ACPScreen(BaseAppScreen):
    """Agent Client Protocol agents, sessions, runtimes, diffs, and terminals."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "acp", **kwargs)

    @staticmethod
    def _column_divider(identifier: str) -> Rule:
        divider = Rule(orientation="vertical", id=identifier)
        divider.add_class("destination-pane-divider")
        return divider

    def compose_content(self) -> ComposeResult:
        with Vertical(id="acp-shell"):
            yield Static(
                "ACP | Agent protocol sessions and runtimes | Runtime needed | Local/Remote",
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
                    yield Static("> Codex local", id="acp-agent-codex-local")
                    yield Static("  No sessions", id="acp-no-sessions")
                    yield Static("  Runtime blocked", id="acp-runtime-blocked")
                    yield Static("  Diffs unavailable", id="acp-diffs-unavailable")
                    yield Static("  Terminal unavailable", id="acp-terminal-unavailable")
                yield self._column_divider("acp-list-detail-divider")
                with Vertical(id="acp-detail-pane", classes="destination-workbench-pane"):
                    yield Static(
                        "Column 2: Session Detail / Runtime Setup",
                        classes="destination-section acp-column-title",
                    )
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
                    yield Static("ACP version: n/a", id="acp-version-status")
                    yield Static("Runtime owner: ACP", id="acp-runtime-owner")
                    yield Static("Launch disabled: runtime missing", id="acp-launch-disabled-reason")
                    yield Static("Console follow disabled: no session", id="acp-follow-disabled-reason")
                    yield Button(
                        "Console follow unavailable",
                        id="acp-follow-in-console",
                        disabled=True,
                        tooltip=ACP_CONSOLE_FOLLOW_UNAVAILABLE.disabled_tooltip,
                    )
                    yield Button(
                        "Launch ACP Agent",
                        id="acp-launch-agent",
                        disabled=True,
                        tooltip=ACP_RUNTIME_NOT_CONFIGURED.disabled_tooltip,
                    )
