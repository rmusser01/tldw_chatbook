"""ACP destination shell for agent sessions and runtimes."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen
from .destination_recovery import DestinationRecoveryState


ACP_RUNTIME_NOT_CONFIGURED = DestinationRecoveryState(
    status_label="Runtime not configured",
    unavailable_what="ACP agent launch",
    why="no ACP-compatible runtime is configured",
    next_action="Configure an ACP runtime in Settings before launch.",
    recovery_action="Settings",
    authority_owner="local app runtime",
    stable_selector="acp-empty-state",
    disabled_tooltip="Configure an ACP-compatible runtime in Settings before launching an ACP agent.",
)

ACP_CONSOLE_FOLLOW_UNAVAILABLE = DestinationRecoveryState(
    status_label="Runtime not configured",
    unavailable_what="Console follow for ACP sessions",
    why="ACP session payloads are not wired yet",
    next_action="Configure an ACP runtime and start a session before following it in Console.",
    recovery_action="ACP",
    authority_owner="local app runtime",
    stable_selector="acp-console-unavailable",
    disabled_tooltip="Configure an ACP runtime and start a session before following it in Console.",
)


class ACPScreen(BaseAppScreen):
    """Agent Client Protocol agents, sessions, runtimes, diffs, and terminals."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "acp", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="acp-shell"):
            yield Static("ACP", id="acp-title", classes="ds-destination-header")
            yield Static(
                "Agent Client Protocol agents, sessions, runtimes, diffs, and terminals.",
                id="acp-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="acp-sections", classes="ds-panel"):
                yield Static("Installed agents", classes="destination-section")
                yield Static("Sessions", classes="destination-section")
                yield Static("Resume", classes="destination-section")
                yield Static("Diffs", classes="destination-section")
                yield Static("Terminal/Shell", classes="destination-section")
                yield Static(
                    ACP_RUNTIME_NOT_CONFIGURED.visible_copy,
                    id=ACP_RUNTIME_NOT_CONFIGURED.stable_selector,
                )
                yield Static(
                    ACP_CONSOLE_FOLLOW_UNAVAILABLE.visible_copy,
                    id=ACP_CONSOLE_FOLLOW_UNAVAILABLE.stable_selector,
                )
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
