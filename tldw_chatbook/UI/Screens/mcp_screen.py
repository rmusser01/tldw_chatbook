"""MCP destination shell for tools, servers, permissions, and audit."""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ...Widgets.destination_workbench import DestinationModeStrip
from ..MCP_Modules.unified_mcp_panel import LAYOUT_MODE_COMPACT_WORKBENCH, UnifiedMCPPanel
from ..Navigation.base_app_screen import BaseAppScreen


class MCPScreen(BaseAppScreen):
    """MCP servers, tools, permissions, auth, and audit surface."""

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "mcp", **kwargs)
        self.mcp_panel: UnifiedMCPPanel | None = None

    def compose_content(self) -> ComposeResult:
        with Vertical(id="mcp-shell"):
            yield Static("MCP", id="mcp-title", classes="ds-destination-header")
            yield Static(
                "Manage MCP servers, scoped tools, permissions, and audit readiness.",
                id="mcp-purpose",
                classes="destination-purpose",
            )
            with DestinationModeStrip(id="mcp-mode-strip", classes="destination-mode-strip"):
                yield Static(
                    "Mode: Servers | Tools | Permissions | Audit",
                    id="mcp-mode-label",
                    classes="destination-section",
                )
            self.mcp_panel = UnifiedMCPPanel(
                self.app_instance,
                id="unified-mcp-panel",
                layout_mode=LAYOUT_MODE_COMPACT_WORKBENCH,
            )
            self.mcp_panel.set_initial_view_state(self.state_data.get("unified_mcp_view_state"))
            yield self.mcp_panel

    def save_state(self) -> dict[str, Any]:
        """Save Unified MCP view state when the screen is switched away.

        Returns:
            Screen state including the embedded Unified MCP panel selection.
        """
        state = super().save_state()
        if self.mcp_panel:
            state["unified_mcp_view_state"] = self.mcp_panel.get_view_state()
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore Unified MCP view state when the screen is revisited.

        Args:
            state: Previously saved screen state.
        """
        super().restore_state(state)
        if self.mcp_panel:
            self.mcp_panel.set_initial_view_state(state.get("unified_mcp_view_state"))

    async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
        """Schedule MCP context refresh when runtime backend/source changes.

        Args:
            runtime_backend: Newly active runtime backend identifier.
        """
        _ = runtime_backend
        if self.mcp_panel:
            self.run_worker(
                self.mcp_panel.load_context(),
                name="mcp-screen-runtime-refresh",
                group="mcp-screen-runtime-refresh",
                exclusive=True,
            )
