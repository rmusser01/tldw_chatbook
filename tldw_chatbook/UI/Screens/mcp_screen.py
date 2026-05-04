"""MCP destination shell for tools, servers, permissions, and audit."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..MCP_Modules.unified_mcp_panel import UnifiedMCPPanel
from ..Navigation.base_app_screen import BaseAppScreen


class MCPScreen(BaseAppScreen):
    """MCP servers, tools, permissions, auth, and audit surface."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "mcp", **kwargs)
        self.mcp_panel: UnifiedMCPPanel | None = None

    def compose_content(self) -> ComposeResult:
        with Vertical(id="mcp-shell"):
            yield Static("MCP", id="mcp-title", classes="ds-destination-header")
            yield Static(
                "MCP owns tools and servers.",
                id="mcp-purpose",
                classes="destination-purpose",
            )
            self.mcp_panel = UnifiedMCPPanel(self.app_instance, id="unified-mcp-panel", classes="ds-panel")
            yield self.mcp_panel

    def save_state(self):
        """Save Unified MCP view state when the screen is switched away."""
        state = super().save_state()
        if self.mcp_panel:
            state["unified_mcp_view_state"] = self.mcp_panel.get_view_state()
        return state

    def restore_state(self, state):
        """Restore Unified MCP view state when the screen is revisited."""
        super().restore_state(state)
        if self.mcp_panel and isinstance(state, dict):
            self.mcp_panel.set_initial_view_state(state.get("unified_mcp_view_state"))

    async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
        """Refresh MCP context when runtime backend/source changes."""
        _ = runtime_backend
        if self.mcp_panel:
            await self.mcp_panel.load_context()
