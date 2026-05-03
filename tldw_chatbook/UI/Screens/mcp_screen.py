"""MCP destination shell for tools, servers, permissions, and audit."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class MCPScreen(BaseAppScreen):
    """MCP servers, tools, permissions, auth, and audit surface."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "mcp", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="mcp-shell"):
            yield Static("MCP", id="mcp-title", classes="ds-destination-header")
            yield Static(
                "MCP servers, tools, permissions, auth, and audit.",
                id="mcp-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="mcp-sections", classes="ds-panel"):
                yield Static("Servers | Tools | Permissions | Auth | Audit")
