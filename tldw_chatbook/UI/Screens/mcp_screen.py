"""MCP destination shell for tools, servers, permissions, and audit."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen


class MCPScreen(BaseAppScreen):
    """MCP servers, tools, permissions, auth, and audit surface."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "mcp", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="mcp-shell"):
            yield Static("MCP", id="mcp-title", classes="ds-destination-header")
            yield Static(
                "MCP owns tools and servers.",
                id="mcp-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="mcp-sections", classes="ds-panel"):
                yield Static("Servers", classes="destination-section")
                yield Static("Tools", classes="destination-section")
                yield Static("Permissions", classes="destination-section")
                yield Static("Auth", classes="destination-section")
                yield Static("Audit", classes="destination-section")
                yield Static("Test Tool", classes="destination-section")
                yield Static(
                    "Unified MCP management is not embedded in this shell yet.",
                    id="mcp-management-unavailable",
                )
                yield Button(
                    "Open MCP Management",
                    id="mcp-open-management",
                    disabled=True,
                    tooltip="Unavailable until MCP management is embedded in this shell.",
                )
