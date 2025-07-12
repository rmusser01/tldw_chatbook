"""
MCP (Model Context Protocol) Integration for tldw_chatbook

This module provides MCP server and client functionality, exposing tldw_chatbook's
features as MCP tools, resources, and prompts that can be used by AI applications
like Claude Desktop.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .server import TldwMCPServer
    from .client import MCPClient

__all__ = ["TldwMCPServer", "MCPClient", "is_mcp_available"]

def is_mcp_available() -> bool:
    """Check if MCP dependencies are available."""
    try:
        import mcp
        return True
    except ImportError:
        return False