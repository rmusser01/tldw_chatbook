"""Capture the local MCP master choice when its displayed button is activated."""

from pathlib import Path

from textual.message import Message
from textual.widgets import Button


class MCPLocalMasterButton(Button):
    """Carry the displayed choice through queued button events and repaints."""

    enabled: bool = False
    config_path: Path | None = None

    def post_message(self, message: Message) -> bool:
        if isinstance(message, Button.Pressed) and message.button is self:
            message.mcp_local_master_choice = (not self.enabled, self.config_path)
        return super().post_message(message)
