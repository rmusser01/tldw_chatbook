"""Capture the local MCP master choice when its displayed button is activated."""

from pathlib import Path

from textual.message import Message
from textual.widgets import Button


class MCPLocalMasterButton(Button):
    """Carry the displayed choice through queued button events and repaints."""

    enabled: bool = False
    config_path: Path | None = None

    def post_message(self, message: Message) -> bool:
        """Capture the intended master choice before queueing this button's press.

        Args:
            message: Framework message; this button's presses receive the intended
                enabled state and configuration path visible at activation time.

        Returns:
            Whether the framework accepted the message for delivery.
        """
        if isinstance(message, Button.Pressed) and message.button is self:
            message.mcp_local_master_choice = (not self.enabled, self.config_path)
        return super().post_message(message)
