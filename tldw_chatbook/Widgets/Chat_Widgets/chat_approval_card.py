from typing import Any, Dict, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static


class ChatApprovalCard(Container):
    """Inline approval card for privileged agent actions."""

    def compose(self) -> ComposeResult:
        yield Static("Approval required", id="approval-title")
        yield Static("", id="approval-summary")
        yield Static("", id="approval-copy")
        with Horizontal(id="approval-actions"):
            yield Button("Allow once", id="approval-allow-once", variant="primary")
            yield Button("Deny", id="approval-deny", variant="error")
            yield Button("Review details", id="approval-details")

    def on_mount(self) -> None:
        self.display = False

    def set_approval(self, approval: Optional[Dict[str, Any]]) -> None:
        """Update the card with the latest approval request."""
        has_approval = bool(approval)
        self.display = has_approval
        if not has_approval:
            return

        summary = approval.get("summary", "Approval required")
        details = approval.get("details", "")
        allow_label = approval.get("allow_label", "Allow once")

        self.query_one("#approval-summary", Static).update(summary)
        self.query_one("#approval-copy", Static).update(details)
        self.query_one("#approval-details", Button).label = approval.get("details_label", "Review details")
        self.query_one("#approval-allow-once", Button).label = allow_label
