"""Single-item Allow/Deny card for an agent-initiated skill install.

Distinct from ChatApprovalCard (the MCP batch card): one URL, one boolean
decision, no per-row Selects and no 5-way vocabulary. The URL is
agent/attacker-influenced, so it is rendered with markup=False.
"""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, Static


class SkillInstallConfirmCard(Container):
    """Prompts the user to allow/deny installing a skill from a URL."""

    class InstallDecided(Message):
        """Posted when the user allows or denies the install."""

        def __init__(self, allow: bool) -> None:
            self.allow = allow
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Static(
            "An agent wants to install a skill:",
            id="skill-install-prompt",
            markup=False,
        )
        yield Static("", id="skill-install-url", markup=False)
        yield Static(
            "It will be installed pending your review and cannot run until "
            "you approve it in Library > Skills.",
            id="skill-install-note",
            markup=False,
        )
        yield Horizontal(
            Button("Allow", id="skill-install-allow", variant="primary"),
            Button("Deny", id="skill-install-deny", variant="error"),
            id="skill-install-buttons",
        )

    def on_mount(self) -> None:
        self.display = False

    def set_install(self, payload: dict[str, Any] | None) -> None:
        """Show the card for ``payload`` (``{"url": ...}``), or hide it if None."""
        if not payload:
            self.display = False
            return
        self.query_one("#skill-install-url", Static).update(str(payload.get("url", "")))
        self.display = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skill-install-allow":
            event.stop()
            self.display = False
            self.post_message(self.InstallDecided(True))
        elif event.button.id == "skill-install-deny":
            event.stop()
            self.display = False
            self.post_message(self.InstallDecided(False))
