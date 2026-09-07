"""Single-item Allow/Deny card for an agent-initiated skill install.

Distinct from ChatApprovalCard (the MCP batch card): one URL, one boolean
decision, no per-row Selects and no 5-way vocabulary. The URL is
agent/attacker-influenced, so it is rendered with markup=False.

TASK-910: this card's ``InstallDecided`` message must echo back the exact
``request_id`` the pending confirm's payload carried (see ``set_install``),
mirroring ``SkillScriptConfirmCard.ScriptDecided``'s identical contract.
``ConsoleChatController.resolve_pending_skill_install`` performs a strict
match against the currently-armed round's id and silently drops any resolve
that doesn't carry it -- this guards against a stale button press from a
just-torn-down round resolving a different (possibly different-session)
round the user never saw.
"""

from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, Static


class SkillInstallConfirmCard(Container):
    """Prompts the user to allow/deny installing a skill from a URL."""

    #: Class-level default so an `InstallDecided` can never be posted with
    #: an `AttributeError` even if `on_button_pressed` were somehow reached
    #: before `on_mount`/`set_install` ran.
    _request_id: Optional[str] = None

    class InstallDecided(Message):
        """Posted when the user allows or denies the install."""

        def __init__(self, allow: bool, request_id: Optional[str] = None) -> None:
            """Initialize the decision payload.

            Args:
                allow: True to allow the pending install, False to deny it.
                request_id: The pending confirm round's id, as read from
                    the payload passed to `set_install`. Must be echoed
                    back unchanged to
                    `ConsoleChatController.resolve_pending_skill_install`,
                    or that call silently drops the decision.
            """
            self.allow = allow
            self.request_id = request_id
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
        """Show the card for ``payload`` (``{"url": ...}``), or hide it if None.

        Stores ``payload["request_id"]`` so it can be echoed back on
        ``InstallDecided`` -- see the class docstring for why this id must
        survive the round-trip unchanged.

        Args:
            payload: The pending confirm's ``{"url", "timeout_seconds",
                "request_id"}`` dict to render, or None to hide the card.
        """
        if not payload:
            self.display = False
            self._request_id = None
            return
        self._request_id = payload.get("request_id")
        self.query_one("#skill-install-url", Static).update(str(payload.get("url", "")))
        self.display = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skill-install-allow":
            event.stop()
            self.display = False
            self.post_message(self.InstallDecided(True, request_id=self._request_id))
        elif event.button.id == "skill-install-deny":
            event.stop()
            self.display = False
            self.post_message(self.InstallDecided(False, request_id=self._request_id))
