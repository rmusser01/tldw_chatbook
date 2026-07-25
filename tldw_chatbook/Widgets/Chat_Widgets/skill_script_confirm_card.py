"""Allow / Always-allow / Deny card for running a skill's bundled script.

The skill name, script path, and args are agent-influenced, so every Static
renders with markup=False.

This card's ``ScriptDecided`` message must echo back the exact
``request_id`` the pending confirm's payload carried (see ``set_script``).
``ConsoleChatController.resolve_pending_skill_script`` (task-5,
``tldw_chatbook/Chat/console_chat_controller.py``) performs a strict match
against the currently-armed round's id and silently drops any resolve that
doesn't carry it -- this guards against a stale button press from a
just-torn-down round authorizing a script the user never saw.
"""

from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, Static


class SkillScriptConfirmCard(Container):
    """Prompts the user to allow or deny running a skill's script."""

    #: Class-level default so a `ScriptDecided` can never be posted with an
    #: `AttributeError` even if `on_button_pressed` were somehow reached
    #: before `on_mount`/`set_script` ran.
    _request_id: Optional[str] = None

    class ScriptDecided(Message):
        """Posted when the user allows, always-allows, or denies the run."""

        def __init__(
            self, allow: bool, remember: bool, request_id: Optional[str] = None
        ) -> None:
            """Initialize the decision payload.

            Args:
                allow: True to run the script this once.
                remember: True to also grant this skill standing permission.
                request_id: The pending confirm round's id, as read from
                    the payload passed to `set_script`. Must be echoed
                    back unchanged to
                    `ConsoleChatController.resolve_pending_skill_script`,
                    or that call silently drops the decision.
            """
            self.allow = allow
            self.remember = remember
            self.request_id = request_id
            super().__init__()

    def compose(self) -> ComposeResult:
        """Build the card's prompt text, target/args statics, and button row.

        Returns:
            The composed child widgets.
        """
        yield Static(
            "An agent wants to run a script from a skill:",
            id="skill-script-prompt",
            markup=False,
        )
        yield Static("", id="skill-script-target", markup=False)
        yield Static("", id="skill-script-args", markup=False)
        yield Static(
            "It runs with a scrubbed environment in a temporary folder (not "
            "the skill's own folder); only its output comes back.",
            id="skill-script-note",
            markup=False,
        )
        yield Horizontal(
            Button("Allow once", id="skill-script-allow", variant="primary"),
            Button("Always allow this skill", id="skill-script-always"),
            Button("Deny", id="skill-script-deny", variant="error"),
            id="skill-script-buttons",
        )

    def on_mount(self) -> None:
        """Hide the card until `set_script` is called with a payload."""
        self.display = False

    def set_script(self, payload: dict[str, Any] | None) -> None:
        """Show the card for ``payload``, or hide it if None.

        Stores ``payload["request_id"]`` so it can be echoed back on
        ``ScriptDecided`` -- see the class docstring for why this id must
        survive the round-trip unchanged.

        Args:
            payload: The pending confirm's dict ({"skill_name",
                "script_path", "mechanism", "interpreter", "is_binary",
                "args", "timeout_seconds", "request_id"}), or None to hide
                the card.
        """
        if not payload:
            self.display = False
            self._request_id = None
            return
        self._request_id = payload.get("request_id")
        skill_name = str(payload.get("skill_name", ""))
        script_path = str(payload.get("script_path", ""))
        mechanism = str(payload.get("mechanism", ""))
        interpreter = str(payload.get("interpreter", ""))
        if mechanism == "direct-exec":
            how = "runs directly"
            if payload.get("is_binary"):
                how = "runs directly (a binary you cannot review as text)"
        else:
            how = f"runs with {interpreter}"
        self.query_one("#skill-script-target", Static).update(
            f"{skill_name} — {script_path} ({how})"
        )
        args = payload.get("args") or []
        self.query_one("#skill-script-args", Static).update(
            ("arguments: " + " ".join(str(a) for a in args)) if args else "no arguments"
        )
        self.display = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate a button press into a `ScriptDecided` message.

        Args:
            event: The Textual button-pressed event.
        """
        decisions = {
            "skill-script-allow": (True, False),
            "skill-script-always": (True, True),
            "skill-script-deny": (False, False),
        }
        decision = decisions.get(event.button.id or "")
        if decision is None:
            return
        event.stop()
        self.display = False
        allow, remember = decision
        self.post_message(
            self.ScriptDecided(allow, remember, request_id=self._request_id)
        )
