"""Allow / Allow-for-session / Deny card for agent-initiated chat creation.

Renders the ``fork_chat`` / ``new_chat`` confirm payload the Console's chat
controller arms (task-6 of the agent-chat-fork-spawn plan). The title,
opening prompt, instructions, and fork-source title are agent-influenced,
so every Static renders with markup=False -- the same consent-surface rule
as ``SkillScriptConfirmCard``.

This card's ``ChatCreateDecided`` message must echo back the exact
``request_id`` the pending confirm's payload carried (see ``set_payload``).
``ConsoleChatController.resolve_pending_chat_create`` (task-5,
``tldw_chatbook/Chat/console_chat_controller.py``) performs a strict match
against the currently-armed round's id and silently drops any resolve that
doesn't carry it -- this guards against a stale button press from a
just-torn-down round creating a chat the user never saw.
"""

from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, Static


class ChatCreateConfirmCard(Container):
    """Prompts the user to allow or deny creating a forked or new chat."""

    #: Class-level defaults so a `ChatCreateDecided` can never be posted
    #: with an `AttributeError` even if `on_button_pressed` were somehow
    #: reached before `on_mount`/`set_payload` ran, and so `on_mount` can
    #: read `_payload` on a card whose payload has not been set yet.
    _request_id: Optional[str] = None
    _payload: Optional[dict[str, Any]] = None

    class ChatCreateDecided(Message):
        """Posted when the user allows, allows-for-session, or denies."""

        def __init__(
            self, allow: bool, remember: bool, request_id: Optional[str] = None
        ) -> None:
            """Initialize the decision payload.

            Args:
                allow: True to create the chat this once.
                remember: True to also grant this session standing
                    permission (no further cards this session).
                request_id: The pending confirm round's id, as read from
                    the payload passed to `set_payload`. Must be echoed
                    back unchanged to
                    `ConsoleChatController.resolve_pending_chat_create`,
                    or that call silently drops the decision.
            """
            self.allow = allow
            self.remember = remember
            self.request_id = request_id
            super().__init__()

    def compose(self) -> ComposeResult:
        """Build the card's header, body, and button row."""
        yield Static("", id="chat-create-header", markup=False)
        yield Static("", id="chat-create-body", markup=False)
        yield Horizontal(
            Button("Allow", id="chat-create-allow", variant="primary"),
            Button("Allow for this session", id="chat-create-allow-remember"),
            Button("Deny", id="chat-create-deny", variant="error"),
            id="chat-create-buttons",
        )

    def on_mount(self) -> None:
        """Hide the card until `set_payload` is called with a payload.

        A payload that arrived before mount (``set_payload`` stores it
        without rendering, since the Static children do not exist yet)
        is rendered and shown here instead.
        """
        if self._payload is not None:
            self._refresh_statics()
            self.display = True
        else:
            self.display = False

    def set_payload(self, payload: dict[str, Any] | None) -> None:
        """Show the card for ``payload``, or hide it if None.

        Stores ``payload["request_id"]`` so it can be echoed back on
        ``ChatCreateDecided`` -- see the class docstring for why this id
        must survive the round-trip unchanged. Also safe before mount
        (the Static children do not exist yet); the stored payload is
        rendered by ``on_mount`` in that case.

        Args:
            payload: The pending confirm's dict ({"tool", "title",
                "opening_prompt", "instructions", "run_id", and for
                fork_chat "fork_source_title"/"fork_message_count" when the
                controller's enrichment succeeded, "timeout_seconds",
                "request_id"}), or None to hide the card.
        """
        if not payload:
            self._payload = None
            self._request_id = None
            self.display = False
            return
        self._payload = payload
        self._request_id = payload.get("request_id")
        if self.is_mounted:
            self._refresh_statics()
        self.display = True

    def _refresh_statics(self) -> None:
        """Render the stored payload onto the header/body Statics."""
        self.query_one("#chat-create-header", Static).update(self._header_text())
        self.query_one("#chat-create-body", Static).update(self._body_text())

    def _header_text(self) -> str:
        """Return the one-line summary naming the action and target title."""
        assert self._payload is not None
        verb = (
            "fork this chat"
            if self._payload.get("tool") == "fork_chat"
            else "create a new chat"
        )
        return f"An agent wants to {verb}: {self._payload.get('title', '')}"

    def _body_text(self) -> str:
        """Return the fork facts plus the draft prompt and instructions."""
        assert self._payload is not None
        lines: list[str] = []
        if self._payload.get("tool") == "fork_chat":
            # Final-review fix wave (Finding 1): render the fork line only
            # when the controller's enrichment keys are PRESENT. An
            # un-enriched payload (or a degraded enrichment) must omit the
            # line entirely rather than render the dead
            # "Copies ? messages from ''". A present count of 0 still
            # renders -- "Copies 0 messages" is the honest summary of an
            # empty fork source.
            count = self._payload.get("fork_message_count")
            source_title = self._payload.get("fork_source_title")
            if count is not None or source_title:
                count_text = str(count) if count is not None else "?"
                lines.append(
                    f"Copies {count_text} messages "
                    f"from {str(source_title or '')!r} into the new chat."
                )
        run_id = self._payload.get("run_id")
        if run_id:
            lines.append(f"Requested by agent run {run_id}.")
        if self._payload.get("opening_prompt"):
            lines.append(
                "Opening prompt (draft for the input box):\n"
                f"{self._payload['opening_prompt']}"
            )
        if self._payload.get("instructions"):
            lines.append(f"System prompt:\n{self._payload['instructions']}")
        return "\n\n".join(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate a button press into a `ChatCreateDecided` message.

        Args:
            event: The Textual button-pressed event.
        """
        decisions = {
            "chat-create-allow": (True, False),
            "chat-create-allow-remember": (True, True),
            "chat-create-deny": (False, False),
        }
        decision = decisions.get(event.button.id or "")
        if decision is None:
            return
        event.stop()
        allow, remember = decision
        self._decide(allow=allow, remember=remember)

    def _decide(self, *, allow: bool, remember: bool) -> None:
        """Tear the card down and post the decision with its round id.

        Args:
            allow: True to allow the chat creation.
            remember: True to also grant standing session permission.
        """
        if self._payload is None:
            return
        request_id = self._request_id
        self.set_payload(None)
        self.post_message(
            self.ChatCreateDecided(allow, remember, request_id=request_id)
        )
