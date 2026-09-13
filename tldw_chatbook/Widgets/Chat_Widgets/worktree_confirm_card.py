"""Disposable, exact-round confirmation for recorded agent work."""

from __future__ import annotations

import unicodedata
from typing import Any

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static

from .chat_task_cards import ChatTaskCards


def plain(value: object, *, multiline: bool = False) -> str:
    """Escape control characters without interpreting terminal or Rich markup."""
    return "".join(
        char
        if (multiline and char == "\n")
        or not unicodedata.category(char).startswith("C")
        else repr(char)[1:-1]
        for char in str(value or "")
    )


class _DecisionButton(Button):
    def __init__(self, label: str, request_id: str, allow: bool) -> None:
        super().__init__(label, classes="worktree-allow" if allow else "worktree-deny")
        self.request_id = request_id
        self.allow = allow


class WorktreeConfirmCard(Vertical):
    """A round owns its buttons; queued old presses cannot relabel consent."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.display = False
        self._payload = None
        self._answered = False

    def compose(self):
        yield Static("", id="worktree-action", markup=False)
        with VerticalScroll(id="worktree-details"):
            yield Static("", id="worktree-source", markup=False)
            yield Static("", id="worktree-destination", markup=False)
            yield Static("", id="worktree-diffstat", markup=False)
            yield Static("", id="worktree-consequence", markup=False)
        yield Horizontal(id="worktree-buttons")

    def set_confirmation(self, payload: dict[str, Any] | None) -> None:
        if payload == self._payload:
            return
        previous_id = (self._payload or {}).get("request_id")
        self._payload = dict(payload) if payload else None
        self.display = bool(payload)
        if not self.is_mounted:
            self.call_after_refresh(self._render_payload)
        else:
            self._render_payload(previous_id)

    def _render_payload(self, previous_id=None):
        payload = self._payload or {}
        request_id = payload.get("request_id")
        if request_id == previous_id:
            return
        self._answered = False
        action = payload.get("action", payload.get("mode", "apply"))
        sentence = {
            "apply": "Apply agent changes as unstaged edits?",
            "merge": "Merge agent work into this repository?",
            "discard": "Discard agent changes and remove its branch?",
        }.get(action, "Confirm agent work operation?")
        for selector, value in (
            ("#worktree-action", sentence),
            (
                "#worktree-source",
                "Source: " + plain(payload.get("source", payload.get("worktree", ""))),
            ),
            (
                "#worktree-destination",
                "Destination: " + plain(payload.get("destination", "")),
            ),
            (
                "#worktree-diffstat",
                plain(str(payload.get("diffstat", ""))[:8192], multiline=True),
            ),
            (
                "#worktree-consequence",
                "A detached baseline checkout is retained."
                if action == "discard"
                else "Source checkout is retained.",
            ),
        ):
            self.query_one(selector, Static).update(value)
        buttons = self.query_one("#worktree-buttons", Horizontal)
        buttons.remove_children()
        if isinstance(request_id, str) and request_id:
            buttons.mount(
                _DecisionButton("Allow once", request_id, True),
                _DecisionButton("Deny", request_id, False),
            )

    @on(Button.Pressed)
    def decide(self, event: Button.Pressed) -> None:
        event.stop()
        button = event.button
        if (
            not isinstance(button, _DecisionButton)
            or self._answered
            or button.request_id != (self._payload or {}).get("request_id")
        ):
            return
        self._answered = True
        for control in self.query(Button):
            control.disabled = True
        self.post_message(
            ChatTaskCards.WorktreeDecided(button.allow, button.request_id)
        )
