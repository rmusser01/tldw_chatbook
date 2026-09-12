"""Reusable speech-only controls for both Buddy conversation and workspace modals."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static


class BuddySpeechControls(Vertical):
    """Project an app-owned queue; unmounting this widget never stops playback."""

    DEFAULT_CSS = """
    BuddySpeechControls { height: auto; }
    BuddySpeechControls Static { height: auto; }
    BuddySpeechControls Horizontal { height: auto; }
    BuddySpeechControls Button { min-width: 8; }
    """

    def __init__(self, coordinator: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.coordinator = coordinator

    def compose(self) -> ComposeResult:
        yield Static("", id="buddy-speech-status", markup=False)
        with Horizontal(id="buddy-speech-actions"):
            yield Button("Pause", id="buddy-speech-pause", compact=True)
            yield Button("Skip", id="buddy-speech-skip", compact=True)
            yield Button("Mute", id="buddy-speech-mute", compact=True)
            yield Button("Confirm speech", id="buddy-speech-confirm", compact=True)

    def on_mount(self) -> None:
        self.refresh_state()
        self.set_interval(0.5, self.refresh_state)

    def refresh_state(self) -> None:
        state = self.coordinator.queue.state
        status = self.coordinator.notice or state.error
        if state.input_active:
            status = "Microphone active; speech waits"
        if not status:
            status = (
                "Speech muted"
                if state.muted
                else "Speech paused · Resume restarts this update"
                if state.paused
                else f"Speaking: {state.current_title}"
                if state.current_title
                else "Buddy speech enabled"
                if self.coordinator.enabled
                else "Buddy speech is off"
            )
        self.query_one("#buddy-speech-status", Static).update(status)
        self.query_one("#buddy-speech-actions").display = bool(
            self.coordinator.enabled
            or self.coordinator.needs_consent
            or state.current_title
            or state.queued
            or state.error
            or self.coordinator.notice
            or state.paused
            or state.muted
            or state.input_active
        )
        pause = self.query_one("#buddy-speech-pause", Button)
        pause.label = "Resume" if state.paused else "Pause"
        pause.disabled = not self.coordinator.enabled or state.input_active
        self.query_one("#buddy-speech-mute", Button).label = (
            "Unmute" if state.muted else "Mute"
        )
        self.query_one("#buddy-speech-skip", Button).disabled = not (
            state.current_title or state.queued
        )
        self.query_one(
            "#buddy-speech-confirm", Button
        ).display = self.coordinator.needs_consent

    @on(Button.Pressed)
    def _control(self, event: Button.Pressed) -> None:
        action = event.button.id
        if not action or not action.startswith("buddy-speech-"):
            return
        event.stop()
        queue = self.coordinator.queue
        if action == "buddy-speech-pause":
            (queue.resume if queue.state.paused else queue.pause)()
        elif action == "buddy-speech-skip":
            queue.skip()
        elif action == "buddy-speech-mute":
            (queue.unmute if queue.state.muted else queue.mute)()
        elif action == "buddy-speech-confirm":
            self.coordinator.request_consent()
        self.refresh_state()
