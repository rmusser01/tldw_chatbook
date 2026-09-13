"""Console speech switches displayed beside the Workbench status."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Static, Switch

#: TASK-32015: the two voice switches sit side by side with no other
#: affordance explaining them, so each tooltip carries the three facts a
#: first-time user cannot discover elsewhere: scope (per-conversation),
#: the relationship between the features (hands-free owns reply speech
#: while it runs -- the two never double-speak), and the interrupt/exit
#: keys that need no voice round-trip. Pinned by
#: `test_voice_switch_tooltips_state_relationship_and_interrupts`.
AUTO_SPEAK_SWITCH_TOOLTIP = (
    "Speak new assistant replies in this conversation (per-conversation, "
    "consent-gated). While Hands-free is on, that mode speaks replies."
)
HANDS_FREE_SWITCH_TOOLTIP = (
    "Voice conversation loop: speak, auto-send after a pause, hear the "
    "reply (Ctrl+Shift+H). Type to interrupt a spoken reply; Esc exits."
)


class ConsoleHandsFreeToggleRequested(Message):
    """User requested a visible Hands-free state change."""

    def __init__(self, enabled: bool) -> None:
        """Initialize the request.

        Args:
            enabled: Requested Hands-free state.
        """
        super().__init__()
        self.enabled = enabled


class ConsoleAutoSpeakChanged(Message):
    """User requested a durable Speak replies state change."""

    def __init__(self, enabled: bool) -> None:
        """Initialize the request.

        Args:
            enabled: Requested automatic-reply-speech state.
        """
        super().__init__()
        self.enabled = enabled


class ConsoleSpeechControls(Horizontal):
    """Keep Console speech switches visible immediately before status."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize authoritative speech-control presentation state.

        Args:
            **kwargs: Additional Textual widget arguments.
        """
        super().__init__(**kwargs)
        self.auto_speak_enabled = False
        self.auto_speak_paused = False
        self.hands_free_active = False
        self.styles.width = "auto"
        self.styles.height = 1
        self.styles.min_height = 1
        self.styles.max_height = 1

    def compose(self) -> ComposeResult:
        with Horizontal(id="console-auto-speak-control") as auto_speak_control:
            auto_speak_control.styles.width = "auto"
            auto_speak_control.styles.height = 1
            auto_speak_label = Static(
                "Speak replies",
                id="console-auto-speak-label",
                markup=False,
            )
            auto_speak_label.styles.width = "auto"
            yield auto_speak_label
            auto_speak_switch = Switch(
                False,
                name="Speak replies",
                id="console-auto-speak",
                tooltip=AUTO_SPEAK_SWITCH_TOOLTIP,
            )
            self._size_switch(auto_speak_switch)
            yield auto_speak_switch
        with Horizontal(id="console-hands-free-control") as hands_free_control:
            hands_free_control.styles.width = "auto"
            hands_free_control.styles.height = 1
            hands_free_label = Static(
                "Hands-free",
                id="console-hands-free-label",
                markup=False,
            )
            hands_free_label.styles.width = "auto"
            yield hands_free_label
            hands_free_switch = Switch(
                False,
                name="Hands-free",
                id="console-hands-free-switch",
                tooltip=HANDS_FREE_SWITCH_TOOLTIP,
            )
            self._size_switch(hands_free_switch)
            yield hands_free_switch

    @staticmethod
    def _size_switch(switch: Switch) -> None:
        """Keep a switch to one terminal row in every stylesheet host."""
        switch.styles.width = 5
        switch.styles.height = 1
        switch.styles.min_height = 1
        switch.styles.max_height = 1
        switch.styles.padding = 0
        switch.styles.border = ("none", "transparent")

    def on_mount(self) -> None:
        """Apply state received before the composed children were mounted."""
        self.sync_auto_speak(
            enabled=self.auto_speak_enabled,
            paused=self.auto_speak_paused,
        )
        self.sync_hands_free_state(self.hands_free_active)

    def sync_auto_speak(self, *, enabled: bool, paused: bool) -> None:
        """Silently repaint the persisted reply-speech state.

        Args:
            enabled: Whether automatic reply speech is enabled.
            paused: Whether automatic speech is paused after a failure.
        """
        self.auto_speak_enabled = enabled is True
        self.auto_speak_paused = paused is True
        try:
            switch = self.query_one("#console-auto-speak", Switch)
        except NoMatches:
            return
        if switch.value is not self.auto_speak_enabled:
            with self.prevent(Switch.Changed):
                switch.value = self.auto_speak_enabled
        switch.disabled = False
        switch.tooltip = (
            "Automatic speech is paused after a failure."
            if self.auto_speak_paused
            else AUTO_SPEAK_SWITCH_TOOLTIP
        )

    def sync_hands_free_state(self, active: bool) -> None:
        """Silently mirror the live Hands-free session state.

        Args:
            active: Whether a Hands-free session is active.
        """
        self.hands_free_active = active is True
        try:
            switch = self.query_one("#console-hands-free-switch", Switch)
        except NoMatches:
            return
        if switch.value is not self.hands_free_active:
            with self.prevent(Switch.Changed):
                switch.value = self.hands_free_active

    @on(Switch.Changed, "#console-hands-free-switch")
    def on_console_hands_free_switch_changed(self, event: Switch.Changed) -> None:
        """Forward one user Hands-free gesture.

        Args:
            event: Textual switch-change event.
        """
        event.stop()
        self.post_message(ConsoleHandsFreeToggleRequested(event.value))

    @on(Switch.Changed, "#console-auto-speak")
    def on_console_auto_speak_changed(self, event: Switch.Changed) -> None:
        """Request persistence while retaining authoritative presentation.

        Args:
            event: Textual switch-change event.
        """
        if event.value is self.auto_speak_enabled:
            return
        event.stop()
        with self.prevent(Switch.Changed):
            event.switch.value = self.auto_speak_enabled
        self.post_message(ConsoleAutoSpeakChanged(event.value))
