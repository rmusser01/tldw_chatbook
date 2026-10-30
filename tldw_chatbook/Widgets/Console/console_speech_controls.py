"""Console speech switches displayed beside the Workbench status."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.cells import cell_len
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.content import Content
from textual.css.query import NoMatches
from textual.geometry import Size
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static, Switch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from textual.app import RenderResult

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
#: TASK-33626.1: shown while the loop waits on the speech model.
HANDS_FREE_STARTING_TOOLTIP = (
    "Hands-free is starting: the speech model is loading. Press Enter or Esc "
    "to cancel."
)

#: TASK-33626.1 (DESIGN.md: a toggle is paired with a text-state word). The
#: header read "Speak replies ▌  Hands-free ▌" for both Off and On; the word
#: is now the state, so the row's text differs between them.
STATE_OFF = "Off"
STATE_ON = "On"
STATE_PAUSED = "Paused"
STATE_STARTING = "Starting…"


class ConsoleStateToggle(Switch):
    """A one-row Switch whose face is its state word, not a slider track.

    Keeps Switch's keys (Enter/Space), click, ``value`` and ``Changed`` so
    every caller of the header controls is unchanged. A squeezed Switch
    track measured 1.42:1 on the header, and focus replaced the knob with
    the app's ``*:focus`` outline, so Off and On looked alike; the word is
    styled by `_console.tcss` (`.-on`, `.-paused`, `.-starting`, `:focus`).
    """

    state_word: reactive[str] = reactive(STATE_OFF, layout=True)

    def render(self) -> "RenderResult":
        """Paint the state word."""
        return Content(self.state_word)

    def get_content_width(self, container: Size, viewport: Size) -> int:
        """Size to the word, so Paused and Starting… are never clipped."""
        return cell_len(self.state_word)

    def show_state(self, word: str, *, paused: bool = False, starting: bool = False) -> None:
        """Set the word and the state classes the stylesheet colours.

        Args:
            word: One of the ``STATE_*`` words.
            paused: Auto-speak is on but paused after a failure.
            starting: Hands-free is on but waiting for the speech model.
        """
        self.state_word = word
        self.set_class(paused, "-paused")
        self.set_class(starting, "-starting")


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
        self.hands_free_starting = False
        self.remove_class(*(name for name in self.classes if name.startswith("w-")))
        self.set_styles(width=None)
        self.add_class("w-auto")
        self.remove_class(*(name for name in self.classes if name.startswith("h-")))
        self.set_styles(height=None)
        self.add_class("h-1")
        self.styles.min_height = 1
        self.styles.max_height = 1

    def compose(self) -> ComposeResult:
        with Horizontal(id="console-auto-speak-control") as auto_speak_control:
            auto_speak_control.remove_class(*(name for name in auto_speak_control.classes if name.startswith("w-")))
            auto_speak_control.set_styles(width=None)
            auto_speak_control.add_class("w-auto")
            auto_speak_control.remove_class(*(name for name in auto_speak_control.classes if name.startswith("h-")))
            auto_speak_control.set_styles(height=None)
            auto_speak_control.add_class("h-1")
            auto_speak_label = Static(
                "Speak replies:",
                id="console-auto-speak-label",
                markup=False,
            )
            auto_speak_label.remove_class(*(name for name in auto_speak_label.classes if name.startswith("w-")))
            auto_speak_label.set_styles(width=None)
            auto_speak_label.add_class("w-auto")
            yield auto_speak_label
            auto_speak_switch = ConsoleStateToggle(
                False,
                animate=False,
                name="Speak replies",
                id="console-auto-speak",
                tooltip=AUTO_SPEAK_SWITCH_TOOLTIP,
            )
            self._size_switch(auto_speak_switch)
            yield auto_speak_switch
        with Horizontal(id="console-hands-free-control") as hands_free_control:
            hands_free_control.remove_class(*(name for name in hands_free_control.classes if name.startswith("w-")))
            hands_free_control.set_styles(width=None)
            hands_free_control.add_class("w-auto")
            hands_free_control.remove_class(*(name for name in hands_free_control.classes if name.startswith("h-")))
            hands_free_control.set_styles(height=None)
            hands_free_control.add_class("h-1")
            hands_free_label = Static(
                "Hands-free:",
                id="console-hands-free-label",
                markup=False,
            )
            hands_free_label.remove_class(*(name for name in hands_free_label.classes if name.startswith("w-")))
            hands_free_label.set_styles(width=None)
            hands_free_label.add_class("w-auto")
            yield hands_free_label
            hands_free_switch = ConsoleStateToggle(
                False,
                animate=False,
                name="Hands-free",
                id="console-hands-free-switch",
                tooltip=HANDS_FREE_SWITCH_TOOLTIP,
            )
            self._size_switch(hands_free_switch)
            yield hands_free_switch

    @staticmethod
    def _size_switch(switch: Switch) -> None:
        """Keep a toggle to one terminal row in every stylesheet host.

        Width follows the state word and padding comes from `_console.tcss`:
        the old ``w-5``/``p-0`` utility classes are ``!important`` and would
        clip "Paused" and "Starting…".
        """
        switch.remove_class(*(name for name in switch.classes if name.startswith("w-")))
        switch.set_styles(width=None)
        switch.add_class("w-auto")
        switch.remove_class(*(name for name in switch.classes if name.startswith("h-")))
        switch.set_styles(height=None)
        switch.add_class("h-1")
        switch.styles.min_height = 1
        switch.styles.max_height = 1
        switch.add_class("border-none")

    def on_mount(self) -> None:
        """Apply state received before the composed children were mounted."""
        self.sync_auto_speak(
            enabled=self.auto_speak_enabled,
            paused=self.auto_speak_paused,
        )
        self.sync_hands_free_state(
            self.hands_free_active, starting=self.hands_free_starting
        )

    def sync_auto_speak(self, *, enabled: bool, paused: bool) -> None:
        """Silently repaint the persisted reply-speech state.

        Args:
            enabled: Whether automatic reply speech is enabled.
            paused: Whether automatic speech is paused after a failure.
        """
        self.auto_speak_enabled = enabled is True
        self.auto_speak_paused = paused is True
        try:
            switch = self.query_one("#console-auto-speak", ConsoleStateToggle)
        except NoMatches:
            return
        if switch.value is not self.auto_speak_enabled:
            with self.prevent(Switch.Changed):
                switch.value = self.auto_speak_enabled
        switch.disabled = False
        # TASK-33626.1 AC#5: paused auto-speak reads Paused, not On.
        paused = self.auto_speak_enabled and self.auto_speak_paused
        switch.show_state(
            STATE_PAUSED if paused else STATE_ON if self.auto_speak_enabled else STATE_OFF,
            paused=paused,
        )
        switch.tooltip = (
            "Automatic speech is paused after a failure."
            if self.auto_speak_paused
            else AUTO_SPEAK_SWITCH_TOOLTIP
        )

    def sync_hands_free_state(self, active: bool, *, starting: bool = False) -> None:
        """Silently mirror the live Hands-free session state.

        Args:
            active: Whether a Hands-free session is active.
            starting: Whether that session is still waiting on the speech
                model (the microphone has not opened yet).
        """
        self.hands_free_active = active is True
        self.hands_free_starting = self.hands_free_active and starting is True
        try:
            switch = self.query_one("#console-hands-free-switch", ConsoleStateToggle)
        except NoMatches:
            return
        if switch.value is not self.hands_free_active:
            with self.prevent(Switch.Changed):
                switch.value = self.hands_free_active
        switch.show_state(
            STATE_STARTING
            if self.hands_free_starting
            else STATE_ON if self.hands_free_active else STATE_OFF,
            starting=self.hands_free_starting,
        )
        switch.tooltip = (
            HANDS_FREE_STARTING_TOOLTIP
            if self.hands_free_starting
            else HANDS_FREE_SWITCH_TOOLTIP
        )

    @on(Switch.Changed, "#console-hands-free-switch")
    def on_console_hands_free_switch_changed(self, event: Switch.Changed) -> None:
        """Forward one user Hands-free gesture.

        Args:
            event: Textual switch-change event.
        """
        event.stop()
        # TASK-33626.1: like Speak replies, the control keeps showing the
        # session's real state. Turning on now asks for consent first, and
        # entry can still be refused, so the request must not repaint it On.
        if event.value is not self.hands_free_active:
            with self.prevent(Switch.Changed):
                event.switch.value = self.hands_free_active
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
