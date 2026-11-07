"""Console speech switches and composer voice-status presentation."""

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

    from .console_composer_bar import ConsoleComposerBar

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


def composer_voice_chip_width_cap(self: ConsoleComposerBar) -> int:
    """Return the max cells the voice chip may take (0 = hide it).

    TASK-24620: mirrors `_send_reason_width_cap` (TASK-24415). The chip
    sized itself against the composer's FULL width with only
    `VOICE_CHIP_MIN_WIDTH` reserved, so a long state message took up to
    53 cells and the ``1fr`` draft (``min_width: 0``) starved beside it
    -- the same starvation class the reason strip had. The budget is
    whatever the live row can spare after the fixed furniture and the
    draft floor; below a legible remainder the chip hides rather than
    blind the composer (the Dictate button's own label still says
    "Dictating", so the mic-live state survives).

    Returns:
        0 when the row cannot spare a legible chip (hide it), else a
        cap of at most ``VOICE_CHIP_MAX_WIDTH`` cells.
    """
    from . import console_composer_bar as _owner

    try:
        row = self.query_one("#console-composer-expanded", _owner.Horizontal)
    except _owner.NoMatches:
        return self.VOICE_CHIP_MAX_WIDTH
    row_width = int(row.content_region.width)
    if row_width <= 0:
        # Not laid out yet; the next resize re-derives the cap.
        return self.VOICE_CHIP_MAX_WIDTH
    reserved = (
        self.LEFT_CLUSTER_WIDTH
        + self._actions_row_width()
        + self.ADVISORY_MARGIN_ALLOWANCE
    )
    budget = row_width - reserved - self.DRAFT_MIN_RENDER_WIDTH
    if budget < self.VOICE_CHIP_MIN_LEGIBLE_WIDTH:
        return 0
    return min(self.VOICE_CHIP_MAX_WIDTH, budget)


def set_composer_voice_status(
    self: ConsoleComposerBar,
    state: str,
    *,
    partial: str = "",
    elapsed_seconds: int = 0,
    message: str = "",
    segment_transcribing: bool = False,
) -> None:
    """Render the dictation state into the inline voice chip.

    Every write goes through `textual.content.Content`, never a bare string:
    a `Static` parses strings as Textual markup, and `rich.markup.escape`
    (which used to guard this) only escapes tags opening with `[a-z#/@]`.
    Whisper's own tokens are uppercase, so `[BLANK_AUDIO]` and `[Music]`
    survived escaping untouched and were then *stripped at paint time* --
    the chip showed "● 0:03    hi" for "● 0:03  [BLANK_AUDIO] [Music] hi".
    `Content` carries plain text with no markup semantics at all, so it
    fixes the swallowing and the opposite failure (`[/tmp/x]` raising
    `MarkupError`) in one move. Nothing in this chip is ever markup: the
    text is either our own constants or recognizer output.

    Args:
        state: One of the `STATE_*` constants from `console_voice_input`.
        partial: In-flight recognizer text. Truncated from the left so the
            newest words stay visible, and dropped entirely on narrow
            terminals so the 1fr draft never collapses.
        elapsed_seconds: Recording duration, rendered as m:ss.
        message: Status or failure text for non-listening states.
        segment_transcribing: True while a per-segment transcription is
            in flight (see `set_voice_segment_transcribing`). Only
            meaningful for `state == "listening"`; takes precedence over
            `partial` there when both happen to be set -- which is a real
            case, not a hypothetical one: an inline command's ack (`¶`,
            e.g.) is left sitting in `_voice_partial` by
            `set_voice_partial`, and an inline command does NOT end the
            capture, so the very next segment's `done=False` can set this
            flag while that ack text is still stored. The chip correctly
            shows the transcribing indication over the stale ack in that
            case; it just means the two are not mutually exclusive the
            way an earlier version of this docstring claimed.
    """
    from . import console_composer_bar as _owner

    _owner._load_console_voice_states()  # TASK-22504
    try:
        chip = self.query_one("#console-voice-status", _owner.Static)
    except _owner.NoMatches:
        return
    # TASK-24620: cache the inputs so `on_resize` can re-derive the
    # chip against the new row width (a shrink must retract the chip,
    # not the draft).
    self._voice_status_last = (
        state,
        partial,
        elapsed_seconds,
        message,
        segment_transcribing,
    )

    if state in ("idle", "unavailable"):
        # PR-2230 review f2: clear the chip-width cache BEFORE the
        # presentation sync -- `_sync_full_width_voice_presentation`
        # re-derives the disabled-reason strip, which subtracts the
        # cache; clearing it after left the strip capped as though the
        # chip were still displayed.
        self._voice_chip_last_width = 0
        self._sync_full_width_voice_presentation(False)
        chip.styles.display = "none"
        _owner._swap_size_class(chip, "w-", "w-0")
        chip.styles.min_width = 0
        _owner._clear_static_content(chip)
        return

    # TASK-24620 / PR-2230 review f3: PREPARING selects its sizing
    # BEFORE the ordinary cap -- it is action feedback for a Dictate
    # press, and its full-width presentation exempts it at every width,
    # not only where the budget hits zero (the 80-column busy-parakeet
    # tests require the WHOLE copy, and intermediate budgets used to
    # truncate it while the presentation collapsed space the chip could
    # have used). The non-listening branch below still shrinks the
    # final width to the message's own length, so short preparing copy
    # does not balloon.
    if state == _owner.STATE_PREPARING:
        total_width = self.size.width or self.VOICE_CHIP_MAX_WIDTH * 2
        width = min(
            self.VOICE_CHIP_MAX_WIDTH,
            max(0, total_width - self.VOICE_CHIP_MIN_WIDTH),
        )
    else:
        # The chip's width is a live-row budget like the send-disabled
        # reason strip's (see `_voice_chip_width_cap`) -- the draft
        # floor wins, and below a legible remainder the chip hides
        # rather than starve it.
        width = self._voice_chip_width_cap()
    if width == 0:
        self._voice_chip_last_width = 0
        self._sync_full_width_voice_presentation(False)
        chip.styles.display = "none"
        _owner._swap_size_class(chip, "w-", "w-0")
        chip.styles.min_width = 0
        _owner._clear_static_content(chip)
        return

    if state == "listening":
        head = f"{_owner.resolve_glyph(_owner.GLYPH_VOICE_RECORDING)} {elapsed_seconds // 60}:{elapsed_seconds % 60:02d}"
        room = width - len(head) - 3
        if segment_transcribing and room > 8:
            # A right-truncating `[-room:]` slice (as used for `partial`
            # just below, correctly -- the newest recognizer words are
            # what matters there) is wrong for a fixed constant: it keeps
            # the TAIL of the label, so at composer widths that land
            # `room` in 9..14 the chip painted "scribing…" -- the label's
            # own trailing ellipsis surviving while its meaningful prefix
            # was cut, reading as garbage rather than a truncated status
            # word (review finding L3). Keep the START instead, and
            # replace the cut-off end with an explicit ellipsis of our
            # own when the label doesn't fit whole -- the label's own
            # trailing "…" only ever survives when nothing was cut.
            label = _owner.resolve_glyph_text(self.VOICE_CHIP_TRANSCRIBING_LABEL)
            if len(label) <= room:
                tail = label
            else:
                tail = f"{label[: max(room - 1, 0)]}…"
            body = f"{head}  {tail}"
        elif partial and room > 8:
            tail = partial[-room:]
            body = f"{head}  {tail}"
        else:
            # Below the floor the counter alone still proves the mic is live.
            body = head
            width = min(width, len(head) + 2)
    else:
        body = message or state
        width = min(width, len(body) + 2)

    # The busy preparation row hides optional chrome. Reserve the actual
    # stable action width plus the chip's three surrounding layout cells.
    full_width_preparing = (
        state == _owner.STATE_PREPARING
        and _owner.cell_len(body) + 2 >= self.VOICE_CHIP_MAX_WIDTH
    )
    if full_width_preparing:
        width = min(
            width,
            max(
                1,
                total_width - self._actions_row_width(attachment_visible=False) - 3,
            ),
        )
        chip.tooltip = body
        if (
            _owner.cell_len(body) + 1 > width
            and body == "Local transcription busy — dictation will run next."
        ):
            compact = "Local transcription busy — queued."
            body = compact if _owner.cell_len(compact) + 1 <= width else "Queued"
    else:
        chip.tooltip = None
    chip.styles.display = "block"
    self._voice_chip_last_width = max(width, 1)
    chip.remove_class(*(name for name in chip.classes if name.startswith("w-")))
    # ds-runtime: Voice status text is fitted into measured remaining composer columns.
    chip.set_styles(width=self._voice_chip_last_width)
    chip.styles.min_width = 0
    chip.remove_class(*(name for name in chip.classes if name.startswith("h-")))
    chip.set_styles(height=None)
    chip.add_class("h-1")
    chip.styles.min_height = 1
    self._sync_full_width_voice_presentation(full_width_preparing)
    if full_width_preparing:
        chip.add_class("p-left-1")
    else:
        chip.remove_class("p-left-1")
    chip.set_styles(margin=None)
    chip.set_class(state == "error", "console-voice-status-error")
    chip.update(_owner.Content(body))
    # TASK-24620: the chip just moved; re-derive the reason strip so it
    # yields whatever the chip now holds (chip has priority).
    self._sync_send_disabled_reason(
        self._send_disabled_reason, muted=not self._send_blocked
    )


def sync_full_width_composer_voice_presentation(
    self: ConsoleComposerBar, active: bool
) -> None:
    """Make room for the persistent executor-wait copy without data loss."""
    from . import console_composer_bar as _owner

    self._voice_full_width_preparing = active
    try:
        controls = (
            self.query_one("#console-composer-collapse", _owner.Button),
            self.query_one("#console-composer-menu", _owner.Button),
            self.query_one("#console-command-visible-text", _owner.Static),
        )
        attachment_indicator = self.query_one(
            "#console-attachment-indicator", _owner.Static
        )
        clear_attachment = self.query_one("#console-clear-attachment", _owner.Button)
        actions = self.query_one("#console-composer-actions", _owner.Horizontal)
    except _owner.NoMatches:
        return

    for control in controls:
        control.styles.display = "none" if active else "block"
    attachment_visible = not active and self._pending_attachment_label is not None
    attachment_indicator.styles.display = "block" if attachment_visible else "none"
    clear_attachment.styles.display = "block" if attachment_visible else "none"
    self._set_actions_row_width(
        actions,
        self._actions_row_width(attachment_visible=attachment_visible),
    )
    self._sync_collapsed_presentation()
    self._sync_send_disabled_reason(
        self._send_disabled_reason,
        muted=not self._send_blocked,
    )


# Capture only the bodies reached by the cold idle projection, at definition.
# Width budgeting belongs to non-idle status and does not authorize that skip.
_IDLE_COMPOSER_PRESENTATION_CALLBACKS = tuple(
    (
        function.__name__,
        function,
        function.__code__,
        function.__defaults__,
        function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()),
        function.__globals__,
        function.__closure__,
    )
    for function in (
        set_composer_voice_status,
        sync_full_width_composer_voice_presentation,
    )
)
