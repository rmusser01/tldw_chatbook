"""Console-native composer action row.

Undo/redo (TASK-1281): history is recorded as flat (draft text, cursor
index) snapshots, never a copy of the live `_DraftSegment` objects. This
keeps the history model simple, and it means undo/redo restores plain
text -- a snapshot never carries a segment's `label` or its `expanded`/
`confirm` display state, so a restored segment is always either an
ordinary literal or a generic collapsed paste token (`_apply_history_
snapshot` re-collapses any restored text over `UNDO_RECOLLAPSE_CHAR_
THRESHOLD`, review NEW-2/W-1/W-2 -- a large, performance-driven threshold
deliberately distinct from the small, cosmetic `paste_collapse_threshold`
a real paste uses), never the exact original presentation (a labeled
file/attachment segment, or one the user had manually unfurled, comes
back as a plain "Pasted Text: N Characters" token if it's still over the
recollapse threshold, or as ordinary literal text otherwise). What
undo/redo does NOT do anymore is repaint a large restored segment as one
giant literal: that used to run the composer's O(n^2) wrap/render path
against the full text on every undo/redo (measured up to 283s frozen for
a 2.4 MB snapshot), which is why the re-collapse exists -- gated on a
threshold sized from the measured render cost, not on the paste-cosmetics
threshold or preference, so it can never itself turn an ordinary
human-typed draft into an opaque token (review W-1) or be disabled by a
user's paste-collapse preference (review W-2).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Literal

from rich.cells import cell_len
from rich.markup import escape
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.content import Content
from textual.css.query import NoMatches
from textual.events import Click, DescendantBlur, DescendantFocus, MouseUp
from textual.geometry import Region
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from ...Chat.console_voice_input import (
    STATE_FINISHING,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_PREPARING,
)
from ...config import (
    DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    coerce_bool_setting,
    coerce_int_setting,
)


_CollapseState = Literal["literal", "collapsed", "confirm", "expanded"]
_DictationState = Literal["idle", "starting", "recording", "transcribing"]
_DraftStyleRange = tuple[int, int, str]

#: Chunk boundary regex mirroring `textwrap.TextWrapper.wordsep_simple_re`
#: (the pattern used whenever `break_on_hyphens=False`, which every wrap call
#: in this module passes). `_cell_wrap_line` needs the identical chunking so
#: its greedy fill only differs from `textwrap.wrap` in how it *measures* a
#: chunk (terminal cells instead of characters), not in where it is willing
#: to break.
_DRAFT_WORD_SPLIT_RE = re.compile(r"([\t\n\x0b\x0c\r ]+)")


@dataclass
class _DraftSegment:
    """Private composer segment with canonical payload and display state."""

    text: str
    collapse_state: _CollapseState = "literal"
    label: str | None = None


@dataclass
class ConsoleDraftStash:
    """A draft captured synchronously at the send keypress (TASK-340).

    Holds the composer's real segment objects so paste provenance and
    collapse state survive a restore, plus the canonical text the send
    path uses as its payload.
    """

    segments: list[_DraftSegment]
    text: str
    has_paste: bool


@dataclass(frozen=True)
class _DraftHistorySnapshot:
    """Undo/redo entry (TASK-1281): the canonical draft text plus caret offset.

    Deliberately flat text+cursor rather than a copy of `_segments` -- the
    architecture this task specified trades away exact display-state
    fidelity across an undo/redo (`_apply_history_snapshot` reconstructs a
    single segment from `text` alone, re-collapsing it into a generic
    paste token when it's over `UNDO_RECOLLAPSE_CHAR_THRESHOLD` -- review
    NEW-2/W-1/W-2, a performance-sized threshold, deliberately NOT the
    cosmetic `paste_collapse_threshold` -- but never recovering the
    ORIGINAL segment's `label` or `expanded`/`confirm` state) for a much
    simpler history model. `restore_stashed_draft`/`ConsoleDraftStash`
    already own the "preserve real segment objects" contract for the send
    flow; this is a separate, narrower one.
    """

    text: str
    cursor_index: int


#: Public alias for the (undo stack, redo stack) pair `export_undo_history`
#: returns and `restore_undo_history` accepts (TASK-1281 N2) -- lets a
#: caller like `ChatScreen` type its own per-session history map without
#: reaching for the private `_DraftHistorySnapshot` name or falling back to
#: `Any`.
ConsoleComposerUndoHistory = tuple[
    list[_DraftHistorySnapshot], list[_DraftHistorySnapshot]
]


@dataclass(frozen=True)
class _DraftSegmentDisplayRange:
    """Visible character range occupied by a segment display token."""

    segment: _DraftSegment
    start: int
    end: int


@dataclass(frozen=True)
class _DraftLineSlice:
    """Wrapped display row with source offsets and synthetic prefix metadata."""

    text: str
    start: int
    end: int
    synthetic_prefix_columns: int = 0


#: Fixed cell width of the composer action row at rest: ☰(4) + Send(8) +
#: Stop(8) + Mic(8). Stop is display-toggled rather than removed, so it is
#: budgeted even while hidden. Attach(10) and Save(8) used to sit here too --
#: they moved into the ☰ menu because this row is width-bounded and every
#: always-present button is space the draft never gets back.
BASE_ACTIONS_WIDTH = 28

#: Width while an attachment is staged, adding the ✕ clear control (4).
ATTACHMENT_ACTIONS_WIDTH = BASE_ACTIONS_WIDTH + 4


class ConsoleComposerBar(Horizontal):
    """Expose Console-owned composer actions while reusing active chat sessions."""

    DEFAULT_STATUS = "No active Console session."
    DRAFT_PLACEHOLDER = "Ask, command, or paste task..."
    PASTE_COLLAPSE_THRESHOLD = DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD
    PASTE_COLLAPSE_ENABLED = True
    MIN_DRAFT_ROWS = 1
    MAX_DRAFT_ROWS = 4
    COMPOSER_CHROME_ROWS = 4
    VOICE_CHIP_MIN_WIDTH = 24
    VOICE_CHIP_MAX_WIDTH = 42
    #: Shown in the chip both for the terminal "stop and transcribe" phase
    #: (`sync_dictation_state`'s "transcribing" branch) and for a per-segment
    #: transcription in flight while still `recording`
    #: (`set_voice_segment_transcribing`) -- same word for "the model is
    #: working on your audio right now" in both places, so there is only one
    #: phrase to recognize rather than two.
    VOICE_CHIP_TRANSCRIBING_LABEL = "◌ Transcribing…"
    FALLBACK_DRAFT_WIDTH = 80
    PASTE_TOKEN_STYLE = "bold cyan"
    PASTE_CONFIRM_STYLE = "bold black on yellow"
    CURSOR_GLYPH = "▌"  # LEFT HALF BLOCK, terminal-style caret
    CURSOR_BLINK_INTERVAL = 0.53
    #: TASK-1281: max entries kept per undo/redo stack; the oldest entry is
    #: dropped once a push would exceed this.
    UNDO_HISTORY_DEPTH_CAP = 100
    #: TASK-1281 review F6 (comment corrected per review NEW-3): max total
    #: characters retained PER STACK -- `_evict_to_char_budget` is applied
    #: to the undo stack and the redo stack independently, so the real
    #: combined ceiling across both is up to ~2x this constant (plus the
    #: never-evict-the-last-entry allowance on each), not this constant
    #: itself. Evicts the oldest entries of a stack first once a push would
    #: put that stack over budget. Entry count alone doesn't bound memory:
    #: every snapshot holds a FULL copy of the draft text, so a single
    #: large inlined attachment (`insert_file_segment`, up to
    #: `MAX_ATTACHMENT_BYTES`) multiplies across every entry recorded after
    #: it. Measured during review: one 1 MB `insert_file_segment` followed
    #: by 20 ordinary pastes retained >20,000,000 characters across just
    #: 21 entries -- nowhere near the 100-entry depth cap.
    UNDO_HISTORY_CHAR_BUDGET = 2_000_000
    #: TASK-1281 review W-1 (HIGH): the perf-guard re-collapse in
    #: `_apply_history_snapshot` must NOT reuse `paste_collapse_threshold`
    #: -- that constant is a cosmetic PASTE-display preference (shipped
    #: default 50 characters) and, applied to undo/redo, converted ORDINARY
    #: TYPED draft text into an opaque "Pasted Text: N Characters" token on
    #: every restore over 50 characters, including the AC's own flagship
    #: Ctrl+U -> Ctrl+Z recovery path (one Backspace then destroyed the
    #: whole recovered draft in a single step, since a collapsed token
    #: deletes as one unit). This constant is chosen from the reviewer's
    #: own measured `_refresh_visible_draft` repaint cost -- 0.01s @ 10K
    #: chars, 0.05s @ 25K, 0.19s @ 50K -- and keeps an undo/redo repaint
    #: comfortably under ~50ms while leaving every ordinary human-typed
    #: draft as plain literal text. Checked UNCONDITIONALLY of
    #: `collapse_large_pastes_enabled` (review W-2): that preference
    #: governs collapse-ON-PASTE cosmetics, not whether the UI thread
    #: freezes repainting a large restored draft -- a performance guard
    #: must not hang off a display preference a user can turn off.
    UNDO_RECOLLAPSE_CHAR_THRESHOLD = 20_000
    #: Shared with the mic button's initial `compose()` tooltip and
    #: `sync_dictation_state`'s idle tooltip, and used as the fallback in
    #: `set_dictation_availability` -- an `Availability(ok=False)` with no
    #: `remedy` text must not blank the tooltip entirely.
    #: Deliberately names no provider, model or language: dictation now streams
    #: through whichever speech-to-text provider `console_voice_input.resolve()`
    #: picks, in `transcription.default_language`. Any static claim about
    #: "English" or "Parakeet v2" here is false on most machines.
    DICTATION_IDLE_TOOLTIP = (
        "Dictate into the draft with the configured speech-to-text provider."
    )

    def __init__(
        self,
        *,
        collapsed: bool = False,
        collapse_large_pastes: bool = True,
        paste_collapse_threshold: int = DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._collapsed = bool(collapsed)
        self.can_focus = not self._collapsed
        self.styles.height = 5
        self.styles.min_height = 5
        self.styles.max_height = self.MAX_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS
        self.collapse_large_pastes = coerce_bool_setting(collapse_large_pastes, True)
        self.paste_collapse_threshold = coerce_int_setting(
            paste_collapse_threshold,
            DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
            minimum=MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
            maximum=MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
        )
        self._segments: list[_DraftSegment] = []
        self._segments_initialized = False
        # Caret position as an offset into the canonical draft text (the
        # concatenation of segment payloads), clamped to [0, len(draft)] on
        # every mutation. Collapsed/confirm paste tokens are single units for
        # caret movement and deletion.
        self._cursor_index = 0
        # TASK-339: bumped by every user-edit entry point (typing/deletes);
        # programmatic load/clear/restore leave it untouched so callers can
        # detect "the user typed since X".
        self._user_edit_serial = 0
        # TASK-1281: undo/redo history. `_undo_stack`/`_redo_stack` hold
        # `_DraftHistorySnapshot`s; `_coalescing_active` is True only while
        # the top of `_undo_stack` is still open to absorbing more
        # consecutive single-character printable inserts (see
        # `_record_undo_snapshot`). Session-scoped export/import lives in
        # `export_undo_history`/`restore_undo_history`.
        self._undo_stack: list[_DraftHistorySnapshot] = []
        self._redo_stack: list[_DraftHistorySnapshot] = []
        self._coalescing_active = False
        self._run_active = False
        self._send_blocked = False
        self._setup_blocked_reason = ""
        self._can_save_chatbook = False
        self._ephemeral = False
        self._dictation_state: _DictationState = "idle"
        #: Whether the last availability probe found both a capture backend
        #: and a transcription provider installed. Only consulted while
        #: `_dictation_state` is "idle" -- once a capture is underway, the
        #: probe already ran (inside `ConsoleVoiceInputController.start()`)
        #: and this flag stays out of the way of the busy-state tooltips.
        self._dictation_available = True
        #: Reason + remedy shown in the mic tooltip while unavailable. Empty
        #: whenever `_dictation_available` is True.
        self._dictation_unavailable_tooltip = ""
        #: Chip-only display state for the active capture. Reset on every
        #: fresh entry into "recording" (`sync_dictation_state`) and updated
        #: live by `set_voice_partial()` / `tick_voice_elapsed()`; preserved
        #: across a redundant `sync_dictation_state("recording")` call (the
        #: 0.2s Console UI-sync tick calls this unconditionally) so that tick
        #: cannot stomp the chip back to "0:00" mid-capture.
        self._voice_partial: str = ""
        self._voice_elapsed_seconds: int = 0
        #: Latest model-preparation status for the chip, held for the same
        #: reason `_voice_partial` is: `sync_dictation_state` is called
        #: unconditionally by every control-bar refresh (changing a provider,
        #: collapsing a rail), and without holding this a single keystroke
        #: elsewhere would rewrite a multi-minute "Preparing speech model…" back
        #: to "Preparing microphone…" -- which by then is also false, because
        #: nothing is preparing a microphone. Cleared only on a genuine
        #: transition into "starting", and on recording/idle.
        self._voice_preparing_message: str = ""
        #: True while a per-segment transcription is in flight (the silence
        #: gate closed a segment and the recognizer is working on it, a call
        #: that can take seconds -- see `VoiceSegmentTranscribing`). Reset
        #: the same way `_voice_partial` is: on every fresh entry into
        #: "recording", cleared by `set_voice_partial()` (the next final or
        #: command lands), and on any OTHER lifecycle state change; preserved
        #: across a redundant `sync_dictation_state("recording")` resync so
        #: an unrelated UI refresh cannot blank an indicator that is still
        #: legitimately showing.
        self._voice_segment_transcribing: bool = False
        self._pending_attachment_label: str | None = None
        self._suppress_next_draft_click = False
        self._draft_selection_all = False
        self._cursor_visible = True
        self._cursor_blink_timer: Any | None = None

    @property
    def collapse_large_pastes_enabled(self) -> bool:
        """Return whether pasted chunks over the threshold should display compactly."""
        return self.collapse_large_pastes

    @property
    def collapsed(self) -> bool:
        """Return whether the compact restore-only presentation is active."""
        return self._collapsed

    @staticmethod
    def _bounded_button(label: str, *, width: int, **kwargs: Any) -> Button:
        kwargs.setdefault("compact", True)
        button = Button(label, **kwargs)
        button.styles.width = width
        button.styles.min_width = width
        button.styles.height = 1
        button.styles.min_height = 1
        return button

    @staticmethod
    def _set_actions_row_width(actions: Horizontal, width: int) -> None:
        """Pin the action row to an exact cell width.

        The row is deliberately fixed rather than ``auto``: every cell it
        claims is a cell the draft does not get, so the budget is stated
        once here and in the two module constants rather than being an
        emergent property of whichever buttons happen to be visible.

        Args:
            actions: The ``#console-composer-actions`` row.
            width: Exact width in cells.
        """
        actions.styles.width = width
        actions.styles.min_width = width
        actions.styles.max_width = width

    def draft_text(self) -> str:
        """Return the canonical native Console draft payload.

        Returns:
            The full message text that will be sent, including expanded content
            from any display-collapsed paste segments.
        """
        if self._segments_initialized:
            return self._canonical_draft_text()
        try:
            return self.query_one("#console-command-input", Input).value
        except NoMatches:
            return ""

    def _canonical_draft_text(self) -> str:
        """Return the full payload represented by composer segments."""
        return "".join(segment.text for segment in self._segments)

    def _has_any_draft_content(self) -> bool:
        """Return whether the canonical draft contains at least one character."""
        if self._segments_initialized:
            return any(segment.text for segment in self._segments)
        try:
            return bool(self.query_one("#console-command-input", Input).value)
        except NoMatches:
            return False

    @property
    def cursor_index(self) -> int:
        """Return the caret offset into the canonical draft text.

        Returns:
            The zero-based character index of the caret in the canonical
            (non-display) draft text.
        """
        return self._cursor_index

    def _clamp_cursor(self) -> None:
        """Keep the caret inside the canonical draft bounds."""
        self._cursor_index = max(
            0,
            min(self._cursor_index, len(self._canonical_draft_text())),
        )

    def _locate_canonical(self, index: int) -> tuple[int, int]:
        """Map a canonical draft offset to (segment index, intra-segment offset).

        An offset exactly on a segment boundary maps to the END of the left
        segment, so a caret right after a paste token resolves to that token.
        """
        if not self._segments:
            return (0, 0)
        remaining = index
        for segment_index, segment in enumerate(self._segments):
            segment_length = len(segment.text)
            if remaining <= segment_length:
                return (segment_index, remaining)
            remaining -= segment_length
        last_index = len(self._segments) - 1
        return (last_index, len(self._segments[last_index].text))

    def _cursor_display_index(self) -> int:
        """Map the canonical caret offset to an unwrapped display-string offset.

        Collapsed/confirm paste tokens render as a short display token, so a
        caret inside one snaps to the token's nearest display edge.
        """
        remaining = self._cursor_index
        display_offset = 0
        for segment in self._segments:
            segment_length = len(segment.text)
            display_length = len(self._segment_display_text(segment))
            if remaining <= segment_length:
                if segment.collapse_state in {"collapsed", "confirm"}:
                    return display_offset + (display_length if remaining else 0)
                return display_offset + remaining
            remaining -= segment_length
            display_offset += display_length
        return display_offset

    def _canonical_index_at_display(self, display_index: int) -> int:
        """Map an unwrapped display-string offset to a canonical draft offset.

        Offsets landing on a collapsed/confirm paste token snap to the
        token's nearest canonical edge (the caret never sits inside a token).
        """
        display_offset = 0
        canonical_offset = 0
        for segment in self._segments:
            display_length = len(self._segment_display_text(segment))
            canonical_length = len(segment.text)
            if display_index < display_offset + display_length:
                if segment.collapse_state in {"collapsed", "confirm"}:
                    within = display_index - display_offset
                    if within * 2 < display_length:
                        return canonical_offset
                    return canonical_offset + canonical_length
                return canonical_offset + (display_index - display_offset)
            display_offset += display_length
            canonical_offset += canonical_length
        return canonical_offset

    def _display_draft_text(self) -> str:
        """Return the display-only draft text represented by composer segments."""
        if not self._segments_initialized:
            try:
                return self.query_one("#console-command-input", Input).value
            except NoMatches:
                return ""
        return "".join(
            self._segment_display_text(segment) for segment in self._segments
        )

    @staticmethod
    def _segment_display_text(segment: _DraftSegment) -> str:
        """Return display text for a single draft segment."""
        if segment.collapse_state == "collapsed":
            if segment.label:
                return segment.label
            return f"Pasted Text: {len(segment.text)} Characters"
        if segment.collapse_state == "confirm":
            return "Unfurl?"
        return segment.text

    def _segment_display_ranges(self) -> list[_DraftSegmentDisplayRange]:
        """Return segment ranges in the unwrapped visible draft string."""
        ranges: list[_DraftSegmentDisplayRange] = []
        offset = 0
        for segment in self._segments:
            display_text = self._segment_display_text(segment)
            next_offset = offset + len(display_text)
            ranges.append(_DraftSegmentDisplayRange(segment, offset, next_offset))
            offset = next_offset
        return ranges

    def _display_draft_style_ranges(self) -> list[_DraftStyleRange]:
        """Return Rich style spans for stateful display-only paste tokens."""
        if not self._segments_initialized:
            return []

        style_ranges: list[_DraftStyleRange] = []
        for display_range in self._segment_display_ranges():
            segment = display_range.segment
            if segment.collapse_state == "collapsed":
                style_ranges.append(
                    (
                        display_range.start,
                        display_range.end,
                        self.PASTE_TOKEN_STYLE,
                    )
                )
            elif segment.collapse_state == "confirm":
                style_ranges.append(
                    (
                        display_range.start,
                        display_range.end,
                        self.PASTE_CONFIRM_STYLE,
                    )
                )
        if self._draft_selection_all:
            display_text = self._display_draft_text()
            if display_text:
                style_ranges.append((0, len(display_text), "reverse"))
        return style_ranges

    def _sync_hidden_input(self) -> None:
        """Keep the hidden compatibility input aligned with canonical payload."""
        try:
            self.query_one(
                "#console-command-input", Input
            ).value = self._canonical_draft_text()
        except NoMatches:
            return

    def _sync_interaction_classes(self) -> None:
        """Mirror focus-within and draft presence onto stable CSS state classes."""
        self.set_class(self.has_focus_within, "console-composer-focused")
        self.set_class(
            self._has_any_draft_content(),
            "console-composer-has-draft",
        )

    def _sync_current_action_state(self) -> None:
        """Refresh action buttons from the current draft and cached run/save state."""
        self.sync_action_state(
            has_draft=bool(self.draft_text().strip()),
            run_active=self._run_active,
            can_save_chatbook=self._can_save_chatbook,
            send_blocked=self._send_blocked,
            setup_blocked_reason=self._setup_blocked_reason,
            ephemeral=self._ephemeral,
        )

    def sync_action_state(
        self,
        *,
        has_draft: bool,
        run_active: bool,
        can_save_chatbook: bool,
        send_blocked: bool = False,
        setup_blocked_reason: str = "",
        ephemeral: bool = False,
    ) -> None:
        """Refresh composer action priority and disabled state.

        Args:
            has_draft: Whether the canonical draft has non-whitespace content.
            run_active: Whether a Console run is currently stoppable.
            can_save_chatbook: Whether a Chatbook artifact is available to save.
            send_blocked: Whether the current run state blocks new sends.
            setup_blocked_reason: Provider/model setup copy when setup blocks Send.
            ephemeral: Whether the active session is temporary, which blocks
                Save Chatbook (a second door onto the same write the
                workbench's Save Chatbook action already gates).
        """
        has_draft = bool(has_draft)
        run_active = bool(run_active)
        can_save_chatbook = bool(can_save_chatbook)
        send_blocked = bool(send_blocked)
        setup_blocked_reason = setup_blocked_reason.strip()
        ephemeral = bool(ephemeral)
        setup_reason_changed = self._setup_blocked_reason != setup_blocked_reason
        self._run_active = run_active
        self._send_blocked = send_blocked
        self._setup_blocked_reason = setup_blocked_reason
        self._can_save_chatbook = can_save_chatbook
        self._ephemeral = ephemeral
        self._sync_collapsed_presentation()

        try:
            send_button = self.query_one("#console-send-message", Button)
            stop_button = self.query_one("#console-stop-generation", Button)
        except NoMatches:
            return

        send_ready = has_draft and not send_blocked

        send_button.disabled = False
        send_button.variant = "primary" if send_ready else "default"
        if send_blocked and setup_blocked_reason:
            send_button.tooltip = setup_blocked_reason
        elif send_blocked:
            send_button.tooltip = (
                "Wait for the active Console run to finish before sending."
            )
        elif has_draft:
            send_button.tooltip = "Send the active Console session draft."
        else:
            send_button.tooltip = None
        send_button.set_class(send_ready, "console-action-primary")
        send_button.set_class(not send_ready, "console-action-subdued")
        send_button.set_class(not send_ready, "console-action-disabled")
        send_button.set_class(send_ready, "console-send-ready")
        send_button.set_class(not has_draft, "console-send-inactive")
        send_button.set_class(send_blocked, "console-send-blocked")
        self.set_class(
            send_blocked and bool(setup_blocked_reason),
            "console-composer-setup-blocked",
        )

        stop_button.disabled = False
        stop_button.variant = "warning" if run_active else "default"
        # Fleet-UX expert review F7 (task-1234): this LIVE sync overrides
        # the button's construction-time tooltip on every action-state
        # refresh, so the compose-time copy alone (see `compose()` above)
        # was never actually what a user hovering an active Stop button
        # saw -- fixed here too, matching the collapsed Stop button (which
        # has no such override).
        stop_button.tooltip = (
            "Stop this tab's run."
            if run_active
            else "No active run to stop in this tab."
        )
        stop_button.set_class(run_active, "console-stop-active")
        stop_button.set_class(not run_active, "console-stop-idle")
        stop_button.set_class(not run_active, "console-action-disabled")
        stop_button.styles.display = "block" if run_active else "none"

        # Attach and Save Chatbook no longer live in this row -- their
        # enabled/disabled presentation (including the temporary-chat block
        # on Save) is now decided in `build_composer_menu_entries`, which
        # reads the same `can_save_chatbook` and `ephemeral` inputs stashed
        # on this widget above and consults the same `blocked_reason`
        # registry. One contract, one place, rendered as a menu row instead
        # of a button.

        if setup_reason_changed and not self.draft_text().strip():
            self._refresh_visible_draft()

    def sync_dictation_state(self, state: _DictationState) -> None:
        """Refresh the microphone action for the current one-shot lifecycle.

        Args:
            state: Current one-shot dictation lifecycle state.
        """
        entering_recording = state == "recording" and self._dictation_state != "recording"
        entering_starting = state == "starting" and self._dictation_state != "starting"
        state_changed = state != self._dictation_state
        self._dictation_state = state
        if state_changed:
            # Any genuine lifecycle transition -- including "recording" ->
            # "transcribing" via the mic button, which never routes through
            # `set_voice_partial()` -- ends a live per-segment transcribing
            # indication. A redundant resync that leaves the state unchanged
            # (the 0.2s Console UI-sync tick) must NOT reset it: see
            # `_voice_segment_transcribing`'s docstring.
            self._voice_segment_transcribing = False
        try:
            button = self.query_one("#console-dictation", Button)
        except NoMatches:
            return
        labels = {
            "idle": "Mic",
            "starting": "Mic…",
            "recording": "Rec ●",
            "transcribing": "STT…",
        }
        tooltips = {
            "idle": self.DICTATION_IDLE_TOOLTIP,
            # A first-run model download is minutes long, so this phase needs a
            # way out. The button stays clickable here (unlike "transcribing",
            # where there is nothing left to cancel) and a press cancels.
            "starting": "Preparing the speech model — press to cancel.",
            "recording": "Stop microphone recording and transcribe.",
            # No provider name here either: the transcribing phase runs on the
            # same resolved provider the idle tooltip declines to name.
            "transcribing": "Transcribing…",
        }
        # Unavailability is cosmetic-only, and only shown at idle: the button
        # stays real-clickable (never Textual `disabled`) so a press can
        # still reach the screen's activation handler, which re-probes and
        # is what actually recovers the button without a remount once, say,
        # the missing extra gets installed mid-run. Real Textual `disabled`
        # would block the Click event from ever being delivered at all --
        # including on a later retry -- which is exactly the dead-end this
        # exists to avoid.
        unavailable = state == "idle" and not self._dictation_available
        button.label = labels[state]
        button.tooltip = (
            self._dictation_unavailable_tooltip if unavailable else tooltips[state]
        )
        # "starting" stays enabled on purpose: it now covers a model load that
        # can run for minutes on a first run, and a disabled button would leave
        # the user with no in-app way out of it.
        button.disabled = state == "transcribing"
        button.variant = "warning" if state == "recording" else "default"
        button.set_class(state == "recording", "console-dictation-recording")
        button.set_class(unavailable, "console-dictation-unavailable")

        # Mirror the lifecycle into the inline voice chip. The chip has its
        # own vocabulary (STATE_* from console_voice_input), so map the
        # button's states explicitly rather than passing the string through.
        if state == "idle":
            self._voice_partial = ""
            self._voice_elapsed_seconds = 0
            self._voice_preparing_message = ""
            self.set_voice_status(STATE_IDLE)
        elif state == "starting":
            if entering_starting:
                self._voice_partial = ""
                self._voice_elapsed_seconds = 0
                self._voice_preparing_message = ""
            # Re-applied, not recomputed: a redundant resync (any control-bar
            # refresh calls this) must not overwrite a live model-preparation
            # message with the generic microphone one.
            self.set_voice_status(
                STATE_PREPARING,
                message=self._voice_preparing_message or "◌ Preparing microphone…",
            )
        elif state == "recording":
            if entering_recording:
                self._voice_partial = ""
                self._voice_elapsed_seconds = 0
                self._voice_preparing_message = ""
            self.set_voice_status(
                STATE_LISTENING,
                partial=self._voice_partial,
                elapsed_seconds=self._voice_elapsed_seconds,
                # Re-applied, not recomputed -- same reasoning as "starting"'s
                # `_voice_preparing_message` above: a redundant resync must
                # not blank a live segment-transcribing indication.
                segment_transcribing=self._voice_segment_transcribing,
            )
        elif state == "transcribing":
            self.set_voice_status(
                STATE_FINISHING, message=self.VOICE_CHIP_TRANSCRIBING_LABEL
            )

    def set_dictation_availability(
        self, *, available: bool, tooltip: str = ""
    ) -> None:
        """Record the last dictation availability probe and refresh the mic button.

        Args:
            available: Whether a probe found both a capture backend and a
                transcription provider installed.
            tooltip: Reason and remedy to show in the mic tooltip while
                unavailable (e.g. naming the missing extra to install).
                Ignored when ``available`` is True. An empty string falls
                back to the ordinary idle tooltip rather than blanking it --
                `Availability(ok=False)` defaults both `reason` and `remedy`
                to `""`, and a blank tooltip would be worse than the generic
                one it replaced.
        """
        self._dictation_available = bool(available)
        self._dictation_unavailable_tooltip = (
            "" if available else (tooltip or self.DICTATION_IDLE_TOOLTIP)
        )
        self.sync_dictation_state(self._dictation_state)

    def set_voice_preparing_message(self, text: str) -> None:
        """Show model-preparation progress in the chip, and keep showing it.

        Held in `_voice_preparing_message` rather than written straight to the
        chip: `sync_dictation_state("starting")` fires from every control-bar
        refresh, and a one-shot write would be erased by the next unrelated UI
        change -- during exactly the multi-minute window this message exists
        for, and replaced by a "Preparing microphone…" that is not even true.

        Args:
            text: Chip-sized status text (the "◌ " prefix included). Ignored
                outside the `starting` lifecycle state, so a notice that drains
                late cannot repaint a chip that has moved on.
        """
        if self._dictation_state != "starting":
            return
        self._voice_preparing_message = text
        self.set_voice_status(STATE_PREPARING, message=text)

    def set_voice_partial(self, text: str) -> None:
        """Render live recognizer text into the chip while recording.

        Args:
            text: In-flight partial transcript from the recognizer. Ignored
                (a no-op) outside the `recording` lifecycle state, so a
                partial that drains after the capture already ended cannot
                resurrect the chip.
        """
        if self._dictation_state != "recording":
            return
        self._voice_partial = text
        # A partial lands exactly when a segment's transcription has
        # finished (see `VoiceSegmentTranscribing`'s docstring: under the
        # segment-at-silence architecture there is no partial *during* the
        # transcription, only once it completes) -- and this same method
        # renders the ack for `VoiceCommand`/clears the chip for `VoiceFinal`
        # too, both of which equally supersede an in-flight indication.
        self._voice_segment_transcribing = False
        self.set_voice_status(
            STATE_LISTENING,
            partial=self._voice_partial,
            elapsed_seconds=self._voice_elapsed_seconds,
        )

    def set_voice_segment_transcribing(self, transcribing: bool) -> None:
        """Show or hide a per-segment transcribing indicator in the chip.

        Fills the gap `VoiceSegmentTranscribing` exists for: the silence gate
        can close a segment and then say nothing at all for seconds while it
        transcribes (no live partial text under the segment-at-silence
        architecture), which otherwise looks identical to a dead capture.

        Args:
            transcribing: True right when that gap starts. Reverted to False
                by `set_voice_partial()` (the next final or command landing)
                or by any `sync_dictation_state()` lifecycle transition --
                never called with False directly by a caller. Ignored (a
                no-op) outside the `recording` lifecycle state, the same
                guard `set_voice_partial` uses.
        """
        if self._dictation_state != "recording":
            return
        self._voice_segment_transcribing = transcribing
        self.set_voice_status(
            STATE_LISTENING,
            partial=self._voice_partial,
            elapsed_seconds=self._voice_elapsed_seconds,
            segment_transcribing=self._voice_segment_transcribing,
        )

    def tick_voice_elapsed(self) -> None:
        """Advance the chip's elapsed-time counter by one second.

        A no-op outside `recording` so a stray tick that fires just after the
        capture ends (the owning timer is stopped on every exit path, but a
        tick already queued for this frame can still land) cannot repaint a
        chip that has already collapsed.
        """
        if self._dictation_state != "recording":
            return
        self._voice_elapsed_seconds += 1
        self.set_voice_status(
            STATE_LISTENING,
            partial=self._voice_partial,
            elapsed_seconds=self._voice_elapsed_seconds,
            # Without this, the 1s elapsed-counter tick would blank a live
            # segment-transcribing indication every second (`set_voice_status`
            # defaults the parameter to False) -- the indicator can easily
            # outlast one tick, since the transcription behind it takes
            # seconds.
            segment_transcribing=self._voice_segment_transcribing,
        )

    @staticmethod
    def _extend_fitting_cells(prefix: str, text: str, width: int) -> str:
        """Return the longest prefix of ``text`` that keeps ``prefix + <it>`` within ``width``.

        Measures the actual **joined** candidate (``prefix + text[:k]``)
        directly with `cell_len` at every step -- never a pre-computed
        numeric budget (``width - cell_len(prefix)``) checked against
        `cell_len` of the candidate piece *alone*. Cell width is not
        additive across a join in every case, and not only in the way
        `_cell_wrap_line`'s own docstring already covers (a boundary
        crossing a narrow_to_wide upgrade): `cell_len` has a still sharper
        edge where a **trailing** ZWJ silently absorbs the character that
        would have followed it in a longer string (confirmed:
        ``cell_len("#ZXY b9\\u200d ")`` -- 9 characters, ending right after
        a ZWJ then a space -- is 7, as if the trailing space were never
        there; append even one more character and the space *and* a
        narrow-to-wide upgrade both reappear in the total). A numeric budget
        derived from `cell_len(prefix)` in isolation cannot see that the
        join itself changes what the tail of `prefix` measures as; measuring
        the literal joined string every time sidesteps needing to know why.

        Also why this doesn't delegate to `rich.cells.chop_cells`/
        `split_graphemes`: those compute cell width via Unicode grapheme
        segmentation, which itself disagrees with `cell_len`'s own
        character-scan algorithm on ZWJ-bearing text (confirmed:
        ``cell_len("\\u200d\\u200dbcb3Z33")`` is 7, but `split_graphemes`
        -- what `chop_cells` trusts -- reports 6 for the identical string).
        Recomputing directly with `cell_len` keeps this self-consistent with
        every other width check in the file, all of which measure that way.

        Returns ``""`` when even a single character of ``text`` doesn't fit
        -- an honest "nothing fits" answer, not a forced minimum.
        `_cell_wrap_line` is responsible for its own forward-progress
        guarantee on an empty result; folding a forced minimum in here
        caused it to fire on ordinary partial-budget rows too (an
        already-full row plus a next chunk that simply has no room left is
        not the same situation as a wholly empty row that cannot fit
        *anything*, and conflating them broke the single-width parity
        guarantee -- caught by 100k-trial differential fuzzing against
        `textwrap.wrap` during development).

        Args:
            prefix: Text already committed to the row being built.
            text: Candidate string to take a fitting extension from.
            width: Total cell width the joined row must fit within.

        Returns:
            The longest fitting extension; ``""`` if nothing fits (including
            an empty ``text``).
        """
        if not text:
            return text
        end = 0
        while end < len(text) and cell_len(prefix + text[: end + 1]) <= width:
            end += 1
        return text[:end]

    @classmethod
    def _cell_wrap_line(cls, line: str, width: int) -> list[str]:
        """Greedy word-wrap ``line`` by terminal cell width, not character count.

        Mirrors ``textwrap.wrap(line, width=width, break_long_words=True,
        break_on_hyphens=False, drop_whitespace=False, replace_whitespace=False)``
        chunk-for-chunk -- same tab expansion, same whitespace-run chunking,
        same greedy fill and long-word hard-break -- except every length
        check measures ``rich.cells.cell_len`` (terminal columns) instead of
        ``len()`` (Python characters). For single-width-only text the two
        measures are identical, so this produces byte-identical output to
        the ``textwrap`` call it replaces (confirmed with 100k-trial
        differential fuzzing against the exact `textwrap.wrap` call above);
        the difference only surfaces on double-width text (CJK, emoji),
        where a character-counted wrap can under-count how many terminal
        columns a row actually occupies and let it silently overflow the
        wrap width at paint time.

        Both the greedy fill and the long-word hard-break measure the
        *actual joined candidate row* directly with `cell_len` -- never a
        running sum or a pre-computed numeric budget checked against a
        piece's own isolated `cell_len` (see `_extend_fitting_cells`'s
        docstring for why cell width is not reliably additive across a
        join). The hard-break takes one fitting bite per call via
        `_extend_fitting_cells`, matching `TextWrapper._handle_long_word`'s
        one-bite-per-line contract, so a chunk that needs several lines to
        exhaust naturally gets the rest on subsequent passes of the outer
        loop below.

        Args:
            line: A single logical line (no newlines) to wrap.
            width: Wrap width in terminal cells.

        Returns:
            Wrapped rows for ``line``; always at least one entry (matching
            ``textwrap.wrap(...) or [""]`` at every call site).
        """
        width = max(1, width)
        # `textwrap.wrap` always expands tabs before splitting into chunks
        # (`expand_tabs` defaults to True and is never overridden by any
        # caller in this module) regardless of `replace_whitespace`, which
        # only controls whether *other* whitespace becomes plain spaces.
        chunks = [
            chunk
            for chunk in _DRAFT_WORD_SPLIT_RE.split(line.expandtabs(8))
            if chunk
        ]
        chunks.reverse()
        lines: list[str] = []
        while chunks:
            current_text = ""
            while chunks:
                candidate = current_text + chunks[-1]
                if cell_len(candidate) > width:
                    break
                current_text = candidate
                chunks.pop()

            if chunks and cell_len(chunks[-1]) > width:
                chunk = chunks[-1]
                piece = cls._extend_fitting_cells(current_text, chunk, width)
                if not piece and not current_text:
                    # A wholly fresh row and even the FULL `width` budget
                    # can't fit this chunk's first character (only possible
                    # at width < 2 with double-width leading content --
                    # unreachable in production, where every call site
                    # floors width at 8, but no caller here relies on that,
                    # so this stays correct if one ever passes a smaller
                    # width). Force exactly one character so the outer loop
                    # is guaranteed to make progress; the CSS
                    # `text_overflow: clip` guard crops the resulting
                    # overflow at paint time.
                    piece = chunk[:1]
                if piece:
                    current_text += piece
                    chunks[-1] = chunk[len(piece) :]
                    if not chunks[-1]:
                        chunks.pop()
                # else: this row already holds prior content and has no
                # room left for even one more character of the next
                # (individually too-wide) chunk -- leave it untouched for a
                # fresh, full-budget row on the next pass of the outer loop,
                # exactly mirroring `TextWrapper._handle_long_word`'s own
                # `space_left == 0` behavior. Forcing a character onto an
                # already-full row here would silently make that row wider
                # than `width` for perfectly ordinary text, not just the
                # width-1 double-width case above (caught by fuzzing during
                # development).

            # `current_text` is always non-empty here: either the greedy
            # fill above consumed at least one chunk, or the hard-break
            # branch's forced-progress fallback fired (guaranteed whenever
            # `current_text` started this iteration empty and nothing else
            # fit, per the note above). The only way to reach here with an
            # unconsumed too-wide `chunks[-1]` and an empty `current_text`
            # would require that fallback to not fire on an empty row, which
            # cannot happen.
            lines.append(current_text)
        return lines or [""]

    @classmethod
    def _wrap_draft_lines(cls, text: str, width: int) -> list[str]:
        """Return wrapped draft lines for the visible bounded composer."""
        width = max(8, width)
        source_lines = text.splitlines() or [text]
        wrapped_lines: list[str] = []
        for line in source_lines:
            if not line:
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(cls._cell_wrap_line(line, width))
        return wrapped_lines or [""]

    @classmethod
    def _wrap_draft_line_slices(cls, text: str, width: int) -> list[_DraftLineSlice]:
        """Return wrapped draft lines with source offsets for style remapping."""
        width = max(8, width)
        source_lines = text.splitlines(keepends=True) or [text]
        wrapped_lines: list[_DraftLineSlice] = []
        source_offset = 0
        for raw_line in source_lines:
            if raw_line.endswith("\r\n"):
                line = raw_line[:-2]
                separator_length = 2
            elif raw_line.endswith(("\n", "\r")):
                line = raw_line[:-1]
                separator_length = 1
            else:
                line = raw_line
                separator_length = 0

            if not line:
                wrapped_lines.append(_DraftLineSlice("", source_offset, source_offset))
                source_offset += separator_length
                continue

            line_offset = 0
            wrapped_segments = cls._cell_wrap_line(line, width)
            for wrapped_segment in wrapped_segments:
                start = source_offset + line_offset
                end = start + len(wrapped_segment)
                wrapped_lines.append(_DraftLineSlice(wrapped_segment, start, end))
                line_offset += len(wrapped_segment)
            source_offset += len(line) + separator_length

        return wrapped_lines or [_DraftLineSlice("", 0, 0)]

    @classmethod
    def _visible_draft_lines(cls, text: str, width: int) -> list[str]:
        """Return the bounded visible draft lines, biased toward the caret end."""
        return [
            line_slice.text
            for line_slice in cls._visible_draft_line_slices(text, width)
        ]

    @staticmethod
    def _row_index_for_source_offset(
        line_slices: list[_DraftLineSlice],
        source_offset: int,
    ) -> int:
        """Return the wrapped row containing a source-text offset.

        For SPLICED offsets only -- callers that pass `line_slices` wrapped
        from a caret-glyph- or placeholder-spliced `render_text`, together
        with the matching spliced offset into it (the two current
        production callers, both via `_visible_draft_line_slices(...,
        cursor_index=...)`: `_draft_renderable`'s glyph splice and
        `_display_index_at`'s space splice). A real character always
        occupies the exact offset being looked up under that contract, so
        the "no row contains this offset" fallback below (unconditionally
        the LAST row) is never actually reachable there.

        That fallback is WRONG for a genuinely unspliced offset -- e.g. an
        offset sitting exactly in the gap between two explicit-newline-
        separated rows, which is common and expected, not an edge case.
        Canonical/unspliced-offset callers (the vertical caret-movement
        path) must use `_row_index_for_canonical_offset` instead, which
        resolves that gap to the row immediately preceding it rather than
        unconditionally the draft's last row.
        """
        for row_index, line_slice in enumerate(line_slices):
            if line_slice.start <= source_offset < line_slice.end:
                return row_index
        return len(line_slices) - 1

    @classmethod
    def _visible_draft_line_slices(
        cls,
        text: str,
        width: int,
        *,
        cursor_index: int | None = None,
    ) -> list[_DraftLineSlice]:
        """Return bounded wrapped draft rows with source-offset mapping.

        When ``cursor_index`` is given (an offset into ``text``, typically the
        reserved caret cell), the visible window scrolls just enough to keep
        the caret row on screen; otherwise it stays biased toward the tail.
        """
        line_slices = cls._wrap_draft_line_slices(text, width)
        if len(line_slices) <= cls.MAX_DRAFT_ROWS:
            return line_slices

        if cursor_index is None:
            first_visible = len(line_slices) - cls.MAX_DRAFT_ROWS
        else:
            caret_row = cls._row_index_for_source_offset(line_slices, cursor_index)
            first_visible = min(
                max(caret_row - (cls.MAX_DRAFT_ROWS - 1), 0),
                len(line_slices) - cls.MAX_DRAFT_ROWS,
            )
        visible_slices = list(
            line_slices[first_visible : first_visible + cls.MAX_DRAFT_ROWS]
        )
        if first_visible == 0:
            # Nothing is scrolled off above this window -- row 0 IS the
            # draft's true first row, not a continuation. Prefixing/trimming
            # it here would delete real leading content (and the caret
            # glyph, when the caret is on this row -- reachable by ordinary
            # Home/click-at-start on any draft over MAX_DRAFT_ROWS) to make
            # room for an ellipsis that has nothing above it to elide.
            return visible_slices
        effective_width = max(8, width)
        first_slice = visible_slices[0]
        first_line_stripped = first_slice.text.lstrip()
        if first_line_stripped:
            prefix = "... "
            lstripped_columns = len(first_slice.text) - len(first_line_stripped)
            # The ellipsis must CONSUME leading content, not extend the row:
            # a whitespace-flush row that already fills `width` would
            # otherwise grow to `width + len(prefix)` once prefixed, and an
            # unbudgeted row that long gets rewrapped at paint time by
            # anything that isn't `no_wrap` -- silently pushing the true
            # last row out of the fixed-height window (the bug this guards
            # against). The candidate REMAINDER is measured directly with
            # `cell_len` on each trimmed slice, not via a running per-
            # character subtraction: a decrement seeded from the whole
            # string's grapheme-aware `cell_len` but applied one character
            # at a time over-subtracts across any grapheme cluster whose
            # per-character sum exceeds the cluster's own width (ZWJ
            # sequences), exiting the trim loop early and leaving the row
            # still over budget -- confirmed via fuzzing during development
            # (`"K‍TCCOtR"` at width 8: cluster-cells=6, per-char-sum=7).
            budget_cells = max(0, effective_width - cell_len(prefix))
            overflow_columns = 0
            while (
                overflow_columns < len(first_line_stripped)
                and cell_len(first_line_stripped[overflow_columns:]) > budget_cells
            ):
                overflow_columns += 1
            visible_text = first_line_stripped[overflow_columns:]
            # Advance the source-offset start past every trimmed character:
            # the whitespace `lstrip()` already dropped, plus the additional
            # leading characters just trimmed to make room for the prefix.
            # Keeps `_row_index_for_source_offset`, style-span remapping, and
            # click-to-position mapping consistent with what is now painted.
            trimmed_columns = lstripped_columns + overflow_columns
            visible_slices[0] = _DraftLineSlice(
                f"{prefix}{visible_text}",
                first_slice.start + trimmed_columns,
                first_slice.end,
                synthetic_prefix_columns=len(prefix),
            )
        else:
            visible_slices[0] = _DraftLineSlice(
                "...",
                first_slice.end,
                first_slice.end,
                synthetic_prefix_columns=3,
            )
        return visible_slices

    @staticmethod
    def _shift_style_ranges_for_caret(
        style_ranges: list[_DraftStyleRange],
        caret_position: int,
    ) -> list[_DraftStyleRange]:
        """Shift style spans right of the spliced caret cell by one column.

        Spans starting exactly at the caret move whole; spans containing the
        caret grow to cover the caret cell; spans ending exactly at the caret
        are untouched.
        """
        shifted: list[_DraftStyleRange] = []
        for style_start, style_end, style in style_ranges:
            shifted.append(
                (
                    style_start + 1 if style_start >= caret_position else style_start,
                    style_end + 1 if style_end > caret_position else style_end,
                    style,
                )
            )
        return shifted

    @classmethod
    def _draft_renderable(
        cls,
        text: str,
        *,
        width: int = FALLBACK_DRAFT_WIDTH,
        style_ranges: list[_DraftStyleRange] | None = None,
        focused: bool = False,
        cursor_visible: bool = True,
        cursor_index: int | None = None,
    ) -> Text:
        if text:
            # While focused, exactly one display cell is always reserved at
            # the caret position inside the wrapped draft -- the caret glyph
            # during the visible blink phase, an ordinary space during the
            # hidden phase -- and it is wrapped in the *same* pass as the
            # draft itself (rather than appended afterward). That keeps the
            # two blink phases layout-identical: whichever character reserves
            # the cell is decided by wrap width alone, never by which literal
            # character it is, so a blink tick can never change how many
            # visual rows the draft occupies (which previously could clip or
            # jitter the composer when the last wrapped line landed exactly
            # at the wrap width). The glyph is left unstyled: the block
            # character is prominent enough on its own, and leaving it
            # unstyled keeps it from being mistaken for a stateful paste
            # token.
            if focused:
                caret_cell = cls.CURSOR_GLYPH if cursor_visible else " "
                caret_position = (
                    len(text)
                    if cursor_index is None
                    else max(0, min(cursor_index, len(text)))
                )
                render_text = (
                    f"{text[:caret_position]}{caret_cell}{text[caret_position:]}"
                )
                if style_ranges:
                    style_ranges = cls._shift_style_ranges_for_caret(
                        style_ranges,
                        caret_position,
                    )
            else:
                caret_position = None
                render_text = text
            line_slices = cls._visible_draft_line_slices(
                render_text,
                width,
                cursor_index=caret_position,
            )
            # `no_wrap`/`overflow="crop"`: defense-in-depth, not the fix for
            # any known bug reachable through this file's own call sites.
            # Each joined row is already budgeted to fit `width` by
            # `_visible_draft_line_slices` -- verified with 500k-trial fuzzing
            # (ASCII, CJK, ZWJ/emoji grapheme clusters) at every width >= 8,
            # the floor both `_wrap_draft_lines` and `_wrap_draft_line_slices`
            # enforce, with zero violations. Below that floor (unreachable in
            # production, but not something `_cell_wrap_line` itself assumes)
            # a genuinely-empty row that can't fit even one character of a
            # double-width chunk is allowed a single bounded overflow rather
            # than hang -- this guard is what actually crops that case. This
            # only otherwise changes what happens if a *future* budgeting bug
            # lets a row overflow again -- cropping the one offending row in
            # place instead of silently rewrapping it into an extra physical
            # row, which would push the fixed 4-row window's true last row out
            # of view without any visible sign something was wrong.
            # Belt-and-suspenders only:
            # Textual's `Static` converts a `rich.Text` to `Content` via
            # `Content.from_rich_text`, which carries over the plain text and
            # spans but *not* `no_wrap`/`overflow` -- the enforcement that
            # actually reaches the screen is the `text_wrap`/`text_overflow`
            # widget styles set on `#console-command-visible-text` in
            # `compose()`. Kept here too for any renderer that (unlike
            # `Static`) does respect `Text`'s own flags, and so this stays
            # correct if a future Textual version stops dropping them.
            rendered = Text(
                "\n".join(line.text for line in line_slices),
                no_wrap=True,
                overflow="crop",
            )
            if style_ranges:
                output_offset = 0
                for line_index, line_slice in enumerate(line_slices):
                    source_to_output_offset = (
                        output_offset
                        + line_slice.synthetic_prefix_columns
                        - line_slice.start
                    )
                    for style_start, style_end, style in style_ranges:
                        span_start = max(style_start, line_slice.start)
                        span_end = min(style_end, line_slice.end)
                        if span_start < span_end:
                            rendered.stylize(
                                style,
                                span_start + source_to_output_offset,
                                span_end + source_to_output_offset,
                            )
                    output_offset += len(line_slice.text)
                    if line_index < len(line_slices) - 1:
                        output_offset += 1
            return rendered

        if focused:
            placeholder = Text(cls.CURSOR_GLYPH if cursor_visible else " ")
            placeholder.append(cls.DRAFT_PLACEHOLDER, style="bright_black")
            return placeholder
        return Text(cls.DRAFT_PLACEHOLDER, style="bright_black")

    def _placeholder_renderable(self, *, width: int) -> Text:
        """Return the empty composer placeholder copy."""
        return self._draft_renderable(
            "",
            width=width,
            focused=self.has_focus_within,
            cursor_visible=getattr(self, "_cursor_visible", True),
        )

    @classmethod
    def _visible_draft_row_count(
        cls,
        text: str,
        width: int,
        *,
        reserve_trailing_cell: bool = False,
    ) -> int:
        if not text:
            return cls.MIN_DRAFT_ROWS
        # Budget for the same reserved trailing cell _draft_renderable adds
        # while focused, computed once here (at focus/blur/mutation time,
        # never on a blink tick) so the exactly-at-width case gets its extra
        # row up front instead of only discovering it needs one mid-blink.
        measured_text = f"{text} " if reserve_trailing_cell else text
        return max(
            cls.MIN_DRAFT_ROWS,
            min(
                cls.MAX_DRAFT_ROWS,
                len(cls._wrap_draft_line_slices(measured_text, width)),
            ),
        )

    def _draft_render_width(self) -> int:
        try:
            visible_draft = self.query_one("#console-command-visible-text", Static)
        except NoMatches:
            return self.FALLBACK_DRAFT_WIDTH
        width = visible_draft.region.width or self.FALLBACK_DRAFT_WIDTH
        # The visible draft keeps one-column horizontal padding on both sides.
        return max(8, width - 2)

    def _apply_draft_height(self, row_count: int) -> None:
        row_count = max(self.MIN_DRAFT_ROWS, min(self.MAX_DRAFT_ROWS, row_count))
        composer_height = row_count + self.COMPOSER_CHROME_ROWS
        try:
            visible_draft = self.query_one("#console-command-visible-text", Static)
            visible_draft.styles.height = row_count
            visible_draft.styles.min_height = row_count
            visible_draft.styles.max_height = self.MAX_DRAFT_ROWS
            # The stylesheet pins `#console-composer-expanded` to height 1
            # (the collapsed sibling genuinely wants that). Left at 1, the
            # row CROPS the grown draft to a single painted line, vertically
            # centered in the taller bar -- the live-gate "text hidden past
            # the cutoff" report. Grow the row with the draft it contains.
            expanded = self.query_one("#console-composer-expanded", Horizontal)
            expanded.styles.height = row_count
            expanded.styles.min_height = row_count
        except NoMatches:
            pass
        self.styles.height = composer_height
        self.styles.min_height = self.MIN_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS
        self.styles.max_height = self.MAX_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS
        self.refresh(layout=True)

    def _apply_collapsed_geometry(self) -> None:
        """Pin the compact presentation to exactly one terminal row."""
        self.styles.height = 1
        self.styles.min_height = 1
        self.styles.max_height = 1
        self.refresh(layout=True)

    def _collapsed_status_text(self) -> str:
        """Build presence-only status copy without exposing retained content."""
        parts = ["Composer hidden"]
        if self._run_active:
            parts.append("Generating")
        if self._has_any_draft_content():
            parts.append("Draft retained")
        if self._pending_attachment_label is not None:
            parts.append("Attachment retained")
        return " · ".join(parts)

    def _sync_collapsed_presentation(self) -> None:
        """Synchronize stable presentation containers from cached widget state."""
        try:
            expanded = self.query_one("#console-composer-expanded", Horizontal)
            collapsed = self.query_one("#console-composer-collapsed", Horizontal)
            status = self.query_one("#console-composer-collapsed-status", Static)
            stop = self.query_one("#console-collapsed-stop-generation", Button)
        except NoMatches:
            return
        expanded.styles.display = "none" if self._collapsed else "block"
        collapsed.styles.display = "block" if self._collapsed else "none"
        status.update(self._collapsed_status_text())
        stop.styles.display = "block" if self._run_active else "none"
        self.set_class(self._collapsed, "console-composer-collapsed")

    def set_collapsed(self, collapsed: bool) -> None:
        """Switch presentation without remounting or clearing editor state.

        Args:
            collapsed: Whether to show the one-row restore presentation.
        """
        self._collapsed = bool(collapsed)
        self.can_focus = not self._collapsed
        self._sync_collapsed_presentation()
        self._sync_cursor_blink_state()
        if self._collapsed:
            self._apply_collapsed_geometry()
        else:
            self._refresh_visible_draft()

    def _insert_literal_at_cursor(self, text: str) -> None:
        """Splice literal text into the draft at the caret, coalescing segments.

        Paste tokens are never spliced into: text typed at a token boundary
        merges into the adjacent literal segment (or starts a new one).
        """
        if not self._segments:
            self._segments = [_DraftSegment(text)]
            self._cursor_index = len(text)
            return
        segment_index, offset = self._locate_canonical(self._cursor_index)
        segment = self._segments[segment_index]
        if segment.collapse_state in {"literal", "expanded"}:
            segment.text = segment.text[:offset] + text + segment.text[offset:]
            self._cursor_index += len(text)
            return
        if offset == len(segment.text):
            # Caret just past a paste token: prepend to the right literal
            # neighbour when possible, else start a new literal segment.
            right_index = segment_index + 1
            if (
                right_index < len(self._segments)
                and self._segments[right_index].collapse_state == "literal"
            ):
                self._segments[right_index].text = (
                    text + self._segments[right_index].text
                )
            else:
                self._segments.insert(right_index, _DraftSegment(text))
            self._cursor_index += len(text)
            return
        # Caret just before a leading paste token (offset == 0).
        left_index = segment_index - 1
        if left_index >= 0 and self._segments[left_index].collapse_state == "literal":
            self._segments[left_index].text += text
        else:
            self._segments.insert(segment_index, _DraftSegment(text))
        self._cursor_index += len(text)

    def _insert_segment_at_cursor(self, segment: _DraftSegment) -> None:
        """Insert a paste/file segment at the caret, splitting literal text."""
        if not self._segments:
            self._segments = [segment]
            self._cursor_index = len(segment.text)
            return
        segment_index, offset = self._locate_canonical(self._cursor_index)
        target = self._segments[segment_index]
        if target.collapse_state in {"collapsed", "confirm"}:
            insert_index = segment_index if offset == 0 else segment_index + 1
            self._segments.insert(insert_index, segment)
        else:
            left_text = target.text[:offset]
            right_text = target.text[offset:]
            replacement: list[_DraftSegment] = []
            if left_text:
                replacement.append(
                    _DraftSegment(
                        left_text,
                        collapse_state=target.collapse_state,
                        label=target.label,
                    )
                )
            replacement.append(segment)
            if right_text:
                replacement.append(
                    _DraftSegment(
                        right_text,
                        collapse_state=target.collapse_state,
                        label=target.label,
                    )
                )
            self._segments[segment_index : segment_index + 1] = replacement
        self._cursor_index += len(segment.text)

    def _delete_canonical_range(self, start: int, end: int) -> None:
        """Delete canonical text in ``[start, end)`` and move the caret there.

        Collapsed/confirm paste tokens intersecting the range are removed as
        whole units; literal and expanded segments keep their uncovered parts.
        """
        if start >= end:
            return
        kept_segments: list[_DraftSegment] = []
        offset = 0
        for segment in self._segments:
            segment_start = offset
            segment_end = offset + len(segment.text)
            offset = segment_end
            if segment_end <= start or segment_start >= end:
                kept_segments.append(segment)
                continue
            if segment.collapse_state in {"collapsed", "confirm"}:
                continue
            kept_text = (
                segment.text[: max(0, start - segment_start)]
                + segment.text[max(0, end - segment_start) :]
            )
            if kept_text:
                kept_segments.append(
                    _DraftSegment(
                        kept_text,
                        collapse_state=segment.collapse_state,
                        label=segment.label,
                    )
                )
        self._segments = kept_segments
        self._cursor_index = start
        self._clamp_cursor()

    def _current_visible_draft_renderable(self, draft: str, width: int) -> Text:
        """Build the Text renderable for the current draft/placeholder state."""
        if draft:
            focused = self.has_focus_within
            return self._draft_renderable(
                draft,
                width=width,
                style_ranges=self._display_draft_style_ranges(),
                focused=focused,
                cursor_visible=getattr(self, "_cursor_visible", True),
                cursor_index=(
                    self._cursor_display_index()
                    if focused and self._segments_initialized
                    else None
                ),
            )
        return self._placeholder_renderable(width=width)

    def _render_visible_draft_only(self) -> None:
        """Re-render the visible-draft Static without recomputing composer height.

        Used by the cursor blink tick, which must stay cheap and must not
        trigger a layout recompute on every blink phase.
        """
        try:
            draft = self._display_draft_text()
            width = self._draft_render_width()
            renderable = self._current_visible_draft_renderable(draft, width)
            self.query_one("#console-command-visible-text", Static).update(renderable)
        except NoMatches:
            return

    def _refresh_visible_draft(self) -> None:
        if self._collapsed:
            self._sync_collapsed_presentation()
            self._apply_collapsed_geometry()
            return
        try:
            # Any draft mutation or focus change shows a solid caret, matching
            # terminal cursor behavior (blink resets while actively editing).
            self._cursor_visible = True
            draft = self._display_draft_text()
            width = self._draft_render_width()
            row_count = self._visible_draft_row_count(
                draft, width, reserve_trailing_cell=self.has_focus_within
            )
            renderable = self._current_visible_draft_renderable(draft, width)
            self.query_one("#console-command-visible-text", Static).update(renderable)
            self._apply_draft_height(row_count)
        except NoMatches:
            return

    def _toggle_cursor_blink(self) -> None:
        """Flip the cursor blink phase and refresh only the visible draft."""
        self._cursor_visible = not self._cursor_visible
        self._render_visible_draft_only()

    def _sync_cursor_blink_state(self) -> None:
        """Start/stop the blink timer and reset caret visibility on focus changes."""
        timer = self._cursor_blink_timer
        self._cursor_visible = True
        if timer is None:
            return
        if self.has_focus_within and not self._collapsed:
            timer.resume()
        else:
            timer.pause()

    def on_mount(self) -> None:
        self._cursor_blink_timer = self.set_interval(
            self.CURSOR_BLINK_INTERVAL,
            self._toggle_cursor_blink,
            pause=True,
        )
        self._sync_cursor_blink_state()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def on_resize(self, event: Any) -> None:
        self._refresh_visible_draft()

    def on_focus(self) -> None:
        self._sync_interaction_classes()
        self._sync_cursor_blink_state()
        self._refresh_visible_draft()

    def on_blur(self) -> None:
        self._sync_interaction_classes()
        self._sync_cursor_blink_state()
        self._refresh_visible_draft()

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        self._sync_interaction_classes()
        self._sync_cursor_blink_state()
        self._refresh_visible_draft()

    def on_descendant_blur(self, event: DescendantBlur) -> None:
        self._sync_interaction_classes()
        self._sync_cursor_blink_state()
        self._refresh_visible_draft()

    def load_draft(self, text: str) -> None:
        """Replace the native Console draft with literal text.

        The caret lands at the end of the restored draft.

        TASK-1281 review F3/F4: this is always a SCOPE change (a session
        switch, or a launch-context prefill), never a recorded edit, so it
        unconditionally wipes any existing undo/redo history rather than
        leaving the previous scope's stale entries reachable -- a caller
        that wants to carry history across the call (the session-switch
        path in `ChatScreen._sync_console_session_draft`) must explicitly
        call `restore_undo_history` afterward, which it already does.
        Resetting `_coalescing_active` matters independently of wiping the
        stacks: left `True`, it would silently swallow the very first
        keystroke recorded against the new scope (nothing to merge into,
        but `_record_undo_snapshot` would still no-op).

        Args:
            text: Draft payload to show and send literally.
        """
        self._draft_selection_all = False
        self._segments = [_DraftSegment(text)] if text else []
        self._segments_initialized = True
        self._cursor_index = len(text)
        self._undo_stack = []
        self._redo_stack = []
        self._coalescing_active = False
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def stash_draft_for_send(self) -> ConsoleDraftStash | None:
        """Capture and clear the draft synchronously at the send keypress.

        Keystrokes processed after this call land in a fresh, empty draft —
        they can never fold into the captured send payload (TASK-340). A
        rejected send hands the stash back via ``restore_stashed_draft``.

        Returns:
            The captured stash, or ``None`` when the draft is empty (an
            image-only send has nothing to capture or restore).
        """
        text = self.draft_text()
        if not text:
            return None
        if not self._segments_initialized:
            self._segments = [_DraftSegment(text)]
            self._segments_initialized = True
        stash = ConsoleDraftStash(
            # Copies, not the live objects: segments are mutable, and a
            # restored draft keeps being edited — the stash must stay a
            # faithful snapshot of the keypress moment.
            segments=[replace(segment) for segment in self._segments],
            text=text,
            has_paste=self.has_paste_segments(),
        )
        self.clear_draft()
        return stash

    def restore_stashed_draft(self, stash: ConsoleDraftStash | None) -> None:
        """Put a stashed draft back, ahead of anything typed since the stash.

        The stashed segments are prepended so a rejected send reads exactly
        as before the keypress, with later keystrokes appended after it;
        paste provenance and collapse state come back untouched.

        Args:
            stash: The capture returned by ``stash_draft_for_send``, or
                ``None`` (image-only send — nothing to restore).
        """
        if stash is None or not stash.segments:
            return
        self._draft_selection_all = False
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
        self._segments = list(stash.segments) + self._segments
        self._cursor_index = len(self._canonical_draft_text())
        # TASK-1281 review F3: this replaces the draft wholesale without
        # recording (a rejected send putting the user's own text back is
        # not itself an edit), but it must still close any run left open
        # from before the send -- otherwise the first keystroke typed after
        # the restore silently coalesces into (and is swallowed by) an
        # unrelated pre-send entry instead of getting its own.
        self._coalescing_active = False
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    @property
    def edit_serial(self) -> int:
        """Monotonic count of user-originated draft edits (TASK-339)."""
        return self._user_edit_serial

    def clear_draft(self, *, record_history: bool = False) -> None:
        """Clear the native Console draft without falling back to stale input.

        Args:
            record_history: Whether this clear is user-intent and should be
                undoable (TASK-1281). Defaults to False -- most callers use
                this to swap draft scope programmatically (session switches,
                the post-send clear, restore-then-replace flows), and none
                of those should be revertable with Ctrl+Z. The one caller
                that must pass ``True`` is the Ctrl+U "clear draft" key
                handler in `ChatScreen.on_key` -- an accidental full clear is
                exactly what undo exists for.
        """
        if record_history and self._has_any_draft_content():
            self._record_undo_snapshot(coalesce=False)
        self._draft_selection_all = False
        self._segments = []
        self._segments_initialized = True
        self._cursor_index = 0
        # TASK-1281 review F3: even the non-recording branch replaces the
        # draft, so it must close any coalescing run left open from before
        # the clear -- otherwise the next typed character merges into (and
        # is swallowed by) a stale pre-clear entry instead of recording its
        # own. The recording branch above already resets this as a
        # side effect of `_record_undo_snapshot`, but the non-recording
        # (default) branch never calls it, so this cannot be conditional.
        self._coalescing_active = False
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def clear_history(self) -> None:
        """Empty both undo/redo stacks (TASK-1281 review F2: send is a history barrier).

        Called once a draft has been irrevocably consumed by an accepted
        send that will never be restored. Clearing just the draft
        (`clear_draft()`, always non-recording for a send) is not enough on
        its own -- the mutations that produced the sent text stay reachable
        on the undo stack, and Ctrl+Z would resurrect already-sent content
        back into the composer (and, via the screen's undo/redo
        re-persist, right back into the store as the "live" draft).
        """
        self._undo_stack = []
        self._redo_stack = []
        self._coalescing_active = False

    # -- Undo/redo history (TASK-1281) ------------------------------------
    #
    # Coalescing rule: consecutive single-character *printable* inserts
    # (ordinary typing) merge into one undo entry, so one Ctrl+Z reverts a
    # whole typed run. Every other mutation kind (paste, a file/attachment
    # segment, a delete, a full clear) always opens a fresh entry, and a
    # cursor reposition between keystrokes also closes the run -- both are
    # implemented by having every mutation/reposition entry point call
    # either `_record_undo_snapshot` (mutations) or set
    # `_coalescing_active = False` directly (repositions).

    def _record_undo_snapshot(self, *, coalesce: bool) -> None:
        """Push the pre-mutation draft state onto the undo stack.

        Must be called with the *current* (pre-mutation) segments/cursor
        still in place -- every call site here calls it after any lazy
        segment initialization but before the mutation itself splices
        anything.

        Args:
            coalesce: Whether this mutation is a candidate to merge into an
                already-open typed run. When True and the immediately
                previous recorded mutation was also coalescable and still
                open, this call is a no-op -- the run's original snapshot
                stays on top so a single undo reverts the whole run.
        """
        if coalesce and self._coalescing_active:
            return
        self._undo_stack.append(
            _DraftHistorySnapshot(text=self.draft_text(), cursor_index=self._cursor_index)
        )
        if len(self._undo_stack) > self.UNDO_HISTORY_DEPTH_CAP:
            del self._undo_stack[0]
        self._evict_to_char_budget(self._undo_stack)
        self._redo_stack.clear()
        self._coalescing_active = coalesce

    @classmethod
    def _evict_to_char_budget(cls, stack: list[_DraftHistorySnapshot]) -> None:
        """Drop the oldest entries of ``stack`` while it exceeds the char budget.

        TASK-1281 review F6. Never evicts the single most recent entry, even
        if it alone exceeds the budget -- a best-effort bound on total
        retained memory, not a hard guarantee, since a single oversized
        snapshot (a large inlined attachment) must still be revertible.
        """
        total = sum(len(entry.text) for entry in stack)
        while total > cls.UNDO_HISTORY_CHAR_BUDGET and len(stack) > 1:
            removed = stack.pop(0)
            total -= len(removed.text)

    def _apply_history_snapshot(self, snapshot: _DraftHistorySnapshot) -> None:
        """Replace the live draft with a recorded undo/redo snapshot.

        TASK-1281 review NEW-2 (fix shape corrected by review W-1/W-2): a
        restored segment over `UNDO_RECOLLAPSE_CHAR_THRESHOLD` is created
        COLLAPSED -- the same paste-token mechanics `insert_pasted_text`
        already uses for a real paste over `paste_collapse_threshold` --
        rather than as one giant literal segment. Restoring it as a flat
        literal used to run `_refresh_visible_draft`'s O(n^2) wrap/render
        path against the FULL text on every undo/redo: measured up to 283s
        frozen on the main thread for a 2.4 MB restored draft. F6's
        char-budget eviction deliberately keeps such large snapshots
        revertible, which is exactly what made this reachable by an
        ordinary Ctrl+Z rather than only a contrived one.

        The gate is `UNDO_RECOLLAPSE_CHAR_THRESHOLD` (20,000 chars),
        deliberately NOT `paste_collapse_threshold` (a cosmetic
        paste-display preference, shipped default 50) and NOT gated on
        `collapse_large_pastes_enabled` -- an earlier version of this fix
        used both, which converted ordinary typed draft text into an
        opaque token on every undo/redo over 50 characters (review W-1,
        HIGH) and let a user disabling the display preference bring the
        freeze back in full (review W-2, LOW). See the constant's own
        docstring for the measurements behind the threshold value.

        A redo landing back on a snapshot taken while a large paste was
        still collapsed correctly shows the collapsed token again, not the
        fully expanded literal text -- see the module docstring for the
        (narrower) limitation that remains: the restored token is always a
        generic "Pasted Text: N Characters" collapse, never the original
        segment's label (a labeled file/attachment segment, or one already
        `expanded`/mid-`confirm`, is not carried through the flat snapshot
        -- only the raw text and whether it crosses the threshold are).

        Collapsed tokens are atomic for the caret everywhere else in this
        widget (no other code path leaves it mid-token), so when the
        restored segment collapses, `snapshot.cursor_index` -- recorded
        against whatever the segment structure was AT SNAPSHOT time, which
        may not have been collapsed at all -- is snapped to whichever
        token edge (0 or the full text length) it was nearer to, rather
        than restored verbatim into what is now the middle of a token.
        """
        self._draft_selection_all = False
        text_length = len(snapshot.text)
        raw_cursor = max(0, min(snapshot.cursor_index, text_length))
        if not snapshot.text:
            self._segments = []
            self._cursor_index = 0
        elif text_length > self.UNDO_RECOLLAPSE_CHAR_THRESHOLD:
            self._segments = [
                _DraftSegment(snapshot.text, collapse_state="collapsed")
            ]
            self._cursor_index = 0 if raw_cursor * 2 < text_length else text_length
        else:
            self._segments = [_DraftSegment(snapshot.text)]
            self._cursor_index = raw_cursor
        self._segments_initialized = True
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def undo(self) -> bool:
        """Revert the most recent recorded composer draft mutation.

        A silent no-op when there is nothing to undo -- callers should not
        toast or bell on an empty stack.

        Returns:
            True when a snapshot was applied (the caller is then
            responsible for re-persisting `draft_text()`, mirroring
            dictation insertion).
        """
        if not self._undo_stack:
            return False
        self._user_edit_serial += 1
        current = _DraftHistorySnapshot(text=self.draft_text(), cursor_index=self._cursor_index)
        self._redo_stack.append(current)
        if len(self._redo_stack) > self.UNDO_HISTORY_DEPTH_CAP:
            del self._redo_stack[0]
        self._evict_to_char_budget(self._redo_stack)
        snapshot = self._undo_stack.pop()
        self._apply_history_snapshot(snapshot)
        self._coalescing_active = False
        return True

    def redo(self) -> bool:
        """Reapply a composer draft mutation that was just undone.

        A silent no-op when there is nothing to redo.

        Returns:
            True when a snapshot was applied.
        """
        if not self._redo_stack:
            return False
        self._user_edit_serial += 1
        current = _DraftHistorySnapshot(text=self.draft_text(), cursor_index=self._cursor_index)
        self._undo_stack.append(current)
        if len(self._undo_stack) > self.UNDO_HISTORY_DEPTH_CAP:
            del self._undo_stack[0]
        self._evict_to_char_budget(self._undo_stack)
        snapshot = self._redo_stack.pop()
        self._apply_history_snapshot(snapshot)
        self._coalescing_active = False
        return True

    def export_undo_history(self) -> ConsoleComposerUndoHistory:
        """Return a copy of this composer's undo/redo stacks.

        Used by `ChatScreen` to scope history per Console session (TASK-1281
        AC4): exported on switch-away, restored after `load_draft` on
        switch-in. The returned lists are copies -- safe for the caller to
        hold in a dict keyed by session id without aliasing this composer's
        live stacks.
        """
        return (list(self._undo_stack), list(self._redo_stack))

    def restore_undo_history(
        self, history: ConsoleComposerUndoHistory | None
    ) -> None:
        """Replace the undo/redo stacks wholesale (TASK-1281 session scoping).

        Args:
            history: A prior `export_undo_history()` result, or None for an
                empty history (a session that has never had a recorded
                edit -- freshly created, or never visited before).
        """
        undo_entries, redo_entries = history if history is not None else ([], [])
        self._undo_stack = list(undo_entries)
        self._redo_stack = list(redo_entries)
        # TASK-1281 review F6: a caller-supplied history (banked across a
        # session switch, potentially from before this composer instance's
        # own char-budget enforcement existed, or simply handed in from
        # elsewhere) is re-enforced here too rather than trusted as already
        # within budget.
        self._evict_to_char_budget(self._undo_stack)
        self._evict_to_char_budget(self._redo_stack)
        self._coalescing_active = False

    def select_all_draft(self) -> bool:
        """Mark the full visible Console draft as selected without mutating it.

        Returns:
            True when there is draft text to select, otherwise False.
        """
        if not self.draft_text():
            self._draft_selection_all = False
            self._refresh_visible_draft()
            return False
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
        self._draft_selection_all = True
        self._cursor_index = len(self._canonical_draft_text())
        # TASK-1281: a select-all is a cursor reposition (to the tail) for
        # undo-coalescing purposes -- it closes any open typed run.
        self._coalescing_active = False
        self._refresh_visible_draft()
        return True

    def has_full_draft_selection(self) -> bool:
        """Return whether the composer currently has a full-draft selection.

        Returns:
            True when the visible draft exists and is fully selected.
        """
        return self._draft_selection_all and bool(self.draft_text())

    def insert_text(self, text: str) -> None:
        """Insert user-entered text into the Console draft at the caret.

        Args:
            text: Typed text to insert without paste-collapse transformation.
        """
        self._user_edit_serial += 1
        if not text:
            self._sync_interaction_classes()
            self._sync_current_action_state()
            return
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
            self._cursor_index = len(existing)
        # TASK-1281: a single printable character (ordinary typing) coalesces
        # into an already-open typed run; anything else (a multi-character
        # insert -- e.g. dictation -- or a non-printable one, like the
        # Shift+Enter newline) always opens a fresh undo entry.
        self._record_undo_snapshot(coalesce=len(text) == 1 and text.isprintable())
        if self._draft_selection_all:
            self._segments = []
            self._draft_selection_all = False
            self._cursor_index = 0
        self._reset_pending_unfurl_state()
        self._clamp_cursor()
        self._insert_literal_at_cursor(text)
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def insert_pasted_text(self, text: str) -> None:
        """Insert pasted text at the caret, collapsing only large chunks for display.

        Args:
            text: Raw text inserted through a paste event.
        """
        if not text:
            self._sync_interaction_classes()
            self._sync_current_action_state()
            return
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
            self._cursor_index = len(existing)
        # TASK-1281: a paste is always its own undo entry, even a
        # single-character one -- it is a different mutation kind from
        # typing and must never silently merge into an open typed run.
        self._record_undo_snapshot(coalesce=False)
        if self._draft_selection_all:
            self._segments = []
            self._draft_selection_all = False
            self._cursor_index = 0
        self._reset_pending_unfurl_state()
        self._clamp_cursor()
        should_collapse = (
            self.collapse_large_pastes_enabled
            and len(text) > self.paste_collapse_threshold
        )
        if should_collapse:
            self._insert_segment_at_cursor(
                _DraftSegment(text, collapse_state="collapsed")
            )
        else:
            self._insert_literal_at_cursor(text)
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def insert_text_as_paste(self, text: str) -> None:
        """Insert ``text`` through the same path a real OS paste event uses.

        Thin, clearly-named public entry point for programmatic insertions
        (the Console `/prompt` command and Library's "Use in Console"
        handoff, Task 12) that must behave exactly like a user pasting the
        same text -- collapsing into a stateful token display when it
        exceeds the paste-collapse threshold, unlike ``insert_text``, which
        always inserts as small literal text regardless of size.

        Args:
            text: Text to insert as if it had just been pasted.
        """
        self._user_edit_serial += 1
        self.insert_pasted_text(text)

    def insert_file_segment(self, text: str, label: str) -> None:
        """Insert inlined file content at the caret as a labeled, display-collapsed segment.

        Args:
            text: Full file text that becomes part of the canonical draft.
            label: Display-only token shown in place of the text (e.g.
                ``"📄 notes.md · 4 KB"``).
        """
        if not text:
            self._sync_interaction_classes()
            self._sync_current_action_state()
            return
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
            self._cursor_index = len(existing)
        # TASK-1281: an attachment/file segment is always its own undo entry.
        self._record_undo_snapshot(coalesce=False)
        if self._draft_selection_all:
            self._segments = []
            self._draft_selection_all = False
            self._cursor_index = 0
        self._reset_pending_unfurl_state()
        self._clamp_cursor()
        self._insert_segment_at_cursor(
            _DraftSegment(text, collapse_state="collapsed", label=label)
        )
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def _ensure_editable_segments(self) -> None:
        """Initialize segments from any legacy draft text, caret at the end."""
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
            self._cursor_index = len(existing)

    def delete_left(self) -> None:
        """Delete the character (or paste token) immediately left of the caret."""
        self._user_edit_serial += 1
        if self._draft_selection_all:
            # TASK-1281: record before dispatching to `clear_draft` -- its
            # own `record_history` default is False (most callers are
            # programmatic), so a Backspace-over-a-full-selection must record
            # its own entry here rather than rely on the callee.
            self._record_undo_snapshot(coalesce=False)
            self.clear_draft()
            return
        # TASK-1281 review NEW-1: previously this branch shortcut through
        # `self.load_draft(self.draft_text()[:-1])` -- but `load_draft` now
        # unconditionally wipes both undo/redo stacks (F4), which silently
        # discarded the snapshot just recorded a line above it, making this
        # deletion the one path in the whole widget that was NOT undoable.
        # `_ensure_editable_segments()` is the same lazy-init helper every
        # other mutator here already uses (`delete_right`/`delete_word_left`
        # included); it initializes segments from the legacy draft without
        # touching history, so the ordinary deletion logic below can record
        # and splice exactly as it does once the composer is initialized.
        self._ensure_editable_segments()
        self._clamp_cursor()
        if not self._segments or self._cursor_index == 0:
            self._sync_interaction_classes()
            self._sync_current_action_state()
            return

        self._record_undo_snapshot(coalesce=False)
        segment_index, offset = self._locate_canonical(self._cursor_index)
        segment = self._segments[segment_index]
        if segment.collapse_state in {"collapsed", "confirm"}:
            # A paste token deletes as a unit; the caret lands where it started.
            self._cursor_index -= offset
            del self._segments[segment_index]
        else:
            segment.text = segment.text[: offset - 1] + segment.text[offset:]
            self._cursor_index -= 1
            if not segment.text:
                del self._segments[segment_index]
        self._clamp_cursor()
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def delete_right(self) -> None:
        """Delete the character (or paste token) immediately right of the caret."""
        self._user_edit_serial += 1
        if self._draft_selection_all:
            self._record_undo_snapshot(coalesce=False)
            self.clear_draft()
            return
        self._ensure_editable_segments()
        self._clamp_cursor()
        if not self._segments or self._cursor_index >= len(
            self._canonical_draft_text()
        ):
            self._sync_interaction_classes()
            self._sync_current_action_state()
            return

        self._record_undo_snapshot(coalesce=False)
        segment_index, offset = self._locate_canonical(self._cursor_index)
        segment = self._segments[segment_index]
        if offset == len(segment.text):
            # Caret on a boundary: the next segment holds the deletion target.
            next_segment = self._segments[segment_index + 1]
            if next_segment.collapse_state in {"collapsed", "confirm"}:
                del self._segments[segment_index + 1]
            else:
                next_segment.text = next_segment.text[1:]
                if not next_segment.text:
                    del self._segments[segment_index + 1]
        elif segment.collapse_state in {"collapsed", "confirm"}:
            del self._segments[segment_index]
        else:
            segment.text = segment.text[:offset] + segment.text[offset + 1 :]
            if not segment.text:
                del self._segments[segment_index]
        self._clamp_cursor()
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def delete_word_left(self) -> bool:
        """Delete the whitespace+word run left of the caret (readline Ctrl+W).

        Collapsed/confirm paste tokens overlapped by the deleted range are
        removed as whole units; a word boundary never splits a token.

        Returns:
            True when text (or a full-draft selection) was deleted.
        """
        self._user_edit_serial += 1
        if self._draft_selection_all:
            self._record_undo_snapshot(coalesce=False)
            self.clear_draft()
            return True
        self._ensure_editable_segments()
        self._clamp_cursor()
        canonical = self._canonical_draft_text()
        cursor = self._cursor_index
        if cursor == 0:
            return False
        token_ranges: list[tuple[int, int]] = []
        offset = 0
        for segment in self._segments:
            segment_end = offset + len(segment.text)
            if segment.collapse_state in {"collapsed", "confirm"}:
                token_ranges.append((offset, segment_end))
            offset = segment_end

        def inside_token(index: int) -> bool:
            return any(start <= index < end for start, end in token_ranges)

        # Readline word-rubout, token-aware: skip the whitespace run left of
        # the caret, then delete one word -- where a collapsed paste token
        # counts as a single opaque word and is never split.
        start = cursor
        while (
            start > 0 and canonical[start - 1].isspace() and not inside_token(start - 1)
        ):
            start -= 1
        if start > 0 and inside_token(start - 1):
            start = min(
                token_start
                for token_start, token_end in token_ranges
                if token_start <= start - 1 < token_end
            )
        else:
            while (
                start > 0
                and not canonical[start - 1].isspace()
                and not inside_token(start - 1)
            ):
                start -= 1
        if start == cursor:
            return False
        self._record_undo_snapshot(coalesce=False)
        self._delete_canonical_range(start, cursor)
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()
        return True

    def _move_cursor_to(self, index: int) -> bool:
        """Move the caret to a canonical offset, collapsing any selection."""
        # TASK-1281: every caller of this helper (arrow keys, Home/End) is a
        # cursor reposition, which always closes an open typed run.
        self._coalescing_active = False
        self._draft_selection_all = False
        if not self._segments_initialized:
            self._refresh_visible_draft()
            return False
        previous = self._cursor_index
        self._cursor_index = index
        self._clamp_cursor()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        return self._cursor_index != previous

    def move_cursor_left(self) -> bool:
        """Move the caret one character left, skipping paste tokens as units.

        Returns:
            True when the caret moved.
        """
        self._clamp_cursor()
        if self._cursor_index <= 0:
            return self._move_cursor_to(0)
        segment_index, offset = self._locate_canonical(self._cursor_index)
        segment = self._segments[segment_index]
        if segment.collapse_state in {"collapsed", "confirm"}:
            return self._move_cursor_to(self._cursor_index - offset)
        return self._move_cursor_to(self._cursor_index - 1)

    def move_cursor_right(self) -> bool:
        """Move the caret one character right, skipping paste tokens as units.

        Returns:
            True when the caret moved.
        """
        self._clamp_cursor()
        if self._cursor_index >= len(self._canonical_draft_text()):
            return self._move_cursor_to(self._cursor_index)
        segment_index, offset = self._locate_canonical(self._cursor_index)
        segment = self._segments[segment_index]
        if offset == len(segment.text):
            # Caret on a boundary: the next segment is the move target.
            next_segment = self._segments[segment_index + 1]
            step = (
                len(next_segment.text)
                if next_segment.collapse_state in {"collapsed", "confirm"}
                else 1
            )
        elif segment.collapse_state in {"collapsed", "confirm"}:
            step = len(segment.text) - offset
        else:
            step = 1
        return self._move_cursor_to(self._cursor_index + step)

    def move_cursor_up(self) -> bool:
        """Move the caret up one visual (wrapped) row, at the same column.

        "Same column" is measured in CHARACTERS within the row's DISPLAY
        text -- the same unit `move_cursor_left`/`move_cursor_right` already
        step through -- and is clamped to the target row's own length when
        it is shorter. There is no goal-column memory across calls: every
        move recomputes its starting column from the caret's row at the
        moment it is called, so a run of consecutive Up presses through rows
        of varying length can drift the apparent column over time, exactly
        as repeated left/right stepping already has no memory of a
        "preferred" position either. On a row containing double-width
        (CJK/emoji) content, "same column" in characters is not "same
        column" in terminal cells -- this can visually drift the caret
        sideways on such a row, though it never lands outside the target
        row's own bounds (the clamp is character-count, applied after the
        drift) or corrupts the draft.

        Returns:
            True when the caret's canonical offset actually changed. False
            covers every case where nothing moved: the caret is already on
            the topmost visual row, the composer has no draft segments
            initialized yet, or (rare -- e.g. snapping to a collapsed paste
            token's edge) the mapped target happens to equal the caret's
            current offset. Even on False, this still routes through the
            same `_move_cursor_to` reposition chokepoint every other move
            method uses: any full-draft selection is collapsed and
            undo-coalescing is broken, exactly as `move_cursor_left`
            already does when called at index 0.
        """
        return self._move_cursor_vertically(-1)

    def move_cursor_down(self) -> bool:
        """Move the caret down one visual (wrapped) row, at the same column.

        See `move_cursor_up` for the full column/clamping/no-goal-column-
        memory/double-width-drift contract -- identical here, mirrored
        downward.

        Returns:
            True when the caret's canonical offset actually changed. False
            covers every case where nothing moved -- see `move_cursor_up`;
            here the boundary case is the bottommost visual row instead of
            the topmost. Even on False this still collapses any full-draft
            selection and breaks undo-coalescing, exactly like every other
            move method's own boundary case.
        """
        return self._move_cursor_vertically(1)

    @staticmethod
    def _row_index_for_canonical_offset(
        line_slices: list[_DraftLineSlice], offset: int
    ) -> int:
        """Return the row an UNSPLICED source offset visually belongs to.

        The row with the greatest `start <= offset`. `line_slices` is always
        in non-decreasing `start` order and rows never overlap, so this
        single rule covers three cases at once:

        - strictly inside a row (`start <= offset < end`) -> that row: no
          later row's `start` can also be `<= offset`, since the next row's
          `start` is always `>= end`;
        - a CONTIGUOUS soft-wrap boundary (row N's `end` == row N+1's
          `start`, no separator between them) -> row N+1 -- matching
          `_row_index_for_source_offset`'s own existing convention for the
          same boundary shape, and how a real caret glyph actually paints
          there (`_draft_renderable`'s own wrap pushes it onto the
          following row, attached to that row's next word);
        - a GAP between two explicit-newline-separated rows (row N's `end`
          is the newline character's own offset, strictly less than row
          N+1's `start`) -> row N, i.e. "the end of that row", which is
          where a caret sitting there visually belongs. This is the case
          `_row_index_for_source_offset` gets wrong for this caller: its
          fallback (`len(line_slices) - 1`, unconditionally the LAST row)
          is only ever exercised by ITS OWN production callers against
          caret-SPLICED text, where a real character always occupies the
          exact offset being looked up and a gap is therefore never
          actually reachable. Called against genuinely unspliced text (this
          method's whole point), a gap offset is an expected, common input
          -- not an edge case -- so it needs its own correct handling
          rather than reuse of a fallback tuned for a different caller.
        """
        row_index = 0
        for index, line_slice in enumerate(line_slices):
            if line_slice.start > offset:
                break
            row_index = index
        return row_index

    def _move_cursor_vertically(self, row_delta: int) -> bool:
        """Shared row-stepping logic behind `move_cursor_up`/`move_cursor_down`.

        Maps the caret to (visual row, column) and back entirely in
        UNSPLICED coordinates -- no caret-glyph/placeholder splice, no
        post-hoc correction constant. An earlier version spliced a
        placeholder character in at the caret before wrapping (mirroring
        `_draft_renderable`'s "reserved caret cell" painting technique) to
        disambiguate the newline-gap case below, then shifted the mapped
        target back by one character to undo the splice's effect. That
        splice only leaves rows AT OR BEFORE the caret's own row unchanged;
        every row AT OR AFTER it shifts by the placeholder's one character
        once the text re-wraps around it. A single trailing `-1` correction
        cannot undo that for a downward move, whose whole target lives in
        the shifted region -- confirmed by a live/differential review
        finding a systematic one-column-left drift on `move_cursor_down`
        across soft-wrapped rows (114/150 sampled positions wrong on one
        fixture), including a degenerate case where Down from column 0
        didn't change rows at all. The same splice also disagreed with
        `_draft_renderable`'s own painted caret at whitespace wrap
        boundaries: a space (this method's old placeholder) extends a
        trailing whitespace run and stays on the earlier row, while
        `CURSOR_GLYPH` (what actually paints) attaches to the following
        word and wraps onto the next one. Operating in unspliced
        coordinates throughout removes both mismatches at the root instead
        of patching the correction: there is no splice to leave stale rows
        behind, and an unspliced offset lands on whichever row it would
        visually belong to once a real character (the eventual typed input,
        or the painted glyph) actually occupied it -- see
        `_row_index_for_canonical_offset` for exactly which row that is,
        including the newline-gap case the naive unspliced lookup
        (`_row_index_for_source_offset`, tuned for spliced callers) still
        gets wrong on its own.

        Row boundaries come from the FULL, unwindowed wrap of the display
        draft (`_wrap_draft_line_slices`) -- not the windowed/prefixed
        `_visible_draft_line_slices` the bounded 4-row composer actually
        paints. This sidesteps the windowed view's synthetic "... " prefix
        on its top visible row entirely: that prefix trims leading DISPLAY
        characters for paint only, and mapping against it would otherwise
        drift column math by the prefix's own width whenever the caret's
        target row happens to be a windowed top row. The caret's actual
        on-screen row is left to resolve on the very next
        `_refresh_visible_draft()` (run by `_move_cursor_to` below), which
        already re-centers the caret-following window -- not reimplemented
        here.

        Column offsets are computed in DISPLAY space (`_display_draft_text`/
        `_cursor_display_index`), then mapped back to a canonical draft
        offset via `_canonical_index_at_display` -- the same display<->
        canonical translation the click-to-position path already performs,
        so a caret landing "inside" a collapsed paste token snaps to the
        token's nearest edge exactly as a click would.

        The column clamp ceiling is the target row's own length -- EXCEPT
        when the target row is soft-wrap-CONTIGUOUS with its own successor
        (no separator between them, i.e. `target_slice.end ==
        line_slices[target_row + 1].start`). Landing exactly at that row's
        full length would place the offset AT that shared join point, which
        `_row_index_for_canonical_offset` (correctly, per its own
        contiguous-boundary convention -- and matching how the painted
        caret glyph resolves the identical join) resolves to the
        SUCCESSOR row, not the intended target: a caret "Up" from a late
        column would silently fail to change rows at all (reading as a
        no-op that still consumes the key), and "Down" from the same shape
        would skip an entire row. The clamp ceiling is one character short
        of the full length in that case instead, keeping the landing
        offset strictly inside the target row. A row that ends in a real
        newline (or is the draft's own last row) keeps the full-length
        clamp unchanged: landing one past its last character is a
        distinct, legitimate caret position there (the separator gives
        "end of row" its own unambiguous offset), with no successor row's
        start to collide with.

        Args:
            row_delta: -1 to move up one row, +1 to move down one row.

        Returns:
            True when a target row exists in that direction AND the mapped
            canonical offset differs from the caret's current one. False
            otherwise -- routed through `_move_cursor_to(self._cursor_index)`
            when there is no target row at all, so a boundary press still
            collapses any full-draft selection and breaks undo-coalescing
            exactly like every other move method's own boundary case.
        """
        self._clamp_cursor()
        if not self._segments_initialized:
            return self._move_cursor_to(self._cursor_index)
        display_text = self._display_draft_text()
        line_slices = self._wrap_draft_line_slices(
            display_text, self._draft_render_width()
        )
        caret_display_index = max(
            0, min(self._cursor_display_index(), len(display_text))
        )
        current_row = self._row_index_for_canonical_offset(
            line_slices, caret_display_index
        )
        target_row = current_row + row_delta
        if target_row < 0 or target_row >= len(line_slices):
            return self._move_cursor_to(self._cursor_index)
        current_slice = line_slices[current_row]
        target_slice = line_slices[target_row]
        column = caret_display_index - current_slice.start
        target_row_length = len(target_slice.text)
        target_contiguous_with_successor = (
            target_row + 1 < len(line_slices)
            and target_slice.end == line_slices[target_row + 1].start
        )
        clamp_ceiling = (
            target_row_length - 1
            if target_contiguous_with_successor and target_row_length > 0
            else target_row_length
        )
        clamped_column = min(column, clamp_ceiling)
        target_display_index = target_slice.start + clamped_column
        canonical_index = self._canonical_index_at_display(target_display_index)
        return self._move_cursor_to(canonical_index)

    def move_cursor_home(self) -> bool:
        """Move the caret to the start of the draft.

        Returns:
            True when the caret moved.
        """
        return self._move_cursor_to(0)

    def move_cursor_end(self) -> bool:
        """Move the caret to the end of the draft.

        Returns:
            True when the caret moved.
        """
        if not self._segments_initialized:
            return self._move_cursor_to(self._cursor_index)
        return self._move_cursor_to(len(self._canonical_draft_text()))

    def position_cursor_from_display_index(self, display_index: int) -> bool:
        """Set the caret from an unwrapped display-string offset (click-to-position).

        Args:
            display_index: Offset into the visible draft text; offsets landing
                on a collapsed paste token snap to the token's nearest edge.

        Returns:
            True when the caret was positioned.
        """
        self._ensure_editable_segments()
        # TASK-1281: click-to-position is a cursor reposition too.
        self._coalescing_active = False
        self._draft_selection_all = False
        self._cursor_index = self._canonical_index_at_display(display_index)
        self._clamp_cursor()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        return True

    def _reset_pending_unfurl_state(self) -> bool:
        """Reset pending paste unfurl confirmations without refreshing display."""
        changed = False
        for segment in self._segments:
            if segment.collapse_state == "confirm":
                segment.collapse_state = "collapsed"
                changed = True
        return changed

    def reset_pending_unfurl(self) -> bool:
        """Reset any pending paste unfurl confirmations back to collapsed tokens.

        Returns:
            True when at least one visible confirmation prompt was reset.
        """
        changed = self._reset_pending_unfurl_state()
        if changed:
            self._refresh_visible_draft()
        return changed

    def has_pending_paste_confirmation(self) -> bool:
        """Return whether a collapsed paste token is waiting on confirm.

        Returns:
            True when at least one pasted segment is showing the `Unfurl?` prompt.
        """
        return any(segment.collapse_state == "confirm" for segment in self._segments)

    def has_paste_segments(self) -> bool:
        """Return whether the draft contains any paste-originated segment.

        A segment keeps its paste provenance (``"collapsed"``, ``"confirm"``,
        or ``"expanded"``) for as long as it exists, even once fully unfurled
        -- an "expanded" segment renders identically to typed literal text, so
        display-string inspection cannot distinguish the two. Callers that
        must not treat pasted content as command input (for example, Console
        slash-command parsing) should gate on this real segment state instead.

        Returns:
            True when at least one segment originated from a paste event.
        """
        return any(segment.collapse_state != "literal" for segment in self._segments)

    def suppress_next_draft_click(self) -> None:
        """Suppress the synthesized Click that may follow terminal mouse events."""
        self._suppress_next_draft_click = True

    def has_suppressed_draft_click(self) -> bool:
        """Return whether a synthesized draft click is currently suppressed.

        Returns:
            True when a prior terminal mouse event already handled the visible draft
            interaction and the next matching click should be ignored.
        """
        return self._suppress_next_draft_click

    def clear_suppressed_draft_click(self) -> None:
        """Clear any pending synthesized draft-click suppression."""
        self._suppress_next_draft_click = False

    def consume_suppressed_draft_click(self) -> bool:
        """Consume pending synthesized click suppression.

        Returns:
            True when a prior mouse-event fallback already handled this click.
        """
        if not self._suppress_next_draft_click:
            return False
        self._suppress_next_draft_click = False
        return True

    def activate_focused_paste_token(self) -> bool:
        """Advance a collapsed paste token for keyboard-only users.

        The visible composer draft renders paste tokens inline rather than as
        individually focusable widgets. When the composer owns focus, Enter
        advances the active confirmation if present, otherwise it prompts the
        first collapsed paste token.

        Returns:
            True when a paste token was advanced.
        """
        if not self._segments_initialized:
            return False

        try:
            visible_draft = self.query_one("#console-command-visible-text", Static)
            refocus_composer = self.app.focused in {self, visible_draft}
        except NoMatches:
            refocus_composer = self.app.focused is self

        for segment in self._segments:
            if segment.collapse_state == "confirm":
                segment.collapse_state = "expanded"
                self._refresh_visible_draft()
                if refocus_composer:
                    self.focus()
                return True

        for segment in self._segments:
            if segment.collapse_state == "collapsed":
                segment.collapse_state = "confirm"
                self._refresh_visible_draft()
                if refocus_composer:
                    self.focus()
                return True

        return False

    def _display_index_at(
        self, click_x: int, click_y: int, *, padding_left: int = 0
    ) -> int | None:
        """Map visible-draft coordinates to an unwrapped display-string offset.

        While the composer is focused the rendered draft carries a reserved
        caret cell at the caret position; the same cell is spliced in here so
        click coordinates line up with the rows actually on screen (a click on
        the caret cell itself maps to the caret position).
        """
        click_x = max(0, click_x - padding_left)
        click_y = max(0, click_y)
        display_text = self._display_draft_text()
        caret_position: int | None = None
        if self.has_focus_within and self._segments_initialized and display_text:
            caret_position = max(
                0, min(self._cursor_display_index(), len(display_text))
            )
            render_text = (
                f"{display_text[:caret_position]} {display_text[caret_position:]}"
            )
        else:
            render_text = display_text
        visible_slices = self._visible_draft_line_slices(
            render_text,
            self._draft_render_width(),
            cursor_index=caret_position,
        )
        if click_y >= len(visible_slices):
            return None
        clicked_slice = visible_slices[click_y]
        if click_x >= len(clicked_slice.text):
            return None
        if clicked_slice.synthetic_prefix_columns:
            if click_x < clicked_slice.synthetic_prefix_columns:
                return None
            source_index = (
                clicked_slice.start + click_x - clicked_slice.synthetic_prefix_columns
            )
        else:
            source_index = clicked_slice.start + click_x
        if caret_position is not None:
            if source_index == caret_position:
                return caret_position
            if source_index > caret_position:
                return source_index - 1
        return source_index

    def _click_display_index(self, event: Click) -> int | None:
        """Map a visible-draft click to an unwrapped display-string offset."""
        widget = getattr(event, "widget", None) or getattr(event, "control", None)
        padding_left = getattr(getattr(widget, "styles", None), "padding", None)
        padding_left = getattr(padding_left, "left", 0)
        return self._display_index_at(event.x, event.y, padding_left=padding_left)

    def _target_unfurl_segment_at(
        self,
        click_x: int,
        click_y: int,
        *,
        padding_left: int = 0,
    ) -> _DraftSegment | None:
        """Return the collapsed paste segment targeted by display coordinates."""
        display_index = self._display_index_at(
            click_x, click_y, padding_left=padding_left
        )
        if display_index is None:
            return None
        for display_range in self._segment_display_ranges():
            segment = display_range.segment
            if (
                display_range.start <= display_index < display_range.end
                and segment.collapse_state in {"collapsed", "confirm"}
            ):
                return segment
        return None

    def _advance_targeted_paste_segment(
        self,
        click_x: int,
        click_y: int,
        *,
        padding_left: int = 0,
    ) -> bool:
        """Advance the simple two-step unfurl flow for a targeted paste segment."""
        segment = self._target_unfurl_segment_at(
            click_x,
            click_y,
            padding_left=padding_left,
        )
        if segment is None:
            changed = bool(self._reset_pending_unfurl_state())
            if changed:
                self._refresh_visible_draft()
            return changed
        if segment.collapse_state == "collapsed":
            segment.collapse_state = "confirm"
        elif segment.collapse_state == "confirm":
            segment.collapse_state = "expanded"
        self._refresh_visible_draft()
        return True

    @staticmethod
    def _screen_region(widget: Widget) -> Region:
        """Return the mounted widget region in screen coordinates.

        Textual versions used by this project do not expose `screen_region`.
        Mounted `Widget.region` values are already screen-relative here, so this
        helper centralizes that contract for event hit testing.
        """
        return widget.region

    def _visible_draft_screen_hit(
        self,
        screen_x: int,
        screen_y: int,
    ) -> tuple[Static, int, int] | None:
        """Map absolute screen coordinates to visible draft-local coordinates."""
        try:
            visible_draft = self.query_one("#console-command-visible-text", Static)
        except NoMatches:
            return None

        visible_region = self._screen_region(visible_draft)
        composer_region = self._screen_region(self)
        local_y = screen_y - visible_region.y
        if local_y == -1 and screen_y >= composer_region.y:
            # textual-web reports some bottom-row composer clicks against the
            # containing row above the Static while visually targeting the
            # visible draft. Treat that boundary as the first draft row.
            local_y = 0
        elif (
            local_y == visible_draft.size.height
            and screen_y < composer_region.y + composer_region.height
        ):
            # textual-web can also report a composer-row click one row below the
            # visible Static. Treat that boundary as the draft row when the x
            # coordinate still targets the draft surface.
            local_y = max(0, visible_draft.size.height - 1)

        if (
            screen_x < visible_region.x
            or screen_x >= visible_region.x + visible_draft.size.width
            or local_y < 0
            or local_y >= visible_draft.size.height
        ):
            return None
        return visible_draft, screen_x - visible_region.x, local_y

    def is_visible_draft_screen_position(self, screen_x: int, screen_y: int) -> bool:
        """Return whether screen coordinates target the visible draft surface.

        Args:
            screen_x: Absolute screen column from a terminal mouse/click event.
            screen_y: Absolute screen row from a terminal mouse/click event.

        Returns:
            True when the coordinates map to the visible draft row or supported
            textual-web boundary fallback rows.
        """
        return self._visible_draft_screen_hit(screen_x, screen_y) is not None

    def activate_visible_draft_screen_position(
        self, screen_x: int, screen_y: int
    ) -> bool:
        """Activate the draft surface from absolute screen coordinates.

        Args:
            screen_x: Absolute screen column from a terminal mouse/click event.
            screen_y: Absolute screen row from a terminal mouse/click event.

        Returns:
            True when the coordinates targeted a collapsed/confirm paste
            token, reset a pending prompt, or positioned the caret.
        """
        hit = self._visible_draft_screen_hit(screen_x, screen_y)
        if hit is None:
            return False
        visible_draft, local_x, local_y = hit

        self.focus()
        padding_left = getattr(getattr(visible_draft, "styles", None), "padding", None)
        padding_left = getattr(padding_left, "left", 0)
        return self._activate_draft_point(
            local_x,
            local_y,
            padding_left=padding_left,
        )

    def _activate_draft_point(
        self,
        click_x: int,
        click_y: int,
        *,
        padding_left: int = 0,
    ) -> bool:
        """Handle a draft-surface pointer activation.

        A click targeting a collapsed/confirm paste token advances the unfurl
        flow; any other in-text click positions the caret there (clearing a
        pending unfurl prompt first, matching the previous click-away reset).

        Returns:
            True when the click advanced a token, reset a pending prompt, or
            positioned the caret.
        """
        if (
            self._target_unfurl_segment_at(click_x, click_y, padding_left=padding_left)
            is not None
        ):
            return self._advance_targeted_paste_segment(
                click_x,
                click_y,
                padding_left=padding_left,
            )
        changed = bool(self._reset_pending_unfurl_state())
        display_index = self._display_index_at(
            click_x, click_y, padding_left=padding_left
        )
        if display_index is not None:
            self.position_cursor_from_display_index(display_index)
            return True
        if changed:
            self._refresh_visible_draft()
        return changed

    @on(Click, "#console-command-visible-text")
    def _handle_visible_draft_click(self, event: Click) -> None:
        """Advance a paste token or position the caret for draft clicks."""
        self.focus()
        if self.consume_suppressed_draft_click():
            event.stop()
            event.prevent_default()
            return
        widget = getattr(event, "widget", None) or getattr(event, "control", None)
        padding_left = getattr(getattr(widget, "styles", None), "padding", None)
        padding_left = getattr(padding_left, "left", 0)
        self._activate_draft_point(
            event.x,
            event.y,
            padding_left=padding_left,
        )
        event.stop()
        event.prevent_default()

    @on(MouseUp, "#console-command-visible-text")
    def _handle_visible_draft_mouse_up(self, event: MouseUp) -> None:
        """Handle terminal mouse events on the draft before Click synthesis."""
        self.focus()
        widget = getattr(event, "widget", None) or getattr(event, "control", None)
        padding_left = getattr(getattr(widget, "styles", None), "padding", None)
        padding_left = getattr(padding_left, "left", 0)
        if self._activate_draft_point(
            event.x,
            event.y,
            padding_left=padding_left,
        ):
            self.suppress_next_draft_click()
        event.stop()
        event.prevent_default()

    def on_click(self, event: Click) -> None:
        """Route row-level terminal clicks in the composer to the visible draft."""
        target = getattr(event, "widget", None) or getattr(event, "control", None)
        target_id = getattr(target, "id", None)
        if target_id == "console-command-visible-text" or isinstance(target, Button):
            return
        if self.consume_suppressed_draft_click():
            event.stop()
            event.prevent_default()
            return
        screen_x = getattr(event, "screen_x", None)
        screen_y = getattr(event, "screen_y", None)
        if screen_x is None or screen_y is None:
            return
        if not self.activate_visible_draft_screen_position(screen_x, screen_y):
            return
        event.stop()
        event.prevent_default()

    def on_mouse_up(self, event: MouseUp) -> None:
        """Route terminal mouse-up events in the composer to the visible draft."""
        target = getattr(event, "widget", None) or getattr(event, "control", None)
        target_id = getattr(target, "id", None)
        if target_id == "console-command-visible-text" or isinstance(target, Button):
            return
        screen_x = getattr(event, "screen_x", None)
        screen_y = getattr(event, "screen_y", None)
        if screen_x is None or screen_y is None:
            return
        if not self.activate_visible_draft_screen_position(screen_x, screen_y):
            return
        self.suppress_next_draft_click()
        event.stop()
        event.prevent_default()

    def sync_session_data(self, session_data: Any | None) -> None:
        """Refresh composer status copy from the active chat session contract."""
        if session_data is None:
            status = self.DEFAULT_STATUS
        else:
            title = getattr(session_data, "title", None) or "Untitled session"
            backend = getattr(session_data, "runtime_backend", None) or "local"
            assistant = (
                getattr(session_data, "assistant_id", None)
                or getattr(
                    session_data,
                    "character_name",
                    None,
                )
                or "General"
            )
            workspace = getattr(session_data, "workspace_id", None) or "global"
            status = (
                f"Active session: {title} | Backend: {backend} | "
                f"Assistant: {assistant} | Scope: {workspace}"
            )

        try:
            self.query_one("#console-composer-status", Static).update(escape(status))
        except NoMatches:
            return

    def set_pending_attachment_label(
        self,
        label: str | None,
        *,
        count: int = 0,
        total: int = 0,
    ) -> None:
        """Show or clear the composer's pending-attachment indicator.

        Args:
            label: User-facing attachment label (e.g. ``"photo.png · 184 B"``)
                to display next to the actions, or None to hide the indicator,
                the clear button, and restore the Attach button label.
            count: Number of files currently staged; drives the count-accurate
                Attach/Clear tooltips (TASK-380). ``0`` falls back to generic copy.
            total: The per-message attachment cap, for the ``count of total``
                tooltip; ``0`` omits the cap.
        """
        normalized = label.strip() if label else None
        self._pending_attachment_label = normalized
        self._sync_collapsed_presentation()
        try:
            indicator = self.query_one("#console-attachment-indicator", Static)
            clear_button = self.query_one("#console-clear-attachment", Button)
            actions = self.query_one("#console-composer-actions", Horizontal)
        except NoMatches:
            return
        # The Attach button moved into the ☰ menu, so the count-aware
        # "Attach +" relabel it used to carry moved with it -- the staged
        # count now reads off the indicator beside this row, which is where
        # a user looks for it anyway. ✕ stays: it is the only control that
        # is meaningless until something is staged, so it costs nothing at
        # rest and burying it would hide the way to undo a visible thing.
        if normalized:
            indicator.update(escape(f"📎 {normalized}"))
            indicator.styles.display = "block"
            indicator.styles.width = "auto"
            indicator.styles.max_width = 28
            clear_button.styles.display = "block"
            self._set_actions_row_width(actions, ATTACHMENT_ACTIONS_WIDTH)
            if count > 1:
                clear_button.tooltip = f"Clear all {count} attachments."
            else:
                clear_button.tooltip = "Clear the attachment."
        else:
            indicator.update("")
            indicator.styles.display = "none"
            indicator.styles.width = 0
            clear_button.styles.display = "none"
            self._set_actions_row_width(actions, BASE_ACTIONS_WIDTH)

    def set_voice_status(
        self,
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
        try:
            chip = self.query_one("#console-voice-status", Static)
        except NoMatches:
            return

        if state in ("idle", "unavailable"):
            chip.styles.display = "none"
            chip.styles.width = 0
            chip.styles.min_width = 0
            chip.update(Content(""))
            return

        # `size` is (0, 0) before the first layout; fall back to the ceiling
        # rather than computing a zero width and rendering an invisible chip.
        total_width = self.size.width or self.VOICE_CHIP_MAX_WIDTH * 2
        available = max(0, total_width - self.VOICE_CHIP_MIN_WIDTH)
        width = min(self.VOICE_CHIP_MAX_WIDTH, available)

        if state == "listening":
            head = f"● {elapsed_seconds // 60}:{elapsed_seconds % 60:02d}"
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
                label = self.VOICE_CHIP_TRANSCRIBING_LABEL
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

        chip.styles.display = "block"
        chip.styles.width = max(width, 1)
        chip.styles.min_width = 0
        chip.styles.height = 1
        chip.styles.min_height = 1
        chip.set_class(state == "error", "console-voice-status-error")
        chip.update(Content(body))

    def compose(self) -> ComposeResult:
        expanded = Horizontal(
            id="console-composer-expanded",
            classes="console-composer-presentation",
        )
        expanded.styles.display = "none" if self._collapsed else "block"
        with expanded:
            yield self._bounded_button(
                "Composer ▾",
                width=14,
                id="console-composer-collapse",
                classes="destination-action-button console-composer-toggle",
                tooltip="Collapse composer for more transcript space.",
            )
            visible_draft = Static(
                self._draft_renderable(""),
                id="console-command-visible-text",
                classes="console-command-visible-text",
            )
            visible_draft.can_focus = False
            visible_draft.styles.width = "1fr"
            visible_draft.styles.min_width = 0
            # Defense-in-depth (see `_draft_renderable`): each row `"\n"`-joined
            # into the update is already budgeted to `_draft_render_width()`
            # -- verified with 500k-trial fuzzing at every width >= 8 (the
            # floor every call site in this file enforces), zero violations
            # -- so this is a no-op for every width this composer actually
            # renders at. If a future budgeting bug lets a row overflow
            # again, `nowrap`/`clip` truncates that one row in place at paint
            # time instead of Textual rewrapping it into an extra physical
            # row -- which, inside this fixed-height 4-row Static, would
            # silently push the true last row out of view.
            visible_draft.styles.text_wrap = "nowrap"
            visible_draft.styles.text_overflow = "clip"
            yield visible_draft
            recovery = Static(
                "",
                id="console-composer-recovery",
                classes="console-composer-recovery",
            )
            recovery.styles.display = "none"
            recovery.styles.width = 0
            recovery.styles.min_width = 0
            recovery.styles.height = 0
            recovery.styles.min_height = 0
            yield recovery
            voice_status = Static(
                "",
                id="console-voice-status",
                classes="console-voice-status",
            )
            voice_status.styles.display = "none"
            voice_status.styles.width = 0
            voice_status.styles.min_width = 0
            voice_status.styles.height = 0
            voice_status.styles.min_height = 0
            yield voice_status
            attachment_indicator = Static(
                "",
                id="console-attachment-indicator",
                classes="console-attachment-indicator",
            )
            attachment_indicator.styles.display = "none"
            attachment_indicator.styles.width = 0
            attachment_indicator.styles.min_width = 0
            attachment_indicator.styles.height = 1
            yield attachment_indicator
            command_input = Input(
                value="",
                id="console-command-input",
                classes="console-command-input",
                placeholder=self.DRAFT_PLACEHOLDER,
                compact=True,
            )
            command_input.can_focus = False
            command_input.disabled = True
            command_input.styles.display = "none"
            command_input.styles.width = 0
            command_input.styles.min_width = 0
            command_input.styles.height = 1
            command_input.styles.min_height = 1
            yield command_input
            status = Static(
                self.DEFAULT_STATUS,
                id="console-composer-status",
                classes="console-composer-status console-hidden-control",
            )
            status.styles.display = "none"
            status.styles.width = 0
            status.styles.min_width = 0
            status.styles.height = 0
            status.styles.min_height = 0
            yield status
            disabled_reason = Static(
                "",
                id="console-send-disabled-reason",
                classes="console-send-disabled-reason",
            )
            disabled_reason.styles.display = "none"
            disabled_reason.styles.width = 0
            disabled_reason.styles.min_width = 0
            disabled_reason.styles.max_width = 0
            disabled_reason.styles.height = 0
            disabled_reason.styles.min_height = 0
            disabled_reason.styles.text_overflow = "ellipsis"
            disabled_reason.styles.text_wrap = "nowrap"
            yield disabled_reason
            actions = Horizontal(
                id="console-composer-actions", classes="console-composer-actions"
            )
            self._set_actions_row_width(actions, BASE_ACTIONS_WIDTH)
            actions.styles.height = 1
            actions.styles.min_height = 1
            actions.styles.max_height = 1
            with actions:
                # task-1680: the ☰ overflow menu sits BEFORE Send, per the
                # requested layout; new composer actions go behind it
                # rather than growing this width-bounded row.
                yield self._bounded_button(
                    "☰",
                    width=4,
                    id="console-composer-menu",
                    classes="destination-action-button console-composer-menu-button",
                    tooltip="More composer actions (image, caption, impersonate).",
                )
                yield self._bounded_button(
                    "Send",
                    width=8,
                    id="console-send-message",
                    classes="destination-action-button console-send-button",
                    variant="primary",
                    tooltip="Send the active Console session draft.",
                )
                stop_button = self._bounded_button(
                    "Stop",
                    width=8,
                    id="console-stop-generation",
                    classes="destination-action-button console-stop-button",
                    # Fleet-UX expert review F7 (task-1234): under parallel
                    # runs "Stop generation in the active Console session"
                    # read as ambiguous scope; the button only ever stops
                    # THIS tab's own run (behavior unchanged) -- say so.
                    tooltip="Stop this tab's run.",
                )
                stop_button.styles.display = "none"
                yield stop_button
                yield self._bounded_button(
                    "Mic",
                    width=8,
                    id="console-dictation",
                    classes="destination-action-button console-dictation-button",
                    tooltip=self.DICTATION_IDLE_TOOLTIP,
                )
                # Attach and Save Chatbook moved into the ☰ menu: this row
                # is width-bounded, so every always-present button here is
                # space the draft never gets back. What remains is Send,
                # Mic, and the two CONDITIONAL controls below (Stop while a
                # run is active, ✕ while an attachment is staged) -- those
                # cost nothing at rest and are time-critical when shown.
                clear_attachment = self._bounded_button(
                    "✕",
                    width=4,
                    id="console-clear-attachment",
                    classes=(
                        "destination-action-button console-clear-attachment-button"
                    ),
                    tooltip="Remove the pending attachment.",
                )
                clear_attachment.styles.display = "none"
                yield clear_attachment

        collapsed = Horizontal(
            id="console-composer-collapsed",
            classes="console-composer-presentation",
        )
        collapsed.styles.display = "block" if self._collapsed else "none"
        with collapsed:
            yield Static(
                self._collapsed_status_text(),
                id="console-composer-collapsed-status",
            )
            collapsed_stop = self._bounded_button(
                "Stop",
                width=8,
                id="console-collapsed-stop-generation",
                classes="destination-action-button console-stop-button",
                variant="warning",
                # Fleet-UX expert review F7 (task-1234): matches the
                # expanded Stop button's tooltip (console-stop-generation).
                tooltip="Stop this tab's run.",
            )
            collapsed_stop.styles.display = "block" if self._run_active else "none"
            yield collapsed_stop
            yield self._bounded_button(
                "Expand ▴",
                width=12,
                id="console-composer-expand",
                classes="destination-action-button console-composer-toggle",
                tooltip="Expand composer and return to the draft.",
            )
