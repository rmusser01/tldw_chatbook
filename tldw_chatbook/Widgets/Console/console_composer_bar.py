"""Console-native composer action row.

Undo/redo (TASK-1281): ordinary-sized history entries retain immutable
segment snapshots so collapsed paste tokens, labels, and generated paste
boundaries survive undo, redo, and session switching. Segment retention is
bounded by both character and segment-count ceilings. Oversized or highly
fragmented drafts keep the established flat-text fallback: text above
`UNDO_RECOLLAPSE_CHAR_THRESHOLD` is restored as one generic collapsed token
to avoid the composer's former O(n^2) repaint cost (measured up to 283s for
a 2.4 MB snapshot), while ordinary flat text remains literal.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from rich.cells import cell_len
from rich.markup import escape
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.content import Content
from textual.css.query import NoMatches
from textual.events import Click, DescendantBlur, DescendantFocus, Key, MouseUp
from textual.geometry import Region
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from ...Chat.console_display_state import build_console_disabled_reason
from ...Chat.console_glyphs import GLYPH_VOICE_RECORDING, GLYPH_VOICE_WORKING
from ...Chat.console_voice_input import (
    STATE_FINISHING,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_PREPARING,
)
from ...Chat.prompt_history import PromptHistory
from ...config import (
    DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    coerce_bool_setting,
    coerce_int_setting,
)
from ...UI.character_display_text import sanitize_character_display_label
from ...Widgets.glyph_fallback import resolve_glyph, resolve_glyph_text

_CollapseState = Literal["literal", "collapsed", "confirm", "expanded"]
_DictationState = Literal["idle", "starting", "recording", "transcribing"]
_DraftStyleRange = tuple[int, int, str]
DraftSegmentOrigin = Literal["literal", "paste", "inline_file"]
DraftSelection = tuple[int, int] | Literal["all"] | None

_VALID_COLLAPSE_STATES = frozenset({"literal", "collapsed", "confirm", "expanded"})
_VALID_DRAFT_ORIGINS = frozenset({"literal", "paste", "inline_file"})
_SNAPSHOT_FINGERPRINT_DOMAIN = b"tldw.console.composer.snapshot.v1\0"
_PROJECTION_FINGERPRINT_DOMAIN = b"tldw.console.composer.projection.v1\0"
_PLACEHOLDER_MAC_DOMAIN = b"tldw.console.composer.placeholder.v1\0"
_PLACEHOLDER_PREFIX = "[[TLDW_PROTECTED:"
_PLACEHOLDER_PATTERN = re.compile(
    r"\[\[TLDW_PROTECTED:([0-9a-f]{20}):(\d+):([0-9a-f]{24})\]\]"
)
_PLACEHOLDER_CANDIDATE_PATTERN = re.compile(r"\[\[TLDW_PROTECTED:[^\]]*\]\]")

#: Chunk boundary regex mirroring `textwrap.TextWrapper.wordsep_simple_re`
#: (the pattern used whenever `break_on_hyphens=False`, which every wrap call
#: in this module passes). `_cell_wrap_line` needs the identical chunking so
#: its greedy fill only differs from `textwrap.wrap` in how it *measures* a
#: chunk (terminal cells instead of characters), not in where it is willing
#: to break.
_DRAFT_WORD_SPLIT_RE = re.compile(r"([\t\n\x0b\x0c\r ]+)")

#: Modifier prefixes that make a key a CHORD rather than text input, even
#: when it carries a printable ``character``. Textual's terminal parser
#: renames the key (``alt+m``) but passes the bare letter through as the
#: character (``_xterm_parser.py``), so ``is_printable`` is True for every
#: ``alt+<letter>``. Without this check the printable-capture branch
#: swallowed the chord as typing -- pressing Alt+M inserted a literal "m"
#: into the draft and `ChatScreen`'s own ``Binding("alt+m", ...)`` never
#: ran (TASK-1800).
#:
#: ``ctrl+``/``super+``/``meta+`` are listed for completeness, not because
#: they leak today: their characters are C0 control bytes, which are not
#: printable, so they never reached that branch. ``alt`` is the one that
#: does. Listing all four keeps the rule "a modified key is not text" true
#: by construction rather than by accident of the control-byte encoding.
#:
#: TASK-3749 moved this here from `chat_screen` along with the
#: printable-key branch it guards: "is this keystroke text?" is a question
#: about the composer's input, and it now has no other caller.
_CHORD_MODIFIER_PREFIXES = ("alt+", "ctrl+", "super+", "meta+")


def _is_modified_chord(key: str) -> bool:
    """Return whether ``key`` names a modifier chord rather than plain text.

    Args:
        key: Textual key name, e.g. ``"m"``, ``"alt+m"``, ``"shift+alt+m"``.

    Returns:
        True when any modifier prefix appears in the name. ``shift+`` alone
        is deliberately NOT a chord -- ``shift+a`` is how a capital letter
        arrives, and treating it as a chord would break typing.
    """
    return any(prefix in key for prefix in _CHORD_MODIFIER_PREFIXES)


class ComposerTransactionValidationError(ValueError):
    """Raised when a composer improvement transaction fails closed."""


@dataclass(frozen=True)
class ComposerDraftSegmentSnapshot:
    """Deeply immutable public representation of one composer segment."""

    text: str
    origin: DraftSegmentOrigin
    collapse_state: _CollapseState
    label: str | None
    generated_boundary: bool = False
    paste_block: bool = False


@dataclass(frozen=True)
class ComposerDraftSnapshot:
    """Exact immutable composer state captured for one improvement request."""

    segments: tuple[ComposerDraftSegmentSnapshot, ...]
    cursor_index: int
    selection: DraftSelection
    edit_serial: int
    generation: int
    fingerprint: str


@dataclass(frozen=True)
class ComposerModelProjection:
    """Provider-safe text plus opaque inline-file placeholder metadata."""

    text: str
    placeholder_nonce: str
    placeholder_ids: tuple[str, ...]
    fingerprint: str


def _snapshot_fingerprint(
    *,
    segments: tuple[ComposerDraftSegmentSnapshot, ...],
    cursor_index: int,
    selection: DraftSelection,
    edit_serial: int,
    generation: int,
) -> str:
    """Return a deterministic, domain-separated digest of exact draft state."""
    payload = {
        "cursor_index": cursor_index,
        "edit_serial": edit_serial,
        "generation": generation,
        "segments": [
            {
                "collapse_state": segment.collapse_state,
                "label": segment.label,
                "origin": segment.origin,
                "generated_boundary": segment.generated_boundary,
                "paste_block": segment.paste_block,
                "text": segment.text,
            }
            for segment in segments
        ],
        "selection": list(selection) if isinstance(selection, tuple) else selection,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(_SNAPSHOT_FINGERPRINT_DOMAIN + encoded).hexdigest()


def _projection_fingerprint(
    text: str, placeholder_nonce: str, placeholder_ids: tuple[str, ...]
) -> str:
    """Return a deterministic, domain-separated digest of a model projection."""
    payload = json.dumps(
        {
            "placeholder_ids": list(placeholder_ids),
            "placeholder_nonce": placeholder_nonce,
            "text": text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(_PROJECTION_FINGERPRINT_DOMAIN + payload).hexdigest()


@dataclass
class _DraftSegment:
    """Private composer segment with canonical payload and display state."""

    text: str
    origin: DraftSegmentOrigin = "literal"
    collapse_state: _CollapseState = "literal"
    label: str | None = None
    generated_boundary: bool = False
    paste_block: bool = False


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
    """Undo/redo entry with bounded structured state and flat compatibility."""

    text: str
    cursor_index: int
    segments: tuple[ComposerDraftSegmentSnapshot, ...] | None = field(
        default=None,
        repr=False,
    )


class _HistoryStack(list[_DraftHistorySnapshot]):
    """List-compatible exported stack carrying the current structured draft."""

    def __init__(
        self,
        entries: list[_DraftHistorySnapshot],
        *,
        current_text: str | None = None,
        current_segments: tuple[ComposerDraftSegmentSnapshot, ...] | None = None,
        current_cursor: int | None = None,
    ) -> None:
        super().__init__(entries)
        self.current_text = current_text
        self.current_segments = current_segments
        self.current_cursor = current_cursor


#: Public alias for the (undo stack, redo stack) pair `export_undo_history`
#: returns and `restore_undo_history` accepts (TASK-1281 N2) -- lets a
#: caller like `ChatScreen` type its own per-session history map without
#: reaching for the private `_DraftHistorySnapshot` name or falling back to
#: `Any`.
ConsoleComposerUndoHistory = tuple[
    list[_DraftHistorySnapshot], list[_DraftHistorySnapshot]
]


@dataclass(frozen=True)
class ComposerTransactionCheckpoint:
    """Opaque rollback state for one coordinated composer mutation."""

    draft: ComposerDraftSnapshot = field(repr=False)
    undo_stack: tuple[_DraftHistorySnapshot, ...] = field(repr=False)
    redo_stack: tuple[_DraftHistorySnapshot, ...] = field(repr=False)
    improvement_undo: ComposerDraftSnapshot | None = field(repr=False)
    coalescing_active: bool = field(repr=False)


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


#: Cells of empty space between Send and Mic. The two were adjacent, so a
#: press aimed at one could land on the other; the buffer makes that
#: near-miss hit nothing.
MIC_SEND_GAP = 2

#: Fixed cell width of the composer action row at rest: Send(6) + gap(2) +
#: Dictate(11) + Stop(6). Stop is display-toggled rather than removed, so it is
#: budgeted even while hidden. The overflow-menu button ("Menu", 6) moved
#: out of this row to sit left of the draft, beside Composer ▾; Attach(10)
#: and Save(8) live behind it because this row is width-bounded and every
#: always-present button is space the draft never gets back. CN-01
#: (TASK-2154.13): Dictate needs 11 for its longest lifecycle label
#: ("Dictating", 9 visible cells + padding); Send and Stop tightened from 8
#: to 6 -- exactly their 4-cell labels + button padding, the same width the
#: "Menu" button already uses -- so the row ends one cell UNDER the old 26
#: instead of three over, and the draft keeps its 32-cell floor.
BASE_ACTIONS_WIDTH = 6 + MIC_SEND_GAP + 11 + 6

#: Width while an attachment is staged, adding the ✕ clear control (4).
ATTACHMENT_ACTIONS_WIDTH = BASE_ACTIONS_WIDTH + 4


class ConsoleComposerBar(Horizontal):
    """Expose Console-owned composer actions while reusing active chat sessions."""

    class DraftChanged(Message):
        """Posted after a Console key handled here has edited the draft.

        TASK-3749. This is the inversion that let the draft-editing key
        branches leave `ChatScreen.on_key`: the screen used to edit the
        draft through a composer method and then call back into itself
        (`_sync_console_workbench_actions_from_draft`, and for the two
        text-adding keys `_dismiss_console_guidance`), which is what
        pinned those branches to the screen. Now the composer announces
        the edit and the screen subscribes.

        It is posted **only from `handle_console_key`**, never from the
        low-level mutation helpers (`insert_text`, `delete_left`, ...).
        Those helpers have other callers -- dictation
        (`Console_Modules/dictation.py`), the session draft restore
        (`Console_Modules/session.py`), the `/prompt` insert, paste --
        each of which already runs its own, DIFFERENT follow-up (the
        dictation path also re-persists the draft to the chat store; the
        session restore deliberately does not dismiss guidance). Posting
        from the helpers would fire this message on all of them and
        silently change what those paths do.

        Attributes:
            composer: The composer whose draft changed.
            is_insertion: True when the edit ADDED text (a printable
                character, or Shift+Enter/Ctrl+J's newline), False when it
                removed text (Backspace/Ctrl+H, Delete, Ctrl+W, Ctrl+U).
                The screen dismisses first-run guidance on insertions only
                -- "the user has started composing" is not something a
                Backspace says -- which is exactly the baseline split, and
                is why this is one message with a flag rather than a bare
                "the draft changed" notification the screen would have to
                treat uniformly.
        """

        def __init__(
            self, composer: "ConsoleComposerBar", *, is_insertion: bool
        ) -> None:
            super().__init__()
            self.composer = composer
            self.is_insertion = is_insertion

        @property
        def control(self) -> "ConsoleComposerBar":
            """The composer that posted this message.

            Textual's `@on(..., selector)` filtering resolves against
            `control`, so this property is what lets a subscriber narrow to
            one composer instance.

            Returns:
                ConsoleComposerBar: The posting composer.
            """
            return self.composer

    DEFAULT_STATUS = "No active Console session."
    DRAFT_PLACEHOLDER = "Ask, command, or paste task..."
    PASTE_COLLAPSE_THRESHOLD = DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD
    PASTE_COLLAPSE_ENABLED = True
    MIN_DRAFT_ROWS = 1
    # TASK-17654: 4 -> 8 (owner call). Every consumer — growth clamps,
    # caret windowing, tail bias — derives from this constant, and the
    # stylesheet caps below move with it.
    MAX_DRAFT_ROWS = 8
    # TASK-17651: the dense-form composer has no border box or vertical
    # padding — total height IS the draft row count (1-4).
    COMPOSER_CHROME_ROWS = 0
    VOICE_CHIP_MIN_WIDTH = 24
    # Fits the 51-cell shared-executor busy copy plus the chip's cell of
    # horizontal padding on each side at ordinary Console widths.
    VOICE_CHIP_MAX_WIDTH = 53
    #: Cell cap for the inline `#console-send-disabled-reason` strip. The
    #: longest copy `build_console_disabled_reason` emits is 49 cells, so 52
    #: renders every reason whole at common widths while the `1fr` draft
    #: yields the space; narrower composers ellipsize (`text-overflow` in
    #: the stylesheet) rather than wrapping into a second row.
    SEND_REASON_MAX_WIDTH = 52
    #: Shown in the chip both for the terminal "stop and transcribe" phase
    #: (`sync_dictation_state`'s "transcribing" branch) and for a per-segment
    #: transcription in flight while still `recording`
    #: (`set_voice_segment_transcribing`) -- same word for "the model is
    #: working on your audio right now" in both places, so there is only one
    #: phrase to recognize rather than two. CN-05 (TASK-2154.13): the marker
    #: glyph is ``GLYPH_VOICE_WORKING`` -- this used to lead with "◌", which
    #: also marks temporary session tabs; one glyph, one meaning.
    VOICE_CHIP_TRANSCRIBING_LABEL = f"{GLYPH_VOICE_WORKING} Transcribing…"
    #: CN-05 (TASK-2154.13): the preparing fallback leads with the same
    #: ``GLYPH_VOICE_WORKING`` marker -- "◌" means temporary session tabs,
    #: not voice work, and the glyph must survive ASCII-fallback mode.
    VOICE_CHIP_PREPARING_LABEL = f"{GLYPH_VOICE_WORKING} Preparing microphone…"
    FALLBACK_DRAFT_WIDTH = 80
    PASTE_TOKEN_STYLE = "bold cyan"
    PASTE_CONFIRM_STYLE = "bold black on yellow"
    #: TASK-1364: ghost-text history suggestions render in the same dim style
    #: as the empty-draft placeholder -- visibly "not your text yet".
    GHOST_TEXT_STYLE = "bright_black"
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
    #: TYPED draft text into an opaque "Pasted text | N characters | Expand" token on
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
    #: Structured history is intentionally bounded independently of the text
    #: budget so a highly fragmented draft cannot multiply metadata objects.
    UNDO_STRUCTURED_SEGMENT_CAP = 512
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
        self.styles.height = self.MIN_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS
        self.styles.min_height = self.MIN_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS
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
        # Monotonic scope/version guard for changes that can retain identical
        # bytes and edit serials (notably a same-text session/load swap).
        # Improvement snapshots carry this value so an old request can never
        # become current again merely because the visible text happens to
        # match. Exact improvement Undo restores user-visible/editor state but
        # deliberately advances this generation rather than rewinding it.
        # Start from an opaque per-instance epoch so a snapshot captured from
        # a replaced composer owner cannot match a fresh composer's generation
        # even when its text/edit serial are byte-identical. The value remains
        # monotonic for this owner as scope replacements advance it.
        self._draft_generation = secrets.randbits(63)
        # Per-composer authentication key for opaque protected placeholders.
        # Tokens reveal neither inline-file metadata nor this key, so edited or
        # user-authored lookalikes cannot be accepted as protected segments.
        self._placeholder_secret = secrets.token_bytes(32)
        self._improvement_undo: ComposerDraftSnapshot | None = None
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
        self._queued_prompt_count = 0
        self._queue_paused = False
        self._send_button_width = 6
        self._send_label = "Send"
        self._send_blocked = False
        self._setup_blocked_reason = ""
        #: Last rendered `#console-send-disabled-reason` copy, tracked so a
        #: reason change can re-window the draft at its new width.
        self._send_disabled_reason = ""
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
        #: Derived presentation latch for the one overlength preparing copy.
        #: It hides only redundant row chrome; canonical draft/editor state
        #: remains untouched and is revealed by every later ordinary repaint.
        self._voice_full_width_preparing = False
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
        self._draft_selection_range: tuple[int, int] | None = None
        self._cursor_visible = True
        self._cursor_blink_timer: Any | None = None
        #: TASK-1364: shared JSONL prompt-history store, injected by the
        #: owning screen (`set_prompt_history`). Drives fish-shell-style
        #: ghost text (most-recent prefix match) and Up/Down recall. None
        #: (e.g. a bare composer in unit tests) disables both.
        self._prompt_history: PromptHistory | None = None
        #: Shell-style recall index: 0 is the live draft pseudo-entry (whose
        #: in-progress text is stashed in the history store while
        #: navigating), negatives walk backwards through stored entries.
        self._history_index: int = 0

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

    def _actions_row_width(self, *, attachment_visible: bool | None = None) -> int:
        """Return the current dynamic-label action-row budget."""

        if attachment_visible is None:
            attachment_visible = self._pending_attachment_label is not None
        return (
            BASE_ACTIONS_WIDTH
            + (self._send_button_width - 6)
            + (4 if attachment_visible else 0)
        )

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

    def _snapshot_selection(self) -> DraftSelection:
        """Return the exact selection representation owned by the composer."""
        if self._draft_selection_all:
            return "all"
        return self._draft_selection_range

    def _clear_draft_selection(self) -> None:
        """Collapse every supported selection representation."""
        self._draft_selection_all = False
        self._draft_selection_range = None

    def _mark_manual_draft_edit(self) -> None:
        """Advance the manual-edit guard and expire temporary improvement Undo."""
        self._user_edit_serial += 1
        self.invalidate_improvement_undo()

    def _advance_draft_generation(self) -> None:
        """Invalidate stale same-text transactions after a scope replacement."""
        self._draft_generation += 1
        self.invalidate_improvement_undo()

    @property
    def improvement_undo_available(self) -> bool:
        """Return whether one exact pre-improvement snapshot is available."""
        return self._improvement_undo is not None

    def invalidate_improvement_undo(self) -> None:
        """Expire the one-shot improvement Undo without touching native history."""
        self._improvement_undo = None

    def take_improvement_undo_snapshot(self) -> ComposerDraftSnapshot | None:
        """Consume and return the current exact improvement Undo snapshot."""
        snapshot = self._improvement_undo
        self._improvement_undo = None
        return snapshot

    def undo_improvement(self) -> bool:
        """Restore and consume the latest exact improvement transaction."""
        snapshot = self.take_improvement_undo_snapshot()
        if snapshot is None:
            return False
        self.restore_snapshot(snapshot)
        return True

    @staticmethod
    def _validate_snapshot_shape(snapshot: ComposerDraftSnapshot) -> None:
        """Validate a public snapshot and its deterministic fingerprint."""
        if not isinstance(snapshot, ComposerDraftSnapshot):
            raise ComposerTransactionValidationError(
                "Composer snapshot has an unsupported shape."
            )
        if type(snapshot.segments) is not tuple:
            raise ComposerTransactionValidationError(
                "Composer snapshot segments must be an immutable tuple."
            )
        for segment in snapshot.segments:
            if not isinstance(segment, ComposerDraftSegmentSnapshot):
                raise ComposerTransactionValidationError(
                    "Composer snapshot contains an unsupported segment."
                )
            if not isinstance(segment.text, str) or not segment.text:
                raise ComposerTransactionValidationError(
                    "Composer snapshot segment text must be a non-empty string."
                )
            if (
                not isinstance(segment.origin, str)
                or segment.origin not in _VALID_DRAFT_ORIGINS
            ):
                raise ComposerTransactionValidationError(
                    "Composer snapshot segment origin is invalid."
                )
            if (
                not isinstance(segment.collapse_state, str)
                or segment.collapse_state not in _VALID_COLLAPSE_STATES
            ):
                raise ComposerTransactionValidationError(
                    "Composer snapshot collapse state is invalid."
                )
            if segment.label is not None and not isinstance(segment.label, str):
                raise ComposerTransactionValidationError(
                    "Composer snapshot segment label is invalid."
                )
            if type(segment.generated_boundary) is not bool:
                raise ComposerTransactionValidationError(
                    "Composer snapshot generated-boundary state is invalid."
                )
            if type(segment.paste_block) is not bool:
                raise ComposerTransactionValidationError(
                    "Composer snapshot paste-block state is invalid."
                )
            if segment.generated_boundary and (
                segment.text != "\n"
                or segment.origin != "literal"
                or segment.collapse_state != "literal"
                or segment.label is not None
                or segment.paste_block
            ):
                raise ComposerTransactionValidationError(
                    "Composer snapshot generated boundary is invalid."
                )
            if segment.paste_block and segment.origin != "paste":
                raise ComposerTransactionValidationError(
                    "Composer snapshot paste-block ownership is invalid."
                )
        if not ConsoleComposerBar._snapshot_generated_boundaries_are_owned(
            snapshot.segments
        ):
            raise ComposerTransactionValidationError(
                "Composer snapshot generated boundary is orphaned."
            )

        draft_length = sum(len(segment.text) for segment in snapshot.segments)
        if (
            type(snapshot.cursor_index) is not int
            or snapshot.cursor_index < 0
            or snapshot.cursor_index > draft_length
        ):
            raise ComposerTransactionValidationError(
                "Composer snapshot cursor is outside the draft."
            )
        selection = snapshot.selection
        if selection == "all":
            if draft_length == 0:
                raise ComposerTransactionValidationError(
                    "An empty composer snapshot cannot have a full selection."
                )
        elif selection is not None:
            if type(selection) is not tuple or len(selection) != 2:
                raise ComposerTransactionValidationError(
                    "Composer snapshot selection is invalid."
                )
            start, end = selection
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or start > end
                or end > draft_length
            ):
                raise ComposerTransactionValidationError(
                    "Composer snapshot selection is outside the draft."
                )
        if type(snapshot.edit_serial) is not int or snapshot.edit_serial < 0:
            raise ComposerTransactionValidationError(
                "Composer snapshot edit serial is invalid."
            )
        if type(snapshot.generation) is not int or snapshot.generation < 0:
            raise ComposerTransactionValidationError(
                "Composer snapshot generation is invalid."
            )
        if not isinstance(snapshot.fingerprint, str):
            raise ComposerTransactionValidationError(
                "Composer snapshot fingerprint is invalid."
            )
        expected = _snapshot_fingerprint(
            segments=snapshot.segments,
            cursor_index=snapshot.cursor_index,
            selection=snapshot.selection,
            edit_serial=snapshot.edit_serial,
            generation=snapshot.generation,
        )
        if not hmac.compare_digest(snapshot.fingerprint, expected):
            raise ComposerTransactionValidationError(
                "Composer snapshot fingerprint does not match its state."
            )

    @staticmethod
    def _private_segments_from_snapshot(
        snapshot: ComposerDraftSnapshot,
    ) -> list[_DraftSegment]:
        """Build detached mutable segments from a validated public snapshot."""
        return [
            _DraftSegment(
                text=segment.text,
                origin=segment.origin,
                collapse_state=segment.collapse_state,
                label=segment.label,
                generated_boundary=segment.generated_boundary,
                paste_block=segment.paste_block,
            )
            for segment in snapshot.segments
        ]

    def capture_draft_snapshot(self) -> ComposerDraftSnapshot:
        """Capture the exact immutable Console draft transaction state."""
        self._ensure_editable_segments()
        segments = self._capture_segment_snapshots()
        selection = self._snapshot_selection()
        fingerprint = _snapshot_fingerprint(
            segments=segments,
            cursor_index=self._cursor_index,
            selection=selection,
            edit_serial=self._user_edit_serial,
            generation=self._draft_generation,
        )
        return ComposerDraftSnapshot(
            segments=segments,
            cursor_index=self._cursor_index,
            selection=selection,
            edit_serial=self._user_edit_serial,
            generation=self._draft_generation,
            fingerprint=fingerprint,
        )

    def _capture_segment_snapshots(
        self,
    ) -> tuple[ComposerDraftSegmentSnapshot, ...]:
        """Return detached immutable snapshots of the live segment structure."""
        return tuple(
            ComposerDraftSegmentSnapshot(
                text=segment.text,
                origin=segment.origin,
                collapse_state=segment.collapse_state,
                label=segment.label,
                generated_boundary=segment.generated_boundary,
                paste_block=segment.paste_block,
            )
            for segment in self._segments
        )

    @staticmethod
    def _snapshot_segment_is_paste_block(
        segment: ComposerDraftSegmentSnapshot,
    ) -> bool:
        """Return whether an immutable segment owns paste-block semantics."""
        return (
            segment.origin == "paste"
            and segment.paste_block
            and segment.collapse_state in {"collapsed", "confirm", "expanded"}
        )

    @classmethod
    def _snapshot_generated_boundaries_are_owned(
        cls,
        segments: tuple[ComposerDraftSegmentSnapshot, ...],
    ) -> bool:
        """Return whether every generated newline joins two paste blocks."""
        return all(
            not segment.generated_boundary
            or (
                index > 0
                and index + 1 < len(segments)
                and cls._snapshot_segment_is_paste_block(segments[index - 1])
                and cls._snapshot_segment_is_paste_block(segments[index + 1])
            )
            for index, segment in enumerate(segments)
        )

    @classmethod
    def _history_entry_is_valid(cls, entry: object) -> bool:
        """Return whether one internal/exported history entry is safe to restore."""
        if (
            not isinstance(entry, _DraftHistorySnapshot)
            or type(entry.text) is not str
            or type(entry.cursor_index) is not int
            or not 0 <= entry.cursor_index <= len(entry.text)
        ):
            return False
        if entry.segments is None:
            return True
        if (
            type(entry.segments) is not tuple
            or len(entry.segments) > cls.UNDO_STRUCTURED_SEGMENT_CAP
            or sum(len(segment.text) for segment in entry.segments) != len(entry.text)
            or "".join(segment.text for segment in entry.segments) != entry.text
        ):
            return False
        for segment in entry.segments:
            if not isinstance(segment, ComposerDraftSegmentSnapshot):
                return False
            if (
                not segment.text
                or segment.origin not in _VALID_DRAFT_ORIGINS
                or segment.collapse_state not in _VALID_COLLAPSE_STATES
                or (segment.label is not None and not isinstance(segment.label, str))
                or type(segment.generated_boundary) is not bool
                or type(segment.paste_block) is not bool
                or (segment.paste_block and segment.origin != "paste")
            ):
                return False
            if segment.generated_boundary and (
                segment.text != "\n"
                or segment.origin != "literal"
                or segment.collapse_state != "literal"
                or segment.label is not None
                or segment.paste_block
            ):
                return False
        return cls._snapshot_generated_boundaries_are_owned(entry.segments)

    def capture_transaction_checkpoint(self) -> ComposerTransactionCheckpoint:
        """Capture draft and undo state for coordinated rollback.

        Returns:
            An opaque, repr-safe checkpoint accepted by
            :meth:`rollback_transaction`.
        """
        return ComposerTransactionCheckpoint(
            draft=self.capture_draft_snapshot(),
            undo_stack=tuple(self._undo_stack),
            redo_stack=tuple(self._redo_stack),
            improvement_undo=self._improvement_undo,
            coalescing_active=self._coalescing_active,
        )

    def rollback_transaction(
        self,
        checkpoint: ComposerTransactionCheckpoint,
    ) -> None:
        """Restore a checkpoint without exposing composer-owned histories.

        Args:
            checkpoint: A prior :meth:`capture_transaction_checkpoint` result.

        Raises:
            ComposerTransactionValidationError: If the checkpoint is not a
                supported composer transaction checkpoint.
        """
        if not isinstance(checkpoint, ComposerTransactionCheckpoint):
            raise ComposerTransactionValidationError(
                "Composer transaction checkpoint has an unsupported shape."
            )
        self._validate_snapshot_shape(checkpoint.draft)
        if checkpoint.improvement_undo is not None:
            self._validate_snapshot_shape(checkpoint.improvement_undo)
        if (
            type(checkpoint.undo_stack) is not tuple
            or type(checkpoint.redo_stack) is not tuple
            or type(checkpoint.coalescing_active) is not bool
        ):
            raise ComposerTransactionValidationError(
                "Composer transaction checkpoint history is invalid."
            )
        history_entries = (*checkpoint.undo_stack, *checkpoint.redo_stack)
        if any(not self._history_entry_is_valid(entry) for entry in history_entries):
            raise ComposerTransactionValidationError(
                "Composer transaction checkpoint history is invalid."
            )

        self.restore_snapshot(checkpoint.draft)
        self._undo_stack = list(checkpoint.undo_stack)
        self._redo_stack = list(checkpoint.redo_stack)
        self._improvement_undo = checkpoint.improvement_undo
        self._coalescing_active = checkpoint.coalescing_active
        self._sync_current_action_state()

    @staticmethod
    def _validate_request_nonce(request_nonce: str) -> None:
        """Reject empty, control-bearing, whitespace, or oversized nonces."""
        if (
            not isinstance(request_nonce, str)
            or not request_nonce
            or len(request_nonce) > 128
            or not request_nonce.isprintable()
            or any(character.isspace() for character in request_nonce)
        ):
            raise ComposerTransactionValidationError(
                "Projection request nonce is invalid."
            )

    @staticmethod
    def _placeholder_nonce_id(request_nonce: str) -> str:
        """Return an opaque request identifier without exposing the raw nonce."""
        return hashlib.sha256(request_nonce.encode("utf-8")).hexdigest()[:20]

    def _placeholder_token(
        self,
        snapshot: ComposerDraftSnapshot,
        *,
        nonce_id: str,
        ordinal: int,
    ) -> str:
        """Build an authenticated opaque placeholder for one protected segment."""
        message = (
            _PLACEHOLDER_MAC_DOMAIN
            + snapshot.fingerprint.encode("ascii")
            + b"\0"
            + nonce_id.encode("ascii")
            + b"\0"
            + str(ordinal).encode("ascii")
        )
        mac = hmac.new(self._placeholder_secret, message, hashlib.sha256).hexdigest()[
            :24
        ]
        return f"{_PLACEHOLDER_PREFIX}{nonce_id}:{ordinal}:{mac}]]"

    def project_snapshot_for_model(
        self,
        snapshot: ComposerDraftSnapshot,
        *,
        request_nonce: str,
    ) -> ComposerModelProjection:
        """Project improvable text while replacing inline files with opaque tokens."""
        self._validate_snapshot_shape(snapshot)
        self._validate_request_nonce(request_nonce)
        improvable_text = "".join(
            segment.text
            for segment in snapshot.segments
            if segment.origin != "inline_file"
        )
        if _PLACEHOLDER_PREFIX in improvable_text:
            raise ComposerTransactionValidationError(
                "Improvable composer text contains reserved placeholder syntax."
            )
        if request_nonce in improvable_text:
            raise ComposerTransactionValidationError(
                "Projection request nonce collision with user-authored text."
            )

        nonce_id = self._placeholder_nonce_id(request_nonce)
        placeholder_ids: list[str] = []
        projected_parts: list[str] = []
        protected_ordinal = 0
        for segment in snapshot.segments:
            if segment.origin != "inline_file":
                projected_parts.append(segment.text)
                continue
            token = self._placeholder_token(
                snapshot,
                nonce_id=nonce_id,
                ordinal=protected_ordinal,
            )
            protected_ordinal += 1
            if token in improvable_text:
                raise ComposerTransactionValidationError(
                    "Protected placeholder collides with user-authored text."
                )
            placeholder_ids.append(token)
            projected_parts.append(token)

        text = "".join(projected_parts)
        ids = tuple(placeholder_ids)
        return ComposerModelProjection(
            text=text,
            placeholder_nonce=request_nonce,
            placeholder_ids=ids,
            fingerprint=_projection_fingerprint(text, request_nonce, ids),
        )

    def _validated_apply_parts(
        self,
        snapshot: ComposerDraftSnapshot,
        rewritten_model_text: str,
    ) -> tuple[list[_DraftSegment], str]:
        """Validate protected tokens and construct the detached apply result."""
        if not isinstance(rewritten_model_text, str):
            raise ComposerTransactionValidationError(
                "Improved composer text must be a string."
            )
        protected = [
            segment for segment in snapshot.segments if segment.origin == "inline_file"
        ]
        candidates = _PLACEHOLDER_CANDIDATE_PATTERN.findall(rewritten_model_text)
        prefix_count = rewritten_model_text.count(_PLACEHOLDER_PREFIX)
        if not protected:
            if prefix_count or candidates:
                raise ComposerTransactionValidationError(
                    "Improved text contains an unexpected protected placeholder."
                )
            return (
                [
                    _DraftSegment(
                        rewritten_model_text,
                        origin="literal",
                        collapse_state="literal",
                    )
                ]
                if rewritten_model_text
                else [],
                "".join(segment.text for segment in snapshot.segments),
            )

        if prefix_count != len(protected) or len(candidates) != len(protected):
            raise ComposerTransactionValidationError(
                "Every protected placeholder must appear exactly once."
            )
        parsed = [_PLACEHOLDER_PATTERN.fullmatch(candidate) for candidate in candidates]
        if any(match is None for match in parsed):
            raise ComposerTransactionValidationError(
                "A protected placeholder was edited or malformed."
            )
        matches = [match for match in parsed if match is not None]
        nonce_ids = {match.group(1) for match in matches}
        if len(nonce_ids) != 1:
            raise ComposerTransactionValidationError(
                "Protected placeholders do not share one request identity."
            )
        nonce_id = next(iter(nonce_ids))
        expected = tuple(
            self._placeholder_token(snapshot, nonce_id=nonce_id, ordinal=index)
            for index in range(len(protected))
        )
        if tuple(candidates) != expected:
            raise ComposerTransactionValidationError(
                "Protected placeholders were edited, duplicated, or reordered."
            )

        remaining = rewritten_model_text
        rebuilt: list[_DraftSegment] = []
        for token, protected_segment in zip(expected, protected):
            leading, separator, remaining = remaining.partition(token)
            if not separator:
                raise ComposerTransactionValidationError(
                    "A protected placeholder is missing from improved text."
                )
            if leading:
                rebuilt.append(
                    _DraftSegment(
                        leading,
                        origin="literal",
                        collapse_state="literal",
                    )
                )
            rebuilt.append(
                _DraftSegment(
                    protected_segment.text,
                    origin=protected_segment.origin,
                    collapse_state=protected_segment.collapse_state,
                    label=protected_segment.label,
                )
            )
        if remaining:
            rebuilt.append(
                _DraftSegment(
                    remaining,
                    origin="literal",
                    collapse_state="literal",
                )
            )
        source_parts: list[str] = []
        protected_index = 0
        for segment in snapshot.segments:
            if segment.origin == "inline_file":
                source_parts.append(expected[protected_index])
                protected_index += 1
            else:
                source_parts.append(segment.text)
        return rebuilt, "".join(source_parts)

    def apply_improvement(
        self,
        snapshot: ComposerDraftSnapshot,
        rewritten_model_text: str,
    ) -> ComposerDraftSnapshot | None:
        """Atomically replace improvable spans after exact stale/token checks."""
        self._validate_snapshot_shape(snapshot)
        live = self.capture_draft_snapshot()
        if (
            snapshot.edit_serial != live.edit_serial
            or snapshot.generation != live.generation
            or not hmac.compare_digest(snapshot.fingerprint, live.fingerprint)
        ):
            raise ComposerTransactionValidationError(
                "Composer snapshot is stale and cannot be applied."
            )
        rebuilt, source_projection = self._validated_apply_parts(
            snapshot, rewritten_model_text
        )
        if rewritten_model_text == source_projection:
            return None

        # All parsing and validation above used detached immutable/local values.
        # This is the single live segment swap for the complete transaction.
        self._segments = rebuilt
        self._segments_initialized = True
        self._cursor_index = len(self._canonical_draft_text())
        self._clear_draft_selection()
        self._mark_manual_draft_edit()
        self._draft_generation += 1
        self._undo_stack = []
        self._redo_stack = []
        self._coalescing_active = False
        self._improvement_undo = snapshot
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()
        return snapshot

    def replace_snapshot_as_paste(
        self,
        snapshot: ComposerDraftSnapshot,
        text: str,
    ) -> ComposerDraftSnapshot | None:
        """Replace one exact complete draft snapshot through paste semantics.

        Args:
            snapshot: Complete composer state captured before asynchronous work.
            text: Final replacement payload; blank text clears the captured draft.

        Returns:
            The captured snapshot when applied, or ``None`` for empty-to-empty.

        Raises:
            ComposerTransactionValidationError: If the snapshot is stale or the
                replacement payload is not text.
        """
        self._validate_snapshot_shape(snapshot)
        if not isinstance(text, str):
            raise ComposerTransactionValidationError(
                "Prompt replacement text must be a string."
            )
        live = self.capture_draft_snapshot()
        if (
            snapshot.edit_serial != live.edit_serial
            or snapshot.generation != live.generation
            or not hmac.compare_digest(snapshot.fingerprint, live.fingerprint)
        ):
            raise ComposerTransactionValidationError(
                "Composer snapshot is stale and cannot be applied."
            )
        if not snapshot.segments and not text:
            return None

        should_collapse = (
            self.collapse_large_pastes_enabled
            and len(text) > self.paste_collapse_threshold
        )
        rebuilt = (
            [
                _DraftSegment(
                    text,
                    origin="paste",
                    collapse_state="collapsed" if should_collapse else "literal",
                )
            ]
            if text
            else []
        )
        self._segments = rebuilt
        self._segments_initialized = True
        self._cursor_index = len(text)
        self._clear_draft_selection()
        self._mark_manual_draft_edit()
        self._draft_generation += 1
        self._undo_stack = []
        self._redo_stack = []
        self._coalescing_active = False
        self._improvement_undo = snapshot
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()
        return snapshot

    def validate_improvement(
        self,
        snapshot: ComposerDraftSnapshot,
        rewritten_model_text: str,
    ) -> None:
        """Validate a proposed improvement against live state without mutation."""
        self._validate_snapshot_shape(snapshot)
        live = self.capture_draft_snapshot()
        if (
            snapshot.edit_serial != live.edit_serial
            or snapshot.generation != live.generation
            or not hmac.compare_digest(snapshot.fingerprint, live.fingerprint)
        ):
            raise ComposerTransactionValidationError(
                "Composer snapshot is stale and cannot be applied."
            )
        self._validated_apply_parts(snapshot, rewritten_model_text)

    def restore_snapshot(self, snapshot: ComposerDraftSnapshot) -> None:
        """Atomically restore exact draft state without calling ``load_draft``."""
        self._validate_snapshot_shape(snapshot)
        rebuilt = self._private_segments_from_snapshot(snapshot)

        # A restore is a new live scope generation even though user-visible
        # bytes/state and edit serial are restored exactly. This prevents an
        # older in-flight result from becoming valid again after Undo.
        next_generation = self._draft_generation + 1
        self._segments = rebuilt
        self._segments_initialized = True
        self._cursor_index = snapshot.cursor_index
        self._draft_selection_all = snapshot.selection == "all"
        self._draft_selection_range = (
            snapshot.selection if isinstance(snapshot.selection, tuple) else None
        )
        self._user_edit_serial = snapshot.edit_serial
        self._draft_generation = next_generation
        self._undo_stack = []
        self._redo_stack = []
        self._coalescing_active = False
        self.invalidate_improvement_undo()
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

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
            token = f"Pasted text | {len(segment.text)} characters | Expand"
            if segment.origin == "paste":
                leading, trailing = ConsoleComposerBar._paste_edge_line_breaks(
                    segment.text
                )
                return leading + token + trailing
            return token
        if segment.collapse_state == "confirm":
            if segment.origin == "paste":
                leading, trailing = ConsoleComposerBar._paste_edge_line_breaks(
                    segment.text
                )
                return leading + "Expand?" + trailing
            return "Expand?"
        return segment.text

    @staticmethod
    def _paste_edge_line_breaks(text: str) -> tuple[str, str]:
        """Return non-overlapping LF/CRLF runs at a paste block's edges."""
        leading_match = re.match(r"(?:(?:\r\n)|\n)+", text)
        leading = leading_match.group(0) if leading_match else ""
        if leading == text and leading:
            first_break_length = 2 if leading.startswith("\r\n") else 1
            if len(leading) > first_break_length:
                return (
                    leading[:first_break_length],
                    leading[first_break_length:],
                )
            return "", leading
        remaining = text[len(leading) :]
        trailing_match = re.search(r"(?:(?:\r\n)|\n)+$", remaining)
        trailing = trailing_match.group(0) if trailing_match else ""
        return leading, trailing

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
            # A staged attachment is sendable payload on its own (the screen
            # passes the same OR at its own sync site) -- without it, a
            # keystroke-side resync would flip Send back to disabled and
            # flash the "type a message" reason while an image waits staged.
            has_draft=bool(self.draft_text().strip())
            or self._pending_attachment_label is not None,
            run_active=self._run_active,
            can_save_chatbook=self._can_save_chatbook,
            send_blocked=self._send_blocked,
            setup_blocked_reason=self._setup_blocked_reason,
            ephemeral=self._ephemeral,
            send_label=self._send_label,
        )

    def _sync_send_disabled_reason(self, reason: str, *, muted: bool) -> None:
        """Render or hide the persistent Send disabled-reason strip.

        FR-04 (TASK-2154.6): the blocked/empty reason must be perceivable
        WITHOUT hovering the (now genuinely disabled) Send button. The strip
        lives inside the single-row expanded composer, display-toggled with
        an ``auto`` width the ``1fr`` draft yields -- it never adds height
        and never touches the fixed actions-row budget
        (``BASE_ACTIONS_WIDTH``).

        Args:
            reason: User-facing copy from ``build_console_disabled_reason``;
                empty hides the strip.
            muted: True for the idle "type a message" guidance (rendered in
                the quiet ``-idle`` style), False for genuine blockers.
        """
        try:
            strip = self.query_one("#console-send-disabled-reason", Static)
        except NoMatches:
            return
        strip.set_class(muted, "console-send-disabled-reason-idle")
        if reason:
            strip.update(Content(reason))
            strip.styles.display = "block"
            strip.styles.width = "auto"
            strip.styles.min_width = 0
            strip.styles.max_width = self.SEND_REASON_MAX_WIDTH
            strip.styles.height = 1
            strip.styles.min_height = 1
        else:
            strip.update(Content(""))
            strip.styles.display = "none"
            strip.styles.width = 0
            strip.styles.min_width = 0
            strip.styles.max_width = 0
            strip.styles.height = 0
            strip.styles.min_height = 0
        if self._voice_full_width_preparing:
            # The exact executor-wait copy and unchanged Mic/Send budget fill
            # the row. Keep this redundant guidance cached but out of layout
            # until the next ordinary voice-status repaint restores it.
            strip.styles.display = "none"

    def sync_action_state(
        self,
        *,
        has_draft: bool,
        run_active: bool,
        can_save_chatbook: bool,
        send_blocked: bool = False,
        setup_blocked_reason: str = "",
        ephemeral: bool = False,
        send_label: str = "Send",
        wake_turn_active: bool = False,
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
            wake_turn_active: Whether the active session is busy with a
                machine-injected auto-wake turn (task-15862 AC#3) -- the
                blocked copy then names the wake, and whatever queue/setup
                copy rode ``setup_blocked_reason`` never paints as a
                provider-setup problem.
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
            actions = self.query_one("#console-composer-actions", Horizontal)
        except NoMatches:
            return

        normalized_send_label = send_label.strip() or "Send"
        self._send_label = normalized_send_label
        self._send_button_width = max(6, cell_len(normalized_send_label) + 2)
        send_button.label = normalized_send_label
        send_button.styles.width = self._send_button_width
        send_button.styles.min_width = self._send_button_width
        send_button.styles.max_width = self._send_button_width
        if not self._voice_full_width_preparing:
            self._set_actions_row_width(actions, self._actions_row_width())

        send_ready = has_draft and not send_blocked

        # TASK-2154.6 (FR-04): Send now carries a REAL disabled state instead
        # of the old CSS-classes-only subdual -- a hover tooltip was the sole
        # pre-click affordance, and an empty draft did not even get that.
        # The keyboard path does not go through this flag's event gating:
        # `on_key`'s Enter branch detects the disabled button and dispatches
        # `handle_console_send_message` directly, so the blocked-attempt
        # feedback (toast + transcript system row) survives the flag.
        send_button.disabled = not send_ready
        send_button.variant = "primary" if send_ready else "default"
        if send_blocked and wake_turn_active:
            # task-15862 AC#3: a wake turn's blocked state names itself --
            # the queue tooltip riding `setup_blocked_reason` mid-wake read
            # as a provider-setup problem.
            send_button.tooltip = (
                "A background sub-agent result is being delivered. "
                "Wait for it to finish."
            )
        elif send_blocked and setup_blocked_reason:
            send_button.tooltip = setup_blocked_reason
        elif send_blocked:
            send_button.tooltip = (
                "Wait for the active Console run to finish before sending."
            )
        elif has_draft:
            send_button.tooltip = "Send the active Console session draft."
        else:
            send_button.tooltip = "Type a message to send."
        send_button.set_class(send_ready, "console-action-primary")
        send_button.set_class(not send_ready, "console-action-subdued")
        send_button.set_class(not send_ready, "console-action-disabled")
        send_button.set_class(send_ready, "console-send-ready")
        send_button.set_class(not has_draft, "console-send-inactive")
        send_button.set_class(send_blocked, "console-send-blocked")
        self.set_class(
            send_blocked and bool(setup_blocked_reason) and not wake_turn_active,
            "console-composer-setup-blocked",
        )
        reason = build_console_disabled_reason(
            action_id="send",
            has_draft=has_draft,
            send_blocked=send_blocked,
            setup_blocked_reason=setup_blocked_reason,
            wake_turn_active=wake_turn_active,
        )
        reason_changed = reason != self._send_disabled_reason
        self._send_disabled_reason = reason
        self._sync_send_disabled_reason(reason, muted=not send_blocked)

        stop_button.disabled = False
        stop_button.variant = "warning" if run_active else "default"
        # Fleet-UX expert review F7 (task-1234): this LIVE sync overrides
        # the button's construction-time tooltip on every action-state
        # refresh, so the compose-time copy alone (see `compose()` above)
        # was never actually what a user hovering an active Stop button
        # saw -- fixed here too, matching the collapsed Stop button (which
        # has no such override).
        # TASK-2154.6 (DS-07): the idle copy ("No active run to stop in
        # this tab.") was unreachable -- the button is display-toggled off
        # while idle, so nothing could ever be hovered to read it. The
        # alternative (a permanently visible disabled-idle Stop) would
        # spend 8 cells of the fixed BASE_ACTIONS_WIDTH budget on a
        # control that is never actionable while shown, so the dead copy
        # goes instead of the budget.
        stop_button.tooltip = "Stop this tab's run." if run_active else None
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
        if reason_changed and self.draft_text().strip():
            # The reason strip's width comes out of the 1fr draft's cells,
            # so a reason (dis)appearance re-windows the draft -- but only
            # after the pending layout has actually moved the draft's
            # region. A synchronous `_refresh_visible_draft()` here would
            # still read the PRE-toggle width (same reason
            # `set_pending_attachment_label` does not re-window inline).
            self.call_after_refresh(self._refresh_visible_draft)

    def sync_dictation_state(self, state: _DictationState) -> None:
        """Refresh the microphone action for the current one-shot lifecycle.

        Args:
            state: Current one-shot dictation lifecycle state.
        """
        entering_recording = (
            state == "recording" and self._dictation_state != "recording"
        )
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
            # CN-01 (TASK-2154.13): one verb family for the whole lifecycle --
            # the feature is "Dictate" (idle), it is working ("Dictate…",
            # starting or transcribing -- the tooltip and voice chip decode
            # which phase), and it is capturing ("Dictating"). Supersedes the
            # TX-07 (TASK-2154.12) "Text…" busy word, which had replaced the
            # jargon "STT…" but still read as a third name for one feature.
            # The recording label also drops its old "●" marker: the red
            # variant and the chip's ◉ timer carry liveness, and "●" means
            # agent-running (CN-05).
            "idle": "Dictate",
            "starting": "Dictate…",
            "recording": "Dictating",
            "transcribing": "Dictate…",
        }
        tooltips = {
            "idle": self.DICTATION_IDLE_TOOLTIP,
            # A first-run model download is minutes long, so this phase needs a
            # way out. The button stays clickable here (unlike "transcribing",
            # where there is nothing left to cancel) and a press cancels.
            "starting": "Preparing the speech model — press to cancel.",
            "recording": "Stop dictating and transcribe.",
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
                message=self._voice_preparing_message
                or resolve_glyph_text(self.VOICE_CHIP_PREPARING_LABEL),
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
                STATE_FINISHING,
                message=resolve_glyph_text(self.VOICE_CHIP_TRANSCRIBING_LABEL),
            )

    def set_dictation_availability(self, *, available: bool, tooltip: str = "") -> None:
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
            chunk for chunk in _DRAFT_WORD_SPLIT_RE.split(line.expandtabs(8)) if chunk
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
        ghost_suffix: str = "",
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
                caret_cell = resolve_glyph(cls.CURSOR_GLYPH) if cursor_visible else " "
                caret_position = (
                    len(text)
                    if cursor_index is None
                    else max(0, min(cursor_index, len(text)))
                )
                # TASK-1364: ghost text is render-only -- appended after the
                # reserved caret cell (only meaningful with the caret at the
                # draft's end, its sole offer condition) and dimmed so it
                # never reads as draft content. It shares the draft's wrap
                # pass; the caret-following window keeps the caret row
                # visible and any overflow tail is simply cropped, and the
                # composer's own row-count math (`_visible_draft_row_count`)
                # never sees it, so a suggestion can never grow the bar.
                ghost = ghost_suffix if caret_position == len(text) else ""
                render_text = (
                    f"{text[:caret_position]}{caret_cell}{text[caret_position:]}{ghost}"
                )
                if style_ranges:
                    style_ranges = cls._shift_style_ranges_for_caret(
                        style_ranges,
                        caret_position,
                    )
                if ghost:
                    style_ranges = list(style_ranges or []) + [
                        (
                            caret_position + 1,
                            caret_position + 1 + len(ghost),
                            cls.GHOST_TEXT_STYLE,
                        )
                    ]
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
            placeholder = Text(
                resolve_glyph(cls.CURSOR_GLYPH) if cursor_visible else " "
            )
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
        if self._queued_prompt_count:
            state = "Paused" if self._queue_paused else "Queued"
            parts.append(f"{state} {self._queued_prompt_count}")
        return " · ".join(parts)

    def sync_prompt_queue_state(self, *, count: int, paused: bool) -> None:
        """Sync the collapsed composer's content-free queue indicator."""

        next_state = (max(0, int(count)), bool(paused))
        if next_state == (self._queued_prompt_count, self._queue_paused):
            return
        self._queued_prompt_count, self._queue_paused = next_state
        self._sync_collapsed_presentation()

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
        if segment.generated_boundary:
            insertion_index = segment_index if offset == 0 else segment_index + 1
            self._segments.insert(insertion_index, _DraftSegment(text))
            self._cursor_index += len(text)
            return
        if segment.origin == "literal" and segment.collapse_state in {
            "literal",
            "expanded",
        }:
            segment.text = segment.text[:offset] + text + segment.text[offset:]
            segment.generated_boundary = False
            self._cursor_index += len(text)
            return
        if offset == len(segment.text):
            # Caret just past a protected/non-literal segment: prepend to the
            # right literal neighbour when possible, else start a new literal.
            right_index = segment_index + 1
            if (
                right_index < len(self._segments)
                and self._segments[right_index].origin == "literal"
                and self._segments[right_index].collapse_state == "literal"
                and not self._segments[right_index].generated_boundary
            ):
                self._segments[right_index].text = (
                    text + self._segments[right_index].text
                )
                self._segments[right_index].generated_boundary = False
            else:
                self._segments.insert(right_index, _DraftSegment(text))
            self._cursor_index += len(text)
            return
        if offset == 0:
            # Caret just before a protected/non-literal segment.
            left_index = segment_index - 1
            if (
                left_index >= 0
                and self._segments[left_index].origin == "literal"
                and self._segments[left_index].collapse_state == "literal"
                and not self._segments[left_index].generated_boundary
            ):
                self._segments[left_index].text += text
                self._segments[left_index].generated_boundary = False
            else:
                self._segments.insert(segment_index, _DraftSegment(text))
            self._cursor_index += len(text)
            return

        # An expanded paste/inline file can expose interior caret positions,
        # but typing there must not silently change the original segment's
        # provenance. Split the surviving source around a new literal segment.
        left_text = segment.text[:offset]
        right_text = segment.text[offset:]
        replacement: list[_DraftSegment] = []
        if left_text:
            replacement.append(replace(segment, text=left_text, paste_block=False))
        replacement.append(_DraftSegment(text))
        if right_text:
            replacement.append(replace(segment, text=right_text, paste_block=False))
        self._segments[segment_index : segment_index + 1] = replacement
        self._cursor_index += len(text)

    def _insert_segment_at_cursor(self, segment: _DraftSegment) -> int:
        """Insert a segment at the caret and return its resulting list index."""
        if not self._segments:
            self._segments = [segment]
            self._cursor_index = len(segment.text)
            return 0
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
                    replace(
                        target,
                        text=left_text,
                        paste_block=(
                            target.paste_block
                            and not (
                                target.collapse_state == "expanded"
                                and 0 < offset < len(target.text)
                            )
                        ),
                    )
                )
            insert_index = segment_index + len(replacement)
            replacement.append(segment)
            if right_text:
                replacement.append(
                    replace(
                        target,
                        text=right_text,
                        paste_block=(
                            target.paste_block
                            and not (
                                target.collapse_state == "expanded"
                                and 0 < offset < len(target.text)
                            )
                        ),
                    )
                )
            self._segments[segment_index : segment_index + 1] = replacement
        self._cursor_index += len(segment.text)
        return insert_index

    @staticmethod
    def _is_paste_block(segment: _DraftSegment) -> bool:
        """Return whether a segment still owns adjacent-paste block semantics."""
        return (
            segment.origin == "paste"
            and segment.paste_block
            and segment.collapse_state in {"collapsed", "confirm", "expanded"}
        )

    @staticmethod
    def _starts_with_line_break(text: str) -> bool:
        return text.startswith(("\n", "\r\n"))

    @staticmethod
    def _ends_with_line_break(text: str) -> bool:
        return text.endswith("\n")

    @classmethod
    def _paste_blocks_need_generated_boundary(
        cls,
        left: _DraftSegment,
        right: _DraftSegment,
    ) -> bool:
        return (
            cls._is_paste_block(left)
            and cls._is_paste_block(right)
            and not cls._ends_with_line_break(left.text)
            and not cls._starts_with_line_break(right.text)
        )

    def _remove_segment_at(self, index: int) -> None:
        """Remove one segment while preserving the canonical caret offset."""
        start = sum(len(segment.text) for segment in self._segments[:index])
        segment = self._segments[index]
        end = start + len(segment.text)
        del self._segments[index]
        if end <= self._cursor_index:
            self._cursor_index -= len(segment.text)
        elif start < self._cursor_index:
            self._cursor_index = start

    def _prune_orphaned_generated_boundaries(self) -> None:
        """Remove generated separators no longer joining two paste blocks."""
        index = 0
        while index < len(self._segments):
            segment = self._segments[index]
            if not segment.generated_boundary:
                index += 1
                continue
            owns_boundary = (
                index > 0
                and index + 1 < len(self._segments)
                and self._is_paste_block(self._segments[index - 1])
                and self._is_paste_block(self._segments[index + 1])
            )
            if owns_boundary:
                index += 1
            else:
                self._remove_segment_at(index)

    def _ensure_new_paste_left_boundary(self, paste_index: int) -> int:
        separator_index = paste_index - 1
        has_generated = (
            separator_index >= 0 and self._segments[separator_index].generated_boundary
        )
        left_index = separator_index - 1 if has_generated else separator_index
        if left_index < 0 or not self._is_paste_block(self._segments[left_index]):
            return paste_index
        needed = self._paste_blocks_need_generated_boundary(
            self._segments[left_index], self._segments[paste_index]
        )
        if has_generated and not needed:
            self._remove_segment_at(separator_index)
            return paste_index - 1
        if not has_generated and needed:
            self._segments.insert(
                paste_index,
                _DraftSegment("\n", generated_boundary=True),
            )
            self._cursor_index += 1
            return paste_index + 1
        return paste_index

    def _ensure_new_paste_right_boundary(self, paste_index: int) -> None:
        separator_index = paste_index + 1
        has_generated = (
            separator_index < len(self._segments)
            and self._segments[separator_index].generated_boundary
        )
        right_index = separator_index + 1 if has_generated else separator_index
        if right_index >= len(self._segments) or not self._is_paste_block(
            self._segments[right_index]
        ):
            return
        needed = self._paste_blocks_need_generated_boundary(
            self._segments[paste_index], self._segments[right_index]
        )
        if has_generated and not needed:
            self._remove_segment_at(separator_index)
        elif not has_generated and needed:
            self._segments.insert(
                separator_index,
                _DraftSegment("\n", generated_boundary=True),
            )

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
                    replace(
                        segment,
                        text=kept_text,
                        generated_boundary=False,
                        paste_block=(
                            False
                            if segment.collapse_state == "expanded"
                            else segment.paste_block
                        ),
                    )
                )
        self._segments = kept_segments
        self._cursor_index = start
        self._prune_orphaned_generated_boundaries()
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
                # Ghost text only shows while focused (the caret it trails is
                # focus-only too) and `_ghost_suffix` self-gates on caret-at-
                # end/selection/live-draft, so this recomputes cleanly on
                # every blink tick and edit.
                ghost_suffix=self._ghost_suffix() if focused else "",
            )
        return self._placeholder_renderable(width=width)

    def _render_visible_draft_only(self) -> None:
        """Re-render the visible-draft Static without recomputing composer height.

        Used by the cursor blink tick (its only caller), which must stay cheap
        and must not trigger a layout recompute on every blink phase.

        TASK-21501: ``Static.update`` defaults to ``layout=True``, so this
        method used to arm a full screen layout pass ~2x/second for as long
        as the composer merely held focus -- measured at 1 ``Screen.
        _refresh_layout`` + 1 ``Compositor.reflow`` + 1 arrangement-cache
        miss per tick. ``layout=False`` is sound here because the rendered
        SIZE cannot differ between the two blink phases:

        * ``_draft_renderable`` reserves exactly one display cell at the
          caret position in BOTH phases -- the glyph while visible, a plain
          space while hidden -- and wraps it in the same pass, so the two
          phases are cell-identical by construction (see its comment). Both
          ``CURSOR_GLYPH`` and its ASCII fallback ``|`` are single-width.
        * The Static's geometry is pinned by inline styles rather than
          derived from its content: ``width: 1fr``, ``text_wrap = "nowrap"``,
          ``text_overflow = "clip"`` (set in ``compose``) and an explicit
          ``height``/``min_height``/``max_height`` written by
          ``_apply_draft_height``, which every size-changing path
          (``_refresh_visible_draft``, resize, collapse) still goes through
          with ``layout=True``. The blink tick changes no state those paths
          read.
        """
        try:
            draft = self._display_draft_text()
            width = self._draft_render_width()
            renderable = self._current_visible_draft_renderable(draft, width)
            self.query_one("#console-command-visible-text", Static).update(
                renderable, layout=False
            )
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
        if self._prompt_history is not None:
            # TASK-1364: warm the history entries so ghost text and recall
            # work on the first keystroke; the file IO itself already runs
            # off the event loop inside `load()`, and the call is idempotent.
            # Note: recall workers run exclusive=True in this same group and
            # may cancel this warm load mid-flight — safe because `get_entry`
            # re-awaits `load()` inline when `_loaded` is still False.
            self.run_worker(
                self._prompt_history.load(),
                exclusive=False,
                group="console-prompt-history",
                exit_on_error=False,
            )

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
        self._advance_draft_generation()
        self._clear_draft_selection()
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
        # A send keypress is a draft-scope barrier even when there is no text
        # to stash (for example, an attachment-only send).
        self.invalidate_improvement_undo()
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
        self._advance_draft_generation()
        self._clear_draft_selection()
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
        self._segments = [
            replace(segment) for segment in stash.segments
        ] + self._segments
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
        self._advance_draft_generation()
        self._clear_draft_selection()
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
        # TASK-1364: an accepted send also ends any prompt-history
        # navigation -- the next Up restarts from the live draft (and
        # `PromptHistory.append` clears the stashed index-0 draft on its
        # own successful write).
        self._history_index = 0

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
        self._undo_stack.append(self._make_history_snapshot())
        if len(self._undo_stack) > self.UNDO_HISTORY_DEPTH_CAP:
            del self._undo_stack[0]
        self._evict_to_char_budget(self._undo_stack)
        self._redo_stack.clear()
        self._coalescing_active = coalesce

    def _make_history_snapshot(self) -> _DraftHistorySnapshot:
        """Capture exact bounded structure, or a flat oversized fallback."""
        self._ensure_editable_segments()
        text = self._canonical_draft_text()
        segments = None
        if (
            len(text) <= self.UNDO_RECOLLAPSE_CHAR_THRESHOLD
            and len(self._segments) <= self.UNDO_STRUCTURED_SEGMENT_CAP
        ):
            segments = self._capture_segment_snapshots()
        return _DraftHistorySnapshot(
            text=text,
            cursor_index=self._cursor_index,
            segments=segments,
        )

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

        Bounded snapshots restore their exact segment structure, including
        collapsed paste identity and generated boundary ownership. For an
        oversized snapshot without structured state, TASK-1281 review NEW-2
        (fix shape corrected by review W-1/W-2) still applies: a
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

        An oversized restored token remains a generic
        "Pasted text | N characters | Expand" collapse because carrying a
        potentially unbounded segment graph would defeat the history bound.

        Collapsed tokens are atomic for the caret everywhere else in this
        widget (no other code path leaves it mid-token), so when the
        restored segment collapses, `snapshot.cursor_index` -- recorded
        against whatever the segment structure was AT SNAPSHOT time, which
        may not have been collapsed at all -- is snapped to whichever
        token edge (0 or the full text length) it was nearer to, rather
        than restored verbatim into what is now the middle of a token.
        """
        self._clear_draft_selection()
        text_length = len(snapshot.text)
        raw_cursor = max(0, min(snapshot.cursor_index, text_length))
        if snapshot.segments is not None and self._history_entry_is_valid(snapshot):
            self._segments = [
                _DraftSegment(
                    text=segment.text,
                    origin=segment.origin,
                    collapse_state=segment.collapse_state,
                    label=segment.label,
                    generated_boundary=segment.generated_boundary,
                    paste_block=segment.paste_block,
                )
                for segment in snapshot.segments
            ]
            self._cursor_index = raw_cursor
        elif not snapshot.text:
            self._segments = []
            self._cursor_index = 0
        elif text_length > self.UNDO_RECOLLAPSE_CHAR_THRESHOLD:
            self._segments = [
                _DraftSegment(
                    snapshot.text,
                    origin="literal",
                    collapse_state="collapsed",
                )
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
        self._mark_manual_draft_edit()
        current = self._make_history_snapshot()
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
        self._mark_manual_draft_edit()
        current = self._make_history_snapshot()
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
        current = self._make_history_snapshot()
        undo_entries = _HistoryStack(
            list(self._undo_stack),
            current_text=current.text,
            current_segments=current.segments,
            current_cursor=current.cursor_index,
        )
        return (undo_entries, list(self._redo_stack))

    def restore_undo_history(self, history: ConsoleComposerUndoHistory | None) -> None:
        """Replace the undo/redo stacks wholesale (TASK-1281 session scoping).

        Args:
            history: A prior `export_undo_history()` result, or None for an
                empty history (a session that has never had a recorded
                edit -- freshly created, or never visited before).
        """
        undo_entries, redo_entries = history if history is not None else ([], [])
        current_text = getattr(undo_entries, "current_text", None)
        current_segments = getattr(undo_entries, "current_segments", None)
        current_cursor = getattr(undo_entries, "current_cursor", None)
        self._undo_stack = [
            entry for entry in undo_entries if self._history_entry_is_valid(entry)
        ]
        self._redo_stack = [
            entry for entry in redo_entries if self._history_entry_is_valid(entry)
        ]
        # TASK-1281 review F6: a caller-supplied history (banked across a
        # session switch, potentially from before this composer instance's
        # own char-budget enforcement existed, or simply handed in from
        # elsewhere) is re-enforced here too rather than trusted as already
        # within budget.
        self._evict_to_char_budget(self._undo_stack)
        self._evict_to_char_budget(self._redo_stack)
        current = _DraftHistorySnapshot(
            text=current_text if type(current_text) is str else "",
            cursor_index=current_cursor if type(current_cursor) is int else -1,
            segments=current_segments if type(current_segments) is tuple else None,
        )
        if (
            current_segments is not None
            and current_text == self.draft_text()
            and self._history_entry_is_valid(current)
        ):
            self._segments = [
                _DraftSegment(
                    text=segment.text,
                    origin=segment.origin,
                    collapse_state=segment.collapse_state,
                    label=segment.label,
                    generated_boundary=segment.generated_boundary,
                    paste_block=segment.paste_block,
                )
                for segment in current.segments or ()
            ]
            self._segments_initialized = True
            self._cursor_index = current.cursor_index
            self._clear_draft_selection()
            self._sync_hidden_input()
            self._refresh_visible_draft()
            self._sync_interaction_classes()
            self._sync_current_action_state()
        self._coalescing_active = False

    def handle_console_key(self, event: Key) -> bool:
        """Consume a Console key that maps onto a composer-only operation.

        Decomposition wave 5: the composer already owned every operation
        these keys invoke -- only the key->method mapping lived on
        `ChatScreen.on_key`. The branches below are the ones that call
        *nothing but* composer methods, so they move here verbatim; the
        screen keeps `on_key` itself (Textual resolves it by name, and its
        "the Console composer is the default printable text target" policy
        is routing, not composer behaviour) along with every branch that
        reaches past the composer -- the clipboard, undo/redo's store
        persistence, send, transcript paging.

        TASK-3749 added the six draft-EDITING keys (Backspace/Ctrl+H,
        Delete, Ctrl+W, Shift+Enter/Ctrl+J, Ctrl+U and the printable
        fallthrough). Wave 5 had to leave those on the screen because each
        one called a screen method AFTER the edit; they now post
        `DraftChanged` instead and the screen reacts to that.

        This is NOT a Textual handler (no `on_`/`_on_` prefix, no
        `BINDINGS`): the composer must not start consuming keys on its own
        the moment it happens to be focused. The screen decides whether the
        keystroke belongs to the composer at all (`_should_capture_console_
        input`, the setup-modal guard, the hands-free/realtime loops, the
        slash-command popup) and only then delegates here.

        Args:
            event: The key event the screen is offering to the composer.

        Returns:
            True when the key was consumed (the event has already been
            stopped and default-prevented); False when the screen should
            keep looking -- which includes Up/Down on a boundary row where
            neither history recall nor caret movement had anything to do.
        """
        if event.key in {"ctrl+a", "super+a", "cmd+a", "meta+a"}:
            self.select_all_draft()
            event.stop()
            event.prevent_default()
            return True
        if event.key == "left":
            self.move_cursor_left()
            event.stop()
            event.prevent_default()
            return True
        if event.key == "right":
            # TASK-1364: with a ghost-text suggestion visible (caret at end,
            # live draft), Right accepts it instead of moving the caret.
            if not self.accept_ghost_text():
                self.move_cursor_right()
            event.stop()
            event.prevent_default()
            return True
        # Vertical caret movement differs from every neighbor above: those
        # always consume the key (there is always somewhere to move -- even
        # at a boundary, left/right/home/end land on a valid, if unchanged,
        # offset). `move_cursor_up`/`move_cursor_down` instead return False
        # on the first/last visual row, and the composer moves nothing at
        # all -- so the event must fall through UNCONSUMED in that case,
        # preserving whatever up/down would otherwise do on this screen
        # (nothing today; a future transcript scroll or default focus
        # behavior must not be silently swallowed by a no-op composer move).
        # TASK-1364: on exactly those boundary rows, Up/Down first offer
        # prompt-history recall (the composer gates on first/last visual row
        # of the wrapped draft); only when recall declines does ordinary
        # caret movement get its chance.
        if event.key == "up":
            if self.recall_history_previous() or self.move_cursor_up():
                event.stop()
                event.prevent_default()
                return True
        if event.key == "down":
            if self.recall_history_next() or self.move_cursor_down():
                event.stop()
                event.prevent_default()
                return True
        if event.key == "home":
            self.move_cursor_home()
            event.stop()
            event.prevent_default()
            return True
        if event.key == "end":
            self.move_cursor_end()
            event.stop()
            event.prevent_default()
            return True
        # TASK-3749: the draft-EDITING keys. These were blocked from wave 5
        # purely by their screen-side follow-up calls; they now announce the
        # edit with `DraftChanged` and the screen does the Workbench resync
        # (and, for insertions, the guidance dismissal) in its subscriber.
        # Ordering note: as a group these ran LATER in `on_key` than they do
        # here -- but every key they match is disjoint from the branches that
        # used to precede them (Ctrl+C's copy, Enter's send, PageUp/PageDown's
        # paging, the undo/redo chords), so precedence is unchanged. The
        # printable fallthrough in particular can never shadow those: their
        # characters are C0 control bytes or CR, none of which are
        # `is_printable`, and all of them are modifier chords besides.
        if event.key in {"backspace", "ctrl+h"}:
            self.delete_left()
            self._post_draft_changed(is_insertion=False)
            event.stop()
            event.prevent_default()
            return True
        if event.key == "delete":
            self.delete_right()
            self._post_draft_changed(is_insertion=False)
            event.stop()
            event.prevent_default()
            return True
        if event.key == "ctrl+w":
            self.delete_word_left()
            self._post_draft_changed(is_insertion=False)
            event.stop()
            event.prevent_default()
            return True
        # TASK-381: Shift+Enter is the natural newline chord, but terminals
        # deliver it as a plain CR (which sends), so also accept Ctrl+J -- a
        # control code that survives every terminal -- as a portable newline.
        if event.key in ("shift+enter", "ctrl+j"):
            self.insert_text("\n")
            self._post_draft_changed(is_insertion=True)
            event.stop()
            event.prevent_default()
            return True
        if event.key == "ctrl+u":
            # TASK-1281: this is the one call site that opts into undo --
            # an accidental full clear is exactly what undo exists for.
            self.clear_draft(record_history=True)
            self._post_draft_changed(is_insertion=False)
            event.stop()
            event.prevent_default()
            return True
        if (
            event.is_printable
            and event.character is not None
            and not _is_modified_chord(event.key)
        ):
            self.insert_text(event.character)
            self._post_draft_changed(is_insertion=True)
            event.stop()
            event.prevent_default()
            return True
        return False

    def _post_draft_changed(self, *, is_insertion: bool) -> None:
        """Announce that a key handled here has edited the draft (TASK-3749).

        Args:
            is_insertion: Whether the edit added text rather than removed it.
        """
        self.post_message(
            ConsoleComposerBar.DraftChanged(self, is_insertion=is_insertion)
        )

    def select_all_draft(self) -> bool:
        """Mark the full visible Console draft as selected without mutating it.

        Returns:
            True when there is draft text to select, otherwise False.
        """
        if not self.draft_text():
            self._clear_draft_selection()
            self._refresh_visible_draft()
            return False
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
        self._draft_selection_all = True
        self._draft_selection_range = None
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
        self._mark_manual_draft_edit()
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
            self._clear_draft_selection()
            self._cursor_index = 0
        self._reset_pending_unfurl_state()
        self._clamp_cursor()
        self._insert_literal_at_cursor(text)
        self._prune_orphaned_generated_boundaries()
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def insert_quote(self, text: str) -> None:
        """Insert a transcript selection as a block quote at the caret.

        Public seam for the console selection menu's "Add to chat" action
        (console selection phase 1): every non-empty line gains a ``> ``
        prefix (blank lines become a bare ``>``, as a real block quote
        renders), then the block splices in wherever the caret sits. The
        caret always exists in the segment model -- it is not focus-bound
        -- so an unfocused composer inserts at the end of the draft (the
        phase spec's fallback). Delegates to ``insert_text`` so the quote
        takes the ordinary typing path verbatim: undo entry (never
        coalesced -- a multi-character insert always opens a fresh one),
        segment lazy-init, paste-token boundary handling, and the standard
        post-edit refresh chain.

        Args:
            text: The raw selection text to quote; blank-only input is a
                no-op (there is nothing worth quoting).
        """
        if not text.strip():
            return
        quoted = "\n".join(
            f"> {line}" if line.strip() else ">" for line in text.splitlines()
        )
        self.insert_text(quoted)

    def insert_pasted_text(self, text: str) -> None:
        """Insert pasted text at the caret, collapsing only large chunks for display.

        Args:
            text: Raw text inserted through a paste event.
        """
        self._mark_manual_draft_edit()
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
            self._clear_draft_selection()
            self._cursor_index = 0
        self._reset_pending_unfurl_state()
        self._clamp_cursor()
        should_collapse = (
            self.collapse_large_pastes_enabled
            and len(text) > self.paste_collapse_threshold
        )
        if should_collapse:
            paste_index = self._insert_segment_at_cursor(
                _DraftSegment(
                    text,
                    origin="paste",
                    collapse_state="collapsed",
                    paste_block=True,
                )
            )
            paste_index = self._ensure_new_paste_left_boundary(paste_index)
            self._ensure_new_paste_right_boundary(paste_index)
            self._prune_orphaned_generated_boundaries()
        else:
            self._insert_segment_at_cursor(
                _DraftSegment(
                    text,
                    origin="paste",
                    collapse_state="literal",
                )
            )
            self._prune_orphaned_generated_boundaries()
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
        self.insert_pasted_text(text)

    def insert_file_segment(self, text: str, label: str) -> None:
        """Insert inlined file content at the caret as a labeled, display-collapsed segment.

        Args:
            text: Full file text that becomes part of the canonical draft.
            label: Display-only token shown in place of the text (e.g.
                ``"📄 notes.md · 4 KB"``).
        """
        self._mark_manual_draft_edit()
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
            self._clear_draft_selection()
            self._cursor_index = 0
        self._reset_pending_unfurl_state()
        self._clamp_cursor()
        self._insert_segment_at_cursor(
            _DraftSegment(
                text,
                origin="inline_file",
                collapse_state="collapsed",
                label=label,
            )
        )
        self._prune_orphaned_generated_boundaries()
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
        self._mark_manual_draft_edit()
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
            segment.generated_boundary = False
            if segment.collapse_state == "expanded":
                segment.paste_block = False
            self._cursor_index -= 1
            if not segment.text:
                del self._segments[segment_index]
        self._prune_orphaned_generated_boundaries()
        self._clamp_cursor()
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def delete_right(self) -> None:
        """Delete the character (or paste token) immediately right of the caret."""
        self._mark_manual_draft_edit()
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
                next_segment.generated_boundary = False
                if next_segment.collapse_state == "expanded":
                    next_segment.paste_block = False
                if not next_segment.text:
                    del self._segments[segment_index + 1]
        elif segment.collapse_state in {"collapsed", "confirm"}:
            del self._segments[segment_index]
        else:
            segment.text = segment.text[:offset] + segment.text[offset + 1 :]
            segment.generated_boundary = False
            if segment.collapse_state == "expanded":
                segment.paste_block = False
            if not segment.text:
                del self._segments[segment_index]
        self._prune_orphaned_generated_boundaries()
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
        self._mark_manual_draft_edit()
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
        self._clear_draft_selection()
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

    # -- Prompt history: ghost text and recall (TASK-1364) ------------------
    #
    # Port of the toad-inspired input UX (`Chat/prompt_history.py` +
    # the deprecated `ChatInputTextArea`) onto this composer's segment/caret
    # model. Ghost text is a render-only suffix appended after the caret in
    # `_draft_renderable` (never part of the draft); recall swaps the whole
    # draft via `load_draft`, with the live draft stashed in the history
    # store's index-0 pseudo-entry while navigating.

    def set_prompt_history(self, history: PromptHistory | None) -> None:
        """Inject the shared prompt-history store used for ghost text/recall.

        Args:
            history: The store to read suggestions and recalled entries from,
                or None to disable both (recall keys fall through to ordinary
                caret movement).
        """
        self._prompt_history = history
        self._history_index = 0

    def _ghost_suffix(self) -> str:
        """Return the ghost-text suffix for the current draft, or ``""``.

        Offered only on the live draft (recall index 0) with an empty
        selection, a non-empty draft, and the caret at the very end of the
        canonical text. Drafts starting with ``/`` never get a suggestion:
        the slash-command popup owns completion there. Typing dismisses the
        ghost implicitly -- it is recomputed from the current draft on every
        render, so a keystroke that breaks the prefix match simply yields no
        suffix.
        """
        history = self._prompt_history
        if history is None or self._history_index != 0:
            return ""
        if not self._segments_initialized:
            return ""
        if self._draft_selection_all or self._draft_selection_range is not None:
            return ""
        canonical = self._canonical_draft_text()
        if not canonical or canonical.startswith("/"):
            return ""
        if self._cursor_index != len(canonical):
            return ""
        match = history.complete(canonical)
        if match is None:
            return ""
        return match[len(canonical) :]

    def accept_ghost_text(self) -> bool:
        """Insert the visible ghost-text suffix into the draft (Right arrow).

        Returns:
            True when a suggestion was accepted (the caller consumes the
            key); False when none is visible, so Right falls back to ordinary
            caret movement.
        """
        suffix = self._ghost_suffix()
        if not suffix:
            return False
        # The ghost is only offered with the caret at the end, so this
        # inserts exactly the suggested suffix as ordinary typed text.
        self.insert_text(suffix)
        return True

    def _caret_visual_row(self) -> tuple[int, int]:
        """Return the caret's (visual row index, row count) in the full wrap.

        Uses the same full, unwindowed wrap of the display draft as
        `_move_cursor_vertically` (never the bounded 4-row painted window).
        An uninitialized composer tracks the caret at the draft tail (every
        lazy-init entry point places it there), so it reports the last row.
        """
        display_text = self._display_draft_text()
        line_slices = self._wrap_draft_line_slices(
            display_text, self._draft_render_width()
        )
        if not self._segments_initialized:
            return len(line_slices) - 1, len(line_slices)
        caret_display_index = max(
            0, min(self._cursor_display_index(), len(display_text))
        )
        return (
            self._row_index_for_canonical_offset(line_slices, caret_display_index),
            len(line_slices),
        )

    def _can_recall_history(self, direction: int) -> bool:
        """Return whether an Up/Down keypress should recall history instead of moving the caret.

        Recall is gated to the draft's first (Up/older) or last (Down/newer)
        visual row of the full wrap, with an empty selection and a loaded,
        non-empty history; anywhere else the key keeps its ordinary caret
        movement. Down only recalls while actually navigating (index < 0) --
        at the live draft there is nothing newer, so it always falls through.
        """
        history = self._prompt_history
        if history is None or history.size == 0:
            return False
        if direction > 0 and self._history_index >= 0:
            return False
        if self._draft_selection_all or self._draft_selection_range is not None:
            return False
        row, row_count = self._caret_visual_row()
        if direction < 0:
            return row == 0
        return row == row_count - 1

    def recall_history_previous(self) -> bool:
        """Consume an Up keypress as older-history recall when gated in.

        Returns:
            True when recall was triggered (the caller consumes the key);
            False when the key should fall through to `move_cursor_up`.
        """
        if not self._can_recall_history(-1):
            return False
        self.run_worker(
            self._move_history(-1),
            exclusive=True,
            group="console-prompt-history",
            exit_on_error=False,
        )
        return True

    def recall_history_next(self) -> bool:
        """Consume a Down keypress as newer-history recall when gated in.

        Returns:
            True when recall was triggered (the caller consumes the key);
            False when the key should fall through to `move_cursor_down`.
        """
        if not self._can_recall_history(1):
            return False
        self.run_worker(
            self._move_history(1),
            exclusive=True,
            group="console-prompt-history",
            exit_on_error=False,
        )
        return True

    async def _move_history(self, direction: int) -> None:
        """Move through prompt history, stashing/restoring the live draft.

        Args:
            direction: -1 for older entries (Up), +1 for newer (Down).
        """
        history = self._prompt_history
        if history is None:
            return
        if self._history_index == 0 and direction < 0:
            # Leaving the live draft -- stash the in-progress text so recall
            # never loses it (restored when navigation returns to index 0).
            history.stash_draft(self.draft_text())
        new_index = history.clamp_index(self._history_index + direction)
        if new_index == self._history_index:
            return
        try:
            entry = await history.get_entry(new_index)
        except IndexError:
            return
        self._history_index = new_index
        # A scope swap, not a recorded edit (mirrors session-switch draft
        # loads): wipes undo/redo and lands the caret at the end.
        self.load_draft(entry["input"])

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
        self._clear_draft_selection()
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
            True when at least one pasted segment is showing the `Expand?` prompt.
        """
        return any(segment.collapse_state == "confirm" for segment in self._segments)

    def has_paste_segments(self) -> bool:
        """Return whether the draft contains non-literal inserted content.

        Explicit ``origin`` distinguishes ordinary paste and inline files even
        when their display state is literal/expanded. Callers that must not
        treat inserted content as command input should gate on this provenance
        rather than display state or the presence of a label.

        Returns:
            True when at least one segment originated from paste or an inline
            file insertion.
        """
        return any(segment.origin != "literal" for segment in self._segments)

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
            title = sanitize_character_display_label(
                getattr(session_data, "title", None),
                max_characters=500,
            ) or "Untitled session"
            backend = sanitize_character_display_label(
                getattr(session_data, "runtime_backend", None),
                max_characters=100,
            ) or "local"
            raw_assistant = (
                getattr(session_data, "assistant_id", None)
                or getattr(
                    session_data,
                    "character_name",
                    None,
                )
                or "General"
            )
            assistant = sanitize_character_display_label(
                raw_assistant,
                max_characters=180,
            ) or "General"
            workspace = sanitize_character_display_label(
                getattr(session_data, "workspace_id", None),
                max_characters=180,
            ) or "global"
            status = (
                f"Active session: {title} | Backend: {backend} | "
                f"Assistant: {assistant} | Scope: {workspace}"
            )

        try:
            self.query_one("#console-composer-status", Static).update(Text(status))
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
        # The Attach button moved into the composer Menu, so the count-aware
        # "Attach +" relabel it used to carry moved with it -- the staged
        # count now reads off the indicator beside this row, which is where
        # a user looks for it anyway. ✕ stays: it is the only control that
        # is meaningless until something is staged, so it costs nothing at
        # rest and burying it would hide the way to undo a visible thing.
        if normalized:
            indicator.update(escape(resolve_glyph_text(f"📎 {normalized}")))
            indicator.styles.display = "block"
            indicator.styles.width = "auto"
            indicator.styles.max_width = 28
            clear_button.styles.display = "block"
            self._set_actions_row_width(
                actions, self._actions_row_width(attachment_visible=True)
            )
            # CN-04 (TASK-2154.13): one phrase with the compose-time tooltip
            # ("Remove the pending attachment."), not a second "Clear" verb.
            if count > 1:
                clear_button.tooltip = f"Remove all {count} pending attachments."
            else:
                clear_button.tooltip = "Remove the pending attachment."
        else:
            indicator.update("")
            indicator.styles.display = "none"
            indicator.styles.width = 0
            clear_button.styles.display = "none"
            self._set_actions_row_width(
                actions, self._actions_row_width(attachment_visible=False)
            )
        if self._voice_full_width_preparing:
            self._sync_full_width_voice_presentation(True)

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
            self._sync_full_width_voice_presentation(False)
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
            head = f"{resolve_glyph(GLYPH_VOICE_RECORDING)} {elapsed_seconds // 60}:{elapsed_seconds % 60:02d}"
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
                label = resolve_glyph_text(self.VOICE_CHIP_TRANSCRIBING_LABEL)
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
        # Production CSS resolves the 53-cell ceiling to 52 cells here. The
        # 51-cell executor-wait copy therefore gives back its trailing padding,
        # keeps the normal one-cell right margin, and temporarily hides only
        # presentation chrome; every ordinary repaint restores both padding
        # and chrome without touching draft/editor state.
        full_width_preparing = (
            state == STATE_PREPARING and cell_len(body) + 2 >= self.VOICE_CHIP_MAX_WIDTH
        )
        self._sync_full_width_voice_presentation(full_width_preparing)
        if full_width_preparing:
            chip.styles.padding = (0, 0, 0, 1)
        else:
            chip.styles.padding = None
        chip.styles.margin = None
        chip.set_class(state == "error", "console-voice-status-error")
        chip.update(Content(body))

    def _sync_full_width_voice_presentation(self, active: bool) -> None:
        """Make room for the persistent executor-wait copy without data loss."""
        self._voice_full_width_preparing = active
        try:
            controls = (
                self.query_one("#console-composer-collapse", Button),
                self.query_one("#console-composer-menu", Button),
                self.query_one("#console-command-visible-text", Static),
            )
            attachment_indicator = self.query_one(
                "#console-attachment-indicator", Static
            )
            clear_attachment = self.query_one("#console-clear-attachment", Button)
            actions = self.query_one("#console-composer-actions", Horizontal)
        except NoMatches:
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

    def compose(self) -> ComposeResult:
        """Build the expanded and collapsed composer presentations.

        The expanded row reads, left to right: the ``Composer ▾`` collapse
        toggle, the ``Menu`` overflow-menu button, the visible draft (with its
        hidden compatibility/status companions and the display-toggled Send
        disabled-reason strip), then the fixed-width action row holding
        ``Send``, the ``MIC_SEND_GAP`` buffer, ``Mic``, and the
        display-toggled ``Stop``/``✕`` controls. The collapsed presentation
        is a one-row line with ``Expand ▴``, status, and ``Stop`` (while a
        run is active). Both presentations are always mounted;
        ``set_collapsed`` display-toggles between them so editor state
        survives without remounting.

        Yields:
            Child widgets for both composer presentations.
        """
        expanded = Horizontal(
            id="console-composer-expanded",
            classes="console-composer-presentation",
        )
        expanded.styles.display = "none" if self._collapsed else "block"
        with expanded:
            # task-2154.14 (DS-01): the left cluster keeps its old 18-cell
            # footprint across the relabel: the toggle tightens 14 -> 12 and
            # the menu button grows 4 -> 6. Both only fit their labels at
            # those widths because `line_pad` is cleared below -- Textual 8's
            # default `line-pad: 1` reserves a column each side of every
            # rendered line (and the TCSS parser rejects `line-pad: 0`, so it
            # must be set inline; see the generated stylesheet's own note).
            # Without that, "Composer ▾" needs 14 and "Menu" needs 8.
            collapse_button = self._bounded_button(
                resolve_glyph_text("Composer ▾"),
                width=12,
                id="console-composer-collapse",
                classes="destination-action-button console-composer-toggle",
                tooltip="Collapse composer for more transcript space.",
            )
            collapse_button.styles.line_pad = 0
            yield collapse_button
            # The Menu overflow button sits LEFT of the draft, beside
            # Composer ▾ (superseding the task-1680 before-Send placement):
            # overflow actions live on the left button cluster, keeping the
            # row right of the draft to the Mic → gap → Send flow. task-2154.14
            # (DS-01/DS-02): the bare ☰ glyph had a tooltip-only identity and
            # under-sold the menu, so the button now says "Menu" in words and
            # the tooltip summarizes the real entries.
            menu_button = self._bounded_button(
                "Menu",
                width=6,
                id="console-composer-menu",
                classes="destination-action-button console-composer-menu-button",
                tooltip=(
                    "Composer menu: prompts, attach, save as Chatbook, "
                    "generate image or caption, impersonate."
                ),
            )
            menu_button.styles.line_pad = 0
            yield menu_button
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
            # TASK-2154.6 (FR-04): the Send disabled reason is no longer a
            # permanently-hidden compat static -- `_sync_send_disabled_reason`
            # display-toggles it (width 0 -> auto, capped at
            # SEND_REASON_MAX_WIDTH) whenever Send is blocked or the draft is
            # empty, sitting on this same single row between the 1fr draft
            # and the fixed actions row so it never adds composer height.
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
            self._set_actions_row_width(actions, self._actions_row_width())
            actions.styles.height = 1
            actions.styles.min_height = 1
            actions.styles.max_height = 1
            with actions:
                # Send hugs the draft it submits (the row is left-aligned
                # so Stop's hidden budget never parks between draft and
                # Send), then Dictate follows across the MIC_SEND_GAP buffer
                # so a press aimed at one cannot land on the other.
                send_button = self._bounded_button(
                    "Send",
                    width=6,
                    id="console-send-message",
                    classes="destination-action-button console-send-button",
                    variant="primary",
                    tooltip="Send the active Console session draft.",
                    # A fresh composer always has an empty draft, so Send
                    # starts genuinely disabled (TASK-2154.6); the first
                    # `sync_action_state` flip re-enables it once there is
                    # something to send.
                    disabled=True,
                )
                send_button.styles.line_pad = 0
                yield send_button
                mic_button = self._bounded_button(
                    "Dictate",
                    width=11,
                    id="console-dictation",
                    classes="destination-action-button console-dictation-button",
                    tooltip=self.DICTATION_IDLE_TOOLTIP,
                )
                mic_button.styles.margin = (0, 0, 0, MIC_SEND_GAP)
                yield mic_button
                stop_button = self._bounded_button(
                    "Stop",
                    width=6,
                    id="console-stop-generation",
                    classes="destination-action-button console-stop-button",
                    # Fleet-UX expert review F7 (task-1234): under parallel
                    # runs "Stop generation in the active Console session"
                    # read as ambiguous scope; the button only ever stops
                    # THIS tab's own run (behavior unchanged) -- say so.
                    tooltip="Stop this tab's run.",
                )
                stop_button.styles.line_pad = 0
                stop_button.styles.display = "none"
                yield stop_button
                # Attach and Save Chatbook moved into the composer Menu: this
                # row is width-bounded, so every always-present button here is
                # space the draft never gets back. What remains is Send,
                # Dictate, and the two CONDITIONAL controls (Stop while a run
                # is active, ✕ while an attachment is staged) -- those cost
                # nothing at rest, are time-critical when shown, and sit
                # AFTER Dictate in Stop's budgeted slot so toggling them never
                # shifts Send or Dictate.
                clear_attachment = self._bounded_button(
                    resolve_glyph("✕"),
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
            yield self._bounded_button(
                "Expand ▴",
                width=12,
                id="console-composer-expand",
                classes="destination-action-button console-composer-toggle",
                tooltip="Expand composer and return to the draft.",
            )
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
