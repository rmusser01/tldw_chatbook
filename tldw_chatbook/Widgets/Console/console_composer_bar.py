"""Console-native composer action row."""

from __future__ import annotations

from dataclasses import dataclass
import textwrap
from typing import Any, Literal

from rich.markup import escape
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.events import Click, DescendantBlur, DescendantFocus, MouseUp
from textual.geometry import Region
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from ...config import (
    DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    coerce_bool_setting,
    coerce_int_setting,
)


_CollapseState = Literal["literal", "collapsed", "confirm", "expanded"]
_DraftStyleRange = tuple[int, int, str]


@dataclass
class _DraftSegment:
    """Private composer segment with canonical payload and display state."""

    text: str
    collapse_state: _CollapseState = "literal"


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


class ConsoleComposerBar(Horizontal):
    """Expose Console-owned composer actions while reusing active chat sessions."""

    DEFAULT_STATUS = "No active Console session."
    DRAFT_PLACEHOLDER = "Ask, command, or paste task..."
    PASTE_COLLAPSE_THRESHOLD = DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD
    PASTE_COLLAPSE_ENABLED = True
    MIN_DRAFT_ROWS = 1
    MAX_DRAFT_ROWS = 4
    COMPOSER_CHROME_ROWS = 4
    FALLBACK_DRAFT_WIDTH = 80
    PASTE_TOKEN_STYLE = "bold cyan"
    PASTE_CONFIRM_STYLE = "bold black on yellow"
    CURSOR_GLYPH = "▌"  # LEFT HALF BLOCK, terminal-style caret
    CURSOR_BLINK_INTERVAL = 0.53

    def __init__(
        self,
        *,
        collapse_large_pastes: bool = True,
        paste_collapse_threshold: int = DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.can_focus = True
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
        self._run_active = False
        self._send_blocked = False
        self._setup_blocked_reason = ""
        self._can_save_chatbook = False
        self._suppress_next_draft_click = False
        self._draft_selection_all = False
        self._cursor_visible = True
        self._cursor_blink_timer: Any | None = None

    @property
    def collapse_large_pastes_enabled(self) -> bool:
        """Return whether pasted chunks over the threshold should display compactly."""
        return self.collapse_large_pastes

    @staticmethod
    def _bounded_button(label: str, *, width: int, **kwargs: Any) -> Button:
        kwargs.setdefault("compact", True)
        button = Button(label, **kwargs)
        button.styles.width = width
        button.styles.min_width = width
        button.styles.height = 1
        button.styles.min_height = 1
        return button

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

    def _display_draft_text(self) -> str:
        """Return the display-only draft text represented by composer segments."""
        if not self._segments_initialized:
            try:
                return self.query_one("#console-command-input", Input).value
            except NoMatches:
                return ""
        return "".join(self._segment_display_text(segment) for segment in self._segments)

    @staticmethod
    def _segment_display_text(segment: _DraftSegment) -> str:
        """Return display text for a single draft segment."""
        if segment.collapse_state == "collapsed":
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
            self.query_one("#console-command-input", Input).value = self._canonical_draft_text()
        except NoMatches:
            return

    def _sync_interaction_classes(self) -> None:
        """Mirror focus-within and draft presence onto stable CSS state classes."""
        has_draft = (
            any(segment.text for segment in self._segments)
            if self._segments_initialized
            else bool(self.draft_text())
        )
        self.set_class(self.has_focus_within, "console-composer-focused")
        self.set_class(has_draft, "console-composer-has-draft")

    def _sync_current_action_state(self) -> None:
        """Refresh action buttons from the current draft and cached run/save state."""
        self.sync_action_state(
            has_draft=bool(self.draft_text().strip()),
            run_active=self._run_active,
            can_save_chatbook=self._can_save_chatbook,
            send_blocked=self._send_blocked,
            setup_blocked_reason=self._setup_blocked_reason,
        )

    def sync_action_state(
        self,
        *,
        has_draft: bool,
        run_active: bool,
        can_save_chatbook: bool,
        send_blocked: bool = False,
        setup_blocked_reason: str = "",
    ) -> None:
        """Refresh composer action priority and disabled state.

        Args:
            has_draft: Whether the canonical draft has non-whitespace content.
            run_active: Whether a Console run is currently stoppable.
            can_save_chatbook: Whether a Chatbook artifact is available to save.
            send_blocked: Whether the current run state blocks new sends.
            setup_blocked_reason: Provider/model setup copy when setup blocks Send.
        """
        has_draft = bool(has_draft)
        run_active = bool(run_active)
        can_save_chatbook = bool(can_save_chatbook)
        send_blocked = bool(send_blocked)
        setup_blocked_reason = setup_blocked_reason.strip()
        setup_reason_changed = self._setup_blocked_reason != setup_blocked_reason
        self._run_active = run_active
        self._send_blocked = send_blocked
        self._setup_blocked_reason = setup_blocked_reason
        self._can_save_chatbook = can_save_chatbook

        try:
            send_button = self.query_one("#console-send-message", Button)
            stop_button = self.query_one("#console-stop-generation", Button)
            attach_button = self.query_one("#console-attach-context", Button)
            save_button = self.query_one("#console-save-chatbook", Button)
        except NoMatches:
            return

        send_ready = has_draft and not send_blocked

        send_button.disabled = False
        send_button.variant = "primary" if send_ready else "default"
        if send_blocked and setup_blocked_reason:
            send_button.tooltip = setup_blocked_reason
        elif send_blocked:
            send_button.tooltip = "Wait for the active Console run to finish before sending."
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
        stop_button.tooltip = (
            "Stop generation in the active Console session."
            if run_active
            else "No active Console run to stop."
        )
        stop_button.set_class(run_active, "console-stop-active")
        stop_button.set_class(not run_active, "console-stop-idle")
        stop_button.set_class(not run_active, "console-action-disabled")
        stop_button.styles.display = "block" if run_active else "none"

        attach_button.disabled = False
        attach_button.variant = "default"
        attach_button.tooltip = "Attach files or context through the active Console session."
        attach_button.set_class(True, "console-action-secondary")
        attach_button.set_class(False, "console-action-disabled")
        attach_button.set_class(False, "console-action-subdued")

        save_button.disabled = False
        save_button.variant = "default"
        save_button.tooltip = (
            "Open the available Chatbook artifact in Artifacts."
            if can_save_chatbook
            else "No Chatbook artifact is available to save yet."
        )
        save_button.set_class(True, "console-action-secondary")
        save_button.set_class(True, "console-save-chatbook-secondary")
        save_button.set_class(can_save_chatbook, "console-save-chatbook-ready")
        save_button.set_class(not can_save_chatbook, "console-action-subdued")
        save_button.set_class(not can_save_chatbook, "console-action-disabled")

        if setup_reason_changed and not self.draft_text().strip():
            self._refresh_visible_draft()

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
            wrapped_lines.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                    drop_whitespace=False,
                    replace_whitespace=False,
                )
                or [""]
            )
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
            wrapped_segments = (
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                    drop_whitespace=False,
                    replace_whitespace=False,
                )
                or [""]
            )
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
        return [line_slice.text for line_slice in cls._visible_draft_line_slices(text, width)]

    @classmethod
    def _visible_draft_line_slices(cls, text: str, width: int) -> list[_DraftLineSlice]:
        """Return bounded wrapped draft rows with source-offset mapping."""
        line_slices = cls._wrap_draft_line_slices(text, width)
        if len(line_slices) <= cls.MAX_DRAFT_ROWS:
            return line_slices

        visible_slices = line_slices[-cls.MAX_DRAFT_ROWS :]
        first_slice = visible_slices[0]
        first_line_stripped = first_slice.text.lstrip()
        if first_line_stripped:
            trimmed_columns = len(first_slice.text) - len(first_line_stripped)
            visible_slices[0] = _DraftLineSlice(
                f"... {first_line_stripped}",
                first_slice.start + trimmed_columns,
                first_slice.end,
                synthetic_prefix_columns=4,
            )
        else:
            visible_slices[0] = _DraftLineSlice(
                "...",
                first_slice.end,
                first_slice.end,
                synthetic_prefix_columns=3,
            )
        return visible_slices

    @classmethod
    def _draft_renderable(
        cls,
        text: str,
        *,
        width: int = FALLBACK_DRAFT_WIDTH,
        style_ranges: list[_DraftStyleRange] | None = None,
        focused: bool = False,
        cursor_visible: bool = True,
    ) -> Text:
        if text:
            # While focused, exactly one trailing display cell is always
            # reserved after the wrapped draft -- the caret glyph during the
            # visible blink phase, an ordinary space during the hidden phase
            # -- and it is wrapped in the *same* pass as the draft itself
            # (rather than appended afterward). That keeps the two blink
            # phases layout-identical: whichever character reserves the row
            # is decided by wrap width alone, never by which literal
            # character it is, so a blink tick can never change how many
            # visual rows the draft occupies (which previously could clip or
            # jitter the composer when the last wrapped line landed exactly
            # at the wrap width). The glyph is left unstyled: the block
            # character is prominent enough on its own, and leaving it
            # unstyled keeps it from being mistaken for a stateful paste
            # token.
            trailing_cell = (cls.CURSOR_GLYPH if cursor_visible else " ") if focused else ""
            line_slices = cls._visible_draft_line_slices(f"{text}{trailing_cell}", width)
            rendered = Text("\n".join(line.text for line in line_slices))
            if style_ranges:
                output_offset = 0
                for line_index, line_slice in enumerate(line_slices):
                    source_to_output_offset = (
                        output_offset + line_slice.synthetic_prefix_columns - line_slice.start
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
            min(cls.MAX_DRAFT_ROWS, len(cls._wrap_draft_line_slices(measured_text, width))),
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
        except NoMatches:
            pass
        self.styles.height = composer_height
        self.styles.min_height = self.MIN_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS
        self.styles.max_height = self.MAX_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS
        self.refresh(layout=True)

    def _append_literal_segment(self, text: str) -> None:
        """Append literal text while coalescing adjacent literal segments."""
        if self._segments and self._segments[-1].collapse_state == "literal":
            self._segments[-1].text += text
        else:
            self._segments.append(_DraftSegment(text))

    def _current_visible_draft_renderable(self, draft: str, width: int) -> Text:
        """Build the Text renderable for the current draft/placeholder state."""
        if draft:
            return self._draft_renderable(
                draft,
                width=width,
                style_ranges=self._display_draft_style_ranges(),
                focused=self.has_focus_within,
                cursor_visible=getattr(self, "_cursor_visible", True),
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
        if self.has_focus_within:
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

        Args:
            text: Draft payload to show and send literally.
        """
        self._draft_selection_all = False
        self._segments = [_DraftSegment(text)] if text else []
        self._segments_initialized = True
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def clear_draft(self) -> None:
        """Clear the native Console draft without falling back to stale input."""
        self._draft_selection_all = False
        self._segments = []
        self._segments_initialized = True
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

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
        self._refresh_visible_draft()
        return True

    def has_full_draft_selection(self) -> bool:
        """Return whether the composer currently has a full-draft selection.

        Returns:
            True when the visible draft exists and is fully selected.
        """
        return self._draft_selection_all and bool(self.draft_text())

    def insert_text(self, text: str) -> None:
        """Append user-entered text to the Console draft as literal text.

        Args:
            text: Typed text to append without paste-collapse transformation.
        """
        if not text:
            self._sync_interaction_classes()
            self._sync_current_action_state()
            return
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
        if self._draft_selection_all:
            self._segments = []
            self._draft_selection_all = False
        self._reset_pending_unfurl_state()
        self._append_literal_segment(text)
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def insert_pasted_text(self, text: str) -> None:
        """Append pasted text, collapsing only large inserted chunks for display.

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
        if self._draft_selection_all:
            self._segments = []
            self._draft_selection_all = False
        self._reset_pending_unfurl_state()
        should_collapse = (
            self.collapse_large_pastes_enabled
            and len(text) > self.paste_collapse_threshold
        )
        if should_collapse:
            self._segments.append(_DraftSegment(text, collapse_state="collapsed"))
        else:
            self._append_literal_segment(text)
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

    def delete_left(self) -> None:
        """Delete the last draft character for simple terminal-style editing."""
        if self._draft_selection_all:
            self.clear_draft()
            return
        if not self._segments_initialized:
            self.load_draft(self.draft_text()[:-1])
            return
        if not self._segments:
            self._sync_interaction_classes()
            self._sync_current_action_state()
            return

        last_segment = self._segments[-1]
        if last_segment.collapse_state in {"collapsed", "confirm"}:
            self._segments.pop()
            self._sync_hidden_input()
            self._refresh_visible_draft()
            self._sync_interaction_classes()
            self._sync_current_action_state()
            return

        last_segment.text = last_segment.text[:-1]
        if not last_segment.text:
            self._segments.pop()
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()
        self._sync_current_action_state()

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

    def _display_index_at(self, click_x: int, click_y: int, *, padding_left: int = 0) -> int | None:
        """Map visible-draft coordinates to an unwrapped display-string offset."""
        click_x = max(0, click_x - padding_left)
        click_y = max(0, click_y)
        display_text = self._display_draft_text()
        visible_slices = self._visible_draft_line_slices(
            display_text,
            self._draft_render_width(),
        )
        if click_y >= len(visible_slices):
            return None
        clicked_slice = visible_slices[click_y]
        if click_x >= len(clicked_slice.text):
            return None
        if clicked_slice.synthetic_prefix_columns:
            if click_x < clicked_slice.synthetic_prefix_columns:
                return None
            return clicked_slice.start + click_x - clicked_slice.synthetic_prefix_columns
        return clicked_slice.start + click_x

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
        display_index = self._display_index_at(click_x, click_y, padding_left=padding_left)
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

    def activate_visible_draft_screen_position(self, screen_x: int, screen_y: int) -> bool:
        """Advance a paste token from absolute screen coordinates in the draft row.

        Args:
            screen_x: Absolute screen column from a terminal mouse/click event.
            screen_y: Absolute screen row from a terminal mouse/click event.

        Returns:
            True when the coordinates targeted a collapsed/confirm paste token.
        """
        hit = self._visible_draft_screen_hit(screen_x, screen_y)
        if hit is None:
            return False
        visible_draft, local_x, local_y = hit

        self.focus()
        padding_left = getattr(getattr(visible_draft, "styles", None), "padding", None)
        padding_left = getattr(padding_left, "left", 0)
        return self._advance_targeted_paste_segment(
            local_x,
            local_y,
            padding_left=padding_left,
        )

    @on(Click, "#console-command-visible-text")
    def _handle_visible_draft_click(self, event: Click) -> None:
        """Advance the simple two-step unfurl flow for collapsed paste segments."""
        self.focus()
        if self.consume_suppressed_draft_click():
            event.stop()
            event.prevent_default()
            return
        widget = getattr(event, "widget", None) or getattr(event, "control", None)
        padding_left = getattr(getattr(widget, "styles", None), "padding", None)
        padding_left = getattr(padding_left, "left", 0)
        if not self._advance_targeted_paste_segment(
            event.x,
            event.y,
            padding_left=padding_left,
        ):
            event.stop()
            event.prevent_default()
            return
        event.stop()
        event.prevent_default()

    @on(MouseUp, "#console-command-visible-text")
    def _handle_visible_draft_mouse_up(self, event: MouseUp) -> None:
        """Advance paste tokens for terminal mouse events before Click synthesis."""
        self.focus()
        widget = getattr(event, "widget", None) or getattr(event, "control", None)
        padding_left = getattr(getattr(widget, "styles", None), "padding", None)
        padding_left = getattr(padding_left, "left", 0)
        if not self._advance_targeted_paste_segment(
            event.x,
            event.y,
            padding_left=padding_left,
        ):
            event.stop()
            event.prevent_default()
            return
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
            assistant = getattr(session_data, "assistant_id", None) or getattr(
                session_data,
                "character_name",
                None,
            ) or "General"
            workspace = getattr(session_data, "workspace_id", None) or "global"
            status = (
                f"Active session: {title} | Backend: {backend} | "
                f"Assistant: {assistant} | Scope: {workspace}"
            )

        try:
            self.query_one("#console-composer-status", Static).update(escape(status))
        except NoMatches:
            return

    def compose(self) -> ComposeResult:
        title = Static("Composer:", id="console-composer-title", classes="destination-section")
        title.styles.width = 10
        title.styles.min_width = 10
        yield title
        visible_draft = Static(
            self._draft_renderable(""),
            id="console-command-visible-text",
            classes="console-command-visible-text",
        )
        visible_draft.can_focus = False
        visible_draft.styles.width = "1fr"
        visible_draft.styles.min_width = 0
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
        actions = Horizontal(id="console-composer-actions", classes="console-composer-actions")
        actions.styles.width = 37
        actions.styles.min_width = 37
        actions.styles.max_width = 37
        actions.styles.height = 1
        actions.styles.min_height = 1
        actions.styles.max_height = 1
        with actions:
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
                tooltip="Stop generation in the active Console session.",
            )
            stop_button.styles.display = "none"
            yield stop_button
            yield self._bounded_button(
                "Attach",
                width=10,
                id="console-attach-context",
                classes="destination-action-button console-attach-button",
                tooltip="Attach files or context through the active Console session.",
            )
            yield self._bounded_button(
                "Save",
                width=8,
                id="console-save-chatbook",
                classes="destination-action-button console-save-chatbook-button",
                tooltip="Open the available Chatbook artifact in Artifacts.",
            )
