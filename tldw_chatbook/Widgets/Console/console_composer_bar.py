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
from textual.events import Click, DescendantBlur, DescendantFocus
from textual.widgets import Button, Input, Static

from ...config import coerce_bool_setting


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
    PASTE_COLLAPSE_THRESHOLD = 50
    PASTE_COLLAPSE_ENABLED = True
    MIN_DRAFT_ROWS = 1
    MAX_DRAFT_ROWS = 4
    COMPOSER_CHROME_ROWS = 4
    FALLBACK_DRAFT_WIDTH = 80
    PASTE_TOKEN_STYLE = "bold cyan"
    PASTE_CONFIRM_STYLE = "bold black on yellow"

    def __init__(self, *, collapse_large_pastes: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.can_focus = True
        self.styles.height = 5
        self.styles.min_height = 5
        self.styles.max_height = self.MAX_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS
        self.collapse_large_pastes = coerce_bool_setting(collapse_large_pastes, True)
        self._segments: list[_DraftSegment] = []
        self._segments_initialized = False

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
        """Return the canonical native Console draft payload."""
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
    ) -> Text:
        if text:
            line_slices = cls._visible_draft_line_slices(text, width)
            rendered = Text("\n".join(line.text for line in line_slices))
            if not style_ranges:
                return rendered

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
        return Text(cls.DRAFT_PLACEHOLDER, style="bright_black")

    @classmethod
    def _visible_draft_row_count(cls, text: str, width: int) -> int:
        if not text:
            return cls.MIN_DRAFT_ROWS
        return max(
            cls.MIN_DRAFT_ROWS,
            min(cls.MAX_DRAFT_ROWS, len(cls._wrap_draft_line_slices(text, width))),
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

    def _refresh_visible_draft(self) -> None:
        try:
            draft = self._display_draft_text()
            width = self._draft_render_width()
            row_count = self._visible_draft_row_count(draft, width)
            self.query_one("#console-command-visible-text", Static).update(
                self._draft_renderable(
                    draft,
                    width=width,
                    style_ranges=self._display_draft_style_ranges(),
                )
            )
            self._apply_draft_height(row_count)
        except NoMatches:
            return

    def on_mount(self) -> None:
        self._refresh_visible_draft()
        self._sync_interaction_classes()

    def on_resize(self, event: Any) -> None:
        self._refresh_visible_draft()

    def on_focus(self) -> None:
        self._sync_interaction_classes()

    def on_blur(self) -> None:
        self._sync_interaction_classes()

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        self._sync_interaction_classes()

    def on_descendant_blur(self, event: DescendantBlur) -> None:
        self._sync_interaction_classes()

    def load_draft(self, text: str) -> None:
        """Replace the native Console draft with literal text."""
        self._segments = [_DraftSegment(text)] if text else []
        self._segments_initialized = True
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()

    def clear_draft(self) -> None:
        """Clear the native Console draft without falling back to stale input."""
        self._segments = []
        self._segments_initialized = True
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()

    def insert_text(self, text: str) -> None:
        """Append user-entered text to the Console draft as literal text."""
        if not text:
            self._sync_interaction_classes()
            return
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
        self._reset_pending_unfurl_state()
        self._append_literal_segment(text)
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()

    def insert_pasted_text(self, text: str) -> None:
        """Append pasted text, collapsing only large inserted chunks for display."""
        if not text:
            self._sync_interaction_classes()
            return
        if not self._segments_initialized:
            existing = self.draft_text()
            self._segments = [_DraftSegment(existing)] if existing else []
            self._segments_initialized = True
        self._reset_pending_unfurl_state()
        should_collapse = (
            self.collapse_large_pastes_enabled
            and len(text) > self.PASTE_COLLAPSE_THRESHOLD
        )
        if should_collapse:
            self._segments.append(_DraftSegment(text, collapse_state="collapsed"))
        else:
            self._append_literal_segment(text)
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()

    def delete_left(self) -> None:
        """Delete the last draft character for simple terminal-style editing."""
        if not self._segments_initialized:
            self.load_draft(self.draft_text()[:-1])
            return
        if not self._segments:
            self._sync_interaction_classes()
            return

        last_segment = self._segments[-1]
        if last_segment.collapse_state in {"collapsed", "confirm"}:
            self._segments.pop()
            self._sync_hidden_input()
            self._refresh_visible_draft()
            self._sync_interaction_classes()
            return

        last_segment.text = last_segment.text[:-1]
        if not last_segment.text:
            self._segments.pop()
        self._sync_hidden_input()
        self._refresh_visible_draft()
        self._sync_interaction_classes()

    def _reset_pending_unfurl_state(self) -> bool:
        """Reset pending paste unfurl confirmations without refreshing display."""
        changed = False
        for segment in self._segments:
            if segment.collapse_state == "confirm":
                segment.collapse_state = "collapsed"
                changed = True
        return changed

    def reset_pending_unfurl(self) -> bool:
        """Reset any pending paste unfurl confirmations back to collapsed tokens."""
        changed = self._reset_pending_unfurl_state()
        if changed:
            self._refresh_visible_draft()
        return changed

    def _click_display_index(self, event: Click) -> int | None:
        """Map a visible-draft click to an unwrapped display-string offset."""
        widget = getattr(event, "widget", None) or getattr(event, "control", None)
        padding_left = getattr(getattr(widget, "styles", None), "padding", None)
        padding_left = getattr(padding_left, "left", 0)
        click_x = max(0, event.x - padding_left)
        click_y = max(0, event.y)
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

    def _target_unfurl_segment(self, event: Click) -> _DraftSegment | None:
        """Return the collapsed paste segment targeted by the click position."""
        display_index = self._click_display_index(event)
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

    @on(Click, "#console-command-visible-text")
    def _handle_visible_draft_click(self, event: Click) -> None:
        """Advance the simple two-step unfurl flow for collapsed paste segments."""
        segment = self._target_unfurl_segment(event)
        if segment is None:
            return
        if segment.collapse_state == "collapsed":
            segment.collapse_state = "confirm"
        elif segment.collapse_state == "confirm":
            segment.collapse_state = "expanded"
        self._refresh_visible_draft()
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
        visible_draft.can_focus = True
        yield visible_draft
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
        yield self._bounded_button(
            "Send",
            width=8,
            id="console-send-message",
            classes="destination-action-button console-send-button",
            variant="primary",
            tooltip="Send the active Console session draft.",
        )
        yield self._bounded_button(
            "Stop",
            width=8,
            id="console-stop-generation",
            classes="destination-action-button console-stop-button",
            tooltip="Stop generation in the active Console session.",
        )
        yield self._bounded_button(
            "Attach",
            width=10,
            id="console-attach-context",
            classes="destination-action-button console-attach-button",
            tooltip="Attach files or context through the active Console session.",
        )
        yield self._bounded_button(
            "Save Chatbook",
            width=22,
            id="console-save-chatbook",
            classes="destination-action-button console-save-chatbook-button",
            tooltip="Compatibility adapter: save Chatbook export is still owned by Artifacts/Chatbooks.",
        )
