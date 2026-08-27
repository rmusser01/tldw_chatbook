"""Native Console transcript widget."""

from __future__ import annotations

import asyncio
import difflib
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from time import monotonic
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Mapping
from weakref import WeakSet

from loguru import logger
from PIL import Image as PILImage
from rich.text import Text
from rich_pixels import Pixels
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content, Span
from textual.css.query import NoMatches, QueryError
from textual.dom import NoScreen
from textual.events import Click, Key, MouseDown, MouseMove, MouseUp
from textual.geometry import Region
from textual.message import Message
from textual.message_pump import NoActiveAppError
from textual.style import Style
from textual.widget import Widget
from textual.visual import VisualType
from textual.widgets import Button, Markdown, Static
from textual_diff_view import DiffView

from tldw_chatbook.Chat.console_chat_models import (
    PROPRIETARY_THINKING_NOTICE,
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleCitationNoticeCode,
    ConsoleCitationPhase,
    ConsoleMessageRole,
    ConsoleThinkingActivityRef,
    FEEDBACK_ACTIVE_RUN_STATUSES,
)
from tldw_chatbook.Chat.assistant_generation_state import (
    render_exported_assistant_content,
)
from tldw_chatbook.Chat.console_turn_grouping import (
    ConsoleAssistantTurn,
    ConsoleTranscriptUnit,
    group_console_transcript_messages,
    ordered_assistant_activities,
    project_thinking_activities,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
)
from tldw_chatbook.Chat.console_image_view import (
    PIXELS_MAX_COLS,
    PIXELS_MAX_LINES,
    ConsoleImageRowSpec,
    fit_image_cell_size,
)
from tldw_chatbook.Chat.console_message_actions import (
    ConsoleHeaderSpeechPresentation,
    ConsoleMessageAction,
    ConsoleMessageActionService,
    ConsoleSpeechPresentationState,
    action_row_guide,
    resolve_console_header_speech,
)
from tldw_chatbook.Chat.console_chat_fork import ConsoleForkEligibility
from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_QUIET_EMPTY_COPY,
    ConsoleSetupCardState,
)
from tldw_chatbook.Chat.console_roleplay_identity import (
    ConsoleMessagePresentation,
    ConsolePresentationContext,
    ConsoleTranscriptStyle,
    resolve_console_message_presentation,
)
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_generation_card import (
    ConsoleGenerationCard,
    ConsoleGenerationCardSpec,
    generation_card_signature,
)
from tldw_chatbook.Widgets.Console.console_assistant_turn import (
    ConsoleActivityActivated,
    ConsoleActivityDisclosure,
    ConsoleAssistantTurnWidget,
)
from tldw_chatbook.Widgets.Console.console_selection import (
    SelectionManager,
    TextSelection,
    cap_quote,
    line_end_offset,
    line_start_offset,
    next_line_offset,
    offset_for_cell,
    prev_line_offset,
    word_back_offset,
    word_forward_offset,
)
from tldw_chatbook.Widgets.Console.console_selection_menu import (
    ConsoleSelectionFeedbackRequested,
    ConsoleSelectionNoteRequested,
    ConsoleSelectionMenu,
    ConsoleSelectionQuoteRequested,
    ConsoleSideChatRequested,
    selection_menus_on_screen,
)
from tldw_chatbook.Widgets.Console.console_message_more_menu import (
    ConsoleMessageMoreMenu,
    message_more_menus_on_screen,
)
from tldw_chatbook.Widgets.Console.console_turn_file_card import ConsoleTurnFileCard
from tldw_chatbook.Widgets.Console.console_video_card import (
    ConsoleVideoCard,
    ConsoleVideoCardSpec,
    video_card_signature,
)
from tldw_chatbook.Widgets.diff_widgets import make_diff
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard

if TYPE_CHECKING:  # pragma: no cover - typing only
    from textual.screen import Screen


# TASK-17658: rule separators paint via the stylesheet's hatch fill
# (.console-transcript-rule), which spans any terminal width — the old
# fixed 200-dash string stopped short on very wide terminals.
CONSOLE_TRANSCRIPT_RULE = ""
CONSOLE_GENERATING_PLACEHOLDER = "Generating…"
#: Console selection phase 3: run statuses during which review feedback
#: (Request changes / LGTM) can be queued behind the active run via the
#: prompt-queue seam. Anything else (or a screen without the run-status
#: seam at all) leaves those two menu actions gated; Comment never gates.
#: Derived (as raw strings, matching the seam's wire format) from the
#: canonical ``FEEDBACK_ACTIVE_RUN_STATUSES`` next to ``ConsoleRunStatus``.
_SELECTION_FEEDBACK_ACTIVE_RUN_STATUSES = frozenset(
    {status.value for status in FEEDBACK_ACTIVE_RUN_STATUSES}
)
#: TASK-1365: virtual-height watermarks (terminal rows) for transcript pruning.
#: 20000 rows is several hundred long messages; rows are cheap to measure but
#: expensive to keep laid out. Mirrored from the legacy chat log pruning
#: (``UI/Chat_Modules/chat_log_pruning.py`` on feat/toad-ui-improvements).
DEFAULT_PRUNE_HIGH_WATERMARK = 20000
DEFAULT_PRUNE_LOW_WATERMARK = 12000
#: TASK-15455: long resumes render only a buffered tail before Markdown is
#: parsed.  The effective budget also scales with the mounted viewport; these
#: floors keep a useful scroll buffer when layout has not settled yet.
DEFAULT_INITIAL_WINDOW_LINES = 144
DEFAULT_SCROLLBACK_CHUNK_LINES = 96
SCROLLBACK_HYDRATION_THRESHOLD = 2
#: task-2154.16 (FB-01): a failed assistant row with no partial content used
#: to render as the bare token ``[failed]``. It now shows this placeholder
#: (dimmed) with the state carried by a separate dim status line. Same copy
#: as the agent runtime's persisted empty-final-text fallback.
CONSOLE_FAILED_EMPTY_PLACEHOLDER = "No response was generated."
#: SP2 /rewind: render-derived (never a tree node) one-line banner shown above
#: the boundary message when "summarize up to here" is in effect.
CONSOLE_SUMMARY_BANNER_COPY = (
    "⤵ Earlier turns summarized for context — full history above"
)
EMPTY_TRANSCRIPT_PROVIDER_ACTION_LABEL = "Choose model"
EMPTY_TRANSCRIPT_PROVIDER_ACTION_TOOLTIP = (
    "Choose the provider and model for this Console session."
)
_SESSION_ID_UNSET = object()
# TASK-362 AC#2: the guide names the single-key shortcuts (j/k/c/e/r/Esc), which
# were otherwise undiscoverable anywhere in the app, alongside the icon meanings.
# task-2154.14 (DS-01): the static line was replaced by `action_row_guide()`,
# which names the row's glyph-only buttons in words derived from the row's own
# actions -- see the "action-help" row in `_transcript_rows` and `to_plain_text`.
_ACTION_TOOLTIPS = {
    "copy": "Copy this message to the clipboard.",
    "speak": "Speak this message aloud using text-to-speech.",
    "speak-stop": "Stop the current speech playback.",
    "edit": "Edit this message before continuing the thread.",
    "fork": "Fork chat from this message.",
    "more": "More message actions.",
    "save-as": "Choose a destination for this message, such as Chatbook or Note.",
    "toggle-image-view": "Cycle image view: pixels, graphics, hidden.",
    "save-image": "Save image to disk.",
    "tool-output": "Show or hide this tool call's full result (o).",
    "review-changes": "Open the Change Review screen for this turn (v).",
    "retry": "Retry the failed response.",
    "regenerate": "Generate another assistant variant for this turn.",
    "continue": "Continue and extend the selected message.",
    "feedback-up": "Mark this response as helpful.",
    "feedback-down": "Mark this response as not helpful.",
    "delete": "Delete this message from the Console transcript.",
    "variant-previous": "Show the previous regenerated variant.",
    "variant-next": "Show the next regenerated variant.",
    "keep": "Keep the browsed variant as this message's canonical image.",
}


def _coerce_card_state(value: object) -> ConsoleSetupCardState:
    """Guard against a transiently non-``ConsoleSetupCardState`` value.

    A flaky resume race can hand the empty panel a stale/incorrect value
    (observed as a bare ``str``) before the real card state lands. Fall back
    to the quiet copy rather than raising ``AttributeError`` deep in
    ``compose()``.
    """
    if isinstance(value, ConsoleSetupCardState):
        return value
    return ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)


def _coerce_prune_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def get_console_prune_watermarks(
    app_config: Mapping[str, object] | None,
) -> tuple[int, int]:
    """Resolve ``(low_mark, high_mark)`` pruning watermarks from config.

    Reads ``[chat_defaults] prune_high_watermark`` / ``prune_low_watermark``,
    falling back to :data:`DEFAULT_PRUNE_HIGH_WATERMARK` /
    :data:`DEFAULT_PRUNE_LOW_WATERMARK` when missing or invalid, and clamps
    ``low <= high``. A ``high_mark <= 0`` disables pruning.

    Args:
        app_config: The loaded application config dict (``app.app_config``).

    Returns:
        Tuple of ``(low_mark, high_mark)`` in virtual rows.
    """
    chat_defaults = (app_config or {}).get("chat_defaults", {})
    if not isinstance(chat_defaults, Mapping):
        chat_defaults = {}
    high = _coerce_prune_int(
        chat_defaults.get("prune_high_watermark"), DEFAULT_PRUNE_HIGH_WATERMARK
    )
    low = _coerce_prune_int(
        chat_defaults.get("prune_low_watermark"), DEFAULT_PRUNE_LOW_WATERMARK
    )
    if low > high:
        low = high
    return low, high


def get_console_transcript_window_lines(
    app_config: Mapping[str, object] | None,
) -> tuple[int, int]:
    """Resolve the transcript window line floors from config.

    Reads ``[chat_defaults] transcript_window_lines`` /
    ``transcript_scrollback_lines``, falling back to
    :data:`DEFAULT_INITIAL_WINDOW_LINES` / :data:`DEFAULT_SCROLLBACK_CHUNK_LINES`
    when missing or invalid. Both are FLOORS: the effective budget is the
    larger of the floor and the viewport-derived value. A
    ``transcript_window_lines <= 0`` disables windowing entirely (the kill
    switch: every message mounts at load, as it did before TASK-15455).

    Args:
        app_config: The loaded application config dict (``app.app_config``).

    Returns:
        Tuple of ``(initial_window_lines, scrollback_chunk_lines)``.
    """
    chat_defaults = (app_config or {}).get("chat_defaults", {})
    if not isinstance(chat_defaults, Mapping):
        chat_defaults = {}
    initial_lines = _coerce_prune_int(
        chat_defaults.get("transcript_window_lines"), DEFAULT_INITIAL_WINDOW_LINES
    )
    chunk_lines = _coerce_prune_int(
        chat_defaults.get("transcript_scrollback_lines"),
        DEFAULT_SCROLLBACK_CHUNK_LINES,
    )
    return initial_lines, max(1, chunk_lines)


def _message_role_label(message: ConsoleChatMessage) -> str:
    role = message.role.value if hasattr(message.role, "value") else str(message.role)
    return role.title()


#: Statuses an assistant row holds while its turn is still in flight. The
#: store starts an empty assistant row at "pending" and only promotes it to
#: "streaming" on the first streamed chunk (``append_stream_chunk``), which
#: a tool-calling agent turn never produces -- so the activity line must
#: cover both or it would never appear on the very turns it exists for.
_IN_FLIGHT_MESSAGE_STATUSES = frozenset({"pending", "streaming"})


def _row_is_in_flight(message: ConsoleChatMessage) -> bool:
    """Whether this row belongs to a turn that has not finished yet."""
    return (
        message.role is ConsoleMessageRole.ASSISTANT
        and message.status in _IN_FLIGHT_MESSAGE_STATUSES
    )


def _message_body(
    message: ConsoleChatMessage,
    presentation: ConsoleMessagePresentation | None = None,
) -> str:
    if presentation is not None:
        content = presentation.content
    elif message.variants is not None:
        content = message.variants.current.content
    else:
        content = message.content
    if _row_is_in_flight(message) and not content.strip() and message.live_activity:
        # The live turn-activity line, when the poll has one for this row.
        # Note the WIDER status gate than the placeholder below: measured on
        # a real agent turn with a tool held in flight, the in-flight row is
        # "pending" (the store only reaches "streaming" on the first streamed
        # chunk, and a tool-calling turn's assistant row never streams -- the
        # fence gate seals it). That is why this row used to render as a bare
        # "Assistant" for the whole multi-round turn.
        return message.live_activity
    if message.status == "streaming" and not content.strip():
        # Between send-accepted and the first streamed token the assistant row
        # has no content; show a visible generating state instead of an empty
        # row (local models can take 30-90s to first token).
        return CONSOLE_GENERATING_PLACEHOLDER
    content = render_exported_assistant_content(
        role=message.role.value,
        content=content,
        state=message.assistant_generation_state,
    )
    if (
        message.role is not ConsoleMessageRole.USER
        and message.status == "failed"
        and not content.strip()
    ):
        # task-2154.16 (FB-01): an empty failed row rendered as the bare
        # token "[failed]" -- meaningless. Show a placeholder instead; the
        # dim status line (`_message_status_line`) carries the state.
        # Render-only: stored content stays empty, and `skip_failed`/retry
        # semantics key off `status`, never this text. The USER-role gate
        # matches the old suffix's: a failed USER echo (TASK-457(a)
        # send-blocked) keeps the user's own text, explained by the SYSTEM
        # block-row.
        return CONSOLE_FAILED_EMPTY_PLACEHOLDER
    # task-2154.16 (FB-01): the raw "[streaming]"/"[stopped]"/"[failed]"
    # tokens are gone from message content -- the state renders as a dim
    # status line appended by `_message_render_text`/`to_plain_text`.
    return content


#: Status lines for assistant-response states (FB-01); rendered dim under
#: the body in place of the old "[status]" content token.
_MESSAGE_STATUS_LINES = {
    "streaming": "Streaming…",
    "stopped": "Stopped",
    "failed": "Failed",
}


def _message_status_line(message: ConsoleChatMessage) -> str:
    """Return the status line for an in-flight/terminal response row, or "".

    Same role gate as the old "[status]" suffix: a USER row only carries
    "failed" via the TASK-457(a) send-blocked echo, where the SYSTEM
    block-row already explains it -- so user text never grows a status line.
    """
    if message.role is ConsoleMessageRole.USER:
        return ""
    return _MESSAGE_STATUS_LINES.get(message.status, "")


def _citation_notice(message: ConsoleChatMessage) -> str:
    """Return exact content-free citation transition copy for one message."""
    presentation = message.citation_presentation
    if presentation is None:
        return ""
    if presentation.phase in {
        ConsoleCitationPhase.CHECKING,
        ConsoleCitationPhase.REPAIRING,
    }:
        return "Checking citations…"
    if presentation.notice_code is ConsoleCitationNoticeCode.REPAIRED:
        if presentation.original_attempt_available:
            return "Citations repaired · View original attempt"
        return "Citations repaired"
    if presentation.notice_code is ConsoleCitationNoticeCode.UNAVAILABLE:
        return "Citation repair unavailable · Original response kept"
    if presentation.notice_code is ConsoleCitationNoticeCode.CANCELED:
        return "Citation repair canceled"
    return ""


def _is_generating_placeholder_body(message: ConsoleChatMessage, body: str) -> bool:
    """Return True when the rendered body is a render-only in-flight line.

    Two bodies qualify: the pre-first-token ``Generating…`` placeholder and
    the live turn-activity line that replaces it while an agent turn works.
    Both are render-derived (never stored content), both render dim, and
    both already say "this turn is running" -- so neither may also grow the
    ``Streaming…`` status line under it.
    """
    if _row_is_in_flight(message) and body and body == message.live_activity:
        return True
    return message.status == "streaming" and body == CONSOLE_GENERATING_PLACEHOLDER


def _is_failed_placeholder_body(message: ConsoleChatMessage, body: str) -> bool:
    """Return True when the rendered body is the empty-failed placeholder."""
    return message.status == "failed" and body == CONSOLE_FAILED_EMPTY_PLACEHOLDER


def _human_size(size: int) -> str:
    """Format a byte count for display, matching ``attachment_core._format_size``.

    Kept as a small local helper (rather than importing ``attachment_core``)
    to keep this widget free of that dependency.
    """
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


def _message_image_chip_legacy(message: ConsoleChatMessage) -> str | None:
    """Return the placeholder chip line for a message carrying an image.

    ``attachment_label`` (e.g. "photo.png · 11 B") only exists in
    memory/screen-state -- the DB stores just ``image_data`` +
    ``image_mime_type``, so a message resumed from the DB has no label. When
    the raw bytes are available, synthesize a "{mime} · {size}" label instead
    of falling back to a bare MIME type. When only metadata was restored
    (``image_data`` is ``None``), keep the bare-mime fallback.
    """
    if message.image_data is None and not message.image_mime_type:
        return None
    if message.attachment_label:
        label = message.attachment_label
    elif message.image_data is not None:
        mime = message.image_mime_type or "image"
        label = f"{mime} · {_human_size(len(message.image_data))}"
    else:
        label = message.image_mime_type or "image"
    return f"🖼 {label}"


def _message_attachment_chips(message: ConsoleChatMessage) -> list[str]:
    """Return one placeholder chip line per attachment (position order).

    If no attachments are present, fall back to the legacy image chip logic
    (zero-attachment behavior unchanged).
    """
    attachments = getattr(message, "attachments", ()) or ()
    if not attachments:
        legacy = _message_image_chip_legacy(message)
        return [legacy] if legacy else []
    chips: list[str] = []
    for attachment in attachments:
        if attachment.display_name:
            chips.append(f"🖼 {attachment.display_name}")
        elif attachment.data is not None:
            chips.append(
                f"🖼 {attachment.mime_type or 'image'} · {_human_size(len(attachment.data))}"
            )
        else:
            chips.append(f"🖼 {attachment.mime_type or 'image'}")
    return chips


#: Inline markdown + roleplay flavor handled in-transcript: **bold**, `code`,
#: *action*, 'inner thought', and "quoted speech" (straight or curly). Matched
#: as closed pairs only, so an unclosed marker mid-stream stays literal until
#: it closes. Order matters: ** before * so bold never half-matches as
#: italics, and a quote swallows any markers inside it (task-1536).
_INLINE_MD_RE = re.compile(
    r"\*\*(.+?)\*\*"
    r"|`([^`]+)`"
    r"|(\"[^\"\n]+\")"
    r"|(“[^”\n]+”)"
    r"|((?<!\w)'(?:[^'\n]|(?<=\w)'(?=\w))+?'(?!\w))"
    r"|((?<!\w)‘(?:[^’\n]|(?<=\w)’(?=\w))+?’(?!\w))"
    r"|\*([^*\n]+)\*"
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

#: Roleplay flavor styles (task-1536). Concrete colors, not theme variables:
#: Content span styles are parsed directly and never resolve CSS ``$`` vars.
#: All four read on the dark default theme and stay distinct from each
#: other and from plain narration.
_BOLD_STYLE = "bold #f7d774"
_SPEECH_STYLE = "#8ecdf7"
_THOUGHT_STYLE = "italic #7de3c3"
_ACTION_STYLE = "italic #b596d8"

_CONSOLE_RP_SPEECH_COMPONENT = "console-rp-speech"
_CONSOLE_RP_THOUGHT_COMPONENT = "console-rp-thought"
_CONSOLE_RP_ACTION_COMPONENT = "console-rp-action"
_CONSOLE_RP_STRONG_COMPONENT = "console-rp-strong"
_CONSOLE_RP_COMPONENTS = frozenset(
    {
        _CONSOLE_RP_SPEECH_COMPONENT,
        _CONSOLE_RP_THOUGHT_COMPONENT,
        _CONSOLE_RP_ACTION_COMPONENT,
        _CONSOLE_RP_STRONG_COMPONENT,
    }
)
_ROLEPLAY_SPEECH_RE = re.compile(r'"[^"\n]+"|“[^”\n]+”')
_ROLEPLAY_THOUGHT_RE = re.compile(
    r"(?<!\w)'(?:[^'\n]|(?<=\w)'(?=\w))+?'(?!\w)"
    r"|(?<!\w)‘(?:[^’\n]|(?<=\w)’(?=\w))+?’(?!\w)"
)


def _inline_markdown_spans(line: str) -> list:
    """Split one line into Content segments, styling inline flavor.

    ``**bold**``, ``“quoted”``/``"quoted"`` speech, ``'inner thought'``, and
    ``*action*`` each get a distinct style; `code` keeps its plain italic.
    Text is always emitted literally (styles are applied via ``(text, style)``
    tuples, never markup parsing), so message text can never inject Rich
    markup. Quotation marks stay visible inside speech and thought spans;
    bold/action marker characters are stripped.

    Args:
        line: A single raw text line.

    Returns:
        A list of ``str`` / ``(str, style)`` segments for ``Content.assemble``.
    """
    out: list = []
    pos = 0
    for match in _INLINE_MD_RE.finditer(line):
        if match.start() > pos:
            out.append(line[pos : match.start()])
        bold, code, quote, curly_quote, thought, curly_thought, action = (
            match.groups()
        )
        if bold is not None:
            out.append((bold, _BOLD_STYLE))
        elif code is not None:
            out.append((code, "italic"))
        elif quote is not None:
            out.append((quote, _SPEECH_STYLE))
        elif curly_quote is not None:
            out.append((curly_quote, _SPEECH_STYLE))
        elif thought is not None:
            out.append((thought, _THOUGHT_STYLE))
        elif curly_thought is not None:
            out.append((curly_thought, _THOUGHT_STYLE))
        else:
            out.append((action, _ACTION_STYLE))
        pos = match.end()
    if pos < len(line):
        out.append(line[pos:])
    return out or [line]


def _markdown_body_spans(body: str) -> list:
    """Render a safe subset of markdown (headings, **bold**, `code`) to segments.

    TASK-372: assistant replies arrive as markdown and were shown raw (literal
    ``###`` / ``**`` / backticks). Headings render as bold underline with the
    ``#`` markers stripped; inline bold/code render via
    ``_inline_markdown_spans``. All styling goes through ``(text, style)`` tuples
    so nothing in the message is markup-parsed.

    Args:
        body: The raw message body text.

    Returns:
        A list of ``str`` / ``(str, style)`` segments for ``Content.assemble``.
    """
    segments: list = []
    for index, line in enumerate(body.split("\n")):
        if index:
            segments.append("\n")
        heading = _HEADING_RE.match(line)
        if heading:
            segments.append((heading.group(2), "bold underline"))
        else:
            segments.extend(_inline_markdown_spans(line))
    return segments


def _roleplay_flavor_content(content: Content) -> Content:
    """Annotate rendered Markdown text with semantic roleplay components.

    Standard Markdown emphasis already supplies ``.em`` and ``.strong``
    spans after stripping its markers. This projection adds Console-owned
    component spans for those roles and for closed straight or curly quoted
    speech. Inline code and links are carved out of speech ranges so their
    existing operational styling remains authoritative. The raw Markdown
    source is never rewritten.

    Args:
        content: Textual's literal inline rendering of one Markdown block.

    Returns:
        The same text with additional semantic component spans.
    """
    protected_ranges: list[tuple[int, int]] = []
    semantic_ranges: list[tuple[int, int, str]] = []
    for span in content.spans:
        if span.style == ".code_inline":
            protected_ranges.append((span.start, span.end))
        elif isinstance(span.style, Style) and span.style.meta.get("@click"):
            protected_ranges.append((span.start, span.end))
        elif span.style == ".em":
            semantic_ranges.append((span.start, span.end, _CONSOLE_RP_ACTION_COMPONENT))
        elif span.style == ".strong":
            semantic_ranges.append((span.start, span.end, _CONSOLE_RP_STRONG_COMPONENT))

    def unprotected_ranges(start: int, end: int) -> list[tuple[int, int]]:
        """Carve code and link spans out of one flavor range."""
        remaining = [(start, end)]
        for protected_start, protected_end in protected_ranges:
            next_ranges: list[tuple[int, int]] = []
            for range_start, range_end in remaining:
                if protected_end <= range_start or protected_start >= range_end:
                    next_ranges.append((range_start, range_end))
                    continue
                if range_start < protected_start:
                    next_ranges.append((range_start, protected_start))
                if protected_end < range_end:
                    next_ranges.append((protected_end, range_end))
            remaining = next_ranges
        return remaining

    flavor_spans: list[Span] = []
    speech_ranges = [match.span() for match in _ROLEPLAY_SPEECH_RE.finditer(content.plain)]
    for match_start, match_end in speech_ranges:
        flavor_spans.extend(
            Span(start, end, f".{_CONSOLE_RP_SPEECH_COMPONENT}")
            for start, end in unprotected_ranges(match_start, match_end)
            if start < end
        )
    for match in _ROLEPLAY_THOUGHT_RE.finditer(content.plain):
        if any(
            speech_start <= match.start() and match.end() <= speech_end
            for speech_start, speech_end in speech_ranges
        ):
            continue
        flavor_spans.extend(
            Span(start, end, f".{_CONSOLE_RP_THOUGHT_COMPONENT}")
            for start, end in unprotected_ranges(match.start(), match.end())
            if start < end
        )
    # Action / emphasis follows speech so nested Markdown emphasis retains its
    # more specific role instead of inheriting the surrounding dialogue color.
    for semantic_start, semantic_end, component in semantic_ranges:
        flavor_spans.extend(
            Span(start, end, f".{component}")
            for start, end in unprotected_ranges(semantic_start, semantic_end)
            if start < end
        )
    return content.add_spans(flavor_spans)


class ConsoleRoleplayFlavorBlockMixin:
    """Add roleplay component spans to an inline-capable Markdown block."""

    def _token_to_content(self, token) -> Content:
        # The compatibility resolver below proves this hook exists before any
        # flavored block type is built. Textual type selectors match concrete
        # widget types, so every block also receives a stable public CSS hook.
        self.add_class("console-roleplay-markdown-block")
        return _roleplay_flavor_content(super()._token_to_content(token))


_ROLEPLAY_BLOCK_KEYS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "paragraph_open",
    "th_open",
    "td_open",
)


def _resolve_textual_roleplay_blocks() -> dict[str, type[Widget]] | None:
    """Resolve the Textual block API required for roleplay annotations.

    Returns:
        The required block classes when their component registry and inline
        conversion hook are compatible, otherwise ``None``. Returning
        ``None`` keeps app startup safe across future Textual upgrades.
    """
    try:
        blocks = {key: Markdown.BLOCKS[key] for key in _ROLEPLAY_BLOCK_KEYS}
    except (AttributeError, KeyError, TypeError):
        return None
    for block_type in blocks.values():
        component_classes = getattr(block_type, "COMPONENT_CLASSES", None)
        token_converter = getattr(block_type, "_token_to_content", None)
        if not isinstance(block_type, type) or not issubclass(block_type, Widget):
            return None
        if not isinstance(component_classes, (set, frozenset)):
            return None
        if not callable(token_converter):
            return None
    return blocks


def _make_roleplay_block_type(
    class_name: str, block_type: type[Widget]
) -> type[Widget]:
    """Build one PascalCase Textual block subtype with RP components."""
    return type(
        class_name,
        (ConsoleRoleplayFlavorBlockMixin, block_type),
        {
            "__doc__": "Markdown block with Console roleplay inline components.",
            "__module__": __name__,
            "COMPONENT_CLASSES": (
                frozenset(block_type.COMPONENT_CLASSES) | _CONSOLE_RP_COMPONENTS
            ),
        },
    )


_resolved_roleplay_blocks = _resolve_textual_roleplay_blocks()
if _resolved_roleplay_blocks is None:
    logger.warning(
        "Console roleplay Markdown flavor is unavailable for this Textual "
        "version; using the standard Markdown renderer"
    )
else:
    _console_roleplay_blocks = dict(Markdown.BLOCKS)
    _block_class_names = {
        "h1": "ConsoleRoleplayMarkdownH1",
        "h2": "ConsoleRoleplayMarkdownH2",
        "h3": "ConsoleRoleplayMarkdownH3",
        "h4": "ConsoleRoleplayMarkdownH4",
        "h5": "ConsoleRoleplayMarkdownH5",
        "h6": "ConsoleRoleplayMarkdownH6",
        "paragraph_open": "ConsoleRoleplayMarkdownParagraph",
        "th_open": "ConsoleRoleplayMarkdownTH",
        "td_open": "ConsoleRoleplayMarkdownTD",
    }
    _console_roleplay_blocks.update(
        {
            key: _make_roleplay_block_type(_block_class_names[key], block_type)
            for key, block_type in _resolved_roleplay_blocks.items()
        }
    )


class ConsoleRoleplayMarkdown(Markdown):
    """Render full Markdown with source-preserving roleplay flavor spans.

    Compatible Textual block APIs receive semantic speech, action, and strong
    component spans. Incompatible future APIs retain standard Markdown
    rendering instead of preventing application startup.

    Args:
        markdown: Initial Markdown source, or ``None`` for an empty widget.
        name: Optional DOM name inherited from ``Markdown``.
        id: Optional DOM identifier inherited from ``Markdown``.
        classes: Optional initial CSS classes.
        parser_factory: Optional Markdown parser factory.
        open_links: Whether Textual should open clicked links automatically.
    """

    if _resolved_roleplay_blocks is not None:
        BLOCKS = _console_roleplay_blocks


def _speaker_label(
    message: ConsoleChatMessage,
    presentation: ConsoleMessagePresentation,
) -> str:
    """Return the literal resolved speaker label plus any sibling position."""
    label = presentation.speaker_label
    if message.sibling_count > 1:
        label = f"{label} ({message.sibling_index + 1}/{message.sibling_count})"
    return label


def _message_render_text(
    message: ConsoleChatMessage,
    *,
    selected: bool,
    presentation: ConsoleMessagePresentation | None = None,
) -> Content:
    """Return the compact transcript row renderable for a message.

    The role label is styled ``"dim"`` while the body keeps full contrast
    (no style). ``Content.plain`` matches the pre-existing plain-string
    rendering exactly (``"{role_label}  {body}"`` or ``"{role_label}\\n{body}"``)
    so plain-text assertions and exports are unaffected.

    Uses Textual's native ``Content`` visual (rather than ``rich.text.Text``)
    because ``Static.update()`` eagerly visualizes its argument: a Rich
    ``Text`` renderable requires an active app (``widget.app.console``) to
    convert, which raises ``NoActiveAppError`` for rows built/updated outside
    a mounted app (as several unit tests do). ``Content`` already satisfies
    Textual's ``Visual`` protocol, so it is used as-is without touching
    ``self.app``.
    """
    if presentation is None:
        presentation = resolve_console_message_presentation(
            message, ConsolePresentationContext()
        )
    role_label = _speaker_label(message, presentation)
    body = _message_body(message, presentation)
    chips = _message_attachment_chips(message)
    if chips:
        chip_lines = "\n".join(chips)
        body = f"{body}\n{chip_lines}" if body else chip_lines
    if _is_generating_placeholder_body(message, body) or _is_failed_placeholder_body(
        message, body
    ):
        body_segments: list = [(body, "dim")]
    elif message.role is ConsoleMessageRole.ASSISTANT:
        # TASK-372: render assistant markdown (headings/**bold**/`code`) with
        # terminal emphasis instead of literal marker characters. Only ASSISTANT
        # replies are markdown -- USER input, SYSTEM diagnostics, and TOOL output
        # stay verbatim, since their #/**/backtick characters may be literal and
        # meaningful (Qodo #823).
        body_segments = _markdown_body_spans(body)
    else:
        body_segments = [body]
    citation_notice = _citation_notice(message)
    if citation_notice:
        body_segments.extend(("\n", (citation_notice, "dim")))
    status_line = _message_status_line(message)
    if status_line and not _is_generating_placeholder_body(message, body):
        # The "Generating…" placeholder already implies streaming -- doubling
        # it with a "Streaming…" status line reads as noise, not signal.
        body_segments.extend(("\n", (status_line, "dim")))
    separator = "  " if not selected and "\n" not in body and len(body) <= 120 else "\n"
    return Content.assemble((role_label, "dim"), separator, *body_segments)


def _message_body_render_text(
    message: ConsoleChatMessage,
    presentation: ConsoleMessagePresentation,
) -> Content:
    """Render the plain row body while keeping labels in a dedicated child."""
    body = _message_body(message, presentation)
    chips = _message_attachment_chips(message)
    if chips:
        chip_lines = "\n".join(chips)
        body = f"{body}\n{chip_lines}" if body else chip_lines
    if _is_generating_placeholder_body(message, body) or _is_failed_placeholder_body(
        message, body
    ):
        body_segments: list = [(body, "dim")]
    elif message.role is ConsoleMessageRole.ASSISTANT:
        body_segments = _markdown_body_spans(body)
    else:
        body_segments = [body]
    citation_notice = _citation_notice(message)
    if citation_notice:
        body_segments.extend(("\n", (citation_notice, "dim")))
    status_line = _message_status_line(message)
    if status_line and not _is_generating_placeholder_body(message, body):
        body_segments.extend(("\n", (status_line, "dim")))
    return Content.assemble(*body_segments)


#: How many notes the inline marker lists before collapsing to "+N more".
#: The marker has no scroll of its own; the notes modal shows them all.
_MARKER_NOTE_PREVIEW_LIMIT = 3


def _annotation_marker_content(notes: tuple[str, ...]) -> Content:
    """One marker row's content: a dim header plus each note, first line only.

    The marker is the transcript-side viewer for persisted Comment
    annotations (task-17169): compact enough to sit inline, complete enough
    that the note is readable without leaving the transcript.
    """
    header = "Review note" if len(notes) == 1 else f"Review notes ({len(notes)})"
    segments: list = [(f"✎ {header}", "dim")]
    # Cap the listed notes: the marker is an inline transcript row with no
    # scroll of its own (the modal has one), so an unbounded list would push
    # the conversation off screen on a heavily-annotated message.
    for note in notes[:_MARKER_NOTE_PREVIEW_LIMIT]:
        first_line = note.splitlines()[0] if note else ""
        if len(first_line) > 200:
            first_line = first_line[:199] + "…"
        segments.extend(("\n", first_line))
    hidden = len(notes) - _MARKER_NOTE_PREVIEW_LIMIT
    if hidden > 0:
        segments.extend(("\n", (f"+{hidden} more", "dim")))
    return Content.assemble(*segments)


def _activity_is_expandable(
    message: ConsoleChatMessage,
    owned_rows: Iterable["_TranscriptRow"] = (),
) -> bool:
    """Return whether one TOOL marker owns any disclosure detail."""
    if message.role is not ConsoleMessageRole.TOOL:
        return False
    if (
        message.content.strip()
        or message.tool_output_full
        or message.tool_diff is not None
    ):
        return True
    return any(
        row.kind not in {"actions", "action-help", "message"}
        or (
            row.kind == "message"
            and row.message is not None
            and bool(row.message.content.strip())
        )
        for row in owned_rows
    )


@dataclass(frozen=True)
class _TranscriptRow:
    key: str
    kind: Literal[
        "rule",
        "banner",
        "message",
        "diff",
        "citations",
        "annotations",
        "original-attempt",
        "image",
        "generation-card",
        "video-card",
        "actions",
        "action-help",
        "assistant-turn",
        "empty",
    ]
    signature: tuple
    message: ConsoleChatMessage | None = None
    selected: bool = False
    renderable: str | Content = ""
    action_label: str = EMPTY_TRANSCRIPT_PROVIDER_ACTION_LABEL
    action_tooltip: str = EMPTY_TRANSCRIPT_PROVIDER_ACTION_TOOLTIP
    card_state: ConsoleSetupCardState | None = None
    image_spec: "ConsoleImageRowSpec | None" = None
    generation_card_spec: "ConsoleGenerationCardSpec | None" = None
    video_card_spec: "ConsoleVideoCardSpec | None" = None
    assistant_turn: ConsoleAssistantTurn | None = None
    nested_rows: tuple["_TranscriptRow", ...] = ()
    activity_rows: tuple[tuple["_TranscriptRow", ...], ...] = ()
    activity_items: tuple[ConsoleChatMessage | ConsoleThinkingActivityRef, ...] = ()
    activity_signature: tuple = ()
    adjunct_signature: tuple = ()


@dataclass(frozen=True)
class _ActivityComponents:
    """One activity disclosure's render children and reconciliation tokens."""

    presentation: ConsoleActivityPresentation
    action_widgets: tuple[Widget, ...]
    detail_widgets: tuple[Widget, ...]
    detail_available: bool
    action_signature: tuple
    detail_signature: tuple


def get_console_assistant_markdown(app_config: Mapping[str, object] | None) -> bool:
    """Resolve the ``[chat_defaults] assistant_markdown`` toggle.

    TASK-1990: assistant replies render through Textual's ``Markdown`` widget
    by default. This config switch restores the span-subset renderer
    (TASK-372) for sessions that prefer its literal safe-subset presentation.

    Args:
        app_config: The loaded application config dict (``app.app_config``).

    Returns:
        True when assistant rows should render full markdown (the default);
        non-bool or missing values resolve to True.
    """
    chat_defaults = (app_config or {}).get("chat_defaults", {})
    if not isinstance(chat_defaults, Mapping):
        return True
    value = chat_defaults.get("assistant_markdown", True)
    return value if isinstance(value, bool) else True


def _assistant_markdown_body(
    message: ConsoleChatMessage,
    presentation: ConsoleMessagePresentation | None = None,
) -> str:
    """Return the raw markdown body for a markdown row (no status suffix).

    The plain-text renderer appends ``" [streaming]"`` to the body; a suffix
    would defeat prefix-diffed appends, so the markdown row keeps status in
    its header line and feeds the Markdown widget content only.
    """
    if presentation is not None:
        content = presentation.content
    elif message.variants is not None:
        content = message.variants.current.content
    else:
        content = message.content
    if _row_is_in_flight(message) and not content.strip():
        # The grouped Markdown header owns healthy live/generating copy.
        return content
    return render_exported_assistant_content(
        role=message.role.value,
        content=content,
        state=message.assistant_generation_state,
    )


#: TASK-15456. A line that -- with <=3 leading spaces, per CommonMark's fence
#: indentation rule -- opens or closes a fenced code block: a run of 3+
#: backticks or tildes, optionally followed by trailing content (an info
#: string on an opening line; nothing but whitespace on a closing line).
_FENCE_DELIMITER_LINE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _console_markdown_body_ends_in_open_fence(body: str) -> bool:
    """Return True only when ``body`` is confidently known to end inside an
    open (unclosed) top-level Markdown code fence.

    Deliberately conservative: this tracks fence-delimiter parity line by
    line rather than running a real Markdown parse, so it can be called on
    every streamed delta without itself becoming the cost it exists to
    avoid. It mirrors just the CommonMark rules that matter for staying
    confident:

    - An opening line is <=3 leading spaces of ```` ``` ```` / ``~~~`` (3+
      repeats), optionally followed by an info string. A backtick-fenced
      info string may not itself contain a backtick (CommonMark); a line
      like ``` ```some`code` ``` therefore never opens a fence here --
      ambiguous stays "not open", the safe direction (see below).
    - A closing line is <=3 leading spaces of the SAME delimiter character,
      a run length >= the opening run, followed by nothing but trailing
      whitespace.
    - Anything else while a fence is open just stays inside it, regardless
      of what it contains (mismatched-delimiter or shorter-run lines can't
      close a real fence either).

    This does not track blockquote/list-item indentation contexts, so a
    fence nested inside one can be mis-scored. That is safe in both
    directions for the caller (a streaming-append throttle): scoring "open"
    when a real parse would say "closed" only delays a flush (content is
    still applied in full, just later); scoring "closed" when a real parse
    would say "open" only skips an optimization opportunity (falls back to
    the unthrottled append, today's behavior). Neither ever drops or
    reorders content -- the throttle that consumes this always applies the
    complete accumulated text on flush, and (fix round 1, TASK-15456) every
    path that can end a message's active streaming --
    ``_append_or_defer_body_delta`` seeing a non-"streaming" status, AND
    ``sync_message``'s no-new-body-text fast path -- flushes any buffered
    text unconditionally, without consulting this function at all. So a
    false "open" verdict here is now purely a *delayed-highlight* risk, not
    a drop risk, regardless of how this scan is wrong.

    Known cost: this rescans the *entire* ``body`` on every call that needs
    a decision (``_append_or_defer_body_delta`` passes ``self._body_text``,
    not just the new delta), so it is O(body length) per streamed tick, not
    O(delta). For a single assistant reply that stays well under a few
    hundred KB (the normal case) this is negligible next to the Pygments
    cost it exists to avoid, but it is a real, unbounded-with-message-size
    cost worth knowing about, not a free check.
    """
    fence_char: str | None = None
    fence_len = 0
    for line in body.split("\n"):
        match = _FENCE_DELIMITER_LINE_RE.match(line)
        if not match:
            continue
        run, rest = match.groups()
        char = run[0]
        if fence_char is None:
            if char == "`" and "`" in rest:
                # Not a valid opening fence line (backtick info strings
                # can't contain backticks) -- ordinary text, ignore it.
                continue
            fence_char = char
            fence_len = len(run)
            continue
        if char == fence_char and len(run) >= fence_len and not rest.strip():
            fence_char = None
            fence_len = 0
    return fence_char is not None


#: TASK-15456. While an assistant reply is streaming inside an open code
#: fence, batch appends for up to this many seconds before flushing to the
#: Markdown widget instead of calling ``Markdown.append()`` (and therefore
#: re-running Pygments over the whole fence-so-far, textual/widgets/
#: _markdown.py:895-901) on every 0.2s sync tick. A wall-clock deadline
#: (rather than a tick counter) keeps the bound meaningful even if a slow
#: model pauses mid-fence between chunks: it depends only on how long
#: content has sat unflushed, not on how many sync ticks happened to land
#: while it did. Total Pygments work across a stream scales down by
#: roughly (this many seconds / 0.2s sync interval) -- see the task's
#: Implementation Notes for the derivation.
_FENCE_APPEND_DEFER_SECONDS = 1.0


def _assistant_markdown_header(
    message: ConsoleChatMessage,
    presentation: ConsoleMessagePresentation | None = None,
) -> Content:
    """Return the dim one-line role/status header for a markdown row."""
    if presentation is None:
        presentation = resolve_console_message_presentation(
            message, ConsolePresentationContext()
        )
    role_label = _speaker_label(message, presentation)
    suffix = ""
    if _row_is_in_flight(message):
        body = _assistant_markdown_body(message, presentation)
        if not body.strip() and message.live_activity:
            # The owner's shape: "Assistant  ⚙ read_file · 4s" -- the live
            # line rides the row's own header, so no new widget is mounted
            # and no sibling can be pushed off screen by one.
            suffix = f"  {message.live_activity}"
        elif message.status == "streaming":
            # task-2154.16 (FB-01): same wording as the plain renderer's dim
            # status line -- never the raw "[streaming]" content token.
            suffix = (
                f"  {CONSOLE_GENERATING_PLACEHOLDER}"
                if not body.strip()
                else "  Streaming…"
            )
    elif message.status in {"stopped", "failed"}:
        suffix = f"  {_MESSAGE_STATUS_LINES[message.status]}"
    return Content.assemble(role_label, (suffix, "dim"))


def _assistant_markdown_footer(message: ConsoleChatMessage) -> Content | None:
    """Return the dim chips/citation footer for a markdown row, or None.

    Attachment chips and the citation transition notice render below the
    markdown body (the plain renderer folds them into its single Content).
    Both are emitted as literal text segments — never markup-parsed.
    """
    lines = list(_message_attachment_chips(message))
    citation_notice = _citation_notice(message)
    if citation_notice:
        lines.append(citation_notice)
    if not lines:
        return None
    return Content.assemble(("\n".join(lines), "dim"))


_MANAGED_MESSAGE_CLASSES = frozenset(
    {
        "console-transcript-message-selected",
        "console-transcript-message-tool",
        "console-transcript-message-system",
        "console-transcript-message-failed",
        "console-transcript-message-roleplay-user",
        "console-transcript-message-roleplay-character",
        "console-transcript-message-role-user",
        "console-transcript-message-role-assistant",
        "console-transcript-message-immersive-user",
        "console-transcript-message-immersive-assistant",
        "console-transcript-message-immersive-character",
    }
)


def _message_row_classes(
    message: ConsoleChatMessage,
    presentation: ConsoleMessagePresentation,
    *,
    selected: bool,
    markdown: bool,
) -> list[str]:
    classes = ["console-transcript-message"]
    if markdown:
        classes.append("console-transcript-message-markdown")
    if presentation.row_class:
        classes.append(presentation.row_class)
    if (
        presentation.transcript_style is ConsoleTranscriptStyle.IMMERSIVE_RP
        and presentation.speaker_tone is not None
    ):
        classes.append(
            f"console-transcript-message-immersive-{presentation.speaker_tone}"
        )
    if message.role is ConsoleMessageRole.TOOL:
        classes.append("console-transcript-message-tool")
    elif message.role is ConsoleMessageRole.SYSTEM:
        classes.append("console-transcript-message-system")
    if message.status == "failed":
        classes.append("console-transcript-message-failed")
    if selected:
        classes.append("console-transcript-message-selected")
    return classes


def _speaker_label_classes(presentation: ConsoleMessagePresentation) -> list[str]:
    classes = ["console-transcript-speaker-label"]
    if presentation.speaker_tone == "user" and presentation.row_class:
        classes.append("console-transcript-roleplay-user-label")
    elif presentation.speaker_tone == "character" and presentation.row_class:
        classes.append("console-transcript-roleplay-character-label")
    elif presentation.speaker_tone == "assistant" and presentation.row_class:
        classes.append("console-transcript-role-assistant-label")
    return classes


def _sync_message_classes(
    widget: Widget,
    message: ConsoleChatMessage,
    presentation: ConsoleMessagePresentation,
    *,
    selected: bool,
    markdown: bool,
) -> None:
    for class_name in _MANAGED_MESSAGE_CLASSES:
        widget.remove_class(class_name)
    for class_name in _message_row_classes(
        message, presentation, selected=selected, markdown=markdown
    ):
        widget.add_class(class_name)


class ConsoleMessageHeader(Horizontal):
    """Stable one-line speaker header with its sole visible speech control."""

    BUNDLED_CSS = """
    ConsoleMessageHeader {
        width: 100%;
        height: 1;
        min-height: 1;
    }

    ConsoleMessageHeader > .console-transcript-speaker-label {
        width: 1fr;
        height: 1;
        min-height: 1;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    ConsoleMessageHeader > .console-message-speech-presentation {
        width: 14;
        min-width: 14;
        height: 1;
        min-height: 1;
    }
    """

    def __init__(
        self,
        message: ConsoleChatMessage,
        presentation: ConsoleMessagePresentation,
        speech_state: ConsoleSpeechPresentationState,
        *,
        markdown: bool,
    ) -> None:
        self._message = message
        self._presentation = presentation
        self._speech_state = speech_state
        self._speech = resolve_console_header_speech(message, speech_state)
        classes = ["console-message-header"]
        if markdown:
            classes.append("console-markdown-header")
        super().__init__(
            id=f"console-message-header-{message.id}",
            classes=" ".join(classes),
        )

    def compose(self) -> ComposeResult:
        yield Static(
            self._speaker_copy(),
            classes=" ".join(_speaker_label_classes(self._presentation)),
            markup=False,
        )
        if self._speech.action is not None:
            yield self._speech_controls(self._speech)

    def _speaker_copy(self) -> Content:
        if self.has_class("console-markdown-header"):
            return _assistant_markdown_header(self._message, self._presentation)
        return Content(self._speaker_label())

    def _speaker_label(self) -> str:
        return _speaker_label(self._message, self._presentation)

    def _speech_controls(self, speech: ConsoleHeaderSpeechPresentation) -> Horizontal:
        assert speech.action is not None
        status = Static(
            speech.status_label,
            id=f"console-message-speech-status-{self._message.id}",
            classes="console-message-speech-status",
            markup=False,
        )
        action = Button(
            speech.action.label,
            id=f"console-message-speech-action-{self._message.id}",
            classes="console-message-speech-action",
            compact=True,
            disabled=not speech.action.enabled,
        )
        action.console_action_id = speech.action.action_id
        action.console_message_id = self._message.id
        action.console_restore_focus = False
        action.tooltip = speech.action.disabled_reason or _ACTION_TOOLTIPS.get(
            speech.action.action_id
        )
        return Horizontal(
            status,
            action,
            id=f"console-message-speech-presentation-{self._message.id}",
            classes="console-message-speech-presentation",
        )

    def sync_header(
        self,
        message: ConsoleChatMessage,
        presentation: ConsoleMessagePresentation,
        speech_state: ConsoleSpeechPresentationState,
    ) -> None:
        """Update copy/state in place, recomposing only the fixed control slot."""
        prior_has_action = self._speech.action is not None
        self._message = message
        self._presentation = presentation
        self._speech_state = speech_state
        self._speech = resolve_console_header_speech(message, speech_state)
        next_has_action = self._speech.action is not None
        try:
            speaker = self.query_one(".console-transcript-speaker-label", Static)
        except NoMatches:
            return
        speaker.set_classes(" ".join(_speaker_label_classes(presentation)))
        speaker.update(self._speaker_copy())
        if prior_has_action != next_has_action:
            self.refresh(recompose=True)
            return
        if self._speech.action is None:
            return
        try:
            status = self.query_one(".console-message-speech-status", Static)
            action = self.query_one(".console-message-speech-action", Button)
        except NoMatches:
            self.refresh(recompose=True)
            return
        status.update(self._speech.status_label)
        had_focus = action.has_focus
        if had_focus and not self._speech.action.enabled:
            action.console_restore_focus = True
        action.label = self._speech.action.label
        action.console_action_id = self._speech.action.action_id
        action.console_message_id = self._message.id
        action.disabled = not self._speech.action.enabled
        action.tooltip = self._speech.action.disabled_reason or _ACTION_TOOLTIPS.get(
            self._speech.action.action_id
        )
        if self._speech.action.enabled and action.console_restore_focus:
            action.console_restore_focus = False
            self.call_after_refresh(action.focus)


class ConsoleMarkdownMessage(Vertical):
    """Assistant transcript row rendered with Textual's Markdown widget.

    TASK-1990 (frogmouth-comparison follow-up). Streaming deltas are applied
    with ``Markdown.append()`` (prefix-diffed against the last applied body)
    so per-tick cost is O(delta), not O(message). Non-prefix changes (variant
    switch, edit) fall back to a full ``Markdown.update()``.

    Link policy (task AC#6): links never auto-open (``open_links=False``). A
    click on an http(s) link opens the system browser and notifies; any other
    scheme notifies with the href and does nothing else.

    Text selection (console selection phase 1, task G): implements the same
    four-method row protocol as ``ConsoleTranscriptMessage`` at LINE
    granularity -- the domain is the markdown SOURCE (``_body_text``), ranges
    snap outward to whole source lines, and the highlight is a reverse-video
    ``Static`` strip below the Markdown widget instead of restyling the
    Markdown renderer's internals (which would fight its block widgets).
    """

    can_focus = False

    DEFAULT_CSS = """
    ConsoleMarkdownMessage {
        height: auto;
    }
    ConsoleMarkdownMessage > Static {
        height: auto;
    }
    ConsoleMarkdownMessage > Markdown {
        height: auto;
        margin: 0;
        padding: 0;
        background: transparent;
    }
    """

    def __init__(
        self,
        message: ConsoleChatMessage,
        presentation: ConsoleMessagePresentation | None = None,
        *,
        selected: bool = False,
        speech_state: ConsoleSpeechPresentationState = "idle",
        show_header: bool = True,
    ) -> None:
        self.message_id = message.id
        self._presentation = presentation or resolve_console_message_presentation(
            message, ConsolePresentationContext()
        )
        classes = " ".join(
            _message_row_classes(
                message, self._presentation, selected=selected, markdown=True
            )
        )
        super().__init__(id=f"console-message-{message.id}", classes=classes)
        self._message = message
        self._speech_state = speech_state
        self._show_header = show_header
        self._body_text = _assistant_markdown_body(message, self._presentation)
        # Text-selection range over the markdown SOURCE at line granularity
        # (task G): offsets are always whole-line bounds. None = no highlight.
        self._selection_line_range: tuple[int, int] | None = None
        # TASK-15456: text appended to ``self._body_text`` above but not yet
        # handed to the Markdown widget (deferred while streaming inside an
        # open fence), plus the monotonic deadline by which it must flush.
        self._pending_fence_delta = ""
        self._fence_defer_deadline: float | None = None

    def compose(self) -> ComposeResult:
        if self._show_header:
            yield ConsoleMessageHeader(
                self._message,
                self._presentation,
                self._speech_state,
                markdown=True,
            )
        yield ConsoleRoleplayMarkdown(
            self._body_text,
            classes="console-markdown-body",
            open_links=False,
        )
        # Selection highlight strip (task G): hidden until a selection exists.
        # Composed once and toggled via ``display`` so selection updates never
        # mount/remove widgets (no DuplicateIds / async-remove lifecycle).
        selection_strip = Static(
            "",
            classes="console-markdown-selection-strip",
            markup=False,
        )
        selection_strip.display = False
        yield selection_strip
        footer_content = _assistant_markdown_footer(self._message)
        footer = Static(
            footer_content or "",
            classes="console-markdown-footer",
        )
        footer.display = footer_content is not None
        yield footer

    # -- Text-selection protocol (console selection phase 1, task G) ----------

    def get_display_text(self) -> str:
        """Return the markdown source this row renders (selection domain)."""
        return self._body_text

    def get_selection_text(self) -> str:
        """Return the selected whole source lines, capped for quoting."""
        if self._selection_line_range is None:
            return ""
        start, end = self._selection_line_range
        text = self.get_display_text()
        start, end = max(0, start), min(end, len(text))
        return cap_quote(text[start:end])

    def set_selection_range(self, start: int, end: int) -> None:
        """Highlight the character range ``[start, end)`` of the source.

        Live-spike feedback: whole-line snapping made any partial drag over
        a one-line reply select the ENTIRE message. The cell-to-offset map
        already resolves character positions, so the range is stored
        as-is; only its visual carrier (the reverse-video strip) still
        spells the text out below the Markdown body.
        """
        start, end = sorted((start, end))
        if end <= start:
            self.clear_selection()
            return
        text = self.get_display_text()
        new_range = (max(0, start), min(end, len(text)))
        if self._selection_line_range == new_range:
            # TASK-21114: unchanged effective range -- the strip already
            # shows it; skip the strip re-render (drags re-send the same
            # range at mouse-move rate). Text changes re-render via
            # ``_clamp_selection_to_text`` on sync.
            return
        self._selection_line_range = new_range
        self._refresh_selection_strip()

    def clear_selection(self) -> None:
        """Remove the highlight strip."""
        if self._selection_line_range is None:
            return
        self._selection_line_range = None
        self._refresh_selection_strip()

    def _clamp_selection_to_text(self) -> None:
        """Clamp the stored line range to the current source, re-snapped.

        Streaming updates grow/replace the markdown source; if the new text
        no longer contains the range start, drop the selection entirely
        (mirrors ``ConsoleTranscriptMessage``'s clamp-on-sync). A non-prefix
        replace can also shift line boundaries under the stored offsets, so
        the surviving range is re-snapped to the NEW text's whole lines --
        clamping offsets alone produced misaligned quotes (stray leading
        newline, partial line) when the old start landed mid-line of the new
        body (fix round 1).
        """
        if self._selection_line_range is None:
            return
        start, end = self._selection_line_range
        new_len = len(self._body_text)
        if start >= new_len:
            self._selection_line_range = None
        else:
            self._selection_line_range = (start, min(end, new_len))
        self._refresh_selection_strip()

    def _refresh_selection_strip(self) -> None:
        """Show/hide the reverse-video strip of selected source lines.

        The strip is the visually equivalent highlight: the Markdown widget
        owns its block layout and is left untouched, while the strip spells
        the selected source lines out below it in reverse video. The strip
        content is capped exactly like the quote (``cap_quote``) so
        select-all on a huge message cannot duplicate the whole body below
        itself (fix round 1).
        """
        try:
            strip = self.query_one(".console-markdown-selection-strip", Static)
        except NoMatches:
            return  # row not composed yet -- protocol state stays valid
        if self._selection_line_range is None:
            strip.display = False
            return
        start, end = self._selection_line_range
        text = self.get_display_text()
        selected = cap_quote(text[max(0, start) : min(end, len(text))])
        strip.update(Text(selected, style="reverse"))
        strip.display = bool(selected)

    def sync_message(
        self,
        message: ConsoleChatMessage,
        presentation: ConsoleMessagePresentation | None = None,
        *,
        selected: bool = False,
        speech_state: ConsoleSpeechPresentationState = "idle",
    ) -> None:
        """Update header/body/footer in place; append-only growth avoids re-parse."""
        presentation = presentation or self._presentation
        self.message_id = message.id
        self._message = message
        self._presentation = presentation
        self._speech_state = speech_state
        _sync_message_classes(
            self,
            message,
            presentation,
            selected=selected,
            markdown=True,
        )
        try:
            markdown = self.query_one(Markdown)
            footer = self.query_one(".console-markdown-footer", Static)
        except NoMatches:
            return
        try:
            header = self.query_one(ConsoleMessageHeader)
        except NoMatches:
            header = None
        if header is not None:
            header.sync_header(message, presentation, speech_state)
        footer_content = _assistant_markdown_footer(message)
        footer.update(footer_content or "")
        footer.display = footer_content is not None
        new_body = _assistant_markdown_body(message, presentation)
        if new_body == self._body_text:
            # TASK-15456 fix round 1: a status-only change (Stop, a
            # length-cutoff, a failure -- all routine mid-open-fence exits
            # from "streaming") reaches this fast path with an UNCHANGED
            # body, because the reconciler calls sync_message whenever the
            # row signature changes and status is part of that signature.
            # `self._body_text` already mirrors the true full content (it
            # advances immediately in the growth branch below, even when
            # the corresponding widget append was deferred) -- so this is
            # the only place a still-buffered fence-interior tail can ever
            # reach the widget once the message stops growing. Returning
            # here without checking would strand that text permanently.
            if self._pending_fence_delta and message.status != "streaming":
                self._flush_pending_fence_delta(markdown)
            return
        if new_body.startswith(self._body_text):
            delta = new_body[len(self._body_text) :]
            self._body_text = new_body
            self._append_or_defer_body_delta(markdown, delta, message)
            # Selection clamp-on-sync (task G): the source grew under a live
            # selection; the stored line bounds stay valid but the strip must
            # re-derive from the new source.
            self._clamp_selection_to_text()
        else:
            # Non-append edit (variant switch, retry, DB-resume rebind): the
            # prior diff base no longer applies, so any deferred fence
            # buffer is void too -- markdown.update(new_body) below replaces
            # the widget's content wholesale, so the (now-irrelevant, would
            # have been superseded anyway) pending text is safe to discard
            # rather than flush.
            self._pending_fence_delta = ""
            self._fence_defer_deadline = None
            markdown.update(new_body)
            self._body_text = new_body
            # Selection clamp-on-sync (task G): a replaced body may no longer
            # contain the selected lines -- clamp or clear like the plain row.
            self._clamp_selection_to_text()

    def _flush_pending_fence_delta(self, markdown: Markdown) -> None:
        """Hand any buffered fence-interior text to the widget and clear state.

        The single place that actually calls ``Markdown.append()`` for
        deferred text, used both by a deadline/status-driven flush inside
        ``_append_or_defer_body_delta`` and by ``sync_message``'s terminal-
        status fast path above -- so there is exactly one flush
        implementation for the "hand over the complete buffered text"
        invariant to hold against.
        """
        pending = self._pending_fence_delta
        self._pending_fence_delta = ""
        self._fence_defer_deadline = None
        if pending:
            markdown.append(pending)

    def _append_or_defer_body_delta(
        self, markdown: Markdown, delta: str, message: ConsoleChatMessage
    ) -> None:
        """Apply a streamed body delta, throttling fence-interior Pygments work.

        TASK-15456: ``Markdown.append()`` is cheap for ordinary prose growth
        (Textual only re-parses from the last completed top-level block), so
        that path is untouched -- this only changes behavior while the body
        currently ends inside an open code fence during active streaming,
        where every ``append()`` re-runs Pygments over the whole
        fence-so-far. In that state, deltas are buffered instead of applied
        immediately, for up to ``_FENCE_APPEND_DEFER_SECONDS``, bounding
        both the buffered memory and the visible staleness window. Every
        branch below still ends by handing the widget the *complete*
        buffered text -- append-order and final content are unaffected, only
        the number of Markdown/Pygments passes changes. A buffer started
        here can also be flushed later from ``sync_message``'s terminal-
        status fast path (see ``_flush_pending_fence_delta``) if the message
        stops growing before this function runs again.
        """
        self._pending_fence_delta += delta
        now = monotonic()
        deadline_due = (
            self._fence_defer_deadline is not None and now >= self._fence_defer_deadline
        )
        if (
            message.status == "streaming"
            and not deadline_due
            and _console_markdown_body_ends_in_open_fence(self._body_text)
        ):
            if self._fence_defer_deadline is None:
                self._fence_defer_deadline = now + _FENCE_APPEND_DEFER_SECONDS
            return
        self._flush_pending_fence_delta(markdown)

    @on(Markdown.LinkClicked)
    def _open_link(self, event: Markdown.LinkClicked) -> None:
        """Apply the explicit link policy: http(s) to the browser, else notify."""
        event.stop()
        href = event.href or ""
        if href.startswith(("http://", "https://")):
            import webbrowser

            webbrowser.open(href)
            self.notify(f"Opened in browser: {href}", timeout=4)
        else:
            self.notify(
                f"Link not opened (unsupported scheme): {href}",
                severity="warning",
                timeout=6,
            )

    async def on_click(self, event: Click) -> None:
        transcript = self.parent
        while transcript is not None and not isinstance(transcript, ConsoleTranscript):
            transcript = transcript.parent
        if isinstance(transcript, ConsoleTranscript):
            await transcript._dismiss_message_more_for_click(event.control)
        if event.control is not None and event.control.has_class(
            "console-message-speech-action"
        ):
            event.stop()
            return
        event.stop()
        if isinstance(transcript, ConsoleTranscript):
            manager = transcript.selection_manager
            if (
                manager.state.active
                or manager.just_finished
                or manager.consume_release_click()
            ):
                # This click completed (or landed during) a text-selection
                # drag on this selectable row (markdown rows arm drags too,
                # task G); it must not toggle message selection (console
                # selection phase 1). Consume the finish flag so the
                # suppression swallows exactly this one click. Live spike
                # 2026-08-16: BOTH tokens must die here and the event must
                # STOP -- a short-circuited ``or`` left release_click_pending
                # armed and the click bubbled to the transcript's on_click,
                # whose _remove_selection_menu() wiped the row selection
                # before the menu's action read it ("buttons don't work
                # after the first one").
                manager.consume_just_finished()
                manager.consume_release_click()
                event.stop()
                return
            transcript.toggle_message_selection(self.message_id)


class ConsoleTranscriptMessage(Vertical):
    """Clickable native Console transcript message row."""

    can_focus = False

    def __init__(
        self,
        message: ConsoleChatMessage,
        presentation: ConsoleMessagePresentation | None = None,
        *,
        selected: bool = False,
        speech_state: ConsoleSpeechPresentationState = "idle",
        show_header: bool = True,
    ) -> None:
        self.message_id = message.id
        self._message = message
        self._presentation = presentation or resolve_console_message_presentation(
            message, ConsolePresentationContext()
        )
        self._selected = selected
        self._speech_state = speech_state
        self._show_header = show_header
        # Text-selection range over the BODY text domain (header excluded),
        # console selection phase 1. None = no highlight.
        self._selection_range: tuple[int, int] | None = None
        # TASK-21114: cached body render (``get_display_text`` derives its
        # ``.plain`` from it). Rebuilt lazily; invalidated in ``sync_message``
        # -- the single seam where ``_message``/``_presentation`` are
        # reassigned after construction -- mirroring how the markdown row's
        # ``_body_text`` only ever changes there. A stale entry here would
        # corrupt selection offsets and copied quotes.
        self._body_render_cache: Content | None = None
        super().__init__(
            id=f"console-message-{message.id}",
            classes=" ".join(
                _message_row_classes(
                    message,
                    self._presentation,
                    selected=selected,
                    markdown=False,
                )
            ),
        )

    @property
    def renderable(self) -> Content:
        """Compatibility projection for unmounted row-level assertions."""
        return _message_render_text(
            self._message,
            selected=self._selected,
            presentation=self._presentation,
        )

    def compose(self) -> ComposeResult:
        if self._show_header:
            yield ConsoleMessageHeader(
                self._message,
                self._presentation,
                self._speech_state,
                markdown=False,
            )
        yield Static(
            self._body_render_content(),
            classes="console-transcript-message-body",
            markup=False,
        )

    def _speaker_label(self) -> str:
        return _speaker_label(self._message, self._presentation)

    # -- Text-selection protocol (console selection phase 1) -----------------
    # Offsets are BODY-only: the speaker header is a separate child widget and
    # never part of the selection domain.

    def _body_render_content(self) -> Content:
        """Return the (cached) body render Content (TASK-21114).

        Derived purely from ``_message`` + ``_presentation``; both are only
        reassigned in ``sync_message``, which invalidates this cache first.
        """
        if self._body_render_cache is None:
            self._body_render_cache = _message_body_render_text(
                self._message, self._presentation
            )
        return self._body_render_cache

    def get_display_text(self) -> str:
        """Return the plain body text this row renders (selection domain)."""
        return self._body_render_content().plain

    def get_selection_text(self) -> str:
        """Return the currently highlighted text, capped for quoting."""
        if self._selection_range is None:
            return ""
        start, end = sorted(self._selection_range)
        text = self.get_display_text()
        start, end = max(0, start), min(end, len(text))
        return cap_quote(text[start:end])

    def set_selection_range(self, start: int, end: int) -> None:
        """Highlight ``[start, end)`` in the body and re-render it."""
        if self._selection_range == (start, end):
            # TASK-21114: unchanged range -- the highlight already shows it;
            # skip the full-body re-render (drags re-send the same range at
            # mouse-move rate). Text changes re-render via ``sync_message``.
            return
        self._selection_range = (start, end)
        self._refresh_body_highlight()

    def clear_selection(self) -> None:
        """Remove any highlight and re-render the plain body."""
        if self._selection_range is None:
            return
        self._selection_range = None
        self._refresh_body_highlight()

    def _clamp_selection_to_text(self) -> None:
        """Clamp the stored range to the current text length (streaming sync).

        Streaming updates shrink/grow the body text; if the new text no longer
        contains the range start, drop the selection entirely.
        """
        if self._selection_range is None:
            return
        start, end = self._selection_range
        new_len = len(self.get_display_text())
        if start >= new_len:
            self._selection_range = None
        else:
            self._selection_range = (min(start, new_len), min(end, new_len))

    def _refresh_body_highlight(self) -> None:
        """Re-render the body Static with a reverse-video span over the range.

        The body Static is ``markup=False``, so a rich ``Text`` with spans is
        safe; with no range the original ``Content`` renderable is restored
        exactly, so non-selecting rows render byte-identically to before.
        """
        try:
            body = self.query_one(".console-transcript-message-body", Static)
        except NoMatches:
            return  # row not composed yet -- protocol state stays valid
        if self._selection_range is None:
            body.update(self._body_render_content())
            return
        plain = self.get_display_text()
        start, end = sorted(self._selection_range)
        start, end = max(0, start), min(end, len(plain))
        rich_text = Text(plain)
        if end > start:
            rich_text.stylize("reverse", start, end)
        body.update(rich_text)

    def sync_message(
        self,
        message: ConsoleChatMessage,
        presentation: ConsoleMessagePresentation | None = None,
        *,
        selected: bool = False,
        speech_state: ConsoleSpeechPresentationState = "idle",
    ) -> None:
        """Update row content and selection styling without remounting the row."""
        presentation = presentation or self._presentation
        self.message_id = message.id
        self._message = message
        self._presentation = presentation
        # TASK-21114: the body render derives from the two fields reassigned
        # above -- invalidate BEFORE anything below (the selection clamp
        # included) reads ``get_display_text``.
        self._body_render_cache = None
        self._selected = selected
        self._speech_state = speech_state
        _sync_message_classes(
            self,
            message,
            presentation,
            selected=selected,
            markdown=False,
        )
        try:
            self.query_one(".console-transcript-message-body", Static)
        except NoMatches:
            return
        try:
            header = self.query_one(ConsoleMessageHeader)
        except NoMatches:
            header = None
        if header is not None:
            header.sync_header(message, presentation, speech_state)
        # Clamp any live text-selection range to the NEW body length before
        # re-rendering: streaming deltas shrink/grow the text under the
        # selection (console selection phase 1).
        self._clamp_selection_to_text()
        self._refresh_body_highlight()

    async def on_click(self, event: Click) -> None:
        transcript = self.parent
        while transcript is not None and not isinstance(transcript, ConsoleTranscript):
            transcript = transcript.parent
        if isinstance(transcript, ConsoleTranscript):
            await transcript._dismiss_message_more_for_click(event.control)
        if event.control is not None and event.control.has_class(
            "console-message-speech-action"
        ):
            event.stop()
            return
        event.stop()
        if isinstance(transcript, ConsoleTranscript):
            manager = transcript.selection_manager
            if (
                manager.state.active
                or manager.just_finished
                or manager.consume_release_click()
            ):
                # This click completed (or landed during) a text-selection
                # drag on this row; it must not toggle message selection
                # (console selection phase 1). A genuine click never reaches
                # this branch: its empty drag finish consumed the flag on
                # MouseUp, so what is left here is the drag-release Click
                # (or a click landing mid-drag). Markdown rows carry the
                # identical guard in their own ``on_click`` (task G). Live
                # spike 2026-08-16: consume BOTH tokens and STOP the event
                # -- a lingering release_click_pending let the artifact
                # click reach the transcript's on_click, whose dismissal
                # cleanup wiped the selection the menu needs.
                manager.consume_just_finished()
                manager.consume_release_click()
                event.stop()
                return
            transcript.toggle_message_selection(self.message_id)


class ConsoleToolDiffRow(Vertical):
    """Inline diff row under an expanded file-write TOOL marker (TASK-1366).

    Mounts empty and fills in asynchronously: the diff is computed off the
    UI thread (``DiffView.prepare``) BEFORE the DiffView mounts, mirroring
    ``tool_message_widgets.ToolExecutionWidget``'s integration. The row is
    render-derived view state -- it exists only while its marker message is
    expanded via the full-output toggle, and disappears with it (or when
    the message leaves the view window).

    Text selection (console selection phase 3, task 1): implements the same
    four-method row protocol as the other rows at LINE granularity -- the
    domain is the deterministic unified-diff projection
    (``_tool_diff_display_text``), ranges snap outward to whole diff lines,
    and the highlight is a reverse-video ``Static`` strip below the DiffView
    instead of restyling the DiffView's internals. Diff content is
    immutable (fixed at append time), so unlike the plain/markdown rows
    there is no streaming clamp; row removal rides the existing
    reconciliation guard.
    """

    can_focus = False

    def __init__(self, message_id: str, diff: tuple[str, str, str]) -> None:
        self.message_id = message_id
        self._diff = diff
        # Text-selection range over the unified-diff projection at LINE
        # granularity (phase 3, task 1). Offsets are always whole-line
        # bounds. None = no highlight.
        self._selection_range: tuple[int, int] | None = None
        self._display_text: str | None = None
        super().__init__(
            id=f"console-tool-diff-{message_id}",
            classes="console-transcript-tool-diff",
        )

    def compose(self) -> ComposeResult:
        # Selection highlight strip (phase 3, task 1): composed once and
        # toggled via ``display`` (same lifecycle rationale as the markdown
        # row's strip -- selection updates never mount/remove widgets). The
        # DiffView mounts BEFORE it in ``_prepare_and_mount`` so the strip
        # always sits below the diff.
        selection_strip = Static(
            "",
            classes="console-tool-diff-selection-strip",
            markup=False,
        )
        selection_strip.display = False
        yield selection_strip

    def on_mount(self) -> None:
        path, old_content, new_content = self._diff
        self.run_worker(
            self._prepare_and_mount(path, old_content, new_content),
            thread=False,
            group="console-tool-diff-mount",
        )

    async def _prepare_and_mount(
        self, path: str, old_content: str, new_content: str
    ) -> None:
        """Prepare the diff off the UI thread, then mount the DiffView."""
        try:
            diff_view = make_diff(path, old_content, new_content)
            await diff_view.prepare()
            if not self.is_mounted:
                # Row was unmounted (collapse/prune/session swap) while the
                # diff prepared off-thread.
                return
            try:
                strip = self.query_one(".console-tool-diff-selection-strip", Static)
            except NoMatches:
                strip = None
            if strip is None:
                await self.mount(diff_view)
            else:
                await self.mount(diff_view, before=strip)
        except Exception as exc:  # noqa: BLE001 — a render failure never breaks the transcript
            logger.opt(exception=True).error(
                f"Failed to render console tool diff for {path}: {exc}"
            )

    # -- Text-selection protocol (console selection phase 3, task 1) ------

    def get_display_text(self) -> str:
        """Return the unified-diff projection this row renders (selection domain)."""
        if self._display_text is None:
            # Immutable contents: compute once and cache forever.
            self._display_text = _tool_diff_display_text(self._diff)
        return self._display_text

    def get_selection_text(self) -> str:
        """Return the selected whole diff lines, capped for quoting."""
        if self._selection_range is None:
            return ""
        start, end = self._selection_range
        text = self.get_display_text()
        start, end = max(0, start), min(end, len(text))
        return cap_quote(text[start:end])

    def set_selection_range(self, start: int, end: int) -> None:
        """Highlight ``[start, end)`` of the projection, snapped to whole lines.

        Diff-row granularity is the whole diff line (spec phase 3): the
        character range grows to cover every projection line it touches,
        so a coarse cell-to-offset map still yields whole-line quotes.
        """
        start, end = sorted((start, end))
        if end <= start:
            self.clear_selection()
            return
        text = self.get_display_text()
        start, end = _snap_to_line_bounds(text, start, end)
        if end <= start:
            self.clear_selection()
            return
        if self._selection_range == (start, end):
            # TASK-21114: unchanged snapped range -- skip the strip
            # re-render (drags re-send the same range at mouse-move rate;
            # line snapping makes repeats especially common here).
            return
        self._selection_range = (start, end)
        self._refresh_selection_strip()

    def clear_selection(self) -> None:
        """Hide the highlight strip."""
        if self._selection_range is None:
            return
        self._selection_range = None
        self._refresh_selection_strip()

    def _refresh_selection_strip(self) -> None:
        """Show/hide the reverse-video strip of selected diff lines.

        Same carrier as the markdown row's strip: the DiffView owns its
        block layout and is left untouched, while the strip spells the
        selected diff lines out below it in reverse video. The strip
        content is capped exactly like the quote (``cap_quote``) so
        select-all on a huge diff cannot duplicate the whole diff below
        itself.
        """
        try:
            strip = self.query_one(".console-tool-diff-selection-strip", Static)
        except NoMatches:
            return  # row not composed yet -- protocol state stays valid
        if self._selection_range is None:
            strip.display = False
            return
        start, end = self._selection_range
        text = self.get_display_text()
        selected = cap_quote(text[max(0, start) : min(end, len(text))])
        strip.update(Text(selected, style="reverse"))
        strip.display = bool(selected)


class ConsoleTranscriptActionButton(Button):
    """Message action button that supports Enter activation in transcript focus mode."""

    def on_key(self, event: Key) -> None:
        if event.key == "enter":
            self.action_activate_action()
            event.stop()
            event.prevent_default()
            return
        if event.key == "tab":
            self.action_focus_next_action()
            event.stop()
            event.prevent_default()
            return
        if event.key == "shift+tab":
            self.action_focus_previous_action()
            event.stop()
            event.prevent_default()
            return
        if event.key == "escape":
            self.action_clear_message_selection()
            event.stop()
            event.prevent_default()

    def action_activate_action(self) -> None:
        """Activate the focused message action.

        Presses the currently focused transcript action button.
        """
        self.press()

    def action_focus_next_action(self) -> None:
        """Move focus to the next visible action.

        Advances focus within the selected-message action row.
        """
        self._focus_relative_action(1)

    def action_focus_previous_action(self) -> None:
        """Move focus to the previous visible action.

        Moves focus backward within the selected-message action row.
        """
        self._focus_relative_action(-1)

    def action_clear_message_selection(self) -> None:
        """Clear transcript selection from a focused action button.

        Delegates to the parent transcript when the action row owns focus.
        """
        transcript = self._parent_transcript()
        if transcript is not None:
            transcript.action_clear_selection()

    def _parent_transcript(self) -> ConsoleTranscript | None:
        parent = self.parent
        while parent is not None and not isinstance(parent, ConsoleTranscript):
            parent = parent.parent
        return parent if isinstance(parent, ConsoleTranscript) else None

    def _focus_relative_action(self, offset: int) -> None:
        parent = self.parent
        if parent is None:
            return
        action_buttons = [
            child
            for child in parent.children
            if isinstance(child, ConsoleTranscriptActionButton) and not child.disabled
        ]
        if not action_buttons:
            return
        try:
            current_index = action_buttons.index(self)
        except ValueError:
            return
        action_buttons[(current_index + offset) % len(action_buttons)].focus()


class ConsoleTranscriptEmptyPanel(RecomposeCaptureGuard, Vertical):
    """Actionable Console transcript empty state, driven by a setup card state."""

    #: TASK-2154.8 (FR-03): in-panel recovery action id; routes through the
    #: same ``WorkbenchActionRequested("provider-recovery")`` channel as the
    #: blocking setup modal's primary action.
    PROVIDER_ACTION_ID = "console-empty-provider-action"

    def __init__(
        self,
        card_state: ConsoleSetupCardState,
        *,
        provider_action_label: str,
        provider_action_tooltip: str,
    ) -> None:
        super().__init__(
            id="console-transcript-empty-state",
            classes="console-transcript-empty-panel",
        )
        self.card_state = _coerce_card_state(card_state)
        self.provider_action_label = provider_action_label
        self.provider_action_tooltip = provider_action_tooltip

    def _provider_action_visible(self) -> bool:
        """Return whether the recovery action should render in the panel.

        The action is offered only when the screen synced in a concrete
        recovery label (provider blocked) AND the blocking setup modal is not
        already covering the transcript (``mode == "card"``) -- rendering a
        second, unreachable button under the overlay would be noise.
        """
        return (
            bool(self.provider_action_label.strip()) and self.card_state.mode != "card"
        )

    def compose(self) -> ComposeResult:
        # The blocking setup card (title + numbered steps + primary action) now
        # lives in ``ConsoleSetupModal``; while setup is incomplete
        # (``mode == "card"``) this in-transcript panel shows only the quiet
        # empty line, dimmed under the overlay. ``ready_line``/``quiet`` render
        # as before. TASK-2154.8 (FR-03): a quiet/ready empty state with a
        # blocked provider also offers the provider recovery action in place,
        # so a fresh session after a broken/removed provider is not a dead end.
        body = Static(
            self.card_state.body_copy or CONSOLE_QUIET_EMPTY_COPY,
            id="console-empty-body",
            classes="console-transcript-empty-body console-transcript-empty-state",
        )
        yield body
        if self._provider_action_visible():
            action = Button(
                self.provider_action_label,
                id=self.PROVIDER_ACTION_ID,
                classes="console-empty-provider-action",
                compact=True,
            )
            action.tooltip = self.provider_action_tooltip
            yield action

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the in-panel recovery action through the owning screen."""
        if event.button.id != self.PROVIDER_ACTION_ID:
            return
        event.stop()
        self.post_message(WorkbenchActionRequested("provider-recovery"))

    def sync_card_state(
        self,
        card_state: ConsoleSetupCardState,
        *,
        provider_action_label: str,
        provider_action_tooltip: str,
    ) -> None:
        """Refresh the onboarding surface from a new card state."""
        self.card_state = _coerce_card_state(card_state)
        self.provider_action_label = provider_action_label
        self.provider_action_tooltip = provider_action_tooltip
        self.refresh(recompose=True)


#: TASK-371: the "jump to latest" pill copy per run status. Absent statuses
#: (idle, blocked) show no pill -- there is no streaming context to jump to.
_JUMP_PILL_STREAMING = "▼ streaming below — jump to latest"
_JUMP_PILL_STOPPED = "▼ stopped — jump to latest"
_JUMP_PILL_READY = "▼ reply ready — jump to latest"
_JUMP_PILL_TEXT: Mapping[str, str] = {
    "validating": _JUMP_PILL_STREAMING,
    "retrying": _JUMP_PILL_STREAMING,
    "streaming": _JUMP_PILL_STREAMING,
    "checking_citations": "▼ checking citations below — jump to latest",
    "stopped": _JUMP_PILL_STOPPED,
    "failed": _JUMP_PILL_STOPPED,
    "completed": _JUMP_PILL_READY,
}


class ConsoleTranscriptJumpPill(Static):
    """Clickable 'jump to latest' pill shown while scrolled up during a run.

    TASK-371: when the reader scrolls off the bottom while a reply streams in,
    the content grows below the fold with no signal. This pill sits docked at the
    transcript bottom, states whether the run is streaming / stopped / ready, and
    on click re-attaches follow and jumps to the newest content.

    TASK-2154.11 (DS-05): keyboard-focusable and Enter/Space-activatable --
    it joins the transcript region's Tab cycle whenever it is visible (it is
    ``display: none`` while hidden, which drops it from the focus chain on
    its own). Activation goes through ``on_key`` (the same pattern as
    ``ConsoleTranscriptActionButton`` above), NOT BINDINGS: a Key event
    bubbles from the focused pill up to ``ConsoleTranscript.on_key``, which
    stops ``enter`` before App-level binding dispatch would ever consult the
    pill's own bindings.
    """

    can_focus = True

    def on_key(self, event: Key) -> None:
        """Activate on Enter/Space before the transcript's on_key can claim it."""
        if event.key in ("enter", "space"):
            self.action_jump_to_latest()
            event.stop()
            event.prevent_default()

    def on_click(self, event: Click) -> None:
        """Jump the parent transcript to its newest content.

        Args:
            event: The click event; stopped so it doesn't bubble to the
                transcript's message-selection handler.
        """
        event.stop()
        self._activate(refocus_transcript=False)

    def action_jump_to_latest(self) -> None:
        """Keyboard activation (Enter/Space): jump, then focus the transcript.

        The jump hides the pill itself (``jump_to_latest`` sets
        ``display = False``), so keyboard focus must move somewhere explicit
        -- the transcript it just scrolled is the natural post-jump context.
        """
        self._activate(refocus_transcript=True)

    def _activate(self, *, refocus_transcript: bool) -> None:
        """Jump the parent transcript to its newest content."""
        transcript = self.parent
        if isinstance(transcript, ConsoleTranscript):
            transcript.jump_to_latest()
            if refocus_transcript:
                transcript.focus()


class ConsoleReviewNotesRequested(Message):
    """Bubbled when the user asks to see a message's review notes.

    task-18515 review-note management, task 1: posted by
    ``ConsoleAnnotationMarker.on_click`` and by
    ``ConsoleTranscript.action_open_review_notes`` (the ``n`` binding).
    Anchored by NATIVE message id -- the owning screen resolves the actual
    note records from its own preview/store bookkeeping; this task only
    produces the request (the notes modal is a later task in the plan).
    """

    def __init__(self, anchor_message_id: str) -> None:
        super().__init__()
        self.anchor_message_id = anchor_message_id


class ConsoleAnnotationMarker(Static):
    """Inline review-note marker; click opens the notes modal (Part 2).

    Phase 4 shipped this as an anonymous Static NOT in
    PROTECTED_CLICK_CLASSES, so clicking it toggled message selection --
    the papercut this widget closes.
    """

    def __init__(
        self,
        renderable: VisualType,
        *,
        anchor_message_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(renderable, **kwargs)
        self.anchor_message_id = anchor_message_id

    def on_click(self, event: Click) -> None:
        event.stop()
        self.post_message(ConsoleReviewNotesRequested(self.anchor_message_id))


@lru_cache(maxsize=4)
def _body_wrap_table(text: str, width: int) -> tuple[tuple[tuple[int, str], ...], int]:
    """Memoized wrap table: each wrapped body line with its source offset.

    TASK-21114: a drag delivers MouseMove at 50-100 Hz and every event needs
    the wrapped layout of the SAME (text, width) -- re-running
    ``Content.wrap`` plus the offset-alignment scan over a multi-KB body per
    event was the dominant per-move cost. The table is pure in its key, so a
    small LRU covers a drag's lifetime while text growth (streaming) and
    width changes (resize mid-drag) each miss into a fresh entry and the old
    one ages out.

    Returns:
        ``(table, total_lines)`` where ``table[i]`` is ``(source_start,
        line_text)`` for wrapped line ``i`` and ``total_lines`` counts ALL
        wrapped lines. ``len(table) < total_lines`` means the alignment scan
        hit a wrap edge case it does not model at index ``len(table)`` --
        cells on or below that line fall back to the single-line mapping
        (exactly where the pre-memoization loop bailed out).
    """
    wrapped = [
        line.plain for line in Content(text, strip_control_codes=False).wrap(width)
    ]
    table: list[tuple[int, str]] = []
    source_offset = 0
    for line in wrapped:
        if line:
            start = text.find(line, source_offset)
            if start == -1 or text[source_offset:start].strip():
                # Wrap edge case not modeled (defensive): stop here; the
                # caller falls back to the single-line mapping for this
                # line and everything after it.
                break
            table.append((start, line))
            source_offset = start + len(line)
        else:
            # Blank wrapped line: anchors at the current position.
            table.append((source_offset, ""))
            # Consume the blank line's own break so later lines stay
            # aligned; any other inter-line whitespace is absorbed by the
            # next line's find() above.
            if source_offset < len(text) and text[source_offset] in "\r\n":
                source_offset += 1
    return tuple(table), len(wrapped)


def _body_cell_to_offset(text: str, width: int, cell_x: int, cell_y: int) -> int:
    """Map a body-local screen cell to a character offset in ``text``.

    Console selection phase 1. Plain-row bodies are ``Static`` widgets whose
    text wraps at the body's content width, so the vertical position must be
    resolved against the wrapped layout, not treated as one long line.
    ``Content.wrap`` mirrors the widget's own fold (leading indentation is
    preserved, the fold space is dropped), so each wrapped line is aligned
    back to its source offset by skipping the whitespace the fold dropped --
    mapping choice verified against ``Content.wrap`` on Textual 8.2.8. The
    wrap-plus-alignment work is memoized per (text, width) in
    ``_body_wrap_table`` (TASK-21114).

    Cells above the body clamp to offset 0, cells below the last wrapped
    line to the end of the text; on the hovered line the x cell maps through
    ``offset_for_cell`` (clamped to that line's extent).

    Args:
        text: The row's display (plain body) text -- the selection domain.
        width: The body Static's content width (cells).
        cell_x: Body-local column of the pointer.
        cell_y: Body-local row of the pointer.

    Returns:
        Character offset into ``text`` for the cell, always in
        ``[0, len(text)]``.
    """
    if width <= 0 or not text:
        # Not laid out (or nothing to select): monotone single-line mapping.
        return offset_for_cell(text, cell_x)
    table, total_lines = _body_wrap_table(text, width)
    if cell_y < 0:
        return 0
    if cell_y >= total_lines:
        return len(text)
    if cell_y >= len(table):
        # On or below an unmodeled wrap edge: fall back to the single-line
        # mapping rather than mis-anchor the drag.
        return offset_for_cell(text, cell_x)
    start, line = table[cell_y]
    return start + offset_for_cell(line, cell_x)


def _snap_to_line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Snap ``[start, end)`` outward to whole ``'\\n'``-delimited lines.

    Console selection phase 1 (task G): markdown-row granularity is the whole
    source line, so a character range grows to cover every line it touches.
    The trailing newline of the last touched line is excluded, so the quoted
    text is exactly the selected lines.
    """
    start, end = sorted((start, end))
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))
    line_start = text.rfind("\n", 0, start) + 1
    newline = text.find("\n", end)
    line_end = len(text) if newline == -1 else newline
    return line_start, line_end


def _markdown_cell_to_offset(text: str, height: int, cell_x: int, cell_y: int) -> int:
    """Map a markdown-body-local cell to a source-line character offset.

    Console selection phase 1 (task G). The ``Markdown`` widget does not
    expose which rendered line belongs to which SOURCE line (blocks re-flow,
    wrap, and pad internally), so the body-local ``cell_y`` is distributed
    evenly across the source lines and clamped to the nearest line -- the
    recorded phase-1 approximation (ADR-068). ``set_selection_range`` then
    snaps outward to whole lines, which bounds the damage of a coarse y map:
    the quoted text is always whole lines regardless.

    Two clamp details keep drags useful when the render COLLAPSES source
    lines (soft-wrapped paragraphs render several source lines as one row,
    so ``height < line count``): cells above the body map to the first line,
    and the LAST rendered row maps to the last source line -- otherwise a
    drag that ends on the bottom row could never reach the final lines.
    ``cell_x`` maps within the target line via ``offset_for_cell``.

    Args:
        text: The row's markdown source text -- the selection domain.
        height: The Markdown widget's rendered height (cells).
        cell_x: Body-local column of the pointer.
        cell_y: Body-local row of the pointer.

    Returns:
        Character offset into ``text`` for the cell, always in
        ``[0, len(text)]``.
    """
    lines = text.split("\n")
    if height <= 0 or len(lines) <= 1:
        # Not laid out (or a single source line): monotone single-line map.
        return offset_for_cell(text, cell_x)
    if cell_y < 0:
        line_index = 0
    elif cell_y >= height:
        return len(text)
    elif cell_y == height - 1:
        line_index = len(lines) - 1
    else:
        line_index = min(int(cell_y * len(lines) / height), len(lines) - 1)
    line_start = sum(len(line) + 1 for line in lines[:line_index])
    return line_start + offset_for_cell(lines[line_index], cell_x)


def _tool_diff_display_text(diff: tuple[str, str, str]) -> str:
    """Deterministic plain-text projection of a tool diff (selection domain).

    Console selection phase 3, task 1. The selection domain of a
    ``ConsoleToolDiffRow`` is the unified diff of its immutable
    ``(path, old, new)`` contents, built with ``keepends=True`` so offsets
    are line-anchored, and ``fromfile=tofile=path`` so the header names the
    file. Pure function on the tuple -- unit-testable without a widget.

    Args:
        diff: ``(file_path, old_content, new_content)`` as captured at the
            provider's strip seam.

    Returns:
        The joined unified-diff text (the row's selection domain).
    """
    path, old_content, new_content = diff
    return "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
    )


def _diff_cell_to_offset(text: str, height: int, cell_x: int, cell_y: int) -> int:
    """Map a DiffView-local cell to a diff-projection character offset.

    Console selection phase 3, task 1. ``DiffView`` renders line-number
    gutters, hunk headers, and (in split mode) paired +/- columns, so its
    rows do not map 1:1 onto any exposed text -- exactly like the Markdown
    rows (task G), the body-local ``cell_y`` is distributed evenly across
    the projection's lines and clamped to the nearest line, and
    ``set_selection_range`` then snaps outward to whole diff lines, which
    bounds the damage of a coarse y map: the quoted text is always whole
    diff lines regardless. ``wrap=False`` (the diff-widgets default) keeps
    long diff lines unwrapped (they scroll horizontally), so ``cell_x``
    maps monotonically within the target line.

    Args:
        text: The row's unified-diff projection -- the selection domain.
        height: The DiffView's rendered height (cells).
        cell_x: Diff-view-local column of the pointer.
        cell_y: Diff-view-local row of the pointer.

    Returns:
        Character offset into ``text`` for the cell, always in
        ``[0, len(text)]``.
    """
    return _markdown_cell_to_offset(text, height, cell_x, cell_y)


#: Character/word/line-granularity motion keys (plain + markdown rows only;
#: diff rows only take ``_KB_LINE_KEYS`` and the ``o`` swap).
_KB_CHAR_KEYS = frozenset({"h", "l", "w", "b", "0", "$"})
#: Line-granularity motion keys -- valid on every eligible row kind.
_KB_LINE_KEYS = frozenset({"j", "k"})

_KB_DIFF_SELECTION_HINT = "j/k lines · o swap · Enter menu · Esc cancel"
_KB_CHAR_SELECTION_HINT = (
    "h/l chars · w/b words · 0/$ line · j/k lines · o swap · Enter menu · Esc cancel"
)


def _kb_selection_hint_text(
    row: "ConsoleTranscriptMessage | ConsoleMarkdownMessage | ConsoleToolDiffRow",
) -> str:
    """Status-line copy for keyboard text-selection mode (phase 5).

    Diff rows only take the line-granularity motions (`j`/`k`/`o`); plain
    and markdown rows take the full char/word/line motion set. Task 3's
    final copy per the SDD brief's Interfaces block.
    """
    if isinstance(row, ConsoleToolDiffRow):
        return _KB_DIFF_SELECTION_HINT
    return _KB_CHAR_SELECTION_HINT


#: Every constructed, not-yet-collected transcript (TASK-21119).
#:
#: Same contract as ``_LIVE_SELECTION_MENUS``, and now the same two hooks:
#: registration in ``__init__`` is synchronous and strictly precedes DOM
#: attachment, so the registry can never MISS a mounted transcript (the
#: direction that would silently break the click-outside cleanup); it may
#: over-report, and attachment is always re-derived from the DOM in
#: ``console_transcripts_on_screen``. ``_on_unmount`` prunes recomposed
#: transcripts out of the candidate set, which is an optimization only.
_LIVE_TRANSCRIPTS: "WeakSet[ConsoleTranscript]" = WeakSet()


def console_transcripts_on_screen(
    screen: "Screen[object]",
) -> list["ConsoleTranscript"]:
    """Transcripts currently attached under ``screen``.

    Replaces ``screen.query(ConsoleTranscript)`` (a full-screen DOM walk) on
    the per-press dismissal path. A Console screen holds one transcript
    (side chats add at most a handful), so the candidate scan is a couple of
    parent-chain walks, not a walk of the whole screen.

    Args:
        screen: The screen whose subtree is being inspected.

    Returns:
        The attached transcripts, in unspecified order.
    """
    transcripts: list[ConsoleTranscript] = []
    for transcript in _LIVE_TRANSCRIPTS:
        if transcript.parent is None:
            continue  # never mounted, or already detached (cheap arm)
        try:
            if transcript.screen is screen:
                transcripts.append(transcript)
        except NoScreen:
            continue  # attached to an orphaned subtree mid-teardown
    return transcripts


class ConsoleTranscript(VerticalScroll):
    """Focusable native Console transcript with compact rule-separated messages."""

    can_focus = True

    class TranscriptTextSelected(Message):
        """Posted when a mouse drag finished with a non-empty text selection.

        Console selection phase 1. This transcript mounts the floating
        selection menu at the release cell (screen coordinates); the event
        is not stopped, so the owning screen may also consume it (selection
        lifecycle).
        """

        def __init__(
            self, selection: TextSelection, screen_x: int, screen_y: int
        ) -> None:
            super().__init__()
            self.selection = selection
            self.screen_x = screen_x
            self.screen_y = screen_y

    BINDINGS = [
        ("down,j", "select_next", "Next message"),
        ("up,k", "select_previous", "Previous message"),
        ("enter", "confirm_selection", "Toggle message selection"),
        ("escape", "clear_selection", "Clear selection"),
        ("s", "enter_text_selection", "Select text"),
        ("c", "invoke_selected_action('copy')", "Copy"),
        ("e", "invoke_selected_action('edit')", "Edit"),
        ("f", "invoke_selected_action('fork')", "Fork"),
        ("r", "invoke_selected_action('regenerate')", "Regenerate"),
        ("o", "invoke_selected_action('tool-output')", "Full output"),
        ("v", "invoke_selected_action('review-changes')", "Review changes"),
        ("n", "open_review_notes", "Notes"),
    ]

    PROTECTED_CLICK_CLASSES: frozenset[str] = frozenset(
        {
            "console-transcript-action-row",
            "console-transcript-action-guide",
            "console-message-speech-presentation",
            "console-message-speech-action",
            "console-transcript-empty-panel",
            "console-transcript-empty-body",
            "console-transcript-empty-state",
            "console-transcript-rule",
            "console-transcript-summary-banner",
            "console-transcript-citation-sources",
            "console-transcript-annotations",
            # Textual scrollbars carry the generic system-widget class; ignore them
            # defensively if a scrollbar click ever bubbles up to the transcript.
            "-textual-system",
            "vertical-scrollbar",
            "horizontal-scrollbar",
            "scrollbar",
        }
    )
    """Widget classes that must keep the current selection active when clicked."""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        #: Turn-file-card spec: a zero-arg callable returning the shared
        #: change-review provider, or ``None``. Late-binding (the same
        #: builder convention as the region's other constructor callables)
        #: so a stale reference is never cached across session switches.
        #: Always set post-construction via ``set_change_review_provider_
        #: factory`` (the screen's sync loop keeps it current on the
        #: mounted instance every tick, mirroring ``set_summary_boundary``/
        #: ``set_image_specs``) -- no constructor kwarg for this, so every
        #: harness that builds this widget directly starts with the card
        #: switched off until the setter runs.
        self._change_review_provider_factory: Callable[[], Any] | None = None
        self._presentation_context = ConsolePresentationContext()
        self._messages: list[ConsoleChatMessage] = []
        self._unit_spans_by_index: tuple[
            tuple[int, int, str, tuple[str, ...]], ...
        ] = ()
        self.selected_message_id: str | None = None
        #: task-501: a selection to apply on the NEXT message ingest that
        #: contains this id. Set by the screen's sibling-swipe handler, which
        #: runs BEFORE the (possibly coalesced) post-swipe view reaches this
        #: widget -- selecting eagerly would either miss the membership guard
        #: or be cleared by reconciliation against the stale set. Applying it
        #: at ingest time keeps the swiped-to sibling selected so repeated
        #: `<`/`>` presses need no re-click.
        self.pending_selection_id: str | None = None
        #: SP2 /rewind: native id of the "summarize up to here" boundary message.
        #: Render-derived only -- a banner row is emitted above this message when
        #: it is among the rendered messages; ``None`` (or a dangling id) shows
        #: no banner. Set by the screen sync path from
        #: ``store.session_context_summary``; never mutates store/tree state.
        self.summary_boundary_message_id: str | None = None
        self._follow_intent_time = 0.0
        self._user_scroll_time = 0.0
        #: TASK-16851: when the last ``scroll_end`` (the End key) was issued.
        #: The prune's restore compares this against its ENTRY capture: an
        #: End that lands inside the entry->restore window engages the raw
        #: anchor AFTER the capture, and without the stamp the restore reads
        #: that engagement as the shrink-clamp's spurious re-attach and
        #: quietly cancels the user's drain (the frame-wide End-during-prune
        #: race the TASK-15777 round-3 review filed).
        self._scroll_end_intent_time = 0.0
        self._refresh_lock = asyncio.Lock()
        self._empty_card_state = ConsoleSetupCardState(
            mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY
        )
        # TASK-2154.8 (FR-03): empty means "no recovery action offered". The
        # screen syncs a concrete label only while the provider is blocked.
        self.empty_state_action_label = ""
        self.empty_state_action_tooltip = ""
        self._row_widgets: dict[str, Widget] = {}
        self._row_signatures: dict[str, tuple] = {}
        self._row_build_counts: dict[str, int] = {}
        self._image_specs: dict[str, ConsoleImageRowSpec] = {}
        self._generation_card_specs: dict[str, ConsoleGenerationCardSpec] = {}
        self._video_card_specs: dict[str, ConsoleVideoCardSpec] = {}
        self._fork_eligibility_by_message_id: dict[str, ConsoleForkEligibility] = {}
        self._original_attempt_previews: dict[str, str] = {}
        self._citation_counts: dict[str, int] = {}
        # task-17169: screen-owned review-note previews keyed by native
        # message id -- a message with entries gains an inline marker row.
        self._annotation_previews: dict[str, tuple[str, ...]] = {}
        self._speech_states: dict[str, ConsoleSpeechPresentationState] = {}
        #: TASK-1860: ids of TOOL markers currently showing their FULL result.
        #: Pure view state, owned here: expansion never touches the store, is
        #: per row (so several calls in one turn expand independently), and is
        #: deliberately dropped when the transcript is rebuilt for another
        #: session rather than following the user across conversations.
        self._expanded_tool_output_ids: set[str] = set()
        #: Trusted model-activity identity/owner projection for the current
        #: session. Full thinking text stays in the Assistant envelope.
        self._thinking_activity_refs: dict[str, ConsoleThinkingActivityRef] = {}
        self._show_model_thinking = True
        self._pending_thinking_auto_collapse: set[str] = set()
        self._manual_thinking_disclosures: set[str] = set()
        self._closed_live_thinking_blocks: set[tuple[str, str]] = set()
        # Optional session-boundary identity supplied by the owning screen.
        # The sentinel preserves the historical id-intersection behavior for
        # direct/legacy callers that do not know about sessions.
        self._session_identity: object = _SESSION_ID_UNSET
        #: This poll tick's live turn-activity line ("⚙ read_file · 4s"),
        #: or "" when no turn is in flight. Pure view state, re-supplied by
        #: the screen on every 0.2s sync tick via `apply_turn_activity`;
        #: never derived here and never stored on the message the store
        #: owns (see `_with_turn_activity`).
        self._turn_activity: str = ""
        # TASK-259: per-message render-signature cache. Maps message id ->
        # (cheap change-token, expensive row signature). `_transcript_rows`
        # re-derives the render payload (Content assembly) only when the
        # token differs, making derivation O(changed messages) per tick.
        # Lives on the widget instance so a recompose starts it fresh.
        self._message_signature_cache: dict[str, tuple[tuple, tuple]] = {}
        self._signature_compute_counts: dict[str, int] = {}
        # TASK-298: every message id seen by set_messages, so a NEW
        # user-role message ANYWHERE in the update (a send) re-engages
        # tail-follow even after the user scrolled up. The send path
        # appends USER + ASSISTANT placeholder together, so the tail
        # alone can miss the send (PR #697 review).
        self._seen_message_ids: set[str] = set()
        #: TASK-371: last run status seen by `sync_jump_indicator`, so a scroll
        #: that detaches the reader can refresh the pill without a status source.
        self._last_run_status = "idle"
        #: Last-applied (visible, text) pill state; lets the 0.2s streaming sync
        #: tick skip the query_one + update when nothing changed (Qodo #826).
        self._jump_pill_state: tuple[bool, str] | None = None
        #: TASK-1365: view-only prune window. Ids of the oldest messages whose
        #: rows were dropped after the virtual height crossed the high
        #: watermark. The store keeps the full history; ``_transcript_rows``
        #: filters these ids, so a refresh or recompose never resurrects them
        #: and the reconciliation stale-key path owns their removal.
        self._pruned_message_ids: set[str] = set()
        self._prune_check_scheduled = False
        #: TASK-15455: scrollback hydration is coalesced separately from the
        #: existing height-prune pass.  Both mutate the same contiguous hidden
        #: prefix, but only hydration removes ids from it.
        self._scrollback_hydration_scheduled = False
        self._hydrating_scrollback = False
        #: Console selection phase 1: drag-selection state over plain rows.
        #: Public so the owning screen (and tests) can inspect/consume it.
        self.selection_manager = SelectionManager()
        #: Row widget the active drag started on. Mouse capture reroutes
        #: subsequent moves to THIS transcript (event control is then the
        #: transcript itself), so extension resolves the origin row here
        #: instead of from the event's control. Cleared on finish/cancel and
        #: by the reconciliation guard when the row is removed/rebuilt.
        #: Plain, markdown, or tool diff row (all implement the selection
        #: protocol).
        self._selection_origin_row: (
            ConsoleTranscriptMessage
            | ConsoleMarkdownMessage
            | ConsoleToolDiffRow
            | None
        ) = None
        #: TASK-15777: view-only hidden TAIL — the second window boundary.
        #: Always a contiguous SUFFIX of ``_messages`` derived from one index
        #: (``_hidden_tail_start``), so mounted rows stay one contiguous slice
        #: by construction: two boundary indices over one list can never
        #: produce islands, which is why no gap-seam row is needed (the
        #: failure mode TASK-15455 rejected came from two independently
        #: mutated sets). Empty in every one-sided regime — the kill switch,
        #: disabled pruning, and degenerate watermarks never populate it.
        self._hidden_tail_ids: set[str] = set()
        self._hidden_tail_start: int | None = None
        self._tailward_hydration_scheduled = False
        #: One-shot id consumed by ``refresh_messages``: after a re-centered
        #: far jump the old scroll offset is meaningless, so the revealed
        #: target is scrolled to the top of the viewport once its row mounts.
        self._reveal_scroll_target: str | None = None
        #: Review E: while the re-center's reconcile + placement are in
        #: flight, the layout transits states (an emptied arrangement, the
        #: target parked at y~0) that look exactly like top-boundary hits and
        #: fired one spurious upward hydration — the jump landed with an
        #: extra chunk mounted ABOVE the target. Latched in
        #: ``_recenter_window_on``, released once the placement lands.
        self._suppress_boundary_hydration = False
        #: Console selection phase 5 (keyboard mode): the row currently
        #: armed for keyboard-driven text selection via `s`, or ``None``
        #: when the mode is off. Distinct from ``_selection_origin_row``
        #: (shared with the mouse-drag path) so exit can restore ownership
        #: cleanly -- entry sets both.
        self._kb_selection_row: (
            ConsoleTranscriptMessage
            | ConsoleMarkdownMessage
            | ConsoleToolDiffRow
            | None
        ) = None
        #: Task 3's motion-cursor endpoints over the armed row's display
        #: text (anchor = fixed end, end = moving end). Kept next to
        #: ``_kb_selection_row`` so exit clears all three together; unused
        #: until Task 3 wires the motion keys.
        self._kb_anchor: int | None = None
        self._kb_end: int | None = None
        # TASK-21119: register BEFORE any mount can happen (Textual delivers
        # ``Mount`` asynchronously), so the screen's click-outside gate can
        # never miss a transcript that is already in the DOM.
        _LIVE_TRANSCRIPTS.add(self)

    def _on_unmount(self) -> None:
        """Prune the transcript registry (TASK-21119).

        Best-effort only, exactly like the menu's: correctness never depends
        on it (``console_transcripts_on_screen`` re-checks attachment, and
        the weak reference expires on its own), it just keeps the candidate
        set from carrying every recomposed transcript until the next
        collection. No ``super()`` call is needed -- Textual dispatches
        ``_on_unmount`` from every class in the MRO, so ``Widget``'s own
        teardown still runs.
        """
        _LIVE_TRANSCRIPTS.discard(self)

    @property
    def has_pending_selection_ui(self) -> bool:
        """Whether the screen's click-outside cleanup would change anything.

        The screen-level dismissal (``ChatScreen._dismiss_console_selection_
        menus_outside_transcript``) does three things per transcript: clear
        the highlighted row, cancel the selection manager, and drop the
        origin row. All three are no-ops when the manager is idle and no
        origin row is held -- including the keyboard-selection mode, which
        arms the manager without mounting a menu (so a menu-only gate would
        leave its reverse-video strip painted after a click elsewhere).
        """
        return (
            self._selection_origin_row is not None or not self.selection_manager.is_idle
        )

    def on_mount(self) -> None:
        """Engage tail-follow: stay scrolled to the newest content.

        Textual's anchor keeps the view pinned to the bottom while content
        grows, releases when the user scrolls up, and re-engages when they
        return to the bottom (TASK-298 -- streamed replies taller than the
        viewport used to finish below the fold with no scroll).
        """
        self.anchor()
        # TASK-1365: a transcript composed with a preloaded (resumed) history
        # can already exceed the watermarks before any refresh_messages call.
        self._schedule_prune_check()

    def compose(self) -> ComposeResult:
        self._row_widgets.clear()
        self._row_signatures.clear()
        self._row_build_counts.clear()
        for row in self._transcript_rows():
            widget = self._build_row_widget(row, track=True)
            self._row_widgets[row.key] = widget
            self._row_signatures[row.key] = row.signature
            yield widget
        # TASK-371: docked (non-scrolling) jump-to-latest pill; hidden until
        # `sync_jump_indicator` shows it while the reader is scrolled up.
        pill = ConsoleTranscriptJumpPill(
            "",
            id="console-transcript-jump-pill",
            classes="console-transcript-jump-pill",
        )
        pill.display = False
        # The fresh pill starts hidden; drop the applied-state cache so the next
        # sync re-applies to this new widget (recompose creates a new pill).
        self._jump_pill_state = None
        yield pill
        # Console selection phase 5: keyboard text-selection mode's status
        # line. Hidden until `_enter_keyboard_selection` shows it (mirrors
        # the jump pill above) -- a fresh recompose always starts hidden;
        # re-entering the mode after one always takes a fresh `s` press.
        hint = Static(
            "",
            id="console-kb-selection-hint",
            classes="console-kb-selection-hint",
        )
        hint.display = False
        yield hint

    async def recompose(self) -> None:
        """Detach screen-owned message overflow UI before rebuilding rows."""
        menus = message_more_menus_on_screen(self.screen) if self.is_mounted else []
        opener_id = menus[0].opener_button_id if menus else ""
        await self.dismiss_message_more_menu(restore_focus=False)
        await super().recompose()
        if menus:
            self._restore_message_action_focus(opener_id)

    @property
    def allow_vertical_scroll(self) -> bool:
        """Accept scroll gestures whenever the transcript holds messages.

        TASK-336 (live mechanism): during heavy row churn (sub-agent runs)
        the arrangement transiently collapses — ``max_scroll_y`` reads 0
        (scroll_y can even go negative via the compositor's anchor path) —
        and the base gate (``is_scrollable and show_vertical_scrollbar``)
        is False at exactly the moment the wheel event arrives. The gesture
        is then silently dropped: no scroll, and crucially no
        ``release_anchor``, so follow never detaches (the review's
        byte-identical wheel evidence). A clamped scroll on a collapsed
        layout is a harmless no-op, but accepting it registers the
        reader's intent; the layout recovers within a tick and subsequent
        gestures scroll normally.
        """
        if self._messages:
            return True
        return super().allow_vertical_scroll

    def _raw_anchor_engaged(self) -> bool:
        """Return Textual's anchor state: pinned to the bottom of the MOUNTED rows."""
        return bool(self.is_anchored and not getattr(self, "_anchor_released", False))

    def _is_following_tail(self) -> bool:
        """Return True when the view is pinned to the NEWEST content.

        TASK-15777 (review A): Textual re-engages the anchor without calling
        ``anchor()`` — ``Widget.scroll_end()`` (the End key) and
        ``_check_anchor()`` both clear ``_anchor_released`` directly — so the
        raw anchor state can be engaged while a hidden tail exists ("ghost
        follow": pinned to the bottom of a stale slice, not the newest
        content). Treating that state as following suppressed the jump pill
        (whose visibility is gated on NOT following) exactly when it was the
        only recovery, and let streamed replies accumulate unseen in the
        hidden tail. The predicate is therefore the belt: whatever path
        flips Textual's flag, a non-empty hidden tail means NOT following.
        The braces are the convergence paths — the ``_hydrate_tailward``
        chain and the ``set_messages`` ghost-follow heal — which drain or
        drop the suffix so the raw state becomes true again.
        """
        return self._raw_anchor_engaged() and not self._hidden_tail_ids

    def sync_jump_indicator(self, run_status: str) -> None:
        """Show/hide the jump-to-latest pill for the current run + scroll state.

        TASK-371: while the reader is detached from the bottom during (or just
        after) a run, a docked pill reports whether the reply is streaming,
        stopped, or ready and offers a one-click jump to the newest content. It
        stays hidden while following the tail or when no run is in play.

        Args:
            run_status: The current Console run status value (e.g. ``streaming``).
        """
        self._last_run_status = run_status
        text = _JUMP_PILL_TEXT.get(run_status, "")
        visible = bool(text and self._messages and not self._is_following_tail())
        target = (visible, text if visible else "")
        # Called on every 0.2s streaming sync tick -- skip the query_one + update
        # when the effective state is unchanged (the common steady-stream case).
        if target == self._jump_pill_state:
            return
        try:
            pill = self.query_one(
                "#console-transcript-jump-pill", ConsoleTranscriptJumpPill
            )
        except NoMatches:
            return
        if visible:
            pill.update(text)
        pill.display = visible
        self._jump_pill_state = target

    def jump_to_latest(self) -> None:
        """Re-engage tail-follow, scroll to the newest content, and hide the pill."""
        self.anchor()
        self.scroll_end(animate=False)
        try:
            self.query_one(
                "#console-transcript-jump-pill", ConsoleTranscriptJumpPill
            ).display = False
        except NoMatches:
            pass
        self._jump_pill_state = (False, "")

    def note_follow_intent(self) -> None:
        """Record a programmatic jump-to-tail intent (send/resume/switch).

        TASK-336: the send-time ``anchor()`` arrives via the coalesced sync
        pass and can land AFTER the user has already wheel-scrolled up —
        yanking them back to the tail mid-stream. ``set_messages`` only
        honors a new-user-send anchor when the most recent of (follow
        intent, user scroll) is the intent; a later user scroll wins.
        """
        self._follow_intent_time = monotonic()

    def release_anchor(self) -> None:
        """Release tail-follow, stamping the scroll as user intent.

        Every user-driven scroll path (wheel, keyboard scroll actions)
        funnels through ``release_anchor`` — the timestamp lets a scroll
        that happens after a send outrank that send's late-arriving anchor
        (see ``note_follow_intent``, TASK-336).
        """
        self._user_scroll_time = monotonic()
        super().release_anchor()
        # TASK-371: surface the jump pill the moment the reader detaches, rather
        # than waiting for the next 0.2s sync tick.
        self.sync_jump_indicator(self._last_run_status)

    def _release_anchor_quietly(self) -> None:
        """Release the raw anchor WITHOUT stamping user scroll intent.

        For programmatic corrections (the prune restoring a detached reader
        the shrink-clamp re-attached — TASK-15777 review round 2). The
        public ``release_anchor`` records the release as a user gesture,
        which would let a maintenance scroll outrank a later send's follow
        intent (TASK-336 ordering).
        """
        super().release_anchor()

    @on(events.MouseScrollUp)
    def _hydrate_scrollback_on_boundary_wheel(
        self, _event: events.MouseScrollUp
    ) -> None:
        """Notice an upward wheel gesture that cannot change ``scroll_y``."""
        if self.scroll_y <= SCROLLBACK_HYDRATION_THRESHOLD:
            self._schedule_scrollback_hydration()

    @on(events.MouseScrollDown)
    def _hydrate_tailward_on_boundary_wheel(
        self, _event: events.MouseScrollDown
    ) -> None:
        """Notice a downward wheel gesture against a hidden tail (TASK-15777)."""
        if self._hidden_tail_ids and self._at_bottom_boundary():
            self._schedule_tailward_hydration()

    def action_page_up(self) -> None:
        """Page upward and hydrate when the current window is already at y=0."""
        at_boundary = self.scroll_y <= SCROLLBACK_HYDRATION_THRESHOLD
        super().action_page_up()
        if at_boundary:
            self._schedule_scrollback_hydration()

    def action_page_down(self) -> None:
        """Page downward and reveal the hidden tail at the bottom boundary."""
        at_boundary = bool(self._hidden_tail_ids) and self._at_bottom_boundary()
        super().action_page_down()
        if at_boundary:
            self._schedule_tailward_hydration()

    def _at_bottom_boundary(self) -> bool:
        """Return True when the view cannot scroll meaningfully further down."""
        return self.scroll_y >= self.max_scroll_y - SCROLLBACK_HYDRATION_THRESHOLD

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Hydrate the neighboring window when a detached reader hits a boundary."""
        super().watch_scroll_y(old_value, new_value)
        if new_value <= old_value and new_value <= SCROLLBACK_HYDRATION_THRESHOLD:
            self._schedule_scrollback_hydration()
        elif (
            new_value >= old_value
            and self._hidden_tail_ids
            and self._at_bottom_boundary()
        ):
            self._schedule_tailward_hydration()

    def _window_viewport_height(self) -> int:
        """Return a stable viewport height for line-budget calculations."""
        return max(1, int(self.size.height or 24))

    def _window_line_settings(self) -> tuple[int, int]:
        """Return the configured ``(initial, scrollback)`` line floors."""
        try:
            app_config = getattr(self.app, "app_config", None)
        except NoActiveAppError:
            app_config = None
        return get_console_transcript_window_lines(app_config)

    def _windowing_enabled(self) -> bool:
        """Return False when the config kill switch asks for the whole history."""
        return self._window_line_settings()[0] > 0

    def _initial_window_line_budget(self) -> int:
        """Return the tail-first render budget in estimated terminal lines."""
        return max(
            self._window_line_settings()[0],
            self._window_viewport_height() * 6,
        )

    def _scrollback_chunk_line_budget(self) -> int:
        """Return the amount of earlier history prepended per boundary request."""
        return max(
            self._window_line_settings()[1],
            self._window_viewport_height() * 4,
        )

    def _estimated_message_lines(self, message: ConsoleChatMessage) -> int:
        """Cheaply estimate a message's rendered height without parsing Markdown."""
        content = (
            message.variants.current.content
            if message.variants is not None
            else message.content
        )
        content_width = max(20, int(self.size.width or 80) - 4)
        wrapped_lines = 0
        for line in content.splitlines() or ("",):
            wrapped_lines += max(1, (len(line) + content_width - 1) // content_width)
        # One separator and one speaker/status line are the stable minimum
        # around every message body.  Rich/Textual may wrap differently, so
        # watermarks remain the authoritative post-layout bound.
        return wrapped_lines + 2

    def _turn_aligned_start(
        self, messages: list[ConsoleChatMessage], start: int
    ) -> int:
        """Move a window boundary back to the nearest user turn."""
        start = max(0, min(start, len(messages)))
        if start < len(messages):
            start, _end, _owner_id, _owned_ids = self._unit_span_at(messages, start)
        while start > 0 and (
            start >= len(messages) or messages[start].role != ConsoleMessageRole.USER
        ):
            start -= 1
        return start

    def _unit_span_at(
        self, messages: list[ConsoleChatMessage], index: int
    ) -> tuple[int, int, str, tuple[str, ...]]:
        """Return the causal span and owner for the unit containing ``index``."""
        index = max(0, min(index, len(messages) - 1))
        if messages is self._messages:
            return self._unit_spans_by_index[index]
        return self._build_unit_spans(messages)[index]

    @staticmethod
    def _build_unit_spans(
        messages: list[ConsoleChatMessage],
        units: Iterable[ConsoleTranscriptUnit] | None = None,
    ) -> tuple[tuple[int, int, str, tuple[str, ...]], ...]:
        """Build one causal span lookup entry per message index."""
        index_by_id = {message.id: offset for offset, message in enumerate(messages)}
        spans: list[tuple[int, int, str, tuple[str, ...]] | None] = [
            None
        ] * len(messages)
        if units is None:
            units = group_console_transcript_messages(messages)
        for unit in units:
            if unit.standalone is not None:
                standalone = unit.standalone
                start = index_by_id[standalone.id]
                spans[start] = (start, start + 1, standalone.id, (standalone.id,))
                continue
            turn = unit.assistant_turn
            assert turn is not None
            start = index_by_id[turn.assistant.id]
            end = start + len(turn.owned_message_ids)
            span = (start, end, turn.assistant.id, turn.owned_message_ids)
            spans[start:end] = [span] * (end - start)
        return tuple(
            span
            if span is not None
            else (index, index + 1, messages[index].id, (messages[index].id,))
            for index, span in enumerate(spans)
        )

    def _ownership_by_message_id(
        self, messages: list[ConsoleChatMessage] | None = None
    ) -> dict[str, tuple[str, tuple[str, ...]]]:
        """Map every causal message id to its top-level owner and unit ids."""
        source = self._messages if messages is None else messages
        ownership: dict[str, tuple[str, tuple[str, ...]]] = {}
        for unit in group_console_transcript_messages(source):
            if unit.standalone is not None:
                message = unit.standalone
                ownership[message.id] = (message.id, (message.id,))
                continue
            turn = unit.assistant_turn
            assert turn is not None
            owner = (turn.assistant.id, turn.owned_message_ids)
            for message_id in turn.owned_message_ids:
                ownership[message_id] = owner
            for ref in project_thinking_activities(assistant=turn.assistant):
                ownership[ref.activity_id] = owner
        return ownership

    def _tail_window_start(
        self,
        messages: list[ConsoleChatMessage],
        *,
        end: int | None = None,
        line_budget: int,
    ) -> int:
        """Return the earliest message in a bounded tail slice."""
        end = len(messages) if end is None else max(0, min(end, len(messages)))
        if end == 0:
            return 0
        used = 0
        start = end
        while start > 0 and used < line_budget:
            start -= 1
            used += self._estimated_message_lines(messages[start])
        return self._turn_aligned_start(messages, start)

    def _first_visible_message_index(self) -> int:
        """Return the contiguous window start in the complete message list."""
        for index, message in enumerate(self._messages):
            if message.id not in self._pruned_message_ids:
                return index
        return len(self._messages)

    def _set_hidden_prefix(self, start: int) -> None:
        """Replace the view-only hidden set with one contiguous prefix."""
        start = max(0, min(start, len(self._messages)))
        if start < len(self._messages):
            start, _end, _owner_id, _owned_ids = self._unit_span_at(
                self._messages, start
            )
        self._pruned_message_ids = {message.id for message in self._messages[:start]}

    def _replace_hidden_tail(self, start: int | None) -> None:
        """Replace the hidden suffix at an already unit-aligned boundary.

        Args:
            start: Index of the first hidden-tail message, or ``None`` (or an
                index at/past the end) to mount through the true tail.
        """
        if start is None or start >= len(self._messages):
            self._hidden_tail_start = None
            self._hidden_tail_ids = set()
            return
        start = max(0, start)
        self._hidden_tail_start = start
        self._hidden_tail_ids = {message.id for message in self._messages[start:]}

    def _hide_tail_from(self, start: int | None) -> None:
        """Hide from ``start``, rounding backward to the containing unit."""
        if start is None or start >= len(self._messages):
            self._replace_hidden_tail(None)
            return
        start = max(0, start)
        unit_start, _end, _owner_id, _owned_ids = self._unit_span_at(
            self._messages, start
        )
        self._replace_hidden_tail(unit_start)

    def _reveal_hidden_tail_through(self, end: int) -> None:
        """Reveal through exclusive ``end``, rounding forward to unit end."""
        end = max(0, min(end, len(self._messages)))
        if end == 0:
            self._replace_hidden_tail(0)
            return
        _start, unit_end, _owner_id, _owned_ids = self._unit_span_at(
            self._messages, end - 1
        )
        self._replace_hidden_tail(unit_end if unit_end < len(self._messages) else None)

    def _hidden_tail_start_index(self) -> int:
        """Return the hidden-tail boundary index (``len`` when no tail is hidden)."""
        if self._hidden_tail_start is None:
            return len(self._messages)
        return self._hidden_tail_start

    def _two_sided_active(self) -> bool:
        """Return True when the hidden-tail boundary is allowed to engage.

        TASK-15777: two-sided windowing requires ALL of
        - windowing enabled (the kill switch restores the exact pre-15455
          behavior, including the pruned prefix being unreachable by scroll),
        - pruning enabled (``high_mark > 0``; with pruning off there is no
          ceiling to lift and no height contract to trim against), and
        - watermarks that can hold at least one scrollback chunk plus a
          couple of viewports of reader context, with a chunk of headroom
          between low and high.

        The last condition is the fixed-point loop-breaker's replacement: with
        sane marks, post-hydration trimming returns the mounted height to
        ~``low`` before the prune check runs — and the trim works in the
        prune's own MEASURED units (review B: an estimated trim held only
        while ``measured/estimated <= high/low``, and ordinary short
        messages exceed that at tight ratios) — so the prune (which fires
        only above ``high``) cannot chase hydration, except while a
        protected group (selection, focus) blocks the trim, where scroll-back
        stalls bounded rather than churning free. Degenerate marks (e.g. the
        45/70 test configuration, where a single chunk overshoots ``high``)
        keep the TASK-15455 refusal-at-low-watermark behavior instead — there
        the refusal IS the loop-breaker and must survive.
        """
        if not self._windowing_enabled():
            return False
        low_mark, high_mark = self._prune_watermarks()
        if high_mark <= 0:
            return False
        chunk = self._scrollback_chunk_line_budget()
        viewport = self._window_viewport_height()
        return low_mark >= chunk + 2 * viewport and high_mark - low_mark >= chunk

    def _reveal_tail_window(self) -> None:
        """Drop the hidden tail and re-window onto a fresh bounded tail.

        Every "take me to the newest content" path (the jump pill, a new
        user send, any ``anchor()``) funnels here when a hidden tail exists:
        simply clearing the suffix would remount everything between the
        reader's scroll-back position and the tail in one pass — the exact
        unbounded reveal this task removes — so the prefix boundary moves
        FORWARD to a fresh tail window instead, the same shape a session
        load produces. The dropped scroll-back stays in the store and
        rehydrates chunk-by-chunk if the reader scrolls up again.
        """
        self._hide_tail_from(None)
        # A jump-to-latest between a re-center and its placement supersedes
        # the placement; do not leave upward hydration suppressed (review E).
        self._suppress_boundary_hydration = False
        if self._windowing_enabled():
            self._set_hidden_prefix(
                self._tail_window_start(
                    self._messages,
                    line_budget=self._initial_window_line_budget(),
                )
            )

    def anchor(self, *args, **kwargs) -> None:
        """Engage tail-follow; a hidden tail is revealed first.

        Anchoring means "follow the newest content", which requires the
        newest content to be mounted. ``jump_to_latest`` and the send path
        handle their windows explicitly before anchoring; this override is
        the safety net for every other caller (TASK-15777).
        """
        if self._hidden_tail_ids:
            self._reveal_tail_window()
            if self.is_mounted:
                self.call_later(self.refresh_messages)
        super().anchor(*args, **kwargs)

    def scroll_end(self, *args, **kwargs) -> None:
        """Jump to the bottom; with a hidden tail, plant the drain's first link.

        Review A, round 2: ``Widget.scroll_end`` (the End key) re-engages
        the raw anchor synchronously, but its actual scroll can be
        superseded by the compositor's anchor path, which moves an anchored
        widget WITHOUT firing the ``scroll_y`` watcher (measured: a pill
        display toggle between End and the deferred scroll left
        ``watch_scroll_y`` completely silent across the 32->580 jump). Every
        scroll-EVENT hook (wheel, PageDown, the watcher) can therefore miss
        this entry entirely, so the End action itself schedules the first
        tailward chunk; from there the ``_hydrate_tailward`` self-chain
        carries the drain on the raw ANCHOR STATE, which needs no scroll
        events. Callers that already handled the window (``jump_to_latest``
        via ``anchor()``, the prune's following-branch restore) reach here
        with no hidden tail, making this a no-op for them.

        TASK-16851: the intent stamp lets a prune whose entry-capture
        predates this call recognize the anchor engagement as the user's
        End rather than the shrink-clamp's re-attach (see
        ``_run_prune_check``'s restore).
        """
        self._scroll_end_intent_time = monotonic()
        super().scroll_end(*args, **kwargs)
        if self._hidden_tail_ids:
            self._schedule_tailward_hydration()

    def _schedule_scrollback_hydration(self) -> None:
        """Coalesce one lazy prepend after explicit detached upward scrolling.

        TASK-15455 (reconciliation): automatic hydration is additionally
        refused once the mounted height reaches the LOW watermark. Without
        that gate, hydration and the watermark walk chase each other forever
        — measured on a 180-message transcript with a 45/70 configuration and
        the reader at the boundary: the hidden prefix oscillated between 169
        and 152 messages (height 47 <-> 115) across every idle frame, because
        the prune's own scroll restoration lands back at the boundary and
        re-triggers the hydration that produced it.

        The gate makes the loop impossible rather than unlikely: the walk only
        fires ABOVE the high mark and always leaves the remainder ABOVE the low
        mark, so a prune can never restore a hydratable state. An explicit
        ``_hydrate_scrollback()`` call is deliberately NOT gated — a caller
        asking for one chunk is not a loop.

        TASK-15777: with sane watermarks the refusal is replaced by two-sided
        windowing — hydration proceeds past the low mark and the tail is
        trimmed back into the hidden suffix instead (see ``_two_sided_active``
        for why the fixed point still holds). The refusal remains for the
        kill switch and degenerate watermark configurations.
        """
        if (
            self._scrollback_hydration_scheduled
            or self._hydrating_scrollback
            or self._suppress_boundary_hydration
            or not self.is_mounted
            or self._is_following_tail()
            or self._first_visible_message_index() <= 0
        ):
            return
        low_mark, high_mark = self._prune_watermarks()
        if (
            high_mark > 0
            and self.virtual_size.height >= low_mark
            and not self._two_sided_active()
        ):
            return
        self._scrollback_hydration_scheduled = True
        self.call_later(self._hydrate_scrollback)

    async def _hydrate_scrollback(self) -> None:
        """Prepend one earlier chunk while keeping the current content fixed."""
        self._scrollback_hydration_scheduled = False
        if (
            self._hydrating_scrollback
            or not self.is_mounted
            or self._is_following_tail()
        ):
            return
        current_start = self._first_visible_message_index()
        if current_start <= 0:
            return
        next_start = self._tail_window_start(
            self._messages,
            end=current_start,
            line_budget=self._scrollback_chunk_line_budget(),
        )
        if next_start >= current_start:
            next_start = current_start - 1

        previous_height = self.virtual_size.height
        previous_scroll_y = float(self.scroll_y)
        self._hydrating_scrollback = True
        self._set_hidden_prefix(next_start)
        try:
            async with self._refresh_lock:
                await self._reconcile_rows(self._transcript_rows())
        except Exception:
            self._hydrating_scrollback = False
            raise

        def _restore_reader() -> None:
            try:
                if not self.is_mounted:
                    return
                added_height = max(0, self.virtual_size.height - previous_height)
                # Use Textual's internal switch only to avoid treating this
                # compensating scroll as a fresh user gesture.  The public
                # scroll_to() always calls release_anchor() first.
                self._scroll_to(
                    y=previous_scroll_y + added_height,
                    animate=False,
                    release_anchor=False,
                )
                # TASK-15777: two-sided windowing — after the reader is back
                # on the same content, trim the far end of the mounted slice
                # into the hidden tail so sustained scroll-back keeps the DOM
                # bounded instead of hitting the low-watermark ceiling. The
                # trim lands the MEASURED height back at ~low BEFORE the
                # prune check runs (refresh_messages schedules it), so the
                # prune (which fires only above the measured high mark)
                # cannot chase hydration — measured, not estimated, is what
                # makes that ordering an argument (review B).
                trim_start = (
                    self._compute_tail_trim_start()
                    if self._two_sided_active()
                    else None
                )
                if trim_start is not None:
                    self._hide_tail_from(trim_start)
                    self.call_later(self.refresh_messages)
                else:
                    self._schedule_prune_check()
            finally:
                self._hydrating_scrollback = False

        self.call_after_refresh(_restore_reader)

    def _compute_tail_trim_start(self) -> int | None:
        """Return the message index where the hidden tail should start, if any.

        Walks the MOUNTED rows from the newest end in MEASURED heights (the
        same ``outer_size.height`` + margin-collapse accounting the prune's
        ``_compute_prunable_prefix`` uses), keeping at least
        ``max(low_mark, scroll_y + 2 viewports)`` mounted — the low watermark
        is the same budget every other mounted state is allowed, and the
        viewport term guarantees the trim can never touch content the reader
        is looking at (or about to reach).

        Measuring is load-bearing, not a refinement (review B): the prune
        fires on measured ``virtual_size.height``, so an ESTIMATED trim only
        kept the prune away while ``measured/estimated <= high/low`` — and
        ordinary short one-line messages measure ~1.35-1.7x their estimate,
        which at tight watermark ratios produced a permanent 2-cycle where
        the prune removed exactly what each hydration added and scroll-back
        never progressed. Trimming in the prune's own units makes the fixed
        point hold for any content shape.

        Protections stop the walk (contiguity forbids skipping): the
        SELECTED message (same contract as the prune — review D: trimming it
        unmounted its action row and made ``j`` teleport) and a group whose
        row holds keyboard focus (removing a focused widget silently steals
        the user's keyboard context). While a protection blocks the trim,
        the prune still bounds total height from the top; scroll-back past
        a bottom-pinned selection stalls until the selection is cleared —
        the same stance the prune already takes when a selection blocks its
        walk. A streaming row is deliberately NOT protected here: hidden, it
        costs nothing and updates nothing; the pill and the ghost-follow
        heal bring it back.

        Returns:
            The new hidden-tail start index, or ``None`` when nothing should
            be trimmed.
        """
        first = self._first_visible_message_index()
        tail_start = self._hidden_tail_start_index()
        if tail_start - first <= 1:
            return None
        low_mark, high_mark = self._prune_watermarks()
        if high_mark <= 0:
            return None
        keep_floor = max(
            low_mark,
            int(self.scroll_y) + 2 * self._window_viewport_height(),
        )
        remaining = self.virtual_size.height
        if remaining <= keep_floor:
            return None
        groups = self._measured_message_groups()
        if len(groups) <= 1:
            return None
        protected_ids: set[str] = set()
        ownership = self._ownership_by_message_id()
        if self.selected_message_id is not None:
            protected_ids.add(
                ownership.get(
                    self.selected_message_id,
                    (self.selected_message_id, (self.selected_message_id,)),
                )[0]
            )
        focused_id = self._focused_row_message_id()
        if focused_id is not None:
            protected_ids.add(ownership.get(focused_id, (focused_id, (focused_id,)))[0])
        index_by_id = {
            message.id: index for index, message in enumerate(self._messages)
        }
        trim_start_id: str | None = None
        blocking_id: str | None = None
        # Never trim the topmost group: the window must stay non-empty.
        for message_id, group_height in reversed(groups[1:]):
            if message_id in protected_ids:
                blocking_id = message_id
                break
            if remaining - group_height <= keep_floor:
                break
            remaining -= group_height
            trim_start_id = message_id
        if trim_start_id is None:
            if blocking_id is not None and remaining > keep_floor:
                # Mirror of the prune's blocked-walk log: without it a
                # paused slide (selection or focus pinned at the mounted
                # bottom) is indistinguishable from the trim simply having
                # nothing to do.
                logger.debug(
                    "Console transcript tail trim blocked: mounted height "
                    f"{remaining} over keep floor {keep_floor} but message "
                    f"{blocking_id!r} (selected or focused) pins the newest "
                    "end of the slice"
                )
            return None
        return index_by_id.get(trim_start_id)

    def _measured_message_groups(self) -> list[tuple[str, int]]:
        """Return measured ``(message_id, height)`` per mounted group, top-down.

        The per-row accounting mirrors ``_compute_prunable_prefix`` (outer
        size + collapsed vertical margins), scoped per message group. The
        walk ends at the first non-message child (trailing end-rule, empty
        panel, docked jump pill), exactly like the prune's walk.

        Latent mirror divergence, deliberate and currently exact (review
        round 2): the prune's walk carries ``group_height``/``group_margin``
        cumulatively ACROSS group boundaries, so a group's first row
        collapses against the previous group's trailing margin; this walk
        resets both per group, skipping that inter-group collapse — a
        per-boundary difference of ``min(prev_bottom, top)``. Today every
        transcript row has zero vertical margin, so the sum matches the
        measured virtual height exactly (probe: constant -3 delta = the
        three non-message chrome rows, at any group count) and the sign is
        conservative. If a row style ever gains a vertical margin, this
        walk will overcount each boundary by that collapse amount — still
        conservative (trims slightly less), but worth knowing.
        """
        key_by_widget_id = {
            id(widget): key for key, widget in self._row_widgets.items()
        }
        message_ids = {message.id for message in self._messages}
        ownership = self._ownership_by_message_id()
        groups: list[tuple[str, int]] = []
        group_id: str | None = None
        group_height = 0
        group_margin = 0
        for child in self.children:
            key = key_by_widget_id.get(id(child))
            row_message_id: str | None = None
            if key is not None and ":" in key:
                candidate = key.split(":", 1)[1]
                if candidate in message_ids:
                    row_message_id = ownership.get(
                        candidate, (candidate, (candidate,))
                    )[0]
            if row_message_id is None:
                break
            if row_message_id != group_id:
                if group_id is not None:
                    groups.append((group_id, group_height))
                group_id = row_message_id
                group_height = 0
                group_margin = 0
            if not child.display:
                continue
            top, _, bottom, _ = child.styles.margin
            group_height = (
                (group_height - group_margin + max(group_margin, top))
                + bottom
                + child.outer_size.height
            )
            group_margin = bottom
        if group_id is not None:
            groups.append((group_id, group_height))
        return groups

    def _focused_row_message_id(self) -> str | None:
        """Return the message id of the row group holding keyboard focus."""
        try:
            focused = self.app.focused
        except NoActiveAppError:
            return None
        if focused is None:
            return None
        node = focused
        while node is not None and node.parent is not self:
            node = node.parent
        if node is None:
            return None
        key = next(
            (key for key, widget in self._row_widgets.items() if widget is node),
            None,
        )
        if key is None or ":" not in key:
            return None
        candidate = key.split(":", 1)[1]
        if any(message.id == candidate for message in self._messages):
            return candidate
        return None

    def _schedule_tailward_hydration(self) -> None:
        """Coalesce one downward reveal after reaching the bottom boundary.

        TASK-15777: the counterpart of ``_schedule_scrollback_hydration`` for
        the hidden tail — without it, everything the tail-trim hid would only
        be reachable via the jump pill or a send, a new reachability hole in
        the other direction.
        """
        if (
            self._tailward_hydration_scheduled
            or self._hydrating_scrollback
            or not self.is_mounted
            or not self._hidden_tail_ids
        ):
            return
        self._tailward_hydration_scheduled = True
        self.call_later(self._hydrate_tailward)

    async def _hydrate_tailward(self) -> None:
        """Reveal one newer chunk from the hidden tail below the reader.

        No scroll compensation is needed: revealed rows mount BELOW the
        reader, which does not move content above them. Growth on this end
        is bounded by the existing prefix prune (over the high mark it trims
        the oldest rows, with its own measured scroll compensation), so an
        up-then-down round trip oscillates between the two marks instead of
        accumulating.

        TASK-16851: that bound is real only while the prune can actually
        make room. A far jump SELECTS its target and lands it at the window
        HEAD, and the prune's walk stops at the first protected group — so a
        head-pinned selection blocks the prune entirely while this chain
        kept revealing (round-3 review: 490 rows / height 2.18x the high
        mark, growing with session length). Hydration must not outrun a
        prune that cannot make room: while the measured height is at/over
        the high mark AND the prune walk is blocked, the reveal is refused
        and the walk stalls BOUNDED instead (the mirror of the trim's
        blocked-by-selection pause on the other boundary — the eviction
        alternative would unmount the selection's action row, review D's
        teleport). Clearing the selection (Esc) or the jump pill restores
        full downward reachability.
        """
        self._tailward_hydration_scheduled = False
        if (
            self._hydrating_scrollback
            or not self.is_mounted
            or not self._hidden_tail_ids
        ):
            return
        async with self._refresh_lock:
            # Re-check under the lock: the guards above ran while another
            # reconcile (a prune's, most often) could still be in flight,
            # and the refusal below walks ``self.children`` — reading them
            # mid-reconcile sees a transient order whose first child may not
            # be a message row, which makes the prune walk look blocked when
            # it is not (measured: the walk broke immediately and stalled a
            # selection-free drain).
            if not self.is_mounted or not self._hidden_tail_ids:
                return
            low_mark, high_mark = self._prune_watermarks()
            if (
                high_mark > 0
                and self.virtual_size.height >= high_mark
                and not self._compute_prunable_prefix(
                    self.virtual_size.height, low_mark
                )[0]
            ):
                # Mirror of the prune's and the trim's blocked-walk logs: an
                # unexplained stalled downward walk must be diagnosable.
                logger.debug(
                    "Console transcript tailward hydration refused: mounted "
                    f"height {self.virtual_size.height} at/over high mark "
                    f"{high_mark} and the prune walk is blocked (a protected "
                    "group holds the window head) — hydration must not "
                    "outrun a prune that cannot make room"
                )
                return
            tail_start = self._hidden_tail_start_index()
            budget = self._scrollback_chunk_line_budget()
            used = 0
            end = tail_start
            while end < len(self._messages) and used < budget:
                used += self._estimated_message_lines(self._messages[end])
                end += 1
            self._hydrating_scrollback = True
            self._reveal_hidden_tail_through(end)
            try:
                await self._reconcile_rows(self._transcript_rows())
            finally:
                self._hydrating_scrollback = False
        self._schedule_prune_check()
        # Chain while the reader is still pinned to the bottom: a reader whose
        # anchor re-engaged at the slice bottom is auto-scrolled to the new
        # bottom DURING the reveal (while the in-flight latch swallows the
        # boundary signal), and once there, further scroll intents produce no
        # ``scroll_y`` change to re-fire the watcher. Re-checking here keeps
        # the walk converging toward the true tail; a detached reader
        # mid-window is left alone (the reveal grew ``max_scroll_y`` past
        # them, so they are no longer at the boundary). Review A: the raw
        # anchor state is part of the condition because the anchor's
        # auto-scroll can land a tick AFTER this re-check reads
        # ``scroll_y`` — an anchored reader must converge regardless.
        if self._hidden_tail_ids and (
            self._at_bottom_boundary() or self._raw_anchor_engaged()
        ):
            self._schedule_tailward_hydration()

    def set_presentation_context(
        self,
        context: ConsolePresentationContext,
        *,
        force: bool = False,
    ) -> None:
        """Apply live display identity without remounting transcript rows.

        Args:
            context: Presentation values used to resolve every message row.
            force: Re-resolve mounted rows even if another sync already stored
                the same context but its deferred repaint has not run yet.
        """
        if context == self._presentation_context and not force:
            return
        self._presentation_context = context
        # Every cached signature includes the presentation revision and names.
        # Dropping the old entries forces exactly the current message rows to
        # resolve again while reconciliation keeps their widget objects.
        self._message_signature_cache.clear()
        if self.is_mounted:
            self.call_later(self.refresh_messages)

    def _message_presentation(
        self, message: ConsoleChatMessage
    ) -> ConsoleMessagePresentation:
        return resolve_console_message_presentation(message, self._presentation_context)

    def _speech_state_store(self) -> dict[str, ConsoleSpeechPresentationState]:
        """Return remount-safe screen state, or a local store in bare harnesses."""
        try:
            screen = self.screen
        except NoScreen:
            return self._speech_states
        states = getattr(screen, "_console_speech_states", None)
        if isinstance(states, dict):
            return states
        try:
            screen._console_speech_states = self._speech_states
        except Exception:
            return self._speech_states
        return self._speech_states

    def _console_speech_state(self, message_id: str) -> ConsoleSpeechPresentationState:
        state = self._speech_state_store().get(message_id)
        if state in {"idle", "generating", "playing", "stopped", "failed"}:
            return state
        if self._console_tts_speaking_message_id() == message_id:
            return "playing"
        return "idle"

    def set_speech_state(
        self,
        message_id: str,
        state: ConsoleSpeechPresentationState,
    ) -> bool:
        """Apply one ordered speech transition and reject stale terminal events."""
        if self._message_by_id(message_id) is None:
            return False
        states = self._speech_state_store()
        current = states.get(message_id, "idle")
        allowed = {
            "idle": {"generating"},
            "generating": {"playing", "stopped", "failed"},
            "playing": {"stopped", "failed"},
            "stopped": {"generating", "idle"},
            "failed": {"generating", "idle"},
        }
        if state not in allowed.get(current, set()):
            return False
        if state in {"generating", "playing"}:
            for other_id, other_state in tuple(states.items()):
                if other_id == message_id:
                    continue
                if other_state in {"generating", "playing"}:
                    states[other_id] = "stopped"
                elif other_state in {"stopped", "failed"}:
                    states.pop(other_id, None)
        if state == "idle":
            states.pop(message_id, None)
        else:
            states[message_id] = state
        self._message_signature_cache.pop(message_id, None)
        if self.is_mounted:
            self.call_later(self.refresh_messages)
        return True

    def _clear_failed_speech_states(self) -> None:
        states = self._speech_state_store()
        failed_ids = [
            message_id for message_id, state in states.items() if state == "failed"
        ]
        for message_id in failed_ids:
            states.pop(message_id, None)
            self._message_signature_cache.pop(message_id, None)

    def set_messages(
        self,
        messages: Iterable[ConsoleChatMessage],
        *,
        session_id: object = _SESSION_ID_UNSET,
    ) -> None:
        """Replace transcript messages and refresh mounted rows when possible.

        Args:
            messages: New transcript messages in display order. Signature
                cache entries for messages no longer present are pruned here
                (delete correctness for the TASK-259 per-message cache).
            session_id: Optional owning-session identity. A change between
                explicit identities clears disclosure expansion even when a
                new session recycles message ids. Omitting it preserves the
                legacy id-intersection behavior.
        """
        previous_thinking_refs = self._thinking_activity_refs
        previous_activity_ids = {
            message.id
            for message in self._messages
            if message.role is ConsoleMessageRole.TOOL
        }
        previous_visible_ids = [
            message.id
            for message in self._messages
            if message.id not in self._pruned_message_ids
            and message.id not in self._hidden_tail_ids
        ]
        previous_hidden_tail_ids = self._hidden_tail_ids
        self._messages = list(messages)
        units = group_console_transcript_messages(self._messages)
        self._unit_spans_by_index = self._build_unit_spans(self._messages, units)
        message_ids = {message.id for message in self._messages}
        session_changed = False
        self._fork_eligibility_by_message_id = {
            message_id: eligibility
            for message_id, eligibility in self._fork_eligibility_by_message_id.items()
            if message_id in message_ids
        }
        if session_id is not _SESSION_ID_UNSET:
            if (
                self._session_identity is not _SESSION_ID_UNSET
                and self._session_identity != session_id
            ):
                session_changed = True
                self._expanded_tool_output_ids.clear()
                self._pending_thinking_auto_collapse.clear()
                self._manual_thinking_disclosures.clear()
                self._closed_live_thinking_blocks.clear()
            self._session_identity = session_id
        thinking_refs: dict[str, ConsoleThinkingActivityRef] = {}
        current_thinking_blocks: set[tuple[str, str]] = set()
        for unit in units:
            turn = unit.assistant_turn
            if turn is None:
                continue
            live_block_id = self._live_thinking_block_id(turn)
            envelope = turn.assistant.thinking
            if isinstance(envelope, ThinkingEnvelope):
                current_thinking_blocks.update(
                    (turn.assistant.id, block.block_id) for block in envelope.blocks
                )
            if live_block_id is not None and any(
                activity.id not in previous_activity_ids
                for activity in turn.activities
            ):
                self._closed_live_thinking_blocks.add(
                    (turn.assistant.id, live_block_id)
                )
                live_block_id = None
            for ref in project_thinking_activities(
                assistant=turn.assistant,
                live_block_id=live_block_id,
            ):
                thinking_refs[ref.activity_id] = ref
                is_live = ref.block_id == live_block_id
                if (
                    is_live
                    and ref.activity_id not in previous_thinking_refs
                    and not session_changed
                ):
                    self._expanded_tool_output_ids.add(ref.activity_id)
                    self._pending_thinking_auto_collapse.add(ref.activity_id)
                elif (
                    not is_live
                    and ref.activity_id in self._pending_thinking_auto_collapse
                ):
                    if ref.activity_id not in self._manual_thinking_disclosures:
                        self._expanded_tool_output_ids.discard(ref.activity_id)
                    self._pending_thinking_auto_collapse.discard(ref.activity_id)
        self._thinking_activity_refs = thinking_refs
        self._closed_live_thinking_blocks &= current_thinking_blocks
        thinking_ids = set(thinking_refs)
        self._pending_thinking_auto_collapse &= thinking_ids
        self._manual_thinking_disclosures &= thinking_ids
        # Expansion is per message id, so ids that left the transcript (a
        # session switch, a deleted branch) must go with them -- otherwise the
        # set grows for the life of the widget and a recycled id would come
        # back already expanded.
        self._expanded_tool_output_ids &= message_ids | thinking_ids
        # TASK-15455: preserve the current contiguous window across streaming
        # updates by anchoring it to the first still-present visible id.  A
        # disjoint session switch has no such id and starts from a bounded tail
        # before any Markdown signature/widget is built.
        index_by_id = {
            message.id: index for index, message in enumerate(self._messages)
        }
        preserved_indices = [
            index_by_id[message_id]
            for message_id in previous_visible_ids
            if message_id in index_by_id
        ]
        preserved_start = min(preserved_indices) if preserved_indices else None
        # TASK-15777: the hidden tail is sticky across streaming ingests,
        # anchored to its first still-present id — otherwise every 0.2s sync
        # tick would remount the trimmed tail under a scrolled-back reader.
        # Because the suffix is derived from ONE index over the new list,
        # ids appended at the end (a streamed reply while the reader is deep
        # in scroll-back) join it automatically. A disjoint ingest (session
        # switch) has no surviving id and clears it.
        surviving_tail_indices = [
            index_by_id[message_id]
            for message_id in previous_hidden_tail_ids
            if message_id in index_by_id
        ]
        self._hide_tail_from(
            min(surviving_tail_indices) if surviving_tail_indices else None
        )
        if self._hidden_tail_ids and self._raw_anchor_engaged():
            # Ghost-follow heal (review A): Textual's anchor re-engaged
            # WITHOUT anchor() (End key / _check_anchor), so the reader is
            # pinned to the bottom of a stale slice while replies pile into
            # the hidden tail — and a reply WITHOUT a new user message never
            # takes the send branch below. Following means the newest content
            # must mount: drop the suffix and re-window onto a fresh tail
            # (``preserved_start = None`` routes the boundary computation
            # below through the same fresh-tail path a session load uses).
            self._hide_tail_from(None)
            preserved_start = None
        if not self._windowing_enabled():
            # TASK-15455 (reconciliation): `[chat_defaults]
            # transcript_window_lines = 0` mounts the whole history, the
            # behaviour that shipped before this task. The escape hatch is the
            # point: a windowing bug must be switchable off without a release.
            #
            # It must still leave the WATERMARK prune alone. Forcing 0 here
            # cleared `_pruned_message_ids` on every ingest, so an
            # over-watermark session re-mounted its entire history on each
            # 0.2s sync tick and pruned it back down again (measured: 180 rows
            # remounted, settled to 11, every tick). Pre-task code kept pruning
            # sticky across ingests (`_pruned_message_ids &= message_ids`);
            # carrying the preserved boundary forward reproduces that, while a
            # fresh or disjoint ingest still starts at 0 = mount everything.
            #
            # Review C: the kill switch must clear the hidden TAIL too — the
            # escape hatch exists so a windowing bug can be switched off
            # without a release, and carrying the sticky suffix forward left
            # the trimmed tail hidden forever after a mid-session flip. Safe
            # against the 15458 per-tick churn: no tail-creating path runs
            # with windowing off, so this clear is one-shot, and the
            # watermark-pruned PREFIX stays sticky exactly as before.
            self._hide_tail_from(None)
            window_start = 0 if preserved_start is None else preserved_start
        elif preserved_start is None:
            window_start = self._tail_window_start(
                self._messages,
                line_budget=self._initial_window_line_budget(),
            )
        else:
            window_start = preserved_start
        new_user_send = any(
            message.id not in self._seen_message_ids
            and message.role == ConsoleMessageRole.USER
            for message in self._messages
        )
        if (
            self.is_mounted
            and new_user_send
            and self._follow_intent_time >= self._user_scroll_time
        ):
            # A send: jump to the tail even if the user had scrolled up
            # (anchor() also re-engages follow for the reply that streams
            # in next). Checked against ALL newly-seen ids, not just the
            # tail -- the send path appends USER + ASSISTANT placeholder
            # together and the first polled update can already have the
            # placeholder at the tail. Appended assistant/tool rows alone
            # never yank a reader. TASK-336: a user scroll AFTER the
            # send/resume intent wins — the coalesced sync can deliver this
            # anchor late, and it must not yank a reader who has already
            # scrolled back.
            #
            # TASK-15777: a send from deep scroll-back re-windows onto a
            # fresh bounded tail. Keeping the preserved (far-back) window
            # start while clearing the hidden tail would remount everything
            # from the reader's position to the tail in one pass. The
            # ``anchor()`` override also clears the suffix, but the boundary
            # it sets would be overwritten by ``_set_hidden_prefix`` below —
            # hence the explicit ``window_start`` recompute here.
            if self._hidden_tail_ids and self._windowing_enabled():
                self._hide_tail_from(None)
                window_start = self._tail_window_start(
                    self._messages,
                    line_budget=self._initial_window_line_budget(),
                )
            self.anchor()
        self._seen_message_ids = message_ids
        # task-501: apply a swipe-handoff selection once its id is actually in
        # the ingested set (see ``pending_selection_id``); checked BEFORE the
        # clear below so a swipe that removed the old selection lands directly
        # on the swiped-to sibling instead of clearing to None.
        if (
            self.pending_selection_id is not None
            and self.pending_selection_id in message_ids
        ):
            self.selected_message_id = self.pending_selection_id
            pending_index = index_by_id[self.pending_selection_id]
            if pending_index < window_start:
                # Branch sibling handoff: the replacement id may sit exactly
                # where the old window boundary id disappeared.  Keep the new
                # selected row in the window rather than mounting a disjoint
                # orphan or clearing a valid selection.
                window_start = self._turn_aligned_start(self._messages, pending_index)
            elif pending_index >= self._hidden_tail_start_index():
                # TASK-15777: same contract on the other boundary — a
                # handed-off selection inside the hidden tail extends the
                # mounted slice down through it.
                self._reveal_hidden_tail_through(pending_index + 1)
            self.pending_selection_id = None
        if (
            self._hidden_tail_start is not None
            and self._hidden_tail_start <= window_start
        ):
            # Degenerate reorder: the two boundaries crossed. Mounting through
            # the tail is always safe; a crossed window never is.
            self._hide_tail_from(None)
        self._set_hidden_prefix(window_start)
        if self.selected_message_id not in message_ids | thinking_ids:
            self.selected_message_id = None
        for stale_id in [
            cached_id
            for cached_id in self._message_signature_cache
            if cached_id not in message_ids
        ]:
            del self._message_signature_cache[stale_id]
            self._signature_compute_counts.pop(stale_id, None)

    def _live_thinking_block_id(self, turn: ConsoleAssistantTurn) -> str | None:
        """Return the current unbounded block, using visible turn boundaries."""
        assistant = turn.assistant
        envelope = assistant.thinking
        if (
            assistant.status not in _IN_FLIGHT_MESSAGE_STATUSES
            or not isinstance(envelope, ThinkingEnvelope)
            or not envelope.blocks
        ):
            return None
        answer = (
            assistant.variants.current.content
            if assistant.variants is not None
            else assistant.content
        )
        if answer:
            return None
        current = envelope.blocks[-1]
        if (assistant.id, current.block_id) in self._closed_live_thinking_blocks:
            return None
        if any(
            activity.activity_round_ordinal == current.round_ordinal
            for activity in turn.activities
        ):
            return None
        return current.block_id

    def set_model_thinking_visible(self, visible: bool) -> bool:
        """Apply the presentation-only thinking gate without replacing turns."""

        visible = bool(visible)
        if visible == self._show_model_thinking:
            return False
        self._show_model_thinking = visible
        thinking_ids = set(self._thinking_activity_refs)
        if visible:
            self._expanded_tool_output_ids.update(
                self._pending_thinking_auto_collapse & thinking_ids
            )
        else:
            self._expanded_tool_output_ids.difference_update(thinking_ids)
            self._manual_thinking_disclosures.difference_update(thinking_ids)
            if self.selected_message_id in thinking_ids:
                self.selected_message_id = None
        if self.is_mounted:
            self.call_later(self.refresh_messages)
        return True

    def set_image_specs(self, specs: Mapping[str, ConsoleImageRowSpec]) -> None:
        """Replace the prebuilt inline-image row payloads keyed by message ID.

        Args:
            specs: Mapping of message ID to its prepared image-row payload.
                Messages absent from the mapping render no image row (covers
                hidden mode, unprepared cache, and metadata-only messages).
        """
        self._image_specs = dict(specs)

    def set_generation_card_specs(
        self, specs: Mapping[str, ConsoleGenerationCardSpec]
    ) -> None:
        """Replace the prebuilt image-generation card row payloads keyed by message ID.

        Args:
            specs: Mapping of message ID to its prepared generation-card
                payload. A message id present here renders a
                ``"generation-card"`` row INSTEAD of any ``"image"`` row
                for that same message (mutually exclusive per message id --
                see ``_transcript_rows``). Messages absent from the mapping
                render no card row (covers non-generation messages and a
                generation message in hidden view mode).
        """
        self._generation_card_specs = dict(specs)

    def set_video_card_specs(self, specs: Mapping[str, ConsoleVideoCardSpec]) -> None:
        """Replace the prebuilt video-generation card row payloads keyed by message ID.

        Args:
            specs: Mapping of message ID to its prepared video-card payload.
                A message id present here renders a ``"video-card"`` row
                INSTEAD of any image/generation-card row for that same
                message (mutually exclusive per message id; a video message
                never has attachments -- ADR-044). Messages absent from the
                mapping render no video row.
        """
        self._video_card_specs = dict(specs)

    def set_fork_eligibilities(
        self, eligibilities: Mapping[str, ConsoleForkEligibility]
    ) -> None:
        """Replace frozen store-owned Fork eligibility by native message ID."""
        self._fork_eligibility_by_message_id = {
            message_id: eligibility
            for message_id, eligibility in eligibilities.items()
            if isinstance(message_id, str)
            and isinstance(eligibility, ConsoleForkEligibility)
        }

    def set_summary_boundary(self, message_id: str | None) -> None:
        """Set the `/rewind` summary boundary message id for the banner.

        The banner is render-derived: ``_transcript_rows`` emits it above the
        matching message when it is present. Refresh is driven by the screen's
        sync path (which folds this id into its refresh key), matching
        ``set_image_specs``; standalone callers/tests refresh explicitly.
        """
        self.summary_boundary_message_id = message_id

    def set_original_attempt_previews(self, previews: Mapping[str, str]) -> None:
        """Replace screen-owned visible original-attempt preview copies."""
        self._original_attempt_previews = dict(previews)

    def set_citation_counts(self, counts: Mapping[str, int]) -> None:
        """Replace screen-owned citation counts keyed by native message ID."""
        self._citation_counts = {
            message_id: count
            for message_id, count in counts.items()
            if isinstance(message_id, str)
            and message_id
            and type(count) is int
            and count > 0
        }

    def set_annotation_previews(self, previews: Mapping[str, tuple[str, ...]]) -> None:
        """Replace screen-owned review-note previews keyed by native message ID.

        task-17169 slice 2: the screen's sync loop pushes this every tick
        (the citation-counts pattern); entries without an id or without at
        least one note are dropped so the row derivation below can treat
        presence as "render a marker".
        """
        self._annotation_previews = {
            message_id: notes
            for message_id, notes in previews.items()
            if isinstance(message_id, str) and message_id and notes
        }

    def set_change_review_provider_factory(
        self, factory: Callable[[], Any] | None
    ) -> None:
        """Update the change-summary turn-file-card's provider factory.

        Screen-owned (mirrors ``set_summary_boundary``/``set_image_specs``):
        the screen's sync loop keeps this current on the mounted instance
        every tick, so a session switch or a bridge becoming available never
        needs a fresh transcript instance to take effect.

        Args:
            factory: Zero-arg callable yielding a change-review provider
                for the active session (may return ``None`` when no run
                is reviewable), or ``None`` to render plain marker rows.
        """
        self._change_review_provider_factory = factory

    def sync_empty_state(
        self,
        card_state: ConsoleSetupCardState,
        *,
        provider_action_label: str = "",
        provider_action_tooltip: str = "",
    ) -> None:
        """Refresh the empty transcript state while preserving message exports.

        TASK-2154.8 (FR-03): an empty ``provider_action_label`` now means "no
        recovery action to offer" (provider ready) and is stored as-is; the
        empty panel only renders the action button for a non-empty label.
        """
        next_card_state = _coerce_card_state(card_state)
        next_action_label = provider_action_label.strip()
        next_action_tooltip = provider_action_tooltip.strip()
        if (
            self._empty_card_state == next_card_state
            and self.empty_state_action_label == next_action_label
            and self.empty_state_action_tooltip == next_action_tooltip
        ):
            return
        self._empty_card_state = next_card_state
        self.empty_state_action_label = next_action_label
        self.empty_state_action_tooltip = next_action_tooltip
        if self.is_mounted and not self._messages:
            self.call_later(self.refresh_messages)

    async def refresh_messages(self) -> None:
        """Reconcile mounted message rows from the current transcript state."""
        menus = message_more_menus_on_screen(self.screen) if self.is_mounted else []
        opener_id = menus[0].opener_button_id if menus else ""
        await self.dismiss_message_more_menu(restore_focus=False)
        async with self._refresh_lock:
            await self._reconcile_rows(self._transcript_rows())
        if menus:
            self._restore_message_action_focus(opener_id)
        # TASK-15777: a re-centered far jump replaced the whole window, so the
        # previous scroll offset points at arbitrary content — put the jump
        # target at the top of the viewport once its row has a layout.
        target_id = self._reveal_scroll_target
        if target_id is not None:
            self._reveal_scroll_target = None
            target_widget = self._row_widgets.get(
                f"assistant-turn:{target_id}"
            ) or self._row_widgets.get(f"message:{target_id}")
            if target_widget is not None:
                self.call_after_refresh(
                    self._scroll_reveal_target_into_view, target_widget
                )
            else:
                # No row to place: release the review-E latch here, since
                # the placement callback that normally releases it will
                # never run.
                self._suppress_boundary_hydration = False
        self._schedule_prune_check()

    def _scroll_reveal_target_into_view(self, widget: Widget) -> None:
        """Scroll a just-revealed jump target to the top of the viewport."""
        try:
            if not self.is_mounted or widget.parent is not self:
                return
            # Textual's internal switch: this is a programmatic placement,
            # not a fresh user gesture (the jump already released the
            # anchor).
            self._scroll_to(
                y=max(0.0, float(widget.virtual_region.y) - 1.0),
                animate=False,
                release_anchor=False,
            )
        finally:
            # The placement has landed (the watcher for it ran synchronously
            # inside the _scroll_to above); boundary hydration is the
            # reader's again.
            self._suppress_boundary_hydration = False

    def _schedule_prune_check(self) -> None:
        """Run one watermark pruning check after the pending refresh settles.

        Heights are only meaningful once layout has run, so the check is
        deferred with ``call_after_refresh`` rather than firing on a timer or
        mid-reconcile. Coalesced: at most one check is ever queued.
        """
        if self._prune_check_scheduled or not self.is_mounted:
            return
        self._prune_check_scheduled = True
        self.call_after_refresh(self._run_prune_check)

    def _prune_watermarks(self) -> tuple[int, int]:
        """Return the configured ``(low_mark, high_mark)`` for this transcript."""
        try:
            app_config = getattr(self.app, "app_config", None)
        except NoActiveAppError:
            app_config = None
        return get_console_prune_watermarks(app_config)

    def _assistant_markdown_enabled(self) -> bool:
        """Return the ``[chat_defaults] assistant_markdown`` toggle (TASK-1990)."""
        try:
            app_config = getattr(self.app, "app_config", None)
        except NoActiveAppError:
            app_config = None
        return get_console_assistant_markdown(app_config)

    async def _run_prune_check(self) -> None:
        """Drop the oldest message rows when virtual height exceeds the marks.

        View-only: ids are added to ``_pruned_message_ids`` and the regular
        reconciliation pass removes the now-undesired rows, so the store and
        ``_messages`` keep the full history. The scroll position is preserved
        for a scrolled-up reader and re-anchored when following the tail.
        """
        self._prune_check_scheduled = False
        if not self.is_mounted:
            return
        low_mark, high_mark = self._prune_watermarks()
        if high_mark <= 0:
            return
        total_height = self.virtual_size.height
        if total_height <= high_mark:
            return
        prune_ids, estimated_height = self._compute_prunable_prefix(
            total_height, low_mark
        )
        if not prune_ids:
            if total_height > high_mark:
                logger.debug(
                    "Console transcript over high watermark "
                    f"({total_height} > {high_mark}) but no prunable prefix: "
                    "a protected or non-message row is blocking the walk"
                )
            return
        logger.debug(
            f"Pruning {len(prune_ids)} console transcript messages "
            f"(estimated height {estimated_height})"
        )
        following = self._is_following_tail()
        raw_anchor_at_entry = self._raw_anchor_engaged()
        entry_time = monotonic()
        anchor_y = self.scroll_y
        self._pruned_message_ids.update(prune_ids)
        logger.info(
            f"Pruned {len(prune_ids)} oldest Console transcript message(s) "
            f"(virtual height {total_height} over high mark {high_mark})"
        )
        async with self._refresh_lock:
            await self._reconcile_rows(self._transcript_rows())

        def _restore_scroll() -> None:
            if not self.is_mounted:
                return
            if following:
                self.anchor()
                self.scroll_end(animate=False)
            else:
                # Restore the reader's state FAITHFULLY, in two parts.
                #
                # Anchor state (review round 2's blocker): the shrink from
                # the reconcile CLAMPS scroll_y to the new maximum, and if
                # the reader happened to sit at the bottom, Textual's
                # `_check_anchor` silently re-engages the raw anchor at
                # that clamp — before this callback runs. The old public
                # `scroll_to` released the anchor as a side effect, which
                # accidentally undid that for detached readers but ALSO
                # disarmed the End-drain's convergence braces (raw anchor
                # engaged + hidden tail), stalling the drain mid-history
                # forever. So: restore the anchor state captured at prune
                # entry — quietly re-release a detached reader the clamp
                # re-attached (no user-intent stamp: a programmatic shift
                # must never outrank a later send's follow intent,
                # TASK-336), and leave an entry-engaged anchor engaged so
                # the drain keeps converging.
                if not raw_anchor_at_entry and self._raw_anchor_engaged():
                    if self._scroll_end_intent_time > entry_time:
                        # TASK-16851 (the round-3 residual): this engagement
                        # is a user End that landed INSIDE the entry->restore
                        # window, not the shrink-clamp's re-attach — its
                        # deferred scroll was enqueued before this callback,
                        # so quietly releasing here would cancel the drain
                        # after ~one chunk (the pill stayed up and a second
                        # End resumed). Honor it instead: keep the anchor,
                        # skip the now-stale entry-offset compensation (the
                        # user asked for the bottom, not their old position),
                        # and re-arm the drain's self-chain.
                        if self._hidden_tail_ids:
                            self._schedule_tailward_hydration()
                        return
                    self._release_anchor_quietly()
                # Content: keep the same rows in view by shifting the
                # offset up by the height actually removed (measured, not
                # estimated), via Textual's internal switch — this is a
                # COMPENSATION, not a user gesture, exactly like the
                # hydration restore above. Ordering matters: release first,
                # or the still-engaged anchor pulls the compensating scroll
                # back to the bottom.
                removed = total_height - self.virtual_size.height
                self._scroll_to(
                    y=max(0.0, anchor_y - removed),
                    animate=False,
                    release_anchor=False,
                )

        self.call_after_refresh(_restore_scroll)

    def _compute_prunable_prefix(
        self, total_height: int, low_mark: int
    ) -> tuple[list[str], int]:
        """Walk mounted rows top-down and pick oldest whole messages to drop.

        Rows of one message (rule, body, citations, image, actions, ...) are
        contiguous, so they are committed as a group: a message is either
        fully pruned or fully kept. The walk stops at the first protected
        group (the in-progress streaming message or the selected message) or
        when dropping another group would push the remaining height below
        ``low_mark`` -- pruning is a prefix removal that errs on keeping
        content. Margin-collapse math mirrors the legacy chat-log pruning.

        Args:
            total_height: Current virtual height of the transcript.
            low_mark: Target remaining height after pruning.

        Returns:
            Tuple of ``(message_ids, prune_height)``.
        """
        key_by_widget_id = {
            id(widget): key for key, widget in self._row_widgets.items()
        }
        message_ids = {message.id for message in self._messages}
        ownership = self._ownership_by_message_id()
        protected_ids = {
            ownership.get(message.id, (message.id, (message.id,)))[0]
            for message in self._messages
            if message.status == "streaming"
        }
        if self.selected_message_id is not None:
            protected_ids.add(
                ownership.get(
                    self.selected_message_id,
                    (self.selected_message_id, (self.selected_message_id,)),
                )[0]
            )

        prune_ids: list[str] = []
        prune_height = 0
        bottom_margin = 0
        group_id: str | None = None
        group_height = 0
        group_margin = 0

        def _try_close_group() -> bool:
            """Commit the finished group, or return False to stop the walk."""
            nonlocal prune_height, bottom_margin
            if group_id is None or group_id in protected_ids:
                return False
            if total_height - group_height <= low_mark:
                return False
            prune_height = group_height
            bottom_margin = group_margin
            prune_ids.extend(ownership.get(group_id, (group_id, (group_id,)))[1])
            return True

        for child in self.children:
            key = key_by_widget_id.get(id(child))
            row_message_id: str | None = None
            if key is not None and ":" in key:
                candidate = key.split(":", 1)[1]
                if candidate in message_ids:
                    row_message_id = ownership.get(
                        candidate, (candidate, (candidate,))
                    )[0]
            if row_message_id is None:
                # The trailing end-rule, the empty panel, or the docked jump
                # pill: nothing prunable lives past a non-message row.
                break
            if row_message_id != group_id:
                if group_id is not None and not _try_close_group():
                    break
                group_id = row_message_id
                group_height = prune_height
                group_margin = bottom_margin
            if not child.display:
                # Hidden rows take no space; the group decision covers them.
                continue
            top, _, bottom, _ = child.styles.margin
            group_height = (
                (group_height - group_margin + max(group_margin, top))
                + bottom
                + child.outer_size.height
            )
            group_margin = bottom
        if group_id is not None:
            _try_close_group()
        return prune_ids, prune_height

    def row_build_counts(self) -> dict[str, int]:
        """Return row build counts for focused reconciliation tests."""
        return dict(self._row_build_counts)

    def row_render_signatures(self) -> dict[str, tuple]:
        """Return active row signatures for focused reconciliation tests."""
        return dict(self._row_signatures)

    def message_signature_compute_counts(self) -> dict[str, int]:
        """Return per-message signature derivation counts for cache tests.

        Returns:
            Mapping of message id to how many times its expensive row
            signature (render Content assembly) was derived since mount.
        """
        return dict(self._signature_compute_counts)

    def message_signature_cache_ids(self) -> tuple[str, ...]:
        """Return the message ids currently held in the signature cache.

        Returns:
            Tuple of cached message ids, for delete-pruning tests.
        """
        return tuple(self._message_signature_cache)

    def display_message(self, message_id: str) -> "ConsoleChatMessage | None":
        """Return the RENDERED row for ``message_id`` — tree node or not.

        TASK-2030: display-only TOOL markers (the ✎/⚙ rows) are never tree
        nodes, so the STORE cannot resolve them by id; the transcript's own
        display model is the authority for what the user actually selected.

        Args:
            message_id: Identifier of a rendered transcript row.

        Returns:
            The display-model message, or ``None`` when nothing rendered
            carries that id.
        """
        return self._message_by_id(message_id)

    def thinking_detail_text(self, activity_id: str) -> str | None:
        """Resolve one trusted model activity body from its owning envelope."""
        ref = self._thinking_activity_refs.get(activity_id)
        if ref is None:
            return None
        assistant = next(
            (
                message
                for message in self._messages
                if message.id == ref.assistant_message_id
            ),
            None,
        )
        envelope = assistant.thinking if assistant is not None else None
        if not isinstance(envelope, ThinkingEnvelope):
            return None
        block = next(
            (block for block in envelope.blocks if block.block_id == ref.block_id),
            None,
        )
        if isinstance(block, DisplayableThinkingBlock):
            return block.text
        if isinstance(block, ProprietaryThinkingBlock):
            return PROPRIETARY_THINKING_NOTICE
        return None

    def thinking_owner_message_id(self, activity_id: str) -> str | None:
        """Return the current Assistant owner for one projected thinking row."""
        ref = self._thinking_activity_refs.get(activity_id)
        return ref.assistant_message_id if ref is not None else None

    def _thinking_display_message(self, activity_id: str) -> ConsoleChatMessage | None:
        """Build a bounded display-only row for selection/copy/Inspector seams."""
        ref = self._thinking_activity_refs.get(activity_id)
        detail = self.thinking_detail_text(activity_id)
        if ref is None or detail is None:
            return None
        return ConsoleChatMessage(
            role=ConsoleMessageRole.TOOL,
            content=detail,
            id=activity_id,
            activity_presentation=ConsoleActivityPresentation(
                "thinking", ref.label, ref.status
            ),
        )

    def reveal_message(self, message_id: str) -> bool:
        """Extend the mounted window back through ``message_id`` when hidden.

        The single implementation behind every "jump to a message the window
        does not currently show" path: selection, the task-501 swipe handoff,
        and reading-state restore. Extending the SAME contiguous boundary is
        what keeps mounted rows one unbroken suffix of the history — no
        islands, so no gap markers are needed anywhere.

        Args:
            message_id: Identifier of the message that must have a row.

        Returns:
            True when the window moved (a refresh is required to see it),
            False when the message was already mounted or is not in this
            transcript. Deliberately does NOT refresh: callers already own a
            refresh, and the restore path needs to sequence its own.

        TASK-15777: a FAR reveal — one where the newly revealed stretch
        between the target and the current window would exceed the low
        watermark in estimated lines — re-centers the window on the target
        instead of extending the boundary, mounting an initial-window-sized
        slice from the target's turn (the same shape a session load produces)
        with everything past it in the hidden tail. Near reveals (a j/k step
        over the boundary, a nearby restore) keep the plain boundary
        extension, so small-session behavior is unchanged. Only meaningful
        under ``_two_sided_active``.
        """
        for requested_index, message in enumerate(self._messages):
            if message.id != message_id:
                continue
            index, unit_end, owner_id, _owned_ids = self._unit_span_at(
                self._messages, requested_index
            )
            first_visible = self._first_visible_message_index()
            tail_start = self._hidden_tail_start_index()
            if first_visible <= index and unit_end <= tail_start:
                return False
            if index < first_visible:
                revealed_start = self._turn_aligned_start(self._messages, index)
                revealed_end = first_visible
            else:
                revealed_start = tail_start
                revealed_end = unit_end
            if (
                self._two_sided_active()
                and self._estimated_window_lines(revealed_start, revealed_end)
                > self._prune_watermarks()[0]
            ):
                self._recenter_window_on(index, owner_id)
                return True
            if index < first_visible:
                self._set_hidden_prefix(revealed_start)
            else:
                self._reveal_hidden_tail_through(revealed_end)
            return True
        return False

    def _estimated_window_lines(self, start: int, end: int) -> int:
        """Return the estimated rendered lines of ``_messages[start:end]``."""
        return sum(
            self._estimated_message_lines(message)
            for message in self._messages[start:end]
        )

    def _recenter_window_on(self, index: int, message_id: str) -> None:
        """Mount a bounded, load-shaped window with the target's turn on top.

        The far jump detaches the reader from the tail (it IS a user
        navigation away from it — a later send's follow intent still
        outranks it, per TASK-336's ordering), and the target row is
        scrolled to the top of the viewport once mounted, because the old
        scroll offset is meaningless in the new window.
        """
        requested_index = next(
            (
                candidate_index
                for candidate_index, message in enumerate(self._messages)
                if message.id == message_id
            ),
            index,
        )
        unit_start, unit_end, owner_id, _owned_ids = self._unit_span_at(
            self._messages, requested_index
        )
        start = self._turn_aligned_start(self._messages, unit_start)
        budget = self._initial_window_line_budget()
        used = 0
        end = unit_start
        while end < len(self._messages) and used < budget:
            used += self._estimated_message_lines(self._messages[end])
            end += 1
        if end < unit_end:
            end = unit_end
        elif end > 0 and end < len(self._messages):
            _included_start, included_end, _included_owner, _included_ids = (
                self._unit_span_at(self._messages, end - 1)
            )
            end = included_end
        self._set_hidden_prefix(start)
        self._reveal_hidden_tail_through(end)
        self.release_anchor()
        self._reveal_scroll_target = owner_id
        # Review E: the reconcile that realizes this window transits an
        # emptied arrangement, and the placement parks the target near y=0 —
        # both read as top-boundary hits and hydrated one spurious chunk
        # ABOVE the jump target. Suppressed until the placement lands.
        self._suppress_boundary_hydration = True

    def select_message(self, message_id: str) -> None:
        """Select one message and show its contextual action row."""
        if self._message_by_id(message_id) is None:
            return
        if self.selected_message_id != message_id:
            self._clear_failed_speech_states()
        try:
            resolver = getattr(self.screen, "_console_fork_eligibility", None)
        except NoScreen:
            resolver = None
        if callable(resolver):
            self._fork_eligibility_by_message_id[message_id] = resolver(message_id)
        # Keep the public pre-windowing contract: callers may select any
        # message in the complete transcript model.  Reveal the contiguous
        # prefix through that turn before mounting its action row.
        if message_id not in self._thinking_activity_refs:
            self.reveal_message(message_id)
        self.selected_message_id = message_id
        if self.is_mounted:
            self.call_later(self.refresh_messages)
            self.call_later(self._notify_selection_changed)

    def toggle_message_selection(self, message_id: str) -> None:
        """Toggle one message's contextual selection state.

        Args:
            message_id: Identifier of the transcript message to select or clear.
        """
        if self._message_by_id(message_id) is None:
            return
        if self.selected_message_id == message_id:
            self.action_clear_selection()
            return
        self.select_message(message_id)

    def focus_action(self, message_id: str, action_id: str) -> None:
        """Focus a selected-message action button by message/action ID."""
        if self.selected_message_id != message_id:
            self.select_message(message_id)
        self.call_later(self._focus_action_button, message_id, action_id)

    def select_next_variant(self, message_id: str) -> None:
        """Select the next rendered variant for a message when available."""
        message = self._message_by_id(message_id)
        if (
            message is None
            or message.variants is None
            or not message.variants.can_go_next
        ):
            return
        message.variants.selected_index += 1
        if self.is_mounted:
            self.call_later(self.refresh_messages)

    def select_previous_variant(self, message_id: str) -> None:
        """Select the previous rendered variant for a message when available."""
        message = self._message_by_id(message_id)
        if (
            message is None
            or message.variants is None
            or not message.variants.can_go_previous
        ):
            return
        message.variants.selected_index -= 1
        if self.is_mounted:
            self.call_later(self.refresh_messages)

    def to_plain_text(self, width: int = 80) -> str:
        """Return an answer-oriented transcript without model thinking."""
        rule = "─" * max(1, width)
        lines: list[str] = []

        def _append_status_and_actions(message: ConsoleChatMessage, body: str) -> None:
            status_line = _message_status_line(message)
            if status_line and not _is_generating_placeholder_body(message, body):
                lines.append(status_line)
            if message.id == self.selected_message_id:
                lines.append(self._plain_action_row(message))
                lines.append(ConsoleMessageActionService().plain_action_guide(message))

        for unit in group_console_transcript_messages(self._messages):
            message = unit.standalone
            turn = unit.assistant_turn
            if message is None:
                assert turn is not None
                message = turn.assistant
            presentation = self._message_presentation(message)
            lines.append(rule)
            if message.id == self.summary_boundary_message_id:
                lines.append(CONSOLE_SUMMARY_BANNER_COPY)
            lines.append(_speaker_label(message, presentation))
            if turn is not None:
                for activity in turn.activities:
                    activity_presentation = activity.activity_presentation
                    if activity_presentation is None:
                        activity_header = "Activity · done"
                    else:
                        activity_header = (
                            f"{activity_presentation.label} · "
                            f"{activity_presentation.status}"
                        )
                    lines.append(activity_header)
                    activity_body = _message_body(activity)
                    if activity_body:
                        lines.append(activity_body)
                    _append_status_and_actions(activity, activity_body)
            body = _message_body(message, presentation)
            lines.append(body)
            _append_status_and_actions(message, body)
        if self._messages:
            lines.append(rule)
        return "\n".join(lines)

    def action_select_next(self) -> None:
        self._select_relative(1)

    def action_select_previous(self) -> None:
        self._select_relative(-1)

    def action_confirm_selection(self) -> None:
        """Select the first message or clear the current transcript selection."""
        if self.selected_message_id is not None:
            self.toggle_message_selection(self.selected_message_id)
            return
        visible = self._visible_messages()
        if visible:
            self.select_message(visible[0].id)

    def action_clear_selection(self) -> None:
        if self.selected_message_id is not None:
            self._clear_failed_speech_states()
        self.selected_message_id = None
        if self.is_mounted:
            self.call_later(self.refresh_messages)
            self.call_later(self._notify_selection_changed)
            self.call_later(self._paint_debug_dump, "after-clear-selection")

    def action_enter_text_selection(self) -> None:
        """`s` binding: arm keyboard text-selection mode.

        Thin wrapper over ``_enter_keyboard_selection`` -- Tasks 3/4 call
        the latter directly (e.g. to re-arm after a motion lands on a new
        row) without going through the binding.
        """
        self._enter_keyboard_selection()

    def action_open_review_notes(self) -> None:
        """`n` binding: request the notes modal for the selected message.

        task-18515 review-note management, task 1: a plain BINDINGS entry
        (no ``on_key`` branch -- the phase-5 probe proved printable-key
        bindings already fire while the transcript holds focus, so a
        speculative interception branch was reverted as unnecessary).
        A selection with no notes toasts instead of posting a request the
        (not-yet-built) modal has nothing to show for.
        """
        message_id = self.selected_message_id
        if message_id is not None and self._annotation_previews.get(message_id):
            self.post_message(ConsoleReviewNotesRequested(message_id))
            return
        self.notify("No review notes on this message.", severity="warning")

    def _enter_keyboard_selection(self) -> bool:
        """Arm keyboard-driven text selection on the j/k-selected row.

        Console selection phase 5 (keyboard mode skeleton). Requires an
        existing message selection (Enter) whose row implements the
        selection protocol (``ConsoleTranscriptMessage`` /
        ``ConsoleMarkdownMessage`` / ``ConsoleToolDiffRow``); anything else
        is a toast, not a mode -- there is no eligible row to arm a cursor
        on. Diff rows are addressed by a different id
        (``console-tool-diff-*``, not ``console-message-*``) and are never
        reachable through the selected MESSAGE's own row, so entry is
        effectively scoped to the message's own row; diff-row keyboard
        entry is out of scope for this phase.

        Seeds a one-character selection at the row's start (offsets
        ``[0, 1)``) through the same ``SelectionManager`` the mouse drag
        drives; Task 3 wires the motion keys that move it from there.

        Returns:
            True if the mode was entered, False if there was no eligible
            target (a toast explains why in that case).
        """
        if self.selected_message_id is None:
            self.notify(
                "Select a message first (j/k, then Enter) to select its text.",
                severity="warning",
            )
            return False
        try:
            row = self.query_one(
                f"#console-message-{self.selected_message_id}",
                (ConsoleTranscriptMessage, ConsoleMarkdownMessage, ConsoleToolDiffRow),
            )
        except QueryError:
            self.notify("This message has no selectable text.", severity="warning")
            return False
        text = row.get_display_text()
        if not text:
            self.notify("This message has no text to select.", severity="warning")
            return False
        row.scroll_visible(animate=False)
        self.selection_manager.begin_drag(row.id, 0)
        self.selection_manager.extend_drag(row.id, 1)
        row.set_selection_range(0, 1)
        self._selection_origin_row = row
        self._kb_selection_row = row
        self._kb_anchor, self._kb_end = 0, 1
        hint = self._kb_selection_hint_widget()
        if hint is not None:
            hint.update(_kb_selection_hint_text(row))
            hint.display = True
        return True

    def _exit_keyboard_selection(self, *, clear: bool = True) -> None:
        """Leave keyboard text-selection mode.

        Args:
            clear: When True (the normal Escape/mouse-takeover path), also
                clears the row's highlight and cancels the selection
                manager. False skips both -- used when the armed row has
                already been destroyed (streaming replacement, prune,
                session switch): the reconciliation guard that removed it
                (``_cancel_selection_if_row_removed``) already cancelled
                the manager, and the row itself can no longer be touched.
        """
        if clear and self._kb_selection_row is not None:
            self._kb_selection_row.clear_selection()
            self.selection_manager.cancel()
        self._kb_selection_row = None
        self._kb_anchor = None
        self._kb_end = None
        self._selection_origin_row = None
        hint = self._kb_selection_hint_widget()
        if hint is not None:
            hint.display = False

    def _kb_selection_hint_widget(self) -> Static | None:
        try:
            return self.query_one("#console-kb-selection-hint", Static)
        except NoMatches:
            return None

    def _kb_apply_motion(self, key: str) -> None:
        """Move the keyboard text-selection cursor (console selection phase 5, Task 3).

        ``key`` is the resolved motion character (``h``/``l``/``w``/``b``/
        ``0``/``$``/``j``/``k``/``o``) -- the caller (the mode's ``on_key``
        branch) has already interception-stopped the raw event and mapped
        it from ``event.character`` for printable keys.

        ``self._kb_anchor`` stays fixed (except across ``o``, which swaps
        it with the end); every motion moves ``self._kb_end``, the active
        cursor. The manager stays the single source the menu path reads by
        re-running ``begin_drag``/``extend_drag`` on every motion rather
        than mutating its offsets directly (the brief's NOTE) -- mouse
        drags never touch ``_kb_anchor``/``_kb_end``, so the two paths
        cannot fight over the same fields.

        Diff rows only take line-granularity motions (``j``/``k``/``o``);
        char/word motions are inert on them (selection unchanged) -- the
        row's own ``set_selection_range`` still line-snaps whatever range
        keyboard ``j``/``k`` produces.

        A floor keeps ``end`` from ever landing on (or being walked past)
        ``anchor`` -- an empty selection has nothing to quote -- by
        clamping the candidate to ``anchor + 1``/``anchor - 1`` whenever a
        motion would cross it; repeated presses past the floor simply stop
        moving rather than oscillating.
        """
        row = self._kb_selection_row
        if row is None or self._kb_anchor is None or self._kb_end is None:
            return
        is_diff_row = isinstance(row, ConsoleToolDiffRow)
        if is_diff_row and key in _KB_CHAR_KEYS:
            return  # diff rows: char/word/line-start/line-end motions do not apply
        anchor = self._kb_anchor
        end = self._kb_end
        if key == "o":
            anchor, end = end, anchor
        elif key in _KB_CHAR_KEYS or key in _KB_LINE_KEYS:
            text = row.get_display_text()
            if key == "h":
                # min(..., len(text)) heals a shrink-stranded end in one
                # press (PR #1813 Qodo bug 6): every other motion already
                # self-clamps through the pure helpers.
                candidate = min(max(end - 1, 0), len(text))
                forward = False
            elif key == "l":
                candidate = min(end + 1, len(text))
                forward = True
            elif key == "w":
                candidate = word_forward_offset(text, end)
                forward = True
            elif key == "b":
                candidate = word_back_offset(text, end)
                forward = False
            elif key == "0":
                candidate = line_start_offset(text, end)
                forward = False
            elif key == "$":
                candidate = line_end_offset(text, end)
                forward = True
            elif key == "j":
                candidate = next_line_offset(text, end)
                forward = True
            else:  # key == "k"
                candidate = prev_line_offset(text, end)
                forward = False
            # Floor: never let the active end reach or cross the anchor.
            if forward and end < anchor and candidate >= anchor:
                candidate = anchor - 1
            elif not forward and end > anchor and candidate <= anchor:
                candidate = anchor + 1
            end = candidate
        else:
            return
        if end == anchor:
            return
        self._kb_anchor, self._kb_end = anchor, end
        self.selection_manager.begin_drag(row.id, anchor)
        self.selection_manager.extend_drag(row.id, end)
        row.set_selection_range(*sorted((anchor, end)))

    def _paint_debug_dump(self, label: str) -> None:
        """task-623 live probe: append DOM truth about action rows to the file
        named by ``TLDW_TRANSCRIPT_PAINT_LOG``. No-op unless the env var is
        set; never raises. The DOM snapshot is taken synchronously (that
        timing is the point) but the file append is deferred off the caller's
        critical section via ``call_later``."""
        import os

        path = os.environ.get("TLDW_TRANSCRIPT_PAINT_LOG")
        if not path:
            return
        try:
            import time

            rows = list(self.query(".console-transcript-action-row"))
            lines = [
                f"{time.strftime('%H:%M:%S')} {label}: selected={self.selected_message_id!r} "
                f"action_rows={len(rows)} children={len(self.children)}"
            ]
            for r in rows:
                lines.append(
                    f"  row id={r.id!r} parent={type(r.parent).__name__}"
                    f" parent_is_transcript={r.parent is self}"
                    f" display={r.display} region={r.region}"
                )
            self.call_later(self._append_paint_log, path, "\n".join(lines) + "\n")
        except Exception:
            logger.opt(exception=True).debug("Paint-debug DOM snapshot failed.")

    def _append_paint_log(self, path: str, text: str) -> None:
        """Best-effort append for `_paint_debug_dump`; warns ONCE if the
        operator-supplied log path is unwritable so a dead probe is visible."""
        try:
            with open(path, "a", encoding="utf-8", errors="backslashreplace") as f:
                f.write(text)
        except OSError as exc:
            if not getattr(self, "_paint_log_warned", False):
                self._paint_log_warned = True
                logger.warning(
                    f"TLDW_TRANSCRIPT_PAINT_LOG write failed ({exc}); "
                    "paint-debug output is being dropped."
                )

    def action_invoke_selected_action(self, action_id: str) -> None:
        """Press the selected message's action button for ``action_id``.

        The action row mounts via ``call_later(refresh_messages)`` after
        ``select_message``, so a fast selection-then-shortcut sequence (e.g.
        Down immediately followed by ``c``) can run before that deferred
        mount lands and find no button. In that case, retry once after the
        pending refresh settles instead of silently no-oping.

        Args:
            action_id: Message action identifier, e.g. ``"copy"``.
        """
        message_id = self.selected_message_id
        if not message_id:
            return
        if action_id == "copy":
            thinking_detail = self.thinking_detail_text(message_id)
            if thinking_detail is not None:
                copy_to_clipboard = getattr(self.app, "copy_to_clipboard", None)
                if callable(copy_to_clipboard):
                    copy_to_clipboard(thinking_detail)
                return
        if action_id == "tool-output" and self._activity_can_expand(message_id):
            self.toggle_tool_output(message_id)
            return
        if self._press_selected_action_button(message_id, action_id):
            return
        self.call_after_refresh(self._invoke_selected_action_retry, action_id)

    def _invoke_selected_action_retry(self, action_id: str) -> None:
        """Retry a selected-message action once, after a deferred row mount settles.

        Gives up silently if the button is still absent (no loops, no
        timers) -- e.g. the selection changed again before the retry ran.

        Args:
            action_id: Message action identifier to retry.
        """
        message_id = self.selected_message_id
        if not message_id:
            return
        self._press_selected_action_button(message_id, action_id)

    def _press_selected_action_button(self, message_id: str, action_id: str) -> bool:
        """Press the action button for ``message_id``/``action_id`` if mounted.

        Args:
            message_id: Selected message id owning the action row.
            action_id: Message action identifier, e.g. ``"copy"``.

        Returns:
            True if the button was found and pressed, False otherwise.
        """
        selector = f"#console-message-action-{action_id}-{message_id}"
        try:
            button = self.query_one(selector, Button)
        except NoMatches:
            if action_id not in {"speak", "speak-stop"}:
                return False
            try:
                button = self.query_one(
                    f"#console-message-speech-action-{message_id}",
                    Button,
                )
            except NoMatches:
                return False
            if getattr(button, "console_action_id", None) != action_id:
                return False
        if button.disabled:
            reason = str(button.tooltip or "This action is unavailable.")
            self.notify(reason, severity="warning")
            return True
        button.press()
        return True

    def _selection_row_for(
        self, widget: Widget | None
    ) -> ConsoleTranscriptMessage | ConsoleMarkdownMessage | ConsoleToolDiffRow | None:
        """Return the selectable message row for a pressed widget, if any.

        Console selection phase 1. Walks parents from the event control to
        the nearest selection-protocol row: plain rows (character
        granularity), markdown rows (line granularity, task G), and tool
        diff rows (whole-diff-line granularity, phase 3 task 1).
        Protected controls (``PROTECTED_CLICK_CLASSES`` -- action rows,
        speech controls, rules, banners, scrollbars) never start a
        selection.
        """
        if widget is None:
            return None
        if any(
            widget.has_class(class_name) for class_name in self.PROTECTED_CLICK_CLASSES
        ):
            return None
        node: Widget | None = widget
        while node is not None:
            if node is self:
                return None
            if isinstance(
                node,
                (ConsoleTranscriptMessage, ConsoleMarkdownMessage, ConsoleToolDiffRow),
            ):
                return node
            node = node.parent
        return None

    def _selection_offset_for(
        self,
        row: ConsoleTranscriptMessage | ConsoleMarkdownMessage | ConsoleToolDiffRow,
        screen_x: int,
        screen_y: int,
    ) -> int:
        """Map a screen cell to a character offset in ``row``'s text domain.

        Plain rows resolve body-local cells wrap-aware through
        ``_body_cell_to_offset``. Markdown rows (task G) resolve against the
        Markdown widget's region at line granularity through
        ``_markdown_cell_to_offset``. Diff rows (phase 3, task 1) resolve
        against the DiffView's region through ``_diff_cell_to_offset``.
        Cells outside the body clamp to the text bounds (above -> 0, below
        -> end), which is the single-row clamp rule for drags that leave
        the row.
        """
        if isinstance(row, ConsoleMarkdownMessage):
            return self._markdown_selection_offset_for(row, screen_x, screen_y)
        if isinstance(row, ConsoleToolDiffRow):
            return self._diff_selection_offset_for(row, screen_x, screen_y)
        text = row.get_display_text()
        try:
            body = row.query_one(".console-transcript-message-body", Static)
        except NoMatches:
            return 0  # row not composed; anchor at the text start
        region = body.region
        width = body.content_region.width or region.width
        if width <= 0:
            return offset_for_cell(text, screen_x - region.x)
        return _body_cell_to_offset(
            text, width, screen_x - region.x, screen_y - region.y
        )

    def _markdown_selection_offset_for(
        self, row: ConsoleMarkdownMessage, screen_x: int, screen_y: int
    ) -> int:
        """Map a screen cell to a markdown row's source-line offset (task G)."""
        text = row.get_display_text()
        try:
            markdown = row.query_one(Markdown)
        except NoMatches:
            return 0  # row not composed; anchor at the source start
        region = markdown.region
        # The transcript CSS zeroes the Markdown child's margin/padding, so
        # its region is the rendered body; height <= 0 means not laid out.
        if region.height <= 0:
            return offset_for_cell(text, screen_x - region.x)
        return _markdown_cell_to_offset(
            text, region.height, screen_x - region.x, screen_y - region.y
        )

    def _diff_selection_offset_for(
        self, row: ConsoleToolDiffRow, screen_x: int, screen_y: int
    ) -> int:
        """Map a screen cell to a diff row's projection-line offset (task 1)."""
        text = row.get_display_text()
        try:
            diff_view = row.query_one(DiffView)
        except NoMatches:
            return 0  # diff not prepared/mounted yet; anchor at the start
        region = diff_view.region
        # The DiffView child carries the rendered body box; height <= 0
        # means not laid out (or the async prepare has not mounted it).
        if region.height <= 0:
            return offset_for_cell(text, screen_x - region.x)
        return _diff_cell_to_offset(
            text, region.height, screen_x - region.x, screen_y - region.y
        )

    def _selection_press_widget(self, event: MouseDown | MouseUp) -> Widget | None:
        """Return the widget a real-terminal mouse press/release hit.

        Textual's screen forwarding dispatches ``MouseDown``/``MouseUp`` to
        the widget under the pointer WITHOUT setting ``event.widget`` (only
        the translated ``MouseMove`` path assigns one), so ``event.control``
        is ``None`` for every press in a live terminal; it is populated only
        on the synthetic events pilot tests post. Live-spike evidence
        (2026-08-15): real drags logged ``ctrl=None`` on every MouseDown
        while synthetic-test events carried controls. Resolve the target
        from screen coordinates, falling back to ``event.control`` for
        synthetic callers.
        """
        if event.control is not None:
            return event.control
        try:
            widget, _offset = self.screen.get_widget_at(event.screen_x, event.screen_y)
        except Exception:
            return None
        return widget

    def on_mouse_down(self, event: MouseDown) -> None:
        """Arm a text-selection drag on a left press over a selectable row."""
        if self._kb_selection_row is not None:
            # Console selection phase 5: a mouse press takes over cleanly --
            # exit keyboard mode first, then let the normal drag-arming
            # logic below run as usual (a press on the same row starts a
            # fresh mouse drag).
            self._exit_keyboard_selection()
        press_control = self._selection_press_widget(event)
        # Click-outside dismissal, row-body half (final review): rows stop
        # their own Clicks (the message-selection toggle), so with a menu
        # open a press on another row's body never reaches this
        # transcript's ``on_click`` removal -- the menu used to stay
        # mounted while the user toggled selections elsewhere. Dismiss
        # mounted menus here, before arming the new drag, EXCEPT when the
        # press originates inside a ``ConsoleSelectionMenu``: the
        # Add-to-chat button's MouseDown precedes its Click, so removing
        # the menu on the press would unmount the button before its Click
        # can activate it.
        press_node: Widget | None = press_control
        while press_node is not None and not isinstance(
            press_node, ConsoleSelectionMenu
        ):
            press_node = press_node.parent
        if press_node is None:
            self._remove_selection_menu()
        # Textual encodes a real left press as button 1 (the XTerm driver
        # maps the left button to ``(buttons + 1) & 3``; 0 means "no button",
        # as in plain mouse-move reports).
        row = self._selection_row_for(press_control) if event.button == 1 else None
        if row is None:
            # A fresh press that cannot arm a drag (non-left button, or a
            # protected/non-row control) ends the drag-release suppression
            # window: the NEXT genuine click must behave normally. The drag's
            # own release Click arrives with no intervening MouseDown, so
            # same-press suppression is intact.
            self.selection_manager.consume_just_finished()
            return
        offset = self._selection_offset_for(row, event.screen_x, event.screen_y)
        self.selection_manager.begin_drag(row.id, offset)
        self._selection_origin_row = row
        # TASK-21114: the stale-highlight sweep over every mounted row used
        # to run on EVERY MouseMove (hundreds of rows under the 20k/12k-line
        # watermarks, at 50-100 Hz). One sweep when the drag arms keeps the
        # same guarantee -- no other row can GAIN a highlight mid-drag (the
        # only writers are this drag's origin row and keyboard mode, which
        # exits above) -- so the moves only ever touch the origin row.
        self._clear_other_selection_highlights(row)
        # Capture the mouse so the terminal MouseUp reaches this transcript
        # even when the pointer is released outside it; otherwise the
        # manager stays active and suppresses row clicks until the next
        # MouseDown (ported from the reference implementation's fix).
        self.capture_mouse(True)

    def _clear_other_selection_highlights(self, active_row: Widget) -> None:
        """Clear any stale text-selection highlight on every OTHER row.

        TASK-21114: called once per drag (at arm time) instead of per
        MouseMove. ``clear_selection`` is a guarded no-op on rows without a
        stored range, so the sweep costs one attribute check per mounted
        selectable row.
        """
        for other in self._row_widgets.values():
            if (
                isinstance(
                    other,
                    (
                        ConsoleTranscriptMessage,
                        ConsoleMarkdownMessage,
                        ConsoleToolDiffRow,
                    ),
                )
                and other.id != active_row.id
            ):
                other.clear_selection()

    def on_mouse_move(self, event: MouseMove) -> None:
        """Extend the active drag over the origin row's body text."""
        if not self.selection_manager.state.active:
            return
        event.stop()
        selection = self.selection_manager.state.selection
        row = self._selection_origin_row
        if (
            selection is None
            or row is None
            or not row.is_attached
            or row.id != selection.row_key
        ):
            return  # origin row went away: hold the last position
        offset = self._selection_offset_for(row, event.screen_x, event.screen_y)
        if not self.selection_manager.extend_drag(row.id, offset):
            # TASK-21114: the pointer moved within the same character cell --
            # nothing to re-render. (Stale highlights on OTHER rows were
            # already swept once when the drag armed, in ``on_mouse_down``.)
            return
        updated = self.selection_manager.state.selection
        if updated is None:
            return
        row.set_selection_range(updated.start, updated.end)

    def on_mouse_up(self, event: MouseUp) -> None:
        """Finish the drag; post a selection message for menu-worthy releases."""
        # Self-guarding: a no-op unless this transcript holds the capture.
        self.release_mouse()
        if not self.selection_manager.state.active:
            return
        event.stop()
        selection = self.selection_manager.finish_drag()
        self._selection_origin_row = None
        if selection is None:
            # Empty finish (a plain click, not a drag): the manager's
            # just_finished flag exists to suppress drag-release clicks, so
            # consume it here and let the following Click select the message.
            self.selection_manager.consume_just_finished()
            return
        self.post_message(
            self.TranscriptTextSelected(
                selection=selection,
                screen_x=event.screen_x,
                screen_y=event.screen_y,
            )
        )

    @on(TranscriptTextSelected)
    async def _text_selected(self, event: TranscriptTextSelected) -> None:
        """Mount the floating selection menu at the drag-release cell.

        Console selection phase 1, live-spike round 3 rework: the menu is
        mounted on the owning SCREEN with an ``absolute_offset`` (the
        tooltip anchoring mechanism). The previous approaches -- plain flow
        child (rendered at the end of the scroll content), and a docked
        transcript child with ``styles.offset`` -- were both broken in a
        live terminal: the flow child sat below the fold, and the docked
        child painted translated by its offset while being CLIPPED to the
        un-translated dock slot (the user saw one button; hit-tests used
        the un-translated region, so the rest were unclickable). Screen
        mounting folds the position into the widget's region, so paint,
        clipping, and hit-testing agree. Deliberately does not stop the
        event: the owning screen also consumes TranscriptTextSelected
        (selection lifecycle).

        The event carries SCREEN coordinates directly; the clamp keeps the
        anchor within the OWNING TRANSCRIPT's visible region (the ``+1``
        sits the menu just below the release row). Live-spike 2026-08-16:
        the real Console layout puts the composer + status bar BELOW the
        transcript, so the transcript's box ends above the screen edge --
        clamping the anchor to SCREEN bounds let a bottom-of-transcript
        release paint the menu over the composer.

        Async and awaiting the previous menu's removal is load-bearing:
        ``Widget.remove()`` only SCHEDULES removal while ``mount()``
        registers the new menu into the DOM synchronously, so a same-id
        remount over a still-attached old menu raises Textual's app-fatal
        ``DuplicateIds`` (consecutive selections crashed before). Menus
        whose removal was merely scheduled (``_pruning``) are awaited too
        rather than skipped: a skipped one can still be attached at the
        remount, and awaiting an already-pruning node's ``remove()`` is
        harmless (it waits out the prune already in flight). The freshly
        mounted menu survives the drag-release Click because that Click is
        stopped by the row's ``on_click`` (drag-release suppression), so it
        never reaches this transcript's own ``on_click`` removal.
        """
        for menu in self._attached_selection_menus():
            await menu.remove()
        # Mounting on the screen triggers a layout refresh that re-engages
        # Textual's bottom anchor and yanks the view to the tail -- away
        # from the selection the user just made. Release the tail-follow
        # for the menu's lifetime (the jump pill appears, the standard
        # detached-reader affordance).
        self.release_anchor()
        # Phase 3: selections in agent output (ASSISTANT/TOOL-role rows,
        # diff rows) additionally offer the review-feedback actions,
        # run-gated through the owning screen's status seam.
        origin_row = self._active_selection_row()
        feedback_available = self._row_supports_selection_feedback(origin_row)
        # Live spike 2026-08-16 8:48: when the measured clamp must pull the
        # menu up off the release point, pinning its bottom to the
        # transcript bottom landed it ON TOP of the just-selected row --
        # the reverse-video highlight strip (the evidence of the selection)
        # hid behind the menu. Hand the menu the row's screen top so its
        # clamp can hop entirely ABOVE the row when there is room (NULL /
        # unmeasured row region passes None -> plain bottom pin).
        origin_region = origin_row.region if origin_row is not None else None
        selection_top = origin_region.y if origin_region else None
        # Clamp the ANCHOR POINT into the transcript's own visible box
        # (never the bare screen): the composer/status bar live below this
        # region, and the menu's measured post-layout clamp (in the menu)
        # finishes the job against the same box.
        bounds = self.region
        if not bounds:
            # Clamp-fix review: pre-layout the region is NULL_REGION (never
            # None in textual 8.2.8), and a zero-size box would collapse
            # both clamp axes to its origin and pin the menu at (0, 0) --
            # fall back to screen-size bounds, mirroring the menu-side
            # guard for unmeasured owners.
            screen_size = self.screen.size
            bounds = Region(0, 0, screen_size.width, screen_size.height)
        self.screen.mount(
            ConsoleSelectionMenu(
                screen_x=self._clamp_menu_offset(
                    event.screen_x, low=bounds.x, high=bounds.right, margin=2
                ),
                screen_y=self._clamp_menu_offset(
                    event.screen_y + 1, low=bounds.y, high=bounds.bottom, margin=2
                ),
                owner=self,
                feedback_available=feedback_available,
                run_active=feedback_available and self._selection_run_active(),
                selection_top=selection_top,
            )
        )

    @staticmethod
    def _clamp_menu_offset(value: int, low: int, high: int, *, margin: int) -> int:
        """Clamp a screen-space menu anchor into ``[low, high - margin]``.

        ``low``/``high`` delimit the owning transcript's visible box on one
        axis, so the anchor can never leave the transcript (the composer
        below it stays clear); the inner ``max(low, ...)`` guards
        degenerate boxes thinner than the margin.
        """
        return max(low, min(value, max(low, high - margin)))

    @on(ConsoleSelectionMenu.AddToChat)
    def _selection_add_to_chat(self, event: ConsoleSelectionMenu.AddToChat) -> None:
        """Quote the active row selection up to the owning screen and clean up."""
        event.stop()
        row = self._active_selection_row()
        if row is not None:
            self.post_message(
                ConsoleSelectionQuoteRequested(
                    quote=cap_quote(row.get_selection_text())
                )
            )
            row.clear_selection()
        self.selection_manager.cancel()
        self._selection_origin_row = None
        self._remove_selection_menu()

    @on(ConsoleSelectionMenu.Dismissed)
    def _selection_menu_dismissed(self, event: ConsoleSelectionMenu.Dismissed) -> None:
        """Escape dismissal clears the whole selection UI (strip included)."""
        event.stop()
        self._remove_selection_menu()
        self.selection_manager.cancel()
        self._selection_origin_row = None

    @on(ConsoleSelectionMenu.MoreDetails)
    def _selection_more_details(self, event: ConsoleSelectionMenu.MoreDetails) -> None:
        """Open a More Details side chat about the active selection."""
        event.stop()
        self._request_side_chat(ConsoleSideChatRequested.MODE_MORE_DETAILS)

    @on(ConsoleSelectionMenu.AskInSideChat)
    def _selection_ask_side_chat(
        self, event: ConsoleSelectionMenu.AskInSideChat
    ) -> None:
        """Open a freeform Ask in Side Chat about the active selection."""
        event.stop()
        self._request_side_chat(ConsoleSideChatRequested.MODE_ASK)

    def _request_side_chat(self, mode: str) -> None:
        """Post a capped-selection side-chat request and clean up (phase 2).

        Same quote plumbing and cleanup as ``_selection_add_to_chat``: the
        selection text is capped by ``cap_quote`` before it leaves the
        transcript, the row range is cleared, the drag manager cancelled,
        and the menu removed.
        """
        row = self._active_selection_row()
        if row is not None:
            self.post_message(
                ConsoleSideChatRequested(
                    quote=cap_quote(row.get_selection_text()), mode=mode
                )
            )
            row.clear_selection()
        self.selection_manager.cancel()
        self._selection_origin_row = None
        self._remove_selection_menu()

    @on(ConsoleSelectionMenu.RequestChanges)
    def _selection_request_changes(
        self, event: ConsoleSelectionMenu.RequestChanges
    ) -> None:
        """Send request-changes review feedback for the active selection."""
        event.stop()
        self._request_selection_feedback(
            ConsoleSelectionFeedbackRequested.ACTION_REQUEST_CHANGES
        )

    @on(ConsoleSelectionMenu.Lgm)
    def _selection_lgm(self, event: ConsoleSelectionMenu.Lgm) -> None:
        """Send LGTM review feedback for the active selection."""
        event.stop()
        self._request_selection_feedback(ConsoleSelectionFeedbackRequested.ACTION_LGM)

    @on(ConsoleSelectionMenu.Comment)
    def _selection_comment(self, event: ConsoleSelectionMenu.Comment) -> None:
        """Send comment feedback for the active selection."""
        event.stop()
        self._request_selection_feedback(
            ConsoleSelectionFeedbackRequested.ACTION_COMMENT
        )

    @on(ConsoleSelectionMenu.CreateNote)
    def _selection_create_note(self, event: ConsoleSelectionMenu.CreateNote) -> None:
        """Save the active selection as a note (task-18156 Task 6).

        Same quote plumbing and cleanup as ``_selection_add_to_chat``: the
        capped quote leaves the transcript in an app-level message; the
        owning screen derives the title and writes the note off-thread.
        """
        event.stop()
        row = self._active_selection_row()
        if row is not None:
            self.post_message(
                ConsoleSelectionNoteRequested(quote=cap_quote(row.get_selection_text()))
            )
            row.clear_selection()
        self.selection_manager.cancel()
        self._selection_origin_row = None
        self._remove_selection_menu()

    def _request_selection_feedback(self, action: str) -> None:
        """Post a capped-selection feedback request and clean up (phase 3).

        Same quote plumbing and cleanup as ``_selection_add_to_chat``: the
        selection text is capped by ``cap_quote`` before it leaves the
        transcript (the screen no-ops empty quotes), the row range is
        cleared, the drag manager cancelled, and the menu removed. The
        structured message composition and prompt-queue routing live on
        the owning screen (phase 3 task 5).
        """
        row = self._active_selection_row()
        if row is not None:
            self.post_message(
                ConsoleSelectionFeedbackRequested(
                    action=action,
                    quote=cap_quote(row.get_selection_text()),
                    anchor_message_id=getattr(row, "message_id", None),
                )
            )
            row.clear_selection()
        self.selection_manager.cancel()
        self._selection_origin_row = None
        self._remove_selection_menu()

    def _row_supports_selection_feedback(
        self,
        row: ConsoleTranscriptMessage
        | ConsoleMarkdownMessage
        | ConsoleToolDiffRow
        | None,
    ) -> bool:
        """Whether the selection's origin row is agent output (phase 3).

        Diff rows exist only under expanded file-write TOOL markers, so
        they are agent output by definition; plain/markdown rows qualify
        when the message they render is ASSISTANT- or TOOL-role. Product
        decision 2026-08-16: the agent's own prose replies (markdown or
        plain) are the most natural review target, so ASSISTANT-role rows
        qualify alongside tool markers/diagnostics; USER-role rows never
        do (the user's own words are not reviewable output). ``None`` (no
        live selection) offers nothing.
        """
        if isinstance(row, ConsoleToolDiffRow):
            return True
        if isinstance(row, (ConsoleTranscriptMessage, ConsoleMarkdownMessage)):
            message = getattr(row, "_message", None)
            return message is not None and message.role in (
                ConsoleMessageRole.ASSISTANT,
                ConsoleMessageRole.TOOL,
            )
        return False

    def _selection_run_active(self) -> bool:
        """Whether the owning screen reports an active console run (phase 3).

        Reads the screen's run-status seam (``ChatScreen``'s
        ``_current_console_run_status_value``) defensively via ``getattr``:
        bare harness screens and non-console hosts do not expose it, which
        simply means "no active run" (Request changes / LGTM stay gated).
        """
        try:
            status_getter = getattr(
                self.screen, "_current_console_run_status_value", None
            )
        except NoScreen:  # pragma: no cover - teardown race only
            return False
        if not callable(status_getter):
            return False
        return str(status_getter()).strip().lower() in (
            _SELECTION_FEEDBACK_ACTIVE_RUN_STATUSES
        )

    def _active_selection_row(
        self,
    ) -> ConsoleTranscriptMessage | ConsoleMarkdownMessage | ConsoleToolDiffRow | None:
        """Resolve the row widget holding the active selection, if any."""
        sel = self.selection_manager.state.selection
        if sel is None:
            return None
        # Query by id without a type expectation: the selected row may be
        # a plain, markdown (task G), or tool diff (phase 3, task 1) row,
        # and a typed query_one would raise WrongType on the other kinds.
        try:
            widget = self.query_one(f"#{sel.row_key}")
        except NoMatches:
            return None
        if isinstance(
            widget,
            (ConsoleTranscriptMessage, ConsoleMarkdownMessage, ConsoleToolDiffRow),
        ):
            return widget
        return None

    def _attached_selection_menus(self) -> list[ConsoleSelectionMenu]:
        """Menus still attached whose removal is not already scheduled.

        Textual marks a widget ``_pruning`` synchronously inside
        ``remove()`` but detaches it only when the prune message is
        processed, so a menu can survive two removal calls; already-pruning
        menus are skipped to keep ``remove()`` single-shot per menu.

        TASK-21119: sourced from the menu registry rather than
        ``self.screen.query(ConsoleSelectionMenu)`` -- same screen scope,
        same result, without a full-screen DOM walk. This runs on every
        in-transcript press (``on_mouse_down``), not just on dismissal.
        ``self.screen`` still resolves first, so a detached transcript
        raises ``NoScreen`` exactly as before.
        """
        screen = self.screen
        return [
            menu
            for menu in selection_menus_on_screen(screen)
            if not getattr(menu, "_pruning", False)
        ]

    def _remove_selection_menu(self) -> None:
        """Dismiss the selection UI: remove the menu AND the highlight.

        Every dismissal path (escape, click-outside, action cleanup)
        clears the text selection as well -- live-spike feedback: the
        markdown highlight strip lingered after the menu closed until the
        user clicked the strip itself. Action handlers read the quote
        BEFORE calling here, so clearing on removal is safe for them.

        Fire-and-forget: suitable for dismissal paths that never remount
        a same-id menu afterwards. The remount path (``_text_selected``)
        must await the removals instead -- see its docstring.
        """
        row = self._active_selection_row()
        if row is not None:
            row.clear_selection()
        # Deliberately NOT cancelling the drag manager here: the drag's own
        # release Click is still in the queue, and its suppression flag is
        # what stops it from toggling the row's message selection. The
        # click cycle consumes that flag (empty finish or non-arming press),
        # and the next armed drag replaces the selection state wholesale.
        for menu in self._attached_selection_menus():
            menu.remove()

    async def on_click(self, event: Click) -> None:
        """Clear selection when the user clicks negative space in the transcript.

        Any click that reaches this handler is outside the floating selection
        menu (the menu stops clicks that land inside it), so the menu is
        removed first: click-outside dismisses it with no other side effect,
        then the normal click handling continues.

        A drag release (``just_finished`` or the one-shot
        ``release_click_pending`` token, which the row guard's short-circuit
        can leave armed) is consumed here instead -- BEFORE any dismissal
        cleanup, which would wipe the row selection the just-opened menu
        exists to act on (live spike 2026-08-16: the release click reached
        this handler with ``just_finished`` already consumed and
        ``_remove_selection_menu()`` erased the quote before the action read
        it).

        Clicks that land on controls with classes in ``PROTECTED_CLICK_CLASSES``
        (message action rows/buttons, rule separators, action-help text, the
        empty-state panel, or scrollbars) keep the current selection active. All
        other clicks that bubble up to the transcript itself clear the selection.
        """
        if (
            self.selection_manager.just_finished
            or self.selection_manager.consume_release_click()
        ):
            event.stop()
            self.selection_manager.consume_just_finished()
            return
        control = event.control
        await self._dismiss_message_more_for_click(control)
        self._remove_selection_menu()
        if self.selection_manager.just_finished:
            event.stop()
            self.selection_manager.consume_just_finished()
            return
        if control is not None and any(
            control.has_class(class_name) for class_name in self.PROTECTED_CLICK_CLASSES
        ):
            event.stop()
            return
        # Capture-routed row clicks (live spike 2026-08-16: 'can't select
        # messages via mouse'): the drag-arm on press captures the mouse,
        # and the synthesized Click is routed to THIS capturer -- the
        # capture only releases when the MouseUp is processed, which lands
        # after the Click was already forwarded. The row the pointer
        # actually targeted never sees the click, so its toggle (and the
        # row-level drag-release suppression) must run here instead.
        row_node: Widget | None = control
        while row_node is not None and not isinstance(
            row_node,
            (ConsoleMarkdownMessage, ConsoleTranscriptMessage, ConsoleToolDiffRow),
        ):
            row_node = row_node.parent
        if row_node is not None:
            event.stop()
            manager = self.selection_manager
            if (
                manager.state.active
                or manager.just_finished
                or manager.consume_release_click()
            ):
                manager.consume_just_finished()
                manager.consume_release_click()
                return
            self.toggle_message_selection(row_node.message_id)
            return
        if control is self:
            self.action_clear_selection()
            event.stop()

    async def _dismiss_message_more_for_click(self, control: Widget | None) -> None:
        """Dismiss More for any transcript click except its own opener."""
        if getattr(control, "console_action_id", None) != "more":
            await self.dismiss_message_more_menu(restore_focus=False)

    def on_key(self, event: Key) -> None:
        if self._kb_selection_row is not None:
            # Console selection phase 5 (keyboard mode). A row destroyed out
            # from under the mode (streaming replacement, prune, session
            # switch) must never crash the next keypress -- the
            # reconciliation guard (`_cancel_selection_if_row_removed`)
            # already cancelled the selection manager when the row was
            # removed, so this only needs to drop the mode's own state.
            if not self._kb_selection_row.is_attached:
                self._exit_keyboard_selection(clear=False)
                return
            if event.key == "escape":
                # Preempt the clear-selection BINDING below: the first Esc
                # only leaves the mode, keeping the message (j/k) selection
                # intact for a second Esc to clear normally.
                self._exit_keyboard_selection()
                event.stop()
                event.prevent_default()
                return
            if event.key == "enter":
                # Task 4: Enter = mouse-release parity. finish_drag() flips
                # the manager to its finished state (which _text_selected's
                # _active_selection_row() reads), and the SAME
                # TranscriptTextSelected message drives the SAME menu path
                # (clamping, feedback gating, above-row hop) -- the only
                # keyboard-specific work is the anchor, derived from the
                # row's laid-out region because there is no release cell.
                event.stop()
                event.prevent_default()
                row = self._kb_selection_row
                selection = self.selection_manager.finish_drag()
                # Keyboard has no release Click to consume the suppression
                # tokens the finish just armed -- drain them, or the NEXT
                # genuine row click's selection toggle is eaten.
                self.selection_manager.consume_release_click()
                self.selection_manager.consume_just_finished()
                # clear=False: the highlight and the manager's finished
                # state ARE the menu's working material; only the mode's
                # own state (and the hint) go away.
                self._exit_keyboard_selection(clear=False)
                if selection is None or row is None:
                    return
                region = row.region
                self.post_message(
                    self.TranscriptTextSelected(
                        selection=selection,
                        screen_x=region.x + min(4, max(0, region.width - 1)),
                        # The handler's +1 lands the menu just below the row.
                        screen_y=region.bottom - 1,
                    )
                )
                return
            if event.is_printable or event.key in {"enter", "up", "down"}:
                # Interception rule (corrected after Task 2's review found
                # the fall-through desync): every printable single
                # character, plus enter/up/down, is claimed here while the
                # mode is armed. Left unclaimed, `j`/`k`/`down`/`up` would
                # move `selected_message_id` to a DIFFERENT message while
                # `_kb_selection_row` (and the manager state, and the hint)
                # stayed pinned to the OLD row -- a silent mode/message-
                # selection desync -- `enter` would toggle message
                # selection out from under the mode (Task 4 wires the
                # mode's own Enter-finish action; this stays a no-op for
                # it), and any other unclaimed letter (e.g. `c`) would fire
                # its normal BINDING (Copy) instead of staying inert.
                # Page-up/page-down and the mouse wheel are NOT claimed --
                # they still scroll the transcript while the mode is on.
                event.stop()
                event.prevent_default()
                motion_key = event.character if event.is_printable else event.key
                if (
                    motion_key in _KB_CHAR_KEYS
                    or motion_key in _KB_LINE_KEYS
                    or motion_key == "o"
                ):
                    self._kb_apply_motion(motion_key)
                return
        if event.key in {"down", "j"}:
            self.action_select_next()
            event.stop()
        elif event.key in {"up", "k"}:
            self.action_select_previous()
            event.stop()
        elif event.key == "enter":
            self.action_confirm_selection()
            event.stop()
        elif event.key == "escape":
            self.action_clear_selection()
            self._remove_selection_menu()
            self.selection_manager.cancel()
            self._selection_origin_row = None
            event.stop()

    def _select_relative(self, offset: int) -> None:
        visible = self._visible_messages()
        if not visible:
            return
        if self.selected_message_id is None:
            index = 0 if offset >= 0 else len(visible) - 1
        else:
            current = next(
                (
                    index
                    for index, message in enumerate(visible)
                    if message.id == self.selected_message_id
                ),
                0,
            )
            index = min(max(current + offset, 0), len(visible) - 1)
        self.select_message(visible[index].id)

    def _message_by_id(self, message_id: str) -> ConsoleChatMessage | None:
        message = next(
            (message for message in self._messages if message.id == message_id), None
        )
        return (
            message
            if message is not None
            else self._thinking_display_message(message_id)
        )

    def _visible_messages(self) -> list[ConsoleChatMessage]:
        """Return the messages with rendered rows (excludes the pruned window).

        Keyboard selection walks this list so j/k never lands on a pruned
        (row-less) message; the store-facing ``_messages`` keeps full history.
        """
        causal_messages = [
            message
            for message in self._messages
            if message.id not in self._pruned_message_ids
            and message.id not in self._hidden_tail_ids
        ]
        visible: list[ConsoleChatMessage] = []
        for unit in group_console_transcript_messages(causal_messages):
            if unit.standalone is not None:
                visible.append(unit.standalone)
                continue
            turn = unit.assistant_turn
            assert turn is not None
            for activity in ordered_assistant_activities(
                turn,
                live_block_id=self._live_thinking_block_id(turn),
            ):
                if isinstance(activity, ConsoleChatMessage):
                    visible.append(activity)
                else:
                    display = self._thinking_display_message(activity.activity_id)
                    if display is not None:
                        visible.append(display)
            visible.append(turn.assistant)
        return visible

    def _notify_selection_changed(self) -> None:
        """Let the owning screen refresh inspector/control surfaces after selection changes."""
        sync_console_control_bar = getattr(
            self.screen, "_sync_console_control_bar", None
        )
        if callable(sync_console_control_bar):
            sync_console_control_bar()

    def _flat_transcript_rows(self) -> list[_TranscriptRow]:
        """Plan the legacy per-message rows reused by standalone and nested UI."""
        rows: list[_TranscriptRow] = []
        # Hoisted: which row (if any) carries this tick's activity line is a
        # property of the message list, not of any one row.
        activity_target_id = (
            self._turn_activity_target_id() if self._turn_activity else None
        )
        for message in self._messages:
            if (
                message.id in self._pruned_message_ids
                or message.id in self._hidden_tail_ids
            ):
                # TASK-1365: pruned by the height watermarks; the store keeps
                # the message, the view window drops every row derived from it.
                # TASK-15777: the hidden tail is the same view-only contract
                # on the other boundary.
                continue
            message = self._with_expanded_tool_output(message)
            message = self._with_turn_activity(message, activity_target_id)
            selected = message.id == self.selected_message_id
            rows.append(
                _TranscriptRow(
                    key=f"rule:{message.id}",
                    kind="rule",
                    signature=("rule", message.id),
                    renderable=CONSOLE_TRANSCRIPT_RULE,
                )
            )
            if message.id == self.summary_boundary_message_id:
                rows.append(
                    _TranscriptRow(
                        key=f"summary-banner:{message.id}",
                        kind="banner",
                        signature=("banner", message.id),
                        renderable=CONSOLE_SUMMARY_BANNER_COPY,
                    )
                )
            rows.append(
                _TranscriptRow(
                    key=f"message:{message.id}",
                    kind="message",
                    signature=self._cached_message_row_signature(
                        message, selected=selected
                    ),
                    message=message,
                    selected=selected,
                )
            )
            if (
                message.id in self._expanded_tool_output_ids
                and message.tool_diff is not None
            ):
                # TASK-1366: inline diff row for a file-write marker,
                # directly under its message row so it stays inside the
                # message group (pruning drops every row derived from a
                # pruned message id at the top of this loop). Signature is
                # stable: a marker's tool_diff is fixed at append time.
                rows.append(
                    _TranscriptRow(
                        key=f"diff:{message.id}",
                        kind="diff",
                        signature=("diff", message.id),
                        message=message,
                    )
                )
            citation_count = self._citation_counts.get(message.id, 0)
            if citation_count > 0:
                rows.append(
                    _TranscriptRow(
                        key=f"citations:{message.id}",
                        kind="citations",
                        signature=("citations", message.id, citation_count),
                        message=message,
                        renderable=f"Cited sources ({citation_count})",
                    )
                )
            annotation_notes = self._annotation_previews.get(message.id)
            if annotation_notes:
                # task-17169: inline review-note marker under the annotated
                # message. The notes ride the signature so an added or edited
                # note re-renders the mounted marker instead of going stale.
                rows.append(
                    _TranscriptRow(
                        key=f"annotations:{message.id}",
                        kind="annotations",
                        signature=("annotations", message.id, annotation_notes),
                        message=message,
                        renderable=_annotation_marker_content(annotation_notes),
                    )
                )
            original_attempt = self._original_attempt_previews.get(message.id)
            if original_attempt is not None:
                rows.append(
                    _TranscriptRow(
                        key=f"original-attempt:{message.id}",
                        kind="original-attempt",
                        signature=(
                            "original-attempt",
                            message.id,
                            original_attempt,
                        ),
                        message=message,
                        renderable=Content.assemble(
                            ("Original attempt (not selected)", "dim"),
                            "\n",
                            original_attempt,
                        ),
                    )
                )
            card_spec = self._generation_card_specs.get(message.id)
            video_spec = self._video_card_specs.get(message.id)
            if video_spec is not None:
                # A video-generation message renders its card row INSTEAD of
                # any image/generation-card row (it never has attachments --
                # mutually exclusive per message id, ADR-044).
                rows.append(
                    _TranscriptRow(
                        key=f"video-card:{message.id}",
                        kind="video-card",
                        signature=video_card_signature(video_spec),
                        message=message,
                        video_card_spec=video_spec,
                    )
                )
            elif card_spec is not None:
                # A generation-card message renders the card row INSTEAD of
                # the plain image row -- mutually exclusive per message id.
                rows.append(
                    _TranscriptRow(
                        key=f"generation-card:{message.id}",
                        kind="generation-card",
                        signature=generation_card_signature(card_spec),
                        message=message,
                        generation_card_spec=card_spec,
                    )
                )
            else:
                image_spec = self._image_specs.get(message.id)
                if image_spec is not None:
                    rows.append(
                        _TranscriptRow(
                            key=f"image:{message.id}",
                            kind="image",
                            signature=("image", message.id, image_spec.mode),
                            message=message,
                            image_spec=image_spec,
                        )
                    )
            if selected:
                rows.append(
                    _TranscriptRow(
                        key=f"actions:{message.id}",
                        kind="actions",
                        signature=self._action_row_signature(message),
                        message=message,
                    )
                )
                # DS-01: the legend under the buttons names this row's
                # glyph-only actions in words, so its text must join the
                # signature -- a static guide would survive a speak -> ⏹
                # swap or a variant set appearing and name glyphs the row
                # no longer shows.
                guide = self._action_guide(message)
                rows.append(
                    _TranscriptRow(
                        key=f"action-help:{message.id}",
                        kind="action-help",
                        signature=("action-help", guide),
                        renderable=guide,
                    )
                )
        if self._messages:
            rows.append(
                _TranscriptRow(
                    key="rule:end",
                    kind="rule",
                    signature=("rule", "end"),
                    renderable=CONSOLE_TRANSCRIPT_RULE,
                )
            )
        else:
            rows.append(
                _TranscriptRow(
                    key="empty",
                    kind="empty",
                    signature=(
                        "empty",
                        self._empty_card_state,
                        self.empty_state_action_label,
                        self.empty_state_action_tooltip,
                    ),
                    action_label=self.empty_state_action_label,
                    action_tooltip=self.empty_state_action_tooltip,
                    card_state=self._empty_card_state,
                )
            )
        return rows

    def _transcript_rows(self) -> list[_TranscriptRow]:
        """Plan top-level rows, grouping owned TOOL markers into Assistant turns."""
        flat_rows = self._flat_transcript_rows()
        visible_messages = [
            message
            for message in self._messages
            if message.id not in self._pruned_message_ids
            and message.id not in self._hidden_tail_ids
        ]
        if not visible_messages:
            return flat_rows

        starts = {
            row.key.removeprefix("rule:"): index
            for index, row in enumerate(flat_rows)
            if row.kind == "rule" and row.key != "rule:end"
        }
        groups: dict[str, tuple[_TranscriptRow, ...]] = {}
        for index, message in enumerate(visible_messages):
            start = starts[message.id]
            if index + 1 < len(visible_messages):
                end = starts[visible_messages[index + 1].id]
            else:
                end = next(
                    (
                        row_index
                        for row_index in range(start + 1, len(flat_rows))
                        if flat_rows[row_index].key == "rule:end"
                    ),
                    len(flat_rows),
                )
            groups[message.id] = tuple(flat_rows[start:end])

        rows: list[_TranscriptRow] = []
        for unit in group_console_transcript_messages(visible_messages):
            if unit.standalone is not None:
                rows.extend(groups[unit.standalone.id])
                continue

            turn = unit.assistant_turn
            assert turn is not None
            assistant_rows = groups[turn.assistant.id]
            message_index = next(
                index
                for index, row in enumerate(assistant_rows)
                if row.kind == "message"
            )
            rows.extend(assistant_rows[:message_index])
            nested_rows = assistant_rows[message_index:]
            selected_id = self.selected_message_id
            owned_selected_id = (
                selected_id if selected_id in turn.owned_message_ids else None
            )
            activity_items = ordered_assistant_activities(
                turn,
                live_block_id=self._live_thinking_block_id(turn),
            )
            if not self._show_model_thinking:
                activity_items = tuple(
                    item
                    for item in activity_items
                    if not isinstance(item, ConsoleThinkingActivityRef)
                )
            activity_ids = tuple(
                item.id if isinstance(item, ConsoleChatMessage) else item.activity_id
                for item in activity_items
            )
            if selected_id in activity_ids:
                owned_selected_id = selected_id
            activity_rows = tuple(
                (
                    tuple(
                        row
                        for row in groups[item.id]
                        if row.kind not in {"rule", "banner"}
                    )
                    if isinstance(item, ConsoleChatMessage)
                    else ()
                )
                for item in activity_items
            )
            activity_signature = tuple(
                (
                    item.id
                    if isinstance(item, ConsoleChatMessage)
                    else item.activity_id,
                    (
                        item.activity_presentation
                        if isinstance(item, ConsoleChatMessage)
                        else (
                            item.label,
                            item.status,
                            self.thinking_detail_text(item.activity_id),
                        )
                    ),
                    (
                        item.id
                        if isinstance(item, ConsoleChatMessage)
                        else item.activity_id
                    )
                    in self._expanded_tool_output_ids,
                    selected_id
                    == (
                        item.id
                        if isinstance(item, ConsoleChatMessage)
                        else item.activity_id
                    ),
                    tuple(row.signature for row in owned_rows),
                )
                for item, owned_rows in zip(activity_items, activity_rows)
            )
            adjunct_signature = tuple(row.signature for row in nested_rows[1:])
            rows.append(
                _TranscriptRow(
                    key=f"assistant-turn:{turn.assistant.id}",
                    kind="assistant-turn",
                    signature=(
                        "assistant-turn",
                        nested_rows[0].signature,
                        activity_signature,
                        activity_ids,
                        owned_selected_id,
                        tuple(
                            sorted(self._expanded_tool_output_ids & set(activity_ids))
                        ),
                        adjunct_signature,
                    ),
                    message=turn.assistant,
                    selected=selected_id == turn.assistant.id,
                    assistant_turn=turn,
                    nested_rows=tuple(nested_rows),
                    activity_rows=activity_rows,
                    activity_items=activity_items,
                    activity_signature=activity_signature,
                    adjunct_signature=adjunct_signature,
                )
            )
        end_rule = next((row for row in flat_rows if row.key == "rule:end"), None)
        if end_rule is not None:
            rows.append(end_rule)
        return rows

    def _message_widgets(self) -> list[Widget]:
        return [
            self._build_row_widget(row, track=False) for row in self._transcript_rows()
        ]

    def _cancel_selection_if_row_removed(self, widget: Widget) -> None:
        """Drop drag-selection state when its row widget is removed/rebuilt.

        A rebuilt row widget does not carry the previous selection range, so
        keeping the manager state would desync highlight vs. domain (ported
        from the reference implementation). Releases mouse capture the same
        way ``on_mouse_up`` does, so a mid-drag row rebuild cannot leave the
        pointer captured.
        """
        selection = self.selection_manager.state.selection
        selection_row: Widget | None = None
        if selection is not None:
            if widget.id == selection.row_key:
                selection_row = widget
            else:
                matches = list(widget.query(f"#{selection.row_key}"))
                selection_row = matches[0] if matches else None
        if isinstance(
            selection_row,
            (ConsoleTranscriptMessage, ConsoleMarkdownMessage, ConsoleToolDiffRow),
        ):
            if self.selection_manager.state.active:
                self.release_mouse()
            self.selection_manager.cancel()
            self._selection_origin_row = None
            # PR #1813 review (Qodo bug 5 + whole-branch warning): a removed
            # row must drop keyboard-selection mode EAGERLY -- lingering
            # state kept the hint advertising a mode that no longer existed
            # until the next keypress noticed the detached row.
            if self._kb_selection_row is not None:
                self._exit_keyboard_selection(clear=False)

    async def _reconcile_rows(self, rows: list[_TranscriptRow]) -> None:
        desired_keys = [row.key for row in rows]
        desired_key_set = set(desired_keys)

        removals: list[Widget] = []
        for stale_key in [
            key for key in self._row_widgets if key not in desired_key_set
        ]:
            removals.append(self._row_widgets.pop(stale_key))
            self._row_signatures.pop(stale_key, None)
            self._row_build_counts.pop(stale_key, None)

        replacements: dict[str, Widget] = {}
        for row in rows:
            widget = self._row_widgets.get(row.key)
            if widget is None or self._row_signatures.get(row.key) == row.signature:
                continue
            if row.kind == "assistant-turn" and isinstance(
                widget, ConsoleAssistantTurnWidget
            ):
                await self._sync_assistant_turn_widget(widget, row)
                self._row_signatures[row.key] = row.signature
                continue
            updated_widget = self._update_row_widget(widget, row)
            if updated_widget is widget:
                self._row_signatures[row.key] = row.signature
                continue
            removals.append(widget)
            replacements[row.key] = updated_widget
            self._row_widgets.pop(row.key, None)
            self._row_signatures.pop(row.key, None)

        # Textual's remove_children() prunes all supplied direct children in
        # one DOM operation.  A session swap therefore has one await instead
        # of two awaits per message (rule + body).
        if removals:
            for widget in removals:
                self._cancel_selection_if_row_removed(widget)
            await self.remove_children(removals)

        pending_widgets: list[Widget] = []
        pending_rows: list[_TranscriptRow] = []

        async def _mount_pending(*, before: Widget | None) -> bool:
            """Mount one contiguous missing run and validate attachment."""
            if not pending_widgets:
                return True
            if self._closing or self._pruning or not self.is_attached:
                for pending_row in pending_rows:
                    self._row_widgets.pop(pending_row.key, None)
                    self._row_signatures.pop(pending_row.key, None)
                return False
            if before is None:
                await self.mount(*pending_widgets)
            else:
                await self.mount(*pending_widgets, before=before)
            if any(widget.parent is not self for widget in pending_widgets):
                for pending_row in pending_rows:
                    self._row_widgets.pop(pending_row.key, None)
                    self._row_signatures.pop(pending_row.key, None)
                return False
            pending_widgets.clear()
            pending_rows.clear()
            return True

        for row in rows:
            if self._closing or self._pruning or not self.is_attached:
                return
            widget = self._row_widgets.get(row.key)
            if widget is not None:
                if not await _mount_pending(before=widget):
                    return
                continue
            widget = replacements.pop(row.key, None)
            if widget is None:
                widget = self._build_row_widget(row, track=True)
            self._row_widgets[row.key] = widget
            self._row_signatures[row.key] = row.signature
            pending_widgets.append(widget)
            pending_rows.append(row)

        if pending_widgets:
            try:
                pill = self.query_one(
                    "#console-transcript-jump-pill", ConsoleTranscriptJumpPill
                )
            except NoMatches:
                pill = None
            if not await _mount_pending(before=pill):
                return
        # Preserve the reorder contract for branch changes after the batched
        # mount/remove work above. TASK-15453 established that move_child is
        # costly, so only move rows that are not already at their target index.
        # Read children fresh on every iteration because an earlier move can
        # shift the remaining indices.
        previous_widget: Widget | None = None
        for index, row in enumerate(rows):
            widget = self._row_widgets[row.key]
            already_in_position = (
                index < len(self.children) and self.children[index] is widget
            )
            if not already_in_position:
                if previous_widget is None:
                    self.move_child(widget, before=0)
                else:
                    self.move_child(widget, after=previous_widget)
            previous_widget = widget
        self._paint_debug_dump("after-reconcile")

    async def _sync_assistant_turn_widget(
        self,
        widget: ConsoleAssistantTurnWidget,
        row: _TranscriptRow,
    ) -> None:
        """Sync one composite row without remounting its answer or shell."""
        assert row.assistant_turn is not None and row.nested_rows
        assistant = row.nested_rows[0].message
        assert assistant is not None
        presentation = self._message_presentation(assistant)
        header = widget.header_widget
        if isinstance(header, ConsoleMessageHeader):
            header.sync_header(
                assistant,
                presentation,
                self._console_speech_state(assistant.id),
            )
        answer = widget.answer_widget
        if isinstance(answer, ConsoleMarkdownMessage):
            answer.sync_message(
                assistant,
                presentation,
                selected=row.selected,
                speech_state=self._console_speech_state(assistant.id),
            )
        elif isinstance(answer, ConsoleTranscriptMessage):
            answer.sync_message(
                assistant,
                presentation,
                selected=row.selected,
                speech_state=self._console_speech_state(assistant.id),
            )

        if (
            getattr(widget, "_console_activity_signature", None)
            != row.activity_signature
        ):
            await self._sync_activity_widgets(widget, row)
            widget._console_activity_signature = row.activity_signature
        if getattr(widget, "_console_adjunct_signature", None) != row.adjunct_signature:
            adjuncts = tuple(
                self._build_row_widget(nested_row, track=False)
                for nested_row in row.nested_rows[1:]
            )
            if widget.adjunct_stack.children:
                self._cancel_selection_if_row_removed(widget.adjunct_stack)
                await widget.adjunct_stack.remove_children()
            if adjuncts:
                await widget.adjunct_stack.mount(*adjuncts)
            widget._console_adjunct_signature = row.adjunct_signature

    def _build_row_widget(self, row: _TranscriptRow, *, track: bool) -> Widget:
        if track:
            self._row_build_counts[row.key] = self._row_build_counts.get(row.key, 0) + 1
        if row.kind == "rule":
            # Rule separators do not need stable IDs; using None avoids
            # DuplicateIds when a recompose/race leaves the previous end-rule
            # widget in the DOM briefly while the new one is mounted.
            return Static(
                row.renderable,
                classes="console-transcript-rule",
            )
        if row.kind == "banner":
            # Non-interactive, render-derived summary banner (never a tree node).
            return Static(
                row.renderable,
                classes="console-transcript-summary-banner",
            )
        if row.kind == "empty":
            assert row.card_state is not None
            return ConsoleTranscriptEmptyPanel(
                row.card_state,
                provider_action_label=row.action_label,
                provider_action_tooltip=row.action_tooltip,
            )
        if row.kind == "action-help":
            return Static(
                row.renderable,
                id=self._row_widget_id(row),
                classes="console-transcript-action-guide",
            )
        if row.kind == "original-attempt" and row.message is not None:
            return Static(
                row.renderable,
                id=f"console-original-attempt-{row.message.id}",
                classes="console-transcript-original-attempt",
            )
        if row.kind == "assistant-turn":
            return self._build_assistant_turn_widget(row)
        if row.kind == "message" and row.message is not None:
            return self._build_message_widget(row.message, selected=row.selected)
        if (
            row.kind == "diff"
            and row.message is not None
            and row.message.tool_diff is not None
        ):
            return ConsoleToolDiffRow(row.message.id, row.message.tool_diff)
        if row.kind == "annotations" and row.message is not None:
            return ConsoleAnnotationMarker(
                row.renderable,
                anchor_message_id=row.message.id,
                id=f"console-annotations-{row.message.id}",
                classes="console-transcript-annotations",
            )
        if row.kind == "citations" and row.message is not None:
            button = Button(
                row.renderable,
                id=f"console-citation-sources-{row.message.id}",
                classes="console-transcript-citation-sources",
            )
            button.native_message_id = row.message.id
            return button
        if row.kind == "image" and row.image_spec is not None:
            return self._image_row_widget(row.image_spec)
        if row.kind == "generation-card" and row.generation_card_spec is not None:
            assert row.message is not None
            return ConsoleGenerationCard(
                row.generation_card_spec,
                actions=self._action_groups(row.message).media,
            )
        if row.kind == "video-card" and row.video_card_spec is not None:
            assert row.message is not None
            return ConsoleVideoCard(
                row.video_card_spec,
                actions=self._action_groups(row.message).media,
            )
        if row.kind == "actions" and row.message is not None:
            return self._action_row(row.message)
        raise ValueError(f"Unsupported transcript row: {row}")

    def _build_message_widget(
        self,
        message: ConsoleChatMessage,
        *,
        selected: bool,
        show_header: bool = True,
    ) -> Widget:
        """Build one message body through the shared standalone/nested seam."""
        review_run_id = getattr(message, "change_review_run_id", None)
        if (
            review_run_id
            and self._change_review_provider_factory is not None
            and bool(get_cli_setting("console", "turn_file_cards", True))
        ):
            return ConsoleTurnFileCard(
                str(message.content),
                str(review_run_id),
                self._change_review_provider_factory,
                message_id=message.id,
                selected=selected,
                id=f"console-turn-file-card-{message.id}",
            )
        presentation = self._message_presentation(message)
        if (
            message.role is ConsoleMessageRole.ASSISTANT
            and self._assistant_markdown_enabled()
        ):
            return ConsoleMarkdownMessage(
                message,
                presentation,
                selected=selected,
                speech_state=self._console_speech_state(message.id),
                show_header=show_header,
            )
        return ConsoleTranscriptMessage(
            message,
            presentation,
            selected=selected,
            speech_state=self._console_speech_state(message.id),
            show_header=show_header,
        )

    def _activity_components(
        self,
        activity: ConsoleChatMessage | ConsoleThinkingActivityRef,
        owned_rows: tuple[_TranscriptRow, ...],
    ) -> _ActivityComponents:
        """Build one disclosure's children through the shared transcript builders."""
        if isinstance(activity, ConsoleThinkingActivityRef):
            activity_id = activity.activity_id
            expanded = activity_id in self._expanded_tool_output_ids
            detail = self.thinking_detail_text(activity_id)
            detail_widgets = (
                (
                    Static(
                        Content(detail),
                        id=f"console-thinking-detail-{activity_id}",
                        classes="console-thinking-detail",
                        markup=False,
                    ),
                )
                if expanded and detail is not None
                else ()
            )
            return _ActivityComponents(
                presentation=ConsoleActivityPresentation(
                    "thinking", activity.label, activity.status
                ),
                action_widgets=(),
                detail_widgets=detail_widgets,
                detail_available=detail is not None,
                action_signature=(),
                detail_signature=(
                    (("thinking-detail", activity_id, detail),) if expanded else ()
                ),
            )
        presentation = activity.activity_presentation or ConsoleActivityPresentation(
            "activity", "Activity", "done"
        )
        action_widgets: list[Widget] = []
        detail_widgets: list[Widget] = []
        action_signature: list[tuple] = []
        detail_signature: list[tuple] = []
        for owned_row in owned_rows:
            if owned_row.kind in {"actions", "action-help"}:
                action_widgets.append(self._build_row_widget(owned_row, track=False))
                action_signature.append(owned_row.signature)
                continue
            if owned_row.kind == "message":
                if owned_row.message is not None and owned_row.message.content.strip():
                    detail_widgets.append(
                        self._build_message_widget(
                            owned_row.message,
                            selected=False,
                            show_header=False,
                        )
                    )
                    detail_signature.append(owned_row.signature)
                continue
            detail_widgets.append(self._build_row_widget(owned_row, track=False))
            detail_signature.append(owned_row.signature)
        if not detail_widgets and _activity_is_expandable(activity, owned_rows):
            # Preserve lazy collapsed rendering while telling the disclosure
            # that hidden full output or a diff exists. The expanded refresh
            # replaces this sentinel with the real shared message/diff rows.
            detail_widgets.append(
                Static("", classes="console-activity-detail-placeholder")
            )
            detail_signature.append(("activity-detail-placeholder", activity.id))
        return _ActivityComponents(
            presentation=presentation,
            action_widgets=tuple(action_widgets),
            detail_widgets=tuple(detail_widgets),
            detail_available=bool(detail_widgets),
            action_signature=tuple(action_signature),
            detail_signature=tuple(detail_signature),
        )

    def _build_activity_disclosure(
        self,
        activity: ConsoleChatMessage | ConsoleThinkingActivityRef,
        owned_rows: tuple[_TranscriptRow, ...],
    ) -> ConsoleActivityDisclosure:
        """Build and stamp one disclosure for later same-id reconciliation."""
        components = self._activity_components(activity, owned_rows)
        activity_id = (
            activity.id
            if isinstance(activity, ConsoleChatMessage)
            else activity.activity_id
        )
        disclosure = ConsoleActivityDisclosure(
            activity_id,
            components.presentation.label,
            components.presentation.status,
            expanded=activity_id in self._expanded_tool_output_ids,
            selected=activity_id == self.selected_message_id,
            action_widgets=components.action_widgets,
            detail_widgets=components.detail_widgets,
            detail_available=components.detail_available,
        )
        disclosure._console_action_signature = components.action_signature
        disclosure._console_detail_signature = components.detail_signature
        return disclosure

    def _build_activity_widgets(self, row: _TranscriptRow) -> tuple[Widget, ...]:
        """Build owned disclosures from the same rows used by standalone messages."""
        turn = row.assistant_turn
        assert turn is not None
        return tuple(
            self._build_activity_disclosure(activity, owned_rows)
            for activity, owned_rows in zip(row.activity_items, row.activity_rows)
        )

    async def _sync_activity_widgets(
        self,
        widget: ConsoleAssistantTurnWidget,
        row: _TranscriptRow,
    ) -> None:
        """Reconcile same-id disclosures without detaching their focused headers."""
        turn = row.assistant_turn
        assert turn is not None
        disclosures = list(widget.activity_stack.children)
        current_ids = tuple(
            disclosure.activity_message_id
            for disclosure in disclosures
            if isinstance(disclosure, ConsoleActivityDisclosure)
        )
        next_ids = tuple(
            activity.id
            if isinstance(activity, ConsoleChatMessage)
            else activity.activity_id
            for activity in row.activity_items
        )
        if len(disclosures) != len(current_ids) or current_ids != next_ids:
            by_id = {
                disclosure.activity_message_id: disclosure
                for disclosure in disclosures
                if isinstance(disclosure, ConsoleActivityDisclosure)
            }
            stale = [
                disclosure
                for activity_id, disclosure in by_id.items()
                if activity_id not in next_ids
            ]
            if stale:
                for disclosure in stale:
                    self._cancel_selection_if_row_removed(disclosure)
                await widget.activity_stack.remove_children(stale)
            disclosures = []
            for index, (activity, owned_rows) in enumerate(
                zip(row.activity_items, row.activity_rows)
            ):
                activity_id = (
                    activity.id
                    if isinstance(activity, ConsoleChatMessage)
                    else activity.activity_id
                )
                disclosure = by_id.get(activity_id)
                if disclosure is None:
                    disclosure = self._build_activity_disclosure(activity, owned_rows)
                    await widget.activity_stack.mount(disclosure)
                disclosures.append(disclosure)
                if widget.activity_stack.children[index] is not disclosure:
                    widget.activity_stack.move_child(disclosure, before=index)

        for disclosure, activity, owned_rows in zip(
            disclosures, row.activity_items, row.activity_rows
        ):
            assert isinstance(disclosure, ConsoleActivityDisclosure)
            activity_id = (
                activity.id
                if isinstance(activity, ConsoleChatMessage)
                else activity.activity_id
            )
            components = self._activity_components(activity, owned_rows)
            if (
                getattr(disclosure, "_console_action_signature", None)
                != components.action_signature
            ):
                if disclosure.action_stack.children:
                    await disclosure.action_stack.remove_children()
                if components.action_widgets:
                    await disclosure.action_stack.mount(*components.action_widgets)
                disclosure._console_action_signature = components.action_signature
            if (
                getattr(disclosure, "_console_detail_signature", None)
                != components.detail_signature
            ):
                self._cancel_selection_if_row_removed(disclosure.detail_stack)
                await disclosure.replace_detail_widgets(components.detail_widgets)
                disclosure._console_detail_signature = components.detail_signature
            disclosure._has_actions = bool(components.action_widgets)
            disclosure.detail_available = components.detail_available
            disclosure.sync_activity(
                components.presentation.label,
                components.presentation.status,
                expanded=activity_id in self._expanded_tool_output_ids,
                selected=activity_id == self.selected_message_id,
            )

    def _build_assistant_turn_widget(self, row: _TranscriptRow) -> Widget:
        """Build one Assistant-owned surface from a composite transcript row."""
        turn = row.assistant_turn
        assert turn is not None and row.nested_rows
        assistant = row.nested_rows[0].message
        assert assistant is not None
        presentation = self._message_presentation(assistant)
        header = ConsoleMessageHeader(
            assistant,
            presentation,
            self._console_speech_state(assistant.id),
            markdown=self._assistant_markdown_enabled(),
        )
        answer = self._build_message_widget(
            assistant,
            selected=row.selected,
            show_header=False,
        )
        adjuncts = tuple(
            self._build_row_widget(nested_row, track=False)
            for nested_row in row.nested_rows[1:]
        )
        widget = ConsoleAssistantTurnWidget(
            assistant.id,
            header,
            self._build_activity_widgets(row),
            answer,
            adjuncts,
        )
        widget._console_activity_signature = row.activity_signature
        widget._console_adjunct_signature = row.adjunct_signature
        return widget

    def _image_row_widget(self, spec: ConsoleImageRowSpec) -> Widget:
        """Build the mounted widget for one inline-image row."""
        widget: Widget | None = None
        if spec.mode == "graphics" and spec.pil is not None:
            try:
                from textual_image.widget import Image as _GraphicsImage

                widget = _GraphicsImage(spec.pil, id=f"console-image-{spec.message_id}")
                # Explicit fitted cell size, not just max-width/max-height:
                # textual_image's "auto" sizing resolves its render region
                # from the parent's settled layout, and mounting a tick before
                # that settles can ask the renderer to scale into a transient
                # 0-width/height region - which PIL's resize() raises on. Fixed
                # ints resolve without waiting on layout, sidestepping the race
                # (the personas avatar preview uses the same guard).
                w_cells, h_cells = fit_image_cell_size(
                    spec.pil.width, spec.pil.height, PIXELS_MAX_COLS, PIXELS_MAX_LINES
                )
                widget.styles.width = w_cells
                widget.styles.height = h_cells
            except Exception:
                logger.opt(exception=True).warning(
                    "textual-image unavailable; falling back to pixels row."
                )
                widget = None
        if widget is None:
            pixels = spec.pixels
            if pixels is None and spec.pil is not None:
                # Graphics import failed and nothing was cached: thumbnail a
                # copy before building, mirroring the cache's bounded build
                # (`ConsoleImageRenderCache.get_pixels`) so this fallback
                # never runs `Pixels.from_image` on the full ≤1024px image.
                scaled = spec.pil.copy()
                scaled.thumbnail(
                    (PIXELS_MAX_COLS, PIXELS_MAX_LINES * 2), PILImage.Resampling.LANCZOS
                )
                pixels = Pixels.from_image(scaled)
            widget = Static(
                pixels if pixels is not None else "",
                id=f"console-image-{spec.message_id}",
            )
            # Pixels render at their baked half-block size; a max cap is safe
            # here (no textual_image auto-sizing race).
            widget.styles.max_width = PIXELS_MAX_COLS
            widget.styles.max_height = PIXELS_MAX_LINES
        widget.add_class("console-transcript-image")
        return widget

    def _update_row_widget(self, widget: Widget, row: _TranscriptRow) -> Widget:
        if (
            row.kind == "message"
            and row.message is not None
            and isinstance(widget, ConsoleTurnFileCard)
        ):
            # A card row's signature (`_message_row_signature`, shared with
            # every other "message" kind row) folds in `selected` -- so
            # moving keyboard/click selection onto or off this row DOES
            # change the signature and reaches this method. Marker text and
            # run id are fixed at append time (TOOL markers never mutate),
            # so a mismatch here can only mean the row identity itself
            # changed underneath the same key -- fall through to a full
            # rebuild for that case; otherwise sync in place. Rebuilding on
            # every selection flip would collapse whatever diffs were
            # expanded and drop the diff cache for no reason.
            review_run_id = getattr(row.message, "change_review_run_id", None)
            still_a_card = (
                review_run_id is not None
                and self._change_review_provider_factory is not None
                and bool(get_cli_setting("console", "turn_file_cards", True))
            )
            if (
                still_a_card
                and widget.marker_text == str(row.message.content)
                and widget.run_id == str(review_run_id)
            ):
                widget.update_selected(row.selected)
                return widget
            return self._build_row_widget(row, track=True)
        if (
            row.kind == "message"
            and row.message is not None
            and isinstance(widget, ConsoleMarkdownMessage)
        ):
            widget.sync_message(
                row.message,
                self._message_presentation(row.message),
                selected=row.selected,
                speech_state=self._console_speech_state(row.message.id),
            )
            return widget
        if (
            row.kind == "message"
            and row.message is not None
            and isinstance(widget, ConsoleTranscriptMessage)
        ):
            widget.sync_message(
                row.message,
                self._message_presentation(row.message),
                selected=row.selected,
                speech_state=self._console_speech_state(row.message.id),
            )
            return widget
        if row.kind == "empty" and isinstance(widget, ConsoleTranscriptEmptyPanel):
            assert row.card_state is not None
            widget.sync_card_state(
                row.card_state,
                provider_action_label=row.action_label,
                provider_action_tooltip=row.action_tooltip,
            )
            return widget
        return self._build_row_widget(row, track=True)

    @staticmethod
    def _row_widget_id(row: _TranscriptRow) -> str:
        return "console-transcript-row-" + row.key.replace(":", "-")

    def _message_signature_token(
        self, message: ConsoleChatMessage, *, selected: bool
    ) -> tuple:
        """Return a cheap change-token covering every render-signature input.

        Captures the exact inputs of ``_message_render_text`` plus the
        non-render signature fields (status, selection, variant identity), so
        token equality guarantees the cached expensive signature is current.
        Unchanged messages keep the same ``str``/``bytes`` object references
        across store snapshots, so tuple comparison short-circuits on
        identity; content edits/streaming rebinds produce new objects and are
        caught by value comparison (never by length alone -- an equal-length
        edit still misses the cache).

        Args:
            message: Transcript message to fingerprint.
            selected: Whether the message row renders as selected.

        Returns:
            Hashable token tuple; any render-affecting change alters it.
        """
        variants = message.variants
        if variants is None:
            variants_token = None
            content = message.content
        else:
            variants_token = (
                variants.selected_index,
                tuple(variant.id for variant in variants.variants),
            )
            content = variants.current.content
        attachments_token = tuple(
            (
                attachment.display_name,
                attachment.mime_type,
                attachment.position,
                None if attachment.data is None else len(attachment.data),
            )
            for attachment in (getattr(message, "attachments", ()) or ())
        )
        presentation = self._message_presentation(message)
        return (
            message.role,
            message.status,
            selected,
            content,
            variants_token,
            attachments_token,
            message.attachment_label,
            message.image_mime_type,
            None if message.image_data is None else len(message.image_data),
            message.citation_presentation,
            # Load-bearing: the activity line is stamped on a `replace()`
            # copy whose every OTHER field is identical tick to tick, so a
            # token without it would hit this cache and the elapsed figure
            # would freeze at the first value it ever rendered.
            message.live_activity,
            presentation.revision_token,
            self._console_speech_state(message.id),
        )

    def _cached_message_row_signature(
        self, message: ConsoleChatMessage, *, selected: bool
    ) -> tuple:
        """Return the row signature, deriving it only when the message changed.

        Args:
            message: Transcript message for the row.
            selected: Whether the message row renders as selected.

        Returns:
            The (possibly cached) expensive row signature tuple.
        """
        token = self._message_signature_token(message, selected=selected)
        cached = self._message_signature_cache.get(message.id)
        if cached is not None and cached[0] == token:
            return cached[1]
        signature = self._message_row_signature(message, selected=selected)
        self._message_signature_cache[message.id] = (token, signature)
        self._signature_compute_counts[message.id] = (
            self._signature_compute_counts.get(message.id, 0) + 1
        )
        return signature

    def _turn_activity_target_id(self) -> str | None:
        """Id of the row the activity line would render on, or ``None``.

        The last in-flight assistant row with no content yet -- the one
        ``_message_body`` would otherwise render blank (or, once streaming
        starts, as ``Generating…``). A row that already has text is showing
        the real reply and must never be overwritten by a status line.
        """
        for message in reversed(self._messages):
            if _row_is_in_flight(message):
                content = (
                    message.variants.current.content
                    if message.variants is not None
                    else message.content
                )
                return message.id if not content.strip() else None
        return None

    def apply_turn_activity(self, activity: str) -> str:
        """Store this poll tick's live activity line; return what will show.

        Returns the EFFECTIVE value -- ``""`` whenever no row is eligible --
        because the screen folds this return into its transcript refresh
        key. A stale ``running`` snapshot (a run that died without a
        terminal publish) therefore cannot tick that key once a second on
        an otherwise idle transcript: task-15664 AC#2 forbids repainting on
        a timer when nothing is live, and this is the check that keeps that
        true no matter what the bridge last published.

        Args:
            activity: The derived line (``console_turn_activity_text``), or
                ``""`` when nothing is live.

        Returns:
            The line that will actually render, or ``""``.
        """
        effective = activity if activity and self._turn_activity_target_id() else ""
        # Unconditional, including the empty case: found by mutation, a row
        # that stays in flight after its run dies (no terminal publish) is
        # exactly where a retained line would sit forever, frozen at its
        # last elapsed -- the frozen look this feature exists to remove.
        self._turn_activity = effective
        return effective

    def _with_turn_activity(
        self, message: ConsoleChatMessage, target_id: str | None
    ) -> ConsoleChatMessage:
        """Return ``message`` carrying this tick's activity line, when it owns it.

        Applied at the ONE walk that plans rows -- the same seam, and for the
        same reason, as ``_with_expanded_tool_output``: the row renderable,
        its cached signature and its action row must all see the identical
        message, or a row would render one thing while its signature claimed
        another and never repaint.

        **Exactly one row, deliberately.** Found by mutation: stamping every
        message instead of just the in-flight one is invisible to every
        display assertion (a row with content never renders the line, and
        only assistant rows can) yet puts ``live_activity`` into EVERY row's
        signature -- so the whole transcript re-derives and re-syncs once a
        second for the entire turn.

        Returns ``message`` UNCHANGED (same object) when there is nothing to
        show, so a transcript with no live turn is byte-for-byte what it was
        before this feature existed.

        Args:
            message: The transcript message about to be rendered.
            target_id: The row that owns the line this pass, hoisted out of
                the loop by the caller (``_turn_activity_target_id``), or
                ``None`` when no row does.

        Returns:
            ``message``, or a render-only copy carrying ``live_activity``.
        """
        if target_id is None or message.id != target_id:
            return message
        return replace(message, live_activity=self._turn_activity)

    def _with_expanded_tool_output(
        self, message: ConsoleChatMessage
    ) -> ConsoleChatMessage:
        """Return ``message`` showing its full tool result, when expanded.

        TASK-1860. Applied at the ONE walk that plans rows, so the row
        renderable, its cached signature and its action row all see the same
        message -- a row that renders expanded while its signature says
        collapsed would never repaint.

        Args:
            message: The transcript message about to be rendered.

        Returns:
            ``message`` unchanged, or a copy whose ``content`` carries the
            full tool result when this row is currently expanded.
        """
        full = message.tool_output_full
        if not full or message.id not in self._expanded_tool_output_ids:
            return message
        head, separator, _preview = message.content.partition(" \u2192 ")
        expanded = (
            f"{head}{separator}{full}" if separator else f"{message.content}\n{full}"
        )
        return replace(message, content=expanded)

    @on(Button.Pressed)
    async def _intercept_transcript_action_press(self, event: Button.Pressed) -> None:
        """Handle transcript-owned actions and close More before other presses.

        Expansion is view state owned by this widget -- it never reaches the
        store and nothing outside the transcript needs to know about it -- so
        routing it through the screen's action dispatch would add a hop that
        carries no information. More opens its captured-target popup; every
        other action first detaches that popup, then still bubbles to
        `ChatScreen`.
        """
        button_id = event.button.id or ""
        more_prefix = "console-message-action-more-"
        if button_id.startswith(more_prefix):
            event.stop()
            await self._open_message_more_menu(
                button_id.removeprefix(more_prefix), event.button
            )
            return
        await self.dismiss_message_more_menu(restore_focus=False)
        tool_prefix = "console-message-action-tool-output-"
        if button_id.startswith(tool_prefix):
            event.stop()
            self.toggle_tool_output(button_id.removeprefix(tool_prefix))

    @on(ConsoleActivityActivated)
    def _on_activity_activated(self, event: ConsoleActivityActivated) -> None:
        """Keep disclosure controls on the original message selection seam."""
        event.stop()
        if event.message_id in self._thinking_activity_refs:
            self._manual_thinking_disclosures.add(event.message_id)
            self._pending_thinking_auto_collapse.discard(event.message_id)
        self.select_message(event.message_id)
        if event.toggle_requested:
            self.toggle_tool_output(event.message_id)

    def toggle_tool_output(self, message_id: str) -> None:
        """Expand or collapse one TOOL marker's full result.

        Args:
            message_id: Id of the marker row to toggle. Unknown ids are
                harmless -- the row simply renders collapsed, and
                ``set_messages`` prunes ids that leave the transcript.
        """
        if not self._activity_can_expand(message_id):
            return
        if message_id in self._thinking_activity_refs:
            self._manual_thinking_disclosures.add(message_id)
            self._pending_thinking_auto_collapse.discard(message_id)
        if message_id in self._expanded_tool_output_ids:
            self._expanded_tool_output_ids.discard(message_id)
        else:
            self._expanded_tool_output_ids.add(message_id)
        self.call_later(self.refresh_messages)

    def _owned_activity_rows(self, message_id: str) -> tuple[_TranscriptRow, ...]:
        """Return planned nested rows for an owned activity marker."""
        for row in self._transcript_rows():
            turn = row.assistant_turn
            if row.kind != "assistant-turn" or turn is None:
                continue
            for activity, owned_rows in zip(row.activity_items, row.activity_rows):
                activity_id = (
                    activity.id
                    if isinstance(activity, ConsoleChatMessage)
                    else activity.activity_id
                )
                if activity_id == message_id:
                    return owned_rows
        return ()

    def _activity_can_expand(self, message_id: str) -> bool:
        """Resolve the one disclosure-detail fact used by click, keys, and `o`."""
        message = next(
            (candidate for candidate in self._messages if candidate.id == message_id),
            None,
        )
        if message_id in self._thinking_activity_refs:
            return self.thinking_detail_text(message_id) is not None
        return message is not None and _activity_is_expandable(
            message,
            self._owned_activity_rows(message_id),
        )

    def _message_row_signature(
        self, message: ConsoleChatMessage, *, selected: bool
    ) -> tuple:
        variants_signature = None
        if message.variants is not None:
            variants_signature = (
                message.variants.selected_index,
                tuple(variant.id for variant in message.variants.variants),
            )
        presentation = self._message_presentation(message)
        return (
            "message",
            _message_render_text(
                message,
                selected=selected,
                presentation=presentation,
            ),
            message.status,
            selected,
            variants_signature,
            # Found by mutation: a MARKDOWN row (the default assistant
            # renderer) carries the activity line in its HEADER, which this
            # signature never renders -- it renders the PLAIN row. So with
            # this field absent the elapsed still ticked, but only as a side
            # effect of `_message_render_text` happening to embed the same
            # text; disabling the plain renderer's activity branch froze the
            # markdown row's elapsed at the first value it painted, with
            # every display test still green. Named explicitly here so the
            # two renderers cannot silently depend on each other again.
            message.live_activity,
            presentation.revision_token,
            self._console_speech_state(message.id),
        )

    def _generation_browsed_index(self, message_id: str, variant_count: int) -> int:
        """Return the screen's ephemeral browsed-variant index for ``message_id``.

        Reads directly off the owning screen's ``_generation_browse`` map
        (the same ephemeral, never-persisted state ``ChatScreen`` uses to
        build ``ConsoleGenerationCardSpec``s) rather than the card-spec map
        this widget also holds, since a card spec can be absent for a
        message currently in "hidden" view mode while the action row (and
        its `<`/`>`/Keep gating) still needs the real browsed index. Falls
        back to 0 -- the canonical variant -- when unmounted (bare
        unit-construction in tests) or the screen hasn't created its browse
        map yet, both of which correctly describe "nothing browsed".
        """
        try:
            browse = getattr(self.screen, "_generation_browse", None)
        except NoScreen:
            browse = None
        browsed_index = (browse or {}).get(message_id, 0)
        if not (0 <= browsed_index < variant_count):
            return 0
        return browsed_index

    def _console_ephemeral_active(self) -> bool:
        """Return whether the owning screen's active session is temporary.

        Mirrors ``_console_tts_speaking_message_id`` below: reads the
        screen's accessor so the message-action row (Save Image) reads the
        same flag the composer menu and workbench state already do. Falls
        back to ``False`` when unmounted (bare unit-construction in tests)
        or the screen hasn't defined the accessor.
        """
        try:
            screen = self.screen
        except NoScreen:
            return False
        is_ephemeral = getattr(screen, "_console_active_session_is_ephemeral", None)
        if not callable(is_ephemeral):
            return False
        return bool(is_ephemeral())

    def _console_tts_speaking_message_id(self) -> str | None:
        """Return the owning screen's ephemeral "currently speaking" id.

        Mirrors ``_generation_browsed_index`` above (task-559 unit 2): reads
        the screen's ``_console_speaking_message_id`` -- purely screen-side,
        never-persisted state set/cleared by ``ChatScreen.handle_console_
        message_action`` around the speak/speak-stop actions -- so the ⏹
        stop swap survives whatever transcript instance a recompose mounts.
        Falls back to ``None`` when unmounted (bare unit-construction in
        tests) or the screen hasn't set the attribute yet.
        """
        try:
            return getattr(self.screen, "_console_speaking_message_id", None)
        except NoScreen:
            return None

    def _generation_action_kwargs(self, message: ConsoleChatMessage) -> dict[str, Any]:
        """Return the ``available_actions()`` generation/video kwargs for ``message``.

        Empty for a plain message, so ``available_actions(message)`` sees its
        old, un-keyworded call shape unchanged (regression guard). A video
        message contributes ``video_file_available`` from the current video
        card specs (the action row's ▶/Save enablement -- task-3401.5).
        """
        variant_count = len(message.generation_metadata)
        kwargs: dict[str, Any] = {}
        if variant_count > 0:
            kwargs["generation_variant_count"] = variant_count
            kwargs["generation_browsed_index"] = self._generation_browsed_index(
                message.id, variant_count
            )
        if getattr(message, "video_metadata", None) is not None:
            spec = self._video_card_specs.get(message.id)
            kwargs["video_file_available"] = bool(
                spec is not None and spec.status == "ready"
            )
        return kwargs

    def _action_groups(self, message: ConsoleChatMessage):
        return ConsoleMessageActionService().action_groups(
            message,
            speaking_message_id=self._console_tts_speaking_message_id(),
            original_attempt_available=bool(
                message.citation_presentation
                and message.citation_presentation.original_attempt_available
            ),
            ephemeral=self._console_ephemeral_active(),
            fork_eligibility=self._fork_eligibility_by_message_id.get(
                message.id, ConsoleForkEligibility(True)
            ),
            **self._generation_action_kwargs(message),
        )

    def _action_row_signature(self, message: ConsoleChatMessage) -> tuple:
        actions = []
        for action in self._action_groups(message).primary:
            if action.action_id == "feedback":
                actions.append(("feedback-up", "👍", True, ""))
                actions.append(("feedback-down", "👎", True, ""))
                continue
            actions.append(
                (
                    action.action_id,
                    action.label,
                    action.enabled,
                    action.disabled_reason or "",
                )
            )
        return ("actions", message.id, tuple(actions))

    def _action_guide(self, message: ConsoleChatMessage) -> str:
        """Return the legend naming row actions and the header speech action.

        Speech remains in the guide because its button moved to the persistent
        message header; only the selected-row button is removed.
        """
        actions = list(self._action_groups(message).primary)
        guide = action_row_guide(actions)
        fork = next((action for action in actions if action.action_id == "fork"), None)
        if fork is not None and not fork.enabled and fork.disabled_reason:
            return f"{guide} · Fork unavailable — {fork.disabled_reason}"
        return guide

    def _action_row(self, message: ConsoleChatMessage) -> Horizontal:
        buttons: list[Button] = []
        for action in self._action_groups(message).primary:
            buttons.append(self._action_button(message, action))
        return Horizontal(
            *buttons,
            id=f"console-message-actions-{message.id}",
            classes="console-transcript-action-row",
        )

    @staticmethod
    def _plain_action_row(message: ConsoleChatMessage) -> str:
        return ConsoleMessageActionService().plain_action_row(message)

    @staticmethod
    def _action_button(
        message: ConsoleChatMessage, action: ConsoleMessageAction
    ) -> Button:
        button = ConsoleTranscriptActionButton(
            action.label,
            id=f"console-message-action-{action.action_id}-{message.id}",
            classes="console-transcript-action-button",
            disabled=not action.enabled,
        )
        if action.disabled_reason:
            button.tooltip = action.disabled_reason
        else:
            button.tooltip = _ACTION_TOOLTIPS.get(action.action_id)
        button.console_action_id = action.action_id
        button.console_message_id = message.id
        return button

    async def _open_message_more_menu(self, message_id: str, opener: Button) -> None:
        """Mount the overflow menu bound to the opener's captured target."""
        message = self._message_by_id(message_id)
        if message is None or self.selected_message_id != message_id:
            return
        await self.dismiss_message_more_menu(restore_focus=False)
        actions = self._action_groups(message).overflow
        if not actions:
            return
        region = opener.region
        menu_width = ConsoleMessageMoreMenu.MENU_WIDTH
        menu_height = len(actions) + 2
        self.screen.mount(
            ConsoleMessageMoreMenu(
                message_id=message_id,
                actions=actions,
                owner=self,
                opener_button_id=opener.id or "",
                screen_x=max(
                    self.region.x, min(region.x, self.region.right - menu_width)
                ),
                screen_y=max(
                    self.region.y, min(region.bottom, self.region.bottom - menu_height)
                ),
            )
        )

    async def dismiss_message_more_menu(self, *, restore_focus: bool = True) -> None:
        """Detach overflow UI without dispatching an action."""
        menus = message_more_menus_on_screen(self.screen) if self.is_mounted else []
        opener_id = menus[0].opener_button_id if menus else ""
        for menu in menus:
            await menu.remove()
        if restore_focus:
            self._restore_message_action_focus(opener_id)

    def _restore_message_action_focus(self, opener_button_id: str) -> None:
        if opener_button_id:
            for opener in self.query(f"#{opener_button_id}"):
                if opener.is_mounted:
                    opener.focus(scroll_visible=False)
                    return
        if self.selected_message_id:
            for row in self.query(f"#console-message-{self.selected_message_id}"):
                if row.is_mounted:
                    row.scroll_visible(animate=False)
                    self.focus(scroll_visible=False)
                    return
        for composer in self.screen.query("#console-native-composer"):
            composer.focus(scroll_visible=False)
            return

    def dispatch_captured_message_action(
        self, message_id: str, action_id: str, *, opener_button_id: str
    ) -> None:
        """Post one controller-compatible action after the menu detached."""
        self._restore_message_action_focus(opener_button_id)
        button = Button("", id=f"console-message-action-{action_id}-{message_id}")
        button.console_action_id = action_id
        button.console_message_id = message_id
        self.post_message(Button.Pressed(button))

    async def choose_captured_message_more_action(
        self, message_id: str, action_id: str, *, opener_button_id: str
    ) -> None:
        """Detach More in a separate message turn, then dispatch its capture."""
        await self.dismiss_message_more_menu(restore_focus=False)
        self.call_later(
            self.dispatch_captured_message_action,
            message_id,
            action_id,
            opener_button_id=opener_button_id,
        )

    def _focus_action_button(self, message_id: str, action_id: str) -> None:
        try:
            self.query_one(
                f"#console-message-action-{action_id}-{message_id}", Button
            ).focus()
        except Exception:
            return
