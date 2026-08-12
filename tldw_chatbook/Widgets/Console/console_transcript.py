"""Native Console transcript widget."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, replace
from time import monotonic
from typing import Any, Iterable, Literal, Mapping

from loguru import logger
from PIL import Image as PILImage
from rich_pixels import Pixels
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content, Span
from textual.css.query import NoMatches
from textual.dom import NoScreen
from textual.events import Click, Key
from textual.message_pump import NoActiveAppError
from textual.style import Style
from textual.widget import Widget
from textual.widgets import Button, Markdown, Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleCitationNoticeCode,
    ConsoleCitationPhase,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_image_view import (
    PIXELS_MAX_COLS,
    PIXELS_MAX_LINES,
    ConsoleImageRowSpec,
    fit_image_cell_size,
)
from tldw_chatbook.Chat.console_message_actions import (
    ConsoleMessageAction,
    ConsoleMessageActionService,
    action_row_guide,
)
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
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_generation_card import (
    ConsoleGenerationCard,
    ConsoleGenerationCardSpec,
    generation_card_signature,
)
from tldw_chatbook.Widgets.Console.console_video_card import (
    ConsoleVideoCard,
    ConsoleVideoCardSpec,
    video_card_signature,
)
from tldw_chatbook.Widgets.diff_widgets import make_diff
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


CONSOLE_TRANSCRIPT_RULE = "─" * 200
CONSOLE_GENERATING_PLACEHOLDER = "Generating…"
#: TASK-1365: virtual-height watermarks (terminal rows) for transcript pruning.
#: 20000 rows is several hundred long messages; rows are cheap to measure but
#: expensive to keep laid out. Mirrored from the legacy chat log pruning
#: (``UI/Chat_Modules/chat_log_pruning.py`` on feat/toad-ui-improvements).
DEFAULT_PRUNE_HIGH_WATERMARK = 20000
DEFAULT_PRUNE_LOW_WATERMARK = 12000
#: TASK-15455: tail-first mount window for a conversation LOAD (session resume
#: or a session switch). The watermarks above bound a transcript that GROWS;
#: they can only act after the rows are mounted and laid out, so a resumed
#: 500-message session used to pay a full mount plus a full-history Markdown
#: parse before anything was trimmed. A load now mounts at most this many of
#: the newest messages, further capped by an estimated line budget, and
#: hydrates older messages when the reader scrolls back.
#: 40 messages is roughly a dozen turns of scrollback -- comfortably more than
#: any terminal shows at once, and above every fixture in the transcript
#: suites, so the window only engages on histories those never reach.
DEFAULT_TRANSCRIPT_WINDOW_MESSAGES = 40
#: Estimated rendered rows (body lines + per-row chrome) allowed in the initial
#: window. A handful of very long messages hit this before the message cap.
DEFAULT_TRANSCRIPT_WINDOW_LINES = 600
#: Messages hydrated per scroll-back step.
DEFAULT_TRANSCRIPT_HYDRATE_MESSAGES = 20
#: The window never shrinks below this many messages, however long they are --
#: a single mounted row would make the transcript look truncated.
MIN_TRANSCRIPT_WINDOW_MESSAGES = 3
#: Per-message row chrome (rule + speaker label) folded into the line estimate.
_MESSAGE_ROW_CHROME_LINES = 2
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


def get_console_transcript_window(
    app_config: Mapping[str, object] | None,
) -> tuple[int, int, int]:
    """Resolve the tail-first mount window settings from config.

    Reads ``[chat_defaults] transcript_window_messages`` /
    ``transcript_window_lines`` / ``transcript_hydrate_messages``, falling back
    to the ``DEFAULT_TRANSCRIPT_*`` constants when missing or invalid. A
    ``window_messages <= 0`` disables windowing entirely (the kill switch:
    every message mounts at load, exactly as before TASK-15455).

    Args:
        app_config: The loaded application config dict (``app.app_config``).

    Returns:
        Tuple of ``(window_messages, window_lines, hydrate_messages)``.
    """
    chat_defaults = (app_config or {}).get("chat_defaults", {})
    if not isinstance(chat_defaults, Mapping):
        chat_defaults = {}
    window_messages = _coerce_prune_int(
        chat_defaults.get("transcript_window_messages"),
        DEFAULT_TRANSCRIPT_WINDOW_MESSAGES,
    )
    window_lines = _coerce_prune_int(
        chat_defaults.get("transcript_window_lines"), DEFAULT_TRANSCRIPT_WINDOW_LINES
    )
    hydrate_messages = _coerce_prune_int(
        chat_defaults.get("transcript_hydrate_messages"),
        DEFAULT_TRANSCRIPT_HYDRATE_MESSAGES,
    )
    return window_messages, max(1, window_lines), max(1, hydrate_messages)


def _estimated_message_lines(message: ConsoleChatMessage) -> int:
    """Estimate a message's rendered height in terminal rows.

    Deliberately cheap and approximate (raw newlines plus fixed row chrome, no
    wrapping): it only sizes the INITIAL mount window, and the real bound on
    mounted height stays the measured height watermarks.

    Args:
        message: The message about to be sized.

    Returns:
        Estimated rows the message's row group occupies.
    """
    content = getattr(message, "content", "") or ""
    return content.count("\n") + 1 + _MESSAGE_ROW_CHROME_LINES


def _message_role_label(message: ConsoleChatMessage) -> str:
    role = message.role.value if hasattr(message.role, "value") else str(message.role)
    return role.title()


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
    if message.status == "streaming" and not content.strip():
        # Between send-accepted and the first streamed token the assistant row
        # has no content; show a visible generating state instead of an empty
        # row (local models can take 30-90s to first token).
        return CONSOLE_GENERATING_PLACEHOLDER
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
    """Return True when the rendered body is the pre-first-token placeholder."""
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
#: *action/inner-monologue*, and "quoted speech" (straight or curly). Matched
#: as closed pairs only, so an unclosed marker mid-stream stays literal until
#: it closes. Order matters: ** before * so bold never half-matches as
#: italics, and a quote swallows any markers inside it (task-1536).
_INLINE_MD_RE = re.compile(
    r"\*\*(.+?)\*\*"
    r"|`([^`]+)`"
    r"|(\"[^\"\n]+\")"
    r"|(“[^”\n]+”)"
    r"|\*([^*\n]+)\*"
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

#: Roleplay flavor styles (task-1536). Concrete colors, not theme variables:
#: Content span styles are parsed directly and never resolve CSS ``$`` vars.
#: All three read on the dark default theme and stay distinct from each
#: other and from plain narration.
_BOLD_STYLE = "bold #f7d774"
_SPEECH_STYLE = "#8ecdf7"
_ACTION_STYLE = "italic #b596d8"

_CONSOLE_RP_SPEECH_COMPONENT = "console-rp-speech"
_CONSOLE_RP_ACTION_COMPONENT = "console-rp-action"
_CONSOLE_RP_STRONG_COMPONENT = "console-rp-strong"
_CONSOLE_RP_COMPONENTS = frozenset(
    {
        _CONSOLE_RP_SPEECH_COMPONENT,
        _CONSOLE_RP_ACTION_COMPONENT,
        _CONSOLE_RP_STRONG_COMPONENT,
    }
)
_ROLEPLAY_SPEECH_RE = re.compile(r'"[^"\n]+"|“[^”\n]+”')


def _inline_markdown_spans(line: str) -> list:
    """Split one line into Content segments, styling inline flavor.

    ``**bold**``, ``“quoted”``/``"quoted"`` speech, and
    ``*action/inner monologue*`` each get a distinct style; `code` keeps its
    plain italic. Text is always emitted literally (styles are applied via
    ``(text, style)`` tuples, never markup parsing), so message text can
    never inject Rich markup. Quotation marks stay visible inside the
    styled speech span; bold/action marker characters are stripped.

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
        bold, code, quote, curly_quote, action = match.groups()
        if bold is not None:
            out.append((bold, _BOLD_STYLE))
        elif code is not None:
            out.append((code, "italic"))
        elif quote is not None:
            out.append((quote, _SPEECH_STYLE))
        elif curly_quote is not None:
            out.append((curly_quote, _SPEECH_STYLE))
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
            semantic_ranges.append(
                (span.start, span.end, _CONSOLE_RP_ACTION_COMPONENT)
            )
        elif span.style == ".strong":
            semantic_ranges.append(
                (span.start, span.end, _CONSOLE_RP_STRONG_COMPONENT)
            )

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
    for match in _ROLEPLAY_SPEECH_RE.finditer(content.plain):
        flavor_spans.extend(
            Span(start, end, f".{_CONSOLE_RP_SPEECH_COMPONENT}")
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


@dataclass(frozen=True)
class _TranscriptRow:
    key: str
    kind: Literal[
        "rule",
        "banner",
        "message",
        "diff",
        "citations",
        "original-attempt",
        "image",
        "generation-card",
        "video-card",
        "actions",
        "action-help",
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
        return presentation.content
    if message.variants is not None:
        return message.variants.current.content
    return message.content


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
    if message.status == "streaming":
        body = _assistant_markdown_body(message, presentation)
        # task-2154.16 (FB-01): same wording as the plain renderer's dim
        # status line -- never the raw "[streaming]" content token.
        suffix = (
            f"  {CONSOLE_GENERATING_PLACEHOLDER}" if not body.strip() else "  Streaming…"
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


class ConsoleMarkdownMessage(Vertical):
    """Assistant transcript row rendered with Textual's Markdown widget.

    TASK-1990 (frogmouth-comparison follow-up). Streaming deltas are applied
    with ``Markdown.append()`` (prefix-diffed against the last applied body)
    so per-tick cost is O(delta), not O(message). Non-prefix changes (variant
    switch, edit) fall back to a full ``Markdown.update()``.

    Link policy (task AC#6): links never auto-open (``open_links=False``). A
    click on an http(s) link opens the system browser and notifies; any other
    scheme notifies with the href and does nothing else.
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
        self._body_text = _assistant_markdown_body(message, self._presentation)
        # TASK-15456: text appended to ``self._body_text`` above but not yet
        # handed to the Markdown widget (deferred while streaming inside an
        # open fence), plus the monotonic deadline by which it must flush.
        self._pending_fence_delta = ""
        self._fence_defer_deadline: float | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            _assistant_markdown_header(self._message, self._presentation),
            classes=" ".join(
                ["console-markdown-header", *_speaker_label_classes(self._presentation)]
            ),
            markup=False,
        )
        yield ConsoleRoleplayMarkdown(
            self._body_text,
            classes="console-markdown-body",
            open_links=False,
        )
        footer_content = _assistant_markdown_footer(self._message)
        footer = Static(
            footer_content or "",
            classes="console-markdown-footer",
        )
        footer.display = footer_content is not None
        yield footer

    def sync_message(
        self,
        message: ConsoleChatMessage,
        presentation: ConsoleMessagePresentation | None = None,
        *,
        selected: bool = False,
    ) -> None:
        """Update header/body/footer in place; append-only growth avoids re-parse."""
        presentation = presentation or self._presentation
        self.message_id = message.id
        self._message = message
        self._presentation = presentation
        _sync_message_classes(
            self,
            message,
            presentation,
            selected=selected,
            markdown=True,
        )
        try:
            header = self.query_one(".console-markdown-header", Static)
            markdown = self.query_one(Markdown)
            footer = self.query_one(".console-markdown-footer", Static)
        except NoMatches:
            return
        header.set_classes(
            " ".join(
                ["console-markdown-header", *_speaker_label_classes(presentation)]
            )
        )
        header.update(_assistant_markdown_header(message, presentation))
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

    def on_click(self, event: Click) -> None:
        event.stop()
        transcript = self.parent
        while transcript is not None and not isinstance(transcript, ConsoleTranscript):
            transcript = transcript.parent
        if isinstance(transcript, ConsoleTranscript):
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
    ) -> None:
        self.message_id = message.id
        self._message = message
        self._presentation = presentation or resolve_console_message_presentation(
            message, ConsolePresentationContext()
        )
        self._selected = selected
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
        yield Static(
            Content(self._speaker_label()),
            classes=" ".join(_speaker_label_classes(self._presentation)),
            markup=False,
        )
        yield Static(
            _message_body_render_text(self._message, self._presentation),
            classes="console-transcript-message-body",
            markup=False,
        )

    def _speaker_label(self) -> str:
        return _speaker_label(self._message, self._presentation)

    def sync_message(
        self,
        message: ConsoleChatMessage,
        presentation: ConsoleMessagePresentation | None = None,
        *,
        selected: bool = False,
    ) -> None:
        """Update row content and selection styling without remounting the row."""
        presentation = presentation or self._presentation
        self.message_id = message.id
        self._message = message
        self._presentation = presentation
        self._selected = selected
        _sync_message_classes(
            self,
            message,
            presentation,
            selected=selected,
            markdown=False,
        )
        try:
            label = self.query_one(".console-transcript-speaker-label", Static)
            body = self.query_one(".console-transcript-message-body", Static)
        except NoMatches:
            return
        label.set_classes(" ".join(_speaker_label_classes(presentation)))
        label.update(Content(self._speaker_label()))
        body.update(_message_body_render_text(message, presentation))

    def on_click(self, event: Click) -> None:
        event.stop()
        transcript = self.parent
        while transcript is not None and not isinstance(transcript, ConsoleTranscript):
            transcript = transcript.parent
        if isinstance(transcript, ConsoleTranscript):
            transcript.toggle_message_selection(self.message_id)


class ConsoleToolDiffRow(Vertical):
    """Inline diff row under an expanded file-write TOOL marker (TASK-1366).

    Mounts empty and fills in asynchronously: the diff is computed off the
    UI thread (``DiffView.prepare``) BEFORE the DiffView mounts, mirroring
    ``tool_message_widgets.ToolExecutionWidget``'s integration. The row is
    render-derived view state -- it exists only while its marker message is
    expanded via the full-output toggle, and disappears with it (or when
    the message leaves the view window).
    """

    can_focus = False

    def __init__(self, message_id: str, diff: tuple[str, str, str]) -> None:
        self.message_id = message_id
        self._diff = diff
        super().__init__(
            id=f"console-tool-diff-{message_id}",
            classes="console-transcript-tool-diff",
        )

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
            await self.mount(diff_view)
        except Exception as exc:  # noqa: BLE001 — a render failure never breaks the transcript
            logger.opt(exception=True).error(
                f"Failed to render console tool diff for {path}: {exc}"
            )


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
        return bool(self.provider_action_label.strip()) and self.card_state.mode != "card"

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


class ConsoleTranscript(VerticalScroll):
    """Focusable native Console transcript with compact rule-separated messages."""

    can_focus = True
    BINDINGS = [
        ("down,j", "select_next", "Next message"),
        ("up,k", "select_previous", "Previous message"),
        ("enter", "confirm_selection", "Toggle message selection"),
        ("escape", "clear_selection", "Clear selection"),
        ("c", "invoke_selected_action('copy')", "Copy"),
        ("e", "invoke_selected_action('edit')", "Edit"),
        ("r", "invoke_selected_action('regenerate')", "Regenerate"),
        ("o", "invoke_selected_action('tool-output')", "Full output"),
        ("v", "invoke_selected_action('review-changes')", "Review changes"),
    ]

    PROTECTED_CLICK_CLASSES: frozenset[str] = frozenset(
        {
            "console-transcript-action-row",
            "console-transcript-action-guide",
            "console-transcript-empty-panel",
            "console-transcript-empty-body",
            "console-transcript-empty-state",
            "console-transcript-rule",
            "console-transcript-summary-banner",
            "console-transcript-citation-sources",
            # Textual scrollbars carry the generic system-widget class; ignore them
            # defensively if a scrollbar click ever bubbles up to the transcript.
            "-textual-system",
            "vertical-scrollbar",
            "horizontal-scrollbar",
            "scrollbar",
        }
    )
    """Widget classes that must keep the current selection active when clicked."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._presentation_context = ConsolePresentationContext()
        self._messages: list[ConsoleChatMessage] = []
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
        self._original_attempt_previews: dict[str, str] = {}
        self._citation_counts: dict[str, int] = {}
        #: TASK-1860: ids of TOOL markers currently showing their FULL result.
        #: Pure view state, owned here: expansion never touches the store, is
        #: per row (so several calls in one turn expand independently), and is
        #: deliberately dropped when the transcript is rebuilt for another
        #: session rather than following the user across conversations.
        self._expanded_tool_output_ids: set[str] = set()
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
        #: TASK-15455: tail-first mount window. Ids of the OLDEST messages a
        #: conversation load deliberately did not mount; ``_transcript_rows``
        #: filters them exactly like the prune window above. Unlike pruning
        #: these come back: scrolling near the top hydrates the next chunk.
        self._unhydrated_message_ids: set[str] = set()
        #: Ids hydrated back into the view since the window was established.
        #: They are protected from the watermark walk while
        #: ``_scrollback_protected`` holds, so scrollback never vanishes
        #: mid-read.
        self._hydrated_message_ids: set[str] = set()
        #: True from the moment scrollback is hydrated until the reader is back
        #: at the tail (jump pill, a send, or simply scrolling to the bottom).
        #: An explicit latch rather than a sampled "is following the tail?":
        #: the anchor re-engages asynchronously, so a check scheduled by the
        #: jump can otherwise run while the widget still reads as detached and
        #: skip the reclaim entirely.
        self._scrollback_protected = False
        self._hydration_check_scheduled = False

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
        # TASK-15455: the same compose path can have applied a mount window;
        # a window shorter than the viewport must fill until it can scroll.
        self._schedule_hydration_check()

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

    def _is_following_tail(self) -> bool:
        """Return True when the view is pinned to the newest content."""
        return bool(self.is_anchored and not getattr(self, "_anchor_released", False))

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
        # TASK-15455: hydrated scrollback is protected from the watermark walk
        # only while the reader is reading it; jumping back to the tail is
        # exactly when that protection lifts, so drop it and run the check.
        self._release_scrollback_protection()

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

    def set_messages(self, messages: Iterable[ConsoleChatMessage]) -> None:
        """Replace transcript messages and refresh mounted rows when possible.

        Args:
            messages: New transcript messages in display order. Signature
                cache entries for messages no longer present are pruned here
                (delete correctness for the TASK-259 per-message cache).
        """
        self._messages = list(messages)
        message_ids = {message.id for message in self._messages}
        # Expansion is per message id, so ids that left the transcript (a
        # session switch, a deleted branch) must go with them -- otherwise the
        # set grows for the life of the widget and a recycled id would come
        # back already expanded.
        self._expanded_tool_output_ids &= message_ids
        # TASK-1365: same lifecycle for the prune window -- a session switch
        # re-renders from scratch, and a deleted message must not linger here.
        self._pruned_message_ids &= message_ids
        # TASK-15455: (re)establish the tail-first mount window on a LOAD --
        # the first ingest, or one whose ids are disjoint from the last (a
        # session switch). An ingest that shares ids is the 0.2s sync tick, a
        # send, or a branch swap: those extend the current view and must keep
        # whatever the reader has already hydrated.
        self._unhydrated_message_ids &= message_ids
        self._hydrated_message_ids &= message_ids
        if not self._seen_message_ids or not (message_ids & self._seen_message_ids):
            self._hydrated_message_ids.clear()
            self._scrollback_protected = False
            self._unhydrated_message_ids = self._initial_unhydrated_ids()
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
            self.anchor()
            # TASK-15455: the send takes the reader out of any scrollback they
            # hydrated, so the watermarks own those rows again.
            self._release_scrollback_protection()
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
            self.pending_selection_id = None
        if self.selected_message_id not in message_ids:
            self.selected_message_id = None
        for stale_id in [
            cached_id
            for cached_id in self._message_signature_cache
            if cached_id not in message_ids
        ]:
            del self._message_signature_cache[stale_id]
            self._signature_compute_counts.pop(stale_id, None)

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

    def set_video_card_specs(
        self, specs: Mapping[str, ConsoleVideoCardSpec]
    ) -> None:
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
        async with self._refresh_lock:
            await self._reconcile_rows(self._transcript_rows())
        self._schedule_prune_check()
        self._schedule_hydration_check()

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

    def _window_settings(self) -> tuple[int, int, int]:
        """Return ``(window_messages, window_lines, hydrate_messages)``."""
        try:
            app_config = getattr(self.app, "app_config", None)
        except NoActiveAppError:
            app_config = None
        return get_console_transcript_window(app_config)

    def _initial_unhydrated_ids(self) -> set[str]:
        """Return the ids a fresh load leaves unmounted (the tail-first window).

        Walks backwards from the newest message, keeping messages until either
        the message cap or the estimated line budget is reached (never fewer
        than :data:`MIN_TRANSCRIPT_WINDOW_MESSAGES`). Everything older is
        reported as unhydrated.

        Returns:
            Ids of the oldest messages to leave unmounted; empty when the whole
            history fits the window or windowing is disabled.
        """
        window_messages, window_lines, _hydrate = self._window_settings()
        total = len(self._messages)
        if window_messages <= 0 or total <= window_messages:
            return set()
        kept = 0
        lines = 0
        for message in reversed(self._messages):
            if kept >= window_messages:
                break
            if kept >= MIN_TRANSCRIPT_WINDOW_MESSAGES and lines >= window_lines:
                break
            lines += _estimated_message_lines(message)
            kept += 1
        if kept >= total:
            return set()
        logger.debug(
            f"Console transcript load window: mounting the newest {kept} of "
            f"{total} messages (~{lines} estimated rows)"
        )
        return {message.id for message in self._messages[: total - kept]}

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Hydrate older scrollback as the reader approaches the top.

        Every scroll path (wheel, keyboard, scrollbar drag, programmatic)
        lands here, so this is the single trigger for scroll-back hydration.
        Costs one attribute read on the common path: transcripts with nothing
        windowed out return immediately.
        """
        super().watch_scroll_y(old_value, new_value)
        if getattr(self, "_scrollback_protected", False) and self._is_following_tail():
            # Back at the tail: the reader has left the scrollback they
            # hydrated, so the watermark walk may reclaim it again (see
            # `_compute_prunable_prefix`).
            self._release_scrollback_protection()
        if not getattr(self, "_unhydrated_message_ids", None):
            return
        if new_value <= max(1, self.container_size.height):
            self._schedule_hydration_check()

    def _release_scrollback_protection(self) -> None:
        """Let the watermarks reclaim hydrated scrollback again.

        Called when the reader returns to the tail by any route (the jump
        pill, a send, or scrolling to the bottom). No-op when nothing is
        protected, so the common scroll path costs one boolean read.
        """
        if not self._scrollback_protected:
            return
        self._scrollback_protected = False
        self._schedule_prune_check()

    def _schedule_hydration_check(self) -> None:
        """Queue one coalesced scroll-back hydration check after the refresh."""
        if self._hydration_check_scheduled or not self.is_mounted:
            return
        if not self._unhydrated_message_ids:
            return
        self._hydration_check_scheduled = True
        self.call_after_refresh(self._run_hydration_check)

    async def _run_hydration_check(self) -> None:
        """Mount the next chunk of older messages when the reader needs it.

        Two conditions gate hydration, and together they make a
        hydrate → prune → hydrate oscillation impossible for ANY watermark
        configuration:

        1. The mounted height must be strictly BELOW the low watermark. The
           watermark walk only fires above the HIGH mark and always stops
           while the remainder is still above the LOW mark, so a prune can
           never put the transcript back into a hydratable state.
        2. The reader must be within one viewport of the top -- or the window
           must be too short to scroll at all, which is the same request
           ("there is more above and I cannot reach it").

        Each step is additionally sized against the room left under the low
        watermark, so a hydration chunk does not vault the transcript over the
        HIGH mark and hand the watermark walk the rows it just mounted.

        A second, independent guard lives in ``_compute_prunable_prefix``:
        hydrated ids are protected from pruning while the reader is detached,
        so an oversized chunk that overshoots the high watermark is never
        yanked back out from under them.
        """
        self._hydration_check_scheduled = False
        if not self.is_mounted or self._closing or self._pruning:
            return
        if not self._unhydrated_message_ids:
            return
        low_mark, high_mark = self._prune_watermarks()
        total_height = self.virtual_size.height
        if high_mark > 0 and total_height >= low_mark:
            return
        if self.scroll_y > max(1, self.container_size.height):
            return
        _window, _lines, chunk = self._window_settings()
        pending = [
            message
            for message in self._messages
            if message.id in self._unhydrated_message_ids
        ]
        if not pending:
            self._unhydrated_message_ids.clear()
            return
        # Size the step against the room left under the low watermark, using
        # the same estimator as the initial window. Without this a fixed-size
        # chunk can vault a small transcript clean over the HIGH mark, and the
        # watermark walk then throws the freshly hydrated rows away (measured:
        # a 3-message window hydrating 10 messages under a 20/40 config mounted
        # 10 rows and pruned 9 of them one tick later). One message is always
        # taken so scroll-back never stalls silently.
        budget = max(0, low_mark - total_height) if high_mark > 0 else None
        hydrate_ids: list[str] = []
        estimated = 0
        for message in reversed(pending):
            if len(hydrate_ids) >= chunk:
                break
            lines = _estimated_message_lines(message)
            if budget is not None and hydrate_ids and estimated + lines > budget:
                break
            estimated += lines
            hydrate_ids.append(message.id)
        hydrate_ids.reverse()
        following = self._is_following_tail()
        anchor_y = self.scroll_y
        self._unhydrated_message_ids.difference_update(hydrate_ids)
        self._hydrated_message_ids.update(hydrate_ids)
        if not following:
            # Reader-driven scroll-back: hold it until they return to the tail.
            # A fill while following the tail is not scrollback the reader is
            # reading, so it stays reclaimable by the watermarks.
            self._scrollback_protected = True
        logger.debug(
            f"Hydrating {len(hydrate_ids)} older Console transcript message(s) "
            f"({len(self._unhydrated_message_ids)} still windowed out)"
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
                # Keep the same content in view: the hydrated rows were added
                # ABOVE the reader, so shift down by the height actually added
                # (measured, not estimated) -- the mirror image of pruning.
                added = self.virtual_size.height - total_height
                self.scroll_to(y=max(0.0, anchor_y + added), animate=False)
            # One chunk may not be enough (a short window under a tall
            # viewport, or tiny messages); re-arm. Bounded by the gates above
            # and by the unhydrated set emptying.
            self._schedule_hydration_check()

        self.call_after_refresh(_restore_scroll)
        self._schedule_prune_check()

    def ensure_message_hydrated(self, message_id: str) -> bool:
        """Mount a windowed-out message (and everything after it) on demand.

        Jump targets -- a citation, a search hit, a programmatic selection --
        can name a message the tail-first window never mounted. Hydrating from
        that message forward keeps the mounted rows one contiguous suffix of
        the history, which is what pruning and reconciliation both assume.

        Args:
            message_id: Identifier of the message that must have a row.

        Returns:
            True when rows were hydrated (a refresh is scheduled), False when
            the message was already mounted or is not in this transcript.
        """
        if message_id not in self._unhydrated_message_ids:
            return False
        ordered = [
            message.id
            for message in self._messages
            if message.id in self._unhydrated_message_ids
        ]
        try:
            index = ordered.index(message_id)
        except ValueError:  # pragma: no cover - membership was just checked
            return False
        hydrate_ids = ordered[index:]
        self._unhydrated_message_ids.difference_update(hydrate_ids)
        self._hydrated_message_ids.update(hydrate_ids)
        self._scrollback_protected = True
        if self.is_mounted:
            self.call_later(self.refresh_messages)
        return True

    def unhydrated_message_ids(self) -> frozenset[str]:
        """Return the ids currently outside the tail-first mount window."""
        return frozenset(self._unhydrated_message_ids)

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
                # Keep the same content in view: shift the offset up by the
                # height actually removed (measured, not estimated).
                removed = total_height - self.virtual_size.height
                self.scroll_to(y=max(0.0, anchor_y - removed), animate=False)

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
        protected_ids = {
            message.id for message in self._messages if message.status == "streaming"
        }
        if self.selected_message_id is not None:
            protected_ids.add(self.selected_message_id)
        if self._scrollback_protected:
            # TASK-15455: scrollback the reader deliberately hydrated is what
            # they are looking at; the walk must not delete it from under
            # them (and an oversized hydration step that overshot the high
            # mark must not be undone, which would restart the cycle).
            # The latch drops the moment they return to the tail, so the
            # watermarks still bound the steady state.
            protected_ids |= self._hydrated_message_ids

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
            prune_ids.append(group_id)
            return True

        for child in self.children:
            key = key_by_widget_id.get(id(child))
            row_message_id: str | None = None
            if key is not None and ":" in key:
                candidate = key.split(":", 1)[1]
                if candidate in message_ids:
                    row_message_id = candidate
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

    def select_message(self, message_id: str) -> None:
        """Select one message and show its contextual action row."""
        if message_id not in {message.id for message in self._messages}:
            return
        # TASK-15455: a jump target (citation, search hit, programmatic focus)
        # can name a message the tail-first window has not mounted; hydrate it
        # first so the selection lands on a real row with its action row.
        self.ensure_message_hydrated(message_id)
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
        """Return a terminal-readable transcript rendering for tests and exports."""
        rule = "─" * max(1, width)
        lines: list[str] = []
        for message in self._messages:
            presentation = self._message_presentation(message)
            lines.append(rule)
            if message.id == self.summary_boundary_message_id:
                lines.append(CONSOLE_SUMMARY_BANNER_COPY)
            lines.extend(
                [
                    _speaker_label(message, presentation),
                    _message_body(message, presentation),
                ]
            )
            status_line = _message_status_line(message)
            if status_line and not _is_generating_placeholder_body(
                message, _message_body(message, presentation)
            ):
                lines.append(status_line)
            if message.id == self.selected_message_id:
                lines.append(self._plain_action_row(message))
                lines.append(ConsoleMessageActionService().plain_action_guide(message))
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
        self.selected_message_id = None
        if self.is_mounted:
            self.call_later(self.refresh_messages)
            self.call_later(self._notify_selection_changed)
            self.call_later(self._paint_debug_dump, "after-clear-selection")

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
            return False
        button.press()
        return True

    def on_click(self, event: Click) -> None:
        """Clear selection when the user clicks negative space in the transcript.

        Clicks that land on controls with classes in ``PROTECTED_CLICK_CLASSES``
        (message action rows/buttons, rule separators, action-help text, the
        empty-state panel, or scrollbars) keep the current selection active. All
        other clicks that bubble up to the transcript itself clear the selection.
        """
        control = event.control
        if control is not None and any(
            control.has_class(class_name) for class_name in self.PROTECTED_CLICK_CLASSES
        ):
            event.stop()
            return
        if control is self:
            self.action_clear_selection()
            event.stop()

    def on_key(self, event: Key) -> None:
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
        return next(
            (message for message in self._messages if message.id == message_id), None
        )

    def _visible_messages(self) -> list[ConsoleChatMessage]:
        """Return the messages with rendered rows (excludes unmounted windows).

        Keyboard selection walks this list so j/k never lands on a pruned or
        not-yet-hydrated (row-less) message; the store-facing ``_messages``
        keeps full history.
        """
        if not self._pruned_message_ids and not self._unhydrated_message_ids:
            return self._messages
        return [
            message
            for message in self._messages
            if message.id not in self._pruned_message_ids
            and message.id not in self._unhydrated_message_ids
        ]

    def _notify_selection_changed(self) -> None:
        """Let the owning screen refresh inspector/control surfaces after selection changes."""
        sync_console_control_bar = getattr(
            self.screen, "_sync_console_control_bar", None
        )
        if callable(sync_console_control_bar):
            sync_console_control_bar()

    def _transcript_rows(self) -> list[_TranscriptRow]:
        rows: list[_TranscriptRow] = []
        for message in self._messages:
            if message.id in self._pruned_message_ids:
                # TASK-1365: pruned by the height watermarks; the store keeps
                # the message, the view window drops every row derived from it.
                continue
            if message.id in self._unhydrated_message_ids:
                # TASK-15455: older than the tail-first load window. Same
                # view-only contract as pruning above, except this one is
                # reversible -- scrolling back hydrates these ids again.
                continue
            message = self._with_expanded_tool_output(message)
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
                        renderable=f"Sources ({citation_count})",
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

    def _message_widgets(self) -> list[Widget]:
        return [
            self._build_row_widget(row, track=False) for row in self._transcript_rows()
        ]

    async def _reconcile_rows(self, rows: list[_TranscriptRow]) -> None:
        desired_keys = [row.key for row in rows]
        desired_key_set = set(desired_keys)

        stale_widgets: list[Widget] = []
        for stale_key in [
            key for key in self._row_widgets if key not in desired_key_set
        ]:
            stale_widgets.append(self._row_widgets.pop(stale_key))
            self._row_signatures.pop(stale_key, None)
            self._row_build_counts.pop(stale_key, None)
        if stale_widgets:
            # TASK-15455: one batched removal instead of one awaited
            # `remove()` per row -- a session switch or a prune drops hundreds
            # of rows at once, and each individual removal is its own DOM
            # walk + await. `remove_children` prunes the whole set in one
            # pass. Widgets that are not (or no longer) our children take the
            # per-widget path they always did.
            own_rows = [widget for widget in stale_widgets if widget.parent is self]
            if own_rows:
                await self.remove_children(own_rows)
            for stale_widget in stale_widgets:
                if stale_widget.parent is not None:
                    await stale_widget.remove()

        previous_widget: Widget | None = None
        # TASK-15455: contiguous freshly built rows are accumulated here and
        # mounted in ONE `mount(*widgets, after=...)` call. Textual inserts a
        # multi-widget mount in argument order, so the resulting child order
        # is identical to mounting them one at a time -- only the number of
        # DOM/layout passes changes. The batch is flushed before any decision
        # that reads the real child list (the in-position check below).
        pending_widgets: list[Widget] = []
        pending_keys: list[str] = []
        pending_signatures: list[tuple] = []
        pending_after: Widget | None = None

        async def _flush_pending_mounts() -> bool:
            """Mount the accumulated new rows; False means abandon the pass."""
            nonlocal previous_widget
            if not pending_widgets:
                return True
            widgets = list(pending_widgets)
            keys = list(pending_keys)
            signatures = list(pending_signatures)
            pending_widgets.clear()
            pending_keys.clear()
            pending_signatures.clear()
            if pending_after is None:
                await self.mount(*widgets, before=0 if self.children else None)
            else:
                await self.mount(*widgets, after=pending_after)
            for key, widget, signature in zip(keys, widgets, signatures):
                if widget.parent is not self:
                    # Version-proof backstop for the pruning check in the
                    # walk: mount() completed without attaching (it no-ops
                    # while the container is being removed). Drop the phantom
                    # map entries for this batch and abandon the pass instead
                    # of poisoning later moves.
                    for pending_key in keys:
                        self._row_widgets.pop(pending_key, None)
                        self._row_signatures.pop(pending_key, None)
                    return False
                self._row_widgets[key] = widget
                self._row_signatures[key] = signature
            previous_widget = widgets[-1]
            return True

        for index, row in enumerate(rows):
            if self._closing or self._pruning or not self.is_attached:
                # This instance is being removed (a parent recompose/session
                # surface swap can prune the transcript between this loop's
                # awaits). Widget.mount() silently no-ops while pruning, so
                # continuing would record detached widgets in the row maps
                # and then crash in move_child. The replacement instance
                # composes fresh state; abandon this pass.
                return
            widget = self._row_widgets.get(row.key)
            row_was_mounted = False
            if widget is None:
                widget = self._build_row_widget(row, track=True)
                if not pending_widgets:
                    pending_after = previous_widget
                pending_widgets.append(widget)
                pending_keys.append(row.key)
                pending_signatures.append(row.signature)
                continue
            # An already-mounted row: every row above it must be in the DOM
            # before its position can be judged, so flush first.
            if not await _flush_pending_mounts():
                return
            if self._row_signatures.get(row.key) != row.signature:
                updated_widget = self._update_row_widget(widget, row)
                if updated_widget is widget:
                    self._row_signatures[row.key] = row.signature
                else:
                    await widget.remove()
                    widget = updated_widget
                    if previous_widget is None:
                        await self.mount(widget, before=0 if self.children else None)
                    else:
                        await self.mount(widget, after=previous_widget)
                    row_was_mounted = True
                    self._row_widgets[row.key] = widget
                    self._row_signatures[row.key] = row.signature

            if row_was_mounted and widget.parent is not self:
                # Version-proof backstop for the pruning check above:
                # mount() completed without attaching (it no-ops while the
                # container is being removed). Drop the phantom map entries
                # and abandon the pass instead of poisoning later moves.
                self._row_widgets.pop(row.key, None)
                self._row_signatures.pop(row.key, None)
                return
            if not row_was_mounted:
                # TASK-15453: `move_child` is several O(rows) NodeList scans
                # plus a `refresh(layout=True)` plus a DOM-version bump --
                # expensive to pay for a row that is already where it needs
                # to be. Every already-processed row (0..index-1) is correct
                # by induction (mounts above always land at the walk's
                # current slot), so this row is in place iff it already
                # sits at `index` in the ACTUAL child list -- read fresh
                # every iteration (never cached) because earlier mounts in
                # this same pass shift indices out from under a snapshot.
                already_in_position = (
                    index < len(self.children) and self.children[index] is widget
                )
                if not already_in_position:
                    if previous_widget is None:
                        self.move_child(widget, before=0)
                    else:
                        self.move_child(widget, after=previous_widget)
            previous_widget = widget
        if self._closing or self._pruning or not self.is_attached:
            return
        if not await _flush_pending_mounts():
            return
        self._paint_debug_dump("after-reconcile")

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
        if row.kind == "message" and row.message is not None:
            presentation = self._message_presentation(row.message)
            if (
                row.message.role is ConsoleMessageRole.ASSISTANT
                and self._assistant_markdown_enabled()
            ):
                return ConsoleMarkdownMessage(
                    row.message, presentation, selected=row.selected
                )
            return ConsoleTranscriptMessage(
                row.message, presentation, selected=row.selected
            )
        if (
            row.kind == "diff"
            and row.message is not None
            and row.message.tool_diff is not None
        ):
            return ConsoleToolDiffRow(row.message.id, row.message.tool_diff)
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
            return ConsoleGenerationCard(row.generation_card_spec)
        if row.kind == "video-card" and row.video_card_spec is not None:
            return ConsoleVideoCard(row.video_card_spec)
        if row.kind == "actions" and row.message is not None:
            return self._action_row(row.message)
        raise ValueError(f"Unsupported transcript row: {row}")

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
            and isinstance(widget, ConsoleMarkdownMessage)
        ):
            widget.sync_message(
                row.message,
                self._message_presentation(row.message),
                selected=row.selected,
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
            presentation.revision_token,
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
        expanded = f"{head}{separator}{full}" if separator else f"{message.content}\n{full}"
        return replace(message, content=expanded)

    @on(Button.Pressed, ".console-transcript-action-button")
    def _intercept_tool_output_press(self, event: Button.Pressed) -> None:
        """Handle the Full-output button here; let every other action bubble.

        Expansion is view state owned by this widget -- it never reaches the
        store and nothing outside the transcript needs to know about it -- so
        routing it through the screen's action dispatch would add a hop that
        carries no information. Every other action id is left untouched and
        still bubbles to `ChatScreen`.
        """
        button_id = event.button.id or ""
        prefix = "console-message-action-tool-output-"
        if not button_id.startswith(prefix):
            return
        event.stop()
        self.toggle_tool_output(button_id.removeprefix(prefix))

    def toggle_tool_output(self, message_id: str) -> None:
        """Expand or collapse one TOOL marker's full result.

        Args:
            message_id: Id of the marker row to toggle. Unknown ids are
                harmless -- the row simply renders collapsed, and
                ``set_messages`` prunes ids that leave the transcript.
        """
        if message_id in self._expanded_tool_output_ids:
            self._expanded_tool_output_ids.discard(message_id)
        else:
            self._expanded_tool_output_ids.add(message_id)
        self.call_later(self.refresh_messages)

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
            presentation.revision_token,
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

    def _action_row_signature(self, message: ConsoleChatMessage) -> tuple:
        actions = []
        for action in ConsoleMessageActionService().available_actions(
            message,
            speaking_message_id=self._console_tts_speaking_message_id(),
            original_attempt_available=bool(
                message.citation_presentation
                and message.citation_presentation.original_attempt_available
            ),
            ephemeral=self._console_ephemeral_active(),
            **self._generation_action_kwargs(message),
        ):
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
        """Return the legend naming ``message``'s glyph-only action buttons.

        Reads the same ``available_actions`` inputs as ``_action_row``/
        ``_action_row_signature`` so the guide always describes the buttons
        actually mounted beside it (DS-01).
        """
        return action_row_guide(
            ConsoleMessageActionService().available_actions(
                message,
                speaking_message_id=self._console_tts_speaking_message_id(),
                original_attempt_available=bool(
                    message.citation_presentation
                    and message.citation_presentation.original_attempt_available
                ),
                ephemeral=self._console_ephemeral_active(),
                **self._generation_action_kwargs(message),
            )
        )

    def _action_row(self, message: ConsoleChatMessage) -> Horizontal:
        buttons: list[Button] = []
        for action in ConsoleMessageActionService().available_actions(
            message,
            speaking_message_id=self._console_tts_speaking_message_id(),
            original_attempt_available=bool(
                message.citation_presentation
                and message.citation_presentation.original_attempt_available
            ),
            ephemeral=self._console_ephemeral_active(),
            **self._generation_action_kwargs(message),
        ):
            if action.action_id == "feedback":
                buttons.append(
                    self._action_button(
                        message, ConsoleMessageAction("feedback-up", "👍")
                    )
                )
                buttons.append(
                    self._action_button(
                        message, ConsoleMessageAction("feedback-down", "👎")
                    )
                )
                continue
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
        return button

    def _focus_action_button(self, message_id: str, action_id: str) -> None:
        try:
            self.query_one(
                f"#console-message-action-{action_id}-{message_id}", Button
            ).focus()
        except Exception:
            return
