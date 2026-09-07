"""Pure identity and presentation contracts for Console roleplay chats."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata

from rich.cells import cell_len

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.UI.character_display_text import sanitize_character_display_label


CHAT_DISPLAY_NAME_MAX_CELLS = 48
CHARACTER_SPEAKER_LABEL_MAX_CHARACTERS = 180


class ConsoleTranscriptStyle(str, Enum):
    """Closed vocabulary for Console transcript role coloring."""

    NEUTRAL = "neutral"
    ROLE_ACCENTS = "role_accents"
    IMMERSIVE_RP = "immersive_rp"


DEFAULT_CONSOLE_TRANSCRIPT_STYLE = ConsoleTranscriptStyle.ROLE_ACCENTS


def normalize_console_transcript_style(value: object) -> ConsoleTranscriptStyle:
    """Return a supported transcript style with the expressive safe default.

    Args:
        value: Candidate enum member or string-like transcript style value.

    Returns:
        The matching transcript style, or the default role-accent style when
        the candidate is unsupported or blank.
    """

    if isinstance(value, ConsoleTranscriptStyle):
        return value
    try:
        return ConsoleTranscriptStyle(str(value or "").strip().lower())
    except ValueError:
        return DEFAULT_CONSOLE_TRANSCRIPT_STYLE


class ChatDisplayNameError(ValueError):
    """A human chat display name is unsafe or too wide."""


def normalize_chat_display_name(value: object, *, blank_means_none: bool) -> str | None:
    """Validate and normalize a human-facing Console chat display name.

    Args:
        value: Candidate display name. Only text or ``None`` is accepted.
        blank_means_none: Return ``None`` for an empty value when true;
            otherwise return the neutral ``"User"`` fallback.

    Returns:
        The stripped, validated display name, ``None``, or ``"User"``.

    Raises:
        ChatDisplayNameError: If the value is not text, contains unsafe control
            characters, or exceeds the terminal-cell limit.
    """
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ChatDisplayNameError("Display name must be text.")
    else:
        text = value.strip()
    if not text:
        return None if blank_means_none else "User"
    if any(
        unicodedata.category(char) in {"Cc", "Cs"}
        or char in {"\u2028", "\u2029"}
        or (unicodedata.category(char) == "Cf" and char not in {"\u200c", "\u200d"})
        for char in text
    ):
        raise ChatDisplayNameError("Display name cannot contain control characters.")
    if cell_len(text) > CHAT_DISPLAY_NAME_MAX_CELLS:
        raise ChatDisplayNameError("Display name must fit within 48 terminal cells.")
    return text


def effective_user_display_name(override: object, global_default: object) -> str:
    """Return the per-chat, global, or neutral human chat display name."""
    local = normalize_chat_display_name(override, blank_means_none=True)
    if local is not None:
        return local
    return normalize_chat_display_name(global_default, blank_means_none=False) or "User"


_TEMPLATE_TOKEN_RE = re.compile(
    r"\{\{user\}\}|\{\{random_user\}\}|<USER>|"
    r"\{\{char\}\}|\{\{character\}\}|\{\{persona\}\}|<CHAR>"
)
_USER_TOKENS = frozenset({"{{user}}", "{{random_user}}", "<USER>"})


def expand_character_template(
    source: str, *, user_name: str, character_name: str
) -> str:
    """Expand trusted character template aliases in one non-recursive pass."""

    def replacement(match: re.Match[str]) -> str:
        return user_name if match.group(0) in _USER_TOKENS else character_name

    return _TEMPLATE_TOKEN_RE.sub(replacement, source)


@dataclass(frozen=True, slots=True)
class ConsolePresentationContext:
    """Identity inputs used to project one Console transcript message."""

    user_name: str = "User"
    assistant_kind: str | None = "generic"
    character_name: str | None = None
    revision: int = 0
    transcript_style: ConsoleTranscriptStyle = DEFAULT_CONSOLE_TRANSCRIPT_STYLE


@dataclass(frozen=True, slots=True)
class ConsoleMessagePresentation:
    """Resolved, render-safe values for one Console transcript message."""

    speaker_label: str
    content: str
    row_class: str | None
    speaker_tone: str | None
    transcript_style: ConsoleTranscriptStyle
    revision_token: tuple[object, ...]


def resolve_console_message_presentation(
    message: ConsoleChatMessage, context: ConsolePresentationContext
) -> ConsoleMessagePresentation:
    """Resolve the visible Console projection without parsing markup."""
    raw_character_name = (
        context.character_name.strip()
        if isinstance(context.character_name, str)
        else ""
    )
    is_character_session = context.assistant_kind == "character" and bool(
        raw_character_name
    )
    character_display_name = sanitize_character_display_label(
        raw_character_name,
        max_characters=CHARACTER_SPEAKER_LABEL_MAX_CHARACTERS,
    )
    transcript_style = normalize_console_transcript_style(context.transcript_style)
    role_accents = transcript_style is not ConsoleTranscriptStyle.NEUTRAL
    content = message.variants.current.content if message.variants else message.content

    if message.role is ConsoleMessageRole.USER:
        speaker_label = sanitize_character_display_label(
            context.user_name,
            max_characters=CHAT_DISPLAY_NAME_MAX_CELLS,
        )
        speaker_tone = "user"
        row_class = None
        if role_accents:
            row_class = (
                "console-transcript-message-roleplay-user"
                if is_character_session
                else "console-transcript-message-role-user"
            )
    elif message.role is ConsoleMessageRole.ASSISTANT:
        speaker_label = character_display_name if is_character_session else "Assistant"
        speaker_tone = "character" if is_character_session else "assistant"
        row_class = None
        if role_accents:
            row_class = (
                "console-transcript-message-roleplay-character"
                if is_character_session
                else "console-transcript-message-role-assistant"
            )
        metadata: MessageMetadata | None = message.metadata
        template_source = getattr(metadata, "template_source", "")
        if (
            is_character_session
            and getattr(metadata, "template_kind", "") == "character_greeting"
            and isinstance(template_source, str)
            and template_source.strip()
        ):
            content = expand_character_template(
                template_source,
                user_name=context.user_name,
                character_name=raw_character_name,
            )
    else:
        speaker_label = message.role.value.title()
        row_class = None
        speaker_tone = None

    revision_token = (
        speaker_label,
        content,
        row_class,
        speaker_tone,
        transcript_style.value,
        context.revision,
    )
    return ConsoleMessagePresentation(
        speaker_label=speaker_label,
        content=content,
        row_class=row_class,
        speaker_tone=speaker_tone,
        transcript_style=transcript_style,
        revision_token=revision_token,
    )
