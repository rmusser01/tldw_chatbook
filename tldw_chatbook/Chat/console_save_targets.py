"""Pure title and payload derivation for Console "Save as..." destinations.

These helpers keep the Console save-as apply paths (Note / Media / Prompt /
Chatbook in ``UI/Screens/chat_screen.py``) free of copy/format logic so the
derivations stay unit-testable without a running app.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from tldw_chatbook.Chat.citation_artifact_ownership import (
    CitationArtifactOwnershipCoordinator,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationArtifactOwnerRequest,
)

CONSOLE_SAVE_TITLE_MAX_CHARS = 80
CONSOLE_SAVE_TITLE_PREFIX = "Console message"
#: task-32146: title for a captured answer that turns out to be all
#: whitespace. The capture action is only offered on non-blank completed
#: assistant rows, so this is a floor, not an expected outcome.
CONSOLE_NOTE_FALLBACK_TITLE = "Console answer"
#: task-32146 fix round 1: a captured answer that opens with a fenced code
#: block or a markdown heading must not be titled by the fence or the
#: hashes -- the Library notes list shows titles verbatim.
_NOTE_TITLE_FENCE_LINE = re.compile(r"^\s*(?:`{3,}|~{3,})")
_NOTE_TITLE_MARKDOWN_PREFIX = re.compile(r"^[\s#>]+")

# Stable bounds for Console chatbook artifacts consumed by Artifacts and Home.
CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS = 20_000
CONSOLE_CHATBOOK_ARTIFACT_DESCRIPTION_MAX_CHARS = 280


def resolve_console_artifact_owner_request(
    *,
    coordinator: CitationArtifactOwnershipCoordinator | None,
    persisted_message_id: str | None,
    message_text: str,
) -> CitationArtifactOwnerRequest | None:
    """Resolve ownership only from the current persisted message row."""

    if (
        coordinator is None
        or getattr(coordinator, "writes_enabled", False) is not True
        or not isinstance(persisted_message_id, str)
        or not persisted_message_id
        or len(persisted_message_id.encode("utf-8")) > 256
    ):
        return None
    repository = getattr(coordinator, "trace_repository", None)
    db = getattr(repository, "db", None)
    get_message = getattr(db, "get_message_by_id", None)
    owner_request = getattr(coordinator, "owner_request_for_message", None)
    if not callable(get_message) or not callable(owner_request):
        return None
    try:
        persisted = get_message(persisted_message_id)
        revision = persisted["version"]
        if (
            persisted.get("id") != persisted_message_id
            or persisted.get("deleted", 0) != 0
            or persisted.get("content") != message_text
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
        ):
            return None
        return owner_request(
            message_id=persisted_message_id,
            message_revision=revision,
            current_body=message_text,
        )
    except Exception:
        return None


def _collapse_whitespace(text: Any) -> str:
    return " ".join(str(text or "").split())


def derive_console_save_title(
    conversation_title: str,
    *,
    role_label: str = "",
    now: datetime | None = None,
    max_length: int = CONSOLE_SAVE_TITLE_MAX_CHARS,
) -> str:
    """Derive a save title like ``Console message — <conversation> (2026-07-11)``.

    Args:
        conversation_title: Title of the Console conversation the message
            belongs to. Blank titles fall back to the prefix alone.
        role_label: Optional message role woven into the prefix, producing
            e.g. ``Console assistant message — ...``.
        now: Timestamp used for the date suffix; defaults to UTC now.
        max_length: Hard cap for the returned title.

    Returns:
        A single-line title bounded to ``max_length`` characters.
    """
    role = _collapse_whitespace(role_label).lower()
    prefix = f"Console {role} message" if role else CONSOLE_SAVE_TITLE_PREFIX
    moment = now if now is not None else datetime.now(timezone.utc)
    date_suffix = f" ({moment.strftime('%Y-%m-%d')})"
    normalized_title = _collapse_whitespace(conversation_title)
    if not normalized_title:
        return f"{prefix}{date_suffix}"[:max_length]
    separator = " — "
    available = max_length - len(prefix) - len(separator) - len(date_suffix)
    if available < 1:
        return f"{prefix}{date_suffix}"[:max_length]
    if len(normalized_title) > available:
        # available >= 1 here, so reserving one slot for the ellipsis keeps
        # the truncated title within budget even at available == 1.
        normalized_title = f"{normalized_title[: available - 1].rstrip()}…"
    return f"{prefix}{separator}{normalized_title}{date_suffix}"


def console_answer_note_title(
    answer_text: Any,
    *,
    max_length: int = CONSOLE_SAVE_TITLE_MAX_CHARS,
) -> str:
    """Title a captured answer with the answer's own first line.

    Unlike ``derive_console_save_title`` (which names the CONVERSATION a
    saved message came from), a captured note is filed by what it SAYS --
    the Library notes list shows titles, and "Console message — Chat 1"
    repeated per capture is unreadable there.

    Args:
        answer_text: The assistant answer being captured.
        max_length: Hard cap for the returned title.

    Returns:
        The first non-blank line that is not a code fence, with any
        leading heading/quote markers dropped, whitespace-collapsed and
        bounded, or ``CONSOLE_NOTE_FALLBACK_TITLE`` when there is no text
        at all.
    """
    for line in str(answer_text or "").splitlines():
        if _NOTE_TITLE_FENCE_LINE.match(line):
            continue
        normalized = _collapse_whitespace(
            _NOTE_TITLE_MARKDOWN_PREFIX.sub("", line, count=1)
        )
        if not normalized:
            continue
        if len(normalized) > max_length:
            return f"{normalized[: max_length - 1].rstrip()}…"
        return normalized
    return CONSOLE_NOTE_FALLBACK_TITLE


def console_note_provenance_keywords(
    *,
    conversation_id: Any,
    message_id: Any,
) -> list[str]:
    """Return the keywords that record where a captured note came from.

    task-32146 AC#2. An unpersisted (temporary) session has no conversation
    id, so that tag is omitted rather than written blank -- a
    ``conversation:`` tag with nothing after it would match every other
    provenance-less note in a keyword search.

    Args:
        conversation_id: Persisted Chat conversation id, if the session has
            one.
        message_id: Console transcript message id being captured.

    Returns:
        ``["console", "conversation:<id>", "message:<id>"]``, minus any part
        with no id to record.
    """
    keywords = ["console"]
    conversation = _collapse_whitespace(conversation_id)
    if conversation:
        keywords.append(f"conversation:{conversation}")
    message = _collapse_whitespace(message_id)
    if message:
        keywords.append(f"message:{message}")
    return keywords


def console_message_preview(
    message_text: str,
    *,
    max_length: int = CONSOLE_CHATBOOK_ARTIFACT_DESCRIPTION_MAX_CHARS,
) -> str:
    """Return a single-line, bounded preview of a Console message.

    Args:
        message_text: Raw message content; whitespace runs are collapsed.
        max_length: Hard cap for the returned preview.

    Returns:
        The collapsed text, truncated with a trailing ``...`` when it
        exceeds ``max_length``.
    """
    preview = _collapse_whitespace(message_text)
    if len(preview) > max_length:
        preview = preview[: max(1, max_length - 3)].rstrip() + "..."
    return preview


def console_chatbook_artifact_payload(
    *,
    title: str,
    message_text: str,
    message_role: str,
    conversation_id: str | None = None,
    message_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Build the ``LocalChatbookService.create_chatbook`` payload for one message.

    The ``artifact_source``/``artifact_kind`` metadata keys mark the record as
    a Console-saved artifact so the Artifacts screen, Home surfaces, and
    ``LocalChatbookService.list_home_artifact_snapshot`` recognize it.

    Args:
        title: Display name for the registry record.
        message_text: Full message content; bounded copy is stored in metadata.
        message_role: User-facing role label for the saved message.
        conversation_id: Optional persisted conversation id for provenance.
        message_id: Optional Console transcript message id.
        provider: Optional provider label active when the message was produced.
        model: Optional model label active when the message was produced.

    Returns:
        Keyword arguments for ``create_chatbook``.
    """
    content = str(message_text or "")
    metadata: dict[str, Any] = {
        "artifact_source": "console",
        "artifact_kind": "assistant-response",
        "message_role": _collapse_whitespace(message_role) or "Assistant",
        "content": content[:CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS],
        "content_truncated": len(content) > CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS,
    }
    for key, value in (
        ("conversation_id", conversation_id),
        ("message_id", message_id),
        ("provider", provider),
        ("model", model),
    ):
        normalized = _collapse_whitespace(value)
        if normalized:
            metadata[key] = normalized
    preview = console_message_preview(message_text)
    description = (
        f"Saved from Console assistant response. Preview: {preview}"
        if preview
        else "Saved from Console assistant response."
    )
    return {
        "name": str(title),
        "description": description,
        "tags": ["console", "artifact"],
        "categories": ["Console", "Artifacts"],
        "metadata": metadata,
    }
