"""Pure markdown rendering for one Console conversation (TASK-25886).

Copy-as-markdown backs the conversation action menu's "Copy as" page: Clean
copies a shareable rendering (role headings plus verbatim user/assistant
content) and Full copies a faithful transcript (tool rows, thinking, and
system prompts included as collapsed ``<details>`` blocks so the document
stays readable in any markdown viewer).

Everything here is pure: a normalized message list in, a string out. No DOM,
no database, no store -- both sources (persisted rows from
``get_messages_for_conversation``, live open sessions from the chat store's
``messages_for_session``) adapt into ``MarkdownMessage`` at the call site,
and the fidelity rules are testable as golden strings.

Deliberate scope limits, recorded so they read as decisions:

* Message CONTENT is verbatim by design -- it is the user's own transcript,
  and the export would be useless laundered. Interpolated METADATA (titles,
  labels, summaries) is escaped and any URL-ish placeholder text is
  protocol-checked, so a hostile title cannot inject structure.
* Citations are NOT exported. The Console's citation sources are served
  through the governed, authorization-gated citation hydration graph (the
  same seam the sources modal uses); a clipboard export must not bypass
  that authorization, so v1 simply does not promise sources.
* Image bytes never enter markdown; image-bearing messages render a
  sanitized placeholder label.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

Fidelity = Literal["clean", "full"]

_ROLE_HEADINGS = {
    "user": "## User",
    "assistant": "## Assistant",
}

_SAFE_LABEL = re.compile(r"[^\w\s.,:;!?'\"()/+-]")


def _safe_label(value: object) -> str:
    """Return an interpolation-safe label for titles and placeholders.

    Args:
        value: Untrusted text (a conversation title, attachment label...).

    Returns:
        The text with markdown/HTML-structural characters stripped to
        word-ish punctuation, single-spaced, bounded to 120 characters --
        safe to interpolate into headings, summaries, and image alt text.
    """
    text = _SAFE_LABEL.sub(" ", str(value or ""))
    return " ".join(text.split())[:120]


@dataclass(frozen=True, slots=True)
class MarkdownMessage:
    """One normalized message, source-agnostic.

    Attributes:
        role: ``user`` / ``assistant`` / ``system`` / ``tool``.
        content: Verbatim text content (may be empty for image messages).
        tool_label: Display name of the tool for ``tool`` rows.
        thinking: The assistant's displayable reasoning text, when captured.
        image_label: Label for image-bearing messages, rendered as a
            placeholder (image bytes never enter markdown).
        attachments: Display names of file attachments.
    """

    role: str
    content: str = ""
    tool_label: str = ""
    thinking: str = ""
    image_label: str = ""
    attachments: tuple[str, ...] = field(default=())


def render_conversation_markdown(
    *,
    title: str,
    rendered_at: str,
    messages: list[MarkdownMessage],
    fidelity: Fidelity,
) -> str | None:
    """Render one conversation as markdown.

    Args:
        title: Conversation title for the document header.
        rendered_at: Render date (``YYYY-MM-DD``) for the header line.
        messages: Transcript-ordered normalized messages.
        fidelity: ``clean`` (user/assistant text only) or ``full``
            (everything, tool/system/thinking as details blocks).

    Returns:
        The markdown document, or None when there are no renderable
        messages (the caller gates the menu entries on this).

    Raises:
        ValueError: On an unknown fidelity value.
    """
    if fidelity not in ("clean", "full"):
        raise ValueError(f"unknown fidelity: {fidelity!r}")

    clean_messages = [
        message
        for message in messages
        if message.role in ("user", "assistant")
        or (fidelity == "full" and message.role in ("system", "tool"))
    ]
    if not clean_messages:
        return None

    lines: list[str] = [
        f"# {_safe_label(title) or 'Untitled conversation'}",
        "",
        f"_{_safe_label(rendered_at)} · {len(clean_messages)} messages_",
        "",
    ]
    for message in clean_messages:
        if message.role == "tool":
            lines.extend(
                _details(f"Tool: {_safe_label(message.tool_label) or 'tool'}", message.content)
            )
            continue
        if message.role == "system":
            lines.extend(_details("System", message.content))
            continue
        lines.append(_ROLE_HEADINGS.get(message.role, f"## {message.role.title()}"))
        lines.append("")
        if message.image_label:
            lines.append(f"![image]({_safe_label(message.image_label)})")
        if message.content:
            lines.append(message.content)
        for attachment in message.attachments:
            lines.append(f"_(attached: {_safe_label(attachment)})_")
        if not (message.image_label or message.content or message.attachments):
            lines.append("_(empty message)_")
        lines.append("")
        if fidelity == "full" and message.thinking:
            lines.extend(_details("Thinking", message.thinking))
    return "\n".join(lines).rstrip("\n") + "\n"


def _details(summary: str, body: str) -> list[str]:
    """Return a collapsed ``details`` block for non-chat content."""
    return [
        "<details>",
        f"<summary>{_safe_label(summary)}</summary>",
        "",
        body,
        "",
        "</details>",
        "",
    ]


def _role_value(role: object) -> str:
    """Return a bare role string from an enum or plain string.

    ``str(ConsoleMessageRole.USER)`` yields the CLASS name, not ``user``
    (PR #2262 review), which silently emptied every live-session export;
    enums are read through ``.value`` and plain strings pass through.
    """
    value = getattr(role, "value", role)
    return str(value or "").strip().lower()


def _thinking_text(thinking: object) -> str:
    """Extract displayable text from a live ThinkingEnvelope.

    Proprietary blocks are content-free by contract and render nothing;
    displayable blocks contribute their ``text`` in envelope order.
    """
    blocks = getattr(thinking, "blocks", None) or ()
    parts = [
        block_text
        for block in blocks
        if (block_text := str(getattr(block, "text", "") or "").strip())
    ]
    return "\n\n".join(parts)


def _attachment_names(attachments: object) -> tuple[str, ...]:
    """Return attachment display names, never their binary payloads.

    Stringifying a ``MessageAttachment`` embeds its ``data`` bytes (PR
    #2262 review); only the display name is safe metadata.
    """
    names: list[str] = []
    for attachment in attachments or ():
        name = getattr(attachment, "display_name", None)
        if name is None:
            continue
        names.append(str(name))
    return tuple(names)


def markdown_messages_from_store(
    store_messages: list[object],
) -> list[MarkdownMessage]:
    """Adapt live chat-store snapshots into normalized messages.

    Args:
        store_messages: ``ConsoleChatMessage`` snapshots from
            ``messages_for_session`` (duck-typed: only the fields copied
            below are read, so tests can pass simple stand-ins).

    Returns:
        Transcript-ordered normalized messages.
    """

    normalized: list[MarkdownMessage] = []
    for message in store_messages:
        role = _role_value(getattr(message, "role", "assistant"))
        content = str(getattr(message, "content", "") or "")
        if role == "tool":
            # Tool rows carry the untruncated result separately; the
            # content field holds the UI preview (PR #2262 review).
            full_output = getattr(message, "tool_output_full", None)
            if full_output:
                content = str(full_output)
        normalized.append(
            MarkdownMessage(
                role=role,
                content=content,
                tool_label=str(getattr(message, "tool_label", "") or ""),
                thinking=_thinking_text(getattr(message, "thinking", None)),
                image_label=str(getattr(message, "attachment_label", "") or ""),
                attachments=_attachment_names(
                    getattr(message, "attachments", ()) or ()
                ),
            )
        )
    return normalized


def markdown_messages_from_db_rows(rows: list[dict]) -> list[MarkdownMessage]:
    """Adapt persisted ``get_messages_for_conversation`` rows.

    Args:
        rows: Message row dicts, transcript order. Tool thinking arrives as
            ``thinking_blocks_json``; image presence is signalled by
            ``image_mime_type`` (there is no attachment label column).

    Returns:
        Transcript-ordered normalized messages.
    """

    normalized: list[MarkdownMessage] = []
    for row in rows:
        thinking = ""
        raw_thinking = row.get("thinking_blocks_json")
        if raw_thinking:
            try:
                blocks = json.loads(raw_thinking)
            except (TypeError, ValueError):
                blocks = None
            if isinstance(blocks, list):
                thinking = "\n\n".join(
                    str(block.get("text", "") or "")
                    for block in blocks
                    if isinstance(block, dict)
                ).strip()
        image_label = (
            "image" if str(row.get("image_mime_type", "") or "") else ""
        )
        normalized.append(
            MarkdownMessage(
                role=_role_value(row.get("sender", "") or "assistant"),
                content=str(row.get("content", "") or ""),
                thinking=thinking,
                image_label=image_label,
            )
        )
    return normalized
