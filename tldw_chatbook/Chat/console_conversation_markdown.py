"""Pure markdown rendering for one Console conversation (TASK-25714).

Copy-as-markdown backs the conversation action menu's "Copy as" page: Clean
copies a shareable rendering (role headings plus verbatim user/assistant
content) and Full copies a faithful transcript (tool rows, thinking, system
prompts, and citations included as collapsed ``<details>`` blocks so the
document stays readable in any markdown viewer).

Everything here is pure: a normalized message list in, a string out. No DOM,
no database, no store -- both sources (persisted rows from
``get_messages_for_conversation``, live open sessions from the chat store's
``messages_for_session``) adapt into ``MarkdownMessage`` at the call site,
and the fidelity rules are testable as golden strings.

Scope note (deliberate): the DB adapter below reconstructs what was
PERSISTED -- thinking blocks and tool message content survive, but live
open sessions render richest because their in-flight tool-call structure
never needed serializing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Fidelity = Literal["clean", "full"]

_ROLE_HEADINGS = {
    "user": "## User",
    "assistant": "## Assistant",
}


@dataclass(frozen=True, slots=True)
class MarkdownMessage:
    """One normalized message, source-agnostic.

    Attributes:
        role: ``user`` / ``assistant`` / ``system`` / ``tool``.
        content: Verbatim text content (may be empty for image messages).
        tool_label: Display name of the tool for ``tool`` rows.
        thinking: The assistant's reasoning text, when captured.
        citations: ``(label, url)`` pairs cited by this message.
        image_label: Attachment label for image-bearing messages, rendered
            as a placeholder (image bytes never enter markdown).
    """

    role: str
    content: str = ""
    tool_label: str = ""
    thinking: str = ""
    citations: tuple[tuple[str, str], ...] = ()
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
        f"# {title or 'Untitled conversation'}",
        "",
        f"_{rendered_at} · {len(clean_messages)} messages_",
        "",
    ]
    for message in clean_messages:
        if message.role == "tool":
            lines.extend(_details(f"Tool: {message.tool_label or 'tool'}", message.content))
            continue
        if message.role == "system":
            lines.extend(_details("System", message.content))
            continue
        lines.append(_ROLE_HEADINGS.get(message.role, f"## {message.role.title()}"))
        lines.append("")
        if message.image_label:
            lines.append(f"![image]({message.image_label})")
        if message.content:
            lines.append(message.content)
        for attachment in message.attachments:
            lines.append(f"_(attached: {attachment})_")
        if not (message.image_label or message.content or message.attachments):
            lines.append("_(empty message)_")
        lines.append("")
        if fidelity == "full" and message.thinking:
            lines.extend(_details("Thinking", message.thinking))
        if message.citations:
            lines.append("**Sources**")
            lines.append("")
            lines.extend(
                f"- [{label}]({url})" for label, url in message.citations
            )
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _details(summary: str, body: str) -> list[str]:
    """Return a collapsed ``details`` block for non-chat content."""
    return [
        "<details>",
        f"<summary>{summary}</summary>",
        "",
        body,
        "",
        "</details>",
        "",
    ]


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
        citations = [
            (
                str(getattr(citation, "label", "") or "source"),
                str(getattr(citation, "url", "") or ""),
            )
            for citation in getattr(message, "citations", ()) or ()
        ]
        normalized.append(
            MarkdownMessage(
                role=str(getattr(message, "role", "assistant")),
                content=str(getattr(message, "content", "") or ""),
                tool_label=str(getattr(message, "tool_label", "") or ""),
                thinking=str(getattr(message, "thinking", "") or ""),
                citations=tuple(
                    (label, url) for label, url in citations if url
                ),
                image_label=str(getattr(message, "attachment_label", "") or ""),
                attachments=tuple(
                    str(attachment)
                    for attachment in getattr(message, "attachments", ()) or ()
                ),
            )
        )
    return normalized


def markdown_messages_from_db_rows(rows: list[dict]) -> list[MarkdownMessage]:
    """Adapt persisted ``get_messages_for_conversation`` rows.

    Args:
        rows: Message row dicts, transcript order. Tool thinking arrives as
            ``thinking_blocks_json``; tool rows are the ``tool`` sender.

    Returns:
        Transcript-ordered normalized messages.
    """

    import json

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
        normalized.append(
            MarkdownMessage(
                role=str(row.get("sender", "") or "assistant").lower(),
                content=str(row.get("content", "") or ""),
                thinking=thinking,
                image_label=str(row.get("attachment_label", "") or ""),
            )
        )
    return normalized
