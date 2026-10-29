"""Pure Markdown projections of retained, displayable Console traces."""

from __future__ import annotations

from collections.abc import Sequence
import re

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleThinkingActivityRef,
)
from tldw_chatbook.Chat.console_turn_grouping import (
    ConsoleAssistantTurn,
    group_console_transcript_messages,
    ordered_assistant_activities,
)
from tldw_chatbook.Chat.thinking_blocks import DisplayableThinkingBlock


def _label(text: str) -> str:
    """Escape metadata for a single Markdown heading."""
    return re.sub(r"([\\`*{}\[\]<>#+|])", r"\\\1", " ".join(text.split()))


def _fenced(text: str, language: str = "text") -> str:
    """Preserve the payload inside a fence it cannot close."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    separator = "" if text.endswith("\n") else "\n"
    return f"{fence}{language}\n{text}{separator}{fence}\n"


def _media(message: ConsoleChatMessage) -> str:
    labels = []
    if message.attachment_label:
        labels.append(message.attachment_label)
    for attachment in message.attachments:
        name = getattr(attachment, "display_name", None)
        if isinstance(name, str) and name not in labels:
            labels.append(name)
    return "".join(f"\nAttachment: {_label(label)}\n" for label in labels)


def _activity(message: ConsoleChatMessage) -> str:
    presentation = message.activity_presentation
    label = presentation.label if presentation is not None else "Activity"
    status = presentation.status if presentation is not None else message.status
    sections = [f"## Tool: {_label(label)}\n\nStatus: {_label(status)}\n"]
    if presentation is not None and presentation.arguments:
        sections.append("\n### Arguments\n\n" + _fenced(presentation.arguments, "json"))
    else:
        sections.append("\nArguments unavailable.\n")
    result = message.tool_output_full
    if result is None:
        if presentation is not None and presentation.result_preview is not None:
            result = presentation.result_preview
        elif presentation is None:
            result = message.content
        elif status not in {"queued", "awaiting_approval", "running"}:
            result = message.content
    if result:
        sections.append("\n### Result\n\n" + _fenced(result))
    else:
        sections.append("\nResult unavailable.\n")
    if message.tool_diff is not None:
        path, before, after = message.tool_diff
        sections.append(f"\n### File change: {_label(path)}\n\nBefore:\n\n")
        sections.append(_fenced(before))
        sections.append("\nAfter:\n\n" + _fenced(after))
    return "".join(sections)


def find_response_turn(
    messages: Sequence[ConsoleChatMessage], message_id: str
) -> ConsoleAssistantTurn | None:
    """Resolve a message or owned tool to its enclosing response.

    Args:
        messages: Active-path snapshots in causal store order.
        message_id: Exact assistant or tool message identifier.

    Returns:
        The owning assistant turn, or None for a missing/standalone message.
    """
    for unit in group_console_transcript_messages(messages):
        turn = unit.assistant_turn
        if turn is not None and message_id in turn.owned_message_ids:
            return turn
    return None


def render_response_trace_markdown(turn: ConsoleAssistantTurn) -> str:
    """Render one response without reading widgets, providers, or storage.

    Args:
        turn: Detached response and its retained activity snapshots.

    Returns:
        Expanded Markdown containing only displayable retained evidence.
    """
    assistant = turn.assistant
    state = assistant.status
    suffix = " (incomplete snapshot)" if state in {"pending", "streaming"} else ""
    sections = [f"# Assistant response\n\nStatus: {_label(state)}{suffix}\n"]
    if assistant.generation_projection_quarantined:
        return (
            "".join(sections)
            + "\nResponse unavailable until its canonical generation is reloaded.\n"
        )
    envelope = assistant.thinking
    blocks = {block.block_id: block for block in envelope.blocks} if envelope else {}
    if not blocks and (assistant.thinking_warning or assistant.opaque_thinking_json):
        sections.append("\n## Thinking\n\nThinking unavailable.\n")
    for activity in ordered_assistant_activities(turn):
        if isinstance(activity, ConsoleThinkingActivityRef):
            block = blocks[activity.block_id]
            sections.append(f"\n## Thinking (round {block.round_ordinal + 1})\n\n")
            if isinstance(block, DisplayableThinkingBlock):
                sections.append(_fenced(block.text))
            else:
                sections.append("Proprietary thinking unavailable.\n")
        else:
            sections.append("\n" + _activity(activity))
    heading = "Final answer" if state == "complete" else "Response so far"
    sections.append(f"\n## {heading}\n\n{assistant.content}")
    if not assistant.content.endswith("\n"):
        sections.append("\n")
    sections.append(_media(assistant))
    return "".join(sections)


def render_chat_trace_markdown(
    messages: Sequence[ConsoleChatMessage], *, title: str
) -> str | None:
    """Compose full response traces and user messages for one active branch.

    Args:
        messages: Detached active-path snapshots, including unmounted history.
        title: Source conversation title used only as escaped metadata.

    Returns:
        Expanded Markdown, or None when no visible messages are present.
    """
    sections = []
    for unit in group_console_transcript_messages(messages):
        if unit.assistant_turn is not None:
            sections.append(render_response_trace_markdown(unit.assistant_turn))
        elif unit.standalone is not None:
            message = unit.standalone
            if message.role == ConsoleMessageRole.USER:
                sections.append(f"# User\n\n{message.content}\n{_media(message)}")
            elif message.role == ConsoleMessageRole.TOOL:
                sections.append(_activity(message))
    if not sections:
        return None
    return f"# Chat: {_label(title) or 'Untitled conversation'}\n\n" + "\n---\n\n".join(
        sections
    )
