"""Pure, lazy descriptors for the Conversation Inspector's three views."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...Chat.console_chat_models import ConsoleContextSnapshot
    from ...Chat.console_cost_tracker import ConsoleCostRow
    from .console_conversation_inspector import InspectorTurn

MAX_DETAIL_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class InspectorSection:
    key: str
    group: str
    label: str
    item_count: int = 0


@dataclass(frozen=True, slots=True)
class InspectorDetail:
    key: str
    text: str
    raw_json: str | None
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class InspectorUsageItem:
    message_key: str
    row: ConsoleCostRow
    title: str


def context_sections(snapshot: ConsoleContextSnapshot) -> tuple[InspectorSection, ...]:
    """Enumerate metadata only; prompt bodies never enter section labels."""
    sections = [
        InspectorSection(
            "current:messages",
            "Current conversation",
            "Retained messages",
            len(snapshot.current_messages),
        )
    ]
    for field, value in snapshot.next_send_payload.items():
        sections.append(
            InspectorSection(
                "preview:" + field,
                "Next send preview",
                field.replace("_", " ").capitalize(),
                len(value) if isinstance(value, list) else 0,
            )
        )
    return tuple(sections)


def context_detail(snapshot: ConsoleContextSnapshot, key: str) -> InspectorDetail:
    """Format only the chosen section; callers perform this work off-loop."""
    if key == "current:messages":
        value = [
            {
                "id": message.id,
                "role": getattr(message.role, "value", message.role),
                "content": message.content,
            }
            for message in snapshot.current_messages
        ]
    elif key.startswith("preview:") and key[8:] in snapshot.next_send_payload:
        value = snapshot.next_send_payload[key[8:]]
    else:
        return InspectorDetail(key, "No content in this section.", None)
    raw = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    if len(raw.encode("utf-8")) > MAX_DETAIL_BYTES:
        return InspectorDetail(
            key,
            "Section exceeds 1 MiB. Use Export payload to view the full prepared request.",
            None,
            True,
        )
    if isinstance(value, str):
        readable = value or "Empty section."
    elif isinstance(value, list) and all(
        isinstance(item, dict) and "content" in item for item in value
    ):
        readable = (
            "\n\n".join(
                f"{item.get('role', 'Message')}\n{item['content']}" for item in value
            )
            or "No messages yet. Write a prompt to prepare the next send."
        )
    else:
        readable = raw
    return InspectorDetail(key, readable, raw)


def usage_items(
    rows: Sequence[ConsoleCostRow], turns: Sequence[InspectorTurn]
) -> tuple[InspectorUsageItem, ...]:
    """Resolve a display index once, refusing missing or ambiguous identities."""
    by_index = Counter(turn.index for turn in turns)
    by_id = Counter(turn.native_message_id for turn in turns)
    unique = {
        turn.index: turn
        for turn in turns
        if by_index[turn.index] == 1
        and turn.native_message_id
        and by_id[turn.native_message_id] == 1
    }
    result = []
    for row in rows:
        turn = unique.get(row.index)
        basis = "Estimated" if row.estimated else "Reported"
        cost = "Unpriced" if row.cost_usd is None else f"${row.cost_usd:.4f}"
        title = f"{row.index + 1:>3} {row.role:<9} {row.uncached_input + row.cache_read + row.cache_write:>7} in {row.output:>6} out  {cost} · {basis}"
        result.append(
            InspectorUsageItem(turn.native_message_id if turn else "", row, title)
        )
    return tuple(result)
