"""Bound Console conversation history by real tokens before dispatch.

Pure counting + whole-turn trimming, consumed by ConsoleChatController at the
dispatch choke point. Depends only on the token_counter seam (get_model_token_
limit / count_tokens_messages), which tasks 320/321 sharpen later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tldw_chatbook.Utils.token_counter import (
    count_tokens_messages,
    get_model_token_limit,
)

DEFAULT_RESPONSE_RESERVATION = 1024
DEFAULT_PER_IMAGE_TOKENS = 1024
_MIN_SAFETY_MARGIN = 512


@dataclass(frozen=True)
class BoundResult:
    """Result of trimming a provider message list to the model window."""

    messages: list[dict[str, Any]]
    dropped_count: int


def count_console_messages_tokens(
    messages: list[dict[str, Any]],
    model: str,
    *,
    per_image_tokens: int = DEFAULT_PER_IMAGE_TOKENS,
) -> int:
    """Token count for Console provider payloads, multimodal-aware.

    ``count_tokens_messages`` assumes string ``content`` and crashes on the
    Console's vision payloads (``content`` is a list of ``{type:text}`` /
    ``{type:image_url}`` parts). This flattens each list ``content`` to its
    concatenated text before delegating to ``count_tokens_messages`` (so
    text counting stays byte-identical, and 320/321 flow through), then adds
    ``per_image_tokens`` per image part.

    Args:
        messages: Provider payload dicts (``role``/``content``).
        model: Model name for the underlying tokenizer.
        per_image_tokens: Flat token estimate charged per image part.

    Returns:
        Estimated total prompt tokens.
    """
    flattened: list[dict[str, Any]] = []
    image_count = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            texts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            image_count += sum(
                1
                for part in content
                if isinstance(part, dict) and part.get("type") != "text"
            )
            flattened.append(
                {
                    **message,
                    "content": " ".join(
                        t for t in texts if isinstance(t, str) and t
                    ),
                }
            )
        else:
            flattened.append(message)
    return count_tokens_messages(flattened, model) + per_image_tokens * image_count


def _group_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group middle history into whole turns (a user + its following rows).

    Any rows before the first user message (e.g. a leading orphan assistant)
    form their own first group. Dropping a whole group never splits a
    user/assistant pair — nor a tool_call/tool_result pair, were tool rows
    ever present in the payload.
    """
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "user" and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    return turns


def bound_messages_to_window(
    messages: list[dict[str, Any]],
    *,
    model: str,
    provider: str,
    response_reservation: int,
    per_image_tokens: int = DEFAULT_PER_IMAGE_TOKENS,
    window: int | None = None,
    count_fn: Callable[[list[dict[str, Any]], str], int] | None = None,
) -> BoundResult:
    """Drop oldest whole turns until the payload fits the model window.

    Always preserves the leading system prefix and the current turn (from the
    last user message to the end). Returns the trimmed list and how many
    history messages were removed.

    Args:
        messages: Full provider payload, post dictionaries/skills.
        model: Model name (tokenizer + window lookup).
        provider: Provider name (window lookup fallback).
        response_reservation: Tokens reserved for the reply.
        per_image_tokens: Per-image token estimate.
        window: Explicit context window; ``None`` uses the token_counter lookup.
        count_fn: Injectable counter ``(messages, model) -> int``; ``None``
            uses ``count_console_messages_tokens``.

    Returns:
        ``BoundResult(messages, dropped_count)``.
    """
    counter = count_fn or (
        lambda msgs, mdl: count_console_messages_tokens(
            msgs, mdl, per_image_tokens=per_image_tokens
        )
    )
    win = window if window is not None else get_model_token_limit(model, provider)
    budget = win - response_reservation - max(_MIN_SAFETY_MARGIN, win // 50)

    # System prefix = contiguous leading system rows.
    sys_end = 0
    while sys_end < len(messages) and messages[sys_end].get("role") == "system":
        sys_end += 1
    system_prefix = messages[:sys_end]
    rest = messages[sys_end:]

    # Current turn = from the last user message to the end.
    last_user = None
    for index in range(len(rest) - 1, -1, -1):
        if rest[index].get("role") == "user":
            last_user = index
            break
    if last_user is None:
        # No user turn to anchor on -- nothing safe to trim.
        return BoundResult(messages, 0)

    current_turn = rest[last_user:]
    kept_turns = _group_turns(rest[:last_user])

    def assemble(drop: int) -> list[dict[str, Any]]:
        return (
            system_prefix
            + [m for turn in kept_turns[drop:] for m in turn]
            + current_turn
        )

    # Drop oldest whole turns until the payload fits. The token count is
    # monotonically non-increasing as more turns drop (each turn contributes
    # >= 0 tokens), so binary-search the minimal number of oldest turns to
    # drop rather than re-counting the whole payload once per dropped turn
    # (O(n^2) on the long histories this trimmer exists for). The chosen drop
    # count -- and thus the returned messages -- is identical to dropping one
    # turn at a time.
    lo, hi = 0, len(kept_turns)
    best = hi  # if nothing fits, drop every middle turn
    while lo <= hi:
        mid = (lo + hi) // 2
        if counter(assemble(mid), model) <= budget:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    dropped = sum(len(turn) for turn in kept_turns[:best])
    return BoundResult(assemble(best), dropped)
