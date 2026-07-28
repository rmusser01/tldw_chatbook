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
    #: How many whole GROUPS (turns, or -- for a caller that supplies its
    #: own `is_turn_boundary` -- whatever unit that predicate delimits)
    #: were dropped. Defaults to 0 so every pre-existing positional
    #: `BoundResult(messages, dropped_count)` construction in this module
    #: stays valid. `dropped_count` alone (a message count) cannot answer
    #: "how many turns" for a caller that wants to report that to a user
    #: or model (task-1272, Phase 3's synthetic eviction note).
    dropped_turns: int = 0


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


def _group_turns(
    messages: list[dict[str, Any]],
    *,
    is_boundary: Callable[[dict[str, Any]], bool] | None = None,
) -> list[list[dict[str, Any]]]:
    """Group middle history into whole turns (a boundary row + its followers).

    Any rows before the first boundary (e.g. a leading orphan assistant)
    form their own first group. Dropping a whole group never splits a
    user/assistant pair — nor a tool_call/tool_result pair, provided
    ``is_boundary`` correctly identifies every row that must stay attached
    to its predecessor rather than start a new group.

    Args:
        messages: The message slice to group (already sans any leading
            system prefix and the current turn).
        is_boundary: Predicate deciding whether a message starts a new
            turn. Defaults to ``role == "user"`` — correct for Console's
            own payloads, which never carry tool rows, and this
            function's original contract. A caller whose payload DOES
            carry tool rows (e.g. an agent run) must supply a
            protocol-aware predicate, or a tool_call/tool_result pair can
            be split across groups — see
            ``Agents.run_log_eviction._make_round_boundary``.

    Returns:
        Turns in original order, each a non-empty list of messages.
    """
    boundary = is_boundary or (lambda message: message.get("role") == "user")
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if boundary(message) and current:
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
    is_turn_boundary: Callable[[dict[str, Any]], bool] | None = None,
) -> BoundResult:
    """Drop oldest whole turns until the payload fits the model window.

    Always preserves the leading system prefix and the current turn (from the
    last boundary row to the end). Returns the trimmed list and how many
    history messages/turns were removed.

    Args:
        messages: Full provider payload, post dictionaries/skills.
        model: Model name (tokenizer + window lookup).
        provider: Provider name (window lookup fallback).
        response_reservation: Tokens reserved for the reply.
        per_image_tokens: Per-image token estimate.
        window: Explicit context window; ``None`` uses the token_counter lookup.
        count_fn: Injectable counter ``(messages, model) -> int``; ``None``
            uses ``count_console_messages_tokens``.
        is_turn_boundary: Optional protocol-aware turn-boundary predicate,
            forwarded to ``_group_turns`` and used to anchor the current
            turn (scanning from the end for the last row this predicate
            accepts). ``None`` (every Console call site) keeps the
            original ``role == "user"`` rule. An agent-run caller passes
            its own predicate here — see
            ``Agents.run_log_eviction._make_round_boundary`` — because an
            agent payload carries tool rows Console's payloads never do.

    Returns:
        ``BoundResult(messages, dropped_count, dropped_turns)``.
    """
    counter = count_fn or (
        lambda msgs, mdl: count_console_messages_tokens(
            msgs, mdl, per_image_tokens=per_image_tokens
        )
    )
    win = window if window is not None else get_model_token_limit(model, provider)
    margin = max(_MIN_SAFETY_MARGIN, win // 50)
    # The reply reservation may never consume the whole window. `max_tokens` is
    # user-facing and routinely set to the full context size (Console's own
    # default did exactly that), which made `budget` negative and silently
    # dropped ALL history on EVERY send -- the model then had no memory of the
    # conversation while still sounding in-character, because the system prefix
    # is always preserved. Clamping to half the usable window guarantees history
    # always keeps a share; a caller asking to reserve more just gets less reply
    # room than requested, which is recoverable, unlike total amnesia.
    usable = max(0, win - margin)
    effective_reservation = min(max(0, response_reservation), usable // 2)
    budget = win - effective_reservation - margin

    # System prefix = contiguous leading system rows.
    sys_end = 0
    while sys_end < len(messages) and messages[sys_end].get("role") == "system":
        sys_end += 1
    system_prefix = messages[:sys_end]
    rest = messages[sys_end:]

    boundary = is_turn_boundary or (lambda message: message.get("role") == "user")

    # Current turn = from the last boundary row to the end.
    last_user = None
    for index in range(len(rest) - 1, -1, -1):
        if boundary(rest[index]):
            last_user = index
            break
    if last_user is None:
        # No turn boundary to anchor on -- nothing safe to trim.
        return BoundResult(messages, 0)

    current_turn = rest[last_user:]
    kept_turns = _group_turns(rest[:last_user], is_boundary=boundary)

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
    return BoundResult(assemble(best), dropped, best)
