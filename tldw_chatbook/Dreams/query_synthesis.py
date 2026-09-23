"""[dreams] Turning the interest-profile snapshot into search queries.

One chat call per cycle (spec §query synthesis): the prompt embeds the
decayed topics and the region verbatim, demands at least one
adjacent/serendipity angle so the cycle cannot collapse into "more of the
same", and on ANY chat failure degrades to a deterministic fallback built
straight from the top topics — a dead LLM still yields a cycle, just a
less interesting one.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

from loguru import logger

from tldw_chatbook.Chat.Chat_Functions import extract_response_content

#: Fixed sampling for the one synthesis call: a short, factual list and a
#: lowish temperature so the model lists queries instead of essaying.
_SYNTHESIS_MAX_TOKENS = 512
_SYNTHESIS_TEMPERATURE = 0.4

SYSTEM_PROMPT = (
    "You turn a user's interest profile into web search queries. Return ONE query "
    "per line, no numbering, no commentary. At least {explore} line(s) must explore "
    "something ADJACENT to their interests rather than the interests themselves "
    "(serendipity, not more of the same). Queries must be self-contained for a "
    "search engine, may include the user's region verbatim when locality helps, "
    "and must never include anything except topics, region, and search terms."
)


async def _invoke(chat: Callable[..., Any], *, system: str, user: str) -> Any:
    """Make the one chat call, accepting a sync or async seam.

    Copies ``briefing_service._invoke_chat``'s discipline: the real
    ``chat_api_call`` is synchronous blocking network I/O, so a sync
    callable is offloaded to a thread rather than run on the event loop;
    the system prompt travels in ``system_message``, not as a message
    role (each provider handler decides how its API wants it delivered).
    Endpoint, model and API key are deliberately NOT set here — Task 4's
    resolver injects ``chat`` already bound to them (a closure over
    ``chat_api_call``), so this helper carries only payload and sampling.

    Args:
        chat: ``chat_api_call``-shaped callable (kwargs per
            ``Chat/Chat_Functions.py``).
        system: System prompt text.
        user: User-turn text.

    Returns:
        The callable's response, awaited when it is awaitable.
    """
    kwargs: dict[str, Any] = {
        "messages_payload": [{"role": "user", "content": user}],
        "system_message": system,
        "streaming": False,
        "max_tokens": _SYNTHESIS_MAX_TOKENS,
        "temp": _SYNTHESIS_TEMPERATURE,
    }
    if inspect.iscoroutinefunction(chat):
        return await chat(**kwargs)
    result = await asyncio.to_thread(chat, **kwargs)
    if inspect.isawaitable(result):  # a sync callable returning an awaitable
        return await result
    return result


async def synthesize_queries(
    chat: Callable[..., Any],
    *,
    snapshot: dict,
    count: int,
    exploration_slots: int,
) -> list[str]:
    """Turn one interest snapshot into ``count`` search queries.

    Exactly one chat call is made. The user payload embeds the topic texts
    and region verbatim; the system prompt demands at least
    ``exploration_slots`` adjacent/serendipity lines. Any chat failure —
    exception, empty or unusably short response — falls back to a
    deterministic list built directly from the top topics, so the cycle
    degrades instead of dying.

    Args:
        chat: ``chat_api_call``-shaped callable (injected; Task 4 binds
            endpoint/model/api-key in a closure).
        snapshot: ``interest_profile.snapshot`` result
            (``{"topics": [...], "region": str}``).
        count: Number of queries to return.
        exploration_slots: How many lines must explore adjacent ground
            (embedded in the prompt; the fallback always contributes one
            adjacent query).

    Returns:
        Up to ``count`` query strings.
    """
    topics = [t["text"] for t in snapshot.get("topics", [])]
    region = (snapshot.get("region") or "").strip()
    user = json.dumps({"topics": topics, "region": region, "count": count},
                      ensure_ascii=False)
    try:
        resp = await _invoke(
            chat, system=SYSTEM_PROMPT.format(explore=exploration_slots), user=user
        )
        lines = [ln.strip() for ln in extract_response_content(resp).splitlines()
                 if ln.strip()][:count]
        if len(lines) >= max(1, min(count, 2)):
            return lines
        logger.debug("Dreams query synthesis returned {} usable lines; falling back", len(lines))
    except Exception as exc:  # noqa: BLE001 - degradation, not a crash path
        logger.warning("Dreams query synthesis chat call failed; using fallback: {}", exc)
    return _fallback_queries(topics, count)


def preview_queries(topics: list[str], count: int) -> list[str]:
    """Public preview seam over the fallback list (Task 7's modal only).

    The Artifacts story-detail modal renders a "what we'll look for"
    preview of the NEXT cycle's queries without ever calling an LLM from
    the UI, so it shows exactly what a degraded cycle would search. That
    is this function: the deterministic fallback, exposed publicly so the
    modal never imports the private helper (controller-authorized wrapper,
    Task 7).

    Args:
        topics: Current snapshot topic texts, heaviest first.
        count: The cycle's query budget (``queries_per_cycle``).

    Returns:
        The same list ``_fallback_queries`` builds.
    """
    return _fallback_queries(topics, count)


def _fallback_queries(topics: list[str], count: int) -> list[str]:
    """Deterministic degraded-cycle queries straight from the top topics.

    Args:
        topics: Snapshot topic texts, heaviest first.
        count: Number of queries to return.

    Returns:
        ``"<topic> recent developments"`` per top topic plus one
        ``"surprising adjacent to <top topic>"`` line, capped at ``count``.
    """
    out = [f"{t} recent developments" for t in topics[: max(count - 1, 1)]]
    out.append(f"surprising adjacent to {topics[0]}" if topics else "curious new things this week")
    return out[:count]
