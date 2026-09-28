"""[dreams] Turning the interest-profile snapshot into search queries.

One chat call per cycle (spec §query synthesis): the prompt embeds the
decayed topics, the SEARCHABLE goals, and the region verbatim, demands at
least one adjacent/serendipity angle so the cycle cannot collapse into
"more of the same", and on ANY chat failure degrades to a deterministic
fallback built straight from the top topics and searchable goals — a dead
LLM still yields a cycle, just a less interesting one.

Privacy gate (spec §Privacy, binding): a goal's text enters the payload or
a fallback line ONLY when its ``searchable`` flag is 1; the filter runs
before any payload or query string is built.
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
    "(serendipity, not more of the same). When goals are given, goal texts drive "
    "event, deal, and social-opportunity query angles (tickets, dates, local "
    "openings) while topics drive content angles. Queries must be "
    "self-contained for a "
    "search engine, may include the user's region verbatim when locality helps, "
    "and must never include anything except topics, goals, region, and search "
    "terms."
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

    Exactly one chat call is made. The user payload embeds the topic
    texts, the SEARCHABLE goal texts, and the region verbatim; the system
    prompt demands at least ``exploration_slots`` adjacent/serendipity
    lines. Any chat failure — exception, empty or unusably short response
    — falls back to a deterministic list built directly from the top
    topics and searchable goals, so the cycle degrades instead of dying.

    Args:
        chat: ``chat_api_call``-shaped callable (injected; Task 4 binds
            endpoint/model/api-key in a closure).
        snapshot: ``interest_profile.snapshot`` result
            (``{"topics": [...], "goals": [...], "region": str}``).
        count: Number of queries to return.
        exploration_slots: How many lines must explore adjacent ground
            (embedded in the prompt; the fallback always contributes one
            adjacent query).

    Returns:
        Up to ``count`` query strings.
    """
    topics = [t["text"] for t in snapshot.get("topics", [])]
    # PRIVACY GATE (spec §Privacy, binding): a goal's text may leave the
    # machine only when its ``searchable`` flag is 1. The filter runs HERE,
    # before the payload exists, so an unsearchable goal's text appears
    # nowhere in anything sent to the model.
    goals = [str(g["text"]) for g in snapshot.get("goals", [])
             if int(g.get("searchable", 1) or 0)]
    region = (snapshot.get("region") or "").strip()
    user = json.dumps(
        {"topics": topics, "goals": goals, "region": region, "count": count},
        ensure_ascii=False,
    )
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
        logger.warning("Dreams query synthesis chat call failed; using fallback: {}",
                       type(exc).__name__)
    return [row["query"] for row in _fallback_queries(topics, goals, count)]


def preview_queries(
    topics: list[str], goals: list[dict], *, count: int
) -> list[dict]:
    """Public preview seam over the fallback list (the modals' only route).

    The Artifacts story-detail modal renders a "what we'll look for"
    preview of the NEXT cycle's queries without ever calling an LLM from
    the UI, so it shows exactly what a degraded cycle would search. That
    is this function: the deterministic fallback, exposed publicly so the
    modal never imports the private helper (controller-authorized wrapper,
    Task 7). Rows carry ``goal_derived`` so the preview can label which
    lines came from goals; the same privacy gate as synthesis applies —
    unsearchable goals are filtered here and never contribute a line.

    Args:
        topics: Current snapshot topic texts, heaviest first.
        goals: The snapshot's goal rows (dicts carrying ``text`` and
            ``searchable``); searchable filtering happens inside.
        count: The cycle's query budget (``queries_per_cycle``).

    Returns:
        ``{"query": str, "goal_derived": bool}`` rows, the same list
        ``_fallback_queries`` builds.
    """
    searchable = [str(g["text"]) for g in goals
                  if int(g.get("searchable", 1) or 0)]
    return _fallback_queries(topics, searchable, count)


def _fallback_queries(
    topics: list[str], goals: list[str], count: int
) -> list[dict]:
    """Deterministic degraded-cycle queries from the top topics and goals.

    Args:
        topics: Snapshot topic texts, heaviest first.
        goals: SEARCHABLE goal texts only (callers filter; this helper is
            the privacy gate's innermost trust boundary).
        count: Number of queries to return.

    Returns:
        ``{"query": str, "goal_derived": bool}`` rows — one
        ``"<topic> recent developments"`` per top topic, at most one
        ``"<goal> events and tickets"`` per goal, plus one
        ``"surprising adjacent to <top topic>"`` line, capped at ``count``
        with one slot always reserved for the exploration line.
    """
    out = [{"query": f"{t} recent developments", "goal_derived": False}
           for t in topics[: max(count - 1, 1)]]
    for goal in goals[: max(count - len(out) - 1, 0)]:
        out.append({"query": f"{goal} events and tickets", "goal_derived": True})
    out.append({"query": f"surprising adjacent to {topics[0]}" if topics
                else "curious new things this week",
                "goal_derived": False})
    return out[:count]
