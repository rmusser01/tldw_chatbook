"""[dreams] Turning the interest-profile snapshot into search queries.

One chat call per cycle (spec §query synthesis): the prompt embeds the
decayed topics, the SEARCHABLE goals, and the region verbatim, demands at
least one adjacent/serendipity angle so the cycle cannot collapse into
"more of the same", and on ANY chat failure degrades to a deterministic
fallback built straight from the top topics and searchable goals — a dead
LLM still yields a cycle, just a less interesting one.

Goal query angles (task-33165): a goal with feedback history carries a
``query_angle`` steering note ("prefer: deal" / "avoid: event"); the
payload's ``goal_angles`` map tells the model how to steer that goal's
phrasing, and the deterministic fallback labels the goal's line with its
angle so the preview keeps showing exactly what a degraded cycle searches.

Privacy gate (spec §Privacy, binding): a goal's text — and, with it, its
angle — enters the payload or a fallback line ONLY when its ``searchable``
flag is 1; the filter runs before any payload or query string is built.

Goal provenance (review of #2921, item 1): every query row says which goal
it was written for, or ``None``. A deterministic goal line knows by
construction; the model is asked to mark each line it writes for a goal
with that goal's position (``[g2] ...``), and a mark in front of the
line or at its end is read and stripped, with the model's list markers
and whole-line quoting, before the query is searched. The cycle carries the row's
goal through the search to the story it produced, so feedback on that
story steers exactly that goal's angle. An unmarked line has unknown provenance and is
attributed to no goal -- provenance is never guessed from shared words.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable, Iterable
from itertools import zip_longest
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
    "openings) while topics drive content angles. When goal_angles are given, each "
    "entry steers that goal's query phrasing: \"prefer: deal\" leans that goal's "
    "queries toward deals and similar opportunities, \"avoid: event\" steers away "
    "from event angles. Begin every line you write for a goal with that goal's "
    "position in the goals list, in the form [g1], [g2], and so on (for example "
    "\"[g2] cheap flights tokyo april\"); a line that comes from topics or "
    "exploration carries no such mark. Queries must be "
    "self-contained for a "
    "search engine, may include the user's region verbatim when locality helps, "
    "and must never include anything except topics, goals, region, and search "
    "terms."
)


#: The mark the model puts on a line it wrote for a goal: ``[g2] cheap
#: flights tokyo april``. The number is the goal's 1-based position in the
#: payload's ``goals`` list. Read loosely -- ``[ G 2 ]``, ``[g-1]``, a
#: doubled bracket ``[[g1]]`` -- because a mark that is not recognised is
#: a mark that gets searched (review of #2921, round 3, item 1). An
#: optional ``/`` covers the model that "closes" its mark: ``[g1] cheap
#: flights [/g1]`` used to be searched as ``cheap flights [/g1]`` (round
#: 4, C2).
#:
#: WHERE it counts is narrower than round 3 had it (round 4, I1). Matched
#: anywhere, text that is no mark at all was read as one: ``arr[g1]
#: indexing in c`` became goal 1 with the query ``arr indexing in c``. A
#: mark stands alone, in FRONT of the line (after a list or quote marker)
#: or at its END -- see ``_LEADING_MARKS_RE``/``_TRAILING_MARKS_RE``.
#:
#: Every repetition in it is BOUNDED (round 5, N1). ``\s*/?\s*`` made a
#: long run of spaces after a "[" cost its square -- 20,000 of them took
#: 1.9 s on the event loop -- and ``\[+`` did the same for a run of
#: brackets. No model pads a mark with more than a space or two.
_GOAL_MARK = (
    r"\[{1,3}\s{0,3}(?:/\s{0,3})?g\s{0,3}(-?\d+)\s{0,3}\]{1,3}")
_GOAL_MARK_RE = re.compile(_GOAL_MARK, re.IGNORECASE)

#: Longest line that can be a search query. A longer one is prose, is
#: dropped, and so never reaches the patterns below (N1: whatever a
#: pattern costs, it costs it on at most this much text per line).
_MAX_QUERY_LINE_LENGTH = 300

#: Leading Markdown quote markers (``> ``, ``>> ``): the model's, never
#: the query's.
_QUOTE_MARKER_RE = re.compile(r"^(?:>\s*)+")

#: A Markdown heading line (``### Queries``): not a query at all. A
#: hashtag (``#rustlang``) has no space after the hashes and is kept.
_HEADING_RE = re.compile(r"^#{1,6}(?:\s|$)")

#: A leading bullet: ``-``, ``*`` or a bullet character, followed by
#: whitespace -- so ``*.rs lint rules`` does not start with one.
_BULLET_RE = re.compile(r"^[-*\u2022]\s+")

#: A leading ``3.`` or ``3)``, followed by whitespace or ending the line
#: (so ``3.5 inch`` is not one). Whether it is a LIST NUMBER is not a
#: property of the line: ``2. Bundesliga Spielplan 2026`` and ``1. FC
#: K\u00f6ln transfer news`` are queries (review of #2921, round 4, C3).
#: See :func:`_is_numbered`.
_NUMBER_RE = re.compile(r"^(\d+)[.)](?:\s+|$)")

#: What a model wraps a whole line (or a mark) in: bold, quotes, code.
_WRAP_PAIRS = (
    ("**", "**"), ("*", "*"), ('"', '"'), ("'", "'"), ("`", "`"),
    ("\u201c", "\u201d"), ("\u2018", "\u2019"),
)

#: A goal mark together with the wrapping that stands DIRECTLY around it
#: (``**[g1]**``, ``"[g1]"``): the model's emphasis of the mark goes with
#: the mark. Wrapping separated from the mark by a space is the query's.
_WRAPPED_MARK = "(?:" + "|".join(
    [re.escape(opening) + _GOAL_MARK + re.escape(closing)
     for opening, closing in _WRAP_PAIRS] + [_GOAL_MARK]) + ")"

#: One or more marks in front of the line, and one or more at its end
#: (set off from the text by whitespace, so ``index into arr[g1]`` has
#: none).
_LEADING_MARKS_RE = re.compile(
    rf"^(?:{_WRAPPED_MARK}\s*)+", re.IGNORECASE)
_TRAILING_MARKS_RE = re.compile(
    rf"(?:^|(?<=\s))(?:{_WRAPPED_MARK}\s*)+$", re.IGNORECASE)

#: Longest digit run read as a goal position; anything longer names no goal.
_GOAL_POSITION_MAX_DIGITS = 9


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


def _angle_of(goal: dict) -> str | None:
    """A goal row's non-blank ``query_angle``, or ``None`` when unset."""
    angle = str(goal.get("query_angle") or "").strip()
    return angle or None


def _goal_line(text: str, angle: str | None) -> str:
    """A goal's fallback line, labeled with its steering angle when set."""
    query = f"{text} events and tickets"
    if angle:
        query += f" (angled: {angle})"
    return query


def _unwrap(text: str) -> str:
    """Remove quotes, backticks or asterisks that wrap the WHOLE text.

    Only as a matching pair with none of its characters inside: round 3
    stripped these characters from both ENDS of every line, which took
    the opening quote off ``"Wednesday 13" tour dates``, the closing one
    off ``tutorial for "rust tui"`` and the asterisk off ``learn C*``
    (review of #2921, round 4, C3).
    """
    unwrapped = True
    while unwrapped:
        unwrapped = False
        for opening, closing in _WRAP_PAIRS:
            if (len(text) > len(opening) + len(closing)
                    and text.startswith(opening) and text.endswith(closing)):
                inner = text[len(opening):len(text) - len(closing)]
                if not any(char in inner for char in opening + closing):
                    text, unwrapped = inner.strip(), True
                    break
    return text


def _is_numbered(lines: list[str]) -> bool:
    """Whether the response's leading numbers are list numbers.

    They are when the lines that start with ``N.``/``N)`` are numbered by
    their position among themselves: 1, 2, 3, ... in order, at least two
    of them. A single ``2. Bundesliga Spielplan 2026`` among plain lines,
    or three lines numbered 2, 2, 1, is not a list; a list after one line
    of preamble still is.
    """
    numbers = [
        int(match.group(1)) for match in map(_NUMBER_RE.match, lines) if match]
    return len(numbers) >= 2 and numbers == list(range(1, len(numbers) + 1))


def _query_row(
    line: str, goals: list[tuple[str, str | None]], *, numbered: bool = False,
) -> dict | None:
    """One model line as a query row: its goal read, the line cleaned.

    What is the model's formatting rather than the query is removed
    before the line can reach a search provider -- and nothing else:

    * one leading bullet (``- ``, ``* ``);
    * a leading ``N.``/``N)`` when the response is a numbered list
      (``numbered``, see :func:`_is_numbered`);
    * leading Markdown quote markers (``> ``);
    * quotes, backticks or asterisks that wrap the whole line as a pair
      (:func:`_unwrap`), before and after the marks go;
    * the goal marks in front of the line and at its end, with wrapping
      that stands directly around them, and a ``:`` or dash a removed
      leading mark leaves in front. A ``[g1]`` inside a word or in the
      middle of the line is text and stays (round 4, I1).

    A Markdown heading line is not a query and gives no row.

    Args:
        line: One non-blank response line.
        goals: The ``(text, angle)`` pairs of the SEARCHABLE goals, in the
            order the payload listed them.
        numbered: The response numbers its lines by position.

    Returns:
        ``{"query", "goal_derived", "goal"}``, or ``None`` when nothing is
        left once the line is cleaned. ``goal`` is set only when the line's
        marks name exactly ONE position and that position is a goal the
        model was given; no mark, an unknown position, or marks naming two
        different positions attribute the line to no goal -- provenance is
        never guessed.
    """
    if len(line) > _MAX_QUERY_LINE_LENGTH:
        return None
    text = _QUOTE_MARKER_RE.sub("", line.strip())
    if _HEADING_RE.match(text):
        return None
    text = _BULLET_RE.sub("", text, count=1)
    number = _NUMBER_RE.match(text)
    if number and (numbered or number.end() == len(text)):
        text = text[number.end():]  # a list number, or nothing but one
    text = _unwrap(text.strip())
    marks = ""
    leading = _LEADING_MARKS_RE.match(text)
    if leading:
        marks += leading.group(0)
        text = re.sub(r"^[:\-\u2013\u2014]\s*", "", text[leading.end():])
    trailing = _TRAILING_MARKS_RE.search(text)
    if trailing:
        marks += trailing.group(0)
        text = text[:trailing.start()]
    positions = {
        int(digits) if len(digits) <= _GOAL_POSITION_MAX_DIGITS else 0
        for digits in _GOAL_MARK_RE.findall(marks)
    }
    text = " ".join(_unwrap(text.strip()).split())
    if not text:
        return None
    goal: str | None = None
    if len(positions) == 1:
        (position,) = positions
        if 1 <= position <= len(goals):
            goal = goals[position - 1][0]
    return {"query": text, "goal_derived": goal is not None, "goal": goal}


def _response_rows(
    response_text: str, goals: list[tuple[str, str | None]],
) -> list[dict]:
    """Every usable line of one model response, as query rows."""
    lines = [_QUOTE_MARKER_RE.sub("", line.strip())
             for line in response_text.splitlines()
             if line.strip() and len(line) <= _MAX_QUERY_LINE_LENGTH]
    numbered = _is_numbered(lines)
    rows = (_query_row(line, goals, numbered=numbered) for line in lines)
    return [row for row in rows if row is not None]


def _distinct(rows: Iterable[dict]) -> list[dict]:
    """Rows whose query was not already listed, compared case-insensitively.

    The FIRST spelling wins (review of #2921, round 3, N1). Two rows with
    the same query are one search: keeping both spent a second search on
    the same text and, since a story's goal is looked up through the query
    that found it, let a goal-marked duplicate attribute the unmarked
    line's stories to the goal as well.
    """
    seen: set[str] = set()
    distinct: list[dict] = []
    for row in rows:
        key = str(row["query"]).casefold()
        if key not in seen:
            seen.add(key)
            distinct.append(row)
    return distinct


async def synthesize_query_rows(
    chat: Callable[..., Any],
    *,
    snapshot: dict,
    count: int,
    exploration_slots: int,
) -> list[dict]:
    """Turn one interest snapshot into ``count`` search queries, with goals.

    Exactly one chat call is made. The user payload embeds the topic
    texts, the SEARCHABLE goal texts, the searchable goals' query angles
    (``goal_angles``, task-33165 -- feedback steering, never weight), and
    the region verbatim; the system prompt demands at least
    ``exploration_slots`` adjacent/serendipity lines and asks for a goal
    mark on every line written for a goal. Any chat failure — exception,
    empty or unusably short response — falls back to a deterministic list
    built directly from the top topics and searchable goals, so the cycle
    degrades instead of dying.

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
        Up to ``count`` rows ``{"query": str, "goal_derived": bool,
        "goal": str | None}``: the query to search, and the stored text of
        the goal it was written for (``None`` for topic, exploration, and
        unmarked lines).
    """
    topics = [t["text"] for t in snapshot.get("topics", [])]
    # PRIVACY GATE (spec §Privacy, binding): a goal's text -- and its
    # angle, which names the user's reaction to that text -- may leave the
    # machine only when its ``searchable`` flag is 1. The filter runs HERE,
    # before the payload exists, so an unsearchable goal's text and angle
    # appear nowhere in anything sent to the model -- and it has no
    # position a line could be marked with.
    searchable = [g for g in snapshot.get("goals", [])
                  if int(g.get("searchable", 1) or 0)]
    goal_pairs = [(str(g["text"]), _angle_of(g)) for g in searchable]
    goal_angles = {text: angle for text, angle in goal_pairs if angle}
    region = (snapshot.get("region") or "").strip()
    user = json.dumps(
        {"topics": topics, "goals": [text for text, _ in goal_pairs],
         "goal_angles": goal_angles, "region": region, "count": count},
        ensure_ascii=False,
    )
    try:
        resp = await _invoke(
            chat, system=SYSTEM_PROMPT.format(explore=exploration_slots), user=user
        )
        rows = _distinct(_response_rows(
            extract_response_content(resp), goal_pairs))[:count]
        if len(rows) >= max(1, min(count, 2)):
            return rows
        logger.debug("Dreams query synthesis returned {} usable lines; falling back", len(rows))
    except Exception as exc:  # noqa: BLE001 - degradation, not a crash path
        logger.warning("Dreams query synthesis chat call failed; using fallback: {}",
                       type(exc).__name__)
    return _fallback_queries(topics, goal_pairs, count)


async def synthesize_queries(
    chat: Callable[..., Any],
    *,
    snapshot: dict,
    count: int,
    exploration_slots: int,
) -> list[str]:
    """Turn one interest snapshot into ``count`` search query strings.

    The string-only view of :func:`synthesize_query_rows` (same single
    chat call, same fallback): the queries exactly as they are searched,
    goal marks already stripped.

    Args:
        chat: ``chat_api_call``-shaped callable.
        snapshot: ``interest_profile.snapshot`` result.
        count: Number of queries to return.
        exploration_slots: How many lines must explore adjacent ground.

    Returns:
        Up to ``count`` query strings.
    """
    rows = await synthesize_query_rows(
        chat, snapshot=snapshot, count=count,
        exploration_slots=exploration_slots)
    return [row["query"] for row in rows]


def preview_queries(
    topics: list[str], goals: list[dict], *, count: int
) -> list[dict]:
    """Public preview seam over the fallback list.

    Two consumers: the Artifacts story-detail modal renders a "what we'll
    look for" preview of the NEXT cycle's queries without ever calling an
    LLM from the UI (so it shows exactly what a degraded cycle would
    search), and the cycle's LLM-exhausted fallback searches this same
    deterministic list verbatim when the daily budget rules out a
    synthesis call. The fallback is exposed publicly so neither consumer
    imports the private helper (controller-authorized wrapper, Task 7).
    Rows carry ``goal_derived`` so the preview can label which lines came
    from goals, and a goal with a query angle gets its line labeled
    ``(angled: ...)`` (task-33165); the same privacy gate as synthesis
    applies — unsearchable goals are filtered here and never contribute a
    line or an angle.

    Args:
        topics: Current snapshot topic texts, heaviest first.
        goals: The snapshot's goal rows (dicts carrying ``text``,
            ``searchable``, and optionally ``query_angle``); searchable
            filtering happens inside.
        count: The cycle's query budget (``queries_per_cycle``).

    Returns:
        ``{"query": str, "goal_derived": bool, "goal": str | None}`` rows,
        the same list ``_fallback_queries`` builds.
    """
    searchable = [(str(g["text"]), _angle_of(g)) for g in goals
                  if int(g.get("searchable", 1) or 0)]
    return _fallback_queries(topics, searchable, count)


def _fallback_queries(
    topics: list[str], goals: list[tuple[str, str | None]], count: int
) -> list[dict]:
    """Deterministic degraded-cycle queries from the top topics and goals.

    Qodo #14 (PR #2890): topic and goal lines INTERLEAVE (topic, goal,
    topic, goal, ...) instead of topics filling every slot first. With the
    default three-query budget and two or more topics, the old
    topics-first order starved goals out of the fallback entirely -- an
    LLM failure then produced no goal-derived search or preview at all
    despite searchable goals being present. Round-robin puts the first
    searchable goal in slot 2 whenever one exists, at every count. A goal
    carrying a query angle gets its line labeled with it (task-33165), so
    the fallback the cycle would actually search stays identical to what
    the preview shows.

    Args:
        topics: Snapshot topic texts, heaviest first.
        goals: ``(text, angle_or_None)`` pairs, SEARCHABLE goals only
            (callers filter; this helper is the privacy gate's innermost
            trust boundary).
        count: Number of queries to return.

    Returns:
        ``{"query": str, "goal_derived": bool, "goal": str | None}`` rows
        -- alternating ``"<topic> recent developments"`` / ``"<goal>
        events and tickets"`` lines (an angled goal's line suffixed
        ``(angled: <note>)``; whichever pool runs out first drains the
        other into the remaining slots), plus one ``"surprising adjacent
        to <top topic>"`` line, capped at ``count`` with one slot always
        reserved for the exploration line. A goal line's ``goal`` is the
        text it was built from (its provenance, exact by construction);
        every other line's is ``None``.
    """
    interleaved: list[dict] = []
    for topic, goal in zip_longest(
        ({"query": f"{t} recent developments", "goal_derived": False,
          "goal": None}
         for t in topics),
        ({"query": _goal_line(text, angle), "goal_derived": True,
          "goal": text}
         for text, angle in goals),
    ):
        if topic is not None:
            interleaved.append(topic)
        if goal is not None:
            interleaved.append(goal)
    out = _distinct(interleaved)[: max(count - 1, 1)]
    out.append({"query": f"surprising adjacent to {topics[0]}" if topics
                else "curious new things this week",
                "goal_derived": False, "goal": None})
    return out[:count]
