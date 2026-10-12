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
with that goal's position (``[g2] ...``), and every mark that stands
alone on the line -- wherever the model put it -- is read and stripped,
with the model's list markers and wrapping, before the query is searched. The cycle carries the row's
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
#: doubled bracket ``[[g1]]``, Markdown-escaped brackets ``\[g1\]``, the
#: "closing" form ``[/g1]`` -- because a mark that is not recognised is a
#: mark that gets searched (review of #2921, round 3, item 1; round 4, C2).
#:
#: Every repetition in it is BOUNDED (round 5, N1). ``\s*/?\s*`` made a
#: long run of spaces after a "[" cost its square -- 20,000 of them took
#: 1.9 s on the event loop -- and ``\[+`` did the same for a run of
#: brackets. No model pads a mark with more than a space or two.
_GOAL_MARK = (
    r"(?:\\?\[){1,3}\s{0,3}(?:/\s{0,3})?g\s{0,3}(-?\d+)\s{0,3}(?:\\?\]){1,3}")
_GOAL_MARK_RE = re.compile(_GOAL_MARK, re.IGNORECASE)

#: A mark as it is REMOVED from a line: with the punctuation a model
#: sets directly after it -- ``[g1]: cheap``, ``[g1]. cheap``, ``[g1],
#: cheap`` (round 5, N2). A ``.``/``,``/``;`` only when a space or the
#: end of the line follows, so ``[g1].5 inch`` keeps its dot.
_MARK_TOKEN_RE = re.compile(
    _GOAL_MARK + r"(?::|[.,;](?=\s|$))?", re.IGNORECASE)

#: Separators a removed mark leaves at the FRONT of what remains (``[g1]
#: - cheap``, ``[g1]) cheap``) and in front of a mark at the END (``cheap
#: flights - [g1]``). A dash, a bracket or a dot only when it stands by
#: itself: ``-5 C tokyo`` keeps its minus and ``...and justice`` its dots.
_LEADING_LEFTOVER_RE = re.compile(
    r"^(?:(?::|[.,;)\]\-\u2013\u2014](?=\s|$))\s*)+")
_TRAILING_LEFTOVER_RE = re.compile(
    r"(?:\s+[:,;(\[\-\u2013\u2014]|:)+$")

#: Longest line that can be a search query. A longer one is prose, is
#: dropped, and so never reaches the patterns below (N1: whatever a
#: pattern costs, it costs it on at most this much text per line).
_MAX_QUERY_LINE_LENGTH = 300

#: What may stand at the edge of a line without being part of it:
#: whitespace, a byte order mark, a zero width space.
_LINE_EDGE = " \t\r\n\ufeff\u200b\u2060"

#: Leading Markdown quote markers (``> ``, ``>> ``): the model's, never
#: the query's.
_QUOTE_MARKER_RE = re.compile(r"^(?:>\s*)+")

#: A Markdown heading line (``### Queries``) or the fence of a code block
#: (three backticks or tildes, with or without a language): not a query
#: at all. A hashtag (``#rustlang``) has no space after the hashes and is
#: kept.
_HEADING_RE = re.compile(r"^(?:#{1,6}(?:\s|$)|(?:`{3,}|~{3,})[\w+-]*$)")

#: Leading bullets: ``-``, ``*``, ``+`` or a bullet character, each
#: followed by whitespace -- so ``*.rs lint rules`` does not start with
#: one.
_BULLET_RE = re.compile(r"^(?:[-*+\u2022]\s+)+")

#: A leading list number as a model writes one: ``3.``, ``3)``, ``(3)``,
#: bold ``**3.**`` or ``**3**.`` -- followed by whitespace or ending the
#: line (so ``3.5 inch`` is not one, and neither is ``2026.``). Whether
#: it IS a list number is not a property of the line: ``2. Bundesliga
#: Spielplan 2026`` and ``1. FC K\u00f6ln transfer news`` are queries
#: (review of #2921, round 4, C3). See :func:`_is_numbered`.
_NUMBER_RE = re.compile(
    r"^(?:\(?(\d{1,3})[.)]"
    r"|(\*\*|__|\*|_)(\d{1,3})[.)]\2"
    r"|(\*\*|__|\*|_)(\d{1,3})\4[.)]"
    r")(?:\s+|$)")

#: What a model wraps a whole line in: bold, italics, quotes, code.
#: Removed from ANY line, so only as a pair with none of its characters
#: inside (:func:`_unwrap`); underscores are not in it, or ``__init__``
#: would lose them.
_WRAP_PAIRS = (
    ("**", "**"), ("*", "*"), ('"', '"'), ("'", "'"), ("`", "`"),
    ("\u201c", "\u201d"), ("\u2018", "\u2019"),
)

#: What a model wraps a MARK in, or opens in front of one: the above plus
#: underscores and parentheses. Next to a mark these are formatting.
_MARK_WRAP_PAIRS = (
    ("**", "**"), ("__", "__"), ("*", "*"), ("_", "_"), ('"', '"'),
    ("'", "'"), ("`", "`"), ("\u201c", "\u201d"), ("\u2018", "\u2019"),
    ("(", ")"),
)
_CLOSER_OF = dict(_MARK_WRAP_PAIRS)
_OPENING_WRAP_CHARS = "*_\"'`\u201c\u2018("
_WRAP_TOKEN_RE = re.compile(
    r"\*\*|__|[*_\"'`\u201c\u201d\u2018\u2019()]")

#: The pieces a line may consist of IN FRONT of its first mark and still
#: be nothing but formatting there: whitespace, a list number, a label
#: (``Query:``, ``Goal query 2:``), quote markers, bullets and wrapping.
_PREFIX_PIECE_RE = re.compile(
    r"\s+"
    r"|\(?\d{1,3}[*_]{0,2}[.)]"
    r"|[^\W\d_][^\W_]*(?: [^\W_]+){0,3} ?:"
    r"|[>\-*+\u2022_\"'`\u201c\u2018(]"
    r"|[)\u201d\u2019]"
)

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


def _unwrap(
    text: str, pairs: tuple[tuple[str, str], ...] = _WRAP_PAIRS,
) -> str:
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
        for opening, closing in pairs:
            if (len(text) > len(opening) + len(closing)
                    and text.startswith(opening) and text.endswith(closing)):
                inner = text[len(opening):len(text) - len(closing)]
                if not any(char in inner for char in opening + closing):
                    text, unwrapped = inner.strip(), True
                    break
    return text


def _peel(text: str) -> str:
    """Remove the formatting that is formatting on ANY line.

    Whitespace and a byte order mark at the edges, Markdown quote
    markers, bullets, and whole-line wrapping (:func:`_unwrap`) -- in
    whatever order a model nested them. Numbers and labels are not here:
    they are only formatting in a context (a numbered list, a mark).
    """
    for _ in range(4):
        before = text
        text = _QUOTE_MARKER_RE.sub("", text.strip(_LINE_EDGE))
        text = _BULLET_RE.sub("", text)
        text = _unwrap(text)
        if text == before:
            break
    return text


def _leading_number(line: str) -> int | None:
    """The list number a line would start with once peeled, or ``None``."""
    match = _NUMBER_RE.match(_peel(line))
    if match is None:
        return None
    return int(match.group(1) or match.group(3) or match.group(5))


def _is_numbered(lines: list[str]) -> bool:
    """Whether the response's leading numbers are list numbers.

    They are when the lines that start with a number -- behind a bullet
    or inside whole-line bold just the same -- count up from 1: each is
    1 or one more than the one before, and there are at least two. So a
    list is a list after a line of preamble, with bold numbers, and when
    it starts again at 1 for a second section (review of #2921, round 5,
    F1: round 4 wanted exactly 1..n on raw lines, and the lines of every
    other list kept their numbers -- and their marks). A single ``2.
    Bundesliga Spielplan 2026`` among plain lines, or lines numbered 2,
    2, 1, is not a list.
    """
    numbers = [
        number for number in map(_leading_number, lines) if number is not None]
    if len(numbers) < 2:
        return False
    previous = 0
    for number in numbers:
        if number != 1 and number != previous + 1:
            return False
        previous = number
    return True


def _is_subscript(text: str, start: int) -> bool:
    """Whether the bracket at ``start`` is glued to a word in front of it.

    ``arr[g1]``, ``arr_[g1]`` and ``f(x)[g1]`` are subscripts -- text. A
    mark may have wrapping in front of it (``**[g1]``, ``"[g1]``), so the
    look goes past opening wrapping to what stands before THAT.
    """
    while start > 0 and text[start - 1] in _OPENING_WRAP_CHARS:
        start -= 1
    return start > 0 and (text[start - 1].isalnum() or text[start - 1] in "])")


def _grown_over_wrapping(text: str, start: int, end: int) -> tuple[int, int]:
    """A mark's span grown over the wrapping that stands directly around it.

    ``**[g1]**``, ``"[g1]"``, ``([g1])``, ``**[g1]:**`` -- the model's
    emphasis of the mark goes with the mark, and so does a ``:`` behind
    it. Wrapping separated from the mark by a space is the query's.
    """
    grown = True
    while grown:
        grown = False
        for opening, closing in _MARK_WRAP_PAIRS:
            if (text.endswith(opening, 0, start)
                    and text.startswith(closing, end)):
                start, end = start - len(opening), end + len(closing)
                if text.startswith(":", end):
                    end += 1
                grown = True
                break
    return start, end


def _unclosed_wrappers(prefix: str) -> list[str]:
    """The closers still owed by wrapping opened in ``prefix``, in order.

    ``**2.** `` owes nothing: the bold is closed again. ``**`` owes one
    ``**``. A token closes what is open only the way Markdown closes it
    -- directly behind something, never behind a space or behind the
    opener itself -- so ``****`` and ``**1. **`` each open twice.
    """
    owed: list[str] = []
    opened_at = 0
    for match in _WRAP_TOKEN_RE.finditer(prefix):
        token = match.group(0)
        inside = prefix[opened_at:match.start()]
        if (owed and owed[-1] == token and inside.strip()
                and not inside[-1].isspace()):
            owed.pop()
        elif token in _CLOSER_OF:
            owed.append(_CLOSER_OF[token])
            opened_at = match.end()
    return owed


def _without_marks(text: str) -> tuple[str, set[int]] | None:
    """A peeled line with its goal marks removed, and the positions they name.

    ``None`` when the line carries no mark. A ``[gN]`` is a mark wherever
    it STANDS ALONE -- in front, at the end, in the middle, with or
    without the space after it -- and is text only where it is glued to
    a word in front of it (:func:`_is_subscript`). A line that has one
    mark loses every ``[gN]`` on it, glued or not (``[g1]x[g1]``).

    What stands in front of the FIRST mark is dropped with it when it is
    nothing but formatting there -- a list number, a label, a bullet,
    opening wrapping (``**2.** [g1] ...``, ``Query: [g1] ...``); wrapping
    opened there is closed at the end of the line, and that goes too.
    Anything else in front of the mark is the query's.
    """
    if all(_is_subscript(text, match.start())
           for match in _MARK_TOKEN_RE.finditer(text)):
        return None
    # On a line with a mark, underscores and parentheses around the whole
    # line are wrapping too ("__cheap flights [g1]__").
    text = _unwrap(text, _MARK_WRAP_PAIRS)
    spans: list[tuple[int, int, str]] = []
    for match in _MARK_TOKEN_RE.finditer(text):
        start, end = match.span()
        if not _is_subscript(text, start):
            start, end = _grown_over_wrapping(text, start, end)
            if spans:
                start = max(start, spans[-1][1])
        spans.append((start, end, match.group(1)))
    positions = {
        int(digits) if len(digits.lstrip("-")) <= _GOAL_POSITION_MAX_DIGITS
        else 0
        for _, _, digits in spans
    }
    prefix = text[:spans[0][0]]
    behind = [
        text[end:spans[index + 1][0] if index + 1 < len(spans) else len(text)]
        for index, (_, end, _digits) in enumerate(spans)]
    # Formatting in front of the mark -- unless nothing stands behind the
    # mark and the "label" has words in it: then it IS the query, with the
    # mark set after it ("cheap flights: [g1]").
    in_front = not _PREFIX_PIECE_RE.sub("", prefix) and (
        any(char.isalnum() for char in "".join(behind))
        or not any(char.isalpha() for char in prefix))
    pieces = behind if in_front else [prefix, *behind]
    at_end = not behind[-1].strip()
    body = " ".join(pieces).strip()
    if in_front:
        for closer in _unclosed_wrappers(prefix):
            if body.endswith(closer):
                body = body[:-len(closer)].rstrip()
        body = _LEADING_LEFTOVER_RE.sub("", body)
    if at_end:
        body = _TRAILING_LEFTOVER_RE.sub("", body)
    return body, positions


def _query_row(
    line: str, goals: list[tuple[str, str | None]], *, numbered: bool = False,
) -> dict | None:
    """One model line as a query row: its goal read, the line cleaned.

    What is the model's formatting rather than the query is removed
    before the line can reach a search provider -- and nothing else:

    * quote markers, bullets and whole-line wrapping (:func:`_peel`);
    * a leading list number when the response is a numbered list
      (``numbered``, see :func:`_is_numbered`);
    * EVERY standalone goal mark, wherever on the line it stands, with
      the wrapping directly around it -- and whatever formatting stands
      in front of the first one (:func:`_without_marks`). No string this
      returns as a query contains a mark the model wrote as one (review
      of #2921, round 5, F1).

    A heading line, a code fence, a label line (``Goal queries:``), a
    line with no letter or digit in it and a line too long to be a query
    give no row.

    Args:
        line: One non-blank response line.
        goals: The ``(text, angle)`` pairs of the SEARCHABLE goals, in the
            order the payload listed them.
        numbered: The response numbers its lines.

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
    text = _peel(line)
    if _HEADING_RE.match(text):
        return None
    number = _NUMBER_RE.match(text)
    if number and (numbered or number.end() == len(text)):
        text = _peel(text[number.end():])  # a list number, or nothing but one
    positions: set[int] = set()
    unmarked = _without_marks(text)
    if unmarked is not None:
        text, positions = unmarked
    elif text.endswith(":"):
        return None  # "Goal queries:" -- a label over the lines that follow
    text = " ".join(_unwrap(text.strip()).split())
    if not any(char.isalnum() for char in text):
        # Nothing to search for: a rule ("---"), a stray bullet, the
        # debris of a mark-only line (round 5, F2 -- round 4 searched
        # these).
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
    lines = [line for line in response_text.splitlines()
             if line.strip(_LINE_EDGE) and len(line) <= _MAX_QUERY_LINE_LENGTH]
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
