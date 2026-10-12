# Tests/Dreams/test_query_synthesis.py
"""Query synthesis: exactly one chat call per cycle, deterministic fallback.

Ruling R1: ``synthesize_queries`` is ``async``; the sync fake chat callables
are offloaded to a thread by the module (the ``briefing_service._invoke_chat``
discipline).
"""
import json

import pytest

from tldw_chatbook.Dreams.query_synthesis import (
    preview_queries,
    synthesize_queries,
    synthesize_query_rows,
)

SNAP = {"topics": [{"facet": "topic", "text": "rust tui", "weight": 0.9},
                   {"facet": "topic", "text": "jazz guitar", "weight": 0.6}],
        "region": "Seattle"}

SNAP_WITH_GOALS = {
    "topics": [{"facet": "topic", "text": "rust tui", "weight": 0.9}],
    "goals": [
        {"facet": "goal", "text": "see Wednesday 13 live", "searchable": 1},
        {"facet": "goal", "text": "private wish", "searchable": 0},
    ],
    "region": "Seattle",
}


def _capture_chat(response_text):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": response_text}}]}

    return chat, calls


@pytest.mark.asyncio
async def test_synthesis_makes_exactly_one_call_and_parses_lines():
    chat, calls = _capture_chat(
        "rust tui new releases 2026\njazz guitar concerts Seattle\n"
        "random curiosity: deep sea cables"
    )
    out = await synthesize_queries(chat, snapshot=SNAP, count=3, exploration_slots=1)
    assert len(calls) == 1
    assert len(out) == 3 and "deep sea cables" in out[2]
    assert "Seattle" in calls[0]["messages_payload"][0]["content"]


@pytest.mark.asyncio
async def test_synthesis_falls_back_deterministically_when_chat_fails():
    def boom(**kwargs):
        raise RuntimeError("provider down")

    out = await synthesize_queries(boom, snapshot=SNAP, count=3, exploration_slots=1)
    assert out[0] == "rust tui recent developments"
    assert any("adjacent" in q for q in out)


@pytest.mark.asyncio
async def test_synthesis_never_leaks_unsearchable_or_regionless_junk():
    chat, calls = _capture_chat("a\nb\nc")
    await synthesize_queries(chat, snapshot=SNAP, count=3, exploration_slots=0)
    system = calls[0]["system_message"]
    assert "queries" in system.lower()


# --- Goals enter synthesis (Phase 2 Task 1) ----------------------------------


@pytest.mark.asyncio
async def test_synthesis_payload_carries_searchable_goals_and_never_private_ones():
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "rust tui news\nconcerts Seattle"}}]}

    await synthesize_queries(chat, snapshot=SNAP_WITH_GOALS, count=3, exploration_slots=1)
    payload = calls[0]["messages_payload"][0]["content"]
    assert "see Wednesday 13 live" in payload
    assert "private wish" not in payload  # searchable=0 never leaves the machine


def test_preview_queries_labels_goal_derived_lines_and_skips_private_goals():
    rows = preview_queries(["rust tui"], SNAP_WITH_GOALS["goals"], count=4)
    queries = [r["query"] for r in rows]
    assert any(r["goal_derived"] for r in rows)
    assert not any("private wish" in q for q in queries)
    assert any("rust tui" in q for q in queries)


@pytest.mark.asyncio
async def test_synthesis_fallback_derives_goal_lines_and_never_private_ones():
    def boom(**kwargs):
        raise RuntimeError("provider down")

    out = await synthesize_queries(
        boom, snapshot=SNAP_WITH_GOALS, count=3, exploration_slots=1
    )
    assert "see Wednesday 13 live events and tickets" in out
    assert not any("private wish" in q for q in out), (
        "a searchable=0 goal must never reach even the fallback queries"
    )
    assert out[0] == "rust tui recent developments"


# --- Qodo #14 (PR #2890): goals must survive the fallback at default counts ---


@pytest.mark.asyncio
async def test_fallback_interleaves_goals_at_default_count():
    """Two topics, one goal, count 3: the fallback must carry the goal.

    Topics-first filled both non-exploration slots with topics, so an LLM
    failure produced no goal-derived query at all despite a searchable
    goal being present. Round-robin interleaving puts the first goal in
    slot 2 whenever one exists.
    """
    def boom(**kwargs):
        raise RuntimeError("provider down")

    snapshot = {
        "topics": [
            {"facet": "topic", "text": "rust tui", "weight": 0.9},
            {"facet": "topic", "text": "jazz guitar", "weight": 0.8},
        ],
        "goals": [
            {"facet": "goal", "text": "see Wednesday 13 live", "searchable": 1},
        ],
        "region": "Seattle",
    }
    out = await synthesize_queries(
        boom, snapshot=snapshot, count=3, exploration_slots=1
    )

    assert "see Wednesday 13 live events and tickets" in out, (
        "a searchable goal must appear at the default three-query budget"
    )
    assert out[0] == "rust tui recent developments", "topics still lead"
    assert any("adjacent" in q for q in out), "the exploration slot stays"


def test_preview_queries_interleaves_and_preserves_goal_labels():
    rows = preview_queries(
        ["rust tui", "jazz guitar", "vinyl collecting"],
        [{"facet": "goal", "text": "gig goal", "searchable": 1},
         {"facet": "goal", "text": "private wish", "searchable": 0}],
        count=5,
    )
    queries = [r["query"] for r in rows]
    # Alternation: topic, goal, topic, exploration (5th slot beyond count).
    assert queries[:2] == ["rust tui recent developments",
                           "gig goal events and tickets"]
    assert "jazz guitar recent developments" in queries
    assert not any("private wish" in q for q in queries)
    assert rows[-1]["goal_derived"] is False, "the exploration line closes"
    assert sum(1 for r in rows if r["goal_derived"]) == 1


def test_fallback_without_goals_matches_the_old_order():
    """No goals: pure topic lines plus exploration, unchanged behavior."""
    rows = preview_queries(["rust tui", "jazz guitar"], [], count=3)
    assert [r["query"] for r in rows] == [
        "rust tui recent developments",
        "jazz guitar recent developments",
        "surprising adjacent to rust tui",
    ]
    assert not any(r["goal_derived"] for r in rows)


# --- Goal query angles steer synthesis (task-33165) ---------------------------
#
# A goal with feedback history carries a query_angle ("prefer: deal" /
# "avoid: event"); synthesis consults it for SEARCHABLE goals only -- the
# privacy gate stays the single egress point.

SNAP_WITH_ANGLES = {
    "topics": [{"facet": "topic", "text": "rust tui", "weight": 0.9}],
    "goals": [
        {"facet": "goal", "text": "see Wednesday 13 live", "searchable": 1,
         "query_angle": "prefer: deal"},
        {"facet": "goal", "text": "private wish", "searchable": 0,
         "query_angle": "avoid: event"},
        {"facet": "goal", "text": "unangled goal", "searchable": 1},
    ],
    "region": "Seattle",
}


@pytest.mark.asyncio
async def test_synthesis_payload_carries_goal_angles_for_searchable_goals_only():
    chat, calls = _capture_chat("rust tui news\nconcerts Seattle")
    await synthesize_queries(
        chat, snapshot=SNAP_WITH_ANGLES, count=3, exploration_slots=1
    )
    payload = json.loads(calls[0]["messages_payload"][0]["content"])
    assert payload["goal_angles"] == {"see Wednesday 13 live": "prefer: deal"}
    # An unsearchable goal's ANGLE never leaves the machine either -- the
    # same single filter point that keeps its text out.
    assert "private wish" not in json.dumps(payload)
    assert "avoid: event" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_synthesis_system_prompt_tells_the_model_angles_steer():
    chat, calls = _capture_chat("a\nb\nc")
    await synthesize_queries(
        chat, snapshot=SNAP_WITH_ANGLES, count=3, exploration_slots=1
    )
    system = calls[0]["system_message"]
    assert "goal_angles" in system
    assert "prefer" in system and "avoid" in system


@pytest.mark.asyncio
async def test_synthesis_fallback_labels_angled_goal_lines():
    """Degraded cycles keep the preview truthful: angles label the lines."""
    def boom(**kwargs):
        raise RuntimeError("provider down")

    out = await synthesize_queries(
        boom, snapshot=SNAP_WITH_ANGLES, count=3, exploration_slots=1
    )
    assert "see Wednesday 13 live events and tickets (angled: prefer: deal)" \
        in out
    assert not any("private wish" in q or "avoid: event" in q for q in out), (
        "an unsearchable goal's text AND angle stay machine-local"
    )


def test_preview_queries_labels_angled_goal_lines_and_hides_private_angles():
    rows = preview_queries(["rust tui"], SNAP_WITH_ANGLES["goals"], count=4)
    queries = [r["query"] for r in rows]
    assert "see Wednesday 13 live events and tickets (angled: prefer: deal)" \
        in queries
    assert "unangled goal events and tickets" in queries, (
        "a goal without an angle keeps its plain line"
    )
    assert not any("(angled" in q and "unangled goal" in q for q in queries)
    assert not any("private wish" in q or "avoid: event" in q for q in queries)


# --- Goal provenance on synthesized queries (review of #2921, item 1) ----------
#
# A story is attributed to the goal whose query surfaced it, so each query
# has to say which goal (if any) it was written for. The deterministic
# lines know by construction; the model is asked to tag its goal lines.

SNAP_TWO_GOALS = {
    "topics": [{"facet": "topic", "text": "rust tui", "weight": 0.9}],
    "goals": [
        {"facet": "goal", "text": "private wish", "searchable": 0},
        {"facet": "goal", "text": "see Wednesday 13 live", "searchable": 1},
        {"facet": "goal", "text": "learn to play the cello", "searchable": 1},
    ],
    "region": "Seattle",
}


@pytest.mark.asyncio
async def test_synthesis_rows_carry_the_goal_each_tagged_line_was_written_for():
    chat, calls = _capture_chat(
        "rust tui new releases\n"
        "[g1] wednesday 13 tour dates seattle\n"
        "[G2]  beginner cello lessons\n"
        "surprising deep sea cables"
    )
    rows = await synthesize_query_rows(
        chat, snapshot=SNAP_TWO_GOALS, count=4, exploration_slots=1)

    assert rows == [
        {"query": "rust tui new releases", "goal_derived": False,
         "goal": None},
        {"query": "wednesday 13 tour dates seattle", "goal_derived": True,
         "goal": "see Wednesday 13 live"},
        {"query": "beginner cello lessons", "goal_derived": True,
         "goal": "learn to play the cello"},
        {"query": "surprising deep sea cables", "goal_derived": False,
         "goal": None},
    ]
    # Positions count the SEARCHABLE goals the model was given: the private
    # goal ahead of them in the profile has no number to be tagged with.
    payload = json.loads(calls[0]["messages_payload"][0]["content"])
    assert payload["goals"] == [
        "see Wednesday 13 live", "learn to play the cello"]
    assert "[g1]" in calls[0]["system_message"], (
        "the model is told how to mark a goal's line"
    )


@pytest.mark.asyncio
async def test_synthesis_rows_attribute_nothing_to_an_unknown_goal_position():
    chat, _calls = _capture_chat(
        "[g7] something else entirely\nrust tui news\n[g0] zero is no goal")
    rows = await synthesize_query_rows(
        chat, snapshot=SNAP_TWO_GOALS, count=3, exploration_slots=1)

    assert [row["goal"] for row in rows] == [None, None, None]
    assert [row["query"] for row in rows] == [
        "something else entirely", "rust tui news", "zero is no goal"], (
        "the tag is stripped either way: it never reaches a search engine"
    )


@pytest.mark.asyncio
async def test_synthesize_queries_returns_the_untagged_query_strings():
    chat, _calls = _capture_chat(
        "rust tui news\n[g1] wednesday 13 tour dates\nsurprising cables")
    out = await synthesize_queries(
        chat, snapshot=SNAP_TWO_GOALS, count=3, exploration_slots=1)

    assert out == [
        "rust tui news", "wednesday 13 tour dates", "surprising cables"]


def test_fallback_rows_name_the_goal_their_line_was_built_from():
    rows = preview_queries(["rust tui"], SNAP_TWO_GOALS["goals"], count=4)

    assert {row["goal"] for row in rows if row["goal_derived"]} == {
        "see Wednesday 13 live", "learn to play the cello"}
    assert all(row["goal"] is None for row in rows if not row["goal_derived"])
    assert not any("private wish" in row["query"] for row in rows)


# --- The goal mark, wherever the model puts it (review round 3, item 1) ---------
#
# The mark was only recognised as the very first characters of a line. A
# model that numbers its lines, bullets them, quotes them (exactly as the
# system prompt's own example is quoted) or bolds the mark sent the mark --
# and the list marker -- to the search provider, and the story lost its goal.

_G1, _G2 = "see Wednesday 13 live", "learn to play the cello"


@pytest.mark.parametrize(
    "line, query, goal",
    [
        ("[g1] tour dates seattle", "tour dates seattle", _G1),
        ("- [g1] tour dates seattle", "tour dates seattle", _G1),
        ("* [g2] cello lessons", "cello lessons", _G2),
        ("\u2022 [g2] cello lessons", "cello lessons", _G2),
        ('"[g2] cello lessons april"', "cello lessons april", _G2),
        ("\u201c[g2] cello lessons\u201d", "cello lessons", _G2),
        ("**[g1]** tour dates", "tour dates", _G1),
        ("- **[g1]** tour dates", "tour dates", _G1),
        ("tour dates seattle [g1]", "tour dates seattle", _G1),
        ("[g 1] tour dates", "tour dates", _G1),
        ("[ G2 ] cello lessons", "cello lessons", _G2),
        ("[g1]: tour dates", "tour dates", _G1),
        ("[g1] - tour dates", "tour dates", _G1),
        # The same goal marked twice is still one goal.
        ("[g1] tour dates [g1]", "tour dates", _G1),
        # Marks that name no goal the model was given: stripped, no goal.
        ("[g-1] tour dates", "tour dates", None),
        ("[g0] tour dates", "tour dates", None),
        ("[g3] tour dates", "tour dates", None),
        ("[g99999999999999999999999999] tour dates", "tour dates", None),
        # Two different goals on one line: every mark goes, nothing is
        # attributed -- provenance is never guessed.
        ("[g1][g2] cello tour bars", "cello tour bars", None),
        ("[g1] cello tour bars [g2]", "cello tour bars", None),
        ("[g1][g9] tour dates", "tour dates", None),
        # A decimal is not a list marker.
        ("3.5 inch floppy drives", "3.5 inch floppy drives", None),
        # Round 4, C2: a model that "closes" its mark. The closing form was
        # searched ("cheap flights [/g1]") and stored as the story's query.
        ("[g1] tour dates seattle [/g1]", "tour dates seattle", _G1),
        ("[g2] cello lessons [ / g2 ]", "cello lessons", _G2),
        ("tour dates seattle [/g1]", "tour dates seattle", _G1),
        ("[g1] cello tour bars [/g2]", "cello tour bars", None),
    ],
)
def test_goal_mark_is_read_and_removed_wherever_it_is(line, query, goal):
    from tldw_chatbook.Dreams.query_synthesis import _query_row

    goals = [(_G1, None), (_G2, None)]
    assert _query_row(line, goals) == {
        "query": query, "goal_derived": goal is not None, "goal": goal}


# --- Round 4, I1: a mark is a mark only where a model puts one ------------------
#
# Round 3 matched the mark anywhere on the line. That reads text that is not
# a mark at all as one: ``arr[g1] indexing in c`` became goal 1 with the
# query ``arr indexing in c``. A mark stands alone, in front of the line
# (after a list or quote marker) or at its end.


@pytest.mark.parametrize(
    "line, query, goal",
    [
        # Inside a word, or in the middle of the line: text, not a mark.
        ("arr[g1] indexing in c", "arr[g1] indexing in c", None),
        ("index into arr[g1]", "index into arr[g1]", None),
        ("tour dates [g1] seattle", "tour dates [g1] seattle", None),
        # A mark the model bracketed twice is still its mark -- and no
        # "[ ]" is left behind.
        ("[[g1]] osaka food", "osaka food", _G1),
        ("osaka food [[g1]]", "osaka food", _G1),
        # No space after the mark.
        ("[g1]tour dates", "tour dates", _G1),
        # Markdown quote markers in front are the model's formatting.
        ("> [g1] tour dates", "tour dates", _G1),
        (">> - [g2] cello lessons", "cello lessons", _G2),
        ("> rust tui news", "rust tui news", None),
        # A hashtag is not a heading.
        ("#rustlang news", "#rustlang news", None),
        ("C# tutorials", "C# tutorials", None),
    ],
)
def test_a_mark_counts_only_in_front_of_or_at_the_end_of_a_line(
        line, query, goal):
    from tldw_chatbook.Dreams.query_synthesis import _query_row

    assert _query_row(line, [(_G1, None), (_G2, None)]) == {
        "query": query, "goal_derived": goal is not None, "goal": goal}


@pytest.mark.parametrize(
    "line", ["### Queries", "# [g1] tour dates", "## 1. Topics", "#"])
def test_a_heading_line_is_not_a_query(line):
    from tldw_chatbook.Dreams.query_synthesis import _query_row

    assert _query_row(line, [(_G1, None), (_G2, None)]) is None


@pytest.mark.asyncio
async def test_synthesis_reads_a_quoted_numbered_response_under_a_heading():
    chat, _calls = _capture_chat(
        "### Queries\n"
        "> 1. [g1] wednesday 13 tour dates\n"
        "> 2. beginner cello lessons [/g2]\n"
        "> 3. arr[g1] indexing in c"
    )
    rows = await synthesize_query_rows(
        chat, snapshot=SNAP_TWO_GOALS, count=5, exploration_slots=1)

    assert [(row["query"], row["goal"]) for row in rows] == [
        ("wednesday 13 tour dates", _G1),
        ("beginner cello lessons", _G2),
        ("arr[g1] indexing in c", None),
    ]


# --- Round 4, C3: cleaning a line must not damage the query ---------------------
#
# Round 3 stripped quotes, backticks and asterisks from both ENDS of every
# line and a leading "N." from every line. That took the opening quote off
# ``"Wednesday 13" tour dates``, the closing one off ``tutorial for "rust
# tui"``, the asterisk off ``learn C*``, and the ordinal off ``2. Bundesliga``.


@pytest.mark.parametrize(
    "line",
    [
        '"Wednesday 13" tour dates 2026',
        'tutorial for "rust tui"',
        '"A" tour "B"',
        "2. Bundesliga Spielplan 2026",
        "2. Weltkrieg Dokumentation",
        "1. FC K\u00f6ln transfer news",
        "learn C*",
        "*.rs lint rules",
        "rock 'n' roll history",
        "what does `ls -la` print",
    ],
)
def test_a_query_is_not_damaged_by_cleaning(line):
    from tldw_chatbook.Dreams.query_synthesis import _query_row

    assert _query_row(line, [(_G1, None), (_G2, None)]) == {
        "query": line, "goal_derived": False, "goal": None}


@pytest.mark.parametrize(
    "line, query, goal",
    [
        # Wrapping is removed only as a matching pair around the WHOLE line...
        ('"cheap flights tokyo"', "cheap flights tokyo", None),
        ("**cheap flights tokyo**", "cheap flights tokyo", None),
        ("`cheap flights tokyo`", "cheap flights tokyo", None),
        ("\u201ccheap flights tokyo\u201d", "cheap flights tokyo", None),
        ('- "cheap flights tokyo"', "cheap flights tokyo", None),
        # ...or directly around a removed mark.
        ("**[g1]** tour dates", "tour dates", _G1),
        ("`[g2]` cello lessons", "cello lessons", _G2),
        # The mark goes; the phrase the model quoted stays quoted.
        ('[g1] "Wednesday 13" tour dates', '"Wednesday 13" tour dates', _G1),
        ('tutorial for "rust tui" [g2]', 'tutorial for "rust tui"', _G2),
    ],
)
def test_wrapping_is_removed_only_as_a_pair(line, query, goal):
    from tldw_chatbook.Dreams.query_synthesis import _query_row

    assert _query_row(line, [(_G1, None), (_G2, None)]) == {
        "query": query, "goal_derived": goal is not None, "goal": goal}


@pytest.mark.parametrize(
    "line, query, goal",
    [
        ("1. rust tui news", "rust tui news", None),
        ("2. [g1] tour dates seattle", "tour dates seattle", _G1),
        ("3) [g2] cello lessons", "cello lessons", _G2),
        ('2. "[g1] tour dates seattle"', "tour dates seattle", _G1),
        ("rust tui news", "rust tui news", None),
    ],
)
def test_a_number_is_a_list_marker_only_in_a_numbered_response(
        line, query, goal):
    from tldw_chatbook.Dreams.query_synthesis import _query_row

    assert _query_row(line, [(_G1, None), (_G2, None)], numbered=True) == {
        "query": query, "goal_derived": goal is not None, "goal": goal}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, queries",
    [
        # Numbers that are NOT the lines' positions are part of the queries.
        ("2. Bundesliga Spielplan 2026\n2. Weltkrieg Dokumentation\n"
         "1. FC K\u00f6ln transfer news",
         ["2. Bundesliga Spielplan 2026", "2. Weltkrieg Dokumentation",
          "1. FC K\u00f6ln transfer news"]),
        # One numbered-looking line among plain ones is a query.
        ("rust tui news\n2. Bundesliga Spielplan 2026\ncello lessons",
         ["rust tui news", "2. Bundesliga Spielplan 2026", "cello lessons"]),
        ("1. FC K\u00f6ln transfer news\nrust tui news",
         ["1. FC K\u00f6ln transfer news", "rust tui news"]),
        # A list numbered 1..n by position is a list -- with or without a
        # line of preamble in front of it.
        ("1. rust tui news\n2. cello lessons\n3) tour dates",
         ["rust tui news", "cello lessons", "tour dates"]),
        ("Here are the queries:\n1. rust tui news\n2. cello lessons",
         ["Here are the queries:", "rust tui news", "cello lessons"]),
        # ...and inside one, an ordinal that belongs to the query survives.
        ("1. 2. Bundesliga Spielplan 2026\n2. rust tui news",
         ["2. Bundesliga Spielplan 2026", "rust tui news"]),
    ],
)
async def test_numbering_is_judged_over_the_whole_response(response, queries):
    chat, _calls = _capture_chat(response)

    rows = await synthesize_query_rows(
        chat, snapshot=SNAP_TWO_GOALS, count=5, exploration_slots=1)

    assert [row["query"] for row in rows] == queries


@pytest.mark.parametrize("line", ["[g1]", "- [g2]", "**[g1]**", '"[g1]":', "1."])
def test_a_line_that_is_empty_once_cleaned_is_dropped(line):
    from tldw_chatbook.Dreams.query_synthesis import _query_row

    assert _query_row(line, [(_G1, None), (_G2, None)]) is None


@pytest.mark.asyncio
async def test_synthesis_cleans_a_numbered_quoted_and_bolded_response():
    chat, _calls = _capture_chat(
        "1. rust tui new releases\n"
        '2. "[g1] wednesday 13 tour dates seattle"\n'
        "- **[g2]** beginner cello lessons\n"
        "[g1]\n"
        "* surprising deep sea cables [g1][g2]"
    )
    rows = await synthesize_query_rows(
        chat, snapshot=SNAP_TWO_GOALS, count=5, exploration_slots=1)

    assert [(row["query"], row["goal"]) for row in rows] == [
        ("rust tui new releases", None),
        ("wednesday 13 tour dates seattle", _G1),
        ("beginner cello lessons", _G2),
        ("surprising deep sea cables", None),
    ]
    assert not any("[" in row["query"] for row in rows)


# --- One search per distinct query (review round 3, N1) -------------------------


@pytest.mark.asyncio
async def test_synthesis_keeps_the_first_of_queries_that_differ_only_by_mark_or_case():
    """``japan travel news`` and ``[g1] japan travel news`` are the same
    search once the mark is stripped. Keeping both spent a second search on
    the same text, and -- attribution being looked up by query string --
    credited BOTH stories to the goal."""
    chat, _calls = _capture_chat(
        "wednesday 13 tour news\n"
        "[g1] wednesday 13 tour news\n"
        "Wednesday 13 Tour News\n"
        "[g2] beginner cello lessons\n"
        "rust tui news"
    )
    rows = await synthesize_query_rows(
        chat, snapshot=SNAP_TWO_GOALS, count=3, exploration_slots=1)

    assert [(row["query"], row["goal"]) for row in rows] == [
        ("wednesday 13 tour news", None),
        ("beginner cello lessons", _G2),
        ("rust tui news", None),
    ], "the first spelling wins, and duplicates do not use up the count"


def test_fallback_rows_are_distinct_queries_too():
    """Two topics that differ only by case are two profile rows (the key
    is exact) but one search."""
    rows = preview_queries(["Rust TUI", "rust tui", "jazz guitar"], [], count=4)

    assert [row["query"] for row in rows] == [
        "Rust TUI recent developments",
        "jazz guitar recent developments",
        "surprising adjacent to Rust TUI",
    ]


# --- Review round 5, N1: reading a mark must not be quadratic --------------------
#
# ``\s*/?\s*`` in the mark pattern made one long run of spaces after a "["
# cost its square: 20,000 spaces took 1.9 s on the event loop. A run of
# brackets followed by a run of spaces did the same through ``\[+``.


@pytest.mark.parametrize(
    "line",
    [
        "[" + " " * 200_000 + "x",
        "[" * 20_000 + " " * 20_000 + "x",
        "[" + " /" * 50_000,
        "[g" + " " * 200_000 + "1",
        "[g1" + " " * 200_000,
        "[ g 1 ]" * 30_000,
        "\\[" * 60_000,
    ],
    ids=["spaces", "brackets-then-spaces", "slashes", "g-spaces", "tail-spaces",
         "many-marks", "escaped-brackets"],
)
def test_reading_marks_in_a_hostile_line_is_fast(line):
    import time

    from tldw_chatbook.Dreams.query_synthesis import (
        _GOAL_MARK_RE,
        _response_rows,
    )

    started = time.perf_counter()
    list(_GOAL_MARK_RE.finditer(line))
    _response_rows(line + "\nrust tui news", [(_G1, None)])
    elapsed = time.perf_counter() - started

    # Linear work on 200,000 characters is a few milliseconds; the
    # quadratic pattern needed minutes for these. Loose on purpose.
    assert elapsed < 2.0, f"{elapsed:.2f}s"


@pytest.mark.parametrize(
    "line, is_mark",
    [
        ("[g1]", True), ("[ g 1 ]", True), ("[  /  g  1  ]", True),
        ("[[[g1]]]", True),
        # More padding than any model writes is not a mark any more.
        ("[" + " " * 10 + "g1]", False),
        ("[g" + " " * 10 + "1]", False),
    ],
)
def test_the_mark_pattern_is_bounded_but_still_loose(line, is_mark):
    from tldw_chatbook.Dreams.query_synthesis import _GOAL_MARK_RE

    assert bool(_GOAL_MARK_RE.fullmatch(line)) is is_mark


def test_a_line_too_long_to_be_a_query_is_not_one():
    """No search query is hundreds of characters long; a line like that is
    prose, and nothing below has to scan it."""
    from tldw_chatbook.Dreams.query_synthesis import _response_rows

    rows = _response_rows(
        "[g1] " + "very " * 200 + "long\n[g1] tour dates", [(_G1, None)])

    assert [(row["query"], row["goal"]) for row in rows] == [
        ("tour dates", _G1)]
