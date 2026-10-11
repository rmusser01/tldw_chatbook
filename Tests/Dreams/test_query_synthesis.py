# Tests/Dreams/test_query_synthesis.py
"""Query synthesis: exactly one chat call per cycle, deterministic fallback.

Ruling R1: ``synthesize_queries`` is ``async``; the sync fake chat callables
are offloaded to a thread by the module (the ``briefing_service._invoke_chat``
discipline).
"""
import pytest

from tldw_chatbook.Dreams.query_synthesis import (
    preview_queries,
    synthesize_queries,
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
