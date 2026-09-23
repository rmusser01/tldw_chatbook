# Tests/Dreams/test_query_synthesis.py
"""Query synthesis: exactly one chat call per cycle, deterministic fallback.

Ruling R1: ``synthesize_queries`` is ``async``; the sync fake chat callables
are offloaded to a thread by the module (the ``briefing_service._invoke_chat``
discipline).
"""
import pytest

from tldw_chatbook.Dreams.query_synthesis import synthesize_queries

SNAP = {"topics": [{"facet": "topic", "text": "rust tui", "weight": 0.9},
                   {"facet": "topic", "text": "jazz guitar", "weight": 0.6}],
        "region": "Seattle"}


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
