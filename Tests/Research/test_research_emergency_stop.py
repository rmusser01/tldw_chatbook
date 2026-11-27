"""TASK-33620.1.12: /research is held by the emergency stop too.

Owner ruling, 2026-11-24: "a /research started during a stop is refused with
the stop copy; one already running stops fetching new pages at its next
step."

The engine checks the stop at every step boundary -- before each search, each
paper lookup, and each phase -- and a held run ends like a cancelled one (what
earlier steps saved is kept) with a progress message naming the stop. The
Console's /research refuses at start through the same copy the composer's
send refusal uses. The stop is the per-test file from Tests/conftest.py's
redirect, read through the module as the engine reads it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook import emergency_stop
from tldw_chatbook.emergency_stop import clear_emergency_stop, set_emergency_stop
from tldw_chatbook.Research_Interop.local_research_engine import LocalResearchEngine
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService

STOPPED = "Stopped by the emergency stop"


def _stop_path():
    return emergency_stop.default_emergency_stop_path()


def _pipeline(on_search=None):
    calls = {"search": 0, "analyze": 0}

    def search_fn(q, params):
        calls["search"] += 1
        if on_search is not None:
            on_search()
        return (
            {"results": [{"title": "One", "url": "https://one.example/"}], "warnings": []},
            {"sub_questions": [], "main_goal": q},
        )

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        calls["analyze"] += 1
        return {
            "final_answer": {
                "text": "Answer [1].",
                "evidence": [
                    {
                        "id": 1,
                        "url": "https://one.example/",
                        "title": "One",
                        "content": "c",
                        "original_content": "o",
                        "reasoning": "r",
                        "chunk_index": 1,
                    }
                ],
                "confidence": 0.5,
                "chunks": [],
                "citation_verification": {},
            },
            "relevant_results": {"1": {}},
            "web_search_results_dict": wsr,
        }

    return search_fn, analyze_fn, calls


def _events(service, run_id):
    return [e["event"] for e in service.list_run_events(run_id)]


def test_a_run_started_during_the_stop_fetches_nothing():
    from loguru import logger

    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
    service = LocalResearchService(":memory:")
    run = service.launch_run(query="held from the start", autonomy_mode="autonomous")
    search_fn, analyze_fn, calls = _pipeline()
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    set_emergency_stop(_stop_path(), reason="research hold")

    try:
        final = asyncio.run(engine.execute_run(run["id"]))
    finally:
        logger.remove(sink)

    assert calls == {"search": 0, "analyze": 0}, "research fetched during the stop"
    assert final["status"] == "cancelled"
    assert final["progress_message"].startswith(STOPPED)
    assert "emergency_stop" in _events(service, run["id"])
    # One diagnostic line names the step and why, never the question.
    held = [line for line in lines if "held by the emergency stop" in line]
    assert len(held) == 1 and "before search" in held[0]
    assert "held from the start" not in held[0]


def test_a_running_research_stops_at_its_next_step():
    """The stop lands during a search: that step finishes, the next never starts."""
    service = LocalResearchService(":memory:")
    run = service.launch_run(query="held mid-way", autonomy_mode="autonomous")
    search_fn, analyze_fn, calls = _pipeline(
        on_search=lambda: set_emergency_stop(_stop_path(), reason="landed mid-run")
    )
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert calls == {"search": 1, "analyze": 0}, "a step ran after the stop landed"
    assert final["status"] == "cancelled"
    assert final["progress_message"].startswith(STOPPED)
    # Like a cancellation, what the run saved before the stop is kept.
    names = {a["artifact_name"] for a in service.get_bundle(run["id"])["artifacts"]}
    assert "plan.json" in names


def test_a_stop_landing_mid_round_stops_before_the_next_search():
    """Within one collecting round, each search is its own step.

    Round 2 researches two gap queries. The stop lands during the first: the
    second is never searched (the round-level check already passed, so only
    the per-search check can catch it).
    """
    service = LocalResearchService(":memory:")
    run = service.launch_run(
        query="two gaps", autonomy_mode="autonomous", limits_json={"max_iterations": 2}
    )
    searched: list[str] = []

    def on_search():
        if len(searched) == 2:  # the first gap query of round 2
            set_emergency_stop(_stop_path(), reason="mid round")

    search_fn, analyze_fn, calls = _pipeline(on_search=on_search)

    def recording_search(q, params):
        searched.append(q)
        return search_fn(q, params)

    async def gap_fn(_context):
        return ["gap one", "gap two"]

    engine = LocalResearchEngine(
        service, search_fn=recording_search, analyze_fn=analyze_fn, gap_fn=gap_fn
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert searched == ["two gaps", "gap one"], "a search ran after the stop landed"
    assert calls["analyze"] == 1
    assert final["status"] == "cancelled"
    assert final["progress_message"].startswith(STOPPED)


def test_a_stop_landing_during_the_web_search_skips_the_paper_lookups():
    """Each paper lookup is a step too."""
    service = LocalResearchService(":memory:")
    run = service.launch_run(query="papers held", autonomy_mode="autonomous")
    search_fn, analyze_fn, calls = _pipeline(
        on_search=lambda: set_emergency_stop(_stop_path(), reason="before papers")
    )
    papers: list[str] = []

    def paper_search_fn(query):
        papers.append(query)
        return []

    engine = LocalResearchEngine(
        service,
        search_fn=search_fn,
        analyze_fn=analyze_fn,
        paper_search_fn=paper_search_fn,
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert papers == [], "a paper lookup ran after the stop landed"
    assert calls == {"search": 1, "analyze": 0}
    assert final["status"] == "cancelled"


def test_a_stop_landing_during_synthesis_skips_packaging():
    service = LocalResearchService(":memory:")
    run = service.launch_run(query="held before the report", autonomy_mode="autonomous")
    search_fn, analyze_fn, calls = _pipeline()

    async def stopping_analyze(*args, **kwargs):
        result = await analyze_fn(*args, **kwargs)
        set_emergency_stop(_stop_path(), reason="before packaging")
        return result

    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=stopping_analyze)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert calls == {"search": 1, "analyze": 1}
    assert final["status"] == "cancelled"
    names = {a["artifact_name"] for a in service.get_bundle(run["id"])["artifacts"]}
    assert "report_v1.md" not in names


@pytest.mark.emergency_stop_unavailable
def test_a_stop_that_cannot_be_read_holds_research(monkeypatch):
    def refuse():
        raise PermissionError("data dir refused")

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", refuse)
    service = LocalResearchService(":memory:")
    run = service.launch_run(query="doubt holds", autonomy_mode="autonomous")
    search_fn, analyze_fn, calls = _pipeline()
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert calls["search"] == 0
    assert final["status"] == "cancelled"
    assert "couldn't be read" in final["progress_message"]


def test_research_runs_normally_once_the_stop_is_cleared():
    set_emergency_stop(_stop_path(), reason="brief")
    clear_emergency_stop(_stop_path())
    service = LocalResearchService(":memory:")
    run = service.launch_run(query="after clear", autonomy_mode="autonomous")
    search_fn, analyze_fn, calls = _pipeline()
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert calls == {"search": 1, "analyze": 1}


# --- the Console's /research refuses at start --------------------------------


class _Screen:
    """The ChatScreen surface /research touches, held by the stop."""

    def __init__(self, app, stop_path) -> None:
        from tldw_chatbook.UI.Console_Modules.emergency_stop_hold import HELD_COPY

        self.app = app
        self.app_instance = app
        self.is_mounted = False
        self._console_emergency_stop_copy = HELD_COPY
        self._console_emergency_stop_path = stop_path
        self.system_messages: list[str] = []
        self.worker_coroutines: list[object] = []

    async def _append_native_console_system_message(self, text: str) -> None:
        self.system_messages.append(text)

    def _current_console_conversation_id(self) -> str:
        return "conv-console-1"

    def run_worker(self, coroutine, **kwargs):
        self.worker_coroutines.append(coroutine)
        return None


async def test_console_research_is_refused_during_the_stop_with_the_stop_copy():
    from tldw_chatbook.UI.Console_Modules.emergency_stop_hold import HELD_COPY
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    launched: list[dict] = []
    service = SimpleNamespace(launch_run=lambda **kwargs: launched.append(kwargs))
    app = SimpleNamespace(
        local_research_service=service,
        research_window_academic_enabled=False,
        chachanotes_db=None,
    )
    path = _stop_path()
    set_emergency_stop(path, reason="no research")
    screen = _Screen(app, path)

    await ChatScreen._console_command_research(
        screen, SimpleNamespace(args="why is the sky blue")
    )

    assert screen.worker_coroutines == [], "a research run started during the stop"
    assert launched == []
    assert screen.system_messages == [HELD_COPY]
