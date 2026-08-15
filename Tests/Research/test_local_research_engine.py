"""Local research execution engine (task-16322, ADR-066).

The engine drives an existing local run through planning -> collecting ->
synthesizing -> packaging by REUSING the deep-search pipeline via injectable
runners. Tests fake ONLY the two pipeline seams (search_fn / analyze_fn);
the engine code, the real LocalResearchService (in-memory SQLite), and the
artifact/event storage run real.
"""

import asyncio

import pytest

from tldw_chatbook.Research_Interop.local_research_engine import LocalResearchEngine
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService


def _make_service() -> LocalResearchService:
    return LocalResearchService(":memory:")


def _make_pipeline(question: str):
    """Return (search_fn, analyze_fn, calls) fakes producing a small run."""
    calls: dict[str, int] = {"search": 0, "analyze": 0}

    def search_fn(q, params):
        assert q == question
        calls["search"] += 1
        return (
            {
                "results": [
                    {"title": "One", "url": "https://one.example/"},
                    {"title": "Two", "url": "https://two.example/"},
                ],
                "warnings": [],
            },
            {"sub_questions": ["sub q1"], "main_goal": question},
        )

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        calls["analyze"] += 1
        return {
            "final_answer": {
                "text": "Answer citing [1] and [2?].",
                "evidence": [
                    {"id": 1, "url": "https://one.example/", "title": "One",
                     "content": "c1", "original_content": "o1", "reasoning": "r1",
                     "chunk_index": 1},
                ],
                "confidence": 0.8,
                "chunks": [],
                "citation_verification": {
                    "markers_total": 2, "markers_resolved": 1,
                    "unknown_marker_ids": [2], "quotes_checked": 0,
                    "quotes_verified": 0, "quotes_misquoted": 0,
                    "uncited_sentences": 0,
                },
            },
            "relevant_results": {"1": {}},
            "web_search_results_dict": wsr,
        }

    return search_fn, analyze_fn, calls


def _events(service, run_id):
    return [e["event"] for e in service.list_run_events(run_id)]


def _artifact_content(bundle, name):
    for artifact in bundle["artifacts"]:
        if artifact["artifact_name"] == name:
            return artifact["content"]
    raise AssertionError(f"artifact {name!r} missing from bundle")


def test_engine_executes_phases_and_completes_run():
    service = _make_service()
    run = service.launch_run(query="How do persistent agents checkpoint?")
    search_fn, analyze_fn, calls = _make_pipeline(run["query"])
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert final["phase"] == "completed"
    assert final["progress_percent"] == 100.0
    assert calls == {"search": 1, "analyze": 1}

    events = _events(service, run["id"])
    assert events[0] == "created"
    assert "progress" in events
    assert events[-1] == "completed"

    bundle = service.get_bundle(run["id"])
    names = {a["artifact_name"] for a in bundle["artifacts"]}
    assert {
        "plan.json",
        "collection_summary.json",
        "sources.json",
        "verification_summary.json",
        "report_v1.md",
        "bundle.json",
    } <= names


def test_engine_artifacts_carry_plan_sources_and_citation_verdict():
    service = _make_service()
    run = service.launch_run(query="What changed in SQLite 3.50?")
    search_fn, analyze_fn, _ = _make_pipeline(run["query"])
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    asyncio.run(engine.execute_run(run["id"]))
    bundle = service.get_bundle(run["id"])

    # JSON artifacts normalize back to dicts; text artifacts stay strings.
    plan = _artifact_content(bundle, "plan.json")
    assert plan["query"] == run["query"]
    assert plan["sub_questions"] == ["sub q1"]

    sources = _artifact_content(bundle, "sources.json")
    assert sources["evidence"][0]["url"] == "https://one.example/"

    verification = _artifact_content(bundle, "verification_summary.json")
    assert verification["confidence"] == 0.8
    assert verification["citation_verification"]["unknown_marker_ids"] == [2]

    report = _artifact_content(bundle, "report_v1.md")
    assert "Answer citing [1] and [2?]." in report
    assert "https://one.example/" in report  # sources rendered into the report


def test_engine_normalizes_draft_run_to_running():
    service = _make_service()
    draft = service.create_run(query="Draft question")
    assert draft["status"] == "draft"
    search_fn, analyze_fn, _ = _make_pipeline("Draft question")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(draft["id"]))

    assert final["status"] == "completed"
    assert "engine_started" in _events(service, draft["id"])


def test_engine_fails_run_and_keeps_partial_artifacts_on_pipeline_error():
    service = _make_service()
    run = service.launch_run(query="Exploding question")

    def search_fn(q, params):
        return ({"results": [{"title": "T", "url": "https://t.example/"}], "warnings": []},
                {"sub_questions": [], "main_goal": q})

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        raise RuntimeError("synthesis provider down")

    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "failed"
    assert "synthesis provider down" in final["progress_message"]
    bundle = service.get_bundle(run["id"])
    names = {a["artifact_name"] for a in bundle["artifacts"]}
    assert "plan.json" in names and "collection_summary.json" in names
    assert "report_v1.md" not in names


def test_engine_pause_between_phases_leaves_run_resumable():
    service = _make_service()
    run = service.launch_run(query="Pause me mid-run")

    def search_fn(q, params):
        service.pause_run(run["id"])  # user pauses while collecting runs
        return ({"results": [{"title": "T", "url": "https://t.example/"}], "warnings": []},
                {"sub_questions": [], "main_goal": q})

    analyze_calls = {"n": 0}

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        analyze_calls["n"] += 1
        return {"final_answer": {"text": "x", "evidence": [], "confidence": 0.1, "chunks": []},
                "relevant_results": {}, "web_search_results_dict": wsr}

    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["control_state"] == "paused"
    assert final["status"] == "running"  # non-terminal: resumable
    assert analyze_calls["n"] == 0        # synthesis never started
    assert "engine_paused" in _events(service, run["id"])


def test_engine_cancel_between_phases_resolves_cancelled_once():
    service = _make_service()
    run = service.launch_run(query="Cancel me mid-run")

    def search_fn(q, params):
        service.cancel_run(run["id"])  # user cancels while collecting runs
        return ({"results": [{"title": "T", "url": "https://t.example/"}], "warnings": []},
                {"sub_questions": [], "main_goal": q})

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        raise AssertionError("analyze must not run after cancellation")

    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "cancelled"
    assert _events(service, run["id"]).count("cancelled") == 1


def test_engine_rejects_terminal_run():
    service = _make_service()
    run = service.launch_run(query="Already done")
    service.complete_run(run["id"])
    search_fn, analyze_fn, _ = _make_pipeline(run["query"])
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    with pytest.raises(ValueError, match="terminal"):
        asyncio.run(engine.execute_run(run["id"]))


# --- budget enforcement (task-16323) -------------------------------------------

def _budget_pipeline(question, *, results=2, captured_params=None):
    def search_fn(q, params):
        if captured_params is not None:
            captured_params.update(params)
        return (
            {
                "results": [
                    {"title": f"R{i}", "url": f"https://r{i}.example/"} for i in range(results)
                ],
                "warnings": [],
            },
            {"sub_questions": ["sub q1"], "main_goal": q},
        )

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        seen["results"] = list(wsr.get("results") or [])
        return {
            "final_answer": {"text": "ok[1]", "evidence": [], "confidence": 0.5, "chunks": []},
            "relevant_results": {},
            "web_search_results_dict": wsr,
        }

    seen: dict = {}
    return search_fn, analyze_fn, seen


def test_engine_caps_search_fanout_at_budget_before_spend():
    service = _make_service()
    run = service.launch_run(query="Budgeted", limits_json={"max_searches": 2})
    captured: dict = {}
    search_fn, analyze_fn, _ = _budget_pipeline("Budgeted", captured_params=captured)
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    # The fan-out cap is applied in the params BEFORE phase 1 can spend.
    assert captured["search_default_max_queries"] <= 2
    ledger = _artifact_content(service.get_bundle(run["id"]), "budget_ledger.json")
    assert ledger["searches_used"] == 2  # question + 1 sub-query


def test_engine_truncates_docs_to_budget():
    service = _make_service()
    run = service.launch_run(query="Budgeted docs", limits_json={"max_fetched_docs": 1})
    search_fn, analyze_fn, seen = _budget_pipeline("Budgeted docs", results=3)
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert len(seen["results"]) == 1  # analyze only saw the budgeted batch
    ledger = _artifact_content(service.get_bundle(run["id"]), "budget_ledger.json")
    assert ledger["docs_used"] == 1


def test_engine_stops_cleanly_when_doc_budget_exhausted():
    service = _make_service()
    run = service.launch_run(query="No docs allowed", limits_json={"max_fetched_docs": 0})
    search_fn, analyze_fn, seen = _budget_pipeline("No docs allowed", results=3)
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "failed"
    assert "research_limit_exceeded:max_fetched_docs" in final["progress_message"]
    assert "results" not in seen  # synthesis never started
    names = {a["artifact_name"] for a in service.get_bundle(run["id"])["artifacts"]}
    assert {"plan.json", "collection_summary.json", "budget_ledger.json"} <= names
    assert "report_v1.md" not in names


def test_engine_runtime_budget_stops_run_at_phase_boundary():
    service = _make_service()
    run = service.launch_run(query="No time", limits_json={"max_runtime_seconds": 0})
    search_fn, analyze_fn, seen = _budget_pipeline("No time")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "failed"
    assert "research_limit_exceeded:max_runtime_seconds" in final["progress_message"]
    assert "results" not in seen
    ledger = _artifact_content(service.get_bundle(run["id"]), "budget_ledger.json")
    assert ledger["limits"]["max_runtime_seconds"] == 0.0
