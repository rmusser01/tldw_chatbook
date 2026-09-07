"""Local research execution engine (task-16322, ADR-068).

The engine drives an existing local run through planning -> collecting ->
synthesizing -> packaging by REUSING the deep-search pipeline via injectable
runners. Tests fake ONLY the two pipeline seams (search_fn / analyze_fn);
the engine code, the real LocalResearchService (in-memory SQLite), and the
artifact/event storage run real.
"""

import asyncio
import concurrent.futures
import json
from unittest.mock import patch

import pytest

from tldw_chatbook.Research_Interop.local_research_engine import LocalResearchEngine
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
from tldw_chatbook.Research_Interop.research_budget import BudgetLedger


def _make_service() -> LocalResearchService:
    return LocalResearchService(":memory:")


def _run_on_loop(loop: "asyncio.AbstractEventLoop", fn):
    """Run a sync callable on ``loop``'s own thread and block the CALLING
    thread until it completes.

    The engine now offloads a synchronous search_fn to a worker thread
    (finding 1: this stops it from starving the lease keep-alive). A fake
    search_fn that pokes the real, thread-affine ``:memory:`` SQLite
    connection to simulate a concurrent user action (pause/cancel) can no
    longer do so directly from that worker thread -- it must hand the call
    back to the loop's own thread, which is where the connection was
    created.
    """
    future: "concurrent.futures.Future" = concurrent.futures.Future()

    def _call() -> None:
        try:
            future.set_result(fn())
        except BaseException as exc:  # noqa: BLE001 - propagated via the future
            future.set_exception(exc)

    loop.call_soon_threadsafe(_call)
    return future.result(timeout=5)


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
    run = service.launch_run(query="How do persistent agents checkpoint?", autonomy_mode="autonomous")
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
    run = service.launch_run(query="What changed in SQLite 3.50?", autonomy_mode="autonomous")
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
    draft = service.create_run(query="Draft question", autonomy_mode="autonomous")
    assert draft["status"] == "draft"
    search_fn, analyze_fn, _ = _make_pipeline("Draft question")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(draft["id"]))

    assert final["status"] == "completed"
    assert "engine_started" in _events(service, draft["id"])


def test_engine_fails_run_and_keeps_partial_artifacts_on_pipeline_error():
    service = _make_service()
    run = service.launch_run(query="Exploding question", autonomy_mode="autonomous")

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
    run = service.launch_run(query="Pause me mid-run", autonomy_mode="autonomous")
    loop_box: dict[str, asyncio.AbstractEventLoop] = {}

    def search_fn(q, params):
        # user pauses while collecting runs -- run_on_loop hands the DB
        # write back to the loop's own thread (see _run_on_loop docstring).
        _run_on_loop(loop_box["loop"], lambda: service.pause_run(run["id"]))
        return ({"results": [{"title": "T", "url": "https://t.example/"}], "warnings": []},
                {"sub_questions": [], "main_goal": q})

    analyze_calls = {"n": 0}

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        analyze_calls["n"] += 1
        return {"final_answer": {"text": "x", "evidence": [], "confidence": 0.1, "chunks": []},
                "relevant_results": {}, "web_search_results_dict": wsr}

    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    async def _run():
        loop_box["loop"] = asyncio.get_running_loop()
        return await engine.execute_run(run["id"])

    final = asyncio.run(_run())

    assert final["control_state"] == "paused"
    assert final["status"] == "running"  # non-terminal: resumable
    assert analyze_calls["n"] == 0        # synthesis never started
    assert "engine_paused" in _events(service, run["id"])


def test_engine_cancel_between_phases_resolves_cancelled_once():
    service = _make_service()
    run = service.launch_run(query="Cancel me mid-run", autonomy_mode="autonomous")
    loop_box: dict[str, asyncio.AbstractEventLoop] = {}

    def search_fn(q, params):
        # user cancels while collecting runs -- see _run_on_loop docstring.
        _run_on_loop(loop_box["loop"], lambda: service.cancel_run(run["id"]))
        return ({"results": [{"title": "T", "url": "https://t.example/"}], "warnings": []},
                {"sub_questions": [], "main_goal": q})

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        raise AssertionError("analyze must not run after cancellation")

    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    async def _run():
        loop_box["loop"] = asyncio.get_running_loop()
        return await engine.execute_run(run["id"])

    final = asyncio.run(_run())

    assert final["status"] == "cancelled"
    assert _events(service, run["id"]).count("cancelled") == 1


def test_engine_rejects_terminal_run():
    service = _make_service()
    run = service.launch_run(query="Already done", autonomy_mode="autonomous")
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
    run = service.launch_run(query="Budgeted", autonomy_mode="autonomous", limits_json={"max_searches": 2})
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
    run = service.launch_run(query="Budgeted docs", autonomy_mode="autonomous", limits_json={"max_fetched_docs": 1})
    search_fn, analyze_fn, seen = _budget_pipeline("Budgeted docs", results=3)
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert len(seen["results"]) == 1  # analyze only saw the budgeted batch
    ledger = _artifact_content(service.get_bundle(run["id"]), "budget_ledger.json")
    assert ledger["docs_used"] == 1


def test_engine_stops_cleanly_when_doc_budget_exhausted():
    service = _make_service()
    run = service.launch_run(query="No docs allowed", autonomy_mode="autonomous", limits_json={"max_fetched_docs": 0})
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
    run = service.launch_run(query="No time", autonomy_mode="autonomous", limits_json={"max_runtime_seconds": 0})
    search_fn, analyze_fn, seen = _budget_pipeline("No time")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "failed"
    assert "research_limit_exceeded:max_runtime_seconds" in final["progress_message"]
    assert "results" not in seen
    ledger = _artifact_content(service.get_bundle(run["id"]), "budget_ledger.json")
    assert ledger["limits"]["max_runtime_seconds"] == 0.0


# --- iterative gap-driven replanning (task-16324) -------------------------------

def _iter_pipeline(question):
    """Pipeline fakes whose search returns one NEW url per call."""
    state = {"search": 0, "analyze": 0, "merged_results": []}

    def search_fn(q, params):
        state["search"] += 1
        n = state["search"]
        return (
            {"results": [{"title": f"R{n}", "url": f"https://r{n}.example/"}], "warnings": []},
            {"sub_questions": [], "main_goal": q},
        )

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        state["analyze"] += 1
        merged = list(wsr.get("results") or [])
        state["merged_results"] = merged
        return {
            "final_answer": {
                "text": f"Round {state['analyze']} answer",
                "evidence": [
                    {"id": i, "url": r.get("url"), "title": r.get("title"),
                     "content": r.get("content"), "original_content": r.get("content"),
                     "reasoning": "", "chunk_index": 1}
                    for i, r in enumerate(merged, 1)
                ],
                "confidence": 0.5, "chunks": [],
            },
            "relevant_results": {},
            "web_search_results_dict": wsr,
        }

    return search_fn, analyze_fn, state


def test_engine_single_pass_by_default_without_gap_llm():
    service = _make_service()
    run = service.launch_run(query="Single pass", autonomy_mode="autonomous")
    search_fn, analyze_fn, state = _iter_pipeline("Single pass")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    # No gap_fn injected and no final_answer_llm in params -> default gap
    # analysis returns no gaps -> exactly one pass (today's behavior).

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert state["search"] == 1 and state["analyze"] == 1


def test_engine_iterates_until_gaps_resolve_within_max_iterations():
    service = _make_service()
    run = service.launch_run(query="Iterate", autonomy_mode="autonomous", limits_json={"max_iterations": 3})
    search_fn, analyze_fn, state = _iter_pipeline("Iterate")
    gap_calls = []

    async def gap_fn(context):
        gap_calls.append(dict(context))
        return ["gap query 1"] if len(gap_calls) == 1 else []

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn, gap_fn=gap_fn
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert state["search"] == 2 and state["analyze"] == 2
    # Gap analysis saw the synthesized answer from the prior pass.
    assert gap_calls[0]["answer_text"] == "Round 1 answer"
    # Evidence merged across iterations reached synthesis.
    assert [r["url"] for r in state["merged_results"]] == [
        "https://r1.example/",
        "https://r2.example/",
    ]
    events = _events(service, run["id"])
    assert "iteration_started" in events
    bundle = service.get_bundle(run["id"])
    bundle_json = _artifact_content(bundle, "bundle.json")
    assert bundle_json["iterations"] == 2
    assert bundle_json["remaining_gaps"] == []


def test_engine_stops_at_max_iterations_and_reports_remaining_gaps():
    service = _make_service()
    run = service.launch_run(query="Always gappy", autonomy_mode="autonomous", limits_json={"max_iterations": 2})
    search_fn, analyze_fn, state = _iter_pipeline("Always gappy")

    async def gap_fn(context):
        return ["unresolved question"]

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn, gap_fn=gap_fn
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert state["search"] == 2  # hard bound despite gaps remaining
    report = _artifact_content(service.get_bundle(run["id"]), "report_v1.md")
    assert "Remaining gaps" in report and "unresolved question" in report
    bundle_json = _artifact_content(service.get_bundle(run["id"]), "bundle.json")
    assert bundle_json["remaining_gaps"] == ["unresolved question"]
    assert bundle_json["iterations"] == 2


def test_engine_gap_iteration_stops_cleanly_when_search_budget_exhausted():
    service = _make_service()
    run = service.launch_run(
        query="Budgeted iteration",
        autonomy_mode="autonomous",
        limits_json={"max_iterations": 5, "max_searches": 1},
    )
    search_fn, analyze_fn, state = _iter_pipeline("Budgeted iteration")

    async def gap_fn(context):
        return ["gap query 1"]

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn, gap_fn=gap_fn
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "failed"
    assert "research_limit_exceeded:max_searches" in final["progress_message"]
    assert "iteration_started" in _events(service, run["id"])


def test_engine_default_gap_fn_returns_empty_without_llm():
    engine = LocalResearchEngine(_make_service())
    gaps = asyncio.run(engine._default_gap_fn({"answer_text": "x", "sub_questions": []}))
    assert gaps == []


# --- claims artifact + follow-up Q&A (task-16325) -------------------------------

_CLAIMS_CV = {
    "markers_total": 2, "markers_resolved": 1, "unknown_marker_ids": [2],
    "quotes_checked": 0, "quotes_verified": 0, "quotes_misquoted": 0,
    "uncited_sentences": 0,
    "claims": [
        {"claim_id": "claim-1", "text": "Supported fact[1].", "source_ids": [1],
         "unknown_marker_ids": [], "quotes_checked": 0, "quotes_verified": 0,
         "status": "supported"},
        {"claim_id": "claim-2", "text": "Shaky claim[2].", "source_ids": [],
         "unknown_marker_ids": [2], "quotes_checked": 0, "quotes_verified": 0,
         "status": "unverified"},
    ],
}


def _claims_pipeline(question):
    def search_fn(q, params):
        return ({"results": [{"title": "T", "url": "https://t.example/"}], "warnings": []},
                {"sub_questions": ["sq1", "sq2"], "main_goal": q})

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        return {
            "final_answer": {
                "text": "Supported fact[1]. Shaky claim[2?].",
                "evidence": [{"id": 1, "url": "https://t.example/", "title": "T",
                              "content": "c", "original_content": "o", "reasoning": "r",
                              "chunk_index": 1}],
                "confidence": 0.7, "chunks": [],
                "citation_verification": dict(_CLAIMS_CV),
            },
            "relevant_results": {"1": {}},
            "web_search_results_dict": wsr,
        }

    return search_fn, analyze_fn


def test_engine_persists_claims_artifact_with_counts():
    service = _make_service()
    run = service.launch_run(query="Claims question", autonomy_mode="autonomous")
    search_fn, analyze_fn = _claims_pipeline("Claims question")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    claims_artifact = _artifact_content(service.get_bundle(run["id"]), "claims.json")
    assert claims_artifact["claim_count"] == 2
    assert claims_artifact["supported_claim_count"] == 1
    assert claims_artifact["unverified_claim_count"] == 1
    assert claims_artifact["claims"][0]["claim_id"] == "claim-1"


def _completed_claims_run(service):
    run = service.launch_run(query="Claims question", autonomy_mode="autonomous")
    search_fn, analyze_fn = _claims_pipeline("Claims question")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    asyncio.run(engine.execute_run(run["id"]))
    return run


def test_follow_up_answers_from_stored_evidence_with_bounded_seed():
    service = _make_service()
    run = _completed_claims_run(service)
    captured = {}

    async def answer_fn(seed, question):
        captured["seed"] = seed
        captured["question"] = question
        return {"sufficient": True, "answer": "From the stored claims: yes."}

    engine = LocalResearchEngine(service)
    result = asyncio.run(
        engine.answer_follow_up(run["id"], "Is the supported fact reliable?", answer_fn=answer_fn)
    )

    assert result["status"] == "answered"
    assert result["answer"] == "From the stored claims: yes."
    seed = captured["seed"]
    # Server follow_up contract bounds: outline <= 7, key claims <= 5,
    # unresolved <= 5, plus verification counts.
    assert len(seed["outline"]) <= 7
    assert len(seed["key_claims"]) <= 5
    assert len(seed["unresolved_questions"]) <= 5
    assert seed["verification_summary"] == {
        "supported_claim_count": 1, "unsupported_claim_count": 1
    }
    assert seed["key_claims"][0]["claim_id"] == "claim-1"
    assert captured["question"] == "Is the supported fact reliable?"
    # The exchange is recorded for auditability.
    assert "follow_up_answered" in _events(service, run["id"])


def test_follow_up_insufficient_evidence_falls_back_explicitly():
    service = _make_service()
    run = _completed_claims_run(service)

    async def answer_fn(seed, question):
        return {"sufficient": False, "answer": None}

    engine = LocalResearchEngine(service)
    result = asyncio.run(engine.answer_follow_up(run["id"], "Unrelated?", answer_fn=answer_fn))

    assert result["status"] == "insufficient_evidence"
    assert result["answer"] is None
    assert result["suggestion"]
    assert "follow_up_insufficient" in _events(service, run["id"])


def test_follow_up_without_claims_artifact_never_calls_the_llm():
    service = _make_service()
    run = service.launch_run(query="No claims here", autonomy_mode="autonomous")
    called = {"n": 0}

    async def answer_fn(seed, question):
        called["n"] += 1
        return {"sufficient": True, "answer": "fabricated"}

    engine = LocalResearchEngine(service)
    result = asyncio.run(engine.answer_follow_up(run["id"], "Anything?", answer_fn=answer_fn))

    assert result["status"] == "insufficient_evidence"
    assert called["n"] == 0


# --- academic lane into the evidence pool (task-16326) --------------------------

def test_engine_merges_academic_papers_with_doi_dedup():
    service = _make_service()
    run = service.launch_run(query="Papers question", autonomy_mode="autonomous", limits_json={"max_iterations": 2})
    search_fn, analyze_fn, state = _iter_pipeline("Papers question")
    paper_rounds = [
        [
            {"title": "Paper v1", "abstract": "abs", "doi": "10.1/x",
             "url": "https://doi.org/10.1/x", "source": "arxiv"},
            {"title": "Paper v1 preprint", "abstract": "abs", "doi": "10.1/x",
             "url": "https://other.example/x", "source": "semantic_scholar"},
        ],
        [
            {"title": "Paper v1 again", "abstract": "abs", "doi": "10.1/x",
             "url": "https://doi.org/10.1/x", "source": "arxiv"},
            {"title": "Paper v2", "abstract": "abs2", "doi": "10.2/y",
             "url": "https://doi.org/10.2/y", "source": "arxiv"},
        ],
    ]

    async def paper_search_fn(query):
        return paper_rounds.pop(0) if paper_rounds else []

    gap_calls = {"n": 0}

    async def gap_fn(context):
        gap_calls["n"] += 1
        return ["gap 1"] if gap_calls["n"] == 1 else []

    engine = LocalResearchEngine(
        service,
        search_fn=search_fn,
        analyze_fn=analyze_fn,
        gap_fn=gap_fn,
        paper_search_fn=paper_search_fn,
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    urls = [r["url"] for r in state["merged_results"]]
    # r1 (web round 1) + paper v1 ONCE (deduped across providers in round 1
    # and across rounds) + r2 (web round 2) + paper v2 (new DOI round 2).
    assert urls == [
        "https://r1.example/",
        "https://doi.org/10.1/x",
        "https://r2.example/",
        "https://doi.org/10.2/y",
    ]
    sources = _artifact_content(service.get_bundle(run["id"]), "sources.json")
    paper_entries = [
        e for e in sources["evidence"] if str(e.get("url", "")).startswith("https://doi.org/")
    ]
    assert len(paper_entries) == 2


# --- token usage settlement + enforcement (task-16329) ---------------------------

def test_engine_settles_recorded_usage_into_ledger():
    from tldw_chatbook.Chat.usage_recorder import active_recorder

    service = _make_service()
    run = service.launch_run(query="Tokens question", autonomy_mode="autonomous")
    search_fn, analyze_fn, _ = _make_pipeline("Tokens question")

    async def analyze_with_usage(wsr, sqd, params, cancel_event=None):
        recorder = active_recorder()
        if recorder is not None:
            recorder.record_usage(prompt_tokens=30, completion_tokens=10)
        return await analyze_fn(wsr, sqd, params, cancel_event=cancel_event)

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_with_usage
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    ledger = _artifact_content(service.get_bundle(run["id"]), "budget_ledger.json")
    assert ledger["tokens_settled"] == 40
    # record_usage counts are provider-exact (task-16814): not estimates.
    assert ledger["tokens_estimated"] is False


def test_engine_enforces_max_tokens_between_llm_calls():
    from tldw_chatbook.Chat.usage_recorder import active_recorder

    service = _make_service()
    run = service.launch_run(query="Token capped", autonomy_mode="autonomous", limits_json={"max_tokens": 25})
    search_fn, analyze_fn, _ = _make_pipeline("Token capped")

    async def analyze_with_usage(wsr, sqd, params, cancel_event=None):
        recorder = active_recorder()
        if recorder is not None:
            recorder.record_usage(prompt_tokens=20, completion_tokens=10)
        return await analyze_fn(wsr, sqd, params, cancel_event=cancel_event)

    async def gap_with_usage(context):
        recorder = active_recorder()
        if recorder is not None:
            recorder.record_usage(prompt_tokens=5, completion_tokens=5)
        return []

    engine = LocalResearchEngine(
        service,
        search_fn=search_fn,
        analyze_fn=analyze_with_usage,
        gap_fn=gap_with_usage,
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    # Synthesis settled 30 of 25 tokens; the gap call is refused at the
    # boundary -> clean research_limit_exceeded stop with partial artifacts.
    assert final["status"] == "failed"
    assert "research_limit_exceeded:max_tokens" in final["progress_message"]
    names = {a["artifact_name"] for a in service.get_bundle(run["id"])["artifacts"]}
    assert "report_v1.md" not in names
    assert "budget_ledger.json" in names


# --- gate block in verification summary (task-16333) -----------------------------

def test_engine_verification_summary_carries_gate_block():
    service = _make_service()
    run = service.launch_run(query="Gate question", autonomy_mode="autonomous")

    def search_fn(q, params):
        return ({"results": [{"title": "T", "url": "https://t.example/"}], "warnings": []},
                {"sub_questions": [], "main_goal": q})

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        return {
            "final_answer": {
                "text": "Answer[1].", "confidence": 0.6, "chunks": [],
                "evidence": [{"id": 1, "url": "https://t.example/", "title": "T",
                              "content": "c", "original_content": "o", "reasoning": "r",
                              "chunk_index": 1, "gate_unverified": True}],
                "gate": {"relevant": 3, "raw": 5, "fallback": True},
            },
            "relevant_results": {"1": {}},
            "web_search_results_dict": wsr,
        }

    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    summary = _artifact_content(service.get_bundle(run["id"]), "verification_summary.json")
    assert summary["gate"] == {"relevant": 3, "raw": 5, "fallback": True}


# --- chat handoff on completion (task-16481) --------------------------------------

def test_engine_fires_completion_handoff_with_report_bundle():
    service = _make_service()
    run = service.launch_run(
        query="Handoff question",
        autonomy_mode="autonomous",
        chat_handoff={"conversation_id": "conv-42", "origin": "console"},
    )
    search_fn, analyze_fn, _ = _make_pipeline("Handoff question")
    fired = []

    def handoff(payload):
        fired.append(payload)

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn, completion_handoff=handoff
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert len(fired) == 1
    payload = fired[0]
    assert payload["run_id"] == run["id"]
    assert payload["question"] == "Handoff question"
    assert payload["chat_handoff"] == {"conversation_id": "conv-42", "origin": "console"}
    assert "Answer citing" in payload["report_markdown"]
    assert payload["bundle"]["query"] == "Handoff question"


def test_engine_skips_handoff_without_chat_handoff_target():
    service = _make_service()
    run = service.launch_run(query="No handoff", autonomy_mode="autonomous")
    search_fn, analyze_fn, _ = _make_pipeline("No handoff")
    fired = []

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn,
        completion_handoff=fired.append,
    )

    asyncio.run(engine.execute_run(run["id"]))
    assert fired == []


def test_engine_handoff_failure_never_fails_the_run():
    service = _make_service()
    run = service.launch_run(
        query="Boom handoff", autonomy_mode="autonomous",
        chat_handoff={"conversation_id": "x"},
    )
    search_fn, analyze_fn, _ = _make_pipeline("Boom handoff")

    def exploding_handoff(payload):
        raise RuntimeError("handoff sink down")

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn,
        completion_handoff=exploding_handoff,
    )

    final = asyncio.run(engine.execute_run(run["id"]))
    assert final["status"] == "completed"  # handoff failure is a warning only


# --- checkpointed autonomy (task-16482) -------------------------------------------

_cp_state = {"search": 0, "analyze": 0}


def _mk_cp_engine(service, question):
    def search_fn(q, params):
        _cp_state["search"] += 1
        return (
            {"results": [{"title": "S1", "url": "https://s1.example/"},
                         {"title": "S2", "url": "https://s2.example/"}], "warnings": []},
            {"sub_questions": [], "main_goal": q},
        )

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        _cp_state["analyze"] += 1
        return {
            "final_answer": {"text": "Report[1].", "confidence": 0.6, "chunks": [],
                             "evidence": [{"id": 1, "url": "https://s1.example/", "title": "S1",
                                           "content": "c", "original_content": "o", "reasoning": "r",
                                           "chunk_index": 1}]},
            "relevant_results": {"1": {}},
            "web_search_results_dict": wsr,
        }

    return LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)


def test_checkpointed_run_pauses_at_plan_review():
    service = _make_service()
    run = service.launch_run(query="Checkpointed question")  # default: checkpointed
    engine = _mk_cp_engine(service, "Checkpointed question")

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "running"  # non-terminal
    assert final["control_state"] == "awaiting_plan_review"
    pending = service.latest_pending_checkpoint(run["id"])
    assert pending["checkpoint_type"] == "plan_review"
    names = {a["artifact_name"] for a in service.get_bundle(run["id"])["artifacts"]}
    assert "plan.json" in names and "report_v1.md" not in names
    assert _cp_state["search"] == 0


def test_approved_plan_checkpoint_advances_to_sources_review():
    _cp_state.update(search=0, analyze=0)
    service = _make_service()
    run = service.launch_run(query="Checkpointed question")
    engine = _mk_cp_engine(service, "Checkpointed question")
    asyncio.run(engine.execute_run(run["id"]))
    plan_cp = service.latest_pending_checkpoint(run["id"])
    service.patch_and_approve_checkpoint(run["id"], plan_cp["id"], patch_payload={"limits": {}})

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["control_state"] == "awaiting_sources_review"
    pending = service.latest_pending_checkpoint(run["id"])
    assert pending["checkpoint_type"] == "sources_review"
    assert set(pending["proposed_payload"]["source_ids"]) == {
        "https://s1.example/", "https://s2.example/",
    }
    assert _cp_state["search"] == 1  # collected once
    assert _cp_state["analyze"] == 0  # synthesis not reached


def test_approved_sources_checkpoint_completes_and_applies_dropped():
    _cp_state.update(search=0, analyze=0)
    service = _make_service()
    run = service.launch_run(query="Checkpointed question")
    engine = _mk_cp_engine(service, "Checkpointed question")
    asyncio.run(engine.execute_run(run["id"]))
    service.patch_and_approve_checkpoint(
        run["id"], service.latest_pending_checkpoint(run["id"])["id"]
    )
    asyncio.run(engine.execute_run(run["id"]))
    service.patch_and_approve_checkpoint(
        run["id"],
        service.latest_pending_checkpoint(run["id"])["id"],
        patch_payload={"dropped_source_ids": ["https://s2.example/"]},
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    sources = _artifact_content(service.get_bundle(run["id"]), "sources.json")
    urls = [e.get("url") for e in sources["evidence"]]
    assert urls == ["https://s1.example/"]


def test_sources_recollect_loops_back_to_collecting():
    _cp_state.update(search=0, analyze=0)
    service = _make_service()
    run = service.launch_run(query="Checkpointed question")
    engine = _mk_cp_engine(service, "Checkpointed question")
    asyncio.run(engine.execute_run(run["id"]))
    service.patch_and_approve_checkpoint(
        run["id"], service.latest_pending_checkpoint(run["id"])["id"]
    )
    asyncio.run(engine.execute_run(run["id"]))
    service.patch_and_approve_checkpoint(
        run["id"],
        service.latest_pending_checkpoint(run["id"])["id"],
        patch_payload={"recollect": {"enabled": True}},
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    # Re-collected and waiting at a NEW sources review, not completed.
    assert _cp_state["search"] == 2
    assert final["control_state"] == "awaiting_sources_review"
    assert service.latest_pending_checkpoint(run["id"])["checkpoint_type"] == "sources_review"


def test_approved_sources_checkpoint_with_empty_patch_still_passes():
    _cp_state.update(search=0, analyze=0)
    service = _make_service()
    run = service.launch_run(query="Checkpointed question")
    engine = _mk_cp_engine(service, "Checkpointed question")
    asyncio.run(engine.execute_run(run["id"]))
    service.patch_and_approve_checkpoint(
        run["id"], service.latest_pending_checkpoint(run["id"])["id"]
    )
    asyncio.run(engine.execute_run(run["id"]))
    # Approve with NO patch: the boundary must PASS, not re-await forever.
    service.patch_and_approve_checkpoint(
        run["id"], service.latest_pending_checkpoint(run["id"])["id"]
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"


# --- source policy + provider overrides (task-16791) -------------------------------

def _policy_pipeline(state):
    def search_fn(q, params):
        state["web"] = state.get("web", 0) + 1
        state["last_params"] = dict(params)
        return ({"results": [{"title": "Web", "url": "https://web.example/"}],
                 "warnings": []},
                {"sub_questions": [], "main_goal": q})

    async def paper_fn(query, **kwargs):
        state.setdefault("papers", []).append(kwargs.get("providers"))
        return [{"title": "Paper", "abstract": "p", "doi": "10.9/z",
                 "url": "https://doi.org/10.9/z", "source": "pubmed"}]

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        state["merged"] = list(wsr.get("results") or [])
        return {
            "final_answer": {"text": "R[1].", "confidence": 0.5, "chunks": [],
                             "evidence": [{"id": 1, "url": "https://web.example/",
                                           "title": "Web", "content": "c",
                                           "original_content": "o", "reasoning": "r",
                                           "chunk_index": 1}]},
            "relevant_results": {"1": {}},
            "web_search_results_dict": wsr,
        }

    return search_fn, analyze_fn, paper_fn


def _run_policy_engine(service, question, *, policy=None, overrides=None):
    state: dict = {}
    search_fn, analyze_fn, paper_fn = _policy_pipeline(state)
    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn,
        paper_search_fn=paper_fn,
    )
    kwargs = {"query": question, "autonomy_mode": "autonomous"}
    if policy:
        kwargs["source_policy"] = policy
    if overrides:
        kwargs["provider_overrides"] = overrides
    run = service.launch_run(**kwargs)
    return run, engine, state


def test_policy_web_only_skips_academic_lane():
    service = _make_service()
    run, engine, state = _run_policy_engine(service, "Q", policy="web_only")
    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert state["web"] >= 1
    assert "papers" not in state
    assert all(r.get("url") == "https://web.example/" for r in state["merged"])


def test_policy_academic_only_skips_web_engine():
    service = _make_service()
    run, engine, state = _run_policy_engine(service, "Q", policy="academic_only")
    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert "web" not in state  # zero web-engine spend
    assert state["papers"]  # academic lane ran
    assert [r.get("url") for r in state["merged"]] == ["https://doi.org/10.9/z"]


def test_policy_academic_first_orders_papers_before_web():
    service = _make_service()
    run, engine, state = _run_policy_engine(service, "Q", policy="academic_first")
    asyncio.run(engine.execute_run(run["id"]))

    urls = [r.get("url") for r in state["merged"]]
    assert urls.index("https://doi.org/10.9/z") < urls.index("https://web.example/")


def test_provider_overrides_reach_params_and_papers():
    service = _make_service()
    run, engine, state = _run_policy_engine(
        service, "Q",
        overrides={"engine": "duckduckgo", "result_count": 3,
                   "academic_providers": ["pubmed"]},
    )
    asyncio.run(engine.execute_run(run["id"]))

    assert state["last_params"]["engine"] == "duckduckgo"
    assert state["last_params"]["result_count"] == 3
    assert state["papers"] == [["pubmed"]]


# --- pipeline params pre-flight (task-17371) --------------------------------
# A run whose engine was constructed WITHOUT search_params used to reach the
# real pipeline and die inside generate_and_search's own validation with
# "Invalid search_params parameter" -- a message that names neither what is
# missing nor where it comes from. Research_Window shipped exactly that
# construction (no search_params at all), so every window-launched run failed
# with it. The engine now refuses the unusable configuration up front, and
# only when the REAL pipeline is the search function.


def test_default_pipeline_without_search_params_fails_legibly():
    """No search_params + the default (real) pipeline: the run must fail
    naming the missing keys and where they come from, not with the
    pipeline's opaque 'Invalid search_params parameter'."""
    service = _make_service()
    engine = LocalResearchEngine(service)  # no search_params -- the window's bug
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "failed"
    # fail_run records the reason in progress_message (service contract).
    message = str(final.get("progress_message") or final.get("error_msg") or "")
    # Names the missing keys...
    assert "engine" in message and "result_count" in message
    # ...and where they come from.
    assert "SearchSettings" in message


def test_injected_search_fn_skips_the_pipeline_preflight():
    """The pre-flight is the REAL pipeline's requirement, not the engine's:
    a caller injecting its own search_fn (every other test here, and any
    future non-web lane) still runs with empty search_params."""
    service = _make_service()
    search_fn, analyze_fn, _calls = _make_pipeline("q")
    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn
    )  # still no search_params
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"


def test_preflight_refuses_params_without_usable_llms():
    """Qodo (PR 1764): search keys alone let a run spend phase-1 searches and
    only then fail for want of an LLM. The tool path refuses both cases before
    phase 1; the engine matches it for every default-pipeline caller."""
    service = _make_service()
    engine = LocalResearchEngine(
        service,
        search_params={
            "engine": "duckduckgo",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 5,
            # both LLM slots present but empty -- the shape a config with no
            # [SearchSettings] LLMs actually produces
            "relevance_analysis_llm": "",
            "final_answer_llm": None,
        },
    )
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "failed"
    message = str(final.get("progress_message") or "")
    assert "relevance_analysis_llm" in message and "final_answer_llm" in message
    assert "SearchSettings" in message


def test_preflight_accepts_fully_configured_params():
    """The complement: a complete assembly must pass the pre-flight, or the
    check would refuse every real run."""
    service = _make_service()
    engine = LocalResearchEngine(service)

    engine._require_pipeline_params(
        {
            "engine": "duckduckgo",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 5,
            "relevance_analysis_llm": "llama_cpp",
            "final_answer_llm": "llama_cpp",
        }
    )


# --- multi-hop on by default (task-17371) -------------------------------------
# Gap-driven replanning shipped but max_iterations defaulted to 1, so every
# real run was single-pass and the mechanism never ran. task-17370 measured
# what it is worth: on the one question whose synthesis path was intact,
# a second round held the gate rate while taking resolved markers from 24 to
# 39 and citation density from 0.77 to 0.95. Deep research defaults to it now.


def _gap_pipeline(question: str, gaps_per_round):
    """search/analyze/gap fakes recording the queries of every round."""
    rounds: list[str] = []
    remaining = list(gaps_per_round)

    def search_fn(q, params):
        rounds.append(q)
        return (
            {"results": [{"title": f"R:{q}", "url": f"https://x.example/{len(rounds)}"}],
             "warnings": []},
            {"sub_questions": [], "main_goal": question},
        )

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        return {
            "final_answer": {"text": "Answer citing [1].", "evidence": [
                {"id": 1, "url": "https://x.example/1", "title": "R"}],
                "confidence": 0.5, "chunks": []},
            "relevant_results": {"0": {"url": "https://x.example/1"}},
        }

    async def gap_fn(context):
        return remaining.pop(0) if remaining else []

    return search_fn, analyze_fn, gap_fn, rounds


def test_multi_hop_runs_a_second_round_by_default():
    """No limits at all: the run must research the gaps its first synthesis
    left open, not stop after one pass."""
    service = _make_service()
    search_fn, analyze_fn, gap_fn, rounds = _gap_pipeline("q", [["gap one"], []])
    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn, gap_fn=gap_fn
    )
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    assert rounds == ["q", "gap one"], rounds


def test_explicit_single_pass_limit_still_wins():
    """A run that asks for one pass gets one pass -- the default must not
    override what a caller (or the baseline recorder) states."""
    service = _make_service()
    search_fn, analyze_fn, gap_fn, rounds = _gap_pipeline("q", [["gap one"], []])
    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn, gap_fn=gap_fn
    )
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    asyncio.run(engine.execute_run(run["id"]))

    assert rounds == ["q"], rounds


def test_configured_iteration_default_is_honoured(monkeypatch):
    """Operators can move the shipped default without editing code."""
    from tldw_chatbook.Research_Interop import local_research_engine as engine_module

    monkeypatch.setattr(
        engine_module, "_configured_max_iterations", lambda: 3
    )
    service = _make_service()
    search_fn, analyze_fn, gap_fn, rounds = _gap_pipeline(
        "q", [["gap one"], ["gap two"], []]
    )
    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn, gap_fn=gap_fn
    )
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    asyncio.run(engine.execute_run(run["id"]))

    assert rounds == ["q", "gap one", "gap two"], rounds


def test_plan_review_patch_bounds_the_iterations():
    """Qodo (PR 1766): an approved plan-review patch reached the budget ledger
    but not the iteration bound, which was re-read from the run record. With
    multi-hop as the default, a run the user had just limited to ONE pass could
    still perform a second -- spending more than the review had approved.

    Driven on an AUTONOMOUS run with a pre-approved plan patch, because a
    resumed CHECKPOINTED run restarts the phase machine from the top (a
    documented v1 limitation), which makes counting rounds through that path
    ambiguous. The merge under test is the same one either way.
    """
    service = _make_service()
    search_fn, analyze_fn, gap_fn, rounds = _gap_pipeline("q", [["gap one"], []])
    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=analyze_fn, gap_fn=gap_fn
    )
    # No max_iterations of its own, so ONLY the approved patch can bound it.
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    checkpoint = service.create_checkpoint(
        run["id"], checkpoint_type="plan_review", proposed_payload={"query": "q"}
    )
    service.patch_and_approve_checkpoint(
        run["id"], checkpoint["id"], patch_payload={"limits": {"max_iterations": 1}}
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed", final.get("progress_message")
    # Without the fix this is ["q", "gap one"]: the ledger honoured the patch
    # while the iteration bound fell back to the shipped default of 2.
    assert rounds == ["q"], rounds


# --- fan-out reaches the academic lane (task-17372) ---------------------------
# Sub-question generation lives inside the WEB pipeline, so generated
# sub-questions only ever drove web searches: the academic lane looped
# round_queries, which is [question] in round 1. Fan-out therefore changed how
# academic evidence was JUDGED (the sub-questions reach the gate) while leaving
# what was RETRIEVED untouched -- which is why task-17370 measured it as flat on
# the repositories lane and could say nothing about retrieval.


def _fanout_pipeline(question: str, sub_questions: list[str]):
    """search_fn returning generated sub-questions, plus a recording paper_fn."""
    paper_queries: list[str] = []

    def search_fn(q, params):
        return (
            {"results": [{"title": "web", "url": "https://w.example/1"}], "warnings": []},
            {"sub_questions": list(sub_questions), "main_goal": question},
        )

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        return {
            "final_answer": {
                "text": "Answer citing [1].",
                "evidence": [{"id": 1, "url": "https://w.example/1", "title": "web"}],
                "confidence": 0.5,
                "chunks": [],
            },
            "relevant_results": {"0": {"url": "https://w.example/1"}},
        }

    def paper_search_fn(query):
        paper_queries.append(query)
        return [
            {
                "title": f"paper for {query}",
                "url": f"https://p.example/{len(paper_queries)}",
                "metadata": {"doi": f"10.1/{len(paper_queries)}"},
            }
        ]

    async def gap_fn(context):
        return []

    return search_fn, analyze_fn, paper_search_fn, gap_fn, paper_queries


def test_generated_sub_questions_reach_the_paper_providers():
    service = _make_service()
    search_fn, analyze_fn, paper_fn, gap_fn, paper_queries = _fanout_pipeline(
        "q", ["facet one", "facet two"]
    )
    engine = LocalResearchEngine(
        service,
        search_fn=search_fn,
        analyze_fn=analyze_fn,
        gap_fn=gap_fn,
        paper_search_fn=paper_fn,
        search_params={"search_default_max_queries": 5},
    )
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    asyncio.run(engine.execute_run(run["id"]))

    assert paper_queries == ["q", "facet one", "facet two"], paper_queries


def test_academic_fan_out_respects_the_query_cap():
    """The lane must obey the same total-queries cap the web lane does."""
    service = _make_service()
    search_fn, analyze_fn, paper_fn, gap_fn, paper_queries = _fanout_pipeline(
        "q", ["facet one", "facet two", "facet three"]
    )
    engine = LocalResearchEngine(
        service,
        search_fn=search_fn,
        analyze_fn=analyze_fn,
        gap_fn=gap_fn,
        paper_search_fn=paper_fn,
        search_params={"search_default_max_queries": 2},
    )
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    asyncio.run(engine.execute_run(run["id"]))

    assert paper_queries == ["q", "facet one"], paper_queries


def test_academic_fan_out_cannot_spend_past_the_search_budget():
    """Extra academic searches are ledger-counted, so a tight max_searches
    cannot be exceeded by the lane fanning out."""
    service = _make_service()
    search_fn, analyze_fn, paper_fn, gap_fn, paper_queries = _fanout_pipeline(
        "q", ["facet one", "facet two", "facet three"]
    )
    engine = LocalResearchEngine(
        service,
        search_fn=search_fn,
        analyze_fn=analyze_fn,
        gap_fn=gap_fn,
        paper_search_fn=paper_fn,
        search_params={"search_default_max_queries": 5},
    )
    # The web call settles 1 + len(sub_questions) = 4 searches, so a budget of 5
    # leaves room for exactly one extra academic query.
    run = service.launch_run(
        query="q",
        autonomy_mode="autonomous",
        limits_json={"max_iterations": 1, "max_searches": 5},
    )

    asyncio.run(engine.execute_run(run["id"]))

    assert paper_queries == ["q", "facet one"], paper_queries


def test_a_sub_question_equal_to_the_question_is_not_searched_twice():
    service = _make_service()
    search_fn, analyze_fn, paper_fn, gap_fn, paper_queries = _fanout_pipeline(
        "q", ["  Q  ", "facet one"]
    )
    engine = LocalResearchEngine(
        service,
        search_fn=search_fn,
        analyze_fn=analyze_fn,
        gap_fn=gap_fn,
        paper_search_fn=paper_fn,
        search_params={"search_default_max_queries": 5},
    )
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    asyncio.run(engine.execute_run(run["id"]))

    assert paper_queries == ["q", "facet one"], paper_queries


# --- _academic_queries directly (Qodo, PR 1772) --------------------------------
# The helper was only exercised through full runs, so its cap, dedup and
# budget arithmetic were inferred from end-to-end behaviour rather than pinned.


def _queries_engine():
    return LocalResearchEngine(_make_service(), search_fn=lambda q, p: ({}, {}))


def test_academic_queries_keeps_primary_queries_and_appends_facets():
    engine = _queries_engine()
    ledger = BudgetLedger.from_limits({})

    queries, reserved = engine._academic_queries(
        ["q"], ["facet a", "facet b"], {"search_default_max_queries": 5}, ledger
    )

    assert queries == ["q", "facet a", "facet b"]
    assert reserved == 2


def test_academic_queries_reserves_without_settling():
    """Reserve-before/settle-after: planning must not record spend."""
    engine = _queries_engine()
    ledger = BudgetLedger.from_limits({"max_searches": 10})

    _queries, reserved = engine._academic_queries(
        ["q"], ["facet a"], {"search_default_max_queries": 5}, ledger
    )

    assert reserved == 1
    snapshot = ledger.snapshot()
    # The reservation is visible, but nothing is settled as spent yet.
    assert snapshot.get("searches_settled", 0) == 0, snapshot


def test_academic_queries_stops_at_the_remaining_budget():
    engine = _queries_engine()
    ledger = BudgetLedger.from_limits({"max_searches": 2})
    ledger.reserve_searches(1)
    ledger.settle_searches(1)

    queries, reserved = engine._academic_queries(
        ["q"], ["facet a", "facet b", "facet c"],
        {"search_default_max_queries": 9}, ledger,
    )

    assert reserved <= 1, reserved
    assert queries[0] == "q"


def test_academic_queries_dedupes_across_primaries_and_facets():
    engine = _queries_engine()
    ledger = BudgetLedger.from_limits({})

    queries, reserved = engine._academic_queries(
        ["q", "  Q "], ["  q  ", "facet a", "FACET A"],
        {"search_default_max_queries": 9}, ledger,
    )

    assert queries == ["q", "facet a"], queries
    assert reserved == 1


def test_academic_queries_falls_back_when_the_cap_is_unusable():
    engine = _queries_engine()
    ledger = BudgetLedger.from_limits({})

    queries, _reserved = engine._academic_queries(
        ["q"], ["a", "b", "c", "d", "e", "f", "g"],
        {"search_default_max_queries": "not-a-number"}, ledger,
    )

    assert len(queries) == 5, queries  # DEFAULT_MAX_QUERIES


def test_a_failed_synthesis_is_recorded_on_the_run():
    """task-17386: a synthesis that never returns leaves no citation verdict,
    which used to make the run indistinguishable from one nobody scored -- it
    completed quietly and vanished from any aggregate. The run must say so."""
    service = _make_service()

    def search_fn(q, params):
        return (
            {"results": [{"title": "One", "url": "https://one.example/"}], "warnings": []},
            {"sub_questions": [], "main_goal": q},
        )

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        # Exactly what aggregate_results returns when synthesis raises.
        return {
            "final_answer": {
                "text": "Could not create the report due to an error (ReadTimeoutError).",
                "evidence": [],
                "confidence": 0.0,
                "chunks": [],
                "synthesis_failed": {
                    "stage": "synthesis",
                    "error_type": "ReadTimeoutError",
                    "evidence_count": 46,
                    "chunk_count": 6,
                },
            },
            "relevant_results": {},
        }

    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed"
    summary = service.get_artifact(run["id"], "verification_summary.json") or {}
    content = summary.get("content") or {}
    assert content.get("synthesis_failed", {}).get("error_type") == "ReadTimeoutError"
    warnings = " ".join(str(w) for w in (content.get("warnings") or []))
    assert "synthesis produced no report" in warnings, content.get("warnings")


def test_a_second_engine_declines_a_leased_run():
    """task-18060: two executors must not run one run. The window's
    exclusive-worker guard is per-session and cannot see a second process.

    Review finding 6: the original version of this test asserted only
    `status != "completed"` and captured `_calls` without ever checking it
    -- it would have passed even if the declined executor had gone ahead
    and run a phase anyway (e.g. if the None-return short-circuit were
    accidentally removed further down the call stack). Assert directly
    that the declined executor performed no work.
    """
    service = _make_service()
    search_fn, analyze_fn, calls = _make_pipeline("q")
    first = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    second = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    service.claim_run(run["id"], worker_id=first.worker_id, lease_seconds=60)
    final = asyncio.run(second.execute_run(run["id"]))

    assert final["status"] != "completed"
    assert final["status"] != "failed"  # declined =/= failed: the run is left alone
    assert "lease_declined" in _events(service, run["id"])
    assert calls == {"search": 0, "analyze": 0}, calls


def test_a_declined_executor_does_not_write_run_state():
    """PR-1822 review follow-up: the declined executor's lease_declined
    handling used ``update_run_progress`` -- an unfenced run-state write by
    a NON-lease-holder that bumped the run's version and stomped the live
    executor's progress message mid-collection, contradicting the
    single-writer principle every fenced write in the engine enforces.

    The decline is still observable: as an append-only event
    (``lease_declined``) in the run's event stream, which any observer may
    append to, and which overwrites nothing.
    """
    service = _make_service()
    search_fn, analyze_fn, _calls = _make_pipeline("q")
    holder = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    declined = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    assert service.claim_run(
        run["id"], worker_id=holder.worker_id, lease_seconds=60
    )
    # The live executor's in-flight progress message must survive.
    service.update_run_progress(
        run["id"], progress_message="Collecting sources (iteration 1)"
    )
    before = service.get_run(run["id"])
    assert before["progress_message"] == "Collecting sources (iteration 1)"

    asyncio.run(declined.execute_run(run["id"]))

    after = service.get_run(run["id"])
    assert after["progress_message"] == "Collecting sources (iteration 1)", after
    assert after["version"] == before["version"], after
    assert "lease_declined" in _events(service, run["id"])


def test_a_run_whose_lease_retry_budget_is_exhausted_is_failed():
    """task-18060 review finding 1: before the fix, exhausting the reclaim
    budget made claim_run return None -- exactly like "another executor
    holds it live" -- so execute_run wrote a lease_declined progress event
    and left the run status=running forever, permanently unclaimable
    (a REGRESSION: before this branch such a run could simply be
    re-executed). A budget-exhausted run must instead be failed, since its
    executor keeps dying rather than merely losing a race.
    """
    service = _make_service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    # Simulate a dead executor: claimed and abandoned (never released) three
    # times in a row, each with an already-expired lease so the next claim
    # can reclaim it deterministically (no time.sleep(), same technique as
    # test_reclaim_stops_at_the_retry_budget in test_research_run_lease.py).
    for _ in range(3):
        assert service.claim_run(
            run["id"], worker_id="dead-executor", lease_seconds=0, max_attempts=3
        )

    engine = LocalResearchEngine(service)  # never reaches the pipeline

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "failed"
    message = str(final.get("progress_message") or "")
    assert "lease" in message.lower() or "executor" in message.lower(), message


def test_a_run_cancelled_between_the_engines_terminal_check_and_its_claim_is_not_executed():
    """task-3 report finding 1: ``claim_run`` restricted acquisition by
    lease expiry only, never by run status, while ``execute_run``'s own
    terminal check (the ``ValueError`` guard at the top of ``execute_run``)
    runs BEFORE the claim. A cancellation landing in that exact gap --
    after the check reads "running", before ``claim_run``'s atomic UPDATE
    -- let a terminal run be claimed and executed (resurrected) anyway.

    Reproduces the race directly at its source: ``service.claim_run`` is
    monkeypatched to cancel the run (simulating a concurrent caller racing
    in right where the finding describes) immediately before delegating to
    the real ``claim_run`` -- so by the time the atomic UPDATE runs, the
    run really is terminal, and the fix (the status condition living in
    that same UPDATE) must refuse the claim.
    """
    service = _make_service()
    search_fn, analyze_fn, calls = _make_pipeline("q")
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    original_claim_run = service.claim_run

    def racing_claim_run(run_id, **kwargs):
        service.cancel_run(run_id)
        return original_claim_run(run_id, **kwargs)

    service.claim_run = racing_claim_run
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "cancelled", (
        "the run must stay exactly as the racing cancellation left it, "
        "never resurrected back to a running/completed pipeline execution"
    )
    assert calls == {"search": 0, "analyze": 0}, (
        f"a terminal run must never reach the pipeline: {calls}"
    )


def test_a_lease_stolen_during_synthesis_blocks_packaging_writes_and_completion():
    """task-18060 review finding 2: regression guard for the nine
    `_require_lease()` fences threaded through `_execute_phases`. Steals the
    run out from under the executing engine from INSIDE a fake pipeline seam
    (`analyze_fn`, which the engine awaits directly on the loop thread, so
    the steal needs no cross-thread plumbing) once collecting has already
    written its artifacts under a still-valid lease. Every packaging write
    that follows the theft must be blocked, and the run must never be
    marked completed.

    Mutation-verified (task report): deleting the `_require_lease()` call
    at the top of `_save_ledger` (the first fence reached after this test's
    theft point, called again at the top of the packaging section) turns
    this test red; restoring it turns it green again.
    """
    service = _make_service()
    search_fn, _default_analyze, _calls = _make_pipeline("q")
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )
    # task-3 report finding 2: `stolen` used to live only inside the seam's
    # local scope, so an in-seam `assert stolen is not None` failing (e.g. if
    # the theft silently did nothing) raised an AssertionError that propagated
    # into execute_run's `except Exception` handler -- which still reaches a
    # non-"completed" terminal status either way (see the outer assertion
    # below), so every "shipped" assertion kept passing even with a no-op
    # theft. Stashing the rescuer's lease id in this outer box lets the test
    # verify, from OUTSIDE execute_run's exception handling, that a real
    # third party actually holds the run's lease afterward -- an assertion
    # that cannot be swallowed by any path through execute_run.
    theft_box: dict[str, str | None] = {}

    async def stealing_analyze_fn(wsr, sqd, params, cancel_event=None):
        # A live (unexpired) lease cannot be reclaimed by another claim_run
        # call -- that non-double-claim guarantee is the whole point of the
        # atomic UPDATE. Releasing it first (using the lease id the engine
        # itself is holding, right now, mid-flight) is what actually
        # simulates "a second executor now owns this run", deterministically
        # and without a real sleep.
        released = service.release_lease(run["id"], lease_id=engine._lease_id)
        assert released is True, "the engine must still hold its own lease at this point"
        stolen = service.claim_run(run["id"], worker_id="rescuer", lease_seconds=60)
        theft_box["lease_id"] = stolen
        assert stolen is not None, "the steal itself must succeed for this test to mean anything"
        return {
            "final_answer": {"text": "Answer[1].", "evidence": [], "confidence": 0.5, "chunks": []},
            "relevant_results": {},
            "web_search_results_dict": wsr,
        }

    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=stealing_analyze_fn)

    complete_calls = {"n": 0}
    original_complete_run = service.complete_run

    def spy_complete_run(*args, **kwargs):
        complete_calls["n"] += 1
        return original_complete_run(*args, **kwargs)

    service.complete_run = spy_complete_run

    final = asyncio.run(engine.execute_run(run["id"]))

    # The positive outcome: a real rescuer must actually hold the run's
    # lease now. This is checked here, in the test's own top-level code --
    # not inside the seam -- so it cannot be short-circuited by execute_run
    # swallowing the in-seam assertion into some other non-"completed"
    # terminal status (fail_run's "failed", or a quiet _LeaseLost return
    # that leaves status at its launch-time "running"). Both of those satisfy
    # a bare `!= "completed"`, which is why that check alone masked a no-op
    # theft; this one cannot be satisfied unless the rescuer's claim really
    # landed.
    assert theft_box.get("lease_id"), "the rescuer's claim_run must have returned a lease id"
    assert service.holds_lease(run["id"], lease_id=theft_box["lease_id"]) is True, (
        "the rescuer must still hold the run's lease after the displaced "
        "executor's execute_run returns"
    )
    assert final["status"] != "completed"
    assert complete_calls["n"] == 0, "the displaced executor must never call complete_run"
    names = {a["artifact_name"] for a in service.get_bundle(run["id"])["artifacts"]}
    # Round-1 collecting wrote its artifacts before the theft (partial-
    # artifact contract) -- packaging must not have written anything after it.
    assert "plan.json" in names
    assert "collection_summary.json" in names
    for packaging_artifact in ("report_v1.md", "sources.json", "verification_summary.json", "bundle.json"):
        assert packaging_artifact not in names, f"{packaging_artifact} must not exist: {names}"


def test_the_last_lease_fence_blocks_completion_after_the_final_write():
    """task-18060 review finding 2 (mutation-check companion): isolates the
    VERY LAST `_require_lease()` call -- immediately before `complete_run`
    -- from every fence before it, by stealing the lease as a side effect of
    the LAST artifact write (`bundle.json`) rather than from inside a
    pipeline seam. Every earlier fence has already passed by the time the
    theft happens, so only that one final fence stands between the theft and
    a wrongly "completed" run.

    Mutation-verified (task report): deleting the `_require_lease()` call
    immediately preceding `self.service.complete_run(...)` turns this test
    red (the run gets marked completed on a stolen lease); restoring it
    turns it green again.
    """
    service = _make_service()
    search_fn, analyze_fn, _calls = _make_pipeline("q")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    # task-3 report finding 2: see the sibling theft test above for why this
    # must be checked from OUTSIDE the seam -- an in-seam `assert stolen is
    # not None` failing still lands execute_run at a non-"completed" terminal
    # status either way (fail_run's "failed", or a quiet _LeaseLost return),
    # so a bare `!= "completed"` check kept passing even for a no-op theft.
    theft_box: dict[str, str | None] = {}

    original_save_artifact = service.save_artifact

    def stealing_save_artifact(run_id, *, artifact_name, content_type, content):
        result = original_save_artifact(
            run_id, artifact_name=artifact_name, content_type=content_type, content=content
        )
        if artifact_name == "bundle.json":
            # The very last write before complete_run -- steal right after
            # it succeeds, so everything up to and including it has already
            # legitimately happened under this engine's own lease. A live
            # lease can't be reclaimed directly (that's the whole point of
            # claim_run's atomicity), so release it first using the id the
            # engine itself is still holding.
            released = service.release_lease(run_id, lease_id=engine._lease_id)
            assert released is True
            stolen = service.claim_run(run_id, worker_id="rescuer", lease_seconds=60)
            theft_box["lease_id"] = stolen
            assert stolen is not None
        return result

    service.save_artifact = stealing_save_artifact

    final = asyncio.run(engine.execute_run(run["id"]))

    # The positive outcome, asserted from the test's own top-level code so it
    # cannot be short-circuited by execute_run swallowing the in-seam
    # assertion: a real rescuer must actually hold the run's lease now.
    assert theft_box.get("lease_id"), "the rescuer's claim_run must have returned a lease id"
    assert service.holds_lease(run["id"], lease_id=theft_box["lease_id"]) is True, (
        "the rescuer must still hold the run's lease after the displaced "
        "executor's execute_run returns"
    )
    assert final["status"] != "completed"
    names = {a["artifact_name"] for a in service.get_bundle(run["id"])["artifacts"]}
    assert "bundle.json" in names  # the write that triggered the steal did land


def test_a_displaced_executor_cannot_advance_the_runs_phase_to_synthesizing():
    """task-3 report finding 2: the engine fenced artifact writes but not
    run-state writes -- progress and phase updates, checkpoint creation,
    and pause/cancel resolution were not fenced at all, so a displaced
    executor could still advance a run's ``phase`` field (and its
    progress/control_state) on a run it no longer owns.

    Isolates the ONE fence guarding the "synthesizing" phase-advance write
    specifically: the lease is stolen as a side effect of
    ``_check_control("synthesizing")`` itself (a synchronous call with no
    ``await`` point before the phase-advance write that immediately
    follows it in ``_execute_phases``), so no OTHER fence gets a chance to
    catch the theft first -- the same isolation technique the sibling
    lease-theft tests above use for artifact writes, applied here to a
    run-state write instead.

    Mutation-verified (task report): deleting the ``_require_lease()`` call
    immediately before the "synthesizing" ``update_run_progress`` in
    ``_execute_phases`` turns this test red (the phase advances to
    "synthesizing" and synthesis runs); restoring it turns it green.
    """
    service = _make_service()
    search_fn, analyze_fn, calls = _make_pipeline("q")
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    theft_box: dict[str, str | None] = {}
    original_check_control = engine._check_control

    def stealing_check_control(run_id, next_phase):
        if next_phase == "synthesizing":
            # Same technique as the sibling theft tests: release the
            # engine's own still-live lease, then have a third party claim
            # it -- deterministic, no real elapsed time needed.
            released = service.release_lease(run_id, lease_id=engine._lease_id)
            assert released is True, "the engine must still hold its own lease at this point"
            stolen = service.claim_run(run_id, worker_id="rescuer", lease_seconds=60)
            theft_box["lease_id"] = stolen
            assert stolen is not None, "the steal itself must succeed for this test to mean anything"
        return original_check_control(run_id, next_phase)

    engine._check_control = stealing_check_control

    final = asyncio.run(engine.execute_run(run["id"]))

    # Checked from the test's own top-level code, not inside the patched
    # seam, for the same reason the sibling theft tests do this: it cannot
    # be short-circuited by execute_run swallowing an in-seam assertion
    # into some other non-"completed" terminal status.
    assert theft_box.get("lease_id"), "the rescuer's claim_run must have returned a lease id"
    assert service.holds_lease(run["id"], lease_id=theft_box["lease_id"]) is True, (
        "the rescuer must still hold the run's lease after the displaced "
        "executor's execute_run returns"
    )
    assert calls["search"] == 1, "round 1's collecting phase ran legitimately, under a valid lease"
    assert calls["analyze"] == 0, "synthesis must never run once the phase-advance write is blocked"

    current = service.get_run(run["id"])
    assert current["phase"] == "collecting", (
        "the displaced executor must not have advanced the phase past its "
        f"last legitimately-written value; got {current['phase']!r}"
    )
    assert final["status"] != "completed"


def test_lease_lost_while_handling_a_pipeline_error_returns_quietly():
    """task-18060 review finding 3: `_save_ledger` is fenced, and it is
    called from INSIDE the `except ResearchLimitExceeded` and
    `except Exception` handlers in execute_run. A `_LeaseLost` raised from
    inside an except body is NOT caught by that same try's sibling
    `except _LeaseLost` clause (Python only matches a raise against a NEW
    try), so before the fix it propagated out of execute_run entirely --
    callers saw "Local research engine error: execution lease lost" instead
    of the quiet return every other fenced write produces, contradicting
    execute_run's own docstring (which promises only ValueError).

    The lease is left to expire naturally (a real elapsed-time wait,
    deterministic because the wait comfortably exceeds `lease_seconds`) and
    reclaimed from inside `search_fn` on the loop's own thread (routed
    through `_run_on_loop`, matching this suite's established pattern for
    touching the thread-affine `:memory:` connection from an offloaded,
    synchronous search_fn) -- then search_fn raises, landing in
    `except Exception`, exactly where finding 3 lives.
    """
    service = _make_service()
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )
    loop_box: dict[str, asyncio.AbstractEventLoop] = {}

    import time as time_module

    def search_fn(q, params):
        time_module.sleep(0.3)  # comfortably past lease_seconds=0.1 below
        stolen = _run_on_loop(
            loop_box["loop"],
            lambda: service.claim_run(run["id"], worker_id="rescuer", lease_seconds=60),
        )
        assert stolen is not None
        raise RuntimeError("pipeline exploded")

    engine = LocalResearchEngine(service, search_fn=search_fn)
    engine.lease_seconds = 0.1
    engine.keepalive_seconds = 999  # never renews inside the test window

    fail_calls = {"n": 0}
    original_fail_run = service.fail_run

    def spy_fail_run(*args, **kwargs):
        fail_calls["n"] += 1
        return original_fail_run(*args, **kwargs)

    service.fail_run = spy_fail_run

    async def _run():
        loop_box["loop"] = asyncio.get_running_loop()
        return await engine.execute_run(run["id"])  # must not raise _LeaseLost

    final = asyncio.run(_run())

    assert fail_calls["n"] == 0, "a displaced executor must not fail a run it no longer owns"
    assert final["status"] != "failed"


def test_a_resumed_run_continues_its_runtime_budget():
    """task-18060 review finding 4: snapshot() records runtime_elapsed_s but
    from_snapshot dropped it, so `_start_monotonic` reset on every resume
    and a resumed run was granted its whole max_runtime_seconds again -- the
    same leak already fixed for searches, docs, and tokens. A run resumed
    with prior elapsed time already past the budget must fail immediately
    rather than get a fresh clock.
    """
    service = _make_service()
    search_fn, analyze_fn, calls = _make_pipeline("q")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(
        query="q",
        autonomy_mode="autonomous",
        limits_json={"max_runtime_seconds": 5, "max_iterations": 1},
    )
    service.save_artifact(
        run["id"],
        artifact_name="budget_ledger.json",
        content_type="application/json",
        content={
            "limits": {"max_runtime_seconds": 5},
            "searches_used": 0,
            "searches_overshoot": 0,
            "docs_used": 0,
            "tokens_settled": 0,
            "runtime_elapsed_s": 999.0,  # already blew the budget in a "previous" execution
        },
    )

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "failed"
    assert "research_limit_exceeded:max_runtime_seconds" in final["progress_message"]
    assert calls == {"search": 0, "analyze": 0}, "must fail at entry, before any pipeline call"


def test_default_gap_fn_offloads_its_blocking_llm_call():
    """task-18060 review finding 5: `_default_gap_fn` is `async def`, so
    `_offload_pipeline_call` routes it INLINE when it is invoked as the
    engine's gap_fn -- but it used to call `chat_api_call` synchronously,
    directly on the loop thread, inside that coroutine. A gap-analysis call
    longer than `lease_seconds` would starve the keep-alive and lapse the
    lease, exactly the bug finding 1 of task-18060's original review fixed
    for `search_fn`. The blocking call must be offloaded the same way.
    """
    import time as time_module

    service = _make_service()
    search_fn, analyze_fn, _calls = _make_pipeline("q")

    def blocking_chat_api_call(**kwargs):
        time_module.sleep(0.3)
        return "[]"

    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    engine.search_params = {"final_answer_llm": "fake-llm"}
    engine.lease_seconds = 0.1
    engine.keepalive_seconds = 0.02
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    renewals = {"count": 0}
    original_renew_lease = service.renew_lease

    def spy_renew_lease(*args, **kwargs):
        renewals["count"] += 1
        return original_renew_lease(*args, **kwargs)

    service.renew_lease = spy_renew_lease

    with patch(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        side_effect=blocking_chat_api_call,
    ):
        final = asyncio.run(engine.execute_run(run["id"]))

    assert renewals["count"] > 0, (
        "the keep-alive never ran while the blocking gap-analysis LLM call was in flight"
    )
    assert final["status"] == "completed", final.get("progress_message")


def test_a_long_silent_phase_keeps_its_lease():
    """The synthesis phase emits no progress for its whole duration, so a
    lease renewed only by progress events would expire inside it."""
    service = _make_service()
    search_fn, _analyze, _calls = _make_pipeline("q")

    async def slow_analyze(wsr, sqd, params, cancel_event=None):
        await asyncio.sleep(0.3)
        return {
            "final_answer": {"text": "Answer citing [1].", "evidence": [],
                             "confidence": 0.5, "chunks": []},
            "relevant_results": {},
        }

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=slow_analyze
    )
    engine.lease_seconds = 0.1
    engine.keepalive_seconds = 0.02
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed", final.get("progress_message")


def test_a_blocking_sync_search_fn_does_not_starve_the_keepalive():
    """task-18060 review finding 1: in production search_fn resolves to a
    plain `def` running a sequential loop of blocking HTTP calls
    (generate_and_search). _collect_round used to await it inline, so the
    call monopolized the single-threaded event loop for its whole duration
    -- the _keepalive task (an asyncio.sleep-based timer) never got a
    chance to run a single tick until the blocking call returned, by which
    point (with a lease shorter than the block) the lease had already
    expired unrenewed. That reopens the double-execution race the lease
    exists to prevent.

    A synchronous, BLOCKING fake search_fn (time.sleep, not asyncio.sleep)
    is the one legitimate use of a real sleep in this suite, since blocking
    the interpreter is exactly the behaviour under test -- an async fake
    would not reproduce the bug at all (awaiting a real coroutine always
    yields control back to the loop, which is why
    test_a_long_silent_phase_keeps_its_lease above can use asyncio.sleep
    for the analyze_fn side and still pass unmodified).
    """
    import time as time_module

    service = _make_service()
    _search_fn, analyze_fn, _calls = _make_pipeline("q")

    def blocking_search_fn(q, params):
        time_module.sleep(0.3)
        return (
            {
                "results": [{"title": "One", "url": "https://one.example/"}],
                "warnings": [],
            },
            {"sub_questions": [], "main_goal": q},
        )

    engine = LocalResearchEngine(
        service, search_fn=blocking_search_fn, analyze_fn=analyze_fn
    )
    # Short enough that the lease WILL lapse across the 0.3s blocking call
    # unless the keep-alive actually gets to run concurrently with it.
    engine.lease_seconds = 0.1
    engine.keepalive_seconds = 0.02
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    renewals = {"count": 0}
    original_renew_lease = service.renew_lease

    def spy_renew_lease(*args, **kwargs):
        renewals["count"] += 1
        return original_renew_lease(*args, **kwargs)

    service.renew_lease = spy_renew_lease

    final = asyncio.run(engine.execute_run(run["id"]))

    assert renewals["count"] > 0, (
        "the keep-alive never ran while the blocking search was in flight"
    )
    assert final["status"] == "completed", final.get("progress_message")


def test_a_failing_renewal_does_not_break_execute_run_or_strand_the_lease():
    """task-3 review follow-up (keepalive containment): ``renew_lease`` can
    raise for reasons that are NOT lease loss -- e.g. a transient
    ``sqlite3.OperationalError: database is locked`` while another process
    writes the same DB file. The keep-alive task had no exception handling,
    so the exception surfaced at ``await keepalive`` inside ``execute_run``'s
    ``finally`` block, which (a) escaped ``execute_run`` entirely (breaking
    its documented ValueError-only contract) and (b) skipped
    ``release_lease``, stranding the lease in the DB with the run left
    status=running and no fail_run recorded -- nobody owned the failure.

    A renewal ERROR must be treated like a lost lease from the keep-alive's
    perspective (stop renewing, warn) -- the next fence the engine hits
    decides what the run's real state is -- and it must never hijack the
    ``finally`` block's release.
    """
    service = _make_service()
    search_fn, analyze_fn, _calls = _make_pipeline("q")

    async def slow_analyze(wsr, sqd, params, cancel_event=None):
        await asyncio.sleep(0.15)
        return {
            "final_answer": {"text": "Answer citing [1].", "evidence": [],
                             "confidence": 0.5, "chunks": []},
            "relevant_results": {},
        }

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=slow_analyze
    )
    engine.lease_seconds = 5.0
    engine.keepalive_seconds = 0.02
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    def exploding_renew(run_id, *, lease_id, lease_seconds):
        raise RuntimeError("database is locked")

    service.renew_lease = exploding_renew

    final = asyncio.run(engine.execute_run(run["id"]))

    # The run completed on its own merits; the renewal error neither failed
    # it nor escaped execute_run.
    assert final["status"] == "completed", final.get("progress_message")
    # The finally block ran to completion: the lease was released and the
    # engine's lease state cleaned up.
    row = service.get_run(run["id"])
    assert engine._lease_id is None
    assert row["lease_id"] is None, row
    assert row["lease_owner"] is None, row
    assert engine._active_ledger is None


def test_execute_run_releases_the_lease_even_when_a_phase_raises():
    """task-3 review follow-up (finally hardening): with the keep-alive's
    exception contained, the only remaining way to strand a lease is a
    failure INSIDE the finally block itself. ``await keepalive`` must not
    raise the keep-alive's stored exception back into cleanup: retrieval
    via ``exception()`` (or a broad suppress) keeps release_lease
    unconditional.

    Reproduces by making the keep-alive DIE (so its exception is stored on
    the task) while the run itself proceeds fine.
    """
    service = _make_service()
    search_fn, analyze_fn, _calls = _make_pipeline("q")

    async def slow_analyze(wsr, sqd, params, cancel_event=None):
        await asyncio.sleep(0.15)
        return {
            "final_answer": {"text": "Answer citing [1].", "evidence": [],
                             "confidence": 0.5, "chunks": []},
            "relevant_results": {},
        }

    engine = LocalResearchEngine(
        service, search_fn=search_fn, analyze_fn=slow_analyze
    )
    engine.lease_seconds = 5.0
    engine.keepalive_seconds = 0.02
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    # Kill the keep-alive with a non-CancelledError exception, mimicking
    # any unhandled failure inside the task body.
    import asyncio as asyncio_module

    async def dying_keepalive(run_id):
        await asyncio_module.sleep(0.05)
        raise RuntimeError("keepalive exploded")

    engine._keepalive = dying_keepalive  # type: ignore[method-assign]

    final = asyncio.run(engine.execute_run(run["id"]))

    assert final["status"] == "completed", final.get("progress_message")
    row = service.get_run(run["id"])
    assert engine._lease_id is None
    assert row["lease_id"] is None, row
    assert row["lease_owner"] is None, row


def test_a_resumed_run_does_not_get_its_search_budget_back():
    """task-18060: the engine rebuilt the ledger from limits on every entry, so
    a run resumed three times was granted its budget three times."""
    service = _make_service()
    search_fn, analyze_fn, _calls = _make_pipeline("q")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(
        query="q",
        autonomy_mode="autonomous",
        limits_json={"max_searches": 10, "max_iterations": 1},
    )
    asyncio.run(engine.execute_run(run["id"]))

    snapshot = (
        service.get_artifact(run["id"], "budget_ledger.json") or {}
    ).get("content") or {}

    assert int(snapshot.get("searches_used") or 0) > 0, snapshot


def test_the_engine_continues_a_pre_existing_ledger():
    """The decisive test: pre-seed spend, then run. If the engine rebuilds from
    limits it starts at zero and the final snapshot shows only this run's spend;
    if it restores, the snapshot continues from the seeded total.

    Written this way because the obvious version -- call from_snapshot in the
    test and assert it reduces the budget -- passes whether or not the ENGINE
    uses it. Verified by mutation: reverting the engine to from_limits fails
    this test and nothing else.
    """
    service = _make_service()
    search_fn, analyze_fn, _calls = _make_pipeline("q")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(
        query="q",
        autonomy_mode="autonomous",
        limits_json={"max_searches": 10, "max_iterations": 1},
    )
    service.save_artifact(
        run["id"],
        artifact_name="budget_ledger.json",
        content_type="application/json",
        content={
            "limits": {"max_searches": 10},
            "searches_used": 5,
            "searches_overshoot": 0,
            "docs_used": 0,
            "tokens_settled": 0,
        },
    )

    asyncio.run(engine.execute_run(run["id"]))

    snapshot = (
        service.get_artifact(run["id"], "budget_ledger.json") or {}
    ).get("content") or {}

    assert int(snapshot.get("searches_used") or 0) > 5, snapshot


# --- the round's evidence is persisted (task-18060) ----------------------------
# collection_summary.json persists counts, sub-questions and warnings -- not the
# evidence -- so a resumed run re-searched everything it had already paid for.


def test_the_rounds_evidence_pool_is_persisted():
    service = _make_service()
    search_fn, analyze_fn, calls = _make_pipeline("q")
    engine = LocalResearchEngine(service, search_fn=search_fn, analyze_fn=analyze_fn)
    run = service.launch_run(
        query="q", autonomy_mode="autonomous", limits_json={"max_iterations": 1}
    )

    asyncio.run(engine.execute_run(run["id"]))

    pool = (service.get_artifact(run["id"], "evidence_pool.json") or {}).get("content")

    assert pool, "the round's evidence must be persisted"
    assert pool["results"], pool
    assert pool["iteration"] == 1
    assert calls["search"] == 1


def test_evidence_beyond_the_cap_persists_without_content():
    """Bounded explicitly: 66 sources of scraped text is roughly 0.7-3 MB per
    round, so beyond the cap entries keep their references and lose their bodies,
    and the artifact records that it happened."""
    service = _make_service()
    engine = LocalResearchEngine(service)
    engine.EVIDENCE_POOL_CAP_BYTES = 200
    bulky = [
        {
            "url": f"https://e.example/{n}",
            "content": "x" * 500,
            "original_content": "y" * 500,
        }
        for n in range(4)
    ]

    payload = engine._bounded_evidence(bulky, iteration=1)

    assert payload["truncated"] is True
    assert any("content" not in entry for entry in payload["results"])
    assert payload["cap_bytes"] == 200
    # References survive truncation -- a resumed run can re-fetch from them.
    assert all(entry["url"] for entry in payload["results"])


def test_a_pool_within_the_cap_keeps_every_body():
    service = _make_service()
    engine = LocalResearchEngine(service)
    small = [{"url": "https://e.example/1", "content": "short"}]

    payload = engine._bounded_evidence(small, iteration=2)

    assert payload["truncated"] is False
    assert payload["results"][0]["content"] == "short"
    assert payload["iteration"] == 2


def test_a_single_entry_larger_than_the_cap_is_dropped_not_persisted_oversized():
    """task-3 report finding 7: ``_bounded_evidence`` stripped bodies once
    the running total passed the cap, but still appended EVERY entry and
    kept accumulating ``used`` past the cap -- so the persisted artifact
    could exceed ``EVIDENCE_POOL_CAP_BYTES``, and a single entry whose
    reference-only (stripped) size alone exceeds the cap could blow it by
    itself with no way for a reader to tell.

    An entry that still does not fit even stripped of its body is dropped
    from the persisted pool entirely (documented decision: its reference is
    lost for this round, and ``dropped_count`` records how many entries
    this happened to) -- the sum of what IS kept must never exceed the cap.
    """
    service = _make_service()
    engine = LocalResearchEngine(service)
    engine.EVIDENCE_POOL_CAP_BYTES = 50
    # The URL alone (stripped of content/original_content) already exceeds
    # a 50-byte cap.
    huge_reference_entry = {
        "url": "https://e.example/" + ("z" * 200),
        "content": "x" * 10,
    }

    payload = engine._bounded_evidence([huge_reference_entry], iteration=1)

    total_kept_bytes = sum(
        len(json.dumps(entry, sort_keys=True)) for entry in payload["results"]
    )
    assert total_kept_bytes <= engine.EVIDENCE_POOL_CAP_BYTES
    assert payload["results"] == []
    assert payload["truncated"] is True
    assert payload["dropped_count"] == 1


def test_bounded_evidence_never_exceeds_the_cap_with_a_mix_of_sizes():
    """A mix of entries that fit, entries that need stripping, and one
    entry too big even stripped -- the cumulative kept size must stay
    within the cap throughout, not just for the single-entry case above."""
    service = _make_service()
    engine = LocalResearchEngine(service)
    engine.EVIDENCE_POOL_CAP_BYTES = 150
    entries = [
        {"url": "https://e.example/1", "content": "short"},  # fits whole
        {  # needs stripping to fit
            "url": "https://e.example/2",
            "content": "x" * 300,
            "original_content": "y" * 300,
        },
        {  # too big even stripped -- must be dropped
            "url": "https://e.example/3/" + ("z" * 300),
            "content": "x" * 10,
        },
    ]

    payload = engine._bounded_evidence(entries, iteration=1)

    total_kept_bytes = sum(
        len(json.dumps(entry, sort_keys=True)) for entry in payload["results"]
    )
    assert total_kept_bytes <= engine.EVIDENCE_POOL_CAP_BYTES
    assert payload["dropped_count"] >= 1
    assert payload["truncated"] is True


def test_a_non_serializable_entry_is_dropped_not_raised():
    """PR-1822 review follow-up: an entry carrying a non-JSON-native value
    (e.g. a datetime leaked from a provider adapter) raised TypeError
    inside ``_bounded_evidence``'s measurement, failing the WHOLE run via
    the generic exception handler instead of degrading the artifact. The
    docstring's graceful-truncation contract requires the entry be dropped
    and counted, like an oversized one -- the previous fix moved the crash
    from save_artifact into the measurement, it did not remove it.
    """
    import datetime as datetime_module

    service = _make_service()
    engine = LocalResearchEngine(service)
    entries = [
        {"url": "https://e.example/1", "content": "fine"},
        {  # datetime is not JSON-native
            "url": "https://e.example/2",
            "content": "fine too",
            "published": datetime_module.datetime(2026, 8, 18),
        },
        {"url": "https://e.example/3", "content": "also fine"},
    ]

    payload = engine._bounded_evidence(entries, iteration=1)

    assert [e["url"] for e in payload["results"]] == [
        "https://e.example/1",
        "https://e.example/3",
    ]
    assert payload["dropped_count"] == 1
    assert payload["truncated"] is False
