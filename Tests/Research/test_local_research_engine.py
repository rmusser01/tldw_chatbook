"""Local research execution engine (task-16322, ADR-068).

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
    run = service.launch_run(query="Cancel me mid-run", autonomy_mode="autonomous")

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
