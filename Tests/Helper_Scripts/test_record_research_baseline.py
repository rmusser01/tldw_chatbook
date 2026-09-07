"""Decomposition controls on the live baseline recorder (task-17370).

Every gate number recorded so far (the 0.29 repositories baseline and the
0.42 source-type-note re-measurement) came from runs with sub-query fan-out
and gap-driven replanning both switched OFF -- hard-coded constants, not
choices. These tests pin the two things that matter once they become flags:
the DEFAULT invocation must stay byte-identical to the spend-bounded runs
that produced the recorded baselines, and the flags must actually reach the
pipeline params and the run's limits.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from Helper_Scripts.Benchmarks import record_research_baseline as recorder


_LLM_SETTINGS = {
    "relevance_analysis_llm": "llama_cpp",
    "final_answer_llm": "llama_cpp",
    "search_provider_default": "duckduckgo",
    "search_enable_subquery": True,  # config says ON: the recorder must still bound it
    "search_default_max_queries": 5,
    "search_result_max": 20,
    "deep_search_timeout_s": 240,
    "relevance_llm_timeout_s": 30,
    "relevance_scrape_timeout_s": 30,
}


@pytest.fixture()
def stub_settings(monkeypatch):
    """Neutralize config so the params assembly is the only variable."""
    from tldw_chatbook.Tools import web_tool_impls

    monkeypatch.setattr(web_tool_impls, "_deep_search_settings", lambda: dict(_LLM_SETTINGS))
    monkeypatch.setattr(web_tool_impls, "_webfetch_settings", lambda: {"respect_robots_txt": True})
    return _LLM_SETTINGS


def test_default_params_stay_spend_bounded(stub_settings):
    """The recorded baselines' configuration, even with config subquery ON."""
    params = recorder._build_search_params(5, engine_override="duckduckgo")

    assert params["subquery_generation"] is False
    assert params["search_default_max_queries"] == 1


def test_max_queries_above_one_enables_decomposition(stub_settings):
    params = recorder._build_search_params(5, engine_override="duckduckgo", max_queries=4)

    assert params["subquery_generation"] is True
    assert params["search_default_max_queries"] == 4


def test_max_queries_of_one_keeps_fan_out_off(stub_settings):
    """1 total query means zero sub-queries -- generating them would be spend
    with nowhere to go, so the flag must not half-enable the feature."""
    params = recorder._build_search_params(5, engine_override="duckduckgo", max_queries=1)

    assert params["subquery_generation"] is False
    assert params["search_default_max_queries"] == 1


def test_default_leaves_the_configured_deadline_untouched(stub_settings):
    params = recorder._build_search_params(5, engine_override="duckduckgo")

    assert params["deep_search_timeout_s"] == 240
    assert params["phase1_time_budget_s"] == 240


def test_deadline_override_reaches_both_budget_keys(stub_settings):
    """A deadline calibrated for one-query runs would truncate a fan-out run
    mid-gate, measuring the deadline instead of the gate."""
    params = recorder._build_search_params(5, engine_override="duckduckgo", deadline_s=1800)

    assert params["deep_search_timeout_s"] == 1800
    assert params["phase1_time_budget_s"] == 1800


class _FakeService:
    def __init__(self):
        self.launch_kwargs = None

    def launch_run(self, **kwargs):
        self.launch_kwargs = kwargs
        return {"id": "run-1"}

    def get_artifact(self, _run_id, _name):
        return {"content": {"citation_verification": {"markers_total": 1, "markers_resolved": 1}}}


class _FakeEngine:
    async def execute_run(self, _run_id):
        return {"status": "completed"}


def test_run_question_defaults_to_a_single_pass():
    service, engine = _FakeService(), _FakeEngine()

    payload = asyncio.run(recorder._run_question(engine, service, "Q"))

    assert payload["citation_verification"]["markers_resolved"] == 1
    assert (service.launch_kwargs["limits_json"] or {}).get("max_iterations", 1) == 1


def test_run_question_carries_the_requested_iteration_bound():
    service, engine = _FakeService(), _FakeEngine()

    asyncio.run(recorder._run_question(engine, service, "Q", max_iterations=3))

    assert service.launch_kwargs["limits_json"]["max_iterations"] == 3


def test_emitted_aggregate_records_the_decomposition_settings():
    """A recorded result must never be readable without knowing whether
    decomposition was on -- that ambiguity is exactly what made the 0.42
    residual unfalsifiable."""
    aggregate = recorder._decorate_aggregate(
        {"sample_count": 2.0, "gate_pass_rate": 0.42},
        args=SimpleNamespace(max_queries=4, max_iterations=2, deadline_s=1800),
    )

    assert aggregate["gate_pass_rate"] == 0.42
    assert aggregate["decomposition"] == {
        "max_queries": 4,
        "max_iterations": 2,
        "deadline_s": 1800,
        # Absent on this Namespace: recorded as None rather than omitted, so a
        # reader can tell "ran at the configured default" from "not recorded".
        "llm_timeout_s": None,
    }


# --- per-call LLM timeout (task-17370, after the task-17382 measurement) ------
# Every per-result summarization in the live arms failed at exactly 30.0s --
# the shipped relevance_llm_timeout_s -- so the pipeline fell back to raw
# source content and NO recorded baseline has ever measured the pipeline with
# summaries actually completing. A baseline cannot investigate that while the
# timeout is fixed at whatever the config happens to hold.


def test_default_leaves_the_configured_llm_timeout_untouched(stub_settings):
    params = recorder._build_search_params(5, engine_override="duckduckgo")

    assert params["relevance_llm_timeout_s"] == 30


def test_llm_timeout_override_reaches_the_pipeline(stub_settings):
    params = recorder._build_search_params(
        5, engine_override="duckduckgo", llm_timeout_s=240
    )

    assert params["relevance_llm_timeout_s"] == 240.0


def test_emitted_aggregate_records_the_llm_timeout():
    """A recorded result must state the timeout it ran under, for the same
    reason it states the decomposition settings."""
    import argparse

    args = argparse.Namespace(
        max_queries=3, max_iterations=2, deadline_s=900.0, llm_timeout_s=240.0
    )
    out = recorder._decorate_aggregate({"gate_pass_rate": 0.5}, args=args)

    assert out["decomposition"]["llm_timeout_s"] == 240.0


# --- unscored runs must be counted, not dropped (Qodo, PR 1782) ---------------
# Recording synthesis_failed on the RUN was only half the fix: the recorder
# still returned None for any payload without citation_verification, so a run
# that produced no report never reached the aggregate at all and the means were
# computed over survivors -- the exact distortion task-17386 exists to stop.


def test_a_run_without_a_report_is_counted_not_dropped():
    """The payload has no metrics to average, but the aggregate must say it
    happened, and say which stage failed."""
    payload = {
        "synthesis_failed": {
            "stage": "synthesis",
            "error_type": "ReadTimeoutError",
            "evidence_count": 46,
            "chunk_count": 6,
        }
    }

    class _Service:
        def launch_run(self, **kwargs):
            return {"id": "run-1"}

        def get_artifact(self, run_id, name):
            return {"content": payload}

    class _Engine:
        async def execute_run(self, run_id):
            return {"id": run_id, "status": "completed"}

    out = asyncio.run(recorder._run_question(_Engine(), _Service(), "Q"))

    assert out is not None, "a failed-synthesis run must not be dropped"
    assert out["__unscored__"]["stage"] == "synthesis"
    assert out["__unscored__"]["error_type"] == "ReadTimeoutError"


def test_aggregate_states_how_many_runs_produced_no_report():
    aggregate = recorder._decorate_aggregate(
        {"sample_count": 2.0, "gate_pass_rate": 0.5},
        args=SimpleNamespace(
            max_queries=3, max_iterations=2, deadline_s=1200, llm_timeout_s=None
        ),
        unscored_runs=[{"stage": "synthesis", "error_type": "ReadTimeoutError"}],
    )

    assert aggregate["sample_count"] == 2.0
    assert aggregate["unscored_runs"]["count"] == 1
    assert aggregate["unscored_runs"]["reasons"][0]["stage"] == "synthesis"
