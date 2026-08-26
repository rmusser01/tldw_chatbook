"""web_deep_search tool core (task-1356). Pipeline faked at the two phase
boundaries; pipeline internals are covered by test_deep_search_pipeline.py."""

import asyncio
import time

import pytest

from tldw_chatbook.Tools import web_tool_impls
from tldw_chatbook.Tools.web_tool_impls import LocalToolError, web_deep_search
from tldw_chatbook.Web_Scraping import WebSearch_APIs


_PHASE1 = {"web_search_results_dict": {"results": [{"title": "T", "url": "https://e.com/"}],
                                        "warnings": []},
           "sub_query_dict": {"sub_questions": ["sq1"], "main_goal": "q"}}

_FINAL = {"text": "Deep answer [1].",
          "evidence": [{"id": 1, "url": "https://e.com/", "title": "T",
                        "content": "c", "original_content": "o", "reasoning": "r",
                        "chunk_index": 0}],
          "confidence": 0.78, "chunks": [{}]}


_DEEP_SETTINGS = {"search_provider_default": "google", "relevance_analysis_llm": "openai",
                  "final_answer_llm": "openai", "search_enable_subquery": False,
                  "search_default_max_queries": 5, "search_result_max": 10,
                  "relevance_llm_timeout_s": 30, "relevance_scrape_timeout_s": 30,
                  "deep_search_timeout_s": 240}


@pytest.fixture
def deep_env(monkeypatch):
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", lambda q, p: _PHASE1)

    async def fake_aa(wsr, sqd, params, cancel_event=None):
        return {"final_answer": dict(_FINAL), "relevant_results": {"1": {}},
                "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa)
    # The tool reads config through the module function _deep_search_settings()
    # (returns a dict of the resolved [SearchSettings] values) -- patched wholesale:
    monkeypatch.setattr(web_tool_impls, "_deep_search_settings", lambda: dict(_DEEP_SETTINGS))
    yield


def test_deep_search_happy_path(deep_env):
    out = web_deep_search("what is love")
    assert "Deep answer [1]." in out
    assert "Sources:" in out and "[1] T — https://e.com/" in out
    assert "Confidence: 0.78" in out and "Engine: google" in out


def test_deep_search_no_synthesis_llm_fails_before_spend(deep_env, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search",
                        lambda q, p: calls.__setitem__("n", calls["n"] + 1) or _PHASE1)
    settings = dict(_DEEP_SETTINGS, final_answer_llm="")
    monkeypatch.setattr(web_tool_impls, "_deep_search_settings", lambda: settings)
    with pytest.raises(LocalToolError, match=r"deep-search-failed.*synthesis"):
        web_deep_search("q")
    assert calls["n"] == 0  # nothing spent


def test_deep_search_no_relevance_llm_fails_before_spend(deep_env, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search",
                        lambda q, p: calls.__setitem__("n", calls["n"] + 1) or _PHASE1)
    settings = dict(_DEEP_SETTINGS, relevance_analysis_llm="")
    monkeypatch.setattr(web_tool_impls, "_deep_search_settings", lambda: settings)
    with pytest.raises(LocalToolError, match=r"deep-search-failed.*relevance"):
        web_deep_search("q")
    assert calls["n"] == 0  # nothing spent


def test_deep_search_runs_inside_running_event_loop(deep_env):
    async def call_from_loop():
        return web_deep_search("what is love")

    out = asyncio.run(call_from_loop())
    assert "Deep answer [1]." in out  # loop-safe runner took the thread path


def test_deep_search_invalid_engine(deep_env, monkeypatch):
    with pytest.raises(LocalToolError, match="invalid-args"):
        web_deep_search("q", engine="not-an-engine")


def test_deep_search_invalid_question(deep_env):
    with pytest.raises(LocalToolError, match="invalid-args"):
        web_deep_search("   ")


def test_deep_search_zero_results_after_search(deep_env, monkeypatch):
    empty_phase1 = {
        "web_search_results_dict": {"results": [], "warnings": ["duckduckgo: rate limited"]},
        "sub_query_dict": {"sub_questions": [], "main_goal": "q"},
    }
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", lambda q, p: empty_phase1)
    with pytest.raises(LocalToolError, match=r"deep-search-failed.*search.*no results"):
        web_deep_search("q")


def test_deep_search_zero_relevant_is_not_an_error(deep_env, monkeypatch):
    async def fake_aa_none(wsr, sqd, params, cancel_event=None):
        return {
            "final_answer": {"text": "No relevant results found. Unable to provide an answer.",
                              "evidence": [], "confidence": 0.0, "chunks": []},
            "relevant_results": {},
            "web_search_results_dict": wsr,
        }

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_none)
    out = web_deep_search("what is love")
    assert "[deep-search-failed]" not in out
    assert "sq1" in out  # sub-queries tried are listed
    assert "what is love" in out


def test_deep_search_deadline_sets_cancel_event(deep_env, monkeypatch):
    observed = {}

    async def fake_aa_deadline(wsr, sqd, params, cancel_event=None):
        await asyncio.sleep(0.05)
        observed["cancelled"] = bool(cancel_event and cancel_event.is_set())
        return {"final_answer": dict(_FINAL), "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_deadline)
    settings = dict(_DEEP_SETTINGS, deep_search_timeout_s=0.01)
    monkeypatch.setattr(web_tool_impls, "_deep_search_settings", lambda: settings)

    out = web_deep_search("what is love")
    assert observed.get("cancelled") is True
    assert "Deep answer" in out
    assert "deadline" in out.lower()


def test_deep_search_answer_byte_capped(deep_env, monkeypatch):
    huge_text = "x" * (web_tool_impls.DEEP_SEARCH_ANSWER_MAX_BYTES + 5000)

    async def fake_aa_huge(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, text=huge_text)
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_huge)
    out = web_deep_search("what is love")
    answer_part = out.split("Sources:", 1)[0]
    assert len(answer_part.encode("utf-8")) <= web_tool_impls.DEEP_SEARCH_ANSWER_MAX_BYTES + 64
    assert "truncated" in out


def test_deep_search_sources_capped_at_max(deep_env, monkeypatch):
    evidence = [
        {"id": i, "url": f"https://e.com/{i}", "title": f"T{i}"}
        for i in range(1, web_tool_impls.DEEP_SEARCH_SOURCES_MAX + 10)
    ]

    async def fake_aa_many(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, evidence=evidence)
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_many)
    out = web_deep_search("what is love")
    assert f"[{web_tool_impls.DEEP_SEARCH_SOURCES_MAX}]" in out
    assert f"[{web_tool_impls.DEEP_SEARCH_SOURCES_MAX + 1}]" not in out


def test_deep_search_footer_fallback_note(deep_env, monkeypatch):
    # "fallback" is a DISTINCT field from "generated" (final review, Important
    # 1): "generated" just means "an LLM produced this summary" -- it is NOT
    # a failure signal on its own (the single-chunk skip path also sets it to
    # False without anything having failed). Only "fallback": True (set by
    # the per-chunk except branch) means "summarization failed, truncated
    # raw text was substituted" -- that's what the footer counts.
    async def fake_aa_fallback(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, chunks=[{"generated": False, "fallback": True}, {"generated": True}])
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_fallback)
    out = web_deep_search("what is love")
    assert "fallback" in out.lower()


# --- Important 1 (final review): fallback-field composition -----------------
# "Tool tests fake analyze_and_aggregate with 'chunks': [{}]  so the
# composition [of aggregate_results' real chunk metadata with the footer's
# reading of it] is never exercised" -- these two tests close that gap by
# running the REAL aggregate_results output through the REAL footer code,
# faking only chat_api_call / Summarization_General_Lib.analyze (the
# established seams), not the chunk-metadata shape itself.

_REL_SINGLE_CHUNK = {"1": {"content": "sum one", "original_content": "orig", "reasoning": "r1",
                            "url": "https://one.example/", "title": "One"}}

_REL_MULTI_CHUNK = {
    "a": {"content": "A" * 4000, "reasoning": "ra", "url": "https://a.example/", "title": "A"},
    "b": {"content": "B" * 4000, "reasoning": "rb", "url": "https://b.example/", "title": "B"},
}


def test_deep_search_footer_no_fallback_mention_for_healthy_single_chunk_run(deep_env, monkeypatch):
    """The single-chunk skip path is the MAJORITY, healthiest case
    (synthesis succeeded, nothing failed) -- reproduced bug: it used to be
    reported as '1 chunk(s) used a fallback summary'."""
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", lambda **kwargs: "Deep answer [1].")

    async def fake_aa_real_single_chunk(wsr, sqd, params, cancel_event=None):
        final_answer = WebSearch_APIs.aggregate_results(_REL_SINGLE_CHUNK, "what is love", [], "openai")
        return {"final_answer": final_answer, "relevant_results": _REL_SINGLE_CHUNK,
                "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_real_single_chunk)
    out = web_deep_search("what is love")
    assert "fallback" not in out.lower()


def test_deep_search_footer_counts_exactly_failed_chunks_as_fallback(deep_env, monkeypatch):
    """A genuine per-chunk MAP-phase summarization failure (multi-chunk) must
    be counted, and counted EXACTLY -- not inflated by the healthy chunk
    alongside it."""
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", lambda **kwargs: "Deep answer [1][2].")

    from tldw_chatbook.LLM_Calls import Summarization_General_Lib
    calls = {"n": 0}

    def fake_analyze(*a, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("summarizer down")
        return "chunk summary ok"

    monkeypatch.setattr(Summarization_General_Lib, "analyze", fake_analyze)

    async def fake_aa_real_multi_chunk(wsr, sqd, params, cancel_event=None):
        final_answer = WebSearch_APIs.aggregate_results(_REL_MULTI_CHUNK, "what is love", [], "openai")
        return {"final_answer": final_answer, "relevant_results": _REL_MULTI_CHUNK,
                "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_real_multi_chunk)
    out = web_deep_search("what is love")
    assert "1 chunk(s) used a fallback summary" in out


def test_deep_search_footer_warning_note(deep_env, monkeypatch):
    phase1_with_warnings = {
        "web_search_results_dict": {"results": [{"title": "T", "url": "https://e.com/"}],
                                     "warnings": ["bing: quota exceeded"]},
        "sub_query_dict": {"sub_questions": ["sq1"], "main_goal": "q"},
    }
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", lambda q, p: phase1_with_warnings)
    out = web_deep_search("what is love")
    assert "warning" in out.lower()


def test_deep_search_footer_surfaces_subquery_generation_failure_warning(deep_env, monkeypatch):
    """task-3221: exhausting all 3 paid sub-query-generation attempts must
    leave a trace the user can see -- otherwise it's indistinguishable from
    the feature being off. Passthrough test (phase-boundary fake, like
    test_deep_search_footer_warning_note above): the warning
    generate_and_search appends on total sub-query-generation failure must
    reach the tool's footer's warning count like any other provider
    warning."""
    warning_text = (
        f"sub-query generation failed after "
        f"{WebSearch_APIs._SUBQUERY_GENERATION_MAX_ATTEMPTS} attempts; "
        "searched only the original query"
    )
    phase1_with_subquery_failure = {
        "web_search_results_dict": {"results": [{"title": "T", "url": "https://e.com/"}],
                                     "warnings": [warning_text]},
        "sub_query_dict": {"sub_questions": [], "main_goal": "q"},
    }
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", lambda q, p: phase1_with_subquery_failure)
    out = web_deep_search("what is love")
    assert "1 search warning(s)" in out  # counted through, like any other provider warning


def test_deep_search_phase1_exception_wrapped(deep_env, monkeypatch):
    def boom(q, p):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", boom)
    with pytest.raises(LocalToolError, match=r"deep-search-failed.*search.*provider exploded"):
        web_deep_search("q")


def test_deep_search_phase2_exception_wrapped(deep_env, monkeypatch):
    async def boom(wsr, sqd, params, cancel_event=None):
        raise RuntimeError("analysis exploded")

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", boom)
    with pytest.raises(LocalToolError, match=r"deep-search-failed.*analysis exploded"):
        web_deep_search("q")


def test_deep_search_places_timeouts_into_search_params(deep_env, monkeypatch):
    """CRITICAL handoff: the pipeline reads relevance_llm_timeout_s /
    relevance_scrape_timeout_s from search_params -- the tool must place the
    config timeout values INTO search_params explicitly."""
    seen = {}

    def fake_generate(q, p):
        seen["search_params"] = dict(p)
        return _PHASE1

    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", fake_generate)
    web_deep_search("q")
    assert seen["search_params"].get("relevance_llm_timeout_s") == 30
    assert seen["search_params"].get("relevance_scrape_timeout_s") == 30


def test_deep_search_places_respect_robots_txt_into_search_params(deep_env, monkeypatch):
    """task-3260: the tool must place the real, configured
    [webfetch] respect_robots_txt setting into search_params -- the
    pydantic-safe channel analyze_and_aggregate/search_result_relevance
    read it from (mirrors the timeouts plumbing above)."""
    monkeypatch.setattr(
        web_tool_impls, "_webfetch_settings", lambda: {"respect_robots_txt": True}
    )
    seen = {}

    def fake_generate(q, p):
        seen["search_params"] = dict(p)
        return _PHASE1

    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", fake_generate)
    web_deep_search("q")
    assert seen["search_params"].get("respect_robots_txt") is True


def test_deep_search_places_respect_robots_txt_false_into_search_params(deep_env, monkeypatch):
    """Same seam, opposite value -- proves this isn't hardcoded True."""
    monkeypatch.setattr(
        web_tool_impls, "_webfetch_settings", lambda: {"respect_robots_txt": False}
    )
    seen = {}

    def fake_generate(q, p):
        seen["search_params"] = dict(p)
        return _PHASE1

    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", fake_generate)
    web_deep_search("q")
    assert seen["search_params"].get("respect_robots_txt") is False


def test_deep_search_places_max_queries_cap_into_search_params(deep_env, monkeypatch):
    """Important 2 (final review): search_default_max_queries is already
    resolved by _deep_search_settings() but was never handed to the
    pipeline -- generate_and_search reads it from search_params (pydantic-safe:
    the same route the timeouts use), so the tool must place it there."""
    seen = {}

    def fake_generate(q, p):
        seen["search_params"] = dict(p)
        return _PHASE1

    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", fake_generate)
    web_deep_search("q")
    assert seen["search_params"].get("search_default_max_queries") == 5


def test_deep_search_places_phase1_time_budget_into_search_params(deep_env, monkeypatch):
    """Important 3a (final review): the tool computes its remaining phase-1
    budget from deep_search_timeout_s at entry and hands it to
    generate_and_search via search_params -- checked at entry, so with
    (almost) no elapsed time yet it should equal the configured timeout."""
    seen = {}

    def fake_generate(q, p):
        seen["search_params"] = dict(p)
        return _PHASE1

    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", fake_generate)
    web_deep_search("q")
    assert seen["search_params"].get("phase1_time_budget_s") == pytest.approx(240, abs=2)


# --- Fix-round: deadline-before-first-relevant honesty (CRITICAL) -----------

def test_deep_search_deadline_before_first_relevant_is_honest(deep_env, monkeypatch):
    """A watchdog firing before ANY result is scored must not report
    "Analyzed 40 result(s)" (zero were analyzed) or advise rephrasing (the
    cause was a timeout, not the query) -- both lies steer a second
    full-price run. Reviewer's repro shape: cancel-at-top-of-loop fake,
    tiny timeout, 40 results."""
    many_results = [{"title": f"T{i}", "url": f"https://e.com/{i}"} for i in range(40)]
    phase1_many = {
        "web_search_results_dict": {"results": many_results, "warnings": []},
        "sub_query_dict": {"sub_questions": ["sq1"], "main_goal": "q"},
    }
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", lambda q, p: phase1_many)

    async def fake_aa_cancel_at_top(wsr, sqd, params, cancel_event=None):
        await asyncio.sleep(0.05)  # let the watchdog fire before we check
        if cancel_event and cancel_event.is_set():
            return {
                "final_answer": {"text": "", "evidence": [], "confidence": 0.0, "chunks": []},
                "relevant_results": {},
                "web_search_results_dict": wsr,
            }
        raise AssertionError("test setup bug: cancel_event never fired")

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_cancel_at_top)
    settings = dict(_DEEP_SETTINGS, deep_search_timeout_s=0.01)
    monkeypatch.setattr(web_tool_impls, "_deep_search_settings", lambda: settings)

    out = web_deep_search("what is love")
    assert "[deep-search-failed]" not in out  # still a normal (non-error) return
    assert "40" in out  # honestly reports what phase 1 found
    assert "deadline" in out.lower()
    assert "analyzed 40" not in out.lower()  # must not claim full coverage
    assert "try rephrasing" not in out.lower()  # wrong diagnosis for a timeout


# --- Fix-round: typed [SearchSettings] coercion (IMPORTANT) -----------------

def test_deep_search_settings_malformed_timeout_falls_back_without_crashing(monkeypatch):
    import tldw_chatbook.config as config_module

    def fake_get_cli_setting(section, key=None, default=None):
        if key == "deep_search_timeout_s":
            return "abc"  # malformed: not float()-able
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    settings = web_tool_impls._deep_search_settings()
    assert settings["deep_search_timeout_s"] == 240  # default, no crash


def test_deep_search_settings_quoted_false_string_does_not_enable_subquery(monkeypatch):
    import tldw_chatbook.config as config_module

    def fake_get_cli_setting(section, key=None, default=None):
        if key == "search_enable_subquery":
            return "false"  # a STRING; bool("false") is True in plain Python
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    settings = web_tool_impls._deep_search_settings()
    assert settings["search_enable_subquery"] is False


def test_deep_search_settings_negative_timeout_falls_back_to_default(monkeypatch):
    import tldw_chatbook.config as config_module

    def fake_get_cli_setting(section, key=None, default=None):
        if key == "relevance_llm_timeout_s":
            return -5
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    settings = web_tool_impls._deep_search_settings()
    assert settings["relevance_llm_timeout_s"] == 30


# --- Fix-round: same-region minors -------------------------------------------

def test_deep_search_invalid_config_engine_default_names_config_key(deep_env, monkeypatch):
    """An invalid ENGINE ARGUMENT is [invalid-args] (caller's mistake); an
    invalid CONFIG default must not blame the caller's (absent) argument."""
    settings = dict(_DEEP_SETTINGS, search_provider_default="not-a-real-engine")
    monkeypatch.setattr(web_tool_impls, "_deep_search_settings", lambda: settings)
    with pytest.raises(LocalToolError) as excinfo:
        web_deep_search("q")  # no engine argument supplied
    msg = str(excinfo.value)
    assert "[invalid-args]" not in msg
    assert "deep-search-failed" in msg
    assert "search_provider_default" in msg


def test_deep_search_malformed_phase1_result_is_structured_error(deep_env, monkeypatch):
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", lambda q, p: {"oops": True})
    with pytest.raises(LocalToolError, match=r"deep-search-failed.*search.*malformed"):
        web_deep_search("q")


def test_deep_search_non_numeric_confidence_does_not_crash(deep_env, monkeypatch):
    async def fake_aa_bad_confidence(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, confidence="high")
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_bad_confidence)
    out = web_deep_search("what is love")
    assert "Confidence: 0.00" in out


def test_deep_search_footer_uses_relevant_scored_wording(deep_env):
    out = web_deep_search("what is love")
    assert "Relevant: 1 of 1 scored" in out
    assert "Analyzed" not in out  # replaced -- K was the relevant count, not an analyzed count


# --- Fix round 2: N4 -- a genuine TOTAL budget (answer + sources + footer) --

def test_deep_search_sources_block_is_byte_capped(deep_env, monkeypatch):
    """A genuinely large answer (consuming essentially the whole answer
    cap) PLUS 20 long-titled sources: exercises the real total-budget
    interaction (task-1356 review round 2, N4) -- previously the Sources
    block had its OWN independent 24KB budget on top of a 16KB answer cap,
    so the combined output could exceed what the agent runtime's
    head-first tool-result truncation actually preserves (16,000 chars).
    Uses a large answer so this assertion tests the TOTAL bound it names,
    not just the sources sub-budget in isolation."""
    huge_text = "A" * (web_tool_impls.DEEP_SEARCH_ANSWER_MAX_BYTES + 5000)
    long_title = "T" * 5000
    evidence = [
        {"id": i, "url": f"https://e.com/{i}", "title": long_title}
        for i in range(1, web_tool_impls.DEEP_SEARCH_SOURCES_MAX + 1)
    ]

    async def fake_aa_large_answer_and_titles(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, text=huge_text, evidence=evidence)
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_large_answer_and_titles)
    out = web_deep_search("what is love")

    total_bytes = len(out.encode("utf-8"))
    slack = 1024  # omission-marker text overhead
    assert total_bytes <= web_tool_impls.DEEP_SEARCH_TOTAL_MAX_BYTES + slack, (
        f"total output was {total_bytes} bytes"
    )
    assert "Confidence:" in out  # footer always survives
    assert "truncated" in out.lower()  # the answer was genuinely capped


def test_deep_search_sources_omission_marker_when_budget_exceeded(deep_env, monkeypatch):
    """A large answer plus 20 sources with BOTH an oversized title AND an
    oversized URL (URL truncation is new in this round -- N4): the sources
    budget is squeezed tight enough by the answer that most sources can't
    fit even after per-field truncation, so the size-cap omission marker
    must fire honestly rather than the block growing unbounded."""
    huge_text = "A" * (web_tool_impls.DEEP_SEARCH_ANSWER_MAX_BYTES + 5000)
    long_url_tail = "x" * 3000
    evidence = [
        {"id": i, "url": f"https://e.com/{long_url_tail}-{i}", "title": "T" * 5000}
        for i in range(1, web_tool_impls.DEEP_SEARCH_SOURCES_MAX + 1)
    ]

    async def fake_aa_large_answer_and_long_fields(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, text=huge_text, evidence=evidence)
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_large_answer_and_long_fields)
    out = web_deep_search("what is love")

    total_bytes = len(out.encode("utf-8"))
    assert total_bytes <= web_tool_impls.DEEP_SEARCH_TOTAL_MAX_BYTES + 1024, (
        f"total output was {total_bytes} bytes"
    )
    assert "Confidence:" in out  # footer survives
    assert "size cap reached" in out.lower()  # size-cap omission marker present
    assert "[1]" in out  # at least the first (newest-relevance) source made it in


def test_deep_search_sources_url_truncated(deep_env, monkeypatch):
    """A pathologically long URL alone (short title, tiny answer -- plenty
    of budget otherwise) must still be truncated per-line (~500B, new in
    this round), not carried through verbatim."""
    long_url = "https://e.com/" + ("x" * 3000)
    evidence = [{"id": 1, "url": long_url, "title": "T"}]

    async def fake_aa_one_long_url(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, evidence=evidence)
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_one_long_url)
    out = web_deep_search("what is love")
    assert long_url not in out
    assert "truncated" in out.lower()
    assert "[1] T" in out


def test_deep_search_sources_count_cap_marker(deep_env, monkeypatch):
    """More evidence than DEEP_SEARCH_SOURCES_MAX allows even considering
    must leave an honest count-cap marker -- distinct from the size-cap
    marker -- rather than silently dropping the excess with no signal."""
    evidence = [
        {"id": i, "url": f"https://e.com/{i}", "title": f"T{i}"}
        for i in range(1, web_tool_impls.DEEP_SEARCH_SOURCES_MAX + 6)  # 5 over the count cap
    ]

    async def fake_aa_over_count_cap(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, evidence=evidence)
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_over_count_cap)
    out = web_deep_search("what is love")

    for i in range(1, web_tool_impls.DEEP_SEARCH_SOURCES_MAX + 1):
        assert f"[{i}]" in out
    assert f"[{web_tool_impls.DEEP_SEARCH_SOURCES_MAX + 1}]" not in out
    assert "count cap reached" in out.lower()
    assert "5" in out  # exactly 5 were dropped by the count cap


def test_deep_search_sources_single_oversized_line_still_emits_marker(deep_env, monkeypatch):
    """When the sources budget is squeezed so tight that even the FIRST
    line can't fit, the block must still say so -- not silently render as
    "Sources: (none)" as if no evidence existed at all. Forced by raising
    the answer cap close to the total and supplying an answer that actually
    fills it, leaving ~0 bytes for sources -- less than even one line."""
    raised_cap = web_tool_impls.DEEP_SEARCH_TOTAL_MAX_BYTES - 100  # leaves less than one source line
    monkeypatch.setattr(web_tool_impls, "DEEP_SEARCH_ANSWER_MAX_BYTES", raised_cap)
    huge_text = "A" * (raised_cap + 5000)
    evidence = [{"id": 1, "url": "https://e.com/1", "title": "T"}]

    async def fake_aa_tiny_budget(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, text=huge_text, evidence=evidence)
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_tiny_budget)
    out = web_deep_search("what is love")
    assert "Sources: (none)" not in out
    assert "size cap reached" in out.lower()
    assert "[1]" not in out  # the one source genuinely didn't fit
    assert "Confidence:" in out  # footer still survives


# --- Fix-round: backstop must hold even when a pipeline call blocks the loop

def test_deep_search_backstop_holds_when_pipeline_blocks_the_loop(deep_env, monkeypatch):
    """Even a misbehaving pipeline call that blocks the event loop
    synchronously (no yield) must not make the tool hang past its deadline
    when invoked from inside an already-running loop: only the loop-safe
    runner's cross-thread thread.join() backstop can preempt that --
    asyncio.wait_for cannot, since it needs the blocked coroutine to yield
    control back to fire its own timeout callback."""
    monkeypatch.setattr(web_tool_impls, "_DEEP_SEARCH_DEADLINE_GRACE_S", 0.05)
    monkeypatch.setattr(web_tool_impls, "_DEEP_SEARCH_THREAD_JOIN_SLACK_S", 0.05)

    async def fake_aa_blocks_loop(wsr, sqd, params, cancel_event=None):
        time.sleep(0.5)  # blocks the event loop thread synchronously
        return {"final_answer": dict(_FINAL), "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_blocks_loop)
    settings = dict(_DEEP_SETTINGS, deep_search_timeout_s=0.01)
    monkeypatch.setattr(web_tool_impls, "_deep_search_settings", lambda: settings)

    async def call_from_loop():
        return web_deep_search("what is love")

    start = time.monotonic()
    with pytest.raises(LocalToolError, match=r"deep-search-failed.*timeout"):
        asyncio.run(call_from_loop())
    elapsed = time.monotonic() - start
    assert elapsed < 0.4, f"backstop did not cut in before the 0.5s block finished (took {elapsed:.2f}s)"


# --- Fix round 2: N1 -- empty-string config value must not fake a default --

def test_deep_search_empty_string_provider_still_blocks_spend(tmp_path, monkeypatch):
    """[SearchSettings] relevance_analysis_llm = "" in REAL TOML must still
    trip spend-check-before-spend, not silently resolve to the "openai"
    default and let a probe call a provider the user never named. Uses the
    real config-loading seam (no wholesale _deep_search_settings monkeypatch)
    so this exercises _str()'s actual coercion, not a test double of it."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[general]\nusers_name = 'test'\n"
        "[SearchSettings]\n"
        'relevance_analysis_llm = ""\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    calls = {"n": 0}

    def fake_generate(q, p):
        calls["n"] += 1
        return _PHASE1

    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", fake_generate)

    with pytest.raises(LocalToolError, match=r"deep-search-failed.*relevance"):
        web_deep_search("q")
    assert calls["n"] == 0  # nothing spent


# --- Fix round 2: N2 -- deadline message must not claim an unknowable count

def test_deep_search_deadline_message_makes_no_scored_count_claim(deep_env, monkeypatch):
    """The pipeline exposes no "how many were scored before cancellation"
    signal -- a watchdog firing partway through the loop (some results
    genuinely examined, just none proved relevant) must get a message that
    claims neither zero nor any other specific scored count, and advice
    that covers both worlds (too little time vs. genuinely no matches)."""
    many_results = [{"title": f"T{i}", "url": f"https://e.com/{i}"} for i in range(40)]
    phase1_many = {
        "web_search_results_dict": {"results": many_results, "warnings": []},
        "sub_query_dict": {"sub_questions": ["sq1"], "main_goal": "q"},
    }
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", lambda q, p: phase1_many)

    async def fake_aa_deadline_midloop(wsr, sqd, params, cancel_event=None):
        await asyncio.sleep(0.05)  # let the watchdog fire mid-"loop"
        return {
            "final_answer": {"text": "", "evidence": [], "confidence": 0.0, "chunks": []},
            "relevant_results": {},
            "web_search_results_dict": wsr,
        }

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_deadline_midloop)
    settings = dict(_DEEP_SETTINGS, deep_search_timeout_s=0.01)
    monkeypatch.setattr(web_tool_impls, "_deep_search_settings", lambda: settings)

    out = web_deep_search("what is love")
    assert "none were scored in time" not in out.lower()  # false zero-claim removed
    assert "unknown number" in out.lower()
    assert "longer" in out.lower() and "deep_search_timeout_s" in out
    assert "rephrasing may help" in out.lower()


def test_deep_search_footer_deadline_note_says_may_be_incomplete(deep_env, monkeypatch):
    """A run whose watchdog fires while a call is STILL genuinely in
    progress, yet the call still goes on to complete fully and
    successfully, must not get flagged as definitely "partial" -- the code
    only knows the deadline was reached, not whether that cost anything
    (the b2 probe: a fully-completed run flagged as partial)."""
    async def fake_aa_completes_anyway(wsr, sqd, params, cancel_event=None):
        await asyncio.sleep(0.05)  # give the tiny-timeout watchdog a chance to fire
        return {"final_answer": dict(_FINAL), "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_completes_anyway)
    settings = dict(_DEEP_SETTINGS, deep_search_timeout_s=0.01)  # fires well before the 0.05s sleep ends
    monkeypatch.setattr(web_tool_impls, "_deep_search_settings", lambda: settings)

    out = web_deep_search("what is love")
    assert "Deep answer [1]." in out  # the run DID fully succeed
    assert "deadline reached: partial synthesis" not in out.lower()
    assert "deadline reached" in out.lower() and "may be incomplete" in out.lower()


def test_deep_search_footer_discloses_gate_fallback(deep_env, monkeypatch):
    final_answer = dict(_FINAL)
    final_answer["gate"] = {"relevant": 3, "raw": 5, "fallback": True}
    final_answer["evidence"] = [
        {"id": 1, "url": "https://e.com/", "title": "T", "content": "c",
         "original_content": "o", "reasoning": "gate fallback", "chunk_index": 0,
         "gate_unverified": True},
    ]

    async def fake_aa(wsr, sqd, params, cancel_event=None):
        return {"final_answer": final_answer, "relevant_results": {"1": {}},
                "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa)
    out = web_deep_search("what is love")
    assert "not relevance-verified" in out


# --- shared pipeline param assembly (task-16484) ----------------------------------

def test_deep_search_pipeline_params_shape_and_overrides():
    from tldw_chatbook.Tools.web_tool_impls import deep_search_pipeline_params

    params = deep_search_pipeline_params()

    for key in (
        "engine", "relevance_analysis_llm", "final_answer_llm",
        "relevance_llm_timeout_s", "relevance_scrape_timeout_s",
        "search_default_max_queries", "result_count", "subquery_generation",
        "phase1_time_budget_s", "respect_robots_txt",
    ):
        assert key in params, key

    bounded = deep_search_pipeline_params(
        engine="duckduckgo", max_results=3, subquery=False, max_queries=1
    )
    assert bounded["engine"] == "duckduckgo"
    assert bounded["result_count"] == 3
    assert bounded["subquery_generation"] is False
    assert bounded["search_default_max_queries"] == 1
