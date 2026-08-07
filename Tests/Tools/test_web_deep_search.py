"""web_deep_search tool core (task-1356). Pipeline faked at the two phase
boundaries; pipeline internals are covered by test_deep_search_pipeline.py."""

import asyncio

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
                  "deep_search_timeout_s": 300}


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
    async def fake_aa_fallback(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, chunks=[{"generated": False}, {"generated": True}])
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_fallback)
    out = web_deep_search("what is love")
    assert "fallback" in out.lower()


def test_deep_search_footer_warning_note(deep_env, monkeypatch):
    phase1_with_warnings = {
        "web_search_results_dict": {"results": [{"title": "T", "url": "https://e.com/"}],
                                     "warnings": ["bing: quota exceeded"]},
        "sub_query_dict": {"sub_questions": ["sq1"], "main_goal": "q"},
    }
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", lambda q, p: phase1_with_warnings)
    out = web_deep_search("what is love")
    assert "warning" in out.lower()


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
