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


# --- Fix-round: total Sources-block byte budget (IMPORTANT) ------------------

def test_deep_search_sources_block_is_byte_capped(deep_env, monkeypatch):
    """20 long-titled sources previously reproduced a ~400KB result even
    with DEEP_SEARCH_SOURCES_MAX already bounding the source COUNT. With
    per-title truncation (~200 bytes) applied, 20 titles land comfortably
    under the 24KB Sources budget on their own -- so this specific
    reproduction is fully absorbed by title truncation alone (no source
    needs to be dropped); the separate byte-budget+omission-marker
    mechanism is exercised by a long-URL scenario below, since the title
    cap does not touch the URL field."""
    long_title = "T" * 5000
    evidence = [
        {"id": i, "url": f"https://e.com/{i}", "title": long_title}
        for i in range(1, web_tool_impls.DEEP_SEARCH_SOURCES_MAX + 1)
    ]

    async def fake_aa_long_titles(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, evidence=evidence)
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_long_titles)
    out = web_deep_search("what is love")

    total_bytes = len(out.encode("utf-8"))
    slack = 4 * 1024  # answer cap + footer + omission-marker overhead
    assert total_bytes <= web_tool_impls.DEEP_SEARCH_TOTAL_MAX_BYTES + slack, (
        f"total output was {total_bytes} bytes"
    )
    assert "Confidence:" in out  # footer survives
    # None of the 20 (comfortably small once titles are truncated) needed
    # to be dropped -- all are present, proving title truncation alone
    # already defeats this specific "long titles" reproduction.
    for i in range(1, web_tool_impls.DEEP_SEARCH_SOURCES_MAX + 1):
        assert f"[{i}]" in out


def test_deep_search_sources_omission_marker_when_budget_exceeded(deep_env, monkeypatch):
    """Title truncation does not touch the URL field -- a pathologically
    long URL must still trip the total Sources-block byte budget and leave
    an honest omission marker rather than growing the block unbounded."""
    long_url_tail = "x" * 3000
    evidence = [
        {"id": i, "url": f"https://e.com/{long_url_tail}-{i}", "title": f"T{i}"}
        for i in range(1, web_tool_impls.DEEP_SEARCH_SOURCES_MAX + 1)
    ]

    async def fake_aa_long_urls(wsr, sqd, params, cancel_event=None):
        final = dict(_FINAL, evidence=evidence)
        return {"final_answer": final, "relevant_results": {"1": {}}, "web_search_results_dict": wsr}

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa_long_urls)
    out = web_deep_search("what is love")

    total_bytes = len(out.encode("utf-8"))
    assert total_bytes <= web_tool_impls.DEEP_SEARCH_TOTAL_MAX_BYTES + 4096, (
        f"total output was {total_bytes} bytes"
    )
    assert "Confidence:" in out  # footer survives
    assert "omitted" in out.lower()  # omission marker present
    assert "[1]" in out  # at least the first (newest-relevance) source made it in


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
