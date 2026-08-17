"""Deep-search pipeline hardening (task-1356). Fakes live ONLY at chat_api_call /
scrape_article / Summarization analyze — the pipeline code runs real."""

import asyncio
import concurrent.futures
import json
import socket
import threading
import time

import httpx
import pytest

from tldw_chatbook.Tools import web_tool_impls
from tldw_chatbook.Web_Scraping import WebSearch_APIs
from tldw_chatbook import config as config_module


def _fake_chat(responses):
    """Return a chat_api_call stand-in popping canned string responses."""
    queue = list(responses)

    def fake(**kwargs):
        return queue.pop(0) if queue else queue_underflow(kwargs)

    def queue_underflow(kwargs):
        raise AssertionError(f"unexpected extra chat_api_call: {kwargs.get('messages_payload')!r:.120}")

    return fake


def _std_result(title, url, content):
    return {"title": title, "url": url, "content": content,
            "metadata": {"snippet": content, "date_published": None, "author": None,
                          "source": None, "language": None, "relevance_score": None}}


# --- _sanitize_sub_questions -------------------------------------------------

def test_sanitize_normalizes_and_dedupes():
    raw = ["  Alpha?  ", {"sub_question": "beta"}, "ALPHA?", "", None, "gamma"]
    out = WebSearch_APIs._sanitize_sub_questions(raw)
    assert out == ["Alpha?", "beta", "gamma"]


def test_sanitize_accepts_dict_shapes():
    assert WebSearch_APIs._sanitize_sub_questions({"sub_questions": ["a", "b"]}) == ["a", "b"]
    assert WebSearch_APIs._sanitize_sub_questions({"search_queries": ["c"]}) == ["c"]
    assert WebSearch_APIs._sanitize_sub_questions(None) == []


# --- analyze_question fallback ----------------------------------------------

def test_analyze_question_total_failure_falls_back_to_empty(monkeypatch):
    def always_garbage(**kwargs):
        return "not json and no quoted strings here"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", always_garbage)
    out = WebSearch_APIs.analyze_question("what is love", "openai")
    assert out["sub_questions"] == []  # NOT [original question]


# --- generate_and_search warnings --------------------------------------------

def _search_params(**over):
    base = {"engine": "google", "content_country": "US", "search_lang": "en",
            "output_lang": "en", "result_count": 3, "subquery_generation": False}
    base.update(over)
    return base


def test_generate_and_search_surfaces_provider_errors(monkeypatch):
    calls = {"n": 0}

    def fake_perform(*a, **k):
        calls["n"] += 1
        return {"results": [], "processing_error": "engine 'google' exploded"}

    monkeypatch.setattr(WebSearch_APIs, "perform_websearch", fake_perform)
    monkeypatch.setattr(WebSearch_APIs.time, "sleep", lambda s: None)
    out = WebSearch_APIs.generate_and_search("q", _search_params())
    wsr = out["web_search_results_dict"]
    assert wsr["warnings"] and "exploded" in wsr["warnings"][0]
    assert wsr["error"] and "exploded" in wsr["error"]  # zero results -> error set


def test_generate_and_search_dedupes_subquery_equal_to_question(monkeypatch):
    seen_queries = []

    def fake_perform(search_engine, search_query, *a, **k):
        seen_queries.append(search_query)
        return {"results": [_std_result("T", "https://e.com/", "c")], "processing_error": None}

    def fake_chat(**kwargs):
        return json.dumps({"sub_questions": ["What Is Love", "real subquery"]})

    monkeypatch.setattr(WebSearch_APIs, "perform_websearch", fake_perform)
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    monkeypatch.setattr(WebSearch_APIs.time, "sleep", lambda s: None)
    WebSearch_APIs.generate_and_search(
        "what is love", _search_params(subquery_generation=True, subquery_generation_llm="openai")
    )
    assert seen_queries == ["what is love", "real subquery"]  # casefold-dup dropped


def test_generate_and_search_caps_fanout_at_search_default_max_queries(monkeypatch):
    """Important 2 (final review): search_default_max_queries resolves but was
    never applied -- an LLM returning 12 sub-questions fanned out to 13
    total searches. Sub-queries are truncated to cap - 1 (the original
    question always counts as one), so total fan-out <= cap."""
    seen_queries = []

    def fake_perform(search_engine, search_query, *a, **k):
        seen_queries.append(search_query)
        return {"results": [_std_result("T", "https://e.com/", "c")], "processing_error": None}

    def fake_chat(**kwargs):
        return json.dumps({"sub_questions": [f"sub question {i}" for i in range(12)]})

    monkeypatch.setattr(WebSearch_APIs, "perform_websearch", fake_perform)
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    monkeypatch.setattr(WebSearch_APIs.time, "sleep", lambda s: None)
    WebSearch_APIs.generate_and_search(
        "what is love",
        _search_params(subquery_generation=True, subquery_generation_llm="openai",
                       search_default_max_queries=5),
    )
    assert len(seen_queries) == 5  # question + 4 sub-queries, NOT question + 12


def test_generate_and_search_stops_fanout_at_phase1_time_budget(monkeypatch):
    """Important 3a (final review): a cheap between-queries deadline check.
    A fake clock advances past the budget after the first search call, so
    the fan-out must stop before the remaining queries are searched, leave
    a warning naming how many of how many were searched, and still return
    the partial results gathered so far. No real sleeps -- the clock is
    faked via WebSearch_APIs.time.monotonic."""
    fake_clock = {"t": 1000.0}

    def fake_monotonic():
        return fake_clock["t"]

    seen_queries = []

    def fake_perform(search_engine, search_query, *a, **k):
        seen_queries.append(search_query)
        fake_clock["t"] += 10.0  # advance well past the budget after each call
        return {"results": [_std_result("T", "https://e.com/", "c")], "processing_error": None}

    def fake_chat(**kwargs):
        return json.dumps({"sub_questions": ["sq1", "sq2", "sq3"]})

    monkeypatch.setattr(WebSearch_APIs, "perform_websearch", fake_perform)
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    monkeypatch.setattr(WebSearch_APIs.time, "sleep", lambda s: None)
    monkeypatch.setattr(WebSearch_APIs.time, "monotonic", fake_monotonic)

    out = WebSearch_APIs.generate_and_search(
        "what is love",
        _search_params(subquery_generation=True, subquery_generation_llm="openai",
                       phase1_time_budget_s=5.0),
    )
    wsr = out["web_search_results_dict"]
    assert len(seen_queries) == 1  # fan-out stopped after the first query
    assert any("deadline reached during search fan-out" in w for w in wsr["warnings"])
    assert wsr["results"]  # partial results still returned, not discarded


def test_generate_and_search_warns_when_subquery_generation_exhausts_attempts(monkeypatch):
    """task-3221: when subquery_generation is on, analyze_question makes up
    to 3 paid LLM attempts; if every attempt fails to produce sub-questions,
    generate_and_search must leave a trace -- otherwise 3 billed calls are
    indistinguishable from the feature being off. Drives the real
    analyze_question/generate_and_search path (only chat_api_call and
    perform_websearch are faked) to a total failure and asserts the warning
    text lands in web_search_results_dict['warnings']."""
    attempts = {"n": 0}

    def always_garbage(**kwargs):
        attempts["n"] += 1
        return "not json and no quoted strings here"

    def fake_perform(search_engine, search_query, *a, **k):
        return {"results": [_std_result("T", "https://e.com/", "c")], "processing_error": None}

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", always_garbage)
    monkeypatch.setattr(WebSearch_APIs, "perform_websearch", fake_perform)
    monkeypatch.setattr(WebSearch_APIs.time, "sleep", lambda s: None)

    out = WebSearch_APIs.generate_and_search(
        "what is love", _search_params(subquery_generation=True, subquery_generation_llm="openai")
    )
    wsr = out["web_search_results_dict"]
    assert attempts["n"] == WebSearch_APIs._SUBQUERY_GENERATION_MAX_ATTEMPTS  # all paid attempts made
    expected = (
        f"sub-query generation failed after "
        f"{WebSearch_APIs._SUBQUERY_GENERATION_MAX_ATTEMPTS} attempts; "
        "searched only the original query"
    )
    assert expected in wsr["warnings"]


def test_generate_and_search_no_warning_when_subquery_generation_disabled(monkeypatch):
    """Sanity check: the new warning must never appear when
    subquery_generation is off -- analyze_question is never even called, so
    there is nothing to report as a "failure"."""
    def fake_perform(search_engine, search_query, *a, **k):
        return {"results": [_std_result("T", "https://e.com/", "c")], "processing_error": None}

    monkeypatch.setattr(WebSearch_APIs, "perform_websearch", fake_perform)
    monkeypatch.setattr(WebSearch_APIs.time, "sleep", lambda s: None)

    out = WebSearch_APIs.generate_and_search("q", _search_params(subquery_generation=False))
    wsr = out["web_search_results_dict"]
    assert not any("sub-query generation failed" in w for w in wsr["warnings"])


def test_generate_and_search_provider_error_not_demoted_by_subquery_warning(monkeypatch):
    """Important 2 (fix-wave 2026-08-07 review): the sub-query-generation-
    exhausted notice used to be appended at warnings[0] BEFORE the fan-out
    loop ran, and the promotion check just below it blindly promotes
    warnings[0] to the top-level `error`. With subquery_generation on AND
    generation exhausted AND a real provider error, that misattributed the
    PROVIDER's own error as the sub-query notice -- reproduced here with
    subquery_generation=True, analyze_question exhausted (garbage LLM
    output every attempt), and perform_websearch returning a
    processing_error with zero results."""
    attempts = {"n": 0}

    def always_garbage(**kwargs):
        attempts["n"] += 1
        return "not json and no quoted strings here"

    def fake_perform(*a, **k):
        return {"results": [], "processing_error": "engine 'google' exploded"}

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", always_garbage)
    monkeypatch.setattr(WebSearch_APIs, "perform_websearch", fake_perform)
    monkeypatch.setattr(WebSearch_APIs.time, "sleep", lambda s: None)

    out = WebSearch_APIs.generate_and_search(
        "what is love", _search_params(subquery_generation=True, subquery_generation_llm="openai")
    )
    wsr = out["web_search_results_dict"]
    assert attempts["n"] == WebSearch_APIs._SUBQUERY_GENERATION_MAX_ATTEMPTS  # all paid attempts made

    subquery_notice = (
        f"sub-query generation failed after "
        f"{WebSearch_APIs._SUBQUERY_GENERATION_MAX_ATTEMPTS} attempts; "
        "searched only the original query"
    )
    # The PROVIDER's own error is promoted -- never the sub-query notice.
    assert wsr["error"] and "exploded" in wsr["error"]
    assert subquery_notice not in wsr["error"]
    # ...but the sub-query notice is still present in warnings (just not
    # promoted, and not occupying the position the provider error owns).
    assert subquery_notice in wsr["warnings"]
    assert "exploded" in wsr["warnings"][0]
    assert wsr["warnings"][-1] == subquery_notice


def test_generate_and_search_no_false_error_when_subquery_warning_is_the_only_warning(monkeypatch):
    """Important 2 continued: a search that legitimately finds nothing --
    zero results, no provider processing_error at all -- must leave `error`
    at None. Before this fix, once the sub-query notice was the ONLY entry
    in `warnings` (subquery_generation exhausted, no provider ever errored),
    the promotion check treated that informational notice as if it were a
    provider error and set a FALSE `error`."""
    attempts = {"n": 0}

    def always_garbage(**kwargs):
        attempts["n"] += 1
        return "not json and no quoted strings here"

    def fake_perform(*a, **k):
        return {"results": [], "processing_error": None}  # legitimately empty, no error

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", always_garbage)
    monkeypatch.setattr(WebSearch_APIs, "perform_websearch", fake_perform)
    monkeypatch.setattr(WebSearch_APIs.time, "sleep", lambda s: None)

    out = WebSearch_APIs.generate_and_search(
        "what is love", _search_params(subquery_generation=True, subquery_generation_llm="openai")
    )
    wsr = out["web_search_results_dict"]
    assert attempts["n"] == WebSearch_APIs._SUBQUERY_GENERATION_MAX_ATTEMPTS

    subquery_notice = (
        f"sub-query generation failed after "
        f"{WebSearch_APIs._SUBQUERY_GENERATION_MAX_ATTEMPTS} attempts; "
        "searched only the original query"
    )
    assert subquery_notice in wsr["warnings"]  # notice still surfaced
    assert wsr["error"] is None  # but never promoted to a false error


# --- chunking / confidence ----------------------------------------------------

def test_build_chunk_infos_packs_and_splits():
    small = ["a" * 100, "b" * 100]
    chunks = WebSearch_APIs._build_chunk_infos(small, max_chars=250)
    assert len(chunks) == 1
    oversized = ["x" * 9000]
    chunks2 = WebSearch_APIs._build_chunk_infos(oversized, max_chars=6000)
    assert len(chunks2) == 1 and len(chunks2[0]["text"]) <= 6000


def test_estimate_confidence_formula_points():
    # Server WebSearch_APIs.py :1119-1133, verbatim: chunk_success is 1.0
    # (nothing failed) when chunk_count == 0, and a fully-clean LLM run
    # (has_llm and failed_chunks == 0) earns a +0.1 bonus, not a flat +0.05.
    f = WebSearch_APIs._estimate_confidence
    assert f(0, 0, 0, True) == 0.0
    assert f(10, 2, 0, True) == pytest.approx(0.9)
    assert f(5, 0, 0, True) == pytest.approx(0.675)  # chunk_count == 0 branch
    assert f(1, 1, 1, False) >= 0.1  # clamp floor


# --- aggregate_results branches ----------------------------------------------

_REL = {"1": {"content": "sum one", "original_content": "orig", "reasoning": "r1",
              "url": "https://one.example/", "title": "One"}}


def test_aggregate_empty_returns_typed_shape():
    out = WebSearch_APIs.aggregate_results({}, "q", [], "openai")
    assert set(out) == {"text", "evidence", "confidence", "chunks"}
    assert out["confidence"] == 0.0 and out["evidence"] == []


def test_aggregate_success_typed_and_numbered(monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured["prompt"] = kwargs["messages_payload"][0]["content"]
        return "Answer citing [1]."

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    # chunk-phase summarizer:
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib
    monkeypatch.setattr(Summarization_General_Lib, "analyze", lambda *a, **k: "chunk summary")
    out = WebSearch_APIs.aggregate_results(_REL, "q", [], "openai")
    # Success branch carries the citation-verification verdict (task-16331);
    # failure/empty branches (pinned by their own tests) omit the key.
    assert set(out) == {"text", "evidence", "confidence", "chunks", "citation_verification"}
    assert out["text"] == "Answer citing [1]."
    cv = out["citation_verification"]
    assert cv["markers_total"] == 1 and cv["markers_resolved"] == 1
    assert cv["unknown_marker_ids"] == []
    assert out["evidence"][0]["id"] == 1
    assert out["evidence"][0]["url"] == "https://one.example/"
    assert "[1]" in captured["prompt"]          # numbered payload shown to the LLM
    assert 0.1 <= out["confidence"] <= 0.99      # computed, not hardcoded


def test_aggregate_llm_failure_still_typed(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", boom)
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib
    monkeypatch.setattr(Summarization_General_Lib, "analyze", lambda *a, **k: "chunk summary")
    out = WebSearch_APIs.aggregate_results(_REL, "q", [], "openai")
    assert set(out) == {"text", "evidence", "confidence", "chunks"}  # no "summary" key ever


def test_aggregate_no_llm_fallback():
    out = WebSearch_APIs.aggregate_results(_REL, "q", [], None)
    assert "sum one" in out["text"] and out["confidence"] > 0.0


def test_aggregate_single_chunk_skips_wasted_map_call(monkeypatch):
    # _REL is one small entry -> exactly one chunk. The MAP-phase chunk
    # summarization would cost a provider round-trip whose output feeds
    # nothing (synthesis already reads the raw numbered evidence directly
    # when there's only one chunk) -- it must not be called at all.
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        return "Answer citing [1]."

    def fake_analyze(*a, **kwargs):
        calls["n"] += 1
        return "chunk summary"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib
    monkeypatch.setattr(Summarization_General_Lib, "analyze", fake_analyze)

    out = WebSearch_APIs.aggregate_results(_REL, "q", [], "openai")
    assert calls["n"] == 1  # only the synthesis call; MAP call skipped


_REL_MULTI = {
    "a": {"content": "A" * 4000, "reasoning": "ra", "url": "https://a.example/", "title": "A"},
    "b": {"content": "B" * 4000, "reasoning": "rb", "url": "https://b.example/", "title": "B"},
}


def test_aggregate_multi_chunk_synthesizes_from_chunk_summaries(monkeypatch):
    # Two ~4000-char entries pack into 2 separate 6000-char chunks, so the
    # MAP phase runs. The synthesis prompt must consume the chunk SUMMARIES
    # (not the raw ~4000-char originals) while the "[n]" markers the
    # summarizer is instructed to preserve still reach the synthesis prompt.
    captured = {}

    def fake_chat(**kwargs):
        captured["prompt"] = kwargs["messages_payload"][0]["content"]
        return "Answer citing [1][2]."

    def fake_analyze(*a, **kwargs):
        input_data = kwargs.get("input_data", "")
        marker = input_data.split("\n", 1)[0]  # e.g. "[1] A"
        return f"{marker} summary of chunk"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib
    monkeypatch.setattr(Summarization_General_Lib, "analyze", fake_analyze)

    out = WebSearch_APIs.aggregate_results(_REL_MULTI, "q", [], "openai")
    prompt = captured["prompt"]
    assert "summary of chunk" in prompt              # built from chunk summaries
    assert "[1]" in prompt and "[2]" in prompt        # citation markers survived the map step
    assert "A" * 4000 not in prompt and "B" * 4000 not in prompt  # not the raw originals


# --- relevance: timeouts, cancel, scrape fallback, url/title capture -----------

@pytest.mark.asyncio
async def test_relevance_scrape_failure_keeps_result_with_fallback(monkeypatch):
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call",
                        _fake_chat(["Selected Answer: True\nReasoning: looks relevant"]))

    async def failing_scrape(url, **k):
        raise RuntimeError("scrape died")

    monkeypatch.setattr(WebSearch_APIs, "scrape_article", failing_scrape)
    results = [_std_result("Kept Title", "https://kept.example/", "snippet text")]
    out = await WebSearch_APIs.search_result_relevance(results, "q", [], "openai")
    assert len(out) == 1
    entry = next(iter(out.values()))
    assert entry["url"] == "https://kept.example/" and entry["title"] == "Kept Title"
    assert "snippet text" in entry["content"] or "Kept Title" in entry["content"]


@pytest.mark.asyncio
async def test_relevance_cancel_event_stops_loop(monkeypatch):
    import asyncio
    evt = asyncio.Event()
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        evt.set()  # cancel after the first result
        return "Selected Answer: False\nReasoning: no"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    results = [_std_result(f"T{i}", f"https://e{i}.example/", "c") for i in range(5)]
    out = await WebSearch_APIs.search_result_relevance(results, "q", [], "openai", cancel_event=evt)
    assert calls["n"] == 1  # loop stopped after cancellation


@pytest.mark.asyncio
async def test_relevance_llm_timeout_counts_as_not_relevant(monkeypatch):
    import asyncio

    def hanging_chat(**kwargs):
        import time as _t
        _t.sleep(0.3)
        return "Selected Answer: True\nReasoning: slow"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", hanging_chat)
    results = [_std_result("T", "https://e.example/", "c")]
    out = await WebSearch_APIs.search_result_relevance(
        results, "q", [], "openai", llm_timeout_s=0.05)
    assert out == {}  # timed out -> skipped, not crashed


@pytest.mark.asyncio
async def test_relevance_refuses_private_url_scrape(monkeypatch):
    # Pre-scrape SSRF guard (task-1356): a relevant result pointing at a
    # cloud metadata IP must never be navigated to by scrape_article --
    # scrape_article is faked here as a spy solely to prove it's never
    # called; the guard refuses BEFORE any fetch, so this test performs
    # no real network I/O either way.
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call",
                        _fake_chat(["Selected Answer: True\nReasoning: relevant"]))
    scraped = []

    async def spy_scrape(url, **k):
        scraped.append(url)
        return {"content": "should not happen", "extraction_successful": True}

    monkeypatch.setattr(WebSearch_APIs, "scrape_article", spy_scrape)
    results = [_std_result("Internal", "http://169.254.169.254/latest", "metadata snippet")]
    out = await WebSearch_APIs.search_result_relevance(results, "q", [], "openai")
    assert scraped == []                                  # never navigated
    entry = next(iter(out.values()))
    assert "metadata snippet" in entry["content"] or "Internal" in entry["content"]  # fallback kept


@pytest.mark.asyncio
async def test_relevance_guard_does_not_block_event_loop(monkeypatch):
    # CRITICAL 2 (task-1356 re-review): is_public_http_url does synchronous
    # DNS resolution (socket.getaddrinfo). Calling it directly inside this
    # async function would stall the whole event loop for however long
    # resolution takes -- reproduced here by monkeypatching the guard to a
    # blocking time.sleep() standing in for a slow resolver, then proving a
    # concurrently-scheduled heartbeat coroutine is NOT stalled: its
    # sleep(0.01) ticks keep landing close to schedule instead of bunching
    # up behind the guard's 0.3s sleep.
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call",
                        _fake_chat(["Selected Answer: True\nReasoning: relevant"]))

    def slow_guard(url):
        time.sleep(0.3)
        return True

    monkeypatch.setattr(WebSearch_APIs, "is_public_http_url", slow_guard)

    async def fast_scrape(url, **k):
        return {"content": "scraped ok", "extraction_successful": True}

    monkeypatch.setattr(WebSearch_APIs, "scrape_article", fast_scrape)
    results = [_std_result("T", "https://e.example/", "c")]

    gaps = []
    stop = asyncio.Event()

    async def heartbeat():
        loop = asyncio.get_event_loop()
        last = loop.time()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = loop.time()
            gaps.append(now - last)
            last = now

    hb_task = asyncio.create_task(heartbeat())
    await WebSearch_APIs.search_result_relevance(results, "q", [], "openai")
    stop.set()
    await hb_task

    assert gaps, "heartbeat never got to run at all"
    # A blocked loop shows one huge gap (~0.3s, the guard's sleep duration);
    # an offloaded guard keeps every gap close to the 0.01s heartbeat interval.
    assert max(gaps) < 0.15, f"event loop stalled: max heartbeat gap {max(gaps):.3f}s"


@pytest.mark.asyncio
async def test_relevance_guard_timeout_falls_back_like_scrape_failure(monkeypatch):
    # CRITICAL 2 continued: a guard that doesn't resolve within
    # scrape_timeout_s must be treated as a refusal -- same fallback path
    # as a scrape failure or a private-IP refusal -- not left to hang or
    # raise out of search_result_relevance.
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call",
                        _fake_chat(["Selected Answer: True\nReasoning: relevant"]))

    def hanging_guard(url):
        time.sleep(1.0)
        return True

    monkeypatch.setattr(WebSearch_APIs, "is_public_http_url", hanging_guard)
    scraped = []

    async def spy_scrape(url, **k):
        scraped.append(url)
        return {"content": "should not happen", "extraction_successful": True}

    monkeypatch.setattr(WebSearch_APIs, "scrape_article", spy_scrape)
    results = [_std_result("Kept Title", "https://kept.example/", "snippet text")]
    out = await WebSearch_APIs.search_result_relevance(
        results, "q", [], "openai", scrape_timeout_s=0.05)
    assert scraped == []                                  # never reached scrape_article
    entry = next(iter(out.values()))
    assert "snippet text" in entry["content"] or "Kept Title" in entry["content"]  # fallback kept


# --- DNS-guard offload isolation (task-3220) -----------------------------------

@pytest.mark.asyncio
async def test_relevance_guard_runs_on_dedicated_dns_guard_executor(monkeypatch):
    """task-3220: the pre-scrape SSRF guard (`is_public_http_url`) must be
    offloaded through the dedicated DNS-guard executor, not
    `asyncio.to_thread`'s shared default one -- proven by capturing the name
    of the worker thread the guard actually ran on."""
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call",
                        _fake_chat(["Selected Answer: True\nReasoning: relevant"]))
    seen_thread_names = []

    def spying_guard(url):
        seen_thread_names.append(threading.current_thread().name)
        return True

    monkeypatch.setattr(WebSearch_APIs, "is_public_http_url", spying_guard)

    async def fast_scrape(url, **k):
        return {"content": "scraped ok", "extraction_successful": True}

    monkeypatch.setattr(WebSearch_APIs, "scrape_article", fast_scrape)
    results = [_std_result("T", "https://e.example/", "c")]
    await WebSearch_APIs.search_result_relevance(results, "q", [], "openai")

    assert seen_thread_names, "guard was never called"
    assert seen_thread_names[0].startswith("deep-search-dns-guard"), (
        f"guard ran on {seen_thread_names[0]!r}; expected the dedicated "
        "DNS-guard executor, not the shared default one"
    )


def test_dns_guard_executor_saturation_does_not_starve_default_executor_offloads(monkeypatch):
    """task-3220 / fix-wave Important 3 (2026-08-07 review, reviewer's option
    b): the PREDECESSOR of this test saturated the dedicated pool from
    OUTSIDE via direct `executor.submit()` calls, then ran a brand-new
    `asyncio.run(run())` with its own brand-new DEFAULT executor -- nothing
    it asserted could ever be affected by any other pool's state (the
    reviewer reproduced this: saturating a completely unrelated executor
    made the test pass identically). It was vacuous.

    This version drives the REAL `search_result_relevance()` -- the actual
    caller of `_get_dns_guard_executor()` -- and blocks `is_public_http_url`
    itself, so every real guard call the function submits saturates the
    dedicated pool from the inside, one call at a time, exactly like
    production. With `n_results = n_workers + 2`, the first
    `_DNS_GUARD_EXECUTOR_MAX_WORKERS` guard calls occupy every pool worker
    (each blocked on `release`, standing in for an abandoned getaddrinfo
    thread); the last two calls queue entirely UNSTARTED behind them -- the
    exact scenario the comment above the guard's `wait_for` call now
    documents (M4). It then proves the faked `chat_api_call` relevance
    offloads -- which run through `asyncio.to_thread` on the DEFAULT
    executor, same as production -- all still complete and land within a
    bounded, predictable time (governed by `scrape_timeout_s`, not by how
    long the abandoned guard threads actually stay blocked), instead of
    hanging or queueing behind them.

    This proves NON-STARVATION (the pipeline keeps making progress and
    every result gets a verdict even with the guard pool fully saturated),
    not the guard's specific executor IDENTITY -- that's
    `test_relevance_guard_runs_on_dedicated_dns_guard_executor`'s job, just
    above. `random.uniform` is patched to 0 to remove the unrelated
    0.2-0.6s pacing jitter `search_result_relevance` inserts before every
    LLM call (not what this test is about, and it would otherwise make an
    N-result run slow for no behavioral benefit). No real DNS, no real
    sleeps beyond `scrape_timeout_s`-scale (0.1s) waits; the guard block is
    released deterministically in `finally`, and the module-level executor
    singleton is reset there too so this test's saturation can never leak
    into a test that runs after it (fix-wave M8).
    """
    n_workers = WebSearch_APIs._DNS_GUARD_EXECUTOR_MAX_WORKERS
    n_results = n_workers + 2  # >= 2 guard calls must queue UNSTARTED once saturated
    release = threading.Event()
    llm_call_times: list = []
    scraped: list = []

    def blocking_guard(url):
        # Every real call blocks until `release` is set in `finally` below --
        # standing in for an abandoned getaddrinfo thread that never returns
        # within scrape_timeout_s (the exact production failure mode
        # task-3220 isolates).
        release.wait(timeout=5)  # released below; 5s is only a safety net
        return True

    def fake_chat(**kwargs):
        llm_call_times.append(time.monotonic())
        return "Selected Answer: True\nReasoning: relevant"

    async def spy_scrape(url, **k):
        scraped.append(url)
        return {"content": "should not be reached", "extraction_successful": True}

    monkeypatch.setattr(WebSearch_APIs, "is_public_http_url", blocking_guard)
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    monkeypatch.setattr(WebSearch_APIs, "scrape_article", spy_scrape)
    monkeypatch.setattr(WebSearch_APIs.random, "uniform", lambda a, b: 0.0)

    results = [_std_result(f"T{i}", f"https://e{i}.example/", "c") for i in range(n_results)]

    try:
        start = time.monotonic()
        out = asyncio.run(
            WebSearch_APIs.search_result_relevance(
                results, "q", [], "openai", scrape_timeout_s=0.1
            )
        )
        elapsed = time.monotonic() - start

        # Every result still got a verdict via the snippet/title/url
        # fallback -- a saturated guard pool must never hang or silently
        # drop results, whether an individual guard call is
        # running-but-blocked or queued entirely unstarted.
        assert len(out) == n_results
        assert scraped == []  # the guard refusal/timeout path was taken every time

        assert len(llm_call_times) == n_results, "not every result reached the relevance LLM call"
        # Bound is intentionally generous, NOT tight around n_results *
        # scrape_timeout_s: this test's job is proving the pipeline makes
        # progress and completes at all (non-starvation), not pinning which
        # executor the guard runs on -- that's the wiring test's job (see
        # module docstring above and the mutation-check note in the
        # fix-wave). A tight bound here would make this test accidentally
        # ALSO discriminate executor identity on machines with a small
        # `os.cpu_count()` (a small default ThreadPoolExecutor can genuinely
        # need one blocked worker's 5s safety-net release before a queued
        # call gets a turn if the guard were ever routed back onto it), and
        # the fix-wave's mutation check #3 explicitly requires this test to
        # keep passing when the guard is pointed at the default executor.
        # 15s comfortably covers that shared-pool/low-core-count case while
        # still catching a genuine per-call deadlock (which would run into
        # tens of seconds to minutes across `n_results` iterations at
        # `llm_timeout_s`-scale each).
        assert elapsed < 15.0, (
            f"search_result_relevance took {elapsed:.3f}s over {n_results} "
            "results with a saturated DNS-guard pool -- expected bounded "
            "progress (scrape_timeout_s-scale per result, or a few "
            "multiples of it under executor contention), not something "
            "resembling a per-result deadlock"
        )
    finally:
        release.set()
        WebSearch_APIs._reset_dns_guard_executor_for_tests()


# --- robots.txt parity for the scrape path (task-3260) -------------------------


@pytest.mark.asyncio
async def test_relevance_robots_disallowed_skips_scrape_others_proceed(monkeypatch):
    """respect_robots_txt=True must skip scraping a robots-disallowed host
    (keeping its existing snippet/title/url fallback content, never
    discarding the result) while an allowed host on the same run scrapes
    normally -- mirrors the SSRF-refusal path's shape exactly.

    THREE fakes required (task-3260 design doc, spec review Important 3 --
    miss one and the test observes the wrong refusal or fail-open):
    (1) WebSearch_APIs.is_public_http_url faked True -- the SSRF guard runs
        FIRST, does real DNS, and fails CLOSED for every .example host in
        this file before any robots check could run;
    (2) web_tool_impls._transport MockTransport serving the robots.txt
        body;
    (3) socket.getaddrinfo faked to a public IP -- _fetch_robots_parser's
        OWN _validate_hop does a SEPARATE DNS check on the robots.txt URL
        and fails OPEN on DNS failure, silently bypassing the MockTransport
        if this isn't faked too.
    """
    web_tool_impls._reset_state_for_tests()
    monkeypatch.setattr(WebSearch_APIs, "is_public_http_url", lambda url: True)
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 80))]
    )
    # Minor 8-style: skip the real 0.2-0.6s pacing jitter search_result_relevance
    # inserts before every LLM call (precedent: test_deep_search_pipeline.py's
    # own DNS-guard saturation test, and task-3060's searx test).
    monkeypatch.setattr(WebSearch_APIs.random, "uniform", lambda a, b: 0.0)

    def robots_handler(request: httpx.Request) -> httpx.Response:
        if "disallowed.example" in str(request.url):
            return httpx.Response(200, content=b"User-agent: *\nDisallow: /\n")
        return httpx.Response(200, content=b"User-agent: *\nAllow: /\n")

    monkeypatch.setattr(web_tool_impls, "_transport", httpx.MockTransport(robots_handler))
    monkeypatch.setattr(
        WebSearch_APIs, "chat_api_call",
        _fake_chat(["Selected Answer: True\nReasoning: relevant"] * 2),
    )

    scraped = []

    async def spy_scrape(url, **k):
        scraped.append(url)
        return {"content": "REAL SCRAPED CONTENT", "extraction_successful": True}

    monkeypatch.setattr(WebSearch_APIs, "scrape_article", spy_scrape)

    results = [
        _std_result("Disallowed", "https://disallowed.example/page", "disallowed snippet"),
        _std_result("Allowed", "https://allowed.example/page", "allowed snippet"),
    ]
    try:
        out = await WebSearch_APIs.search_result_relevance(
            results, "q", [], "openai", respect_robots_txt=True,
        )
    finally:
        web_tool_impls._reset_state_for_tests()

    assert scraped == ["https://allowed.example/page"], (
        "the disallowed host must never reach scrape_article"
    )
    assert len(out) == 2  # both results kept -- disallowed is a fallback, not a discard
    disallowed_entry = next(v for v in out.values() if v["url"] == "https://disallowed.example/page")
    assert "REAL SCRAPED CONTENT" not in disallowed_entry["content"]
    # Pin the ACTUAL _build_result_fallback_content shape (Minor 5) rather
    # than a loose "either field" OR -- the disallowed result's summary
    # step deterministically falls back to source_content unmodified (no
    # real LLM configured in this test), so this is an exact match, not a
    # heuristic one.
    assert disallowed_entry["content"] == WebSearch_APIs._build_result_fallback_content(
        results[0]
    )
    allowed_entry = next(v for v in out.values() if v["url"] == "https://allowed.example/page")
    assert "REAL SCRAPED CONTENT" in allowed_entry["content"]


@pytest.mark.asyncio
async def test_relevance_robots_off_by_default_makes_no_robots_fetch(monkeypatch):
    """Parity pin: respect_robots_txt defaults to False when the caller
    doesn't pass it -- the dead-wired research-service caller never sets
    this, and must keep making ZERO robots.txt fetches (transport-call
    count) with the scrape proceeding exactly like before task-3260. The
    registered robots.txt disallows everything, so if this test somehow DID
    reach a robots check, the scrape would incorrectly get skipped."""
    web_tool_impls._reset_state_for_tests()
    monkeypatch.setattr(WebSearch_APIs, "is_public_http_url", lambda url: True)
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 80))]
    )
    monkeypatch.setattr(WebSearch_APIs.random, "uniform", lambda a, b: 0.0)

    transport_calls = []

    def robots_handler(request: httpx.Request) -> httpx.Response:
        transport_calls.append(str(request.url))
        return httpx.Response(200, content=b"User-agent: *\nDisallow: /\n")

    monkeypatch.setattr(web_tool_impls, "_transport", httpx.MockTransport(robots_handler))
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call",
                        _fake_chat(["Selected Answer: True\nReasoning: relevant"]))

    scraped = []

    async def spy_scrape(url, **k):
        scraped.append(url)
        return {"content": "scraped ok", "extraction_successful": True}

    monkeypatch.setattr(WebSearch_APIs, "scrape_article", spy_scrape)
    results = [_std_result("T", "https://would-be-blocked.example/", "c")]
    try:
        # respect_robots_txt intentionally OMITTED -- must default to False.
        out = await WebSearch_APIs.search_result_relevance(results, "q", [], "openai")
    finally:
        web_tool_impls._reset_state_for_tests()

    assert transport_calls == [], "no robots.txt fetch should happen when the toggle is absent"
    assert scraped == ["https://would-be-blocked.example/"]


@pytest.mark.asyncio
async def test_relevance_robots_unreachable_fails_open_and_scrapes(monkeypatch):
    """A robots.txt that cannot be fetched (network error) must fail OPEN
    and proceed to scrape -- deliberately the OPPOSITE of the SSRF guard's
    own timeout/refusal just above it, matching _fetch_robots_parser's
    existing fail-open for web_fetch/web_crawl (ruling 5)."""
    web_tool_impls._reset_state_for_tests()
    monkeypatch.setattr(WebSearch_APIs, "is_public_http_url", lambda url: True)
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 80))]
    )
    monkeypatch.setattr(WebSearch_APIs.random, "uniform", lambda a, b: 0.0)

    def robots_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated robots.txt fetch failure", request=request)

    monkeypatch.setattr(web_tool_impls, "_transport", httpx.MockTransport(robots_handler))
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call",
                        _fake_chat(["Selected Answer: True\nReasoning: relevant"]))

    scraped = []

    async def spy_scrape(url, **k):
        scraped.append(url)
        return {"content": "scraped ok", "extraction_successful": True}

    monkeypatch.setattr(WebSearch_APIs, "scrape_article", spy_scrape)
    results = [_std_result("T", "https://unreachable-robots.example/", "c")]
    try:
        out = await WebSearch_APIs.search_result_relevance(
            results, "q", [], "openai", respect_robots_txt=True,
        )
    finally:
        web_tool_impls._reset_state_for_tests()

    assert scraped == ["https://unreachable-robots.example/"]  # fail-open: scrape proceeded


# --- pure review ---------------------------------------------------------------

def test_review_no_selector_passes_all():
    wsr = {"results": [_std_result("A", "https://a.example/", "c")]}
    out = WebSearch_APIs.review_and_select_results(wsr)
    assert len(out["results"]) == 1


def test_review_never_blocks_on_input(monkeypatch):
    import builtins
    def no_input(*a, **k):
        raise AssertionError("input() must never be called")
    monkeypatch.setattr(builtins, "input", no_input)
    WebSearch_APIs.review_and_select_results({"results": []})


# --- [SearchSettings] loader (task-1356 Task 4) -------------------------------

def test_search_settings_timeout_keys_with_defaults(tmp_path, monkeypatch):
    """New timeout keys load with defaults when absent from TOML.

    Acceptance: search_settings_general contains three new int keys:
    - relevance_llm_timeout_s = 30
    - relevance_scrape_timeout_s = 30
    - deep_search_timeout_s = 240 (task-1356 review ruling; fix round 1: the
      agent runtime now derives web_deep_search's own per-call timeout from
      this value via LocalToolProvider.timeout_for, so a deadline-hit run
      can still return its partial synthesis instead of being killed by the
      runtime's own ceiling first, for any configured value)
    """
    config_path = tmp_path / "config.toml"
    # Minimal config with no SearchSettings section
    config_path.write_text(
        "[general]\nusers_name = 'test'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    search_settings = settings["search_settings_general"]
    assert search_settings["relevance_llm_timeout_s"] == 30
    assert search_settings["relevance_scrape_timeout_s"] == 30
    assert search_settings["deep_search_timeout_s"] == 240


def test_search_settings_timeout_keys_from_toml(tmp_path, monkeypatch):
    """Timeout keys load from TOML when present.

    Acceptance: custom TOML values override defaults.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[SearchSettings]\n"
        "relevance_llm_timeout_s = 45\n"
        "relevance_scrape_timeout_s = 60\n"
        "deep_search_timeout_s = 600\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    search_settings = settings["search_settings_general"]
    assert search_settings["relevance_llm_timeout_s"] == 45
    assert search_settings["relevance_scrape_timeout_s"] == 60
    assert search_settings["deep_search_timeout_s"] == 600


def test_search_settings_timeout_keys_coerce_int(tmp_path, monkeypatch):
    """Timeout keys coerce string values to int.

    Acceptance: "30" (string) → 30 (int).
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[SearchSettings]\n"
        'relevance_llm_timeout_s = "45"\n'
        'relevance_scrape_timeout_s = "60"\n'
        'deep_search_timeout_s = "600"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    search_settings = settings["search_settings_general"]
    assert search_settings["relevance_llm_timeout_s"] == 45
    assert isinstance(search_settings["relevance_llm_timeout_s"], int)
    assert search_settings["relevance_scrape_timeout_s"] == 60
    assert isinstance(search_settings["relevance_scrape_timeout_s"], int)
    assert search_settings["deep_search_timeout_s"] == 600
    assert isinstance(search_settings["deep_search_timeout_s"], int)


def test_search_settings_timeout_keys_malformed_value_degrades_to_default(
    tmp_path, monkeypatch, caplog
):
    """Malformed timeout values degrade to defaults with a warning log.

    Acceptance: "30s" (malformed) → 30 (default), logged warning.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[SearchSettings]\n"
        'relevance_llm_timeout_s = "30s"\n'
        'relevance_scrape_timeout_s = true\n'
        'deep_search_timeout_s = "300x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    search_settings = settings["search_settings_general"]
    # Malformed values should degrade to defaults
    assert search_settings["relevance_llm_timeout_s"] == 30
    assert search_settings["relevance_scrape_timeout_s"] == 30
    assert search_settings["deep_search_timeout_s"] == 240


# --- Config template and non-positive timeout guards -----------------------

def test_config_template_contains_tools_section():
    """The CONFIG_TOML_CONTENT template includes the [tools] section with web_deep_search_enabled key.

    Acceptance: uncommmented template contains literal 'web_deep_search_enabled'
    (when the key is uncommented by a user, it is a valid TOML config).
    """
    template = config_module.CONFIG_TOML_CONTENT
    assert "web_deep_search_enabled" in template, (
        "CONFIG_TOML_CONTENT must include 'web_deep_search_enabled' key in the [tools] section"
    )


def test_non_positive_timeout_values_degrade_to_default(tmp_path, monkeypatch, caplog):
    """Non-positive timeout values (zero and negative) degrade to defaults with warnings.

    Acceptance: 0 → 30 (default), logged warning; -5 → 30 (default), logged warning.
    """
    import logging
    from loguru import logger as loguru_logger

    # Bridge loguru to caplog for this test
    class PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    handler_id = loguru_logger.add(PropagateHandler(), format="{message}")
    try:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "[SearchSettings]\n"
            "relevance_llm_timeout_s = 0\n"
            "relevance_scrape_timeout_s = -5\n"
            "deep_search_timeout_s = 100\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

        with caplog.at_level(logging.WARNING):
            settings = config_module.load_settings(force_reload=True)

        search_settings = settings["search_settings_general"]
        # Non-positive values should degrade to defaults
        assert search_settings["relevance_llm_timeout_s"] == 30
        assert search_settings["relevance_scrape_timeout_s"] == 30
        assert search_settings["deep_search_timeout_s"] == 100

        # Check warnings were logged for the non-positive cases
        assert "non-positive" in caplog.text.lower() or "not valid for timeout" in caplog.text.lower()
    finally:
        loguru_logger.remove(handler_id)


# --- analyze_and_aggregate must not block the event loop (task-1356 review) --

def test_analyze_and_aggregate_offloads_aggregate_results_so_wait_for_can_fire(monkeypatch):
    """aggregate_results is synchronous; calling it directly on the event
    loop thread blocks the whole loop for its duration, which means an
    outer asyncio.wait_for wrapped around analyze_and_aggregate can never
    actually fire its timeout -- the scheduled cancellation callback needs
    the loop to be idle to run, and a synchronous call never yields it one.
    Reproduced here with a blocking aggregate_results stand-in; the fix
    (task-1356 review) offloads the real call via asyncio.to_thread.

    Before the fix: this returns the LATE result (after the full 0.3s
    block) without ever raising TimeoutError -- wait_for's deadline is
    silently missed. After the fix: TimeoutError fires close to the
    intended ~0.05s timeout.

    Note on measurement: once aggregate_results is offloaded to a worker
    thread, wait_for's cancellation of the *awaiting* coroutine is prompt,
    but the underlying concurrent.futures.Future backing that thread cannot
    actually be cancelled once it has started running (a documented
    to_thread/run_in_executor limitation) -- so asyncio.run()'s own shutdown
    sequence still blocks for the full 0.3s waiting for that orphaned
    thread to finish before the process can unwind. That wait is measuring
    asyncio's cleanup, not whether the fix worked, so this test times the
    TimeoutError from INSIDE the coroutine (immediately when wait_for
    raises it) rather than timing the outer asyncio.run() call.
    """

    def blocking_aggregate(relevant_results, question, sub_questions, api_endpoint):
        time.sleep(0.3)  # simulates a slow synchronous LLM call
        return {"text": "late", "evidence": [], "confidence": 0.5, "chunks": []}

    async def fake_relevance(*_a, **_k):
        return {"1": {"url": "https://e.com/", "title": "T", "content": "c"}}

    monkeypatch.setattr(WebSearch_APIs, "aggregate_results", blocking_aggregate)
    monkeypatch.setattr(WebSearch_APIs, "search_result_relevance", fake_relevance)

    wsr = {"results": [{"title": "T", "url": "https://e.com/"}], "warnings": []}
    sqd = {"main_goal": "q", "sub_questions": []}
    params = {"relevance_analysis_llm": "openai", "final_answer_llm": "openai"}

    async def run() -> float:
        start_inner = time.monotonic()
        try:
            await asyncio.wait_for(
                WebSearch_APIs.analyze_and_aggregate(wsr, sqd, params),
                timeout=0.05,
            )
        except asyncio.TimeoutError:
            return time.monotonic() - start_inner
        raise AssertionError("expected analyze_and_aggregate to time out")

    elapsed = asyncio.run(run())
    assert elapsed < 0.2, (
        f"wait_for did not fire near its 0.05s deadline (took {elapsed:.2f}s) -- "
        "aggregate_results is still blocking the event loop"
    )


# --- discriminating timeout pass-through (task-1356 final review, Minor 6) ---

@pytest.mark.asyncio
async def test_analyze_and_aggregate_forwards_nondefault_relevance_llm_timeout(monkeypatch):
    """No test previously drove a NON-default timeout through
    analyze_and_aggregate into search_result_relevance -- a test using the
    30s default would pass even if the forwarding at WebSearch_APIs.py
    :596-597 were deleted (30.0 is also the fallback default). This uses 45
    specifically so deleting that forwarding turns it red."""
    captured = {}

    async def fake_relevance(*args, **kwargs):
        captured["llm_timeout_s"] = kwargs.get("llm_timeout_s")
        return {}

    monkeypatch.setattr(WebSearch_APIs, "search_result_relevance", fake_relevance)

    wsr = {"results": [], "warnings": []}
    sqd = {"main_goal": "q", "sub_questions": []}
    params = {"relevance_analysis_llm": "openai", "final_answer_llm": "openai",
              "relevance_llm_timeout_s": 45}

    await WebSearch_APIs.analyze_and_aggregate(wsr, sqd, params)
    assert captured["llm_timeout_s"] == 45


@pytest.mark.asyncio
async def test_analyze_and_aggregate_forwards_respect_robots_txt_true(monkeypatch):
    """task-3260's ONLY production link is this forwarding, and nothing
    previously exercised it: every other robots test either pins the
    tool's search_params write (test_web_deep_search.py) or calls
    search_result_relevance directly WITH the kwarg already supplied
    (test_deep_search_pipeline.py's own robots tests above) -- so deleting
    the forwarding at analyze_and_aggregate's call site left the full
    suite green (fix-round mutation finding). This calls the REAL
    analyze_and_aggregate with a spy on search_result_relevance."""
    captured = {}

    async def fake_relevance(*args, **kwargs):
        captured["respect_robots_txt"] = kwargs.get("respect_robots_txt")
        return {}

    monkeypatch.setattr(WebSearch_APIs, "search_result_relevance", fake_relevance)

    wsr = {"results": [], "warnings": []}
    sqd = {"main_goal": "q", "sub_questions": []}
    params = {"relevance_analysis_llm": "openai", "final_answer_llm": "openai",
              "respect_robots_txt": True}

    await WebSearch_APIs.analyze_and_aggregate(wsr, sqd, params)
    assert captured["respect_robots_txt"] is True


@pytest.mark.asyncio
async def test_analyze_and_aggregate_string_false_does_not_enable_robots(monkeypatch):
    """Qodo PR #1451 (the arc's FOURTH bool("false") catch): a stringly
    caller serializing search_params must not ENABLE enforcement with
    "false" -- only "true"/"1" strings (or a real bool) enable; anything
    else forwards False."""
    captured = {}

    async def fake_relevance(*args, **kwargs):
        captured["respect_robots_txt"] = kwargs.get("respect_robots_txt")
        return {}

    monkeypatch.setattr(WebSearch_APIs, "search_result_relevance", fake_relevance)
    wsr = {"results": [], "warnings": []}
    sqd = {"main_goal": "q", "sub_questions": []}
    for raw, expected in (("false", False), ("true", True), ("1", True), ("no", False), (0, False)):
        captured.clear()
        params = {"relevance_analysis_llm": "openai", "final_answer_llm": "openai",
                  "respect_robots_txt": raw}
        await WebSearch_APIs.analyze_and_aggregate(wsr, sqd, params)
        assert captured["respect_robots_txt"] is expected, f"raw={raw!r}"


@pytest.mark.asyncio
async def test_analyze_and_aggregate_forwards_respect_robots_txt_false_when_absent(monkeypatch):
    """Companion case: an absent key must forward False (not None, not
    missing), proving the forwarding isn't hardcoded True and the
    documented default really does reach search_result_relevance -- parity
    with the dead-wired research-service caller that never sets this key."""
    captured = {}

    async def fake_relevance(*args, **kwargs):
        captured["respect_robots_txt"] = kwargs.get("respect_robots_txt")
        return {}

    monkeypatch.setattr(WebSearch_APIs, "search_result_relevance", fake_relevance)

    wsr = {"results": [], "warnings": []}
    sqd = {"main_goal": "q", "sub_questions": []}
    params = {"relevance_analysis_llm": "openai", "final_answer_llm": "openai"}
    # respect_robots_txt intentionally OMITTED.

    await WebSearch_APIs.analyze_and_aggregate(wsr, sqd, params)
    assert captured["respect_robots_txt"] is False


# --- relevance gate robustness (task-16333) --------------------------------------

@pytest.mark.asyncio
async def test_relevance_judgment_runs_at_classification_temperature(monkeypatch):
    captured = []

    def fake_chat(**kwargs):
        captured.append(kwargs)
        return "Selected Answer: False\nReasoning: no"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    monkeypatch.setattr(WebSearch_APIs, "scrape_article", failing_scrape_noop)
    results = [_std_result("T", "https://e.example/", "c")]
    await WebSearch_APIs.search_result_relevance(results, "q", [], "openai")

    eval_calls = [
        c for c in captured
        if str(c["messages_payload"][0]["content"]).startswith("Evaluate the relevance")
    ]
    assert eval_calls, "the judgment call must be identifiable by its input"
    assert all(c["temp"] <= 0.2 for c in eval_calls)


async def failing_scrape_noop(url, **k):
    raise RuntimeError("no scrape")


@pytest.mark.asyncio
async def test_zero_relevant_falls_back_to_flagged_top_results(monkeypatch):
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call",
                        _fake_chat(["Selected Answer: False\nReasoning: off-topic"] * 5))
    results = [_std_result(f"T{i}", f"https://e{i}.example/", f"snippet {i}") for i in range(5)]

    out = await WebSearch_APIs.search_result_relevance(results, "q", [], "openai")

    assert out, "all-rejected must not produce an empty evidence set when raw results exist"
    assert len(out) <= 3  # bounded fallback
    first = next(iter(out.values()))
    assert first["gate_unverified"] is True
    assert first["url"] == "https://e0.example/"  # original rank order
    assert "snippet 0" in (first["content"] or "")  # snippet-level, no summarize spend


@pytest.mark.asyncio
async def test_zero_relevant_with_cancel_keeps_empty(monkeypatch):
    import asyncio
    evt = asyncio.Event()

    def fake_chat(**kwargs):
        evt.set()
        return "Selected Answer: False\nReasoning: no"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    results = [_std_result("T", "https://e.example/", "c")]

    out = await WebSearch_APIs.search_result_relevance(results, "q", [], "openai", cancel_event=evt)

    assert out == {}  # a cancelled/deadline run reports honestly, no fallback


@pytest.mark.asyncio
async def test_zero_relevant_with_unevaluated_results_keeps_empty(monkeypatch):
    import time as _t

    def hanging_chat(**kwargs):
        _t.sleep(0.3)
        return "Selected Answer: True\nReasoning: slow"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", hanging_chat)
    results = [_std_result("T", "https://e.example/", "c")]

    out = await WebSearch_APIs.search_result_relevance(results, "q", [], "openai", llm_timeout_s=0.05)

    assert out == {}  # never-evaluated results are not promoted (existing pin, unchanged)


@pytest.mark.asyncio
async def test_aggregate_carries_gate_unverified_flag_into_evidence(monkeypatch):
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", lambda **kwargs: "A[1].")
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib
    monkeypatch.setattr(Summarization_General_Lib, "analyze", lambda *a, **k: "s")

    out = WebSearch_APIs.aggregate_results(
        {"1": {"content": "c", "original_content": "o", "reasoning": "gate fallback",
               "url": "https://e.example/", "title": "T", "gate_unverified": True}},
        "q", [], "openai",
    )
    assert out["evidence"][0]["gate_unverified"] is True


# --- source-type-aware gate prompt (task-17066) -------------------------------------

@pytest.mark.asyncio
async def test_relevance_gate_carries_source_note_for_repository_records(monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured["prompt"] = kwargs["messages_payload"][0]["content"]
        return "Selected Answer: False\nReasoning: no"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    result = _std_result("Folding dataset", "https://zenodo.org/records/1", "Simulations of folding")
    result["metadata"] = {"source": "academic", "provider": "zenodo", "doi": "10.5281/x"}

    await WebSearch_APIs.search_result_relevance([result], "how do proteins fold", [], "openai")

    assert "repository record" in captured["prompt"]
    assert "does NOT need to directly answer" in captured["prompt"]


@pytest.mark.asyncio
async def test_relevance_gate_carries_source_note_for_metadata_records(monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured["prompt"] = kwargs["messages_payload"][0]["content"]
        return "Selected Answer: False\nReasoning: no"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    result = _std_result("Registry record", "https://openalex.org/W1", "Citation metadata")
    result["metadata"] = {"source": "academic", "provider": "openalex"}

    await WebSearch_APIs.search_result_relevance([result], "any question", [], "openai")

    assert "metadata record" in captured["prompt"]


@pytest.mark.asyncio
async def test_relevance_gate_prompt_unchanged_for_papers_and_web(monkeypatch):
    prompts = []

    def fake_chat(**kwargs):
        prompts.append(kwargs["messages_payload"][0]["content"])
        return "Selected Answer: False\nReasoning: no"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    paper = _std_result("A paper", "https://arxiv.org/abs/1", "Full text")
    paper["metadata"] = {"source": "academic", "provider": "arxiv"}
    web = _std_result("A page", "https://example.com/", "content")

    await WebSearch_APIs.search_result_relevance([paper, web], "q", [], "openai")

    # Byte-identical eval INPUT for both kinds (prefix equality -- absence
    # checks alone would miss any other drift in the input line).
    assert len(prompts) == 2
    prefix = "Evaluate the relevance of the search result."
    for prompt in prompts:
        assert prompt.split("\n\n", 1)[0] == prefix


@pytest.mark.asyncio
async def test_relevance_gate_paper_prompt_is_byte_identical(monkeypatch):
    captured = []

    def fake_chat(**kwargs):
        captured.append(kwargs["messages_payload"][0]["content"])
        return "Selected Answer: False\nReasoning: no"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat)
    plain_result = _std_result("T", "https://e.example/", "c")
    paper = _std_result("P", "https://arxiv.org/abs/1", "c")
    paper["metadata"] = {"source": "academic", "provider": "arxiv"}

    await WebSearch_APIs.search_result_relevance(
        [plain_result, paper], "q", [], "openai"
    )

    # The eval INPUT (everything before the "\n\nSearch Results" payload)
    # must be byte-identical for unclassified and paper-classified results
    # -- only the embedded result content differs.
    prefix = "Evaluate the relevance of the search result."
    assert len(captured) == 2
    for prompt in captured:
        input_line = prompt.split("\n\n", 1)[0]
        assert input_line == prefix  # no note, no extra text


# --- provider error strings must never become evidence (task-17382) ----------
# The summarizers report failure by RETURNING a string, and the caller only
# recognized the "Error:" prefix. A llama.cpp failure returns "Llama: Error
# occurred while processing summary with Llama: 'llama_api'", which sailed
# through the guard and was stored as the result's content -- so the synthesis
# was built from an error message where the source body belonged.


@pytest.mark.parametrize(
    "failure_text",
    [
        "Llama: Error occurred while processing summary with Llama: 'llama_api'",
        "Kobold: Error occurred while processing summary with Kobold: boom",
        "Llama: API request failed: 502 Bad Gateway",
        "Ollama: JSON parse error from summarization API.",
        "Custom OpenAI API: Unexpected error occurred: boom",
        "Custom OpenAI API-2: Error making API request: boom",
        "Llama: No choices in response data",
        "Error: legacy prefix still detected",
        "Error summarizing with Oobabooga: boom",
    ],
)
def test_provider_failure_strings_are_recognized(failure_text):
    from tldw_chatbook.Web_Scraping.WebSearch_APIs import _is_summary_failure

    assert _is_summary_failure(failure_text) is True


@pytest.mark.parametrize(
    "summary_text",
    [
        "This dataset reports error rates for retrieval augmented generation.",
        "The paper analyses failure modes and error propagation in MoE routing.",
        "Graph neural networks aggregate messages over edges.",
    ],
)
def test_real_summaries_are_not_mistaken_for_failures(summary_text):
    """The detector must not eat legitimate prose that merely says "error"."""
    from tldw_chatbook.Web_Scraping.WebSearch_APIs import _is_summary_failure

    assert _is_summary_failure(summary_text) is False
