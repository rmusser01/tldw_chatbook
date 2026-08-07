"""Deep-search pipeline hardening (task-1356). Fakes live ONLY at chat_api_call /
scrape_article / Summarization analyze — the pipeline code runs real."""

import asyncio
import json
import time

import pytest

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
    assert set(out) == {"text", "evidence", "confidence", "chunks"}
    assert out["text"] == "Answer citing [1]."
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
    - deep_search_timeout_s = 240 (task-1356 review ruling: must undercut the
      agent runtime's 300s max_tool_call_seconds so a deadline-hit run can
      still return its partial synthesis instead of being killed first)
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
