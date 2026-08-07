"""Deep-search pipeline hardening (task-1356). Fakes live ONLY at chat_api_call /
scrape_article / Summarization analyze — the pipeline code runs real."""

import json

import pytest

from tldw_chatbook.Web_Scraping import WebSearch_APIs


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
