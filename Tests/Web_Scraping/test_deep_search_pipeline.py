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
