# Tests/Web_Scraping/test_websearch_internal_prompts.py
"""Overrides in config.toml must change the text handed to the LLM transport.
Fakes live ONLY at chat_api_call — the pipeline code runs real.

The `scratch_config` fixture comes from Tests/Web_Scraping/conftest.py, which
re-exports the canonical definition in Tests/Internal_Prompts/conftest.py
(plain import, not `pytest_plugins`), because declaring `pytest_plugins =
["Tests.Internal_Prompts.conftest"]` — in this test module or in
Tests/Web_Scraping/conftest.py — collides with pytest's implicit auto-load
of Tests/Internal_Prompts/conftest.py under --import-mode=importlib when
both test directories are collected in the same session (ValueError: Plugin
already registered under a different name)."""

import pytest


def test_sub_question_prompt_override_reaches_transport(scratch_config, monkeypatch):
    from tldw_chatbook.Web_Scraping import WebSearch_APIs

    scratch_config(
        "[internal_prompts.websearch]\n"
        'sub_question_generation = "CUSTOM SUBQ PROMPT for: {original_query}"\n'
    )

    captured = {}

    def fake_chat_api_call(*args, **kwargs):
        captured["payload"] = kwargs.get("messages_payload") or args[1]
        return '{"sub_questions": ["a", "b"]}'

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat_api_call)

    result = WebSearch_APIs.analyze_question("what is love", api_endpoint="openai")

    content = captured["payload"][0]["content"]
    assert "CUSTOM SUBQ PROMPT for: what is love" in content
    assert result["sub_questions"] == ["a", "b"]


def test_default_used_when_no_override(scratch_config, monkeypatch):
    from tldw_chatbook.Web_Scraping import WebSearch_APIs

    captured = {}

    def fake_chat_api_call(*args, **kwargs):
        captured["payload"] = kwargs.get("messages_payload") or args[1]
        return '{"sub_questions": ["a"]}'

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat_api_call)

    WebSearch_APIs.analyze_question("what is love", api_endpoint="openai")

    content = captured["payload"][0]["content"]
    assert "You are an AI assistant that helps generate search queries" in content
    assert "Original query: what is love" in content


# --- sites 2-4: one transport-boundary test each (task 460) ------------------


def test_answer_synthesis_override_reaches_transport(scratch_config, monkeypatch):
    from tldw_chatbook.Web_Scraping import WebSearch_APIs

    scratch_config(
        "[internal_prompts.websearch]\n"
        'answer_synthesis = "CUSTOM SYNTH for {question} :: {concatenated_texts} :: {current_date}"\n'
    )
    captured = {}

    def fake_chat_api_call(*args, **kwargs):
        captured["payload"] = kwargs.get("messages_payload") or args[1]
        return "final report"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat_api_call)

    WebSearch_APIs.aggregate_results(
        {"1": {"content": "some content", "reasoning": "because"}},
        question="what is love",
        sub_questions=["a"],
        api_endpoint="openai",
    )

    content = captured["payload"][0]["content"]
    assert "CUSTOM SYNTH for what is love" in content
    assert "some content" in content  # concatenated_texts flowed through


@pytest.mark.asyncio
async def test_result_relevance_eval_override_reaches_transport(scratch_config, monkeypatch):
    from tldw_chatbook.Web_Scraping import WebSearch_APIs

    scratch_config(
        "[internal_prompts.websearch]\n"
        'result_relevance_eval = "CUSTOM EVAL {original_question} :: {sub_questions} :: {content}"\n'
    )
    captured = {}

    def fake_chat_api_call(*args, **kwargs):
        captured["payload"] = kwargs.get("messages_payload") or args[1]
        # "False" keeps the result out of the scrape/summarize path
        return "Selected Answer: False\nReasoning: not relevant"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat_api_call)

    await WebSearch_APIs.search_result_relevance(
        [{"content": "page text", "url": "https://example.com", "id": "1"}],
        original_question="what is love",
        sub_questions=["a", "b"],
        api_endpoint="openai",
    )

    content = captured["payload"][0]["content"]
    assert "CUSTOM EVAL what is love" in content
    assert "page text" in content


@pytest.mark.asyncio
async def test_result_summarization_override_reaches_transport(scratch_config, monkeypatch):
    from tldw_chatbook.Web_Scraping import WebSearch_APIs
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib

    scratch_config(
        "[internal_prompts.websearch]\n"
        'result_summarization = "CUSTOM SUMMARIZE {question} :: {content}"\n'
    )

    def fake_chat_api_call(*args, **kwargs):
        # "True" routes into the scrape -> summarize path
        return "Selected Answer: True\nReasoning: relevant"

    async def fake_scrape_article(url):
        return {"content": "scraped body"}

    captured = {}

    def fake_analyze(*args, **kwargs):
        captured["custom_prompt_arg"] = kwargs.get("custom_prompt_arg")
        return "summary"

    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", fake_chat_api_call)
    monkeypatch.setattr(WebSearch_APIs, "scrape_article", fake_scrape_article)
    # analyze is imported inside the function from Summarization_General_Lib
    monkeypatch.setattr(Summarization_General_Lib, "analyze", fake_analyze)

    await WebSearch_APIs.search_result_relevance(
        [{"content": "page text", "url": "https://example.com", "id": "1"}],
        original_question="what is love",
        sub_questions=["a"],
        api_endpoint="openai",
    )

    assert captured.get("custom_prompt_arg", "").startswith("CUSTOM SUMMARIZE what is love")
    assert "scraped body" in captured["custom_prompt_arg"]
