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
