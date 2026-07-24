# Tests/Internal_Prompts/test_subscriptions_migration.py
"""Overrides must reach the subscriptions prompt payloads; caller/per-
subscription override channels still win. Fakes are used only where a real
LLM call would otherwise be required; the prompt-producing units
(`_build_analysis_prompt`, `_get_system_prompt`) are exercised directly
where possible since they require no LLM call at all.

Adaptations from the brief's skeleton (see task-8-report.md for detail):

- The brief names `ContentSummarizer.analyze_content` (from the scout
  notes). Reading Subscriptions/content_processor.py turned up that this is
  approximate: the real methods are on `ContentProcessor` —
  `_analyze_content` (async; builds the messages list and calls
  `chat_api_call` WITHOUT awaiting it inside an async method, a pre-
  existing, out-of-scope quirk unrelated to this migration) and
  `_build_analysis_prompt` (sync, the actual prompt-producing unit, no LLM
  call at all). Per the brief's disproportionate-effort escape hatch, the
  content_processor cases below call `_build_analysis_prompt` directly
  rather than routing through `_analyze_content`.
- `BriefingGenerator._generate_sections_with_llm` is exercised through a
  real `BriefingGenerator` instance constructed with `subscriptions_db=None`
  (the method under test never touches `self.db`) and an async
  `chat_api_call` fake installed in `briefing_generator`'s module namespace
  (that call site does `await chat_api_call(...)`).
- No `asyncio_mode` is configured for this repo's pytest (strict pytest-
  asyncio default), so the one async-under-test case below follows the
  `@pytest.mark.asyncio` pattern already used in
  Tests/RAG/test_reranker_internal_prompts.py rather than a bare `async def
  test_...`, which would silently no-op under strict mode.
"""

import json

import pytest

from tldw_chatbook.Internal_Prompts import get_internal_prompt
from tldw_chatbook.Subscriptions.briefing_generator import BriefingGenerator
from tldw_chatbook.Subscriptions.content_processor import ContentProcessor
from tldw_chatbook.Subscriptions.recursive_summarizer import RecursiveSummarizer


def _feed_item():
    return {
        "title": "Some Title",
        "url": "https://example.com/a",
        "published_date": "2026-01-01",
    }


def _feed_subscription(processing_options=None):
    sub = {"name": "Example Feed", "type": "rss", "source": "https://example.com/feed"}
    if processing_options is not None:
        sub["processing_options"] = processing_options
    return sub


# ---------------------------------------------------------------------------
# (a) content_processor: three-way precedence for the per-item analysis
#     prompt (registry default -> registry override -> per-subscription
#     processing_options.analysis_prompt override, which must still win).
# ---------------------------------------------------------------------------


def test_feed_analysis_default_reaches_prompt():
    processor = ContentProcessor()
    prompt = processor._build_analysis_prompt(
        "Some content body", _feed_item(), _feed_subscription()
    )
    assert "Analyze this article from Example Feed:" in prompt
    assert "Title: Some Title" in prompt
    assert "URL: https://example.com/a" in prompt
    assert "Published: 2026-01-01" in prompt
    assert "Some content body" in prompt


def test_feed_analysis_registry_override_reaches_prompt(scratch_config):
    scratch_config(
        "[internal_prompts.subscriptions]\n"
        'feed_analysis = "CUSTOM {name} | {title} | {url} | {published} | {content}"\n'
    )
    processor = ContentProcessor()
    prompt = processor._build_analysis_prompt(
        "Some content body", _feed_item(), _feed_subscription()
    )
    assert (
        prompt
        == "CUSTOM Example Feed | Some Title | https://example.com/a | 2026-01-01 | Some content body"
    )


def test_feed_analysis_per_subscription_override_wins_over_registry_override(
    scratch_config,
):
    """The processing_options.analysis_prompt code-side .replace channel is
    the highest-priority override, ahead of even a registry override."""
    scratch_config(
        "[internal_prompts.subscriptions]\n"
        'feed_analysis = "CUSTOM {name} | {title} | {url} | {published} | {content}"\n'
    )
    processor = ContentProcessor()
    processing_options = json.dumps(
        {"analysis_prompt": "PER-SUB {content} / {title} / {source} / {url}"}
    )
    prompt = processor._build_analysis_prompt(
        "Some content body",
        _feed_item(),
        _feed_subscription(processing_options=processing_options),
    )
    assert (
        prompt
        == "PER-SUB Some content body / Some Title / Example Feed / https://example.com/a"
    )
    assert "CUSTOM" not in prompt


# ---------------------------------------------------------------------------
# Sanity coverage for the other three content_processor branches (not
# required by the brief's three cases, but cheap and proves each branch's
# precomputed tokens line up with its registered spec's required
# placeholders).
# ---------------------------------------------------------------------------


def test_url_change_analysis_default_reaches_prompt():
    processor = ContentProcessor()
    item = {"url": "https://example.com/page", "change_percentage": 0.4321}
    subscription = {"type": "url_change", "source": "https://example.com/page"}
    prompt = processor._build_analysis_prompt("New content body", item, subscription)
    assert "URL: https://example.com/page" in prompt
    assert "Change: 43.2% of content changed" in prompt
    assert "New content body" in prompt


def test_url_change_analysis_url_falls_back_to_subscription_source():
    processor = ContentProcessor()
    item = {"change_percentage": 0.1}  # no "url" key
    subscription = {"type": "url_change", "source": "https://example.com/fallback"}
    prompt = processor._build_analysis_prompt("content", item, subscription)
    assert "URL: https://example.com/fallback" in prompt


def test_podcast_analysis_default_reaches_prompt():
    processor = ContentProcessor()
    item = {"title": "Episode 1", "published_date": "2026-02-02"}
    subscription = {"name": "My Podcast", "type": "podcast"}
    # Non-periodic content (each word index is unique) so a slice-boundary
    # check below can't coincidentally match elsewhere in a repeating string.
    long_description = " ".join(f"word{i}" for i in range(1000))
    prompt = processor._build_analysis_prompt(long_description, item, subscription)
    assert "Analyze this podcast episode from My Podcast:" in prompt
    assert "Title: Episode 1" in prompt
    assert "Published: 2026-02-02" in prompt
    # podcast branch slices to [:3000], not [:5000]
    assert len(long_description[:3000]) == 3000
    assert long_description[:3000] in prompt
    assert long_description[3000:3050] not in prompt


def test_generic_analysis_default_reaches_prompt():
    processor = ContentProcessor()
    item = {"title": "Some Item"}
    subscription = {"name": "Generic Source", "type": "webhook"}
    prompt = processor._build_analysis_prompt("Generic content", item, subscription)
    assert "Analyze this content from Generic Source:" in prompt
    assert "Title: Some Item" in prompt
    assert "Type: webhook" in prompt
    assert "Generic content" in prompt


def test_analysis_system_prompt_default_and_override(scratch_config):
    assert get_internal_prompt("subscriptions.analysis_system") == (
        "You are a helpful assistant that analyzes and summarizes content "
        "from subscriptions."
    )
    scratch_config(
        '[internal_prompts.subscriptions]\nanalysis_system = "CUSTOM SYSTEM ROLE"\n'
    )
    assert get_internal_prompt("subscriptions.analysis_system") == "CUSTOM SYSTEM ROLE"


# ---------------------------------------------------------------------------
# (c) recursive summarizer system prompt override reaches its payload.
# ---------------------------------------------------------------------------


def test_recursive_summarizer_system_prompt_default():
    summarizer = RecursiveSummarizer()
    assert "expert content summarizer" in summarizer._get_system_prompt()


def test_recursive_summarizer_system_prompt_override_reaches_payload(scratch_config):
    scratch_config(
        "[internal_prompts.subscriptions]\n"
        'recursive_summarizer_system = "CUSTOM RECURSIVE SYSTEM PROMPT"\n'
    )
    summarizer = RecursiveSummarizer()
    assert summarizer._get_system_prompt() == "CUSTOM RECURSIVE SYSTEM PROMPT"


# ---------------------------------------------------------------------------
# (b) briefing generator: else-branch override reaches the LLM payload, and
#     _parse_llm_sections still parses a canned four-section response
#     (contract intact).
# ---------------------------------------------------------------------------

_CANNED_LLM_RESPONSE = """Executive Summary
This is the top headline of the day, in a single paragraph.

Key Insights
- insight one
- insight two

Trending Topics
- topic one

Recommended Actions
- action one
"""
# Note: `_parse_llm_sections` re-triggers a new section on ANY line
# containing the bare word "summary" (not just a header line), so the
# executive-summary body text above deliberately avoids that word --
# otherwise the parser would treat the body line itself as a second header
# and the section's content would come out empty. This is a pre-existing
# quirk of `_parse_llm_sections`, left untouched (out of scope).


def _briefing_items():
    return [
        {
            "title": "Item One",
            "subscription_name": "Feed A",
            "content": "Some article content here.",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "title": "Item Two",
            "subscription_name": "Feed B",
            "content": "Other article content.",
            "created_at": "2026-01-02T00:00:00Z",
        },
    ]


@pytest.mark.asyncio
async def test_briefing_default_prompt_reaches_llm_and_parse_contract_intact(
    monkeypatch,
):
    import tldw_chatbook.Subscriptions.briefing_generator as bg_mod

    generator = BriefingGenerator(
        subscriptions_db=None, llm_provider="test-provider", llm_model="test-model"
    )
    items = _briefing_items()
    items_by_source = generator._group_by_source(items)

    captured = {}

    async def fake_chat_api_call(**kwargs):
        captured["messages"] = kwargs["messages_payload"]
        return {"content": _CANNED_LLM_RESPONSE}

    monkeypatch.setattr(bg_mod, "chat_api_call", fake_chat_api_call)
    sections = await generator._generate_sections_with_llm(items, items_by_source, None)

    user_prompt = captured["messages"][1]["content"]
    assert (
        "Analyze the following content from various subscriptions and "
        "generate a comprehensive briefing:" in user_prompt
    )
    assert "Feed A" in user_prompt
    assert "Item One" in user_prompt

    # _parse_llm_sections contract intact: the four labels still route to
    # their sections.
    assert "top headline of the day" in sections["executive_summary"]
    assert "insight one" in sections["insights_section"]
    assert "topic one" in sections["trends_section"]
    assert "action one" in sections["actions_section"]
    # _format_sources_section still ran on top.
    assert "sources_section" in sections


@pytest.mark.asyncio
async def test_briefing_registry_override_reaches_llm_payload(
    scratch_config, monkeypatch
):
    import tldw_chatbook.Subscriptions.briefing_generator as bg_mod

    scratch_config(
        "[internal_prompts.subscriptions]\n"
        'briefing = "CUSTOM BRIEFING PROMPT ::: {content_summary}"\n'
    )
    generator = BriefingGenerator(
        subscriptions_db=None, llm_provider="test-provider", llm_model="test-model"
    )
    items = _briefing_items()
    items_by_source = generator._group_by_source(items)

    captured = {}

    async def fake_chat_api_call(**kwargs):
        captured["messages"] = kwargs["messages_payload"]
        return {"content": _CANNED_LLM_RESPONSE}

    monkeypatch.setattr(bg_mod, "chat_api_call", fake_chat_api_call)
    await generator._generate_sections_with_llm(items, items_by_source, None)

    user_prompt = captured["messages"][1]["content"]
    assert user_prompt.startswith("CUSTOM BRIEFING PROMPT ::: ")
    assert "Feed A" in user_prompt


@pytest.mark.asyncio
async def test_briefing_custom_prompt_argument_wins_over_registry_override(
    scratch_config, monkeypatch
):
    """The generate_briefing(analysis_prompt=...) caller channel (passed
    through as `custom_prompt`) still outranks a registry override."""
    import tldw_chatbook.Subscriptions.briefing_generator as bg_mod

    scratch_config(
        "[internal_prompts.subscriptions]\n"
        'briefing = "SHOULD NOT APPEAR ::: {content_summary}"\n'
    )
    generator = BriefingGenerator(
        subscriptions_db=None, llm_provider="test-provider", llm_model="test-model"
    )
    items = _briefing_items()
    items_by_source = generator._group_by_source(items)

    captured = {}

    async def fake_chat_api_call(**kwargs):
        captured["messages"] = kwargs["messages_payload"]
        return {"content": _CANNED_LLM_RESPONSE}

    monkeypatch.setattr(bg_mod, "chat_api_call", fake_chat_api_call)
    await generator._generate_sections_with_llm(
        items, items_by_source, "CALLER PROMPT {content}"
    )

    user_prompt = captured["messages"][1]["content"]
    assert user_prompt.startswith("CALLER PROMPT ")
    assert "SHOULD NOT APPEAR" not in user_prompt


def test_briefing_system_role_still_a_literal_not_registry():
    """Scope note: the briefing system role is deliberately deferred, not
    migrated in this task. Documents that current behavior (a hardcoded
    literal) is unchanged."""
    import inspect

    import tldw_chatbook.Subscriptions.briefing_generator as bg_mod

    source = inspect.getsource(bg_mod.BriefingGenerator._generate_sections_with_llm)
    # Still an inline string literal for the "content" key, not a registry
    # call -- if this had been migrated, the value would instead be a
    # get_internal_prompt(...)/render_internal_prompt(...) call.
    assert (
        '"content": "You are an expert analyst creating executive briefings '
        'from aggregated content.",' in source
    )
    # Exactly one render_internal_prompt call in this method: the migrated
    # analysis/user prompt. The system role above is untouched (asserted above).
    assert source.count("render_internal_prompt") == 1
