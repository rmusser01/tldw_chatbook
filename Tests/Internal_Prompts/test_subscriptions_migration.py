# Tests/Internal_Prompts/test_subscriptions_migration.py
"""Overrides must reach the subscriptions prompt payloads; caller/per-
subscription override channels still win. The prompt-producing unit
(`_build_analysis_prompt`) is exercised directly since it requires no LLM
call at all.

Notes:

- The real methods are on `ContentProcessor`: `_analyze_content` (async;
  builds the messages list and calls `chat_api_call` WITHOUT awaiting it
  inside an async method, a pre-existing quirk unrelated to this migration)
  and `_build_analysis_prompt` (sync, the actual prompt-producing unit). The
  cases below call `_build_analysis_prompt` directly rather than routing
  through `_analyze_content`.
- The `BriefingGenerator` and `RecursiveSummarizer` cases that used to live
  here were removed with their modules in TASK-1211 -- both were unreachable
  in the shipped app, and their prompt specs were unregistered with them.
"""

import json

import pytest

from tldw_chatbook.Internal_Prompts import get_internal_prompt
from tldw_chatbook.Subscriptions.content_processor import ContentProcessor


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
