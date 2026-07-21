# Tests/Internal_Prompts/test_subscriptions_prompt_parity.py
"""Golden parity: registry defaults render byte-identical text to the
original Subscriptions/content_processor.py, recursive_summarizer.py, and
briefing_generator.py literals.

The four per-type analysis prompts and the briefing prompt were f-strings in
source with complex interpolations (dict .get() fallbacks, content[:5000]/
content[:3000] slices, an f"{...:.1f}" format spec). Each test below embeds
the ORIGINAL f-string verbatim (locals named to match the source variables:
`subscription`, `item`, `content`, `content_summary`), evaluates it, then
precomputes the same values into named tokens exactly the way calling code
must (Task 8) and asserts render_internal_prompt(...) reproduces the
original string exactly. This proves the registry template equals the
original text with only the interpolation syntax swapped out.
"""

from tldw_chatbook.Internal_Prompts import CATALOG, render_internal_prompt


def test_analysis_system_matches_source_literal():
    # content_processor.py:272 — one-liner, zero placeholders.
    original = "You are a helpful assistant that analyzes and summarizes content from subscriptions."
    assert CATALOG["subscriptions.analysis_system"].default == original


def test_feed_analysis_parity():
    # content_processor.py _build_analysis_prompt, rss/atom/json_feed branch.
    subscription = {"name": "Hacker News", "type": "rss"}
    item = {
        "title": "Big Announcement",
        "url": "https://example.com/a",
        "published_date": "2026-07-20",
    }
    content = "A" * 6000  # longer than the 5000-char slice to prove it matters
    original = f"""Analyze this article from {subscription["name"]}:

Title: {item.get("title", "Untitled")}
URL: {item.get("url", "N/A")}
Published: {item.get("published_date", "Unknown")}

Content:
{content[:5000]}

Please provide:
1. A concise summary (2-3 sentences)
2. Key points or insights (bullet points)
3. Why this might be important or relevant
4. Any action items or implications

Keep the analysis focused and practical."""
    rendered = render_internal_prompt(
        "subscriptions.feed_analysis",
        name=subscription["name"],
        title=item.get("title", "Untitled"),
        url=item.get("url", "N/A"),
        published=item.get("published_date", "Unknown"),
        content=content[:5000],
    )
    assert rendered == original


def test_feed_analysis_parity_missing_metadata_defaults():
    # Same branch, but item metadata absent -> the .get() fallbacks fire.
    subscription = {"name": "Empty Feed", "type": "atom"}
    item = {}
    content = "short body"
    original = f"""Analyze this article from {subscription["name"]}:

Title: {item.get("title", "Untitled")}
URL: {item.get("url", "N/A")}
Published: {item.get("published_date", "Unknown")}

Content:
{content[:5000]}

Please provide:
1. A concise summary (2-3 sentences)
2. Key points or insights (bullet points)
3. Why this might be important or relevant
4. Any action items or implications

Keep the analysis focused and practical."""
    rendered = render_internal_prompt(
        "subscriptions.feed_analysis",
        name=subscription["name"],
        title=item.get("title", "Untitled"),
        url=item.get("url", "N/A"),
        published=item.get("published_date", "Unknown"),
        content=content[:5000],
    )
    assert rendered == original


def test_url_change_analysis_parity():
    # content_processor.py _build_analysis_prompt, url_change branch.
    subscription = {"name": "Docs Page", "type": "url_change", "source": "https://example.com/docs"}
    item = {"url": "https://example.com/docs/page", "change_percentage": 0.3217}
    content = "B" * 5500
    original = f"""A monitored webpage has changed:

URL: {item.get("url", subscription["source"])}
Change: {item.get("change_percentage", 0) * 100:.1f}% of content changed

New content:
{content[:5000]}

Please:
1. Summarize what has changed
2. Highlight the most important updates
3. Assess the significance of these changes
4. Suggest any follow-up actions if needed"""
    rendered = render_internal_prompt(
        "subscriptions.url_change_analysis",
        url=item.get("url", subscription["source"]),
        change_percentage=f"{item.get('change_percentage', 0) * 100:.1f}",
        content=content[:5000],
    )
    assert rendered == original


def test_url_change_analysis_parity_url_fallback_and_zero_change():
    # item has no "url" key (falls back to subscription["source"]) and no
    # "change_percentage" key (defaults to 0).
    subscription = {"name": "Docs Page", "type": "url_change", "source": "https://example.com/docs"}
    item = {}
    content = "unchanged-ish content"
    original = f"""A monitored webpage has changed:

URL: {item.get("url", subscription["source"])}
Change: {item.get("change_percentage", 0) * 100:.1f}% of content changed

New content:
{content[:5000]}

Please:
1. Summarize what has changed
2. Highlight the most important updates
3. Assess the significance of these changes
4. Suggest any follow-up actions if needed"""
    rendered = render_internal_prompt(
        "subscriptions.url_change_analysis",
        url=item.get("url", subscription["source"]),
        change_percentage=f"{item.get('change_percentage', 0) * 100:.1f}",
        content=content[:5000],
    )
    assert rendered == original


def test_podcast_analysis_parity():
    # content_processor.py _build_analysis_prompt, podcast branch. Note the
    # content slice here is [:3000], not [:5000].
    subscription = {"name": "Tech Talk Weekly", "type": "podcast"}
    item = {"title": "Episode 42", "published_date": "2026-07-15"}
    content = "C" * 4000
    original = f"""Analyze this podcast episode from {subscription["name"]}:

Title: {item.get("title", "Untitled")}
Published: {item.get("published_date", "Unknown")}

Description:
{content[:3000]}

Please provide:
1. Episode summary
2. Main topics discussed
3. Key takeaways
4. Whether this episode is worth listening to and why"""
    rendered = render_internal_prompt(
        "subscriptions.podcast_analysis",
        name=subscription["name"],
        title=item.get("title", "Untitled"),
        published=item.get("published_date", "Unknown"),
        content=content[:3000],
    )
    assert rendered == original


def test_generic_analysis_parity():
    # content_processor.py _build_analysis_prompt, else (generic) branch.
    subscription = {"name": "Custom Source", "type": "custom_scraper"}
    item = {"title": "Some Item"}
    content = "D" * 5200
    original = f"""Analyze this content from {subscription["name"]}:

Title: {item.get("title", "Untitled")}
Type: {subscription["type"]}

Content:
{content[:5000]}

Please provide a helpful analysis including:
1. Summary
2. Key information
3. Relevance or importance
4. Any recommended actions"""
    rendered = render_internal_prompt(
        "subscriptions.generic_analysis",
        name=subscription["name"],
        title=item.get("title", "Untitled"),
        type=subscription["type"],
        content=content[:5000],
    )
    assert rendered == original


def test_recursive_summarizer_system_matches_source_literal():
    # recursive_summarizer.py _get_system_prompt — static literal, zero
    # placeholders. Note the trailing space after "length." before the
    # blank line, preserved verbatim from source.
    original = """You are an expert content summarizer. Your task is to create clear, accurate summaries that preserve the key information while significantly reducing length. 

Key principles:
1. Maintain factual accuracy
2. Preserve the most important information
3. Use clear, concise language
4. Maintain logical flow
5. Respect the requested token limit"""
    assert CATALOG["subscriptions.recursive_summarizer_system"].default == original


def test_briefing_parity():
    # briefing_generator.py _generate_sections_with_llm, default (non-custom)
    # prompt branch.
    content_summary = "Source A: 3 new items.\nSource B: 1 new item with {braces} in it."
    original = f"""Analyze the following content from various subscriptions and generate a comprehensive briefing:

{content_summary}

Please provide:
1. Executive Summary (2-3 paragraphs highlighting the most important developments)
2. Key Insights (bullet points of significant findings or patterns)
3. Trending Topics (identify common themes across sources)
4. Recommended Actions (actionable items based on the content)

Format each section clearly with appropriate headers."""
    rendered = render_internal_prompt(
        "subscriptions.briefing", content_summary=content_summary
    )
    assert rendered == original


def test_briefing_contract_note_pins_section_labels():
    # _parse_llm_sections substring-matches these four labels; if the
    # default text or the contract note ever drops one, this test catches it.
    default_text = CATALOG["subscriptions.briefing"].default
    for label in (
        "Executive Summary",
        "Key Insights",
        "Trending Topics",
        "Recommended Actions",
    ):
        assert label in default_text
        assert label in (CATALOG["subscriptions.briefing"].contract_note or "")
