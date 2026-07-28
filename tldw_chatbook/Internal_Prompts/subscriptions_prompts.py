# tldw_chatbook/Internal_Prompts/subscriptions_prompts.py
"""Subscriptions prompt specs. Defaults moved verbatim from
Subscriptions/content_processor.py (analysis_system, and the four
_build_analysis_prompt branches for rss/atom/json_feed, url_change, podcast,
and the generic fallback).

The four per-type analysis prompts were f-strings in source with complex
interpolations (dict .get() calls, content[:5000]/content[:3000] slices, an
f"{...:.1f}" format spec). Registry templates use simple named tokens instead
({name}, {content}, {change_percentage}, ...); the slicing, `.get()` fallback
lookups, and numeric formatting are precomputed in code before calling
render_internal_prompt. All other text is verbatim.

Removed in TASK-1211: `subscriptions.recursive_summarizer_system` and
`subscriptions.briefing` described `recursive_summarizer.py` and
`briefing_generator.py`, which were retired as unreachable. A spec whose
`used_in` points at a deleted module is worse than no spec -- it presents a
customisation surface for code that cannot run. Their text is recoverable from
git history if the briefing work wants it back.
"""

from .catalog import PromptSpec, register

register(
    PromptSpec(
        id="subscriptions.analysis_system",
        subsystem="subscriptions",
        title="Subscription item analysis — system prompt",
        description="System prompt for LLM analysis of a single subscription item.",
        used_in="Subscriptions/content_processor.py (ContentProcessor._analyze_content, system message)",
        default="You are a helpful assistant that analyzes and summarizes content from subscriptions.",
    )
)

register(
    PromptSpec(
        id="subscriptions.feed_analysis",
        subsystem="subscriptions",
        title="Feed item analysis (rss/atom/json_feed)",
        description="Analysis prompt for a single item from an rss, atom, or json_feed subscription.",
        used_in="Subscriptions/content_processor.py (ContentProcessor._build_analysis_prompt, rss/atom/json_feed branch)",
        default="""Analyze this article from {name}:

Title: {title}
URL: {url}
Published: {published}

Content:
{content}

Please provide:
1. A concise summary (2-3 sentences)
2. Key points or insights (bullet points)
3. Why this might be important or relevant
4. Any action items or implications

Keep the analysis focused and practical.""",
        required_placeholders=("name", "title", "url", "published", "content"),
        contract_note=(
            "The processing_options.analysis_prompt per-subscription "
            "override (code-side .replace channel) outranks the registry."
        ),
    )
)

register(
    PromptSpec(
        id="subscriptions.url_change_analysis",
        subsystem="subscriptions",
        title="URL change analysis",
        description="Analysis prompt for a detected change on a monitored webpage.",
        used_in="Subscriptions/content_processor.py (ContentProcessor._build_analysis_prompt, url_change branch)",
        default="""A monitored webpage has changed:

URL: {url}
Change: {change_percentage}% of content changed

New content:
{content}

Please:
1. Summarize what has changed
2. Highlight the most important updates
3. Assess the significance of these changes
4. Suggest any follow-up actions if needed""",
        required_placeholders=("url", "change_percentage", "content"),
        contract_note=(
            "The processing_options.analysis_prompt per-subscription "
            "override (code-side .replace channel) outranks the registry. "
            "change_percentage is precomputed in code as "
            '`f"{item.get(\'change_percentage\', 0) * 100:.1f}"` before '
            "rendering — the registry template only sees the formatted string."
        ),
    )
)

register(
    PromptSpec(
        id="subscriptions.podcast_analysis",
        subsystem="subscriptions",
        title="Podcast episode analysis",
        description="Analysis prompt for a single podcast episode item.",
        used_in="Subscriptions/content_processor.py (ContentProcessor._build_analysis_prompt, podcast branch)",
        default="""Analyze this podcast episode from {name}:

Title: {title}
Published: {published}

Description:
{content}

Please provide:
1. Episode summary
2. Main topics discussed
3. Key takeaways
4. Whether this episode is worth listening to and why""",
        required_placeholders=("name", "title", "published", "content"),
        contract_note=(
            "The processing_options.analysis_prompt per-subscription "
            "override (code-side .replace channel) outranks the registry."
        ),
    )
)

register(
    PromptSpec(
        id="subscriptions.generic_analysis",
        subsystem="subscriptions",
        title="Generic subscription item analysis",
        description="Fallback analysis prompt for subscription types other than feed/url_change/podcast.",
        used_in="Subscriptions/content_processor.py (ContentProcessor._build_analysis_prompt, else branch)",
        default="""Analyze this content from {name}:

Title: {title}
Type: {type}

Content:
{content}

Please provide a helpful analysis including:
1. Summary
2. Key information
3. Relevance or importance
4. Any recommended actions""",
        required_placeholders=("name", "title", "type", "content"),
        contract_note=(
            "The processing_options.analysis_prompt per-subscription "
            "override (code-side .replace channel) outranks the registry."
        ),
    )
)
