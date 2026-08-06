"""Regression tests for legacy WebSearchTool result mapping (task-1341)."""

from tldw_chatbook.Tools.web_search_tool import WebSearchTool


def _real_payload():
    """The shape perform_websearch actually returns (WebSearch_APIs.py:
    process_web_search_results): body text is top-level `content`;
    `snippet` exists only inside `metadata`."""
    return {
        "results": [
            {
                "title": "Result 1",
                "url": "https://example.com/1",
                "content": "the actual body text",
                "metadata": {"snippet": "raw engine snippet"},
            },
            {
                "title": "Result 2",
                "url": "https://example.com/2",
                "content": "body two",
                "metadata": {},
            },
        ]
    }


def test_snippet_falls_back_to_content(monkeypatch):
    tool = WebSearchTool()
    monkeypatch.setattr(
        "tldw_chatbook.Tools.web_search_tool.perform_websearch",
        lambda *a, **k: _real_payload(),
    )
    import asyncio

    out = asyncio.run(tool.execute(query="test"))
    snippets = [r["snippet"] for r in out["results"]]
    # Top-level `snippet` is absent in the real payload shape, so both
    # results fall back to `content` (previously: always "No description
    # available").
    assert snippets == ["the actual body text", "body two"]
