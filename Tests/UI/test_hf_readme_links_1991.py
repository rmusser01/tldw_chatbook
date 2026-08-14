"""TASK-1991: HF README viewer resolves relative links and #anchors.

Frogmouth-style tiers: absolute URLs open in the browser, relative paths
join the repo's blob root, #anchors scroll the rendered document, anything
else notifies instead of failing silently.
"""

import webbrowser

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Markdown

from tldw_chatbook.Widgets.HuggingFace.model_card_viewer import (
    ModelCardViewer,
    resolve_readme_href,
)


def test_resolver_tiers():
    # Anchors.
    assert resolve_readme_href("#quickstart", None) == ("anchor", "quickstart")
    # Absolute URLs and mailto pass through.
    assert resolve_readme_href("https://x.y/z", None) == ("browser", "https://x.y/z")
    assert resolve_readme_href("http://x.y", "o/m") == ("browser", "http://x.y")
    assert resolve_readme_href("mailto:a@b.c", None) == ("browser", "mailto:a@b.c")
    # Protocol-relative gets https.
    assert resolve_readme_href("//host/p", None) == ("browser", "https://host/p")
    # Relative paths join the repo blob root.
    assert resolve_readme_href("docs/usage.md", "org/model") == (
        "browser",
        "https://huggingface.co/org/model/blob/main/docs/usage.md",
    )
    assert resolve_readme_href("./assets/img.png", "org/model") == (
        "browser",
        "https://huggingface.co/org/model/blob/main/assets/img.png",
    )
    # Relative without a known repo is unresolvable, not a guess.
    assert resolve_readme_href("docs/usage.md", None)[0] == "unresolvable"
    # Dangerous/unknown schemes never open.
    assert resolve_readme_href("javascript:alert(1)", "o/m")[0] == "unresolvable"
    assert resolve_readme_href("file:///etc/passwd", "o/m")[0] == "unresolvable"
    assert resolve_readme_href("", "o/m")[0] == "unresolvable"


class ViewerHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ModelCardViewer(id="viewer")


README = "# Title\n\nIntro.\n\n## Quickstart\n\nSteps here.\n"


@pytest.mark.asyncio
async def test_link_clicks_follow_the_tiers(monkeypatch):
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    # Keep watch_model_info from hitting the network.
    monkeypatch.setattr(ModelCardViewer, "load_model_details", lambda self, r: None)

    app = ViewerHarness()
    async with app.run_test() as pilot:
        viewer = app.query_one(ModelCardViewer)
        viewer.model_info = {"id": "test-org/test-model"}
        viewer.readme_content = README
        await pilot.pause()

        md = viewer.query_one("#readme-display", Markdown)
        notices = []
        monkeypatch.setattr(
            viewer, "notify", lambda msg, **kw: notices.append((msg, kw))
        )

        # Absolute URL -> browser.
        viewer._handle_readme_link(Markdown.LinkClicked(md, "https://example.com"))
        assert opened == ["https://example.com"]

        # Relative path -> repo blob root.
        viewer._handle_readme_link(Markdown.LinkClicked(md, "docs/usage.md"))
        assert opened[-1] == (
            "https://huggingface.co/test-org/test-model/blob/main/docs/usage.md"
        )

        # Present anchor -> scrolls, no warning.
        before = len(notices)
        viewer._handle_readme_link(Markdown.LinkClicked(md, "#quickstart"))
        await pilot.pause()
        assert all("No such section" not in n[0] for n in notices[before:])

        # Missing anchor -> visible warning.
        viewer._handle_readme_link(Markdown.LinkClicked(md, "#nope"))
        assert any("No such section" in n[0] for n in notices)

        # Unsupported scheme -> warning, nothing opened.
        count = len(opened)
        viewer._handle_readme_link(Markdown.LinkClicked(md, "javascript:alert(1)"))
        assert len(opened) == count
        assert any("Cannot handle this link" in n[0] for n in notices)
