"""TASK-1993: markdown previews consume YAML front matter instead of rendering it.

Covers the shared factory (present/absent dependency) and its wiring into the
HF README display and the Library note preview.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Markdown

import tldw_chatbook.Utils.markdown_parsing as markdown_parsing
from tldw_chatbook.Utils.markdown_parsing import front_matter_parser_factory
from tldw_chatbook.Widgets.HuggingFace.model_card_viewer import ModelCardViewer

FRONT_MATTERED = "---\ntags:\n  - unsloth\nlicense: apache-2.0\n---\n# Title\n\nBody.\n"


def test_factory_parser_consumes_front_matter():
    factory = front_matter_parser_factory()
    assert factory is not None, "mdit-py-plugins installed in the dev venv"
    parser = factory()
    tokens = parser.parse(FRONT_MATTERED)
    assert tokens[0].type == "front_matter"
    # The document proper still parses normally.
    assert any(t.type == "heading_open" for t in tokens)
    # And gfm-like table support survives the plugin chain.
    table_tokens = parser.parse("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert any(t.type == "table_open" for t in table_tokens)


def test_factory_degrades_to_none_without_dependency(monkeypatch):
    monkeypatch.setattr(
        markdown_parsing, "check_dependency", lambda *a, **k: False
    )
    assert front_matter_parser_factory() is None


class ViewerHarness(App):
    def compose(self) -> ComposeResult:
        yield ModelCardViewer(id="viewer")


@pytest.mark.asyncio
async def test_readme_display_strips_front_matter():
    app = ViewerHarness()
    async with app.run_test() as pilot:
        viewer = app.query_one(ModelCardViewer)
        viewer.readme_content = FRONT_MATTERED
        await pilot.pause()
        await pilot.pause()
        md = viewer.query_one("#readme-display", Markdown)
        rendered_text = " ".join(
            getattr(getattr(block, "_content", None), "plain", "") or ""
            for block in md.walk_children()
        )
        assert "apache-2.0" not in rendered_text
        assert "unsloth" not in rendered_text
        assert "Title" in rendered_text
