"""TASK-1992: heading-tree table of contents for the HF README pane.

The TOC is hidden by default (compact layouts stay clean), toggled by the
toolbar button, populated from the rendered README's headings, and selecting
an entry scrolls the document to that heading.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Markdown, TabbedContent
from textual.widgets.markdown import MarkdownTableOfContents

from tldw_chatbook.Widgets.HuggingFace.model_card_viewer import ModelCardViewer


class ViewerHarness(App):
    CSS = "ModelCardViewer { height: 40; }"

    def compose(self) -> ComposeResult:
        yield ModelCardViewer(id="viewer")


LONG_README = "# Title\n\n" + "\n\n".join(
    f"## Section {i}\n\n" + ("filler line\n" * 8) for i in range(1, 9)
)


@pytest.mark.asyncio
async def test_toc_hidden_by_default_and_toggles():
    app = ViewerHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        viewer = app.query_one(ModelCardViewer)
        toc = viewer.query_one("#readme-toc", MarkdownTableOfContents)
        assert toc.display is False

        viewer.query_one("#readme-toc-toggle", Button).press()
        await pilot.pause()
        assert toc.display is True

        viewer.query_one("#readme-toc-toggle", Button).press()
        await pilot.pause()
        assert toc.display is False


@pytest.mark.asyncio
async def test_toc_populates_from_readme_headings():
    app = ViewerHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        viewer = app.query_one(ModelCardViewer)
        viewer.readme_content = LONG_README
        # TableOfContentsUpdated arrives after the markdown finishes parsing.
        for _ in range(20):
            await pilot.pause()
            toc = viewer.query_one("#readme-toc", MarkdownTableOfContents)
            if toc.table_of_contents:
                break
        assert toc.table_of_contents, "TOC never received the heading tree"
        headings = [entry[1] for entry in toc.table_of_contents]
        assert "Title" in headings
        assert "Section 8" in headings


@pytest.mark.asyncio
async def test_toc_selection_scrolls_document():
    app = ViewerHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        viewer = app.query_one(ModelCardViewer)
        # The README pane must be the active tab to have scroll geometry.
        viewer.query_one(TabbedContent).active = "readme-tab"
        viewer.readme_content = LONG_README
        for _ in range(20):
            await pilot.pause()
            toc = viewer.query_one("#readme-toc", MarkdownTableOfContents)
            if toc.table_of_contents:
                break
        assert toc.table_of_contents

        # Late heading's block id from the TOC payload itself.
        block_id = next(
            entry[2] for entry in toc.table_of_contents if entry[1] == "Section 8"
        )
        md = viewer.query_one("#readme-display", Markdown)
        scroll = viewer.query_one("#readme-scroll")
        assert scroll.scroll_y == 0

        viewer.on_markdown_table_of_contents_selected(
            Markdown.TableOfContentsSelected(md, block_id)
        )
        for _ in range(10):
            await pilot.pause()
            if scroll.scroll_y > 0:
                break
        assert scroll.scroll_y > 0, "selection did not scroll the README"
