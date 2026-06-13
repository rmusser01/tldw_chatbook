import pytest

from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.Widgets.MindmapViewer import MindmapViewer


@pytest.mark.asyncio
async def test_mindmap_viewer_compact_controls_have_descriptive_tooltips():
    """Mindmap's dense controls should explain what each compact label does."""

    class MindmapViewerApp(App):
        def compose(self) -> ComposeResult:
            yield MindmapViewer()

    app = MindmapViewerApp()

    async with app.run_test() as pilot:
        viewer = app.query_one(MindmapViewer)
        expected_tooltips = {
            "collapse-all": "Collapse every expanded mindmap node.",
            "expand-all": "Expand every mindmap node.",
            "search-btn": "Open mindmap node search.",
            "refresh-btn": "Refresh the current mindmap view.",
            "view-tree": "Show the mindmap as an indented tree.",
            "view-outline": "Show the mindmap as a structured outline.",
            "view-ascii": "Show the mindmap as ASCII art.",
            "search-next": "Jump to the next matching node.",
            "search-prev": "Jump to the previous matching node.",
            "search-close": "Close search and clear current matches.",
        }

        for button_id, expected_tooltip in expected_tooltips.items():
            button = viewer.query_one(f"#{button_id}", Button)
            assert str(button.tooltip) == expected_tooltip
