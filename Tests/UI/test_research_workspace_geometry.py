"""Production-CSS geometry evidence for the Research Workspace shell."""

from __future__ import annotations

from html import unescape
import re
from types import SimpleNamespace

import pytest
from textual.widgets import Button
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.app import TldwCli


class _ProductionWorkspaceHarness(ConsolidatedCSSApp):
    CSS_PATH = TldwCli.CSS_PATH

    async def on_mount(self) -> None:
        from tldw_chatbook.UI.Screens.research_workspace_screen import (
            ResearchWorkspaceScreen,
        )

        await self.push_screen(ResearchWorkspaceScreen(SimpleNamespace()))


class _ProductionRunsHarness(ConsolidatedCSSApp):
    CSS_PATH = TldwCli.CSS_PATH

    async def on_mount(self) -> None:
        from tldw_chatbook.UI.Screens.research_screen import ResearchScreen

        await self.push_screen(ResearchScreen(SimpleNamespace()))


def _painted_text(svg: str) -> str:
    """Flatten the screenshot's encoded SVG text nodes for frame assertions."""
    plain = unescape(re.sub(r"<[^>]+>", " ", svg)).replace("\N{NO-BREAK SPACE}", " ")
    return " ".join(plain.split())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "visible_panes"),
    [
        ((160, 40), ("sources", "chat", "studio")),
        ((120, 30), ("sources", "chat")),
        ((100, 30), ("sources", "chat")),
        ((84, 24), ("chat",)),
        ((80, 24), ("chat",)),
        ((60, 20), ("chat",)),
    ],
)
async def test_production_hierarchy_paints_contained_workspace_frames(
    size: tuple[int, int], visible_panes: tuple[str, ...]
) -> None:
    app = _ProductionWorkspaceHarness()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        grid = screen.query_one("#research-workspace-grid")
        painted = _painted_text(app.export_screenshot(simplify=True))

        assert "Research Workspace" in painted
        assert "Workspace data:" in painted
        assert "Processing:" in painted
        assert "Foundation ready" in painted
        for pane_id in visible_panes:
            pane = screen.query_one(f"#research-{pane_id}-pane")
            assert pane.display
            assert pane.region.width > 0 and pane.region.height > 0
            assert grid.content_region.contains_region(pane.region), (size, pane_id)
            assert screen.region.contains_region(pane.region), (size, pane_id)

        for pane_id in {"sources", "chat", "studio"} - set(visible_panes):
            assert not screen.query_one(f"#research-{pane_id}-pane").display


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size", [(160, 40), (120, 30), (100, 30), (84, 24), (80, 24), (60, 20)]
)
async def test_active_sources_keeps_essential_controls_painted_and_reachable(
    size: tuple[int, int],
) -> None:
    app = _ProductionWorkspaceHarness()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        sources = screen.query_one("#research-sources-pane")
        if not sources.display:
            screen.query_one("#research-pane-mode-sources", Button).press()
            await pilot.pause()
        painted = _painted_text(app.export_screenshot(simplify=True))

        assert sources.display
        assert screen.region.contains_region(sources.region)
        for text in (
            "Add Sources",
            "Quick add URL",
            "Filter current page",
            "Select all",
            "No workspace selected",
        ):
            assert text in painted, (size, text, painted)
        for widget_id in (
            "research-source-add",
            "research-source-search",
            "research-source-select-all",
            "research-source-recovery",
        ):
            widget = screen.query_one(f"#{widget_id}")
            assert (
                widget.display and widget.region.width > 0 and widget.region.height > 0
            )
            assert sources.region.overlaps(widget.region), (
                size,
                widget_id,
                widget.region,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 40), (60, 20)])
async def test_production_runs_window_stays_inside_remaining_screen_content(
    size: tuple[int, int],
) -> None:
    """The shared mode strip must not make the legacy 100%-high Runs window clip."""
    app = _ProductionRunsHarness()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        content = screen.query_one("#screen-content")
        mode_strip = screen.query_one("#research-mode-strip")
        window = screen.query_one("#research-window")

        assert window.region.y >= mode_strip.region.bottom
        assert content.content_region.contains_region(window.region), (
            size,
            content.content_region,
            window.region,
        )


@pytest.mark.asyncio
async def test_wide_chat_uses_grid_except_two_fixed_reveal_handles() -> None:
    from textual.widgets import Button

    app = _ProductionWorkspaceHarness()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        screen.query_one("#research-sources-collapse", Button).press()
        screen.query_one("#research-studio-collapse", Button).press()
        await pilot.pause()

        grid = screen.query_one("#research-workspace-grid")
        chat = screen.query_one("#research-chat-pane")
        sources_handle = screen.query_one("#research-sources-handle")
        studio_handle = screen.query_one("#research-studio-handle")

        assert not screen.query_one("#research-sources-pane").display
        assert chat.display
        assert not screen.query_one("#research-studio-pane").display
        assert sources_handle.region.width == 4
        assert studio_handle.region.width == 4
        assert chat.region.width == grid.content_region.width - 8
