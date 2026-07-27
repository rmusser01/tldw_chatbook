import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Watchlists_Modules.watchlists_tab_strip import (
    SectionSelected,
    WatchlistsTabStrip,
)


class _StripApp(App):
    def __init__(self, active="overview"):
        super().__init__()
        self._active = active
        self.selected: list[str] = []

    def compose(self) -> ComposeResult:
        yield WatchlistsTabStrip(active_section=self._active, id="wl-tabs")

    def on_section_selected(self, message: SectionSelected) -> None:
        self.selected.append(message.section_id)


@pytest.mark.asyncio
async def test_every_section_has_a_tab():
    app = _StripApp()
    async with app.run_test():
        for section in ("overview", "sources", "items", "runs", "rules", "notifications"):
            assert app.query(f"#wl-tab-{section}"), f"missing tab for {section}"


@pytest.mark.asyncio
async def test_clicking_a_tab_posts_section_selected():
    app = _StripApp()
    async with app.run_test() as pilot:
        await pilot.click("#wl-tab-runs")
        await pilot.pause()
        assert app.selected == ["runs"]


@pytest.mark.asyncio
async def test_the_active_tab_is_marked():
    app = _StripApp(active="rules")
    async with app.run_test():
        assert app.query_one("#wl-tab-rules").has_class("is-active")
        assert not app.query_one("#wl-tab-runs").has_class("is-active")


@pytest.mark.asyncio
async def test_the_strip_is_one_row_tall():
    app = _StripApp()
    async with app.run_test():
        assert app.query_one(WatchlistsTabStrip).styles.height.value == 1
