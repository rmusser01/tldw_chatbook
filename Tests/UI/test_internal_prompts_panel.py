"""Panel browse: renders grouped rows, badges reflect config, search filters
without a rebuild."""

import pytest
from textual.app import App

from tldw_chatbook.Internal_Prompts.catalog import CATALOG
from tldw_chatbook.Widgets.settings_internal_prompts_panel import InternalPromptsPanel


class _Host(App):
    def compose(self):
        yield InternalPromptsPanel(id="p")


@pytest.mark.asyncio
async def test_renders_one_row_per_prompt(scratch_config):
    scratch_config("")
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        rows = app.query(".internal-prompt-row")
        assert len(rows) == len(CATALOG)


@pytest.mark.asyncio
async def test_customized_badge_reflects_override(scratch_config):
    scratch_config(
        '[internal_prompts.agents]\nsubagent_system = "X"\n'
    )
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        row = app.query_one("#prompt-row-agents__subagent_system")
        assert "row-customized" in row.classes


@pytest.mark.asyncio
async def test_search_hides_nonmatching_rows_without_rebuild(scratch_config):
    scratch_config("")
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        rows_before = {r.id for r in app.query(".internal-prompt-row")}
        search = app.query_one("#internal-prompts-search")
        search.value = "answer synthesis"
        await pilot.pause()
        rows_after = {r.id for r in app.query(".internal-prompt-row")}
        assert rows_before == rows_after  # same widgets, not rebuilt
        visible = [r for r in app.query(".internal-prompt-row") if r.display]
        assert all("answer" in r.tooltip.lower() or "answer" in str(r.label).lower()
                   for r in visible)
        assert len(visible) < len(rows_before)
