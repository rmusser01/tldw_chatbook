"""Row activation -> modal -> save/reset persists and refreshes the row."""

import pytest
from textual.app import App

from tldw_chatbook.Internal_Prompts import authoring
from tldw_chatbook.Widgets.settings_internal_prompts_panel import InternalPromptsPanel


class _Host(App):
    def compose(self):
        yield InternalPromptsPanel(id="p")


@pytest.mark.asyncio
async def test_save_via_panel_persists_and_marks_row(scratch_config):
    scratch_config("")
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(InternalPromptsPanel)
        # drive the panel's save path directly (modal UI covered in Task 2)
        await panel._apply_editor_result(
            "agents.subagent_system", {"action": "save", "text": "PANEL EDIT"}
        )
        await pilot.pause()
        assert authoring.override_state("agents.subagent_system").active_text == "PANEL EDIT"
        row = app.query_one("#prompt-row-agents__subagent_system")
        assert "row-customized" in row.classes


@pytest.mark.asyncio
async def test_reset_via_panel_clears_override(scratch_config):
    scratch_config('[internal_prompts.agents]\nsubagent_system = "X"\n')
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(InternalPromptsPanel)
        await panel._apply_editor_result(
            "agents.subagent_system", {"action": "reset"}
        )
        await pilot.pause()
        assert authoring.override_state("agents.subagent_system").customized is False
        row = app.query_one("#prompt-row-agents__subagent_system")
        assert "row-customized" not in row.classes
