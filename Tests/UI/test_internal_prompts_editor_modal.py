"""Editor modal: dismiss values, Save validation, preview presence."""

import pytest
from textual.app import App

from tldw_chatbook.Internal_Prompts.catalog import CATALOG
from tldw_chatbook.Widgets.settings_internal_prompts_editor_modal import (
    InternalPromptEditorModal,
)


class _Host(App):
    def __init__(self, spec, active):
        super().__init__()
        self._spec, self._active = spec, active
        self.result = "UNSET"

    def on_mount(self):
        def cb(value):
            self.result = value
        self.push_screen(
            InternalPromptEditorModal(spec=self._spec, active_text=self._active),
            cb,
        )


@pytest.mark.asyncio
async def test_cancel_returns_none():
    spec = CATALOG["agents.subagent_system"]
    app = _Host(spec, spec.default)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


@pytest.mark.asyncio
async def test_save_returns_action_and_text():
    spec = CATALOG["agents.subagent_system"]
    app = _Host(spec, spec.default)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#internal-prompt-editor-text").text = "EDITED TEXT"
        await modal._save_from_test()  # helper invokes the same path as the Save button
        await pilot.pause()
    assert app.result == {"action": "save", "text": "EDITED TEXT"}


@pytest.mark.asyncio
async def test_save_blocks_on_missing_required_placeholder():
    spec = CATALOG["rag_reranker.pointwise_template"]  # has required placeholders
    app = _Host(spec, spec.default)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#internal-prompt-editor-text").text = "no tokens here"
        await modal._save_from_test()
        await pilot.pause()
        assert app.result == "UNSET"  # did NOT dismiss
        assert modal.query_one("#internal-prompt-editor-error").renderable != ""


@pytest.mark.asyncio
async def test_preview_present_for_templated_absent_for_plain():
    templated = CATALOG["rag_reranker.pointwise_template"]
    app = _Host(templated, templated.default)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query("#internal-prompt-editor-preview")
    plain = CATALOG["agents.subagent_system"]
    app2 = _Host(plain, plain.default)
    async with app2.run_test() as pilot:
        await pilot.pause()
        assert not app2.screen.query("#internal-prompt-editor-preview")
