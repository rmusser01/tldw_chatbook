"""Mounted multiple-skill candidate choice behavior."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, OptionList

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Library_Modules.skill_import_choice_modal import (
    SkillImportChoiceModal,
)


class _ChoiceHost(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield Button("Open", id="open")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((120, 36), (72, 22)))
async def test_choice_modal_lists_bounded_candidates_and_imports_highlighted(size):
    app = _ChoiceHost()
    results: list[str | None] = []
    async with app.run_test(size=size) as pilot:
        app.push_screen(
            SkillImportChoiceModal(("skills/alpha", "skills/zeta")),
            results.append,
        )
        await pilot.pause()
        choices = app.screen.query_one("#skill-import-choice-list", OptionList)
        assert choices.option_count == 2
        assert "skills/alpha" in app.export_screenshot()
        choices.highlighted = 1
        app.screen.query_one("#skill-import-choice-import", Button).press()
        await pilot.pause()
    assert results == ["skills/zeta"]


@pytest.mark.asyncio
async def test_choice_modal_cancel_returns_none():
    app = _ChoiceHost()
    results: list[str | None] = []
    async with app.run_test(size=(72, 22)) as pilot:
        app.push_screen(SkillImportChoiceModal(("skills/alpha",)), results.append)
        await pilot.pause()
        app.screen.query_one("#skill-import-choice-cancel", Button).press()
        await pilot.pause()
    assert results == [None]
