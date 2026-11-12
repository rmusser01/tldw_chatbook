"""Prompt editor actions remain painted and usable in a small terminal."""

import asyncio
from typing import ClassVar

import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_dialog_action_alignment import assert_painted
from tldw_chatbook.Widgets.Console.console_system_prompt_modal import (
    APPLY_BUTTON_ID,
    CANCEL_BUTTON_ID,
    CLEAR_BUTTON_ID,
    NAME_INPUT_ID,
    SAVE_LIBRARY_BUTTON_ID,
    SAVE_STATUS_ID,
    TEXT_AREA_ID,
    ConsoleSystemPromptModal,
)

FEEDBACK = "Could not save this prompt. Choose another name and try again. " * 3


class PromptEditorHarness(ConsolidatedCSSApp):
    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

    def __init__(self):
        super().__init__()
        self.results = []
        self.saved = []
        self.editor = ConsoleSystemPromptModal(
            system_prompt="Original instructions.", save_to_library=self.save_prompt
        )

    async def save_prompt(self, name, text):
        self.saved.append((name, text))
        return FEEDBACK

    async def on_mount(self):
        await self.push_screen(self.editor, self.results.append)


def assert_actions_painted(editor):
    for control in (APPLY_BUTTON_ID, CLEAR_BUTTON_ID, CANCEL_BUTTON_ID):
        button = editor.query_one(f"#{control}", Button)
        assert button in editor._compositor.visible_widgets, control
        assert_painted(editor, button)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (140, 42)])
@pytest.mark.parametrize(
    ("control", "expected"),
    [
        (APPLY_BUTTON_ID, "Edited instructions."),
        (CLEAR_BUTTON_ID, ""),
        (CANCEL_BUTTON_ID, None),
    ],
)
@private_profile_test
async def test_prompt_editor_actions_are_painted_and_clickable(
    request, size, control, expected
):
    host = PromptEditorHarness()
    async with host.run_test(size=size) as pilot:
        await pilot.pause()
        editor = host.editor
        editor.query_one(f"#{TEXT_AREA_ID}", TextArea).load_text(
            "  Edited instructions.\n"
        )
        await pilot.pause()
        assert_actions_painted(editor)
        button = editor.query_one(f"#{control}", Button)
        assert await pilot.click(
            f"#{control}", offset=(button.region.width // 2, button.region.height // 2)
        )
        async with asyncio.timeout(5):
            while not host.results:
                await pilot.pause(0.01)
        assert host.results == [expected]


@pytest.mark.asyncio
@private_profile_test
async def test_prompt_editor_resize_and_library_feedback_preserve_actions(request):
    host = PromptEditorHarness()
    async with host.run_test(size=(140, 42)) as pilot:
        editor = host.editor
        text = "Retain this long prompt.\n" * 100
        editor.query_one(f"#{TEXT_AREA_ID}", TextArea).load_text(text)
        editor.query_one(f"#{NAME_INPUT_ID}", Input).value = "UAT prompt"
        for size in ((80, 24), (140, 42), (80, 24)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert_actions_painted(editor)
            assert editor.query_one(f"#{TEXT_AREA_ID}", TextArea).text == text

        save = editor.query_one(f"#{SAVE_LIBRARY_BUTTON_ID}", Button)
        save.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert_actions_painted(editor)
        assert await pilot.click(
            f"#{SAVE_LIBRARY_BUTTON_ID}",
            offset=(save.region.width // 2, save.region.height // 2),
        )
        status = editor.query_one(f"#{SAVE_STATUS_ID}", Static)
        async with asyncio.timeout(5):
            while str(status.content) != FEEDBACK:
                await pilot.pause(0.01)
        await pilot.pause()
        assert host.saved == [("UAT prompt", text.strip())]
        region, clip = editor._compositor.visible_widgets[status]
        assert region.area and region.intersection(clip) == region
        painted = "\n".join(
            strip.crop(region.x, region.right).text
            for strip in editor._compositor.render_strips()[region.y : region.bottom]
        )
        assert " ".join(FEEDBACK.split()) in " ".join(painted.split())
        assert_actions_painted(editor)

        editor.query_one(f"#{CLEAR_BUTTON_ID}", Button).focus()
        await pilot.press("tab")
        assert host.focused is editor.query_one(f"#{CANCEL_BUTTON_ID}", Button)
        assert_actions_painted(editor)
        await pilot.press("tab", "enter")
        async with asyncio.timeout(5):
            while not host.results:
                await pilot.pause(0.01)
        assert host.results == [text.strip()]
