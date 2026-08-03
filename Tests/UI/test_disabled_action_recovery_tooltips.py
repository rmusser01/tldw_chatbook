import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.template_selector import (
    TemplatePreviewWidget,
    TemplateSelectorDialog,
)


class _WidgetHost(App):
    def __init__(self, widget):
        super().__init__()
        self.widget_under_test = widget

    def compose(self) -> ComposeResult:
        yield self.widget_under_test


class _ScreenHost(App):
    def __init__(self, screen):
        super().__init__()
        self.screen_under_test = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


def _assert_button_tooltips(root, expected_tooltips: dict[str, str]) -> None:
    for button_id, expected_tooltip in expected_tooltips.items():
        button = root.query_one(f"#{button_id}", Button)
        assert str(button.tooltip) == expected_tooltip


def _static_text(static: Static) -> str:
    return str(static.renderable)


@pytest.mark.asyncio
async def test_template_preview_actions_explain_selection_requirement():
    app = _WidgetHost(TemplatePreviewWidget())

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "create-task-btn": "Select an evaluation template before creating a task.",
                "export-template-btn": "Select an evaluation template before exporting it.",
            },
        )

        app.widget_under_test.update_preview(
            {
                "name": "QA Template",
                "description": "Answer quality checks.",
                "category": "quality",
                "difficulty": "medium",
                "task_type": "qa",
            }
        )

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "create-task-btn": "Create an evaluation task from this template.",
                "export-template-btn": "Export this evaluation template.",
            },
        )


@pytest.mark.asyncio
async def test_template_selector_select_action_explains_selection_requirement(
    monkeypatch,
):
    monkeypatch.setattr(TemplateSelectorDialog, "_load_templates", lambda self: None)
    app = _ScreenHost(TemplateSelectorDialog())

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.screen_under_test,
            {"select-button": "Select an evaluation template before continuing."},
        )

        app.screen_under_test._on_template_selected(
            {
                "name": "QA Template",
                "description": "Answer quality checks.",
                "category": "quality",
                "difficulty": "medium",
                "task_type": "qa",
            }
        )

        _assert_button_tooltips(
            app.screen_under_test,
            {"select-button": "Use the selected evaluation template."},
        )
