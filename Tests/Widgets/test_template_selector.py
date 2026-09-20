# TASK-32831: ``TemplateSelectorDialog._load_templates`` (and
# ``QuickTemplateSelector._load_templates``) imported ``get_eval_templates``
# from ``tldw_chatbook.Evals``, whose ``__init__`` no longer exports it -- the
# ImportError was swallowed and the template list was always empty (documented
# in .superpowers/wave2-report.md). The symbol lives in
# ``tldw_chatbook.Evals.eval_templates``. These tests pin that the selector
# actually lists the built-in evaluation templates.

from textual.app import App
from textual.widgets import Button, Label, ListItem

from tldw_chatbook.Widgets.template_selector import (
    QuickTemplateSelector,
    TemplateSelectorDialog,
)


class _ScreenHost(App):
    def __init__(self, screen):
        super().__init__()
        self.screen_under_test = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


class _WidgetHost(App):
    def __init__(self, widget):
        super().__init__()
        self.widget_under_test = widget

    def compose(self):
        yield self.widget_under_test


async def test_template_selector_dialog_lists_builtin_eval_templates():
    dialog = TemplateSelectorDialog()
    app = _ScreenHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        # The built-in evaluation templates (reasoning, language, coding,
        # safety, creative, multimodal categories) are loaded as dict
        # records the list/preview widgets consume.
        assert len(dialog.templates) >= 1
        first = dialog.templates[0]
        assert isinstance(first, dict)
        assert first.get("name")
        assert first.get("display_name")
        assert first.get("category")

        # ...and they are actually rendered: the categorized ListViews carry
        # at least one ListItem and a known built-in template is displayed.
        items = dialog.query(ListItem)
        assert len(items) >= 1
        labels = [str(item.query_one(Label).renderable) for item in items]
        assert "GSM8K" in labels

        # Selection wiring still works off the rendered records.
        dialog._on_template_selected(first)
        assert dialog.selected_template is first
        assert dialog.query_one("#select-button", Button).disabled is False


async def test_quick_template_selector_lists_builtin_eval_templates():
    widget = QuickTemplateSelector()
    app = _WidgetHost(widget)

    async with app.run_test() as pilot:
        await pilot.pause()

        assert len(widget.templates) >= 1
        quick_buttons = widget.query(".template-quick-button")
        assert len(quick_buttons) >= 1
