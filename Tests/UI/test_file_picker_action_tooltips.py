from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.Chatbooks_Window_Improved import EmptyStateWidget
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.ChatbookImportWizard import FileSelectionStep
from tldw_chatbook.UI.Wizards.EmbeddingSteps import SmartContentSelector
from tldw_chatbook.Widgets.Evals.eval_additional_dialogs import FileUploadDialog
from tldw_chatbook.Widgets.NewIngest.SmartFileDropZone import SmartFileDropZone
from tldw_chatbook.Widgets.file_picker_dialog import QuickPickerWidget


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


@pytest.mark.asyncio
async def test_eval_file_upload_actions_explain_browse_and_upload():
    app = _ScreenHost(FileUploadDialog())

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.screen_under_test,
            {
                "browse-button": "Choose an evaluation task or dataset file from disk.",
                "upload-button": "Upload the selected evaluation file.",
            },
        )


@pytest.mark.asyncio
async def test_quick_file_picker_browse_action_explains_file_type_scope():
    app = _WidgetHost(QuickPickerWidget(file_types="evaluation files"))

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "browse-button": "Choose evaluation files from disk.",
            },
        )


@pytest.mark.asyncio
async def test_chatbooks_empty_import_and_template_actions_have_tooltips():
    app = _WidgetHost(EmptyStateWidget())

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "empty-import-btn": "Import a local Chatbook pack from disk.",
                "empty-templates-btn": "Browse Chatbook templates for a faster start.",
            },
        )


@pytest.mark.asyncio
async def test_chatbook_import_wizard_browse_action_explains_zip_scope():
    wizard = SimpleNamespace(app_instance=Mock(), refresh_current_step=Mock())
    step = FileSelectionStep(
        wizard=wizard,
        config=WizardStepConfig(
            id="file-selection",
            title="Select File",
            description="Choose chatbook to import",
            step_number=1,
        ),
    )
    app = _WidgetHost(step)

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "browse-file": "Choose a .zip Chatbook pack from disk.",
            },
        )


@pytest.mark.asyncio
async def test_embedding_file_browse_action_explains_index_scope():
    app = _WidgetHost(SmartContentSelector("files"))

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "browse-files": "Choose local files to include in this embedding collection.",
            },
        )


@pytest.mark.asyncio
async def test_ingest_drop_zone_browse_action_explains_file_selection():
    app = _WidgetHost(SmartFileDropZone())

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "browse-overlay": "Choose files from disk for ingestion.",
            },
        )

