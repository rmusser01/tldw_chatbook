from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Third_Party.textual_fspicker.parts.directory_navigation import (
    DirectoryEntry,
)
from tldw_chatbook.UI.CCP_Modules.ccp_character_handler import CCPCharacterHandler
from tldw_chatbook.UI.Chatbooks_Window_Improved import EmptyStateWidget
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.ChatbookImportWizard import FileSelectionStep
from tldw_chatbook.Widgets import enhanced_file_picker as efp
from tldw_chatbook.Widgets.file_picker_dialog import QuickPickerWidget
from tldw_chatbook.Widgets.NewIngest.SmartFileDropZone import SmartFileDropZone


class _WidgetHost(App):
    def __init__(self, widget):
        super().__init__()
        self.widget_under_test = widget

    def compose(self) -> ComposeResult:
        yield self.widget_under_test


def _assert_button_tooltips(root, expected_tooltips: dict[str, str]) -> None:
    for button_id, expected_tooltip in expected_tooltips.items():
        button = root.query_one(f"#{button_id}", Button)
        assert str(button.tooltip) == expected_tooltip


def _patch_clean_picker_config(monkeypatch) -> None:
    monkeypatch.setattr(
        efp,
        "get_cli_setting",
        lambda _section, _key, default=None: default,
    )
    monkeypatch.setattr(efp, "save_setting_to_cli_config", lambda *_args: None)


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


@pytest.mark.asyncio
async def test_character_import_picker_actions_explain_open_and_cancel(
    tmp_path, monkeypatch
):
    _patch_clean_picker_config(monkeypatch)
    picker = efp.EnhancedFileOpen(
        location=tmp_path,
        title="Import Character Card",
        context="character_import",
    )
    app = App()

    async with app.run_test() as pilot:
        app.push_screen(picker)
        await pilot.pause()

        _assert_button_tooltips(
            picker,
            {
                "select": "Import the selected character card.",
                "cancel": "Close without importing a character card.",
            },
        )
        assert str(picker.query_one("#select", Button).label) == "Import"


@pytest.mark.asyncio
async def test_character_import_picker_preserves_explicit_primary_action_label(
    tmp_path, monkeypatch
):
    _patch_clean_picker_config(monkeypatch)
    picker = efp.EnhancedFileOpen(
        location=tmp_path,
        context="character_import",
        select_button="Choose card",
    )
    app = App()

    async with app.run_test() as pilot:
        app.push_screen(picker)
        await pilot.pause()

        assert str(picker.query_one("#select", Button).label) == "Choose card"


@pytest.mark.asyncio
async def test_character_picker_selected_row_keeps_focus_and_selection_states(
    tmp_path, monkeypatch
):
    _patch_clean_picker_config(monkeypatch)
    card = tmp_path / "ann.json"
    card.write_text("{}", encoding="utf-8")
    picker = efp.EnhancedFileOpen(
        location=tmp_path,
        context="character_import",
        multi_select=True,
    )
    app = App()

    async with app.run_test(size=(60, 24)) as pilot:
        app.push_screen(picker)
        await pilot.pause()
        nav_type = efp.EnhancedDirectoryNavigation
        navigation = picker.query_one(nav_type)
        index = next(
            index
            for index, option in enumerate(navigation.options)
            if isinstance(option, DirectoryEntry) and option.location == card
        )
        navigation.focus()
        navigation.highlighted = index
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()

        assert navigation.row_state_classes(index) == {"-focused", "-selected"}
        frame = app.export_screenshot()
        assert "✓" in frame
        assert "ann.json" in frame


@pytest.mark.asyncio
async def test_legacy_character_import_action_uses_import_label(monkeypatch):
    captured: dict[str, object] = {}

    class _RecordingPicker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(efp, "EnhancedFileOpen", _RecordingPicker)
    window = SimpleNamespace(
        app_instance=SimpleNamespace(),
        app=SimpleNamespace(push_screen=AsyncMock(return_value=None)),
    )

    await CCPCharacterHandler(window).handle_import()

    assert captured["context"] == "character_import"
    assert captured["select_button"] == "Import"
