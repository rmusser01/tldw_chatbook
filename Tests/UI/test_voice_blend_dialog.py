"""Test voice blend dialog functionality"""
import pytest
from textual.app import App
from tldw_chatbook.Widgets.voice_blend_dialog import VoiceBlendDialog, VoiceBlendEntry


class VoiceBlendDialogApp(App):
    """Test app for voice blend dialog"""
    
    def compose(self):
        yield VoiceBlendDialog()


async def _type_into_widget(pilot, widget, text: str, *, clear: bool = False) -> None:
    """Focus a text input widget and enter text using the current Pilot API."""
    widget.focus()
    await pilot.pause()

    if clear and hasattr(widget, "clear"):
        widget.clear()
        await pilot.pause()

    await pilot.press(*text)
    await pilot.pause()


@pytest.mark.asyncio
async def test_voice_blend_dialog_create():
    """Test creating a new voice blend"""
    app = VoiceBlendDialogApp()
    async with app.run_test() as pilot:
        # Get the dialog
        dialog = app.query_one(VoiceBlendDialog)
        captured = {}
        
        # Fill in the form
        name_input = dialog.query_one("#blend-name-input")
        name_input.value = "My Test Blend"
        
        desc_input = dialog.query_one("#blend-description-input")
        desc_input.value = "A test voice blend"
        
        # Set voice and weight
        voice_select = dialog.query_one("#voice-select-0")
        voice_select.value = "af_bella"
        
        weight_input = dialog.query_one("#weight-input-0")
        weight_input.value = "1.0"
        
        dialog.dismiss = lambda result: captured.setdefault("result", result)
        dialog.save_blend()

        assert captured["result"]["name"] == "My Test Blend"
        assert captured["result"]["description"] == "A test voice blend"
        assert captured["result"]["voices"] == [("af_bella", 1.0)]


@pytest.mark.asyncio
async def test_voice_blend_dialog_multiple_voices():
    """Test creating a blend with multiple voices"""
    app = VoiceBlendDialogApp()
    async with app.run_test() as pilot:
        dialog = app.query_one(VoiceBlendDialog)
        
        # Add another voice
        await dialog.add_voice_entry()
        await pilot.pause()
        
        # Should now have 2 voice entries
        entries = dialog.query(VoiceBlendEntry)
        assert len(entries) == 2
        
        # Set second voice
        voice_select_1 = dialog.query_one("#voice-select-1")
        voice_select_1.value = "am_michael"
        
        weight_input_1 = dialog.query_one("#weight-input-1")
        await _type_into_widget(pilot, weight_input_1, "0.5", clear=True)


@pytest.mark.asyncio
async def test_voice_blend_dialog_remove_voice():
    """Test removing a voice entry"""
    app = VoiceBlendDialogApp()
    async with app.run_test() as pilot:
        dialog = app.query_one(VoiceBlendDialog)
        
        # Add two more voices
        await dialog.add_voice_entry()
        await dialog.add_voice_entry()
        await pilot.pause()
        
        # Should have 3 entries
        entries = dialog.query(VoiceBlendEntry)
        assert len(entries) == 3
        
        # Remove the middle one
        await dialog.on_voice_blend_entry_removed(VoiceBlendEntry.Removed(entries[1]))
        await pilot.pause()
        
        # Should have 2 entries now
        entries = dialog.query(VoiceBlendEntry)
        assert len(entries) == 2


@pytest.mark.asyncio
async def test_voice_blend_dialog_cancel():
    """Test canceling the dialog"""
    app = VoiceBlendDialogApp()
    async with app.run_test() as pilot:
        dialog = app.query_one(VoiceBlendDialog)
        
        # Click cancel
        cancel_btn = dialog.query_one("#cancel-btn")
        await pilot.click(cancel_btn)
        
        # Dialog should be dismissed with None result


@pytest.mark.asyncio
async def test_voice_blend_dialog_validation():
    """Test validation of blend name"""
    app = VoiceBlendDialogApp()
    async with app.run_test() as pilot:
        dialog = app.query_one(VoiceBlendDialog)
        
        # Try to save without a name
        save_btn = dialog.query_one("#save-btn")
        await pilot.click(save_btn)
        
        # Should show error notification
        # (In actual app, we'd check for the notification)
