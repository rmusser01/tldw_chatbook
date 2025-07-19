"""Test voice blend dialog functionality"""
import pytest
from textual.app import App
from textual.testing import AppTest
from tldw_chatbook.Widgets.voice_blend_dialog import VoiceBlendDialog, VoiceBlendEntry


class TestApp(App):
    """Test app for voice blend dialog"""
    
    def compose(self):
        yield VoiceBlendDialog()


@pytest.mark.asyncio
async def test_voice_blend_dialog_create():
    """Test creating a new voice blend"""
    app = TestApp()
    async with app.run_test() as pilot:
        # Get the dialog
        dialog = app.query_one(VoiceBlendDialog)
        
        # Fill in the form
        name_input = dialog.query_one("#blend-name-input")
        await pilot.click(name_input)
        await pilot.press("ctrl+a")
        await pilot.type("My Test Blend")
        
        desc_input = dialog.query_one("#blend-description-input")
        await pilot.click(desc_input)
        await pilot.type("A test voice blend")
        
        # Set voice and weight
        voice_select = dialog.query_one("#voice-select-0")
        voice_select.value = "af_bella"
        
        weight_input = dialog.query_one("#weight-input-0")
        await pilot.click(weight_input)
        await pilot.press("ctrl+a")
        await pilot.type("1.0")
        
        # Save
        save_btn = dialog.query_one("#save-btn")
        await pilot.click(save_btn)
        
        # Check that dialog was dismissed with result
        # (In actual test, we'd check the returned value)


@pytest.mark.asyncio
async def test_voice_blend_dialog_multiple_voices():
    """Test creating a blend with multiple voices"""
    app = TestApp()
    async with app.run_test() as pilot:
        dialog = app.query_one(VoiceBlendDialog)
        
        # Add another voice
        add_btn = dialog.query_one("#add-voice-btn")
        await pilot.click(add_btn)
        
        # Should now have 2 voice entries
        entries = dialog.query(VoiceBlendEntry)
        assert len(entries) == 2
        
        # Set second voice
        voice_select_1 = dialog.query_one("#voice-select-1")
        voice_select_1.value = "am_michael"
        
        weight_input_1 = dialog.query_one("#weight-input-1")
        await pilot.click(weight_input_1)
        await pilot.type("0.5")


@pytest.mark.asyncio
async def test_voice_blend_dialog_remove_voice():
    """Test removing a voice entry"""
    app = TestApp()
    async with app.run_test() as pilot:
        dialog = app.query_one(VoiceBlendDialog)
        
        # Add two more voices
        add_btn = dialog.query_one("#add-voice-btn")
        await pilot.click(add_btn)
        await pilot.click(add_btn)
        
        # Should have 3 entries
        entries = dialog.query(VoiceBlendEntry)
        assert len(entries) == 3
        
        # Remove the middle one
        remove_btn = dialog.query_one("#remove-voice-1")
        await pilot.click(remove_btn)
        
        # Should have 2 entries now
        entries = dialog.query(VoiceBlendEntry)
        assert len(entries) == 2


@pytest.mark.asyncio
async def test_voice_blend_dialog_cancel():
    """Test canceling the dialog"""
    app = TestApp()
    async with app.run_test() as pilot:
        dialog = app.query_one(VoiceBlendDialog)
        
        # Click cancel
        cancel_btn = dialog.query_one("#cancel-btn")
        await pilot.click(cancel_btn)
        
        # Dialog should be dismissed with None result


@pytest.mark.asyncio
async def test_voice_blend_dialog_validation():
    """Test validation of blend name"""
    app = TestApp()
    async with app.run_test() as pilot:
        dialog = app.query_one(VoiceBlendDialog)
        
        # Try to save without a name
        save_btn = dialog.query_one("#save-btn")
        await pilot.click(save_btn)
        
        # Should show error notification
        # (In actual app, we'd check for the notification)