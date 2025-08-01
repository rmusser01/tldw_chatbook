# test_voice_input_widget.py
"""
Unit tests for VoiceInputWidget.
Tests UI component, event handling, and integration with dictation service.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from textual.app import App
from textual.widgets import Button, Static, Select, TextArea

from tldw_chatbook.Widgets.voice_input_widget import (
    VoiceInputWidget,
    VoiceInputMessage
)
from tldw_chatbook.Audio import DictationState
from tldw_chatbook.Event_Handlers.Audio_Events import (
    DictationStartedEvent,
    DictationStoppedEvent,
    PartialTranscriptEvent,
    FinalTranscriptEvent,
    VoiceCommandEvent,
    DictationErrorEvent,
    AudioLevelUpdateEvent
)


class TestVoiceInputWidget:
    """Unit tests for VoiceInputWidget."""
    
    @pytest.fixture
    def mock_dictation_service(self):
        """Mock LiveDictationService."""
        with patch('tldw_chatbook.Widgets.voice_input_widget.LiveDictationService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            # Setup default return values
            mock_instance.get_audio_devices.return_value = [
                {'id': 0, 'name': 'Default Mic', 'is_default': True},
                {'id': 1, 'name': 'USB Mic', 'is_default': False}
            ]
            mock_instance.start_dictation.return_value = True
            mock_instance.stop_dictation.return_value = Mock(
                transcript="Test transcript",
                duration=5.0,
                word_count=2
            )
            mock_instance.get_audio_level.return_value = 0.5
            mock_instance.get_full_transcript.return_value = "Full transcript"
            
            yield mock_instance
    
    @pytest.fixture
    async def widget_app(self, mock_dictation_service):
        """Create app with VoiceInputWidget for testing."""
        class TestApp(App):
            def compose(self):
                yield VoiceInputWidget(
                    show_device_selector=True,
                    show_transcript_preview=True
                )
        
        app = TestApp()
        async with app.run_test() as pilot:
            yield pilot
    
    @pytest.mark.asyncio
    async def test_widget_initialization(self, widget_app):
        """Test widget initializes correctly."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        
        assert widget is not None
        assert widget.state == DictationState.IDLE
        assert widget.audio_level == 0.0
        assert widget.current_transcript == ""
        assert widget.error_message == ""
    
    @pytest.mark.asyncio
    async def test_compose_ui_elements(self, widget_app):
        """Test UI elements are composed correctly."""
        # Check main elements exist
        assert widget_app.app.query_one("#record-button") is not None
        assert widget_app.app.query_one("#audio-level-bar") is not None
        assert widget_app.app.query_one("#device-selector") is not None
        assert widget_app.app.query_one("#transcript-preview") is not None
        assert widget_app.app.query_one("#voice-status") is not None
        assert widget_app.app.query_one("#voice-error") is not None
    
    @pytest.mark.asyncio
    async def test_device_selector_populated(self, widget_app, mock_dictation_service):
        """Test audio devices are loaded into selector."""
        # Wait for mount
        await asyncio.sleep(0.1)
        
        device_selector = widget_app.app.query_one("#device-selector", Select)
        
        # Check options were set
        assert len(device_selector._options) == 2
        assert device_selector._options[0][0] == "Default Mic (Default)"
        assert device_selector._options[1][0] == "USB Mic"
    
    @pytest.mark.asyncio
    async def test_record_button_starts_recording(self, widget_app, mock_dictation_service):
        """Test clicking record button starts recording."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        button = widget_app.app.query_one("#record-button", Button)
        
        # Click button
        await widget_app.click("#record-button")
        await asyncio.sleep(0.1)
        
        # Check recording started
        mock_dictation_service.start_dictation.assert_called_once()
        assert widget.state == DictationState.LISTENING
    
    @pytest.mark.asyncio
    async def test_record_button_stops_recording(self, widget_app, mock_dictation_service):
        """Test clicking record button while recording stops it."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        widget.state = DictationState.LISTENING
        widget.is_voice_recording = True
        
        # Click button to stop
        await widget_app.click("#record-button")
        await asyncio.sleep(0.1)
        
        # Check recording stopped
        mock_dictation_service.stop_dictation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_device_selection_changes(self, widget_app, mock_dictation_service):
        """Test changing device selection."""
        device_selector = widget_app.app.query_one("#device-selector", Select)
        
        # Change selection
        device_selector.value = 1
        await asyncio.sleep(0.1)
        
        # Check device was set
        mock_dictation_service.set_audio_device.assert_called_with(1)
    
    @pytest.mark.asyncio
    async def test_partial_transcript_update(self, widget_app):
        """Test partial transcript updates UI."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        preview = widget_app.app.query_one("#transcript-preview", Static)
        
        # Simulate partial transcript
        widget._on_partial_transcript("Hello world")
        widget._update_transcript_preview()
        
        assert widget.current_transcript == "Hello world"
        assert "Hello world" in preview.renderable
    
    @pytest.mark.asyncio
    async def test_final_transcript_message(self, widget_app):
        """Test final transcript sends message."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        messages = []
        
        # Capture messages
        widget.post_message = Mock(side_effect=lambda msg: messages.append(msg))
        
        # Simulate final transcript
        widget._on_final_transcript("Final text")
        
        # Check messages
        assert len(messages) == 1
        assert isinstance(messages[0], FinalTranscriptEvent)
        assert messages[0].text == "Final text"
    
    @pytest.mark.asyncio
    async def test_voice_command_detection(self, widget_app):
        """Test voice command sends appropriate message."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        messages = []
        
        widget.post_message = Mock(side_effect=lambda msg: messages.append(msg))
        
        # Simulate command
        widget._on_command("new_paragraph")
        
        # Check command message
        command_msgs = [m for m in messages if isinstance(m, VoiceCommandEvent)]
        assert len(command_msgs) == 1
        assert command_msgs[0].command == "new_paragraph"
    
    @pytest.mark.asyncio
    async def test_error_display(self, widget_app):
        """Test error messages are displayed."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        error_display = widget_app.app.query_one("#voice-error", Static)
        
        # Set error
        widget.error_message = "Test error"
        
        # Check display updated
        assert "Test error" in error_display.renderable
    
    @pytest.mark.asyncio
    async def test_audio_level_visualization(self, widget_app):
        """Test audio level updates visual indicator."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        level_bar = widget_app.app.query_one("#audio-level-bar", Static)
        
        # Update level
        widget._update_level_display(0.75)
        
        # Check width updated
        assert level_bar.styles.width == "75%"
    
    @pytest.mark.asyncio
    async def test_state_changes_update_ui(self, widget_app):
        """Test state changes update button and status."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        button = widget_app.app.query_one("#record-button", Button)
        status = widget_app.app.query_one("#voice-status", Static)
        
        # Change to listening state
        widget.state = DictationState.LISTENING
        widget._update_ui_state()
        
        assert "Stop" in button.label
        assert "recording" in button.classes
        assert "Listening..." in status.renderable
        
        # Change to paused state
        widget.state = DictationState.PAUSED
        widget._update_ui_state()
        
        assert "Resume" in button.label
        assert "paused" in button.classes
        assert "Paused" in status.renderable
    
    @pytest.mark.asyncio
    async def test_get_transcript(self, widget_app, mock_dictation_service):
        """Test getting full transcript."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        
        transcript = widget.get_transcript()
        
        assert transcript == "Full transcript"
        mock_dictation_service.get_full_transcript.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_transcript(self, widget_app):
        """Test clearing transcript."""
        widget = widget_app.app.query_one(VoiceInputWidget)
        preview = widget_app.app.query_one("#transcript-preview", Static)
        
        # Set some transcript
        widget.current_transcript = "Some text"
        widget._update_transcript_preview()
        
        # Clear it
        widget.clear_transcript()
        
        assert widget.current_transcript == ""
        assert widget.placeholder in preview.renderable
    
    @pytest.mark.asyncio
    async def test_widget_without_device_selector(self):
        """Test widget without device selector."""
        class TestApp(App):
            def compose(self):
                yield VoiceInputWidget(
                    show_device_selector=False,
                    show_transcript_preview=True
                )
        
        app = TestApp()
        async with app.run_test() as pilot:
            # Device selector should not exist
            with pytest.raises(Exception):
                pilot.app.query_one("#device-selector")
    
    @pytest.mark.asyncio
    async def test_widget_without_transcript_preview(self):
        """Test widget without transcript preview."""
        class TestApp(App):
            def compose(self):
                yield VoiceInputWidget(
                    show_device_selector=True,
                    show_transcript_preview=False
                )
        
        app = TestApp()
        async with app.run_test() as pilot:
            # Transcript preview should not exist
            with pytest.raises(Exception):
                pilot.app.query_one("#transcript-preview")
    
    @pytest.mark.asyncio
    async def test_dictation_service_initialization_failure(self):
        """Test handling dictation service initialization failure."""
        with patch('tldw_chatbook.Widgets.voice_input_widget.LiveDictationService') as mock_class:
            mock_class.side_effect = Exception("Audio not available")
            
            class TestApp(App):
                def compose(self):
                    yield VoiceInputWidget()
            
            app = TestApp()
            async with app.run_test() as pilot:
                widget = pilot.app.query_one(VoiceInputWidget)
                
                # Should show error
                assert "Voice input unavailable" in widget.error_message
                assert widget.dictation_service is None


class TestVoiceInputMessage:
    """Test VoiceInputMessage class."""
    
    def test_message_creation(self):
        """Test creating voice input messages."""
        msg = VoiceInputMessage("Hello world", is_final=True)
        
        assert msg.text == "Hello world"
        assert msg.is_final is True
    
    def test_partial_message(self):
        """Test partial transcript message."""
        msg = VoiceInputMessage("Partial", is_final=False)
        
        assert msg.text == "Partial"
        assert msg.is_final is False


class TestVoiceInputIntegration:
    """Integration tests for voice input widget."""
    
    @pytest.mark.asyncio
    async def test_full_recording_flow(self):
        """Test complete recording flow from start to finish."""
        with patch('tldw_chatbook.Widgets.voice_input_widget.LiveDictationService') as mock_dict_class:
            # Setup mock
            mock_service = Mock()
            mock_dict_class.return_value = mock_service
            
            # Track callbacks
            partial_callback = None
            final_callback = None
            state_callback = None
            
            def capture_start_dictation(**kwargs):
                nonlocal partial_callback, final_callback, state_callback
                partial_callback = kwargs.get('on_partial_transcript')
                final_callback = kwargs.get('on_final_transcript')
                state_callback = kwargs.get('on_state_change')
                return True
            
            mock_service.start_dictation.side_effect = capture_start_dictation
            mock_service.get_audio_devices.return_value = []
            
            class TestApp(App):
                messages_received = []
                
                def compose(self):
                    yield VoiceInputWidget()
                
                def on_voice_input_message(self, event):
                    self.messages_received.append(event)
            
            app = TestApp()
            async with app.run_test() as pilot:
                widget = pilot.app.query_one(VoiceInputWidget)
                
                # Start recording
                await pilot.click("#record-button")
                await asyncio.sleep(0.1)
                
                # Simulate transcription callbacks
                if partial_callback:
                    partial_callback("Hello")
                    partial_callback("Hello world")
                
                if final_callback:
                    final_callback("Hello world")
                
                if state_callback:
                    state_callback(DictationState.LISTENING)
                
                # Check messages
                assert len(app.messages_received) >= 2
                
                # Stop recording
                mock_service.stop_dictation.return_value = Mock(
                    transcript="Hello world",
                    duration=2.0,
                    word_count=2
                )
                
                await pilot.click("#record-button")
                await asyncio.sleep(0.1)
                
                # Should have final message
                final_messages = [m for m in app.messages_received if m.is_final]
                assert len(final_messages) >= 1
                assert final_messages[-1].text == "Hello world"
    
    @pytest.mark.asyncio
    async def test_audio_level_monitoring(self):
        """Test audio level monitoring during recording."""
        with patch('tldw_chatbook.Widgets.voice_input_widget.LiveDictationService') as mock_dict_class:
            mock_service = Mock()
            mock_dict_class.return_value = mock_service
            mock_service.get_audio_devices.return_value = []
            mock_service.start_dictation.return_value = True
            
            # Simulate changing audio levels
            levels = [0.1, 0.3, 0.5, 0.7, 0.2]
            level_index = 0
            
            def get_level():
                nonlocal level_index
                if level_index < len(levels):
                    level = levels[level_index]
                    level_index += 1
                    return level
                return 0.0
            
            mock_service.get_audio_level.side_effect = get_level
            
            class TestApp(App):
                def compose(self):
                    yield VoiceInputWidget()
            
            app = TestApp()
            async with app.run_test() as pilot:
                widget = pilot.app.query_one(VoiceInputWidget)
                
                # Start recording
                await pilot.click("#record-button")
                
                # Wait for level monitoring
                await asyncio.sleep(0.6)  # Should get several updates
                
                # Check that level was updated
                assert widget.audio_level > 0