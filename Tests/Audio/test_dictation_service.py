# test_dictation_service.py
"""
Unit tests for LiveDictationService.
Tests dictation, transcription integration, and state management.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import numpy as np

from tldw_chatbook.Audio.dictation_service import (
    LiveDictationService,
    DictationResult,
    DictationState
)
from tldw_chatbook.Audio.recording_service import AudioRecordingError


class TestLiveDictationService:
    """Unit tests for LiveDictationService."""
    
    @pytest.fixture
    def mock_recording_service(self):
        """Mock AudioRecordingService."""
        with patch('tldw_chatbook.Audio.dictation_service.AudioRecordingService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            mock_instance.start_recording.return_value = True
            mock_instance.stop_recording.return_value = b'test_audio'
            yield mock_instance
    
    @pytest.fixture
    def mock_transcription_service(self):
        """Mock TranscriptionService."""
        with patch('tldw_chatbook.Audio.dictation_service.TranscriptionService') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def dictation_service(self, mock_recording_service, mock_transcription_service):
        """Create a LiveDictationService instance with mocked dependencies."""
        service = LiveDictationService(
            transcription_provider='test',
            language='en'
        )
        return service
    
    def test_initialization(self, dictation_service):
        """Test service initialization."""
        assert dictation_service.state == DictationState.IDLE
        assert dictation_service.language == 'en'
        assert dictation_service.enable_punctuation is True
        assert dictation_service.enable_commands is True
        assert dictation_service.transcript_segments == []
        assert dictation_service.current_transcript == ""
    
    def test_initialization_with_recording_error(self, mock_transcription_service):
        """Test initialization fails gracefully with recording service error."""
        with patch('tldw_chatbook.Audio.dictation_service.AudioRecordingService') as mock_class:
            mock_class.side_effect = AudioRecordingError("No audio backend")
            
            with pytest.raises(AudioRecordingError):
                LiveDictationService()
    
    def test_start_dictation_success(self, dictation_service, mock_recording_service):
        """Test successful start of dictation."""
        # Setup callbacks
        on_partial = Mock()
        on_final = Mock()
        on_state = Mock()
        
        # Start dictation
        result = dictation_service.start_dictation(
            on_partial_transcript=on_partial,
            on_final_transcript=on_final,
            on_state_change=on_state
        )
        
        assert result is True
        assert dictation_service.state == DictationState.LISTENING
        mock_recording_service.start_recording.assert_called_once()
        on_state.assert_called_with(DictationState.LISTENING)
    
    def test_start_dictation_already_running(self, dictation_service):
        """Test starting dictation when already running."""
        dictation_service.state = DictationState.LISTENING
        
        result = dictation_service.start_dictation()
        assert result is False
    
    def test_start_dictation_recording_failure(self, dictation_service, mock_recording_service):
        """Test handling recording service failure."""
        mock_recording_service.start_recording.return_value = False
        
        result = dictation_service.start_dictation()
        assert result is False
        assert dictation_service.state == DictationState.IDLE
    
    def test_stop_dictation(self, dictation_service, mock_recording_service):
        """Test stopping dictation."""
        # Start first
        dictation_service.start_dictation()
        
        # Add some segments
        dictation_service.transcript_segments = [
            {'text': 'Hello', 'timestamp': 0},
            {'text': 'world', 'timestamp': 1}
        ]
        dictation_service.start_time = time.time() - 5  # 5 seconds ago
        
        # Stop dictation
        result = dictation_service.stop_dictation()
        
        assert isinstance(result, DictationResult)
        assert result.transcript == 'Hello world'
        assert len(result.segments) == 2
        assert result.duration > 0
        assert dictation_service.state == DictationState.IDLE
        mock_recording_service.stop_recording.assert_called_once()
    
    def test_stop_dictation_not_running(self, dictation_service):
        """Test stopping when not dictating."""
        result = dictation_service.stop_dictation()
        
        assert isinstance(result, DictationResult)
        assert result.transcript == ""
        assert result.segments == []
        assert result.duration == 0
    
    def test_pause_resume_dictation(self, dictation_service):
        """Test pause and resume functionality."""
        # Start dictation
        dictation_service.start_dictation()
        assert dictation_service.state == DictationState.LISTENING
        
        # Pause
        result = dictation_service.pause_dictation()
        assert result is True
        assert dictation_service.state == DictationState.PAUSED
        
        # Resume
        result = dictation_service.resume_dictation()
        assert result is True
        assert dictation_service.state == DictationState.LISTENING
    
    def test_pause_when_not_listening(self, dictation_service):
        """Test pausing when not in listening state."""
        result = dictation_service.pause_dictation()
        assert result is False
    
    def test_resume_when_not_paused(self, dictation_service):
        """Test resuming when not paused."""
        result = dictation_service.resume_dictation()
        assert result is False
    
    def test_audio_callback(self, dictation_service):
        """Test audio callback processing."""
        dictation_service.start_dictation()
        
        # Simulate audio callback
        audio_chunk = b'\x00\x01' * 512
        dictation_service._audio_callback(audio_chunk)
        
        # Check buffer and queue
        assert len(dictation_service.audio_buffer) == 1
        assert not dictation_service.processing_queue.empty()
        assert dictation_service.last_speech_time > 0
    
    def test_handle_partial_transcript(self, dictation_service):
        """Test handling partial transcript updates."""
        callback = Mock()
        dictation_service.on_partial_transcript = callback
        
        dictation_service._handle_partial_transcript("Hello world")
        
        assert dictation_service.current_transcript == "Hello world"
        callback.assert_called_once_with("Hello world")
    
    def test_handle_partial_transcript_with_punctuation(self, dictation_service):
        """Test automatic punctuation."""
        dictation_service.enable_punctuation = True
        
        dictation_service._handle_partial_transcript("hello world")
        assert dictation_service.current_transcript == "Hello world."
    
    def test_handle_final_transcript(self, dictation_service):
        """Test handling final transcript segments."""
        callback = Mock()
        dictation_service.on_final_transcript = callback
        dictation_service.start_time = time.time()
        
        dictation_service._handle_final_transcript("Hello world")
        
        assert len(dictation_service.transcript_segments) == 1
        assert dictation_service.transcript_segments[0]['text'] == "Hello world"
        assert dictation_service.current_transcript == ""
        callback.assert_called_once_with("Hello world")
    
    def test_command_detection(self, dictation_service):
        """Test voice command detection."""
        callback = Mock()
        dictation_service.on_command = callback
        dictation_service.enable_commands = True
        
        # Test various commands
        commands = [
            ("Say new paragraph please", "new_paragraph"),
            ("Add a new line here", "new_line"),
            ("Insert comma", "insert_comma"),
            ("Stop dictation now", "stop_dictation")
        ]
        
        for text, expected_command in commands:
            command = dictation_service._detect_command(text)
            assert command == expected_command
    
    def test_command_execution(self, dictation_service):
        """Test command processing in transcript."""
        dictation_service.on_command = Mock()
        dictation_service.enable_commands = True
        dictation_service.start_time = time.time()
        
        dictation_service._handle_partial_transcript("new paragraph")
        
        dictation_service.on_command.assert_called_once_with("new_paragraph")
    
    def test_streaming_transcriber_initialization(self, dictation_service, mock_transcription_service):
        """Test streaming transcriber setup."""
        mock_transcriber = Mock()
        mock_transcription_service.create_streaming_transcriber.return_value = mock_transcriber
        
        dictation_service._initialize_streaming_transcriber()
        
        assert dictation_service.streaming_transcriber == mock_transcriber
        mock_transcription_service.create_streaming_transcriber.assert_called_once_with(
            provider='test',
            model=None,
            language='en'
        )
    
    def test_process_audio_buffer_with_streaming(self, dictation_service):
        """Test audio processing with streaming transcriber."""
        # Setup streaming transcriber
        mock_transcriber = Mock()
        mock_transcriber.process_audio.return_value = {
            'partial': 'Hello',
            'final': None
        }
        dictation_service.streaming_transcriber = mock_transcriber
        dictation_service.state = DictationState.LISTENING
        
        # Process audio
        audio_data = b'\x00\x01' * 1000
        dictation_service._process_audio_buffer(audio_data)
        
        mock_transcriber.process_audio.assert_called_once()
        assert dictation_service.current_transcript == "Hello."
    
    def test_process_audio_buffer_without_streaming(self, dictation_service, mock_transcription_service):
        """Test audio processing without streaming (chunked mode)."""
        dictation_service.streaming_transcriber = None
        dictation_service.state = DictationState.LISTENING
        
        # Setup transcription result
        mock_transcription_service.transcribe_buffer.return_value = {
            'text': 'Test transcription'
        }
        
        # Process audio
        audio_data = b'\x00\x01' * 1000
        dictation_service._process_audio_buffer(audio_data)
        
        mock_transcription_service.transcribe_buffer.assert_called_once()
        assert dictation_service.current_transcript == "Test transcription."
    
    def test_get_state(self, dictation_service):
        """Test state getter."""
        assert dictation_service.get_state() == DictationState.IDLE
        
        dictation_service.state = DictationState.LISTENING
        assert dictation_service.get_state() == DictationState.LISTENING
    
    def test_get_current_transcript(self, dictation_service):
        """Test getting current partial transcript."""
        dictation_service.current_transcript = "Hello"
        assert dictation_service.get_current_transcript() == "Hello"
    
    def test_get_full_transcript(self, dictation_service):
        """Test getting full transcript including partial."""
        dictation_service.transcript_segments = [
            {'text': 'Hello', 'timestamp': 0},
            {'text': 'world', 'timestamp': 1}
        ]
        dictation_service.current_transcript = "how are you"
        
        full = dictation_service.get_full_transcript()
        assert full == "Hello world how are you"
    
    def test_get_audio_devices(self, dictation_service, mock_recording_service):
        """Test getting audio devices."""
        mock_devices = [{'id': 0, 'name': 'Microphone'}]
        mock_recording_service.get_audio_devices.return_value = mock_devices
        
        devices = dictation_service.get_audio_devices()
        assert devices == mock_devices
        mock_recording_service.get_audio_devices.assert_called_once()
    
    def test_set_audio_device(self, dictation_service, mock_recording_service):
        """Test setting audio device."""
        mock_recording_service.set_device.return_value = True
        
        result = dictation_service.set_audio_device(1)
        assert result is True
        mock_recording_service.set_device.assert_called_once_with(1)
    
    def test_get_audio_level(self, dictation_service, mock_recording_service):
        """Test getting audio level."""
        mock_recording_service.get_audio_level.return_value = 0.5
        
        level = dictation_service.get_audio_level()
        assert level == 0.5
        mock_recording_service.get_audio_level.assert_called_once()
    
    def test_is_available(self, dictation_service, mock_recording_service):
        """Test checking availability."""
        mock_recording_service.is_available.return_value = True
        
        available = dictation_service.is_available()
        assert available is True
        mock_recording_service.is_available.assert_called_once()
    
    def test_error_callback(self, dictation_service):
        """Test error callback handling."""
        error_callback = Mock()
        dictation_service.on_error = error_callback
        
        test_error = ValueError("Test error")
        dictation_service._notify_error(test_error)
        
        assert dictation_service.state == DictationState.ERROR
        error_callback.assert_called_once_with(test_error)
    
    def test_processing_loop_with_silence_timeout(self, dictation_service):
        """Test processing loop finalizes segment after silence."""
        dictation_service.current_transcript = "Test"
        dictation_service.last_speech_time = time.time() - 3  # 3 seconds ago
        
        # Mock finalize method
        with patch.object(dictation_service, '_finalize_current_segment') as mock_finalize:
            # Run one iteration of processing loop
            dictation_service.stop_processing.set()  # Stop after one iteration
            dictation_service._processing_loop()
            
            mock_finalize.assert_called_once()


class TestDictationResult:
    """Test DictationResult dataclass."""
    
    def test_dictation_result_creation(self):
        """Test creating a DictationResult."""
        result = DictationResult(
            transcript="Hello world",
            segments=[{'text': 'Hello world', 'timestamp': 0}],
            duration=5.0,
            audio_data=b'audio'
        )
        
        assert result.transcript == "Hello world"
        assert len(result.segments) == 1
        assert result.duration == 5.0
        assert result.audio_data == b'audio'
        assert isinstance(result.timestamp, datetime)
    
    def test_dictation_result_defaults(self):
        """Test DictationResult with defaults."""
        result = DictationResult(
            transcript="Test",
            segments=[],
            duration=1.0
        )
        
        assert result.audio_data is None
        assert result.timestamp is not None


class TestDictationIntegration:
    """Integration tests for dictation service."""
    
    def test_full_dictation_flow(self):
        """Test complete dictation flow from start to stop."""
        with patch('tldw_chatbook.Audio.dictation_service.AudioRecordingService') as mock_rec_class:
            with patch('tldw_chatbook.Audio.dictation_service.TranscriptionService') as mock_trans_class:
                # Setup mocks
                mock_recording = Mock()
                mock_recording.start_recording.return_value = True
                mock_recording.stop_recording.return_value = b'test_audio'
                mock_recording.is_available.return_value = True
                mock_rec_class.return_value = mock_recording
                
                mock_transcription = Mock()
                mock_transcription.create_streaming_transcriber.return_value = None
                mock_trans_class.return_value = mock_transcription
                
                # Create service
                service = LiveDictationService()
                
                # Track callbacks
                transcripts = []
                states = []
                
                # Start dictation
                result = service.start_dictation(
                    on_partial_transcript=transcripts.append,
                    on_state_change=states.append
                )
                assert result is True
                
                # Simulate some transcription
                service._handle_partial_transcript("Hello")
                service._handle_final_transcript("Hello")
                service._handle_partial_transcript("world")
                
                # Stop dictation
                result = service.stop_dictation()
                
                assert result.transcript == "Hello"
                assert len(result.segments) == 1
                assert len(transcripts) == 2
                assert states == [DictationState.LISTENING, DictationState.STOPPING, DictationState.IDLE]
    
    def test_error_recovery(self):
        """Test error recovery during dictation."""
        with patch('tldw_chatbook.Audio.dictation_service.AudioRecordingService') as mock_rec_class:
            with patch('tldw_chatbook.Audio.dictation_service.TranscriptionService') as mock_trans_class:
                mock_recording = Mock()
                mock_rec_class.return_value = mock_recording
                mock_trans_class.return_value = Mock()
                
                service = LiveDictationService()
                
                errors = []
                service.start_dictation(on_error=errors.append)
                
                # Simulate processing error
                service._process_audio_buffer(b'invalid')
                
                # Service should continue running
                assert service.state == DictationState.LISTENING
                
                # Stop normally
                service.stop_dictation()
                assert service.state == DictationState.IDLE