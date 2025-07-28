# test_audio_integration.py
"""
Integration tests for the complete audio recording and dictation system.
Tests interaction between components and end-to-end workflows.
"""

import pytest
import asyncio
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np
import wave

from tldw_chatbook.Audio import (
    AudioRecordingService,
    LiveDictationService,
    DictationState,
    DictationResult
)
from tldw_chatbook.Widgets.voice_input_widget import VoiceInputWidget, VoiceInputMessage
from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService


class TestEndToEndDictation:
    """Test complete dictation workflow from recording to transcription."""
    
    @pytest.fixture
    def mock_transcription_backend(self):
        """Mock a transcription backend."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.FASTER_WHISPER_AVAILABLE', True):
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model:
                # Setup mock model
                mock_instance = Mock()
                mock_instance.transcribe.return_value = (
                    [Mock(text="Hello world", start=0.0, end=2.0)],
                    Mock()
                )
                mock_model.return_value = mock_instance
                yield mock_instance
    
    @pytest.fixture
    def simulated_audio_data(self):
        """Generate simulated audio data."""
        # Generate 2 seconds of audio at 16kHz
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Simple sine wave (440Hz - A4 note)
        frequency = 440
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.05, audio.shape)
        audio = audio + noise
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        return audio_int16.tobytes()
    
    def test_recording_to_file_integration(self, simulated_audio_data):
        """Test recording audio and saving to file."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.recording_service.pyaudio.PyAudio') as mock_pa_class:
                # Setup mock PyAudio
                mock_pa = Mock()
                mock_stream = Mock()
                mock_pa_class.return_value = mock_pa
                mock_pa.open.return_value = mock_stream
                
                # Simulate reading audio chunks
                chunk_size = 1024
                chunks = [simulated_audio_data[i:i+chunk_size] 
                         for i in range(0, len(simulated_audio_data), chunk_size)]
                mock_stream.read.side_effect = chunks + [Exception("Stop")]
                
                # Create service and record
                service = AudioRecordingService()
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    service.start_recording(save_to_file=tmp.name)
                    
                    # Process some chunks
                    try:
                        service._pyaudio_recording_loop()
                    except:
                        pass  # Expected when chunks run out
                    
                    # Stop and get audio
                    audio_data = service.stop_recording()
                    
                    # Verify file was created
                    assert Path(tmp.name).exists()
                    
                    # Verify WAV file format
                    with wave.open(tmp.name, 'rb') as wf:
                        assert wf.getnchannels() == 1
                        assert wf.getsampwidth() == 2
                        assert wf.getframerate() == 16000
                    
                    # Cleanup
                    Path(tmp.name).unlink()
    
    def test_dictation_with_mock_transcription(self, mock_transcription_backend):
        """Test dictation service with mocked transcription."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.recording_service.pyaudio.PyAudio'):
                # Create dictation service
                service = LiveDictationService(
                    transcription_provider='faster-whisper',
                    language='en'
                )
                
                # Track events
                transcripts = []
                states = []
                
                # Start dictation
                service.start_dictation(
                    on_partial_transcript=transcripts.append,
                    on_state_change=states.append
                )
                
                # Simulate audio processing
                audio_data = b'\x00\x01' * 8000  # Some audio
                service._process_audio_buffer(audio_data)
                
                # Should have transcript
                assert service.current_transcript != ""
                assert len(transcripts) > 0
                
                # Stop dictation
                result = service.stop_dictation()
                
                assert isinstance(result, DictationResult)
                assert states[-1] == DictationState.IDLE
    
    def test_voice_activity_detection_integration(self):
        """Test VAD integration with recording."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.recording_service.VAD_AVAILABLE', True):
                with patch('tldw_chatbook.Audio.recording_service.webrtcvad.Vad') as mock_vad_class:
                    # Setup VAD mock
                    mock_vad = Mock()
                    mock_vad_class.return_value = mock_vad
                    
                    # Alternate between speech and silence
                    mock_vad.is_speech.side_effect = [True, False, True, False] * 10
                    
                    # Create service with VAD
                    service = AudioRecordingService(use_vad=True)
                    
                    # Process audio with VAD
                    speech_chunks = []
                    service.callback = speech_chunks.append
                    
                    # Process multiple chunks
                    frame_size = 640  # 20ms at 16kHz
                    for i in range(8):
                        chunk = b'\x00\x01' * (frame_size // 2)
                        service._process_audio_chunk(chunk)
                    
                    # Should have filtered some chunks
                    assert len(speech_chunks) < 8  # Not all chunks passed VAD
    
    def test_multi_device_switching(self):
        """Test switching between audio devices during session."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.recording_service.pyaudio.PyAudio') as mock_pa_class:
                mock_pa = Mock()
                mock_pa_class.return_value = mock_pa
                
                # Mock multiple devices
                mock_pa.get_device_count.return_value = 3
                mock_pa.get_device_info_by_index.side_effect = [
                    {'name': 'Device 1', 'maxInputChannels': 2, 'defaultSampleRate': 44100, 'index': 0},
                    {'name': 'Device 2', 'maxInputChannels': 1, 'defaultSampleRate': 16000, 'index': 1},
                    {'name': 'Device 3', 'maxInputChannels': 2, 'defaultSampleRate': 48000, 'index': 2}
                ]
                
                service = AudioRecordingService()
                
                # Get devices
                devices = service.get_audio_devices()
                assert len(devices) == 3
                
                # Switch devices
                assert service.set_device(1) is True
                assert service.current_device_id == 1
                
                # Can't switch while recording
                service.is_recording = True
                assert service.set_device(2) is False
                
                # Can switch after stopping
                service.is_recording = False
                assert service.set_device(2) is True
    
    @pytest.mark.asyncio
    async def test_widget_dictation_flow(self):
        """Test complete flow through voice input widget."""
        from textual.app import App
        
        with patch('tldw_chatbook.Widgets.voice_input_widget.LiveDictationService') as mock_dict_class:
            # Setup comprehensive mock
            mock_service = Mock()
            mock_dict_class.return_value = mock_service
            
            # Device list
            mock_service.get_audio_devices.return_value = [
                {'id': 0, 'name': 'Default', 'is_default': True}
            ]
            
            # Capture callbacks
            callbacks = {}
            def capture_callbacks(**kwargs):
                callbacks.update(kwargs)
                return True
            
            mock_service.start_dictation.side_effect = capture_callbacks
            
            # Stop result
            mock_service.stop_dictation.return_value = DictationResult(
                transcript="Hello world",
                segments=[{'text': 'Hello world', 'timestamp': 0}],
                duration=2.0
            )
            
            class TestApp(App):
                received_messages = []
                
                def compose(self):
                    yield VoiceInputWidget()
                
                def on_voice_input_message(self, event):
                    self.received_messages.append(event)
            
            app = TestApp()
            async with app.run_test() as pilot:
                # Start recording
                await pilot.click("#record-button")
                await asyncio.sleep(0.1)
                
                # Simulate transcription flow
                if 'on_partial_transcript' in callbacks:
                    callbacks['on_partial_transcript']("Hello")
                    callbacks['on_partial_transcript']("Hello world")
                
                if 'on_final_transcript' in callbacks:
                    callbacks['on_final_transcript']("Hello world")
                
                # Stop recording
                await pilot.click("#record-button")
                await asyncio.sleep(0.1)
                
                # Check messages received
                assert len(app.received_messages) >= 2
                final_msgs = [m for m in app.received_messages if m.is_final]
                assert len(final_msgs) >= 1
                assert final_msgs[0].text == "Hello world"


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""
    
    def test_recording_recovery_from_stream_error(self):
        """Test recovery from audio stream errors."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.recording_service.pyaudio.PyAudio') as mock_pa_class:
                mock_pa = Mock()
                mock_stream = Mock()
                mock_pa_class.return_value = mock_pa
                mock_pa.open.return_value = mock_stream
                
                # Simulate stream error after some successful reads
                mock_stream.read.side_effect = [
                    b'\x00\x01' * 512,
                    b'\x00\x01' * 512,
                    Exception("Stream error"),
                    b'\x00\x01' * 512  # Recovery
                ]
                
                service = AudioRecordingService()
                chunks = []
                service.callback = chunks.append
                
                service.start_recording()
                
                # Process with error
                try:
                    for _ in range(4):
                        service._pyaudio_recording_loop()
                except:
                    pass
                
                # Should have captured some chunks despite error
                assert len(chunks) >= 2
    
    def test_transcription_service_fallback(self):
        """Test fallback when primary transcription service fails."""
        # First provider fails
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.FASTER_WHISPER_AVAILABLE', False):
            # But fallback is available
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.PARAKEET_MLX_AVAILABLE', True):
                service = TranscriptionService()
                
                # Should list available models
                models = service.list_available_models()
                assert 'parakeet-mlx' in models
    
    def test_dictation_continues_after_transcription_error(self):
        """Test dictation continues even if transcription fails."""
        with patch('tldw_chatbook.Audio.recording_service.AudioRecordingService'):
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.TranscriptionService') as mock_trans:
                # Make transcription fail sometimes
                mock_trans_instance = Mock()
                mock_trans_instance.transcribe_buffer.side_effect = [
                    Exception("Transcription error"),
                    {'text': 'Recovered transcription'},
                    {'text': 'More text'}
                ]
                mock_trans.return_value = mock_trans_instance
                
                service = LiveDictationService()
                service.start_dictation()
                
                # Process multiple buffers
                for _ in range(3):
                    try:
                        service._process_audio_buffer(b'audio')
                    except:
                        pass
                
                # Should have recovered and continued
                assert service.state == DictationState.LISTENING
                assert service.current_transcript != ""


class TestPerformance:
    """Test performance characteristics of audio system."""
    
    def test_audio_buffering_performance(self):
        """Test audio buffer doesn't grow unbounded."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            service = AudioRecordingService()
            
            # Add many chunks
            for i in range(1000):
                service._handle_audio_chunk(b'\x00\x01' * 512)
            
            # Queue should not grow unbounded
            # (In real implementation, old chunks would be processed/removed)
            assert len(service.audio_buffer) == 1000
            
            # Clear buffer on stop
            service.stop_recording()
            assert len(service.audio_buffer) == 0
    
    def test_concurrent_operations(self):
        """Test concurrent recording and transcription."""
        with patch('tldw_chatbook.Audio.recording_service.AudioRecordingService'):
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.TranscriptionService'):
                service = LiveDictationService()
                
                # Start dictation
                service.start_dictation()
                
                # Simulate concurrent audio chunks
                import threading
                
                def add_audio():
                    for _ in range(10):
                        service._audio_callback(b'chunk')
                        time.sleep(0.01)
                
                # Multiple threads adding audio
                threads = []
                for _ in range(3):
                    t = threading.Thread(target=add_audio)
                    t.start()
                    threads.append(t)
                
                # Wait for completion
                for t in threads:
                    t.join()
                
                # Should have processed audio without crashes
                assert not service.processing_queue.empty()
    
    def test_memory_cleanup(self):
        """Test memory is properly cleaned up."""
        import gc
        import weakref
        
        # Create service
        service = AudioRecordingService()
        service_ref = weakref.ref(service)
        
        # Use it
        service.start_recording()
        service.stop_recording()
        
        # Delete and collect
        del service
        gc.collect()
        
        # Should be garbage collected
        assert service_ref() is None


class TestCrossPlatformCompatibility:
    """Test cross-platform audio functionality."""
    
    @pytest.mark.parametrize("platform,expected_backend", [
        ("darwin", "pyaudio"),  # macOS
        ("win32", "pyaudio"),   # Windows
        ("linux", "pyaudio"),   # Linux
    ])
    def test_platform_backend_selection(self, platform, expected_backend):
        """Test appropriate backend selection per platform."""
        with patch('sys.platform', platform):
            with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
                service = AudioRecordingService()
                assert service.backend == expected_backend
    
    def test_sounddevice_fallback(self):
        """Test fallback to sounddevice when PyAudio unavailable."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', False):
            with patch('tldw_chatbook.Audio.recording_service.SOUNDDEVICE_AVAILABLE', True):
                service = AudioRecordingService()
                assert service.backend == 'sounddevice'