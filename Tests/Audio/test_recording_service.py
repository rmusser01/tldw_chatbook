# test_recording_service.py
"""
Unit tests for AudioRecordingService.
Tests audio capture, device enumeration, and recording functionality.
"""

import pytest
import time
import threading
import queue
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from pathlib import Path
import tempfile
import wave

from tldw_chatbook.Audio.recording_service import (
    AudioRecordingService,
    AudioRecordingError,
    NoAudioBackendError,
    AudioDeviceError
)


class TestAudioRecordingService:
    """Unit tests for AudioRecordingService."""
    
    @pytest.fixture
    def mock_pyaudio(self):
        """Mock PyAudio for testing."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.recording_service.pyaudio') as mock_pa:
                yield mock_pa
    
    @pytest.fixture
    def mock_sounddevice(self):
        """Mock sounddevice for testing."""
        with patch('tldw_chatbook.Audio.recording_service.SOUNDDEVICE_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.recording_service.sd') as mock_sd:
                yield mock_sd
    
    @pytest.fixture
    def mock_vad(self):
        """Mock WebRTC VAD for testing."""
        with patch('tldw_chatbook.Audio.recording_service.VAD_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.recording_service.webrtcvad') as mock_vad:
                yield mock_vad
    
    def test_initialization_with_pyaudio(self, mock_pyaudio):
        """Test initialization with PyAudio backend."""
        service = AudioRecordingService(backend='pyaudio')
        assert service.backend == 'pyaudio'
        assert service.sample_rate == 16000
        assert service.channels == 1
        assert service.chunk_size == 1024
    
    def test_initialization_with_sounddevice(self, mock_sounddevice):
        """Test initialization with sounddevice backend."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', False):
            service = AudioRecordingService(backend='sounddevice')
            assert service.backend == 'sounddevice'
    
    def test_initialization_auto_backend_selection(self, mock_pyaudio):
        """Test automatic backend selection."""
        service = AudioRecordingService()
        assert service.backend == 'pyaudio'  # PyAudio is preferred
    
    def test_initialization_no_backend_available(self):
        """Test initialization fails when no backend is available."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', False):
            with patch('tldw_chatbook.Audio.recording_service.SOUNDDEVICE_AVAILABLE', False):
                with pytest.raises(NoAudioBackendError):
                    AudioRecordingService()
    
    def test_initialization_with_vad(self, mock_pyaudio, mock_vad):
        """Test initialization with VAD enabled."""
        mock_vad_instance = Mock()
        mock_vad.Vad.return_value = mock_vad_instance
        
        service = AudioRecordingService(use_vad=True, vad_aggressiveness=3)
        
        assert service.use_vad is True
        assert service.vad is not None
        mock_vad.Vad.assert_called_once()
        mock_vad_instance.set_mode.assert_called_once_with(3)
    
    def test_get_audio_devices_pyaudio(self, mock_pyaudio):
        """Test getting audio devices with PyAudio."""
        # Mock PyAudio instance
        mock_pa_instance = Mock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        
        # Mock device info
        mock_pa_instance.get_device_count.return_value = 2
        mock_pa_instance.get_device_info_by_index.side_effect = [
            {
                'name': 'Microphone',
                'maxInputChannels': 2,
                'defaultSampleRate': 44100,
                'index': 0
            },
            {
                'name': 'Webcam Mic',
                'maxInputChannels': 1,
                'defaultSampleRate': 16000,
                'index': 1
            }
        ]
        mock_pa_instance.get_default_input_device_info.return_value = {'index': 0}
        
        service = AudioRecordingService(backend='pyaudio')
        devices = service.get_audio_devices()
        
        assert len(devices) == 2
        assert devices[0]['name'] == 'Microphone'
        assert devices[0]['is_default'] is True
        assert devices[1]['name'] == 'Webcam Mic'
        assert devices[1]['is_default'] is False
    
    def test_get_audio_devices_sounddevice(self, mock_sounddevice):
        """Test getting audio devices with sounddevice."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', False):
            # Mock device list
            mock_sounddevice.query_devices.return_value = [
                {
                    'name': 'Built-in Mic',
                    'max_input_channels': 2,
                    'default_samplerate': 48000
                },
                {
                    'name': 'USB Mic',
                    'max_input_channels': 1,
                    'default_samplerate': 44100
                }
            ]
            mock_sounddevice.default.device = (0, 0)
            
            service = AudioRecordingService(backend='sounddevice')
            devices = service.get_audio_devices()
            
            assert len(devices) == 2
            assert devices[0]['name'] == 'Built-in Mic'
            assert devices[0]['is_default'] is True
    
    def test_set_device_valid(self, mock_pyaudio):
        """Test setting a valid audio device."""
        service = AudioRecordingService()
        
        # Mock get_audio_devices to return a device
        with patch.object(service, 'get_audio_devices', return_value=[
            {'id': 0, 'name': 'Device 1'},
            {'id': 1, 'name': 'Device 2'}
        ]):
            result = service.set_device(1)
            assert result is True
            assert service.current_device_id == 1
    
    def test_set_device_invalid(self, mock_pyaudio):
        """Test setting an invalid audio device."""
        service = AudioRecordingService()
        
        # Mock get_audio_devices to return a device
        with patch.object(service, 'get_audio_devices', return_value=[
            {'id': 0, 'name': 'Device 1'}
        ]):
            result = service.set_device(99)  # Non-existent device
            assert result is False
    
    def test_set_device_while_recording(self, mock_pyaudio):
        """Test that device cannot be changed while recording."""
        service = AudioRecordingService()
        service.is_recording = True
        
        result = service.set_device(0)
        assert result is False
    
    def test_start_recording_success(self, mock_pyaudio):
        """Test successful start of recording."""
        service = AudioRecordingService()
        
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            result = service.start_recording()
            
            assert result is True
            assert service.is_recording is True
            assert service.callback is None
            mock_thread_instance.start.assert_called_once()
    
    def test_start_recording_with_callback(self, mock_pyaudio):
        """Test starting recording with callback."""
        service = AudioRecordingService()
        callback = Mock()
        
        with patch('threading.Thread'):
            result = service.start_recording(callback=callback)
            
            assert result is True
            assert service.callback is callback
    
    def test_start_recording_already_recording(self, mock_pyaudio):
        """Test that starting recording twice fails."""
        service = AudioRecordingService()
        service.is_recording = True
        
        result = service.start_recording()
        assert result is False
    
    def test_stop_recording_success(self, mock_pyaudio):
        """Test successful stop of recording."""
        service = AudioRecordingService()
        service.is_recording = True
        service.audio_buffer = [b'audio', b'data']
        
        # Mock thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        service.recording_thread = mock_thread
        
        # Mock PyAudio instance
        mock_pa_instance = Mock()
        service.pyaudio_instance = mock_pa_instance
        
        result = service.stop_recording()
        
        assert result == b'audiodata'
        assert service.is_recording is False
        assert service.audio_buffer == []
        mock_pa_instance.terminate.assert_called_once()
    
    def test_stop_recording_not_recording(self, mock_pyaudio):
        """Test stopping when not recording."""
        service = AudioRecordingService()
        service.is_recording = False
        
        result = service.stop_recording()
        assert result is None
    
    def test_audio_callback_processing(self, mock_pyaudio):
        """Test audio chunk processing with callback."""
        service = AudioRecordingService()
        callback = Mock()
        service.callback = callback
        service.audio_buffer = []
        
        # Test audio chunk
        chunk = b'\x00\x01' * 512  # 1024 bytes
        
        service._handle_audio_chunk(chunk)
        
        assert len(service.audio_buffer) == 1
        assert service.audio_buffer[0] == chunk
        callback.assert_called_once_with(chunk)
    
    def test_vad_processing(self, mock_pyaudio, mock_vad):
        """Test Voice Activity Detection processing."""
        # Setup VAD
        mock_vad_instance = Mock()
        mock_vad_instance.is_speech.return_value = True
        mock_vad.Vad.return_value = mock_vad_instance
        
        service = AudioRecordingService(use_vad=True)
        service.vad = mock_vad_instance
        service.audio_buffer = []
        
        # Create test audio chunk (640 bytes = 20ms at 16kHz)
        chunk = b'\x00\x01' * 320
        
        service._process_audio_chunk(chunk)
        
        # VAD should be called
        assert mock_vad_instance.is_speech.called
        assert len(service.audio_buffer) > 0
    
    def test_get_audio_level(self, mock_pyaudio):
        """Test audio level calculation."""
        service = AudioRecordingService()
        
        # Add some audio data to queue
        audio_data = np.array([0, 16383, -16384, 8192], dtype=np.int16)
        service.audio_queue.put(audio_data.tobytes())
        
        level = service.get_audio_level()
        
        assert 0.0 <= level <= 1.0
    
    def test_get_audio_level_empty_queue(self, mock_pyaudio):
        """Test audio level with empty queue."""
        service = AudioRecordingService()
        
        level = service.get_audio_level()
        assert level == 0.0
    
    def test_save_audio_file(self, mock_pyaudio):
        """Test saving audio to WAV file."""
        service = AudioRecordingService()
        audio_data = b'\x00\x01' * 1000
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            service._save_audio_file(audio_data, tmp.name)
            
            # Verify file was created and has correct format
            assert Path(tmp.name).exists()
            
            with wave.open(tmp.name, 'rb') as wf:
                assert wf.getnchannels() == service.channels
                assert wf.getsampwidth() == 2  # 16-bit
                assert wf.getframerate() == service.sample_rate
            
            # Cleanup
            Path(tmp.name).unlink()
    
    def test_recording_session_context_manager(self, mock_pyaudio):
        """Test recording session context manager."""
        service = AudioRecordingService()
        
        with patch.object(service, 'start_recording', return_value=True) as mock_start:
            with patch.object(service, 'stop_recording', return_value=b'audio') as mock_stop:
                with service.recording_session() as session:
                    assert session is service
                
                mock_start.assert_called_once()
                mock_stop.assert_called_once()
    
    def test_is_available(self, mock_pyaudio):
        """Test checking if recording is available."""
        service = AudioRecordingService()
        assert service.is_available() is True
        
        service.backend = None
        assert service.is_available() is False
    
    def test_cleanup_on_delete(self, mock_pyaudio):
        """Test cleanup when service is deleted."""
        service = AudioRecordingService()
        service.is_recording = True
        
        # Mock stop_recording
        with patch.object(service, 'stop_recording') as mock_stop:
            service.__del__()
            mock_stop.assert_called_once()
    
    @pytest.mark.parametrize("sample_rate,channels,chunk_size", [
        (8000, 1, 512),
        (16000, 2, 1024),
        (44100, 1, 2048),
        (48000, 2, 4096)
    ])
    def test_various_audio_configurations(self, mock_pyaudio, sample_rate, channels, chunk_size):
        """Test service with various audio configurations."""
        service = AudioRecordingService(
            sample_rate=sample_rate,
            channels=channels,
            chunk_size=chunk_size
        )
        
        assert service.sample_rate == sample_rate
        assert service.channels == channels
        assert service.chunk_size == chunk_size


class TestAudioRecordingIntegration:
    """Integration tests for audio recording with mocked backends."""
    
    def test_pyaudio_recording_flow(self):
        """Test complete recording flow with PyAudio backend."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.recording_service.pyaudio.PyAudio') as mock_pa_class:
                # Setup mocks
                mock_pa_instance = Mock()
                mock_stream = Mock()
                mock_pa_class.return_value = mock_pa_instance
                mock_pa_instance.open.return_value = mock_stream
                
                # Simulate audio data
                test_audio = b'\x00\x01' * 512
                mock_stream.read.return_value = test_audio
                
                service = AudioRecordingService(backend='pyaudio')
                
                # Record for a short time
                chunks_received = []
                def callback(chunk):
                    chunks_received.append(chunk)
                    if len(chunks_received) >= 3:
                        service.is_recording = False
                
                service.start_recording(callback=callback)
                
                # Run recording loop briefly
                service._pyaudio_recording_loop()
                
                assert len(chunks_received) >= 3
                assert all(chunk == test_audio for chunk in chunks_received)
    
    def test_sounddevice_recording_flow(self):
        """Test complete recording flow with sounddevice backend."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', False):
            with patch('tldw_chatbook.Audio.recording_service.SOUNDDEVICE_AVAILABLE', True):
                with patch('tldw_chatbook.Audio.recording_service.sd.InputStream') as mock_stream_class:
                    # Setup context manager mock
                    mock_stream = MagicMock()
                    mock_stream_class.return_value.__enter__.return_value = mock_stream
                    mock_stream_class.return_value.__exit__.return_value = None
                    
                    service = AudioRecordingService(backend='sounddevice')
                    
                    # Start recording
                    service.start_recording()
                    
                    # Simulate callback being called
                    callback_func = mock_stream_class.call_args[1]['callback']
                    test_data = np.array([[0.5], [-0.5], [0.25], [-0.25]], dtype=np.float32)
                    callback_func(test_data, 4, None, None)
                    
                    # Stop recording
                    service.is_recording = False
                    
                    # Check that audio was processed
                    assert not service.audio_queue.empty()


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_callback_error_handling(self):
        """Test that callback errors don't crash recording."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            service = AudioRecordingService()
            
            # Callback that raises exception
            def bad_callback(chunk):
                raise ValueError("Test error")
            
            service.callback = bad_callback
            service.audio_buffer = []
            
            # Should not raise
            service._handle_audio_chunk(b'test')
            
            # Audio should still be buffered
            assert len(service.audio_buffer) == 1
    
    def test_device_enumeration_error(self):
        """Test handling of device enumeration errors."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.recording_service.pyaudio.PyAudio') as mock_pa_class:
                mock_pa_instance = Mock()
                mock_pa_class.return_value = mock_pa_instance
                
                # Make device enumeration fail
                mock_pa_instance.get_device_count.side_effect = Exception("Audio system error")
                
                service = AudioRecordingService()
                devices = service.get_audio_devices()
                
                # Should return empty list on error
                assert devices == []
    
    def test_recording_thread_error(self):
        """Test handling of recording thread errors."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            service = AudioRecordingService()
            
            # Mock recording loop to raise exception
            with patch.object(service, '_pyaudio_recording_loop', side_effect=Exception("Recording error")):
                # Start recording in thread
                service.start_recording()
                
                # Wait a bit
                time.sleep(0.1)
                
                # Should have stopped due to error
                assert service.is_recording is False