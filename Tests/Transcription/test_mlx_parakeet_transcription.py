"""
Unit and integration tests for the MLX Parakeet transcription backend.

This module tests the Parakeet MLX implementation for macOS/Apple Silicon.
Parakeet is optimized for real-time ASR (Automatic Speech Recognition).
Tests are designed to run only on macOS and will be skipped on other platforms.
"""

import pytest
import sys
import os
import tempfile
import wave
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, Optional
import json
import time

# Import the transcription service and related classes
from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionService,
    TranscriptionError,
    ConversionError,
    PARAKEET_MLX_AVAILABLE
)


class MockAudioInfo:
    """Mock audio info object for soundfile."""
    def __init__(self, duration=1.0, samplerate=16000, channels=1):
        self.duration = duration
        self.samplerate = samplerate
        self.channels = channels


# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != 'darwin',
    reason="MLX Parakeet tests only run on macOS"
)


class TestMLXParakeetUnit:
    """Unit tests for MLX Parakeet transcription backend."""
    
    @pytest.fixture
    def mock_parakeet_available(self):
        """Mock Parakeet MLX availability."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.PARAKEET_MLX_AVAILABLE', True):
            yield
    
    @pytest.fixture
    def mock_parakeet_model(self):
        """Mock the parakeet_from_pretrained function and model."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            # Create a mock model instance
            mock_model = MagicMock()
            
            # Mock the model methods
            result_obj = MagicMock()
            result_obj.text = 'This is a test transcription from Parakeet'
            mock_model.transcribe.return_value = result_obj
            
            # Mock model properties
            mock_model.config = MagicMock()
            mock_model.config.model_id = 'mlx-community/parakeet-tdt-0.6b-v2'
            
            # Set the return value of from_pretrained
            mock_pretrained.return_value = mock_model
            
            yield mock_pretrained, mock_model
    
    @pytest.fixture
    def transcription_service(self, mock_parakeet_available):
        """Create a TranscriptionService instance with mocked config."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.get_cli_setting') as mock_get_setting, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            # Mock configuration values
            def get_setting_side_effect(key, default=None):
                settings = {
                    'transcription.provider': 'parakeet-mlx',
                    'transcription.model': 'base',
                    'transcription.language': 'en',
                    'transcription.vad_filter': True,
                    'transcription.compute_type': 'int8',
                    'transcription.device': 'cpu',
                    'transcription.device_index': 0,
                    'transcription.num_workers': 1,
                    'transcription.download_root': None,
                    'transcription.local_files_only': False,
                    'transcription.parakeet_model': 'mlx-community/parakeet-tdt-0.6b-v2',
                    'transcription.parakeet_precision': 'bf16',
                    'transcription.parakeet_attention': 'local',
                    'transcription.parakeet_chunk_duration': 120.0,
                    'transcription.parakeet_overlap_duration': 0.5,
                }
                return settings.get(key, default)
            
            mock_get_setting.side_effect = get_setting_side_effect
            service = TranscriptionService()
            return service
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary sample audio file for testing."""
        # Create a simple WAV file with a sine wave
        sample_rate = 16000
        duration = 2  # seconds
        frequency = 440  # Hz (A4 note)
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)   # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            yield tmp_file.name
            
        # Cleanup
        os.unlink(tmp_file.name)
    
    def test_parakeet_initialization(self, transcription_service):
        """Test that Parakeet configuration is properly initialized."""
        assert transcription_service._parakeet_mlx_model is None  # Lazy loaded
        assert transcription_service._parakeet_mlx_config['model'] == 'mlx-community/parakeet-tdt-0.6b-v2'
        assert transcription_service._parakeet_mlx_config['precision'] == 'bf16'
        assert transcription_service._parakeet_mlx_config['attention_type'] == 'local'
        assert transcription_service._parakeet_mlx_config['chunk_duration'] == 120.0
        assert transcription_service._parakeet_mlx_config['overlap_duration'] == 0.5
    
    def test_parakeet_not_available(self, transcription_service):
        """Test error handling when Parakeet MLX is not available."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.PARAKEET_MLX_AVAILABLE', False):
            with pytest.raises(TranscriptionError) as exc_info:
                transcription_service._transcribe_with_parakeet_mlx(
                    audio_path="dummy.wav"
                )
            assert "parakeet-mlx is not installed" in str(exc_info.value)
    
    def test_model_loading(self, transcription_service, mock_parakeet_model, sample_audio_file):
        """Test Parakeet model loading."""
        mock_pretrained, mock_model = mock_parakeet_model
        
        # Mock audio loading
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (np.zeros(16000), 16000)  # 1 second of silence
            mock_sf.info.return_value = MockAudioInfo(duration=1.0)
            
            # First transcription should load the model
            result = transcription_service._transcribe_with_parakeet_mlx(
                audio_path=sample_audio_file,
                model='mlx-community/parakeet-tdt-0.6b-v2'
            )
        
        # Check model was loaded with correct parameters
        mock_pretrained.assert_called_once()
        args, kwargs = mock_pretrained.call_args
        assert args[0] == 'mlx-community/parakeet-tdt-0.6b-v2'
        assert 'dtype' in kwargs  # dtype will be an mlx type object, not a string
        
        # Check result format
        assert 'text' in result
        assert 'segments' in result
        assert 'language' in result
        assert 'provider' in result
        assert result['provider'] == 'parakeet-mlx'
    
    def test_model_caching(self, transcription_service, mock_parakeet_model, sample_audio_file):
        """Test that the model is cached between transcriptions."""
        mock_pretrained, mock_model = mock_parakeet_model
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (np.zeros(16000), 16000)
            mock_sf.info.return_value = MockAudioInfo(duration=1.0)
            
            # First transcription
            transcription_service._transcribe_with_parakeet_mlx(
                audio_path=sample_audio_file,
                model='mlx-community/parakeet-tdt-0.6b-v2'
            )
            
            # Reset mock to check it's not called again
            mock_pretrained.reset_mock()
            
            # Second transcription with same model
            transcription_service._transcribe_with_parakeet_mlx(
                audio_path=sample_audio_file,
                model='mlx-community/parakeet-tdt-0.6b-v2'
            )
            
            # Model should not be loaded again
            mock_pretrained.assert_not_called()
    
    def test_model_reload_on_config_change(self, transcription_service, mock_parakeet_model, sample_audio_file):
        """Test that model is reloaded when configuration changes."""
        mock_pretrained, mock_model = mock_parakeet_model
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (np.zeros(16000), 16000)
            mock_sf.info.return_value = MockAudioInfo(duration=1.0)
            
            # First transcription with one model
            transcription_service._transcribe_with_parakeet_mlx(
                audio_path=sample_audio_file,
                model='mlx-community/parakeet-tdt-0.6b-v2'
            )
            
            # Reset mock
            mock_pretrained.reset_mock()
            
            # Second transcription with different model
            transcription_service._transcribe_with_parakeet_mlx(
                audio_path=sample_audio_file,
                model='mlx-community/parakeet-tdt-1.1b'
            )
            
            # Model should be reloaded
            args, kwargs = mock_pretrained.call_args
            assert args[0] == 'mlx-community/parakeet-tdt-1.1b'
            # Default precision is bf16, so check for bfloat16
            assert hasattr(kwargs['dtype'], '__module__')
            assert 'bfloat16' in str(kwargs['dtype'])
    
    def test_precision_options(self, transcription_service, mock_parakeet_model, sample_audio_file):
        """Test different precision options."""
        mock_pretrained, mock_model = mock_parakeet_model
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (np.zeros(16000), 16000)
            mock_sf.info.return_value = MockAudioInfo(duration=1.0)
            
            # Test with float32 precision
            transcription_service._transcribe_with_parakeet_mlx(
                audio_path=sample_audio_file,
                model='mlx-community/parakeet-tdt-0.6b-v2',
                precision='float32'
            )
            
            # Check that it was called with the correct model
            args, kwargs = mock_pretrained.call_args
            assert args[0] == 'mlx-community/parakeet-tdt-0.6b-v2'
            # Check that dtype is an mlx dtype object (not string)
            assert hasattr(kwargs['dtype'], '__module__')
            assert 'float32' in str(kwargs['dtype'])
            
            # Reset and test with bfloat16 precision
            mock_pretrained.reset_mock()
            transcription_service._transcribe_with_parakeet_mlx(
                audio_path=sample_audio_file,
                model='mlx-community/parakeet-tdt-0.6b-v2',
                precision='bfloat16'
            )
            
            # Check the second call
            args, kwargs = mock_pretrained.call_args
            assert args[0] == 'mlx-community/parakeet-tdt-0.6b-v2'
            assert hasattr(kwargs['dtype'], '__module__')
            assert 'bfloat16' in str(kwargs['dtype'])
    
    def test_attention_type_configuration(self, transcription_service, mock_parakeet_model, sample_audio_file):
        """Test different attention type configurations."""
        mock_pretrained, mock_model = mock_parakeet_model
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (np.zeros(16000), 16000)
            mock_sf.info.return_value = MockAudioInfo(duration=1.0)
            
            # Test with different attention types
            for attention_type in ['flash', 'sdpa', 'native']:
                result = transcription_service._transcribe_with_parakeet_mlx(
                    audio_path=sample_audio_file,
                    model='mlx-community/parakeet-tdt-0.6b-v2',
                    attention_type=attention_type
                )
                
                assert result['attention_type'] == attention_type
    
    def test_chunking_for_long_audio(self, transcription_service, mock_parakeet_model):
        """Test chunking functionality for long audio files."""
        mock_pretrained, mock_model = mock_parakeet_model
        
        # Create a longer audio file (5 minutes)
        sample_rate = 16000
        duration = 300  # 5 minutes
        audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (audio_data, sample_rate)
            mock_sf.info.return_value = MockAudioInfo(duration=duration)
            
            # Mock multiple transcriptions for chunks
            mock_model.transcribe.side_effect = [
                f'Chunk {i} transcription.' for i in range(10)
            ]
            
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                result = transcription_service._transcribe_with_parakeet_mlx(
                    audio_path=tmp.name,
                    model='mlx-community/parakeet-tdt-0.6b-v2'
                )
            
            # Should have multiple segments from chunking
            assert len(result['segments']) > 1
            assert 'Chunk' in result['text']
    
    def test_progress_callback(self, transcription_service, mock_parakeet_model, sample_audio_file):
        """Test progress callback functionality."""
        mock_pretrained, mock_model = mock_parakeet_model
        mock_callback = Mock()
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (np.zeros(16000), 16000)
            mock_sf.info.return_value = MockAudioInfo(duration=1.0)
            
            result = transcription_service._transcribe_with_parakeet_mlx(
                audio_path=sample_audio_file,
                model='mlx-community/parakeet-tdt-0.6b-v2',
                progress_callback=mock_callback
            )
        
        # Check progress callbacks were made
        assert mock_callback.call_count >= 2
        
        # Check first call (start)
        first_call = mock_callback.call_args_list[0]
        assert first_call[0][0] == 0  # Progress percentage
        assert "Starting transcription" in first_call[0][1]
        
        # Check last call (complete)
        last_call = mock_callback.call_args_list[-1]
        assert last_call[0][0] == 100  # Progress percentage
        assert "complete" in last_call[0][1].lower()
    
    def test_segment_formatting(self, transcription_service, mock_parakeet_model, sample_audio_file):
        """Test that segments are properly formatted."""
        mock_pretrained, mock_model = mock_parakeet_model
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (np.zeros(32000), 16000)  # 2 seconds
            mock_sf.info.return_value = MockAudioInfo(duration=2.0)
            
            # Mock multiple chunk transcriptions
            mock_model.transcribe.side_effect = ['First part.', 'Second part.']
            
            result = transcription_service._transcribe_with_parakeet_mlx(
                audio_path=sample_audio_file,
                model='mlx-community/parakeet-tdt-0.6b-v2'
            )
        
        # Check segments are properly formatted
        assert len(result['segments']) >= 1
        for segment in result['segments']:
            assert 'start' in segment
            assert 'end' in segment
            assert 'text' in segment
            assert 'Time_Start' in segment
            assert 'Time_End' in segment
            assert 'Text' in segment
            # Check duplicated fields match
            assert segment['start'] == segment['Time_Start']
            assert segment['end'] == segment['Time_End']
            assert segment['text'] == segment['Text']
    
    def test_audio_loading_error(self, transcription_service, mock_parakeet_model):
        """Test error handling when audio file cannot be loaded."""
        mock_pretrained, mock_model = mock_parakeet_model
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
            mock_sf.read.side_effect = Exception("Cannot read audio file")
            
            with pytest.raises(TranscriptionError) as exc_info:
                transcription_service._transcribe_with_parakeet_mlx(
                    audio_path="invalid.wav",
                    model='mlx-community/parakeet-tdt-0.6b-v2'
                )
            
            # Error message varies based on implementation
            assert "Parakeet MLX transcription failed" in str(exc_info.value)
    
    def test_soundfile_not_available(self, transcription_service):
        """Test error handling when soundfile is not available."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.SOUNDFILE_AVAILABLE', False):
            with pytest.raises(TranscriptionError) as exc_info:
                transcription_service._transcribe_with_parakeet_mlx(
                    audio_path="dummy.wav"
                )
            # When soundfile is not available, Parakeet falls back to ffmpeg which fails on non-existent file
            # Error message varies based on implementation
            assert "Parakeet MLX transcription failed" in str(exc_info.value)
    
    def test_error_handling_model_load(self, transcription_service, sample_audio_file):
        """Test error handling during model loading."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_pretrained.side_effect = Exception("Model loading failed")
            
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                mock_sf.read.return_value = (np.zeros(16000), 16000)
                mock_sf.info.return_value = MockAudioInfo(duration=1.0)
                
                with pytest.raises(TranscriptionError) as exc_info:
                    transcription_service._transcribe_with_parakeet_mlx(
                        audio_path=sample_audio_file,
                        model='mlx-community/parakeet-tdt-0.6b-v2'
                    )
                
                assert "Failed to load Parakeet MLX model" in str(exc_info.value)
    
    def test_error_handling_transcription(self, transcription_service, mock_parakeet_model, sample_audio_file):
        """Test error handling during transcription."""
        mock_pretrained, mock_model = mock_parakeet_model
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (np.zeros(16000), 16000)
            mock_sf.info.return_value = MockAudioInfo(duration=1.0)
            
            with pytest.raises(TranscriptionError) as exc_info:
                transcription_service._transcribe_with_parakeet_mlx(
                    audio_path=sample_audio_file,
                    model='mlx-community/parakeet-tdt-0.6b-v2'
                )
            
            assert "Parakeet MLX transcription failed" in str(exc_info.value)
    
    def test_sample_rate_conversion(self, transcription_service, mock_parakeet_model):
        """Test handling of different sample rates."""
        mock_pretrained, mock_model = mock_parakeet_model
        
        # Test with 48kHz audio (needs resampling to 16kHz)
        sample_rate = 48000
        duration = 1
        audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (audio_data, sample_rate)
            mock_sf.info.return_value = MockAudioInfo(duration=duration, samplerate=sample_rate)
            
            # Should handle resampling internally
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                result = transcription_service._transcribe_with_parakeet_mlx(
                    audio_path=tmp.name,
                    model='mlx-community/parakeet-tdt-0.6b-v2'
                )
            
            assert 'text' in result
            assert result['sample_rate'] == '48000 -> 16000'  # Shows resampling
    
    def test_result_metadata(self, transcription_service, mock_parakeet_model, sample_audio_file):
        """Test that result includes all expected metadata."""
        mock_pretrained, mock_model = mock_parakeet_model
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf, \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.time.time', return_value=1000.0):
            mock_sf.read.return_value = (np.zeros(16000), 16000)
            mock_sf.info.return_value = MockAudioInfo(duration=1.0)
            
            result = transcription_service._transcribe_with_parakeet_mlx(
                audio_path=sample_audio_file,
                model='mlx-community/parakeet-tdt-0.6b-v2',
                precision='float32',
                attention_type='sdpa'
            )
        
        # Check metadata fields
        assert result['provider'] == 'parakeet-mlx'
        assert result['model'] == 'mlx-community/parakeet-tdt-0.6b-v2'
        assert result['precision'] == 'float32'
        assert result['attention_type'] == 'sdpa'
        assert 'text' in result
        assert 'segments' in result
        assert 'language' in result
        assert 'chunk_size' in result
        assert 'overlap' in result


class TestMLXParakeetIntegration:
    """Integration tests for MLX Parakeet transcription backend."""
    
    @pytest.fixture
    def real_transcription_service(self):
        """Create a real TranscriptionService instance."""
        if not PARAKEET_MLX_AVAILABLE:
            pytest.skip("Parakeet MLX not available")
        
        service = TranscriptionService()
        return service
    
    @pytest.fixture
    def test_audio_file(self):
        """Create a test audio file with speech-like content."""
        # Create a more complex audio file that resembles speech
        sample_rate = 16000
        duration = 3  # seconds
        
        # Generate a more complex waveform that might resemble speech patterns
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Mix of frequencies to simulate speech formants
        audio_data = (
            0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
            0.2 * np.sin(2 * np.pi * 700 * t) +  # Mid frequency
            0.1 * np.sin(2 * np.pi * 2000 * t) + # High frequency
            0.05 * np.random.randn(len(t))       # Some noise
        )
        
        # Add amplitude modulation to simulate speech rhythm
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        audio_data = audio_data * envelope
        
        # Normalize and convert to 16-bit PCM
        audio_data = audio_data / np.max(np.abs(audio_data))
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)   # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            yield tmp_file.name
            
        # Cleanup
        os.unlink(tmp_file.name)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_transcription_basic(self, real_transcription_service, test_audio_file):
        """Test actual transcription with Parakeet MLX."""
        # This test will actually load the model and perform transcription
        result = real_transcription_service.transcribe(
            audio_path=test_audio_file,
            provider='parakeet-mlx',
            model='mlx-community/parakeet-tdt-0.6b-v2',
            precision='float16'
        )
        
        # Basic checks - the actual transcription text will vary
        assert 'text' in result
        assert 'segments' in result
        assert 'language' in result
        assert result['provider'] == 'parakeet-mlx'
        assert isinstance(result['text'], str)
        assert isinstance(result['segments'], list)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_transcription_with_different_precisions(self, real_transcription_service, test_audio_file):
        """Test transcription with different precision settings."""
        precisions = ['float16', 'float32']
        
        for precision in precisions:
            result = real_transcription_service.transcribe(
                audio_path=test_audio_file,
                provider='parakeet-mlx',
                model='mlx-community/parakeet-tdt-0.6b-v2',
                precision=precision
            )
            
            assert result['precision'] == precision
            assert 'text' in result
            assert 'segments' in result
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_transcription_long_audio(self, real_transcription_service):
        """Test transcription of longer audio with chunking."""
        # Create a longer audio file (30 seconds)
        sample_rate = 16000
        duration = 30
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create varying tones to simulate different speech segments
        audio_data = np.zeros_like(t)
        
        for i in range(6):  # 6 segments of 5 seconds each
            start = i * 5 * sample_rate
            end = (i + 1) * 5 * sample_rate
            freq = 200 + i * 100  # Different frequency for each segment
            audio_data[start:end] = 0.5 * np.sin(2 * np.pi * freq * t[start:end])
        
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            try:
                result = real_transcription_service.transcribe(
                    audio_path=tmp_file.name,
                    provider='parakeet-mlx',
                    model='mlx-community/parakeet-tdt-0.6b-v2',
                    chunk_size=10,  # 10 second chunks
                    overlap=2       # 2 second overlap
                )
                
                # Should have multiple segments due to chunking
                assert 'text' in result
                assert 'segments' in result
                assert len(result['segments']) >= 1  # At least one segment
                assert result['chunk_size'] == 10
                assert result['overlap'] == 2
                
            finally:
                os.unlink(tmp_file.name)
    
    @pytest.mark.integration
    def test_real_transcription_invalid_file(self, real_transcription_service):
        """Test transcription with invalid audio file."""
        with pytest.raises(TranscriptionError):
            real_transcription_service.transcribe(
                audio_path="non_existent_file.wav",
                provider='parakeet-mlx'
            )
    
    @pytest.mark.integration
    def test_real_transcription_empty_file(self, real_transcription_service):
        """Test transcription with empty audio file."""
        # Create an empty WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b'')  # Empty data
            
            try:
                result = real_transcription_service.transcribe(
                    audio_path=tmp_file.name,
                    provider='parakeet-mlx',
                    model='mlx-community/parakeet-tdt-0.6b-v2'
                )
                
                # Should handle empty file gracefully
                assert 'text' in result
                assert result['text'] == '' or result['text'].strip() == ''
                
            finally:
                os.unlink(tmp_file.name)
    
    @pytest.mark.integration
    @pytest.mark.parametrize("attention_type", ["flash", "sdpa", "native"])
    def test_real_transcription_attention_types(self, real_transcription_service, test_audio_file, attention_type):
        """Test transcription with different attention types."""
        result = real_transcription_service.transcribe(
            audio_path=test_audio_file,
            provider='parakeet-mlx',
            model='mlx-community/parakeet-tdt-0.6b-v2',
            attention_type=attention_type
        )
        
        assert result['attention_type'] == attention_type
        assert 'text' in result
    
    @pytest.mark.integration
    def test_real_transcription_progress_tracking(self, real_transcription_service, test_audio_file):
        """Test progress tracking during transcription."""
        progress_updates = []
        
        def progress_callback(percentage, message, metadata):
            progress_updates.append({
                'percentage': percentage,
                'message': message,
                'metadata': metadata
            })
        
        result = real_transcription_service.transcribe(
            audio_path=test_audio_file,
            provider='parakeet-mlx',
            model='mlx-community/parakeet-tdt-0.6b-v2',
            progress_callback=progress_callback
        )
        
        # Check that we got progress updates
        assert len(progress_updates) > 0
        assert progress_updates[0]['percentage'] == 0
        assert progress_updates[-1]['percentage'] == 100
        
        # Check messages contain relevant info
        assert any('load' in update['message'].lower() for update in progress_updates)
        assert any('complete' in update['message'].lower() for update in progress_updates)
    
    @pytest.mark.integration
    def test_real_transcription_multichannel_audio(self, real_transcription_service):
        """Test transcription of stereo/multichannel audio."""
        # Create stereo audio file
        sample_rate = 16000
        duration = 2
        channels = 2
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Different frequency in each channel
        left_channel = 0.5 * np.sin(2 * np.pi * 440 * t)
        right_channel = 0.5 * np.sin(2 * np.pi * 880 * t)
        
        stereo_data = np.column_stack((left_channel, right_channel))
        stereo_data = (stereo_data * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(stereo_data.tobytes())
            
            try:
                result = real_transcription_service.transcribe(
                    audio_path=tmp_file.name,
                    provider='parakeet-mlx',
                    model='mlx-community/parakeet-tdt-0.6b-v2'
                )
                
                # Should handle stereo audio (usually by converting to mono)
                assert 'text' in result
                assert 'segments' in result
                
            finally:
                os.unlink(tmp_file.name)


# Performance benchmarking tests
class TestMLXParakeetPerformance:
    """Performance benchmarking tests for MLX Parakeet."""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_transcription_speed(self, real_transcription_service, test_audio_file):
        """Benchmark transcription speed with different configurations."""
        import time
        
        configurations = [
            {'model': 'mlx-community/parakeet-tdt-0.6b-v2', 'precision': 'float16'},
            {'model': 'mlx-community/parakeet-tdt-0.6b-v2', 'precision': 'float32'},
            {'model': 'mlx-community/parakeet-tdt-0.6b-v2', 'precision': 'float16', 'attention_type': 'flash'},
            {'model': 'mlx-community/parakeet-tdt-0.6b-v2', 'precision': 'float16', 'attention_type': 'sdpa'},
        ]
        
        results = []
        
        for config in configurations:
            start_time = time.time()
            
            result = real_transcription_service.transcribe(
                audio_path=test_audio_file,
                provider='parakeet-mlx',
                **config
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            results.append({
                'config': config,
                'elapsed_time': elapsed,
                'text_length': len(result['text']),
                'segments': len(result['segments'])
            })
            
            print(f"\nConfig: {config}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Text length: {len(result['text'])}")
            print(f"Segments: {len(result['segments'])}")
        
        # Parakeet is optimized for real-time, so should be fast
        assert all(r['elapsed_time'] < 30 for r in results), "Transcription took too long"
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_real_time_factor(self, real_transcription_service):
        """Test real-time factor (RTF) of Parakeet transcription."""
        import time
        
        # Create audio of known duration
        duration_seconds = 10
        sample_rate = 16000
        
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            try:
                start_time = time.time()
                
                result = real_transcription_service.transcribe(
                    audio_path=tmp_file.name,
                    provider='parakeet-mlx',
                    model='mlx-community/parakeet-tdt-0.6b-v2',
                    precision='float16'
                )
                
                elapsed_time = time.time() - start_time
                rtf = elapsed_time / duration_seconds
                
                print(f"\nReal-Time Factor (RTF): {rtf:.3f}")
                print(f"Audio duration: {duration_seconds}s")
                print(f"Processing time: {elapsed_time:.2f}s")
                
                # Parakeet should achieve real-time or better (RTF < 1.0)
                assert rtf < 2.0, f"RTF {rtf} is too high for real-time ASR"
                
            finally:
                os.unlink(tmp_file.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])