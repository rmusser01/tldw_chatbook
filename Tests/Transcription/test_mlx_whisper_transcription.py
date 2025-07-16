"""
Unit and integration tests for the MLX Whisper transcription backend.

This module tests the Lightning Whisper MLX implementation for macOS/Apple Silicon.
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

# Import the transcription service and related classes
from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionService,
    TranscriptionError,
    ConversionError,
    LIGHTNING_WHISPER_AVAILABLE
)


# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != 'darwin',
    reason="MLX Whisper tests only run on macOS"
)


class TestMLXWhisperUnit:
    """Unit tests for MLX Whisper transcription backend."""
    
    @pytest.fixture
    def mock_lightning_whisper_available(self):
        """Mock Lightning Whisper availability."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LIGHTNING_WHISPER_AVAILABLE', True):
            yield
    
    @pytest.fixture
    def mock_lightning_whisper_model(self):
        """Mock the LightningWhisperMLX class."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock:
            # Create a mock instance
            mock_instance = MagicMock()
            mock_instance.transcribe.return_value = {
                'text': 'This is a test transcription',
                'segments': [
                    {
                        'start': 0.0,
                        'end': 2.5,
                        'text': 'This is a test transcription'
                    }
                ],
                'language': 'en'
            }
            mock.return_value = mock_instance
            yield mock, mock_instance
    
    @pytest.fixture
    def transcription_service(self, mock_lightning_whisper_available):
        """Create a TranscriptionService instance with mocked config."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.get_cli_setting') as mock_get_setting:
            # Mock configuration values
            def get_setting_side_effect(key, default=None):
                settings = {
                    'transcription.provider': 'lightning-whisper-mlx',
                    'transcription.model': 'base',
                    'transcription.language': 'en',
                    'transcription.vad_filter': True,
                    'transcription.compute_type': 'int8',
                    'transcription.device': 'cpu',
                    'transcription.device_index': 0,
                    'transcription.num_workers': 1,
                    'transcription.download_root': None,
                    'transcription.local_files_only': False,
                    'transcription.lightning_batch_size': 12,
                    'transcription.lightning_quant': None,
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
    
    def test_lightning_whisper_initialization(self, transcription_service):
        """Test that Lightning Whisper configuration is properly initialized."""
        assert transcription_service._lightning_whisper_model is None  # Lazy loaded
        assert transcription_service._lightning_whisper_config['batch_size'] == 12
        assert transcription_service._lightning_whisper_config['quant'] is None
    
    def test_lightning_whisper_not_available(self, transcription_service):
        """Test error handling when Lightning Whisper is not available."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LIGHTNING_WHISPER_AVAILABLE', False):
            with pytest.raises(TranscriptionError) as exc_info:
                transcription_service._transcribe_with_lightning_whisper_mlx(
                    audio_path="dummy.wav"
                )
            assert "lightning-whisper-mlx is not installed" in str(exc_info.value)
    
    def test_model_loading(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test Lightning Whisper model loading."""
        mock_class, mock_instance = mock_lightning_whisper_model
        
        # First transcription should load the model
        result = transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base'
        )
        
        # Check model was instantiated with correct parameters
        mock_class.assert_called_once_with(
            model='base',
            batch_size=12,
            quant=None
        )
        
        # Check transcription was called
        mock_instance.transcribe.assert_called_once_with(sample_audio_file)
        
        # Check result format
        assert 'text' in result
        assert 'segments' in result
        assert 'language' in result
        assert 'provider' in result
        assert result['provider'] == 'lightning-whisper-mlx'
    
    def test_model_caching(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test that the model is cached between transcriptions."""
        mock_class, mock_instance = mock_lightning_whisper_model
        
        # First transcription
        transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base'
        )
        
        # Reset mock to check it's not called again
        mock_class.reset_mock()
        
        # Second transcription with same model
        transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base'
        )
        
        # Model should not be loaded again
        mock_class.assert_not_called()
    
    def test_model_reload_on_config_change(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test that model is reloaded when configuration changes."""
        mock_class, mock_instance = mock_lightning_whisper_model
        
        # First transcription with base model
        transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base'
        )
        
        # Reset mock
        mock_class.reset_mock()
        
        # Second transcription with different model
        transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='large-v3'
        )
        
        # Model should be reloaded
        mock_class.assert_called_once_with(
            model='large-v3',
            batch_size=12,
            quant=None
        )
    
    def test_quantization_options(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test different quantization options."""
        mock_class, mock_instance = mock_lightning_whisper_model
        
        # Test with 4-bit quantization
        transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base',
            quant='4bit'
        )
        
        mock_class.assert_called_with(
            model='base',
            batch_size=12,
            quant='4bit'
        )
        
        # Reset and test with 8-bit quantization
        mock_class.reset_mock()
        transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base',
            quant='8bit'
        )
        
        mock_class.assert_called_with(
            model='base',
            batch_size=12,
            quant='8bit'
        )
    
    def test_batch_size_configuration(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test custom batch size configuration."""
        mock_class, mock_instance = mock_lightning_whisper_model
        
        # Test with custom batch size
        transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base',
            batch_size=24
        )
        
        mock_class.assert_called_with(
            model='base',
            batch_size=24,
            quant=None
        )
    
    def test_progress_callback(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test progress callback functionality."""
        mock_class, mock_instance = mock_lightning_whisper_model
        mock_callback = Mock()
        
        result = transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base',
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
    
    def test_segment_formatting(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test that segments are properly formatted."""
        mock_class, mock_instance = mock_lightning_whisper_model
        
        # Mock response with multiple segments
        mock_instance.transcribe.return_value = {
            'text': 'First segment. Second segment.',
            'segments': [
                {'start': 0.0, 'end': 1.5, 'text': 'First segment.'},
                {'start': 1.5, 'end': 3.0, 'text': 'Second segment.'}
            ],
            'language': 'en'
        }
        
        result = transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base'
        )
        
        # Check segments are properly formatted
        assert len(result['segments']) == 2
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
    
    def test_empty_segments_handling(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test handling of response without segments."""
        mock_class, mock_instance = mock_lightning_whisper_model
        
        # Mock response without segments
        mock_instance.transcribe.return_value = {
            'text': 'Transcription without segments',
            'language': 'en'
        }
        
        result = transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base'
        )
        
        # Should create a single segment
        assert len(result['segments']) == 1
        assert result['segments'][0]['text'] == 'Transcription without segments'
        assert result['segments'][0]['start'] == 0.0
        assert result['segments'][0]['end'] == 0.0
    
    def test_translation_support(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test translation support."""
        mock_class, mock_instance = mock_lightning_whisper_model
        
        # Mock response with translation
        mock_instance.transcribe.return_value = {
            'text': 'This is English text',
            'translation': 'This is translated text',
            'language': 'es',
            'segments': []
        }
        
        result = transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base',
            source_lang='es',
            target_lang='en'
        )
        
        # Check translation fields
        assert 'translation' in result
        assert result['translation'] == 'This is translated text'
        assert result['source_language'] == 'es'
        assert result['target_language'] == 'en'
    
    def test_error_handling_model_load(self, transcription_service, sample_audio_file):
        """Test error handling during model loading."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_class:
            mock_class.side_effect = Exception("Model loading failed")
            
            with pytest.raises(TranscriptionError) as exc_info:
                transcription_service._transcribe_with_lightning_whisper_mlx(
                    audio_path=sample_audio_file,
                    model='base'
                )
            
            assert "Failed to load Lightning Whisper MLX model" in str(exc_info.value)
    
    def test_error_handling_transcription(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test error handling during transcription."""
        mock_class, mock_instance = mock_lightning_whisper_model
        mock_instance.transcribe.side_effect = Exception("Transcription failed")
        
        with pytest.raises(TranscriptionError) as exc_info:
            transcription_service._transcribe_with_lightning_whisper_mlx(
                audio_path=sample_audio_file,
                model='base'
            )
        
        assert "Lightning Whisper MLX transcription failed" in str(exc_info.value)
    
    def test_model_name_mapping(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test model name mapping for common variants."""
        mock_class, mock_instance = mock_lightning_whisper_model
        
        # Test mapping of 'large' to 'large-v3'
        transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='large'
        )
        
        mock_class.assert_called_with(
            model='large-v3',
            batch_size=12,
            quant=None
        )
    
    def test_result_metadata(self, transcription_service, mock_lightning_whisper_model, sample_audio_file):
        """Test that result includes all expected metadata."""
        mock_class, mock_instance = mock_lightning_whisper_model
        
        result = transcription_service._transcribe_with_lightning_whisper_mlx(
            audio_path=sample_audio_file,
            model='base',
            batch_size=24,
            quant='4bit'
        )
        
        # Check metadata fields
        assert result['provider'] == 'lightning-whisper-mlx'
        assert result['model'] == 'base'
        assert result['batch_size'] == 24
        assert result['quantization'] == '4bit'
        assert 'text' in result
        assert 'segments' in result
        assert 'language' in result


class TestMLXWhisperIntegration:
    """Integration tests for MLX Whisper transcription backend."""
    
    @pytest.fixture
    def real_transcription_service(self):
        """Create a real TranscriptionService instance."""
        if not LIGHTNING_WHISPER_AVAILABLE:
            pytest.skip("Lightning Whisper MLX not available")
        
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
        """Test actual transcription with Lightning Whisper MLX."""
        # This test will actually load the model and perform transcription
        result = real_transcription_service.transcribe(
            audio_file_path=test_audio_file,
            provider='lightning-whisper-mlx',
            model='tiny',  # Use tiny model for faster testing
            batch_size=4   # Smaller batch size for testing
        )
        
        # Basic checks - the actual transcription text will vary
        assert 'text' in result
        assert 'segments' in result
        assert 'language' in result
        assert result['provider'] == 'lightning-whisper-mlx'
        assert isinstance(result['text'], str)
        assert isinstance(result['segments'], list)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_transcription_with_quantization(self, real_transcription_service, test_audio_file):
        """Test transcription with quantization."""
        result = real_transcription_service.transcribe(
            audio_file_path=test_audio_file,
            provider='lightning-whisper-mlx',
            model='tiny',
            quant='4bit'
        )
        
        assert result['quantization'] == '4bit'
        assert 'text' in result
        assert 'segments' in result
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_transcription_different_models(self, real_transcription_service, test_audio_file):
        """Test transcription with different model sizes."""
        models_to_test = ['tiny', 'base']  # Test smaller models for speed
        
        for model in models_to_test:
            result = real_transcription_service.transcribe(
                audio_file_path=test_audio_file,
                provider='lightning-whisper-mlx',
                model=model
            )
            
            assert result['model'] == model
            assert 'text' in result
            assert 'segments' in result
    
    @pytest.mark.integration
    def test_real_transcription_invalid_file(self, real_transcription_service):
        """Test transcription with invalid audio file."""
        with pytest.raises(TranscriptionError):
            real_transcription_service.transcribe(
                audio_file_path="non_existent_file.wav",
                provider='lightning-whisper-mlx'
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
                    audio_file_path=tmp_file.name,
                    provider='lightning-whisper-mlx',
                    model='tiny'
                )
                
                # Should handle empty file gracefully
                assert 'text' in result
                assert result['text'] == '' or result['text'].strip() == ''
                
            finally:
                os.unlink(tmp_file.name)
    
    @pytest.mark.integration
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_real_transcription_batch_sizes(self, real_transcription_service, test_audio_file, batch_size):
        """Test transcription with different batch sizes."""
        result = real_transcription_service.transcribe(
            audio_file_path=test_audio_file,
            provider='lightning-whisper-mlx',
            model='tiny',
            batch_size=batch_size
        )
        
        assert result['batch_size'] == batch_size
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
            audio_file_path=test_audio_file,
            provider='lightning-whisper-mlx',
            model='tiny',
            progress_callback=progress_callback
        )
        
        # Check that we got progress updates
        assert len(progress_updates) > 0
        assert progress_updates[0]['percentage'] == 0
        assert progress_updates[-1]['percentage'] == 100
        
        # Check messages contain relevant info
        assert any('start' in update['message'].lower() for update in progress_updates)
        assert any('complete' in update['message'].lower() for update in progress_updates)


# Performance benchmarking tests
class TestMLXWhisperPerformance:
    """Performance benchmarking tests for MLX Whisper."""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_transcription_speed(self, real_transcription_service, test_audio_file):
        """Benchmark transcription speed with different configurations."""
        import time
        
        configurations = [
            {'model': 'tiny', 'batch_size': 12, 'quant': None},
            {'model': 'tiny', 'batch_size': 12, 'quant': '4bit'},
            {'model': 'base', 'batch_size': 12, 'quant': None},
            {'model': 'base', 'batch_size': 12, 'quant': '4bit'},
        ]
        
        results = []
        
        for config in configurations:
            start_time = time.time()
            
            result = real_transcription_service.transcribe(
                audio_file_path=test_audio_file,
                provider='lightning-whisper-mlx',
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
        
        # The fastest configuration should be tiny model with 4bit quantization
        assert all(r['elapsed_time'] < 60 for r in results), "Transcription took too long"


# Test fixtures for mocking file system operations
@pytest.fixture
def mock_file_operations():
    """Mock file system operations for unit tests."""
    with patch('os.path.exists') as mock_exists, \
         patch('os.path.isfile') as mock_isfile:
        mock_exists.return_value = True
        mock_isfile.return_value = True
        yield mock_exists, mock_isfile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])