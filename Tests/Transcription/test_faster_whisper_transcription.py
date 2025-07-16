"""
Unit and integration tests for the faster-whisper transcription backend.

This module tests the faster-whisper implementation which is a CPU/GPU optimized
version of OpenAI's Whisper model using CTranslate2.
"""

import pytest
import os
import tempfile
import wave
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, PropertyMock
from typing import Dict, Any, Optional, Generator, NamedTuple
import time

# Import the transcription service and related classes
from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionService,
    TranscriptionError,
    ConversionError,
    FASTER_WHISPER_AVAILABLE,
    protect_file_descriptors
)


class MockSegment(NamedTuple):
    """Mock segment matching faster-whisper's segment structure."""
    start: float
    end: float
    text: str
    tokens: list = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class MockTranscriptionInfo(NamedTuple):
    """Mock info object matching faster-whisper's info structure."""
    language: str
    language_probability: float
    duration: float
    all_language_probs: Optional[Dict[str, float]] = None
    transcription_options: Optional[Dict[str, Any]] = None
    vad_options: Optional[Dict[str, Any]] = None


class TestFasterWhisperUnit:
    """Unit tests for faster-whisper transcription backend."""
    
    @pytest.fixture
    def mock_faster_whisper_available(self):
        """Mock faster-whisper availability."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.FASTER_WHISPER_AVAILABLE', True):
            yield
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock the WhisperModel class."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock:
            # Create a mock instance
            mock_instance = MagicMock()
            
            # Create mock segments
            segments = [
                MockSegment(0.0, 2.5, "This is the first segment."),
                MockSegment(2.5, 5.0, "This is the second segment."),
                MockSegment(5.0, 7.5, "This is the third segment.")
            ]
            
            # Create mock info
            info = MockTranscriptionInfo(
                language="en",
                language_probability=0.99,
                duration=7.5
            )
            
            # Mock transcribe method to return generator and info
            def mock_transcribe(*args, **kwargs):
                return iter(segments), info
            
            mock_instance.transcribe.side_effect = mock_transcribe
            mock.return_value = mock_instance
            yield mock, mock_instance
    
    @pytest.fixture
    def transcription_service(self, mock_faster_whisper_available):
        """Create a TranscriptionService instance with mocked config."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.get_cli_setting') as mock_get_setting:
            # Mock configuration values
            def get_setting_side_effect(key, default=None):
                settings = {
                    'transcription.provider': 'faster-whisper',
                    'transcription.model': 'base',
                    'transcription.language': 'en',
                    'transcription.vad_filter': True,
                    'transcription.compute_type': 'int8',
                    'transcription.device': 'cpu',
                    'transcription.device_index': 0,
                    'transcription.num_workers': 1,
                    'transcription.download_root': None,
                    'transcription.local_files_only': False,
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
    
    def test_faster_whisper_initialization(self, transcription_service):
        """Test that faster-whisper configuration is properly initialized."""
        assert transcription_service.config['provider'] == 'faster-whisper'
        assert transcription_service.config['model'] == 'base'
        assert transcription_service.config['device'] == 'cpu'
        assert transcription_service.config['compute_type'] == 'int8'
        assert transcription_service._model_cache == {}  # Model cache starts empty
    
    def test_faster_whisper_not_available(self, transcription_service):
        """Test error handling when faster-whisper is not available."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.FASTER_WHISPER_AVAILABLE', False):
            with pytest.raises(TranscriptionError) as exc_info:
                transcription_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model="base",
                    language="en",
                    vad_filter=True
                )
            assert "faster-whisper is not installed" in str(exc_info.value)
    
    def test_model_loading_and_caching(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test faster-whisper model loading and caching."""
        mock_class, mock_instance = mock_whisper_model
        
        # First transcription should load the model
        result = transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='en',
            vad_filter=True
        )
        
        # Check model was instantiated with correct parameters
        mock_class.assert_called_once_with(
            'base',
            device='cpu',
            compute_type='int8',
            download_root=None,
            local_files_only=False
        )
        
        # Check transcription was called
        assert mock_instance.transcribe.called
        call_args = mock_instance.transcribe.call_args
        assert call_args[0][0] == sample_audio_file
        assert call_args[1]['language'] == 'en'
        assert call_args[1]['vad_filter'] is True
        assert call_args[1]['task'] == 'transcribe'
        
        # Check result format
        assert 'text' in result
        assert 'segments' in result
        assert 'language' in result
        assert 'provider' in result
        assert result['provider'] == 'faster-whisper'
        
        # Verify model is cached
        cache_key = ('base', 'cpu', 'int8')
        assert cache_key in transcription_service._model_cache
    
    def test_model_cache_reuse(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test that the model is reused from cache."""
        mock_class, mock_instance = mock_whisper_model
        
        # First transcription
        transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='en',
            vad_filter=True
        )
        
        # Reset mock to check it's not called again
        mock_class.reset_mock()
        
        # Second transcription with same model
        transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='en',
            vad_filter=True
        )
        
        # Model should not be loaded again
        mock_class.assert_not_called()
        # But transcribe should be called again
        assert mock_instance.transcribe.call_count == 2
    
    def test_different_model_configurations(self, transcription_service, mock_whisper_model):
        """Test that different configurations create different cache entries."""
        mock_class, mock_instance = mock_whisper_model
        
        # Test different device/compute_type combinations
        configurations = [
            ('base', 'cpu', 'int8'),
            ('base', 'cuda', 'int8'),
            ('base', 'cpu', 'float16'),
            ('large', 'cpu', 'int8'),
        ]
        
        for model, device, compute_type in configurations:
            # Update config
            transcription_service.config['device'] = device
            transcription_service.config['compute_type'] = compute_type
            
            # Clear specific cache entry if it exists
            cache_key = (model, device, compute_type)
            if cache_key in transcription_service._model_cache:
                del transcription_service._model_cache[cache_key]
            
            mock_class.reset_mock()
            
            # Transcribe
            with patch('os.path.exists', return_value=True):
                transcription_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model=model,
                    language='en',
                    vad_filter=True
                )
            
            # Should create new model for this configuration
            mock_class.assert_called_once()
            assert cache_key in transcription_service._model_cache
    
    def test_language_detection(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test automatic language detection."""
        mock_class, mock_instance = mock_whisper_model
        
        # Test with 'auto' language
        result = transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='auto',
            vad_filter=True
        )
        
        # Check that language was not specified in transcribe call
        call_args = mock_instance.transcribe.call_args
        assert call_args[1]['language'] is None  # None triggers auto-detection
        
        # Result should include detected language
        assert result['language'] == 'en'
        assert result['language_probability'] == 0.99
    
    def test_translation_task(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test translation functionality."""
        mock_class, mock_instance = mock_whisper_model
        
        # Test translation from Spanish to English
        result = transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='es',
            vad_filter=True,
            source_lang='es',
            target_lang='en'
        )
        
        # Check that task was set to 'translate'
        call_args = mock_instance.transcribe.call_args
        assert call_args[1]['task'] == 'translate'
        assert call_args[1]['language'] == 'es'
        
        # Result should include translation info
        assert result['task'] == 'translation'
        assert result['source_language'] == 'es'
        assert result['target_language'] == 'en'
        assert 'translation' in result
    
    def test_vad_filter_options(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test VAD (Voice Activity Detection) filter options."""
        mock_class, mock_instance = mock_whisper_model
        
        # Test with VAD enabled
        result_vad = transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='en',
            vad_filter=True
        )
        
        call_args = mock_instance.transcribe.call_args
        assert call_args[1]['vad_filter'] is True
        
        # Reset and test with VAD disabled
        mock_instance.reset_mock()
        
        result_no_vad = transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='en',
            vad_filter=False
        )
        
        call_args = mock_instance.transcribe.call_args
        assert call_args[1]['vad_filter'] is False
    
    def test_progress_callback(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test progress callback functionality."""
        mock_class, mock_instance = mock_whisper_model
        mock_callback = Mock()
        
        result = transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='en',
            vad_filter=True,
            progress_callback=mock_callback
        )
        
        # Check progress callbacks were made
        assert mock_callback.call_count >= 4  # Start, language detection, segments, complete
        
        # Check first call (start)
        first_call = mock_callback.call_args_list[0]
        assert first_call[0][0] == 0  # Progress percentage
        assert "Starting transcription" in first_call[0][1]
        
        # Check language detection call
        lang_calls = [call for call in mock_callback.call_args_list if "Language detected" in call[0][1]]
        assert len(lang_calls) > 0
        assert lang_calls[0][0][0] == 5  # 5% progress
        
        # Check last call (complete)
        last_call = mock_callback.call_args_list[-1]
        assert last_call[0][0] == 100  # Progress percentage
        assert "complete" in last_call[0][1].lower()
        assert last_call[0][2]['total_segments'] == 3
    
    def test_segment_processing(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test segment processing and formatting."""
        mock_class, mock_instance = mock_whisper_model
        
        result = transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='en',
            vad_filter=True
        )
        
        # Check segments are properly formatted
        assert len(result['segments']) == 3
        
        for i, segment in enumerate(result['segments']):
            # Check required fields
            assert 'start' in segment
            assert 'end' in segment
            assert 'text' in segment
            # Check legacy format fields
            assert 'Time_Start' in segment
            assert 'Time_End' in segment
            assert 'Text' in segment
            # Verify values match
            assert segment['start'] == segment['Time_Start']
            assert segment['end'] == segment['Time_End']
            assert segment['text'] == segment['Text']
            # Check text is stripped
            assert segment['text'] == segment['text'].strip()
        
        # Check full text is properly joined
        expected_text = "This is the first segment. This is the second segment. This is the third segment."
        assert result['text'] == expected_text
    
    def test_error_handling_model_load(self, transcription_service, sample_audio_file):
        """Test error handling during model loading."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_class:
            # Test file descriptor error
            mock_class.side_effect = Exception("bad value(s) in fds_to_keep")
            
            with pytest.raises(TranscriptionError) as exc_info:
                transcription_service._transcribe_with_faster_whisper(
                    audio_path=sample_audio_file,
                    model='base',
                    language='en',
                    vad_filter=True
                )
            
            error_msg = str(exc_info.value)
            assert "Failed to load model" in error_msg
            assert "file descriptors" in error_msg
            assert "OBJC_DISABLE_INITIALIZE_FORK_SAFETY" in error_msg
    
    def test_error_handling_transcription(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test error handling during transcription."""
        mock_class, mock_instance = mock_whisper_model
        
        # Make transcribe raise an error
        mock_instance.transcribe.side_effect = Exception("Audio processing failed")
        
        with pytest.raises(TranscriptionError) as exc_info:
            transcription_service._transcribe_with_faster_whisper(
                audio_path=sample_audio_file,
                model='base',
                language='en',
                vad_filter=True
            )
        
        assert "Transcription failed" in str(exc_info.value)
        assert "Audio processing failed" in str(exc_info.value)
    
    def test_beam_size_configuration(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test beam search configuration."""
        mock_class, mock_instance = mock_whisper_model
        
        result = transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='en',
            vad_filter=True
        )
        
        # Check default beam_size and best_of values
        call_args = mock_instance.transcribe.call_args
        assert call_args[1]['beam_size'] == 5
        assert call_args[1]['best_of'] == 5
    
    def test_empty_segments_handling(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test handling of empty segments."""
        mock_class, mock_instance = mock_whisper_model
        
        # Mock empty transcription
        empty_segments = []
        empty_info = MockTranscriptionInfo(
            language="en",
            language_probability=0.99,
            duration=0.0
        )
        
        mock_instance.transcribe.side_effect = lambda *args, **kwargs: (iter(empty_segments), empty_info)
        
        result = transcription_service._transcribe_with_faster_whisper(
            audio_path=sample_audio_file,
            model='base',
            language='en',
            vad_filter=True
        )
        
        # Should handle empty transcription gracefully
        assert result['text'] == ""
        assert len(result['segments']) == 0
        assert result['duration'] == 0.0
    
    def test_speed_calculation(self, transcription_service, mock_whisper_model, sample_audio_file):
        """Test real-time speed calculation."""
        mock_class, mock_instance = mock_whisper_model
        
        # Mock segments with known timing
        with patch('time.time') as mock_time:
            # Set up time sequence: start, after model load, after transcribe start, end
            mock_time.side_effect = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5]
            
            result = transcription_service._transcribe_with_faster_whisper(
                audio_path=sample_audio_file,
                model='base',
                language='en',
                vad_filter=True
            )
            
            # Duration is 7.5 seconds, processing took 1.5 seconds
            # Speed should be 7.5 / 1.5 = 5x realtime
            # (This is logged, not returned in result)
            assert result['duration'] == 7.5


class TestFasterWhisperIntegration:
    """Integration tests for faster-whisper transcription backend."""
    
    @pytest.fixture
    def real_transcription_service(self):
        """Create a real TranscriptionService instance."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")
        
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
        """Test actual transcription with faster-whisper."""
        # This test will actually load the model and perform transcription
        result = real_transcription_service.transcribe(
            audio_file_path=test_audio_file,
            provider='faster-whisper',
            model='tiny',  # Use tiny model for faster testing
            language='en',
            vad_filter=True
        )
        
        # Basic checks - the actual transcription text will vary
        assert 'text' in result
        assert 'segments' in result
        assert 'language' in result
        assert result['provider'] == 'faster-whisper'
        assert isinstance(result['text'], str)
        assert isinstance(result['segments'], list)
        assert result['language_probability'] > 0.0
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_transcription_auto_language(self, real_transcription_service, test_audio_file):
        """Test automatic language detection."""
        result = real_transcription_service.transcribe(
            audio_file_path=test_audio_file,
            provider='faster-whisper',
            model='tiny',
            language='auto',
            vad_filter=True
        )
        
        # Should detect language
        assert 'language' in result
        assert result['language'] is not None
        assert 'language_probability' in result
        assert result['language_probability'] > 0.0
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.parametrize("model", ["tiny", "base"])
    def test_real_transcription_different_models(self, real_transcription_service, test_audio_file, model):
        """Test transcription with different model sizes."""
        result = real_transcription_service.transcribe(
            audio_file_path=test_audio_file,
            provider='faster-whisper',
            model=model,
            language='en',
            vad_filter=True
        )
        
        assert result['model'] == model
        assert 'text' in result
        assert 'segments' in result
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.parametrize("compute_type", ["int8", "float16", "float32"])
    def test_real_transcription_compute_types(self, real_transcription_service, test_audio_file, compute_type):
        """Test transcription with different compute types."""
        # Some compute types may not be available on all hardware
        try:
            result = real_transcription_service.transcribe(
                audio_file_path=test_audio_file,
                provider='faster-whisper',
                model='tiny',
                language='en',
                vad_filter=True,
                compute_type=compute_type
            )
            
            assert 'text' in result
            assert 'segments' in result
        except TranscriptionError as e:
            if "compute type" in str(e).lower():
                pytest.skip(f"Compute type {compute_type} not supported on this hardware")
            else:
                raise
    
    @pytest.mark.integration
    def test_real_transcription_invalid_file(self, real_transcription_service):
        """Test transcription with invalid audio file."""
        with pytest.raises(TranscriptionError):
            real_transcription_service.transcribe(
                audio_file_path="non_existent_file.wav",
                provider='faster-whisper'
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
                    provider='faster-whisper',
                    model='tiny'
                )
                
                # Should handle empty file gracefully
                assert 'text' in result
                assert result['text'] == '' or result['text'].strip() == ''
                
            finally:
                os.unlink(tmp_file.name)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_transcription_with_vad(self, real_transcription_service, test_audio_file):
        """Test transcription with and without VAD filter."""
        # With VAD
        result_vad = real_transcription_service.transcribe(
            audio_file_path=test_audio_file,
            provider='faster-whisper',
            model='tiny',
            language='en',
            vad_filter=True
        )
        
        # Without VAD
        result_no_vad = real_transcription_service.transcribe(
            audio_file_path=test_audio_file,
            provider='faster-whisper',
            model='tiny',
            language='en',
            vad_filter=False
        )
        
        # Both should succeed
        assert 'text' in result_vad
        assert 'text' in result_no_vad
        
        # VAD might filter out some segments
        # Can't guarantee exact behavior without knowing audio content
    
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
            provider='faster-whisper',
            model='tiny',
            language='en',
            vad_filter=True,
            progress_callback=progress_callback
        )
        
        # Check that we got progress updates
        assert len(progress_updates) > 0
        assert progress_updates[0]['percentage'] == 0
        assert progress_updates[-1]['percentage'] == 100
        
        # Check for language detection update
        lang_updates = [u for u in progress_updates if 'language' in u.get('metadata', {})]
        assert len(lang_updates) > 0
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_transcription_translation(self, real_transcription_service):
        """Test translation functionality (if supported by model)."""
        # Create a simple audio file
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            try:
                # Try translation from French to English
                result = real_transcription_service.transcribe(
                    audio_file_path=tmp_file.name,
                    provider='faster-whisper',
                    model='tiny',
                    source_lang='fr',
                    target_lang='en'
                )
                
                # Check translation fields
                assert result['task'] == 'translation'
                assert result['source_language'] == 'fr'
                assert result['target_language'] == 'en'
                assert 'translation' in result
                
            finally:
                os.unlink(tmp_file.name)


class TestFasterWhisperPerformance:
    """Performance benchmarking tests for faster-whisper."""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_transcription_speed(self, real_transcription_service, test_audio_file):
        """Benchmark transcription speed with different configurations."""
        import time
        
        configurations = [
            {'model': 'tiny', 'compute_type': 'int8'},
            {'model': 'tiny', 'compute_type': 'float16'},
            {'model': 'base', 'compute_type': 'int8'},
        ]
        
        results = []
        
        for config in configurations:
            try:
                start_time = time.time()
                
                result = real_transcription_service.transcribe(
                    audio_file_path=test_audio_file,
                    provider='faster-whisper',
                    vad_filter=True,
                    **config
                )
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                results.append({
                    'config': config,
                    'elapsed_time': elapsed,
                    'text_length': len(result['text']),
                    'segments': len(result['segments']),
                    'duration': result.get('duration', 0)
                })
                
                print(f"\nConfig: {config}")
                print(f"Time: {elapsed:.2f}s")
                print(f"Text length: {len(result['text'])}")
                print(f"Segments: {len(result['segments'])}")
                if result.get('duration'):
                    rtf = elapsed / result['duration']
                    print(f"RTF: {rtf:.2f}")
                
            except Exception as e:
                print(f"\nConfig {config} failed: {e}")
                continue
        
        # All should complete reasonably quickly
        assert len(results) > 0
        assert all(r['elapsed_time'] < 60 for r in results), "Transcription took too long"
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_model_loading_time(self, real_transcription_service):
        """Test model loading time for different models."""
        import time
        
        models = ['tiny', 'base']
        
        for model in models:
            # Clear cache to force reload
            real_transcription_service._model_cache.clear()
            
            start_time = time.time()
            
            # Create simple audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                with wave.open(tmp.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
                
                try:
                    result = real_transcription_service.transcribe(
                        audio_file_path=tmp.name,
                        provider='faster-whisper',
                        model=model
                    )
                    
                    load_time = time.time() - start_time
                    print(f"\nModel {model} load time: {load_time:.2f}s")
                    
                    # Model loading should complete in reasonable time
                    assert load_time < 120, f"Model {model} took too long to load"
                    
                finally:
                    os.unlink(tmp.name)


class TestFasterWhisperFileDescriptorProtection:
    """Test file descriptor protection mechanism."""
    
    def test_protect_file_descriptors_context(self):
        """Test the protect_file_descriptors context manager."""
        import sys
        
        # Save original
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Create mock file objects
        mock_stdout = MagicMock()
        mock_stdout.fileno.side_effect = ValueError("I/O operation on closed file")
        
        sys.stdout = mock_stdout
        sys.stderr = mock_stdout
        
        try:
            with protect_file_descriptors():
                # Should have valid file descriptors inside context
                assert sys.stdout != mock_stdout
                assert sys.stderr != mock_stdout
                
                # Should be able to get file descriptors
                stdout_fd = sys.stdout.fileno()
                stderr_fd = sys.stderr.fileno()
                
                assert isinstance(stdout_fd, int)
                assert isinstance(stderr_fd, int)
            
            # Should be restored after context
            assert sys.stdout == mock_stdout
            assert sys.stderr == mock_stdout
            
        finally:
            # Restore original
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    def test_protect_file_descriptors_with_exception(self):
        """Test file descriptor protection handles exceptions properly."""
        import sys
        
        original_stdout = sys.stdout
        
        mock_stdout = MagicMock()
        mock_stdout.fileno.side_effect = ValueError("Closed file")
        
        sys.stdout = mock_stdout
        
        try:
            with pytest.raises(RuntimeError):
                with protect_file_descriptors():
                    # Should have valid stdout
                    assert sys.stdout != mock_stdout
                    # Raise exception
                    raise RuntimeError("Test exception")
            
            # Should still restore stdout
            assert sys.stdout == mock_stdout
            
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])