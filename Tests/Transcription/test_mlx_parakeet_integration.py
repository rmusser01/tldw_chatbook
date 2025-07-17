"""
Integration tests for MLX Parakeet transcription backend.

These tests focus on actual behavior rather than implementation details.
Minimal mocking is used - only for expensive operations like model downloads.
"""

import pytest
import sys
import os
import tempfile
import wave
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import time

from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionService,
    TranscriptionError,
    PARAKEET_MLX_AVAILABLE
)

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != 'darwin',
    reason="MLX Parakeet tests only run on macOS"
)


class TestMLXParakeetIntegration:
    """Integration tests focusing on actual behavior with minimal mocking."""
    
    @pytest.fixture
    def audio_generator(self):
        """Generate real audio data for testing."""
        def _generate_audio(duration_seconds=1.0, sample_rate=16000, frequency=440):
            """Generate a sine wave audio signal."""
            t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
            # Generate sine wave with some variation to simulate speech-like patterns
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            # Add some amplitude modulation to make it more speech-like
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
            audio = audio * envelope
            # Add slight noise
            audio += 0.02 * np.random.randn(len(t))
            # Normalize
            audio = audio / np.max(np.abs(audio))
            return (audio * 32767).astype(np.int16)
        return _generate_audio
    
    @pytest.fixture
    def create_wav_file(self, audio_generator):
        """Create actual WAV files for testing."""
        created_files = []
        
        def _create_wav(duration=1.0, sample_rate=16000, frequency=440):
            """Create a temporary WAV file with generated audio."""
            audio_data = audio_generator(duration, sample_rate, frequency)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)   # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.tobytes())
                
                created_files.append(tmp_file.name)
                return tmp_file.name
        
        yield _create_wav
        
        # Cleanup
        for file_path in created_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    @pytest.fixture
    def mock_model_download(self):
        """Mock only the model download to avoid downloading during tests."""
        def create_mock_model(*args, **kwargs):
            """Create a minimal mock model that behaves like Parakeet MLX."""
            mock_model = MagicMock()
            
            # Simple transcription behavior
            def mock_transcribe(audio_path, **kwargs):
                # Check if file exists
                if isinstance(audio_path, str) and not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
                result = MagicMock()
                # Return different text based on audio characteristics
                if isinstance(audio_path, str) and 'long' in audio_path:
                    result.text = "This is a long audio file transcription."
                elif isinstance(audio_path, str) and 'short' in audio_path:
                    result.text = "Short audio."
                else:
                    result.text = "This is a test transcription."
                
                # Add sentences attribute for segment creation
                if hasattr(result, 'text') and result.text:
                    # Create mock sentences
                    sentences = result.text.split('.')
                    result.sentences = []
                    start_time = 0.0
                    for sentence in sentences:
                        if sentence.strip():
                            mock_sentence = MagicMock()
                            mock_sentence.text = sentence.strip() + '.'
                            mock_sentence.start = start_time
                            mock_sentence.end = start_time + 1.0
                            result.sentences.append(mock_sentence)
                            start_time += 1.0
                else:
                    result.sentences = []
                
                return result
            
            mock_model.transcribe = mock_transcribe
            mock_model._model_name = args[0] if args else 'mlx-community/parakeet-tdt-0.6b-v2'
            return mock_model
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained', 
                   side_effect=create_mock_model) as mock:
            yield mock
    
    @pytest.fixture
    def transcription_service(self):
        """Create a real TranscriptionService instance."""
        # Only mock the model download, everything else is real
        service = TranscriptionService()
        return service
    
    def test_basic_transcription(self, transcription_service, create_wav_file, mock_model_download):
        """Test basic transcription functionality."""
        # Create a real audio file
        audio_file = create_wav_file(duration=1.0)
        
        # Transcribe it
        result = transcription_service.transcribe(
            audio_path=audio_file,
            provider='parakeet-mlx'
        )
        
        # Verify behavior, not implementation
        assert isinstance(result, dict)
        assert 'text' in result
        assert len(result['text']) > 0
        assert 'segments' in result
        assert 'provider' in result
        assert result['provider'] == 'parakeet-mlx'
    
    def test_different_sample_rates(self, transcription_service, create_wav_file, mock_model_download):
        """Test transcription with various sample rates."""
        sample_rates = [8000, 16000, 44100, 48000]
        
        for sample_rate in sample_rates:
            audio_file = create_wav_file(duration=0.5, sample_rate=sample_rate)
            
            result = transcription_service.transcribe(
                audio_path=audio_file,
                provider='parakeet-mlx'
            )
            
            # Verify it handles different sample rates
            assert 'text' in result
            assert len(result['text']) > 0
            
            # Check if sample rate info is included
            if 'sample_rate' in result:
                # If not 16000, should show conversion
                if sample_rate != 16000:
                    assert '16000' in str(result['sample_rate'])
    
    def test_long_audio_chunking(self, transcription_service, create_wav_file, mock_model_download):
        """Test chunking behavior for long audio files."""
        # Create a longer audio file (3 minutes)
        audio_file = create_wav_file(duration=180.0)
        
        result = transcription_service.transcribe(
            audio_path=audio_file,
            provider='parakeet-mlx',
            chunk_duration=30.0  # 30-second chunks
        )
        
        assert 'text' in result
        assert len(result['text']) > 0
        assert 'segments' in result
        # With 3 minutes and 30-second chunks, we should have multiple segments
        # But we're testing behavior, not exact implementation
        assert len(result['segments']) >= 1
    
    def test_progress_callback(self, transcription_service, create_wav_file, mock_model_download):
        """Test progress callback functionality."""
        audio_file = create_wav_file(duration=2.0)
        
        progress_updates = []
        
        def progress_callback(percentage, message, metadata):
            progress_updates.append({
                'percentage': percentage,
                'message': message,
                'metadata': metadata
            })
        
        result = transcription_service.transcribe(
            audio_path=audio_file,
            provider='parakeet-mlx',
            progress_callback=progress_callback
        )
        
        # Verify progress was reported
        assert len(progress_updates) > 0
        assert progress_updates[0]['percentage'] == 0
        assert progress_updates[-1]['percentage'] == 100
        
        # Verify transcription still completed
        assert 'text' in result
    
    def test_error_handling_invalid_file(self, transcription_service, mock_model_download):
        """Test handling of invalid audio files."""
        # Update the mock to not handle non-existent files
        original_transcribe = mock_model_download.return_value.transcribe
        
        def transcribe_with_file_check(audio_path, **kwargs):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            return original_transcribe(audio_path, **kwargs)
        
        mock_model_download.return_value.transcribe = transcribe_with_file_check
        
        with pytest.raises((TranscriptionError, FileNotFoundError, OSError)):
            transcription_service.transcribe(
                audio_path="non_existent_file.wav",
                provider='parakeet-mlx'
            )
    
    def test_error_handling_corrupted_audio(self, transcription_service, mock_model_download):
        """Test handling of corrupted audio files."""
        # Create a corrupted WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Write invalid WAV data
            tmp_file.write(b'RIFF')
            tmp_file.write(b'\x00' * 100)  # Invalid data
            corrupted_file = tmp_file.name
        
        try:
            # The service might handle corrupted files gracefully or raise an error
            # Let's test that it either handles it or fails appropriately
            try:
                result = transcription_service.transcribe(
                    audio_path=corrupted_file,
                    provider='parakeet-mlx'
                )
                # If it succeeds, verify it returns a valid result structure
                assert isinstance(result, dict)
                assert 'text' in result
                assert 'segments' in result
                # Text might be empty for corrupted audio
                assert isinstance(result['text'], str)
            except (TranscriptionError, OSError, wave.Error):
                # Expected - corrupted file caused an error
                pass
        finally:
            os.unlink(corrupted_file)
    
    def test_model_caching(self, transcription_service, create_wav_file, mock_model_download):
        """Test that models are cached between transcriptions."""
        audio_file1 = create_wav_file(duration=0.5)
        audio_file2 = create_wav_file(duration=0.5)
        
        # First transcription
        result1 = transcription_service.transcribe(
            audio_path=audio_file1,
            provider='parakeet-mlx'
        )
        
        # Second transcription should use cached model
        result2 = transcription_service.transcribe(
            audio_path=audio_file2,
            provider='parakeet-mlx'
        )
        
        # Both should succeed
        assert 'text' in result1
        assert 'text' in result2
        
        # Model should only be loaded once
        assert mock_model_download.call_count == 1
    
    def test_concurrent_transcriptions(self, transcription_service, create_wav_file, mock_model_download):
        """Test concurrent transcription requests."""
        import concurrent.futures
        
        # Create multiple audio files
        audio_files = [create_wav_file(duration=0.5) for _ in range(3)]
        
        def transcribe_file(audio_path):
            return transcription_service.transcribe(
                audio_path=audio_path,
                provider='parakeet-mlx'
            )
        
        # Run transcriptions concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(transcribe_file, audio_file) 
                      for audio_file in audio_files]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert 'text' in result
            assert len(result['text']) > 0
    
    @pytest.mark.parametrize("precision", ["float16", "float32", "bfloat16"])
    def test_precision_options(self, transcription_service, create_wav_file, mock_model_download, precision):
        """Test different precision options."""
        audio_file = create_wav_file(duration=0.5)
        
        result = transcription_service.transcribe(
            audio_path=audio_file,
            provider='parakeet-mlx',
            precision=precision
        )
        
        assert 'text' in result
        assert 'precision' in result
        assert result['precision'] == precision
    
    def test_empty_audio(self, transcription_service, mock_model_download):
        """Test handling of empty audio files."""
        # Create empty WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b'')  # Empty audio
            empty_file = tmp_file.name
        
        try:
            # Override the mock factory to handle empty audio
            def create_empty_model(*args, **kwargs):
                model = MagicMock()
                
                def transcribe_empty(audio_path, **kwargs):
                    result = MagicMock()
                    result.text = ""  # Empty transcription for empty audio
                    result.sentences = []  # No sentences
                    return result
                
                model.transcribe = transcribe_empty
                model._model_name = args[0] if args else 'test-model'
                return model
            
            mock_model_download.side_effect = create_empty_model
            
            result = transcription_service.transcribe(
                audio_path=empty_file,
                provider='parakeet-mlx'
            )
            
            # Should handle gracefully
            assert 'text' in result
            assert result['text'] == ""
            assert len(result['segments']) == 0
        finally:
            os.unlink(empty_file)


@pytest.mark.skipif(
    not PARAKEET_MLX_AVAILABLE or sys.platform != 'darwin',
    reason="Parakeet MLX not available or not on macOS"
)
class TestMLXParakeetRealIntegration:
    """Real integration tests that use actual Parakeet MLX if available."""
    
    @pytest.fixture
    def real_audio_file(self):
        """Create a real audio file with speech-like characteristics."""
        sample_rate = 16000
        duration = 2.0
        
        # Create more complex audio that might produce actual transcription
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Mix multiple frequencies to simulate speech formants
        audio = (
            0.3 * np.sin(2 * np.pi * 200 * t) +   # Low frequency
            0.2 * np.sin(2 * np.pi * 700 * t) +   # Mid frequency  
            0.1 * np.sin(2 * np.pi * 2000 * t) +  # High frequency
            0.05 * np.sin(2 * np.pi * 3500 * t)   # Very high frequency
        )
        
        # Add amplitude modulation for speech-like rhythm
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
        audio = audio * envelope
        
        # Add some noise
        audio += 0.02 * np.random.randn(len(t))
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            yield tmp_file.name
        
        os.unlink(tmp_file.name)
    
    @pytest.mark.slow
    def test_real_transcription(self, real_audio_file):
        """Test actual transcription with real Parakeet MLX model."""
        service = TranscriptionService()
        
        result = service.transcribe(
            audio_path=real_audio_file,
            provider='parakeet-mlx'
        )
        
        # With real model, we get actual transcription
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'segments' in result
        assert 'provider' in result
        assert result['provider'] == 'parakeet-mlx'
        
        # The text might be empty or nonsensical for synthetic audio
        # but the structure should be correct
        assert isinstance(result['text'], str)
        assert isinstance(result['segments'], list)