"""
Edge case tests for MLX Parakeet transcription backend.

This module tests edge cases, error conditions, and special scenarios
for the Parakeet MLX implementation.
"""

import pytest
import sys
import os
import tempfile
import wave
import struct
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import concurrent.futures
import threading
import time
import gc

from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionService,
    TranscriptionError,
    ConversionError,
    protect_file_descriptors
)


pytestmark = pytest.mark.skipif(
    sys.platform != 'darwin',
    reason="MLX Parakeet tests only run on macOS"
)


class TestMLXParakeetEdgeCases:
    """Edge case tests for MLX Parakeet transcription."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mocked transcription service."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.PARAKEET_MLX_AVAILABLE', True), \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.SOUNDFILE_AVAILABLE', True), \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.get_cli_setting') as mock_settings:
            
            mock_settings.return_value = None  # Use defaults
            service = TranscriptionService()
            return service
    
    @pytest.fixture
    def corrupted_audio_file(self):
        """Create a corrupted audio file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Write invalid WAV header
            tmp_file.write(b'RIFF')
            tmp_file.write(struct.pack('<I', 100))  # File size
            tmp_file.write(b'WAVE')
            tmp_file.write(b'fmt ')
            tmp_file.write(struct.pack('<I', 16))   # Subchunk size
            tmp_file.write(struct.pack('<H', 1))    # Audio format (PCM)
            tmp_file.write(struct.pack('<H', 1))    # Num channels
            tmp_file.write(struct.pack('<I', 44100)) # Sample rate
            # Intentionally truncate the file here
            
            yield tmp_file.name
            
        os.unlink(tmp_file.name)
    
    @pytest.fixture
    def high_sample_rate_audio(self):
        """Create audio file with very high sample rate."""
        sample_rate = 192000  # 192 kHz
        duration = 1
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            yield tmp_file.name
            
        os.unlink(tmp_file.name)
    
    def test_concurrent_model_loading(self, mock_service):
        """Test concurrent model loading attempts."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            # Simulate slow model loading
            def slow_model_load(*args, **kwargs):
                time.sleep(0.5)
                model = MagicMock()
                model.transcribe_audio.return_value = 'Concurrent test'
                return model
            
            mock_pretrained.side_effect = slow_model_load
            
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                mock_sf.read.return_value = (np.zeros(16000), 16000)
                
                # Try to load model concurrently
                results = []
                errors = []
                
                def load_and_transcribe():
                    try:
                        result = mock_service._transcribe_with_parakeet_mlx(
                            audio_path="dummy.wav",
                            model='mlx-community/parakeet-tdt-0.6b-v2'
                        )
                        results.append(result)
                    except Exception as e:
                        errors.append(e)
                
                threads = [threading.Thread(target=load_and_transcribe) for _ in range(3)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                
                # All should succeed
                assert len(results) == 3
                assert len(errors) == 0
                # Model should be loaded multiple times due to race condition
                # (In a real implementation, this should be protected by a lock)
                assert mock_pretrained.call_count >= 1
    
    def test_audio_resampling_edge_cases(self, mock_service):
        """Test edge cases in audio resampling."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            mock_model.transcribe_audio.return_value = 'Resampled audio'
            mock_pretrained.return_value = mock_model
            
            # Test various sample rates
            test_cases = [
                (8000, "Low sample rate"),     # Below target
                (16000, "Target sample rate"),  # Exact match
                (22050, "CD quality half"),     # Common rate
                (44100, "CD quality"),          # Common rate
                (48000, "DVD quality"),         # Common rate
                (96000, "High quality"),        # High rate
                (192000, "Ultra high quality"), # Very high rate
            ]
            
            for sample_rate, description in test_cases:
                with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                    audio_data = np.zeros(sample_rate, dtype=np.float32)  # 1 second
                    mock_sf.read.return_value = (audio_data, sample_rate)
                    
                    result = mock_service._transcribe_with_parakeet_mlx(
                        audio_path=f"audio_{sample_rate}.wav",
                        model='mlx-community/parakeet-tdt-0.6b-v2'
                    )
                    
                    assert 'text' in result
                    if sample_rate != 16000:
                        assert result['sample_rate'] == f'{sample_rate} -> 16000'
                    else:
                        assert result['sample_rate'] == '16000'
    
    def test_extreme_audio_lengths(self, mock_service):
        """Test handling of extremely short and long audio."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            mock_pretrained.return_value = mock_model
            
            # Test very short audio (10ms)
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                short_audio = np.zeros(160, dtype=np.float32)  # 10ms at 16kHz
                mock_sf.read.return_value = (short_audio, 16000)
                
                mock_model.transcribe_audio.return_value = ''
                
                result = mock_service._transcribe_with_parakeet_mlx(
                    audio_path="short.wav",
                    model='mlx-community/parakeet-tdt-0.6b-v2'
                )
                
                assert result['text'] == ''
                assert len(result['segments']) == 1
            
            # Test very long audio (1 hour)
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                # Don't actually create 1 hour of data, just simulate it
                long_audio_samples = 16000 * 3600  # 1 hour
                mock_sf.read.return_value = (np.zeros(100), 16000)  # Return small array
                mock_sf.read.return_value[0].__len__ = lambda: long_audio_samples  # Mock length
                
                # Mock multiple chunk transcriptions
                chunk_count = (3600 + 29) // 30  # 30-second chunks
                mock_model.transcribe_audio.side_effect = [f'Chunk {i}' for i in range(chunk_count)]
                
                result = mock_service._transcribe_with_parakeet_mlx(
                    audio_path="long.wav",
                    model='mlx-community/parakeet-tdt-0.6b-v2'
                )
                
                assert 'text' in result
                assert len(result['segments']) > 1  # Should have multiple segments
    
    def test_memory_constrained_transcription(self, mock_service):
        """Test transcription under memory constraints."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            
            # First call succeeds, subsequent calls fail with memory error
            call_count = 0
            def transcribe_with_memory_pressure(audio):
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    raise MemoryError("Insufficient memory for transcription")
                return f"Transcription {call_count}"
            
            mock_model.transcribe_audio.side_effect = transcribe_with_memory_pressure
            mock_pretrained.return_value = mock_model
            
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                # Large audio that requires chunking
                mock_sf.read.return_value = (np.zeros(16000 * 60), 16000)  # 60 seconds
                
                with pytest.raises(TranscriptionError) as exc_info:
                    mock_service._transcribe_with_parakeet_mlx(
                        audio_path="large.wav",
                        model='mlx-community/parakeet-tdt-0.6b-v2',
                        chunk_size=20  # 20-second chunks
                    )
                
                assert "transcription failed" in str(exc_info.value).lower()
    
    def test_invalid_precision_values(self, mock_service):
        """Test handling of invalid precision values."""
        test_cases = [
            ('float64', False),    # Valid but unusual
            ('int8', True),        # Invalid for Parakeet
            ('bfloat16', False),   # Valid
            ('float16', False),    # Valid
            ('float32', False),    # Valid
            ('invalid', True),     # Invalid
            ('', True),            # Empty string
            (None, False),         # None should use default
        ]
        
        for precision, should_fail in test_cases:
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
                if should_fail:
                    mock_pretrained.side_effect = ValueError(f"Invalid dtype: {precision}")
                else:
                    mock_model = MagicMock()
                    mock_model.transcribe_audio.return_value = f'Precision: {precision}'
                    mock_pretrained.return_value = mock_model
                
                with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                    mock_sf.read.return_value = (np.zeros(16000), 16000)
                    
                    if should_fail:
                        with pytest.raises(TranscriptionError):
                            mock_service._transcribe_with_parakeet_mlx(
                                audio_path="test.wav",
                                model='mlx-community/parakeet-tdt-0.6b-v2',
                                precision=precision
                            )
                    else:
                        result = mock_service._transcribe_with_parakeet_mlx(
                            audio_path="test.wav",
                            model='mlx-community/parakeet-tdt-0.6b-v2',
                            precision=precision
                        )
                        assert 'text' in result
    
    def test_attention_type_compatibility(self, mock_service):
        """Test attention type compatibility with different models."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            # Some attention types might not be supported by all models
            def create_model_with_attention_check(attention_type):
                if attention_type == 'flash' and 'old-model' in mock_service._parakeet_mlx_config.get('model', ''):
                    raise ValueError(f"Attention type '{attention_type}' not supported")
                
                model = MagicMock()
                model.transcribe_audio.return_value = f'Attention: {attention_type}'
                return model
            
            mock_pretrained.side_effect = create_model_with_attention_check
            
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                mock_sf.read.return_value = (np.zeros(16000), 16000)
                
                # Test with supported attention type
                result = mock_service._transcribe_with_parakeet_mlx(
                    audio_path="test.wav",
                    model='mlx-community/parakeet-tdt-0.6b-v2',
                    attention_type='sdpa'
                )
                assert result['attention_type'] == 'sdpa'
    
    def test_audio_channel_edge_cases(self, mock_service):
        """Test handling of various audio channel configurations."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            mock_model.transcribe_audio.return_value = 'Multi-channel test'
            mock_pretrained.return_value = mock_model
            
            channel_configs = [
                (1, "Mono"),
                (2, "Stereo"),
                (3, "3-channel"),
                (4, "Quad"),
                (5, "5-channel"),
                (6, "5.1 surround"),
                (8, "7.1 surround"),
            ]
            
            for channels, description in channel_configs:
                with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                    # Create multi-channel audio
                    audio_data = np.zeros((16000, channels), dtype=np.float32)
                    mock_sf.read.return_value = (audio_data, 16000)
                    
                    result = mock_service._transcribe_with_parakeet_mlx(
                        audio_path=f"{description}.wav",
                        model='mlx-community/parakeet-tdt-0.6b-v2'
                    )
                    
                    assert 'text' in result
                    # Should handle multi-channel by converting to mono
    
    def test_chunking_boundary_conditions(self, mock_service):
        """Test edge cases in audio chunking."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            mock_pretrained.return_value = mock_model
            
            test_cases = [
                # (duration, chunk_size, overlap, description)
                (30, 30, 0, "Exact single chunk"),
                (30, 31, 0, "Chunk larger than audio"),
                (30, 15, 0, "Exact two chunks"),
                (30, 15, 5, "Two chunks with overlap"),
                (30, 10, 9, "Maximum overlap"),
                (30, 10, 10, "Overlap equals chunk size"),  # Should be adjusted
                (30, 10, 15, "Overlap larger than chunk"),  # Should be adjusted
            ]
            
            for duration, chunk_size, overlap, description in test_cases:
                with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                    audio_samples = 16000 * duration
                    mock_sf.read.return_value = (np.zeros(audio_samples, dtype=np.float32), 16000)
                    
                    # Calculate expected chunks
                    effective_overlap = min(overlap, chunk_size - 1)
                    if chunk_size >= duration:
                        expected_chunks = 1
                    else:
                        expected_chunks = 1 + (duration - chunk_size) // (chunk_size - effective_overlap)
                        if (duration - chunk_size) % (chunk_size - effective_overlap) > 0:
                            expected_chunks += 1
                    
                    mock_model.transcribe_audio.side_effect = [
                        f'Chunk {i}' for i in range(expected_chunks + 5)  # Extra for safety
                    ]
                    
                    result = mock_service._transcribe_with_parakeet_mlx(
                        audio_path=f"{description}.wav",
                        model='mlx-community/parakeet-tdt-0.6b-v2',
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                    
                    assert 'text' in result
                    assert 'segments' in result
    
    def test_nan_inf_in_audio_data(self, mock_service):
        """Test handling of NaN and Inf values in audio data."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            mock_model.transcribe_audio.return_value = 'Cleaned audio'
            mock_pretrained.return_value = mock_model
            
            test_cases = [
                (np.array([np.nan] * 16000, dtype=np.float32), "NaN values"),
                (np.array([np.inf] * 16000, dtype=np.float32), "Inf values"),
                (np.array([-np.inf] * 16000, dtype=np.float32), "-Inf values"),
                (np.array([np.nan, 0.5, -0.5, np.inf] * 4000, dtype=np.float32), "Mixed values"),
            ]
            
            for audio_data, description in test_cases:
                with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                    mock_sf.read.return_value = (audio_data, 16000)
                    
                    # Should handle invalid values gracefully
                    result = mock_service._transcribe_with_parakeet_mlx(
                        audio_path=f"{description}.wav",
                        model='mlx-community/parakeet-tdt-0.6b-v2'
                    )
                    
                    assert 'text' in result
    
    def test_progress_callback_with_errors(self, mock_service):
        """Test progress callback behavior when errors occur."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            mock_pretrained.return_value = mock_model
            
            progress_calls = []
            
            def progress_callback(percentage, message, metadata):
                progress_calls.append((percentage, message, metadata))
                if percentage == 50:
                    raise ValueError("Progress callback error")
            
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                # Multi-chunk audio
                mock_sf.read.return_value = (np.zeros(16000 * 60), 16000)  # 60 seconds
                
                # First chunk succeeds, second fails
                mock_model.transcribe_audio.side_effect = [
                    'Chunk 1',
                    Exception("Transcription error in chunk 2")
                ]
                
                with pytest.raises(TranscriptionError):
                    mock_service._transcribe_with_parakeet_mlx(
                        audio_path="test.wav",
                        model='mlx-community/parakeet-tdt-0.6b-v2',
                        chunk_size=30,
                        progress_callback=progress_callback
                    )
                
                # Should have some progress calls before error
                assert len(progress_calls) > 0
                assert progress_calls[0][0] == 0  # Start
    
    def test_model_switching_during_transcription(self, mock_service):
        """Test behavior when model is switched during transcription."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            models = []
            
            def create_model(*args, **kwargs):
                model = MagicMock()
                model._model_name = args[0]
                model.transcribe_audio.return_value = f'Model: {args[0]}'
                models.append(model)
                return model
            
            mock_pretrained.side_effect = create_model
            
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                mock_sf.read.return_value = (np.zeros(16000), 16000)
                
                # Start two transcriptions with different models in parallel
                results = []
                
                def transcribe_with_model(model_name):
                    result = mock_service._transcribe_with_parakeet_mlx(
                        audio_path="test.wav",
                        model=model_name
                    )
                    results.append(result)
                
                thread1 = threading.Thread(
                    target=transcribe_with_model,
                    args=('mlx-community/parakeet-tdt-0.6b-v2',)
                )
                thread2 = threading.Thread(
                    target=transcribe_with_model,
                    args=('mlx-community/parakeet-tdt-1.1b',)
                )
                
                thread1.start()
                thread2.start()
                thread1.join()
                thread2.join()
                
                # Both should complete
                assert len(results) == 2
                # May have loaded 2 different models
                assert len(models) >= 1
    
    def test_garbage_collection_during_transcription(self, mock_service):
        """Test behavior when garbage collection occurs during transcription."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            
            call_count = 0
            def transcribe_with_gc(audio):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    # Force garbage collection
                    gc.collect()
                return f'Chunk {call_count}'
            
            mock_model.transcribe_audio.side_effect = transcribe_with_gc
            mock_pretrained.return_value = mock_model
            
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                # Multi-chunk audio
                mock_sf.read.return_value = (np.zeros(16000 * 90), 16000)  # 90 seconds
                
                result = mock_service._transcribe_with_parakeet_mlx(
                    audio_path="test.wav",
                    model='mlx-community/parakeet-tdt-0.6b-v2',
                    chunk_size=30
                )
                
                assert 'text' in result
                assert 'Chunk' in result['text']
                assert call_count >= 3  # Should have processed multiple chunks


class TestMLXParakeetRobustness:
    """Robustness tests for MLX Parakeet transcription."""
    
    def test_model_download_interruption(self, mock_service):
        """Test handling of interrupted model downloads."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            # Simulate download interruption
            mock_pretrained.side_effect = [
                ConnectionError("Download interrupted"),
                TimeoutError("Connection timeout"),
                MagicMock()  # Third attempt succeeds
            ]
            
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                mock_sf.read.return_value = (np.zeros(16000), 16000)
                
                # First two attempts should fail
                for _ in range(2):
                    with pytest.raises(TranscriptionError):
                        mock_service._transcribe_with_parakeet_mlx(
                            audio_path="test.wav",
                            model='mlx-community/parakeet-tdt-0.6b-v2'
                        )
                
                # Reset model cache
                mock_service._parakeet_mlx_model = None
                
                # Third attempt should succeed
                mock_pretrained.return_value.transcribe_audio.return_value = 'Success'
                result = mock_service._transcribe_with_parakeet_mlx(
                    audio_path="test.wav",
                    model='mlx-community/parakeet-tdt-0.6b-v2'
                )
                
                assert result['text'] == 'Success'
    
    def test_audio_format_edge_cases(self, mock_service):
        """Test handling of various audio format edge cases."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            mock_model.transcribe_audio.return_value = 'Format test'
            mock_pretrained.return_value = mock_model
            
            # Test different audio data types
            test_cases = [
                (np.int16, -32768, 32767, "int16"),
                (np.int32, -2147483648, 2147483647, "int32"),
                (np.float32, -1.0, 1.0, "float32"),
                (np.float64, -1.0, 1.0, "float64"),
            ]
            
            for dtype, min_val, max_val, description in test_cases:
                with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                    # Create audio with extreme values
                    audio_data = np.array([min_val, max_val] * 8000, dtype=dtype)
                    mock_sf.read.return_value = (audio_data, 16000)
                    
                    result = mock_service._transcribe_with_parakeet_mlx(
                        audio_path=f"{description}.wav",
                        model='mlx-community/parakeet-tdt-0.6b-v2'
                    )
                    
                    assert 'text' in result
    
    def test_unicode_and_special_chars_in_transcription(self, mock_service):
        """Test handling of Unicode and special characters in transcription results."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            
            # Test various special characters and Unicode
            test_texts = [
                "Hello 世界",  # Chinese
                "Привет мир",  # Russian
                "مرحبا بالعالم",  # Arabic
                "🎵 Music 🎶 Symbols 🎵",  # Emojis
                "Special\nChars\tTab\r\nNewlines",  # Control characters
                "Math: ∑∏∫∞ ≠ ≈ ≤ ≥",  # Math symbols
                "Quotes: \"double\" 'single' «guillemets»",  # Various quotes
            ]
            
            mock_pretrained.return_value = mock_model
            
            for test_text in test_texts:
                with patch('tldw_chatbook.Local_Ingestion.transcription_service.sf') as mock_sf:
                    mock_sf.read.return_value = (np.zeros(16000), 16000)
                    mock_model.transcribe_audio.return_value = test_text
                    
                    result = mock_service._transcribe_with_parakeet_mlx(
                        audio_path="unicode.wav",
                        model='mlx-community/parakeet-tdt-0.6b-v2'
                    )
                    
                    assert result['text'] == test_text
                    assert len(result['segments']) >= 1
                    assert result['segments'][0]['text'] == test_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])