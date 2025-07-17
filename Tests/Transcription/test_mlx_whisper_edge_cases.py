"""
Edge case tests for MLX Whisper transcription backend.

This module tests edge cases, error conditions, and special scenarios
for the Lightning Whisper MLX implementation.
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

from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionService,
    TranscriptionError,
    ConversionError,
    protect_file_descriptors
)


pytestmark = pytest.mark.skipif(
    sys.platform != 'darwin',
    reason="MLX Whisper tests only run on macOS"
)


class TestMLXWhisperEdgeCases:
    """Edge case tests for MLX Whisper transcription."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mocked transcription service."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LIGHTNING_WHISPER_AVAILABLE', True), \
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
    def large_audio_file(self):
        """Create a large audio file for memory testing."""
        sample_rate = 48000
        duration = 60  # 1 minute
        channels = 2   # Stereo
        
        # Generate large audio data
        samples = int(sample_rate * duration)
        audio_data = np.random.randint(-32768, 32767, size=(samples, channels), dtype=np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            yield tmp_file.name
            
        os.unlink(tmp_file.name)
    
    def test_concurrent_transcriptions(self, mock_service):
        """Test concurrent transcription requests."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            mock_instance = MagicMock()
            mock_instance.transcribe.return_value = {
                'text': 'Concurrent transcription',
                'segments': [],
                'language': 'en'
            }
            mock_mlx.return_value = mock_instance
            
            # Create multiple audio files
            audio_files = []
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    # Create minimal valid WAV
                    with wave.open(tmp.name, 'wb') as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(16000)
                        wav.writeframes(b'\x00' * 1000)
                    audio_files.append(tmp.name)
            
            try:
                # Run concurrent transcriptions
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [
                        executor.submit(
                            mock_service._transcribe_with_lightning_whisper_mlx,
                            audio_path=audio_file
                        )
                        for audio_file in audio_files
                    ]
                    
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                
                # All should succeed
                assert len(results) == 3
                assert all(r['text'] == 'Concurrent transcription' for r in results)
                
                # Model should only be loaded once (due to caching)
                assert mock_mlx.call_count == 1
                
            finally:
                # Cleanup
                for audio_file in audio_files:
                    os.unlink(audio_file)
    
    def test_memory_pressure(self, mock_service, large_audio_file):
        """Test transcription under memory pressure."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            mock_instance = MagicMock()
            
            # Simulate memory pressure by raising MemoryError
            mock_instance.transcribe.side_effect = MemoryError("Insufficient memory")
            mock_mlx.return_value = mock_instance
            
            with pytest.raises(TranscriptionError) as exc_info:
                mock_service._transcribe_with_lightning_whisper_mlx(
                    audio_path=large_audio_file
                )
            
            assert "transcription failed" in str(exc_info.value).lower()
    
    def test_file_descriptor_protection(self):
        """Test the file descriptor protection context manager."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Mock stdout/stderr to simulate captured output
        mock_stdout = MagicMock()
        mock_stdout.fileno.side_effect = ValueError("I/O operation on closed file")
        
        sys.stdout = mock_stdout
        sys.stderr = mock_stdout
        
        try:
            with protect_file_descriptors():
                # Inside the context, stdout/stderr should be valid
                assert sys.stdout != mock_stdout
                assert sys.stderr != mock_stdout
                
                # Should be able to write
                sys.stdout.write("test")
                sys.stderr.write("test")
            
            # After context, should be restored
            assert sys.stdout == mock_stdout
            assert sys.stderr == mock_stdout
            
        finally:
            # Restore original
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    def test_model_switching_race_condition(self, mock_service):
        """Test race condition when switching models."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            mock_instances = []
            call_count = 0
            
            def create_mock_instance(*args, **kwargs):
                nonlocal call_count
                instance = MagicMock()
                instance.transcribe.return_value = {
                    'text': f'Transcription {call_count}',
                    'segments': [],
                    'language': 'en'
                }
                instance._model_name = kwargs['model']
                mock_instances.append(instance)
                call_count += 1
                
                # Simulate slow model loading
                time.sleep(0.1)
                return instance
            
            mock_mlx.side_effect = create_mock_instance
            
            # Create a barrier to synchronize thread starts
            barrier = threading.Barrier(2)
            results = {}
            errors = {}
            
            def transcribe_with_model(model_name):
                try:
                    barrier.wait()  # Wait for both threads to be ready
                    result = mock_service._transcribe_with_lightning_whisper_mlx(
                        audio_path="dummy.wav",
                        model=model_name
                    )
                    results[model_name] = result
                except Exception as e:
                    errors[model_name] = e
            
            # Start two threads that try to load different models simultaneously
            thread1 = threading.Thread(target=transcribe_with_model, args=('base',))
            thread2 = threading.Thread(target=transcribe_with_model, args=('large',))
            
            thread1.start()
            thread2.start()
            
            thread1.join()
            thread2.join()
            
            # Both should succeed
            assert len(errors) == 0
            assert 'base' in results
            assert 'large' in results
            
            # Should have created two model instances
            assert mock_mlx.call_count == 2
    
    def test_special_characters_in_transcription(self, mock_service):
        """Test handling of special characters in transcription results."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            mock_instance = MagicMock()
            
            # Test various special characters
            special_text = "Hello\nWorld\t[Special] <Characters> & \"Quotes\" 'More' \u2022 Bullet"
            
            mock_instance.transcribe.return_value = {
                'text': special_text,
                'segments': [
                    {'start': 0.0, 'end': 1.0, 'text': 'Hello\nWorld'},
                    {'start': 1.0, 'end': 2.0, 'text': '\t[Special] <Characters>'},
                    {'start': 2.0, 'end': 3.0, 'text': '& "Quotes" \'More\''},
                    {'start': 3.0, 'end': 4.0, 'text': '\u2022 Bullet'}
                ],
                'language': 'en'
            }
            mock_mlx.return_value = mock_instance
            
            result = mock_service._transcribe_with_lightning_whisper_mlx(
                audio_path="dummy.wav"
            )
            
            # Check that special characters are preserved
            assert result['text'] == special_text
            assert len(result['segments']) == 4
            
            # Check each segment - note that text is stripped
            assert result['segments'][0]['text'] == 'Hello\nWorld'  # Newline preserved within text
            assert result['segments'][1]['text'] == '[Special] <Characters>'  # Tab is stripped by strip()
            assert '"' in result['segments'][2]['text']
            assert '\u2022' in result['segments'][3]['text']
    
    def test_very_long_transcription(self, mock_service):
        """Test handling of very long transcription results."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            mock_instance = MagicMock()
            
            # Generate very long text
            long_text = " ".join([f"Sentence {i}." for i in range(10000)])
            
            # Generate many segments
            segments = []
            words_per_segment = 10
            words = long_text.split()
            
            for i in range(0, len(words), words_per_segment):
                segment_text = " ".join(words[i:i+words_per_segment])
                segments.append({
                    'start': i * 0.5,
                    'end': (i + words_per_segment) * 0.5,
                    'text': segment_text
                })
            
            mock_instance.transcribe.return_value = {
                'text': long_text,
                'segments': segments,
                'language': 'en'
            }
            mock_mlx.return_value = mock_instance
            
            result = mock_service._transcribe_with_lightning_whisper_mlx(
                audio_path="dummy.wav"
            )
            
            # Should handle long transcription
            assert len(result['text']) == len(long_text)
            assert len(result['segments']) == len(segments)
            
            # All segments should be properly formatted
            for seg in result['segments']:
                assert 'text' in seg
                assert 'start' in seg
                assert 'end' in seg
                assert 'Text' in seg
                assert 'Time_Start' in seg
                assert 'Time_End' in seg
    
    def test_zero_length_audio(self, mock_service):
        """Test handling of zero-length audio files."""
        # Create a valid but empty WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                # Write zero frames
                wav_file.writeframes(b'')
            
            audio_file = tmp_file.name
        
        try:
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
                mock_instance = MagicMock()
                mock_instance.transcribe.return_value = {
                    'text': '',
                    'segments': [],
                    'language': 'en'
                }
                mock_mlx.return_value = mock_instance
                
                result = mock_service._transcribe_with_lightning_whisper_mlx(
                    audio_path=audio_file
                )
                
                # Should handle empty audio gracefully
                assert result['text'] == ''
                assert len(result['segments']) == 0  # No segments for empty audio
                
        finally:
            os.unlink(audio_file)
    
    def test_progress_callback_exception(self, mock_service):
        """Test that exceptions in progress callback don't break transcription."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            mock_instance = MagicMock()
            mock_instance.transcribe.return_value = {
                'text': 'Test transcription',
                'segments': [],
                'language': 'en'
            }
            mock_mlx.return_value = mock_instance
            
            # Create a progress callback that raises an exception
            def bad_callback(percentage, message, metadata):
                raise ValueError("Callback error")
            
            # Should still complete transcription despite callback error
            result = mock_service._transcribe_with_lightning_whisper_mlx(
                audio_path="dummy.wav",
                progress_callback=bad_callback
            )
            
            assert result['text'] == 'Test transcription'
    
    def test_model_cache_invalidation(self, mock_service):
        """Test that model cache is properly invalidated."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            mock_instance1 = MagicMock()
            mock_instance1.transcribe.return_value = {
                'text': 'First model',
                'segments': [],
                'language': 'en'
            }
            mock_instance1._model_name = 'base'
            mock_instance1._batch_size = 12
            mock_instance1._quant = None
            
            mock_instance2 = MagicMock()
            mock_instance2.transcribe.return_value = {
                'text': 'Second model',
                'segments': [],
                'language': 'en'
            }
            mock_instance2._model_name = 'base'
            mock_instance2._batch_size = 24
            mock_instance2._quant = None
            
            mock_mlx.side_effect = [mock_instance1, mock_instance2]
            
            # First transcription
            result1 = mock_service._transcribe_with_lightning_whisper_mlx(
                audio_path="dummy.wav",
                model='base',
                batch_size=12
            )
            
            # Change batch size - should reload model
            result2 = mock_service._transcribe_with_lightning_whisper_mlx(
                audio_path="dummy.wav",
                model='base',
                batch_size=24
            )
            
            # Model should be reloaded due to batch size change
            assert mock_mlx.call_count == 2
            assert result1['text'] == 'First model'
            assert result2['text'] == 'Second model'
    
    @pytest.mark.parametrize("quant_value", [None, '4bit', '8bit', 'invalid'])
    def test_quantization_values(self, mock_service, quant_value):
        """Test various quantization values."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            if quant_value == 'invalid':
                # Invalid quantization should raise an error during model creation
                mock_mlx.side_effect = ValueError(f"Invalid quantization: {quant_value}")
                
                with pytest.raises(TranscriptionError):
                    mock_service._transcribe_with_lightning_whisper_mlx(
                        audio_path="dummy.wav",
                        quant=quant_value
                    )
            else:
                mock_instance = MagicMock()
                mock_instance.transcribe.return_value = {
                    'text': f'Quantization: {quant_value}',
                    'segments': [],
                    'language': 'en'
                }
                mock_mlx.return_value = mock_instance
                
                result = mock_service._transcribe_with_lightning_whisper_mlx(
                    audio_path="dummy.wav",
                    model='base',
                    quant=quant_value
                )
                
                assert result['quantization'] == quant_value
                
                # Check model was created with correct quantization
                mock_mlx.assert_called_with(
                    model='base',
                    batch_size=12,
                    quant=quant_value
                )


class TestMLXWhisperRobustness:
    """Robustness tests for MLX Whisper transcription."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mocked transcription service."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LIGHTNING_WHISPER_AVAILABLE', True), \
             patch('tldw_chatbook.Local_Ingestion.transcription_service.get_cli_setting') as mock_settings:
            
            mock_settings.return_value = None  # Use defaults
            service = TranscriptionService()
            return service
    
    def test_network_timeout_during_model_download(self, mock_service):
        """Test handling of network timeout during model download."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            # Simulate network timeout
            mock_mlx.side_effect = TimeoutError("Network timeout during model download")
            
            with pytest.raises(TranscriptionError) as exc_info:
                mock_service._transcribe_with_lightning_whisper_mlx(
                    audio_path="dummy.wav",
                    model='large-v3'
                )
            
            assert "Failed to load Lightning Whisper MLX model" in str(exc_info.value)
    
    def test_disk_full_during_transcription(self, mock_service):
        """Test handling of disk full error during transcription."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            mock_instance = MagicMock()
            mock_instance.transcribe.side_effect = OSError("No space left on device")
            mock_mlx.return_value = mock_instance
            
            with pytest.raises(TranscriptionError) as exc_info:
                mock_service._transcribe_with_lightning_whisper_mlx(
                    audio_path="dummy.wav"
                )
            
            assert "transcription failed" in str(exc_info.value).lower()
    
    def test_interrupted_transcription(self, mock_service):
        """Test handling of interrupted transcription."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            mock_instance = MagicMock()
            mock_instance.transcribe.side_effect = KeyboardInterrupt()
            mock_mlx.return_value = mock_instance
            
            with pytest.raises(KeyboardInterrupt):
                mock_service._transcribe_with_lightning_whisper_mlx(
                    audio_path="dummy.wav"
                )
    
    def test_unicode_in_file_path(self, mock_service):
        """Test handling of Unicode characters in file paths."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.LightningWhisperMLX') as mock_mlx:
            mock_instance = MagicMock()
            mock_instance.transcribe.return_value = {
                'text': 'Unicode path test',
                'segments': [],
                'language': 'en'
            }
            mock_mlx.return_value = mock_instance
            
            # Test with Unicode characters in path
            unicode_path = "/tmp/音声ファイル.wav"
            
            result = mock_service._transcribe_with_lightning_whisper_mlx(
                audio_path=unicode_path
            )
            
            assert result['text'] == 'Unicode path test'
            mock_instance.transcribe.assert_called_with(unicode_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])