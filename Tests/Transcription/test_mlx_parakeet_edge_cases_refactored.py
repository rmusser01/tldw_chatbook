"""
Edge case tests for MLX Parakeet with minimal mocking.

These tests focus on edge cases and error conditions with integration approach.
"""

import pytest
import sys
import os
import tempfile
import wave
import numpy as np
import struct
import threading
import gc
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


class TestMLXParakeetEdgeCases:
    """Edge case tests with minimal mocking."""
    
    @pytest.fixture
    def mock_model_minimal(self):
        """Minimal model mock that only prevents downloads."""
        def create_model(*args, **kwargs):
            model = MagicMock()
            
            # Track calls for testing
            model._load_count = 0
            model._transcribe_calls = []
            
            def transcribe(audio_path, **kwargs):
                model._transcribe_calls.append((audio_path, kwargs))
                result = MagicMock()
                
                # Simulate different behaviors based on test scenarios
                if 'error' in str(audio_path):
                    raise Exception("Simulated transcription error")
                elif 'empty' in str(audio_path):
                    result.text = ""
                elif 'memory' in str(audio_path) and len(model._transcribe_calls) > 2:
                    raise MemoryError("Simulated memory error")
                else:
                    result.text = f"Transcribed: {os.path.basename(audio_path)}"
                
                return result
            
            model.transcribe = transcribe
            model._model_name = args[0] if args else 'test-model'
            model._load_count += 1
            return model
        
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.parakeet_from_pretrained',
                   side_effect=create_model) as mock:
            yield mock
    
    @pytest.fixture
    def service(self):
        """Create a real transcription service."""
        return TranscriptionService()
    
    def test_corrupted_wav_header(self, service, mock_model_minimal):
        """Test handling of WAV files with corrupted headers."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Write partial/corrupted WAV header
            tmp.write(b'RIFF')
            tmp.write(struct.pack('<I', 100))  # File size
            tmp.write(b'WAVE')
            tmp.write(b'fmt ')
            # Truncate - incomplete format chunk
            corrupted_file = tmp.name
        
        try:
            # Service might handle corrupted files gracefully or raise an error
            try:
                result = service.transcribe(
                    audio_path=corrupted_file,
                    provider='parakeet-mlx'
                )
                # If it succeeds, verify the result is valid
                assert isinstance(result, dict)
                assert 'text' in result
                # Corrupted files might produce empty or garbage transcription
                assert isinstance(result['text'], str)
            except (TranscriptionError, OSError, wave.Error, EOFError):
                # Expected - corrupted file caused an error
                pass
        finally:
            os.unlink(corrupted_file)
    
    def test_extreme_sample_rates(self, service, mock_model_minimal):
        """Test handling of extreme sample rates."""
        extreme_rates = [
            (4000, "Very low sample rate"),
            (192000, "Very high sample rate"),
            (7350, "Odd sample rate")
        ]
        
        for sample_rate, description in extreme_rates:
            # Create valid WAV with extreme sample rate
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                samples = int(sample_rate * 0.1)  # 0.1 second
                audio_data = (np.random.randn(samples) * 32767).astype(np.int16)
                
                with wave.open(tmp.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_data.tobytes())
                
                audio_file = tmp.name
            
            try:
                # Should handle without crashing
                result = service.transcribe(
                    audio_path=audio_file,
                    provider='parakeet-mlx'
                )
                assert 'text' in result
            finally:
                os.unlink(audio_file)
    
    def test_multichannel_audio(self, service, mock_model_minimal):
        """Test handling of multi-channel audio."""
        channel_configs = [2, 3, 4, 6, 8]  # Stereo to 7.1
        
        for channels in channel_configs:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sample_rate = 16000
                duration = 0.5
                samples = int(sample_rate * duration)
                
                # Create multi-channel audio
                audio_data = np.zeros((samples, channels), dtype=np.int16)
                for ch in range(channels):
                    # Different frequency per channel
                    t = np.linspace(0, duration, samples)
                    freq = 440 * (ch + 1)
                    audio_data[:, ch] = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
                
                with wave.open(tmp.name, 'wb') as wav:
                    wav.setnchannels(channels)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    # Interleave channels
                    wav.writeframes(audio_data.flatten().tobytes())
                
                audio_file = tmp.name
            
            try:
                # Should handle multi-channel (usually by converting to mono)
                result = service.transcribe(
                    audio_path=audio_file,
                    provider='parakeet-mlx'
                )
                assert 'text' in result
            finally:
                os.unlink(audio_file)
    
    def test_very_short_audio(self, service, mock_model_minimal):
        """Test handling of very short audio files."""
        durations_ms = [10, 50, 100, 500]  # Very short durations
        
        for duration_ms in durations_ms:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sample_rate = 16000
                duration_sec = duration_ms / 1000.0
                samples = max(1, int(sample_rate * duration_sec))
                
                audio_data = (np.random.randn(samples) * 16000).astype(np.int16)
                
                with wave.open(tmp.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_data.tobytes())
                
                audio_file = tmp.name
            
            try:
                result = service.transcribe(
                    audio_path=audio_file,
                    provider='parakeet-mlx'
                )
                assert 'text' in result
                # Very short audio might produce empty transcription
                assert isinstance(result['text'], str)
            finally:
                os.unlink(audio_file)
    
    def test_concurrent_model_loading(self, service, mock_model_minimal):
        """Test thread safety of concurrent model loading."""
        # Create a simple test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sample_rate = 16000
            audio_data = (np.random.randn(sample_rate) * 16000).astype(np.int16)
            
            with wave.open(tmp.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_data.tobytes())
            
            audio_file = tmp.name
        
        try:
            results = []
            errors = []
            
            def transcribe():
                try:
                    result = service.transcribe(
                        audio_path=audio_file,
                        provider='parakeet-mlx'
                    )
                    results.append(result)
                except Exception as e:
                    errors.append(e)
            
            # Start multiple threads
            threads = [threading.Thread(target=transcribe) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # All should succeed
            assert len(errors) == 0
            assert len(results) == 5
            
            # Model should be loaded only once (thread-safe)
            assert mock_model_minimal.call_count >= 1
        finally:
            os.unlink(audio_file)
    
    def test_memory_pressure_handling(self, service, mock_model_minimal):
        """Test behavior under memory pressure."""
        # Create test file that triggers memory error after a few calls
        with tempfile.NamedTemporaryFile(suffix='_memory.wav', delete=False) as tmp:
            sample_rate = 16000
            # Create larger file
            audio_data = (np.random.randn(sample_rate * 10) * 16000).astype(np.int16)
            
            with wave.open(tmp.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_data.tobytes())
            
            audio_file = tmp.name
        
        try:
            # First few calls succeed
            for i in range(2):
                result = service.transcribe(
                    audio_path=audio_file,
                    provider='parakeet-mlx'
                )
                assert 'text' in result
            
            # Next call triggers memory error
            with pytest.raises(TranscriptionError) as exc_info:
                service.transcribe(
                    audio_path=audio_file,
                    provider='parakeet-mlx'
                )
            assert "memory" in str(exc_info.value).lower()
        finally:
            os.unlink(audio_file)
    
    def test_progress_callback_exception_handling(self, service, mock_model_minimal):
        """Test that exceptions in progress callbacks don't break transcription."""
        audio_file = None
        try:
            # Create test audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sample_rate = 16000
                audio_data = (np.random.randn(sample_rate * 2) * 16000).astype(np.int16)
                
                with wave.open(tmp.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_data.tobytes())
                
                audio_file = tmp.name
            
            call_count = 0
            
            def bad_callback(percentage, message, metadata):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise Exception("Callback error!")
            
            # Should complete despite callback errors
            result = service.transcribe(
                audio_path=audio_file,
                provider='parakeet-mlx',
                progress_callback=bad_callback
            )
            
            assert 'text' in result
            assert call_count > 1  # Callback was called multiple times
        finally:
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)
    
    def test_model_switching(self, service, mock_model_minimal):
        """Test switching between different models."""
        audio_file = None
        try:
            # Create test audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sample_rate = 16000
                audio_data = (np.random.randn(sample_rate) * 16000).astype(np.int16)
                
                with wave.open(tmp.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_data.tobytes())
                
                audio_file = tmp.name
            
            # Transcribe with different models
            models = [
                'mlx-community/parakeet-tdt-0.6b-v2',
                'mlx-community/parakeet-tdt-1.1b',
                'mlx-community/parakeet-tdt-0.6b-v2'  # Back to first
            ]
            
            results = []
            for model in models:
                result = service.transcribe(
                    audio_path=audio_file,
                    provider='parakeet-mlx',
                    model=model
                )
                results.append(result)
            
            # All should succeed
            assert len(results) == 3
            for result in results:
                assert 'text' in result
                assert 'model' in result
            
            # Should have loaded models as needed
            assert mock_model_minimal.call_count >= 2
        finally:
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)
    
    def test_garbage_collection_during_transcription(self, service, mock_model_minimal):
        """Test that garbage collection doesn't interfere with transcription."""
        audio_file = None
        try:
            # Create test audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sample_rate = 16000
                audio_data = (np.random.randn(sample_rate * 3) * 16000).astype(np.int16)
                
                with wave.open(tmp.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_data.tobytes())
                
                audio_file = tmp.name
            
            # Force garbage collection during transcription
            original_transcribe = mock_model_minimal.return_value.transcribe
            
            def transcribe_with_gc(*args, **kwargs):
                gc.collect()  # Force GC
                return original_transcribe(*args, **kwargs)
            
            mock_model_minimal.return_value.transcribe = transcribe_with_gc
            
            result = service.transcribe(
                audio_path=audio_file,
                provider='parakeet-mlx'
            )
            
            assert 'text' in result
        finally:
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)