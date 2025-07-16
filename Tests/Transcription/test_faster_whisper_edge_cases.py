"""
Edge case tests for faster-whisper transcription backend.

This module tests edge cases, error conditions, and special scenarios
for the faster-whisper implementation.
"""

import pytest
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
import json
from typing import List, Generator

from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionService,
    TranscriptionError,
    ConversionError,
    protect_file_descriptors
)


class MockSegment:
    """Mock segment for edge case testing."""
    def __init__(self, start, end, text, **kwargs):
        self.start = start
        self.end = end
        self.text = text
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockInfo:
    """Mock info object for edge case testing."""
    def __init__(self, language="en", language_probability=0.99, duration=10.0, **kwargs):
        self.language = language
        self.language_probability = language_probability
        self.duration = duration
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestFasterWhisperEdgeCases:
    """Edge case tests for faster-whisper transcription."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mocked transcription service."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.FASTER_WHISPER_AVAILABLE', True), \
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
    
    def test_concurrent_model_loading(self, mock_service):
        """Test concurrent model loading attempts."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            # Simulate slow model loading
            load_times = []
            
            def slow_model_init(*args, **kwargs):
                load_times.append(time.time())
                time.sleep(0.5)  # Simulate loading time
                model = MagicMock()
                model.transcribe.return_value = (
                    iter([MockSegment(0, 1, "Test")]),
                    MockInfo()
                )
                return model
            
            mock_model_class.side_effect = slow_model_init
            
            # Try to load model concurrently from multiple threads
            results = []
            errors = []
            
            def transcribe():
                try:
                    result = mock_service._transcribe_with_faster_whisper(
                        audio_path="dummy.wav",
                        model="base",
                        language="en",
                        vad_filter=True
                    )
                    results.append(result)
                except Exception as e:
                    errors.append(e)
            
            threads = [threading.Thread(target=transcribe) for _ in range(3)]
            
            start_time = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # All should succeed
            assert len(results) == 3
            assert len(errors) == 0
            
            # Model should only be loaded once due to caching
            assert mock_model_class.call_count == 1
            
            # Load times should be close together (within thread scheduling tolerance)
            if len(load_times) > 1:
                time_diff = max(load_times) - min(load_times)
                assert time_diff < 0.1  # Should be nearly simultaneous
    
    def test_model_cache_key_variations(self, mock_service):
        """Test that cache keys properly differentiate configurations."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = (
                iter([MockSegment(0, 1, "Test")]),
                MockInfo()
            )
            mock_model_class.return_value = mock_model
            
            # Test different configurations that should create different cache entries
            configs = [
                ('base', 'cpu', 'int8'),
                ('base', 'cpu', 'float16'),
                ('base', 'cuda', 'int8'),
                ('base', 'cuda:0', 'int8'),
                ('base', 'cuda:1', 'int8'),
                ('large', 'cpu', 'int8'),
                ('large-v2', 'cpu', 'int8'),
                ('large-v3', 'cpu', 'int8'),
            ]
            
            for model, device, compute_type in configs:
                # Update service config
                mock_service.config['device'] = device
                mock_service.config['compute_type'] = compute_type
                
                # Clear this specific cache entry
                cache_key = (model, device, compute_type)
                if cache_key in mock_service._model_cache:
                    del mock_service._model_cache[cache_key]
                
                mock_model_class.reset_mock()
                
                # Transcribe
                mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model=model,
                    language="en",
                    vad_filter=True
                )
                
                # Should create new model
                mock_model_class.assert_called_once()
                assert cache_key in mock_service._model_cache
    
    def test_segment_generator_exceptions(self, mock_service):
        """Test handling of exceptions during segment generation."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            # Create a generator that fails partway through
            def failing_generator():
                yield MockSegment(0, 1, "First segment")
                yield MockSegment(1, 2, "Second segment")
                raise RuntimeError("Generator failed")
            
            mock_model.transcribe.return_value = (
                failing_generator(),
                MockInfo(duration=10.0)
            )
            mock_model_class.return_value = mock_model
            
            with pytest.raises(TranscriptionError) as exc_info:
                mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model="base",
                    language="en",
                    vad_filter=True
                )
            
            assert "Transcription failed" in str(exc_info.value)
            assert "Generator failed" in str(exc_info.value)
    
    def test_progress_callback_exceptions(self, mock_service):
        """Test that exceptions in progress callbacks don't break transcription."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = (
                iter([
                    MockSegment(0, 1, "First"),
                    MockSegment(1, 2, "Second")
                ]),
                MockInfo(duration=2.0)
            )
            mock_model_class.return_value = mock_model
            
            call_count = 0
            
            def bad_callback(percentage, message, metadata):
                nonlocal call_count
                call_count += 1
                if call_count > 2:  # Fail after a few calls
                    raise ValueError("Callback error")
            
            # Should complete despite callback errors
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.logger') as mock_logger:
                result = mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model="base",
                    language="en",
                    vad_filter=True,
                    progress_callback=bad_callback
                )
            
            assert result['text'] == "First Second"
            assert len(result['segments']) == 2
    
    def test_language_detection_edge_cases(self, mock_service):
        """Test edge cases in language detection."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            # Test various language detection scenarios
            test_cases = [
                # (detected_lang, probability, should_warn)
                (None, None, True),  # No language detected
                ("en", 0.1, True),   # Very low confidence
                ("en", 0.5, False),  # Medium confidence
                ("zh", 0.99, False), # High confidence non-English
                ("", 0.9, True),     # Empty language code
            ]
            
            for detected_lang, probability, should_warn in test_cases:
                mock_model.transcribe.return_value = (
                    iter([MockSegment(0, 1, "Test")]),
                    MockInfo(
                        language=detected_lang,
                        language_probability=probability,
                        duration=1.0
                    )
                )
                mock_model_class.return_value = mock_model
                
                result = mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model="base",
                    language="auto",  # Auto-detect
                    vad_filter=True
                )
                
                assert result['language'] == detected_lang
                if probability is not None:
                    assert result['language_probability'] == probability
    
    def test_very_long_segments(self, mock_service):
        """Test handling of very long individual segments."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            # Create a very long segment (10 minutes)
            long_text = " ".join([f"Word {i}" for i in range(10000)])
            
            mock_model.transcribe.return_value = (
                iter([
                    MockSegment(0, 600, long_text),  # 10-minute segment
                    MockSegment(600, 601, "Short segment")
                ]),
                MockInfo(duration=601)
            )
            mock_model_class.return_value = mock_model
            
            result = mock_service._transcribe_with_faster_whisper(
                audio_path="dummy.wav",
                model="base",
                language="en",
                vad_filter=True
            )
            
            # Should handle long segments
            assert len(result['segments']) == 2
            assert len(result['segments'][0]['text']) == len(long_text)
            assert result['duration'] == 601
    
    def test_overlapping_segments(self, mock_service):
        """Test handling of overlapping segment timestamps."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            # Create overlapping segments
            mock_model.transcribe.return_value = (
                iter([
                    MockSegment(0, 2, "First"),
                    MockSegment(1, 3, "Overlapping"),  # Overlaps with first
                    MockSegment(2.5, 4, "Third"),
                    MockSegment(3, 5, "Fourth")  # Gap before this
                ]),
                MockInfo(duration=5)
            )
            mock_model_class.return_value = mock_model
            
            result = mock_service._transcribe_with_faster_whisper(
                audio_path="dummy.wav",
                model="base",
                language="en",
                vad_filter=True
            )
            
            # Should preserve all segments even if overlapping
            assert len(result['segments']) == 4
            
            # Check that timestamps are preserved
            assert result['segments'][0]['start'] == 0
            assert result['segments'][0]['end'] == 2
            assert result['segments'][1]['start'] == 1  # Overlapping
            assert result['segments'][1]['end'] == 3
    
    def test_negative_timestamps(self, mock_service):
        """Test handling of negative timestamps (shouldn't happen but test anyway)."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            # Create segments with invalid timestamps
            mock_model.transcribe.return_value = (
                iter([
                    MockSegment(-1, 0, "Before zero?"),
                    MockSegment(0, 1, "Normal"),
                    MockSegment(1, 0.5, "Backwards?"),  # End before start
                ]),
                MockInfo(duration=1)
            )
            mock_model_class.return_value = mock_model
            
            result = mock_service._transcribe_with_faster_whisper(
                audio_path="dummy.wav",
                model="base",
                language="en",
                vad_filter=True
            )
            
            # Should still process segments
            assert len(result['segments']) == 3
            # Timestamps are preserved as-is (Whisper's responsibility to provide valid ones)
            assert result['segments'][0]['start'] == -1
            assert result['segments'][2]['end'] == 0.5
    
    def test_unicode_in_segments(self, mock_service):
        """Test handling of Unicode characters in transcription."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            # Test various Unicode scenarios
            unicode_segments = [
                MockSegment(0, 1, "Hello 世界"),  # Chinese
                MockSegment(1, 2, "Привет мир"),  # Russian
                MockSegment(2, 3, "مرحبا بالعالم"),  # Arabic
                MockSegment(3, 4, "🎵🎶 Music symbols 🎵🎶"),  # Emojis
                MockSegment(4, 5, "Café résumé naïve"),  # Accented characters
                MockSegment(5, 6, "Math: ∑∏∫ ≠ ≈"),  # Math symbols
            ]
            
            mock_model.transcribe.return_value = (
                iter(unicode_segments),
                MockInfo(duration=6)
            )
            mock_model_class.return_value = mock_model
            
            result = mock_service._transcribe_with_faster_whisper(
                audio_path="dummy.wav",
                model="base",
                language="multilingual",
                vad_filter=True
            )
            
            # All Unicode should be preserved
            assert len(result['segments']) == 6
            assert "世界" in result['text']
            assert "🎵" in result['text']
            assert "∑" in result['text']
    
    def test_empty_segment_text(self, mock_service):
        """Test handling of segments with empty or whitespace-only text."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            mock_model.transcribe.return_value = (
                iter([
                    MockSegment(0, 1, "Normal text"),
                    MockSegment(1, 2, ""),  # Empty
                    MockSegment(2, 3, "   "),  # Whitespace only
                    MockSegment(3, 4, "\n\t"),  # Other whitespace
                    MockSegment(4, 5, "More text"),
                ]),
                MockInfo(duration=5)
            )
            mock_model_class.return_value = mock_model
            
            result = mock_service._transcribe_with_faster_whisper(
                audio_path="dummy.wav",
                model="base",
                language="en",
                vad_filter=True
            )
            
            # Empty segments are included but text is stripped
            assert len(result['segments']) == 5
            assert result['segments'][1]['text'] == ""
            assert result['segments'][2]['text'] == ""
            assert result['segments'][3]['text'] == ""
            
            # Full text joins non-empty segments
            assert result['text'] == "Normal text More text"
    
    def test_beam_search_edge_cases(self, mock_service):
        """Test edge cases with beam search parameters."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            # Track transcribe calls
            transcribe_calls = []
            
            def track_transcribe(*args, **kwargs):
                transcribe_calls.append(kwargs)
                return iter([MockSegment(0, 1, "Test")]), MockInfo()
            
            mock_model.transcribe.side_effect = track_transcribe
            mock_model_class.return_value = mock_model
            
            # Default beam search
            mock_service._transcribe_with_faster_whisper(
                audio_path="dummy.wav",
                model="base",
                language="en",
                vad_filter=True
            )
            
            # Check default values
            assert transcribe_calls[-1]['beam_size'] == 5
            assert transcribe_calls[-1]['best_of'] == 5
    
    def test_translation_language_combinations(self, mock_service):
        """Test various source/target language combinations for translation."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            test_cases = [
                # (source_lang, target_lang, expected_task)
                ('es', 'en', 'translate'),     # Spanish to English
                ('fr', 'en', 'translate'),     # French to English
                ('en', 'en', 'transcribe'),    # English to English (no translation)
                ('es', 'es', 'transcribe'),    # Spanish to Spanish (no translation)
                ('es', 'fr', 'transcribe'),    # Spanish to French (not supported, transcribe only)
                (None, 'en', 'transcribe'),    # Auto-detect to English
                ('auto', 'en', 'transcribe'),  # Auto to English
            ]
            
            for source_lang, target_lang, expected_task in test_cases:
                transcribe_calls = []
                
                def track_transcribe(*args, **kwargs):
                    transcribe_calls.append(kwargs)
                    return iter([MockSegment(0, 1, "Test")]), MockInfo(language=source_lang or 'es')
                
                mock_model.transcribe.side_effect = track_transcribe
                mock_model_class.return_value = mock_model
                
                result = mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model="base",
                    language=source_lang or 'auto',
                    vad_filter=True,
                    source_lang=source_lang,
                    target_lang=target_lang
                )
                
                # Check task
                assert transcribe_calls[-1]['task'] == expected_task
                
                if expected_task == 'translate':
                    assert result['task'] == 'translation'
                    assert result['target_language'] == 'en'
    
    def test_model_loading_with_file_descriptor_issues(self, mock_service):
        """Test model loading with file descriptor issues."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            # First attempt fails with file descriptor error
            mock_model_class.side_effect = [
                OSError("bad value(s) in fds_to_keep"),
            ]
            
            with pytest.raises(TranscriptionError) as exc_info:
                mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model="base",
                    language="en",
                    vad_filter=True
                )
            
            # Check error message includes helpful information
            error_msg = str(exc_info.value)
            assert "file descriptors" in error_msg
            assert "OBJC_DISABLE_INITIALIZE_FORK_SAFETY" in error_msg
            assert "huggingface-cli download" in error_msg
    
    def test_zero_duration_audio(self, mock_service):
        """Test handling of audio with zero or invalid duration."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            # Test with zero duration
            mock_model.transcribe.return_value = (
                iter([MockSegment(0, 0, "Zero duration?")]),
                MockInfo(duration=0)
            )
            mock_model_class.return_value = mock_model
            
            result = mock_service._transcribe_with_faster_whisper(
                audio_path="dummy.wav",
                model="base",
                language="en",
                vad_filter=True
            )
            
            assert result['duration'] == 0
            # Should not crash on zero duration
            assert 'text' in result
            
            # Test with None duration
            mock_model.transcribe.return_value = (
                iter([MockSegment(0, 1, "No duration")]),
                MockInfo(duration=None)
            )
            
            result = mock_service._transcribe_with_faster_whisper(
                audio_path="dummy.wav",
                model="base",
                language="en",
                vad_filter=True
            )
            
            # Should handle None duration
            assert result['duration'] is None or result['duration'] == 0
    
    def test_vad_filter_edge_cases(self, mock_service):
        """Test edge cases with VAD (Voice Activity Detection) filter."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            # VAD might filter out all segments
            mock_model.transcribe.return_value = (
                iter([]),  # No segments after VAD
                MockInfo(duration=10)
            )
            mock_model_class.return_value = mock_model
            
            result = mock_service._transcribe_with_faster_whisper(
                audio_path="silence.wav",
                model="base",
                language="en",
                vad_filter=True
            )
            
            # Should handle empty result
            assert result['text'] == ""
            assert len(result['segments']) == 0
            assert result['duration'] == 10
    
    def test_compute_type_compatibility(self, mock_service):
        """Test compute type compatibility with different devices."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            
            # Test invalid compute type for device
            mock_model_class.side_effect = RuntimeError("Compute type 'float16' is not supported on CPU")
            
            mock_service.config['device'] = 'cpu'
            mock_service.config['compute_type'] = 'float16'
            
            with pytest.raises(TranscriptionError) as exc_info:
                mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model="base",
                    language="en",
                    vad_filter=True
                )
            
            assert "Compute type" in str(exc_info.value)


class TestFasterWhisperConcurrency:
    """Test concurrent access and thread safety."""
    
    def test_concurrent_transcriptions_same_model(self, mock_service):
        """Test multiple concurrent transcriptions using the same model."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            # Track concurrent calls
            active_transcriptions = []
            max_concurrent = 0
            lock = threading.Lock()
            
            def concurrent_transcribe(*args, **kwargs):
                with lock:
                    active_transcriptions.append(threading.current_thread().name)
                    max_concurrent_local = len(active_transcriptions)
                
                time.sleep(0.1)  # Simulate transcription time
                
                with lock:
                    active_transcriptions.remove(threading.current_thread().name)
                
                return iter([MockSegment(0, 1, f"Thread {threading.current_thread().name}")]), MockInfo()
            
            mock_model.transcribe.side_effect = concurrent_transcribe
            mock_model_class.return_value = mock_model
            
            # Run concurrent transcriptions
            results = []
            threads = []
            
            for i in range(5):
                def transcribe(idx):
                    result = mock_service._transcribe_with_faster_whisper(
                        audio_path=f"dummy{idx}.wav",
                        model="base",
                        language="en",
                        vad_filter=True
                    )
                    results.append(result)
                
                thread = threading.Thread(target=transcribe, args=(i,), name=f"Thread-{i}")
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # All should complete
            assert len(results) == 5
            
            # Model should be shared (loaded only once)
            assert mock_model_class.call_count == 1
    
    def test_model_switching_during_transcription(self, mock_service):
        """Test loading different models while transcription is in progress."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            models = {}
            
            def create_model(model_name, *args, **kwargs):
                model = MagicMock()
                
                def slow_transcribe(*args, **kwargs):
                    time.sleep(0.5)  # Simulate slow transcription
                    return iter([MockSegment(0, 1, f"Model: {model_name}")]), MockInfo()
                
                model.transcribe.side_effect = slow_transcribe
                models[model_name] = model
                return model
            
            mock_model_class.side_effect = create_model
            
            results = []
            
            def transcribe_with_model(model_name):
                result = mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model=model_name,
                    language="en",
                    vad_filter=True
                )
                results.append((model_name, result))
            
            # Start transcriptions with different models concurrently
            threads = []
            for model in ['tiny', 'base', 'small']:
                thread = threading.Thread(target=transcribe_with_model, args=(model,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # All should complete
            assert len(results) == 3
            
            # Each model should be loaded
            assert len(models) == 3
            
            # Results should match their models
            for model_name, result in results:
                assert f"Model: {model_name}" in result['text']
    
    def test_cache_cleanup_during_use(self, mock_service):
        """Test behavior when cache is cleared while model is in use."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            transcription_started = threading.Event()
            transcription_can_continue = threading.Event()
            
            def slow_transcribe(*args, **kwargs):
                transcription_started.set()
                transcription_can_continue.wait()
                return iter([MockSegment(0, 1, "Test")]), MockInfo()
            
            mock_model.transcribe.side_effect = slow_transcribe
            mock_model_class.return_value = mock_model
            
            # Start transcription in thread
            result_container = []
            
            def transcribe():
                result = mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model="base",
                    language="en",
                    vad_filter=True
                )
                result_container.append(result)
            
            thread = threading.Thread(target=transcribe)
            thread.start()
            
            # Wait for transcription to start
            transcription_started.wait()
            
            # Clear cache while transcription is in progress
            mock_service._model_cache.clear()
            
            # Let transcription continue
            transcription_can_continue.set()
            thread.join()
            
            # Should complete successfully
            assert len(result_container) == 1
            assert result_container[0]['text'] == "Test"


class TestFasterWhisperMemoryManagement:
    """Test memory management and resource cleanup."""
    
    def test_model_memory_cleanup(self, mock_service):
        """Test that models are properly cleaned up when removed from cache."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_models = []
            
            def create_model(*args, **kwargs):
                model = MagicMock()
                model.transcribe.return_value = (
                    iter([MockSegment(0, 1, "Test")]),
                    MockInfo()
                )
                mock_models.append(model)
                return model
            
            mock_model_class.side_effect = create_model
            
            # Load multiple models
            for i, model_name in enumerate(['tiny', 'base', 'small']):
                mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model=model_name,
                    language="en",
                    vad_filter=True
                )
            
            # Should have 3 models in cache
            assert len(mock_service._model_cache) == 3
            assert len(mock_models) == 3
            
            # Clear cache
            mock_service._model_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Models should be available for cleanup
            # (In real implementation, this would free GPU/CPU memory)
    
    def test_generator_cleanup_on_exception(self, mock_service):
        """Test that segment generators are properly cleaned up on exceptions."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            
            generator_closed = False
            
            def segment_generator():
                nonlocal generator_closed
                try:
                    yield MockSegment(0, 1, "First")
                    yield MockSegment(1, 2, "Second")
                    raise RuntimeError("Generator error")
                finally:
                    generator_closed = True
            
            mock_model.transcribe.return_value = (
                segment_generator(),
                MockInfo()
            )
            mock_model_class.return_value = mock_model
            
            with pytest.raises(TranscriptionError):
                mock_service._transcribe_with_faster_whisper(
                    audio_path="dummy.wav",
                    model="base",
                    language="en",
                    vad_filter=True
                )
            
            # Generator should be properly closed
            assert generator_closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])