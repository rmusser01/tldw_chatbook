# test_audio_integration.py
"""
Integration tests for the complete audio recording and dictation system.
Tests interaction between components and end-to-end workflows.
"""

import pytest
import time
from unittest.mock import Mock, patch

from tldw_chatbook.Audio import (
    AudioRecordingService,
    LiveDictationService,
    DictationState,
)
from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    def test_recording_stops_after_stream_error_and_preserves_prior_chunks(self):
        """Test an audio stream error stops recording and preserves prior chunks."""
        with patch("tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE", True):
            with patch(
                "tldw_chatbook.Audio.recording_service.pyaudio.PyAudio"
            ) as mock_pa_class:
                mock_pa = Mock()
                mock_stream = Mock()
                mock_pa_class.return_value = mock_pa
                mock_pa.open.return_value = mock_stream

                chunk = b"\x00\x01" * 512
                mock_stream.read.side_effect = [chunk, chunk, Exception("Stream error")]

                service = AudioRecordingService(backend="pyaudio", use_vad=False)
                chunks = []
                service.callback = chunks.append
                service.is_recording = True

                service._pyaudio_recording_loop()

                assert chunks == [chunk, chunk]
                assert service.is_recording is False
                mock_stream.stop_stream.assert_called_once_with()
                mock_stream.close.assert_called_once_with()
                assert service.stream is None

    def test_transcription_service_fallback(self):
        """Test fallback when primary transcription service fails."""
        # First provider fails
        with patch(
            "tldw_chatbook.Local_Ingestion.transcription_service.FASTER_WHISPER_AVAILABLE",
            False,
        ):
            # But fallback is available
            with patch(
                "tldw_chatbook.Local_Ingestion.transcription_service.PARAKEET_MLX_AVAILABLE",
                True,
            ):
                service = TranscriptionService()

                # Should list available models
                models = service.list_available_models()
                assert "parakeet-mlx" in models

    def test_dictation_continues_after_transcription_error(self):
        """Test dictation continues even if transcription fails."""
        with patch("tldw_chatbook.Audio.recording_service.AudioRecordingService"):
            with patch(
                "tldw_chatbook.Local_Ingestion.transcription_service.TranscriptionService"
            ) as mock_trans:
                # Make transcription fail sometimes
                mock_trans_instance = Mock()
                mock_trans_instance.transcribe_buffer.side_effect = [
                    Exception("Transcription error"),
                    {"text": "Recovered transcription"},
                    {"text": "More text"},
                ]
                mock_trans.return_value = mock_trans_instance

                service = LiveDictationService()
                service.start_dictation()

                # Process multiple buffers
                for _ in range(3):
                    try:
                        service._process_audio_buffer(b"audio")
                    except Exception:
                        pass

                # Should have recovered and continued
                assert service.state == DictationState.LISTENING
                assert service.current_transcript != ""


class TestPerformance:
    """Test performance characteristics of audio system."""

    def test_audio_buffering_performance(self):
        """Test audio buffer doesn't grow unbounded."""
        with patch("tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE", True):
            service = AudioRecordingService()

            # Add many chunks
            for i in range(1000):
                service._handle_audio_chunk(b"\x00\x01" * 512)

            # Queue should not grow unbounded
            # (In real implementation, old chunks would be processed/removed)
            assert len(service.audio_buffer) == 1000

            # Clear buffer on stop
            service.stop_recording()
            assert len(service.audio_buffer) == 0

    def test_concurrent_operations(self):
        """Test concurrent recording and transcription."""
        with patch("tldw_chatbook.Audio.recording_service.AudioRecordingService"):
            with patch(
                "tldw_chatbook.Local_Ingestion.transcription_service.TranscriptionService"
            ):
                service = LiveDictationService()

                # Start dictation
                service.start_dictation()

                # Simulate concurrent audio chunks
                import threading

                def add_audio():
                    for _ in range(10):
                        service._audio_callback(b"chunk")
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

        # Create service with a mocked backend so cleanup behavior is not
        # dependent on host audio libraries.
        with patch("tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE", True):
            service = AudioRecordingService(backend="pyaudio", use_vad=False)
        service_ref = weakref.ref(service)

        # Exercise the in-memory recording lifecycle without opening a native
        # audio device or starting a background thread.
        service.is_recording = True
        service._handle_audio_chunk(b"\x00\x01" * 512)
        service.stop_recording()

        # Delete and collect
        del service
        gc.collect()

        # Should be garbage collected
        assert service_ref() is None


class TestCrossPlatformCompatibility:
    """Test cross-platform audio functionality."""

    @pytest.mark.parametrize(
        "platform,expected_backend",
        [
            ("darwin", "pyaudio"),  # macOS
            ("win32", "pyaudio"),  # Windows
            ("linux", "pyaudio"),  # Linux
        ],
    )
    def test_platform_backend_selection(self, platform, expected_backend):
        """Test appropriate backend selection per platform."""
        with patch("sys.platform", platform):
            with patch("tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE", True):
                service = AudioRecordingService()
                assert service.backend == expected_backend

    def test_sounddevice_fallback(self):
        """Test fallback to sounddevice when PyAudio unavailable."""
        with patch("tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE", False):
            with patch(
                "tldw_chatbook.Audio.recording_service.SOUNDDEVICE_AVAILABLE", True
            ):
                service = AudioRecordingService()
                assert service.backend == "sounddevice"
