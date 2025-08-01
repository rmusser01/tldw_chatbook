# test_property_based.py
"""
Property-based tests for the audio recording and dictation system.
Uses hypothesis to test invariants and edge cases.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays
from unittest.mock import Mock, patch
import wave
import tempfile
from pathlib import Path

from tldw_chatbook.Audio import (
    AudioRecordingService,
    LiveDictationService,
    DictationResult,
    DictationState,
    AudioRecordingError
)
from tldw_chatbook.Widgets.voice_input_widget import VoiceInputWidget, VoiceInputMessage


class TestAudioRecordingProperties:
    """Property-based tests for AudioRecordingService."""
    
    @given(
        sample_rate=st.sampled_from([8000, 16000, 22050, 44100, 48000]),
        channels=st.integers(min_value=1, max_value=8),
        chunk_size=st.sampled_from([256, 512, 1024, 2048, 4096])
    )
    def test_initialization_invariants(self, sample_rate, channels, chunk_size):
        """Test that initialization maintains invariants for any valid config."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            service = AudioRecordingService(
                sample_rate=sample_rate,
                channels=channels,
                chunk_size=chunk_size
            )
            
            # Invariants
            assert service.sample_rate == sample_rate
            assert service.channels == channels
            assert service.chunk_size == chunk_size
            assert service.is_recording is False
            assert service.audio_buffer == []
            assert service.audio_queue.empty()
    
    @given(
        audio_data=arrays(
            dtype=np.int16,
            shape=st.tuples(st.integers(min_value=100, max_value=10000)),
            elements=st.integers(min_value=-32768, max_value=32767)
        )
    )
    def test_audio_buffer_accumulation(self, audio_data):
        """Test that audio buffer correctly accumulates any valid audio data."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            service = AudioRecordingService()
            
            # Convert to bytes
            audio_bytes = audio_data.tobytes()
            
            # Add to buffer
            service._handle_audio_chunk(audio_bytes)
            
            # Verify accumulation
            assert len(service.audio_buffer) == 1
            assert service.audio_buffer[0] == audio_bytes
            
            # Add more
            service._handle_audio_chunk(audio_bytes)
            assert len(service.audio_buffer) == 2
            
            # Test concatenation
            combined = b''.join(service.audio_buffer)
            assert len(combined) == len(audio_bytes) * 2
    
    @given(
        num_devices=st.integers(min_value=0, max_value=10),
        default_device=st.integers(min_value=0, max_value=9)
    )
    def test_device_selection_properties(self, num_devices, default_device):
        """Test device selection with various device configurations."""
        assume(default_device < num_devices or num_devices == 0)
        
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            service = AudioRecordingService()
            
            # Mock devices
            devices = [
                {'id': i, 'name': f'Device {i}', 'is_default': i == default_device}
                for i in range(num_devices)
            ]
            
            with patch.object(service, 'get_audio_devices', return_value=devices):
                # Test setting each device
                for device_id in range(num_devices):
                    result = service.set_device(device_id)
                    assert result is True
                    assert service.current_device_id == device_id
                
                # Test invalid device
                if num_devices > 0:
                    result = service.set_device(num_devices + 1)
                    assert result is False
    
    @given(
        audio_chunks=st.lists(
            arrays(
                dtype=np.int16,
                shape=(1024,),
                elements=st.integers(min_value=-32768, max_value=32767)
            ),
            min_size=1,
            max_size=10
        )
    )
    def test_audio_level_calculation_properties(self, audio_chunks):
        """Test audio level calculation maintains expected properties."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            service = AudioRecordingService()
            
            for chunk in audio_chunks:
                # Add to queue
                service.audio_queue.put(chunk.tobytes())
                
                # Calculate level
                level = service.get_audio_level()
                
                # Properties
                assert 0.0 <= level <= 1.0
                
                # If all zeros, level should be 0
                if np.all(chunk == 0):
                    assert level == 0.0
                
                # If max values, level should be close to 1
                if np.any(np.abs(chunk) == 32767):
                    assert level > 0.5
    
    @given(
        file_path=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=48),
            min_size=1,
            max_size=50
        ).map(lambda s: f"/tmp/test_{s}.wav"),
        audio_data=arrays(
            dtype=np.int16,
            shape=st.tuples(st.integers(min_value=1000, max_value=48000)),
            elements=st.integers(min_value=-32768, max_value=32767)
        )
    )
    @settings(max_examples=10)  # Limit file operations
    def test_save_audio_file_properties(self, file_path, audio_data):
        """Test audio file saving maintains WAV format properties."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            service = AudioRecordingService()
            
            audio_bytes = audio_data.tobytes()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                full_path = Path(tmpdir) / Path(file_path).name
                
                # Save audio
                service._save_audio_file(audio_bytes, str(full_path))
                
                # Verify file properties
                assert full_path.exists()
                
                with wave.open(str(full_path), 'rb') as wf:
                    # WAV properties
                    assert wf.getnchannels() == service.channels
                    assert wf.getsampwidth() == 2  # 16-bit
                    assert wf.getframerate() == service.sample_rate
                    
                    # Data integrity
                    read_data = wf.readframes(wf.getnframes())
                    assert len(read_data) == len(audio_bytes)


class TestDictationServiceProperties:
    """Property-based tests for LiveDictationService."""
    
    @given(
        language=st.sampled_from(['en', 'es', 'fr', 'de', 'ja', 'zh']),
        enable_punctuation=st.booleans(),
        enable_commands=st.booleans(),
        silence_threshold=st.floats(min_value=0.1, max_value=5.0)
    )
    def test_dictation_initialization_properties(self, language, enable_punctuation, 
                                                enable_commands, silence_threshold):
        """Test dictation service initialization with various configurations."""
        with patch('tldw_chatbook.Audio.dictation_service.AudioRecordingService'):
            with patch('tldw_chatbook.Audio.dictation_service.TranscriptionService'):
                service = LiveDictationService(
                    language=language,
                    enable_punctuation=enable_punctuation,
                    enable_commands=enable_commands,
                    silence_threshold=silence_threshold
                )
                
                # Invariants
                assert service.language == language
                assert service.enable_punctuation == enable_punctuation
                assert service.enable_commands == enable_commands
                assert service.silence_threshold == silence_threshold
                assert service.state == DictationState.IDLE
                assert service.transcript_segments == []
    
    @given(
        transcripts=st.lists(
            st.text(min_size=1, max_size=100),
            min_size=0,
            max_size=10
        )
    )
    def test_transcript_accumulation_properties(self, transcripts):
        """Test transcript accumulation maintains expected properties."""
        with patch('tldw_chatbook.Audio.dictation_service.AudioRecordingService'):
            with patch('tldw_chatbook.Audio.dictation_service.TranscriptionService'):
                service = LiveDictationService()
                service.start_time = 0  # Fixed start time
                
                # Add transcripts
                for i, text in enumerate(transcripts):
                    service._handle_final_transcript(text)
                
                # Properties
                assert len(service.transcript_segments) == len(transcripts)
                
                # Timestamps should be monotonic
                timestamps = [seg['timestamp'] for seg in service.transcript_segments]
                assert timestamps == sorted(timestamps)
                
                # Full transcript should be space-joined
                expected = ' '.join(transcripts) if transcripts else ""
                assert service.get_full_transcript().strip() == expected.strip()
    
    @given(
        states=st.lists(
            st.sampled_from([
                DictationState.IDLE,
                DictationState.STARTING,
                DictationState.LISTENING,
                DictationState.PROCESSING,
                DictationState.PAUSED,
                DictationState.STOPPING,
                DictationState.ERROR
            ]),
            min_size=1,
            max_size=20
        )
    )
    def test_state_transition_properties(self, states):
        """Test state transitions maintain valid properties."""
        with patch('tldw_chatbook.Audio.dictation_service.AudioRecordingService'):
            with patch('tldw_chatbook.Audio.dictation_service.TranscriptionService'):
                service = LiveDictationService()
                
                valid_transitions = {
                    DictationState.IDLE: [DictationState.LISTENING, DictationState.ERROR],
                    DictationState.LISTENING: [DictationState.PAUSED, DictationState.STOPPING, 
                                              DictationState.ERROR, DictationState.IDLE],
                    DictationState.PAUSED: [DictationState.LISTENING, DictationState.STOPPING, 
                                           DictationState.ERROR, DictationState.IDLE],
                    DictationState.STOPPING: [DictationState.IDLE, DictationState.ERROR],
                    DictationState.ERROR: [DictationState.IDLE]
                }
                
                for state in states:
                    old_state = service.state
                    
                    # Only set valid transitions
                    if state in valid_transitions.get(old_state, []) or old_state == state:
                        service.state = state
                        assert service.state == state
    
    @given(
        text=st.text(min_size=1, max_size=200),
        enable_punctuation=st.booleans()
    )
    def test_punctuation_properties(self, text, enable_punctuation):
        """Test automatic punctuation maintains text properties."""
        with patch('tldw_chatbook.Audio.dictation_service.AudioRecordingService'):
            with patch('tldw_chatbook.Audio.dictation_service.TranscriptionService'):
                service = LiveDictationService(enable_punctuation=enable_punctuation)
                
                result = service._add_punctuation(text)
                
                # Properties
                if enable_punctuation and text and text[-1] not in '.!?':
                    assert result.endswith('.')
                
                # Should capitalize first letter
                if enable_punctuation and text:
                    assert result[0].isupper() or not result[0].isalpha()
                
                # Original text should be preserved (minus casing/punctuation)
                assert text.lower().rstrip('.!?') in result.lower()
    
    @given(
        duration=st.floats(min_value=0.0, max_value=3600.0),
        num_segments=st.integers(min_value=0, max_value=100),
        audio_size=st.integers(min_value=0, max_value=1000000)
    )
    def test_dictation_result_properties(self, duration, num_segments, audio_size):
        """Test DictationResult maintains expected properties."""
        segments = [
            {'text': f'Segment {i}', 'timestamp': i * 0.5}
            for i in range(num_segments)
        ]
        
        transcript = ' '.join(seg['text'] for seg in segments)
        audio_data = b'\x00' * audio_size if audio_size > 0 else None
        
        result = DictationResult(
            transcript=transcript,
            segments=segments,
            duration=duration,
            audio_data=audio_data
        )
        
        # Properties
        assert result.duration >= 0
        assert len(result.segments) == num_segments
        assert result.word_count == len(transcript.split()) if transcript else 0
        
        if audio_data:
            assert len(result.audio_data) == audio_size


class TestVoiceInputWidgetProperties:
    """Property-based tests for VoiceInputWidget."""
    
    @given(
        messages=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=100),
                st.booleans()
            ),
            min_size=0,
            max_size=10
        )
    )
    def test_voice_input_message_properties(self, messages):
        """Test VoiceInputMessage creation and properties."""
        for text, is_final in messages:
            msg = VoiceInputMessage(text, is_final)
            
            # Properties
            assert msg.text == text
            assert msg.is_final == is_final
            assert hasattr(msg, 'bubble')  # Should be a Message subclass
    
    @given(
        show_device_selector=st.booleans(),
        show_transcript_preview=st.booleans(),
        placeholder=st.text(max_size=50)
    )
    def test_widget_configuration_properties(self, show_device_selector, 
                                            show_transcript_preview, placeholder):
        """Test widget configuration maintains properties."""
        with patch('tldw_chatbook.Widgets.voice_input_widget.LiveDictationService'):
            widget = VoiceInputWidget(
                show_device_selector=show_device_selector,
                show_transcript_preview=show_transcript_preview,
                placeholder=placeholder or "default"
            )
            
            # Properties
            assert widget.show_device_selector == show_device_selector
            assert widget.show_transcript_preview == show_transcript_preview
            if placeholder:
                assert widget.placeholder == placeholder
    
    @given(
        audio_levels=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=20
        )
    )
    def test_audio_level_display_properties(self, audio_levels):
        """Test audio level visualization maintains properties."""
        with patch('tldw_chatbook.Widgets.voice_input_widget.LiveDictationService'):
            widget = VoiceInputWidget()
            
            for level in audio_levels:
                widget._update_level_display(level)
                
                # Properties
                assert 0.0 <= widget.audio_level <= 1.0
                assert widget.audio_level == level
    
    @given(
        states=st.sampled_from([
            DictationState.IDLE,
            DictationState.STARTING,
            DictationState.LISTENING,
            DictationState.PROCESSING,
            DictationState.PAUSED,
            DictationState.STOPPING,
            DictationState.ERROR
        ])
    )
    def test_widget_state_display_properties(self, states):
        """Test widget state display maintains consistency."""
        with patch('tldw_chatbook.Widgets.voice_input_widget.LiveDictationService'):
            widget = VoiceInputWidget()
            
            widget.state = states
            widget._update_ui_state()
            
            # State-specific properties
            if states == DictationState.IDLE:
                assert not widget.is_voice_recording
            elif states == DictationState.LISTENING:
                assert widget.is_voice_recording
            elif states == DictationState.ERROR:
                assert widget.error_message != ""


class TestSystemIntegrationProperties:
    """Property-based tests for system integration."""
    
    @given(
        num_chunks=st.integers(min_value=1, max_value=50),
        chunk_sizes=st.lists(
            st.integers(min_value=100, max_value=4096),
            min_size=1,
            max_size=50
        )
    )
    def test_audio_pipeline_properties(self, num_chunks, chunk_sizes):
        """Test audio pipeline maintains data integrity."""
        assume(len(chunk_sizes) >= num_chunks)
        
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.dictation_service.TranscriptionService'):
                recording_service = AudioRecordingService()
                dictation_service = LiveDictationService()
                
                total_bytes = 0
                for i in range(num_chunks):
                    chunk_size = chunk_sizes[i % len(chunk_sizes)]
                    chunk = np.random.randint(-32768, 32767, chunk_size, dtype=np.int16)
                    chunk_bytes = chunk.tobytes()
                    
                    # Process through pipeline
                    recording_service._handle_audio_chunk(chunk_bytes)
                    total_bytes += len(chunk_bytes)
                
                # Properties
                assert len(recording_service.audio_buffer) == num_chunks
                total_buffered = sum(len(chunk) for chunk in recording_service.audio_buffer)
                assert total_buffered == total_bytes
    
    @given(
        error_rate=st.floats(min_value=0.0, max_value=0.5),
        num_operations=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=10)
    def test_error_resilience_properties(self, error_rate, num_operations):
        """Test system resilience to errors."""
        with patch('tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE', True):
            with patch('tldw_chatbook.Audio.dictation_service.TranscriptionService'):
                service = LiveDictationService()
                
                errors_encountered = 0
                successful_ops = 0
                
                for i in range(num_operations):
                    if np.random.random() < error_rate:
                        # Simulate error
                        try:
                            service._notify_error(Exception("Test error"))
                            errors_encountered += 1
                        except:
                            pass
                    else:
                        # Normal operation
                        service._handle_partial_transcript(f"Text {i}")
                        successful_ops += 1
                
                # System should remain functional
                assert service.get_full_transcript() != ""
                assert successful_ops > 0
                
                # Error rate should roughly match
                if num_operations > 20:
                    actual_error_rate = errors_encountered / num_operations
                    assert abs(actual_error_rate - error_rate) < 0.2