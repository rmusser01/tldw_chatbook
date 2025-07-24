# test_diarization_integration.py
"""
Integration tests for speaker diarization with transcription service.
"""

import os
import sys
import tempfile
import unittest
import wave
import struct
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService


class TestDiarizationIntegration(unittest.TestCase):
    """Integration tests for diarization with transcription."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = TranscriptionService()
        
    def create_test_audio(self, duration_seconds=3.0, num_speakers=2):
        """Create a test audio file with simulated speakers."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio_path = tmp_file.name
            
        sample_rate = 16000
        with wave.open(audio_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)   # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            
            # Simulate different speakers with different frequencies
            samples_per_speaker = int((duration_seconds * sample_rate) / num_speakers)
            
            for speaker in range(num_speakers):
                # Different frequency for each "speaker"
                frequency = 440 * (speaker + 1)  # 440Hz, 880Hz, etc.
                
                for i in range(samples_per_speaker):
                    # Generate sine wave
                    t = i / sample_rate
                    value = int(32767 * 0.5 * (1 + (i / samples_per_speaker) * 0.5) * 
                              (0.5 * (1 + speaker * 0.3)) * 
                              (0.8 + 0.2 * (i % 1000) / 1000))
                    if i % 2 == 0:
                        value = int(value * 0.7)
                    wav_file.writeframes(struct.pack('<h', value))
                
                # Add short silence between speakers
                if speaker < num_speakers - 1:
                    for _ in range(int(0.2 * sample_rate)):  # 0.2s silence
                        wav_file.writeframes(struct.pack('<h', 0))
        
        return audio_path
        
    def test_real_transcription_with_diarization(self):
        """Test actual transcription with diarization on real audio."""
        # Check if all dependencies are available
        if not self.service.is_diarization_available():
            self.skipTest("Diarization dependencies not available")
            
        # Create test audio
        audio_path = self.create_test_audio(duration_seconds=5.0, num_speakers=2)
        
        try:
            # Run actual transcription with diarization
            result = self.service.transcribe(
                audio_path=audio_path,
                provider='faster-whisper',
                model='tiny',  # Use tiny model for speed
                language='en',
                diarize=True
            )
            
            # Verify results
            self.assertIsNotNone(result)
            self.assertIn('text', result)
            self.assertIn('segments', result)
            
            # Check diarization was performed
            if result.get('diarization_performed', False):
                self.assertIn('num_speakers', result)
                self.assertIn('speakers', result)
                
                # Check segments have speaker info
                for segment in result['segments']:
                    self.assertIn('speaker_id', segment)
                    self.assertIn('speaker_label', segment)
                    
                # We created audio with 2 different patterns, should detect at least 1 speaker
                self.assertGreaterEqual(result['num_speakers'], 1)
            
        finally:
            # Clean up
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_transcription_without_diarization(self):
        """Test that transcription works normally when diarization is disabled."""
        # Create test audio
        audio_path = self.create_test_audio(duration_seconds=2.0)
        
        try:
            # Run transcription without diarization
            result = self.service.transcribe(
                audio_path=audio_path,
                provider='faster-whisper',
                model='tiny',
                language='en',
                diarize=False
            )
            
            # Verify results
            self.assertIsNotNone(result)
            self.assertIn('text', result)
            self.assertIn('segments', result)
            
            # Should not have diarization info
            self.assertNotIn('num_speakers', result)
            self.assertNotIn('speakers', result)
            
            # Segments should not have speaker info
            if result.get('segments'):
                for segment in result['segments']:
                    self.assertNotIn('speaker_id', segment)
                    self.assertNotIn('speaker_label', segment)
                    
        finally:
            # Clean up
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_diarization_with_known_speakers(self):
        """Test diarization when number of speakers is specified."""
        if not self.service.is_diarization_available():
            self.skipTest("Diarization dependencies not available")
            
        audio_path = self.create_test_audio(duration_seconds=4.0, num_speakers=3)
        
        try:
            # Run with specified number of speakers
            result = self.service.transcribe(
                audio_path=audio_path,
                provider='faster-whisper',
                model='tiny',
                diarize=True,
                num_speakers=3  # Specify expected speakers
            )
            
            if result.get('diarization_performed', False):
                # Should respect the specified number if possible
                # (may be less if not enough distinct speech patterns)
                self.assertLessEqual(result['num_speakers'], 3)
                
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def test_format_segments_with_speakers(self):
        """Test formatting segments with speaker information."""
        # Create mock segments with speaker info
        segments = [
            {
                'start': 0.0,
                'end': 2.0,
                'text': 'Hello there',
                'speaker_id': 0,
                'speaker_label': 'SPEAKER_0'
            },
            {
                'start': 2.5,
                'end': 4.5,
                'text': 'Hi, how are you?',
                'speaker_id': 1,
                'speaker_label': 'SPEAKER_1'
            }
        ]
        
        # Test with speakers
        formatted = self.service.format_segments_with_timestamps(
            segments,
            include_timestamps=True,
            include_speakers=True
        )
        
        lines = formatted.split('\n')
        self.assertEqual(len(lines), 2)
        self.assertIn('[SPEAKER_0]', lines[0])
        self.assertIn('[SPEAKER_1]', lines[1])
        self.assertIn('Hello there', lines[0])
        self.assertIn('Hi, how are you?', lines[1])
        
        # Test without speakers
        formatted_no_speakers = self.service.format_segments_with_timestamps(
            segments,
            include_timestamps=True,
            include_speakers=False
        )
        
        lines_no_speakers = formatted_no_speakers.split('\n')
        for line in lines_no_speakers:
            self.assertNotIn('SPEAKER_', line)
    
    def test_is_diarization_available(self):
        """Test checking diarization availability."""
        is_available = self.service.is_diarization_available()
        self.assertIsInstance(is_available, bool)
        
        requirements = self.service.get_diarization_requirements()
        self.assertIsInstance(requirements, dict)
        self.assertIn('torch', requirements)
        self.assertIn('speechbrain', requirements)
        
        # If diarization is available, all requirements should be True
        if is_available:
            for dep, status in requirements.items():
                if dep in ['torch', 'speechbrain', 'sklearn']:  # Core requirements
                    self.assertTrue(status, f"Core dependency {dep} should be available")
    
    def test_progress_callback(self):
        """Test that progress callbacks work during diarization."""
        if not self.service.is_diarization_available():
            self.skipTest("Diarization dependencies not available")
            
        audio_path = self.create_test_audio(duration_seconds=3.0)
        progress_updates = []
        
        def progress_callback(progress, status, data):
            progress_updates.append({
                'progress': progress,
                'status': status,
                'data': data
            })
        
        try:
            result = self.service.transcribe(
                audio_path=audio_path,
                provider='faster-whisper',
                model='tiny',
                diarize=True,
                progress_callback=progress_callback
            )
            
            # Should have received progress updates
            self.assertGreater(len(progress_updates), 0)
            
            # Check for diarization-specific updates
            diarization_updates = [u for u in progress_updates 
                                 if 'diarization' in u['status'].lower() or
                                    (u['data'] and 'diarization_stage' in u.get('data', {}))]
            
            if result.get('diarization_performed', False):
                self.assertGreater(len(diarization_updates), 0)
                
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)


if __name__ == '__main__':
    unittest.main()