# test_diarization_service.py
"""
Unit tests for the speaker diarization service.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Local_Ingestion.diarization_service import (
    DiarizationService, 
    DiarizationError
)


class TestDiarizationService(unittest.TestCase):
    """Test cases for DiarizationService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = DiarizationService()
        
    def test_initialization(self):
        """Test service initialization."""
        self.assertIsNotNone(self.service)
        self.assertIsNotNone(self.service.config)
        self.assertEqual(self.service.config['vad_threshold'], 0.5)
        self.assertEqual(self.service.config['segment_duration'], 2.0)
        
    def test_availability_check(self):
        """Test checking if diarization is available."""
        # This will depend on whether dependencies are installed
        is_available = self.service.is_diarization_available()
        self.assertIsInstance(is_available, bool)
        
        # Check requirements
        requirements = self.service.get_requirements()
        self.assertIsInstance(requirements, dict)
        self.assertIn('torch', requirements)
        self.assertIn('speechbrain', requirements)
        
    @patch('tldw_chatbook.Local_Ingestion.diarization_service.TORCH_AVAILABLE', False)
    def test_unavailable_dependencies(self):
        """Test behavior when dependencies are missing."""
        service = DiarizationService()
        self.assertFalse(service.is_available)
        
        # Should raise error when trying to diarize
        with self.assertRaises(DiarizationError):
            service.diarize("dummy.wav")
    
    def test_device_detection(self):
        """Test device detection logic."""
        # Test auto detection
        self.service.config['embedding_device'] = 'auto'
        device = self.service._get_device()
        self.assertIn(device, ['cpu', 'cuda'])
        
        # Test explicit device
        self.service.config['embedding_device'] = 'cpu'
        device = self.service._get_device()
        self.assertEqual(device, 'cpu')
    
    def test_speech_detection(self):
        """Test speech detection with VAD."""
        # Skip if dependencies not available
        if not self.service.is_available:
            self.skipTest("Diarization dependencies not available")
        
        # Mock the VAD utils
        mock_get_timestamps = MagicMock()
        mock_get_timestamps.return_value = [
            {'start': 1600, 'end': 4800},  # 0.1s - 0.3s at 16kHz
            {'start': 8000, 'end': 12800}   # 0.5s - 0.8s at 16kHz
        ]
        
        # Mock VAD model and utils
        self.service._vad_model = MagicMock()
        self.service._vad_utils = {
            'get_speech_timestamps': mock_get_timestamps,
            'save_audio': MagicMock(),
            'read_audio': MagicMock(),
            'VADIterator': MagicMock(),
            'collect_chunks': MagicMock()
        }
        
        # Create a mock tensor for waveform
        mock_waveform = MagicMock()
        
        # Test speech detection
        segments = self.service._detect_speech(mock_waveform, 16000)
        
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0]['start'], 0.1, places=1)
        self.assertAlmostEqual(segments[0]['end'], 0.3, places=1)
        
    def test_segment_creation(self):
        """Test creating analysis segments."""
        # Create dummy waveform
        waveform = MagicMock()
        waveform.__getitem__ = lambda _, idx: MagicMock()
        
        speech_timestamps = [
            {'start': 0.0, 'end': 5.0},
            {'start': 6.0, 'end': 10.0}
        ]
        
        segments = self.service._create_segments(waveform, speech_timestamps, 16000)
        
        # Should create overlapping segments
        self.assertGreater(len(segments), 0)
        
        # Check segment properties
        for segment in segments:
            self.assertIn('start', segment)
            self.assertIn('end', segment)
            self.assertIn('waveform', segment)
            duration = segment['end'] - segment['start']
            self.assertLessEqual(duration, self.service.config['max_segment_duration'])
    
    def test_embedding_extraction_mock(self):
        """Test embedding extraction with mocked model."""
        # Skip if dependencies not available
        if not self.service.is_available:
            self.skipTest("Diarization dependencies not available")
        
        # Import torch if available
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not available for testing")
        
        # Mock the embedding model
        mock_model = MagicMock()
        # Return batch of embeddings matching the input batch size
        mock_model.encode_batch.return_value = MagicMock(
            cpu=lambda: MagicMock(
                numpy=lambda: np.random.randn(2, 192)  # Batch of 2 embeddings
            )
        )
        self.service._embedding_model = mock_model
        
        # Create test segments with proper torch tensors
        segments = [
            {'waveform': torch.randn(16000)},  # 1 second of audio at 16kHz
            {'waveform': torch.randn(16000)}
        ]
        
        # Extract embeddings
        embeddings = self.service._extract_embeddings(segments)
        
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 192)  # Expected embedding dimension
    
    def test_speaker_estimation(self):
        """Test estimating number of speakers."""
        # Create mock embeddings with clear clusters
        np.random.seed(42)
        
        # 2 speakers with 5 segments each
        embeddings1 = np.random.randn(5, 192) + np.array([1.0] * 192)
        embeddings2 = np.random.randn(5, 192) + np.array([-1.0] * 192)
        embeddings = np.vstack([embeddings1, embeddings2])
        
        # Normalize
        from sklearn.preprocessing import normalize
        embeddings = normalize(embeddings, axis=1, norm='l2')
        
        # Estimate speakers
        num_speakers = self.service._estimate_num_speakers(embeddings)
        
        # Should detect approximately 2 speakers
        self.assertGreaterEqual(num_speakers, 2)
        self.assertLessEqual(num_speakers, 3)
    
    def test_segment_merging(self):
        """Test merging consecutive segments from same speaker."""
        segments = [
            {'start': 0.0, 'end': 2.0, 'speaker_id': 0},
            {'start': 2.1, 'end': 4.0, 'speaker_id': 0},  # Should merge
            {'start': 4.5, 'end': 6.0, 'speaker_id': 1},  # Different speaker
            {'start': 6.2, 'end': 8.0, 'speaker_id': 1},  # Should merge
            {'start': 10.0, 'end': 12.0, 'speaker_id': 0} # Too far, won't merge
        ]
        
        merged = self.service._merge_segments(segments)
        
        # Should have 3 merged segments
        self.assertEqual(len(merged), 3)
        
        # First segment should be merged 0-4.0
        self.assertEqual(merged[0]['start'], 0.0)
        self.assertEqual(merged[0]['end'], 4.0)
        self.assertEqual(merged[0]['speaker_id'], 0)
        
        # Second segment should be merged 4.5-8.0
        self.assertEqual(merged[1]['start'], 4.5)
        self.assertEqual(merged[1]['end'], 8.0)
        self.assertEqual(merged[1]['speaker_id'], 1)
    
    def test_transcription_alignment(self):
        """Test aligning diarization with transcription segments."""
        diarization_segments = [
            {'start': 0.0, 'end': 5.0, 'speaker_id': 0},
            {'start': 5.0, 'end': 10.0, 'speaker_id': 1}
        ]
        
        transcription_segments = [
            {'start': 0.5, 'end': 2.5, 'text': 'Hello there'},
            {'start': 3.0, 'end': 4.5, 'text': 'How are you'},
            {'start': 5.5, 'end': 7.0, 'text': 'I am fine'},
            {'start': 8.0, 'end': 9.5, 'text': 'Thank you'}
        ]
        
        aligned = self.service._align_with_transcription(
            diarization_segments,
            transcription_segments
        )
        
        # All transcription segments should be aligned
        self.assertEqual(len(aligned), 4)
        
        # Check speaker assignments
        self.assertEqual(aligned[0]['speaker_id'], 0)  # First two from speaker 0
        self.assertEqual(aligned[1]['speaker_id'], 0)
        self.assertEqual(aligned[2]['speaker_id'], 1)  # Last two from speaker 1
        self.assertEqual(aligned[3]['speaker_id'], 1)
    
    def test_speaker_statistics(self):
        """Test calculating speaker statistics."""
        segments = [
            {'start': 0.0, 'end': 2.0, 'speaker_id': 0},
            {'start': 2.0, 'end': 5.0, 'speaker_id': 1},
            {'start': 5.0, 'end': 7.0, 'speaker_id': 0},
            {'start': 7.0, 'end': 8.0, 'speaker_id': 1}
        ]
        
        stats = self.service._calculate_speaker_stats(segments)
        
        self.assertEqual(len(stats), 2)
        
        # Check speaker 0 stats
        speaker0_stats = next(s for s in stats if s['speaker_id'] == 0)
        self.assertEqual(speaker0_stats['total_time'], 4.0)  # 2 + 2 seconds
        self.assertEqual(speaker0_stats['segment_count'], 2)
        
        # Check speaker 1 stats
        speaker1_stats = next(s for s in stats if s['speaker_id'] == 1)
        self.assertEqual(speaker1_stats['total_time'], 4.0)  # 3 + 1 seconds
        self.assertEqual(speaker1_stats['segment_count'], 2)
    
    @patch('tldw_chatbook.Local_Ingestion.diarization_service.DiarizationService._load_audio')
    @patch('tldw_chatbook.Local_Ingestion.diarization_service.DiarizationService._detect_speech')
    @patch('tldw_chatbook.Local_Ingestion.diarization_service.DiarizationService._create_segments')
    @patch('tldw_chatbook.Local_Ingestion.diarization_service.DiarizationService._extract_embeddings')
    @patch('tldw_chatbook.Local_Ingestion.diarization_service.DiarizationService._cluster_speakers')
    def test_full_diarization_pipeline(self, mock_cluster, mock_extract, 
                                     mock_create, mock_detect, mock_load):
        """Test the full diarization pipeline."""
        # Skip if dependencies not available
        if not self.service.is_available:
            self.skipTest("Diarization dependencies not available")
        
        # Mock all steps
        mock_waveform = MagicMock()
        mock_waveform.__len__ = lambda _: 160000  # 10 seconds at 16kHz
        mock_load.return_value = mock_waveform
        
        mock_detect.return_value = [
            {'start': 0.0, 'end': 5.0},
            {'start': 6.0, 'end': 10.0}
        ]
        
        mock_segments = [
            {'start': 0.0, 'end': 2.0, 'waveform': MagicMock()},
            {'start': 2.0, 'end': 4.0, 'waveform': MagicMock()},
            {'start': 6.0, 'end': 8.0, 'waveform': MagicMock()},
            {'start': 8.0, 'end': 10.0, 'waveform': MagicMock()}
        ]
        mock_create.return_value = mock_segments
        
        mock_extract.return_value = np.random.randn(4, 192)
        mock_cluster.return_value = np.array([0, 0, 1, 1])
        
        # Run diarization
        result = self.service.diarize("test.wav")
        
        # Check result structure
        self.assertIn('segments', result)
        self.assertIn('speakers', result)
        self.assertIn('num_speakers', result)
        self.assertIn('duration', result)
        self.assertIn('processing_time', result)
        
        self.assertEqual(result['num_speakers'], 2)
        self.assertEqual(len(result['segments']), 2)  # After merging
        
    def test_progress_callback(self):
        """Test progress reporting during diarization."""
        # Skip if dependencies not available
        if not self.service.is_available:
            self.skipTest("Diarization dependencies not available")
        
        progress_calls = []
        
        def progress_callback(progress, status, data):
            progress_calls.append({
                'progress': progress,
                'status': status,
                'data': data
            })
        
        # Mock the pipeline
        with patch.object(self.service, '_load_audio') as mock_load, \
             patch.object(self.service, '_detect_speech') as mock_detect, \
             patch.object(self.service, '_create_segments') as mock_create, \
             patch.object(self.service, '_extract_embeddings') as mock_extract, \
             patch.object(self.service, '_cluster_speakers') as mock_cluster:
            
            # Setup mocks
            mock_waveform = MagicMock()
            mock_waveform.__len__ = lambda _: 160000
            mock_load.return_value = mock_waveform
            mock_detect.return_value = [{'start': 0, 'end': 5}]
            mock_create.return_value = [{'start': 0, 'end': 2, 'waveform': MagicMock()}]
            mock_extract.return_value = np.random.randn(1, 192)
            mock_cluster.return_value = np.array([0])
            
            # Run with progress callback
            self.service.diarize("test.wav", progress_callback=progress_callback)
            
            # Check progress was reported
            self.assertGreater(len(progress_calls), 0)
            
            # Check progress values
            progress_values = [call['progress'] for call in progress_calls]
            self.assertEqual(progress_values[-1], 100)  # Should end at 100%


    def test_single_speaker_case(self):
        """Test handling of single speaker case."""
        # Skip if dependencies not available
        if not self.service.is_available:
            self.skipTest("Diarization dependencies not available")
        
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not available for testing")
        
        # Create embeddings that are very similar (single speaker)
        embeddings = np.random.randn(10, 192)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Test explicit num_speakers=1
        labels = self.service._cluster_speakers(embeddings, num_speakers=1)
        self.assertTrue(np.all(labels == 0))
        
        # Test automatic single speaker detection
        # Create very similar embeddings
        base_embedding = np.random.randn(192)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        similar_embeddings = np.array([base_embedding + 0.01 * np.random.randn(192) for _ in range(10)])
        similar_embeddings = similar_embeddings / np.linalg.norm(similar_embeddings, axis=1, keepdims=True)
        
        is_single = self.service._is_single_speaker(similar_embeddings)
        self.assertTrue(is_single)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Get a valid config and modify it to test validation
        service = DiarizationService()
        base_config = service.config.copy()
        
        # Test invalid vad_threshold
        with self.assertRaises(ValueError):
            invalid_config = base_config.copy()
            invalid_config['vad_threshold'] = 1.5
            service._validate_config(invalid_config)
        
        # Test invalid segment overlap
        with self.assertRaises(ValueError):
            invalid_config = base_config.copy()
            invalid_config['segment_overlap'] = 3.0  # Greater than segment_duration
            service._validate_config(invalid_config)
        
        # Test invalid min_speakers
        with self.assertRaises(ValueError):
            invalid_config = base_config.copy()
            invalid_config['min_speakers'] = 0
            service._validate_config(invalid_config)


if __name__ == '__main__':
    unittest.main()