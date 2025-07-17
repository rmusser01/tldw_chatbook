"""
Tests for NVIDIA Parakeet transcription support.
"""

import pytest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService, TranscriptionError


class TestParakeetTranscription:
    """Test suite for Parakeet transcription functionality."""
    
    @pytest.fixture
    def service(self):
        """Create a TranscriptionService instance."""
        return TranscriptionService()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary WAV file for testing."""
        import wave
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Create a simple WAV file
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                # Generate 1 second of silence
                wav.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
            yield f.name
        os.unlink(f.name)
    
    def test_parakeet_not_available(self, service, sample_audio_file):
        """Test that proper error is raised when NeMo is not available."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', False):
            with pytest.raises(ValueError) as exc_info:
                service.transcribe(sample_audio_file, provider='parakeet')
            assert "NeMo toolkit is not installed" in str(exc_info.value)
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    def test_parakeet_transcription_success(self, service, sample_audio_file):
        """Test successful transcription with Parakeet."""
        # Mock the internal _transcribe_with_parakeet method
        with patch.object(service, '_transcribe_with_parakeet') as mock_transcribe:
            expected_result = {
                'text': 'This is a test transcription.',
                'segments': [{'text': 'This is a test transcription.', 'start': 0.0, 'end': 2.0}],
                'timestamps': [(0.0, 2.0)],
                'provider': 'parakeet',
                'model': 'nvidia/parakeet-tdt-1.1b',
                'metadata': {'provider': 'parakeet', 'model': 'nvidia/parakeet-tdt-1.1b'}
            }
            mock_transcribe.return_value = expected_result
            
            # Perform transcription
            result = service.transcribe(sample_audio_file, provider='parakeet')
            
            # Verify results
            assert result['text'] == "This is a test transcription."
            assert len(result['segments']) == 1
            assert result['segments'][0]['text'] == "This is a test transcription."
            assert result['provider'] == 'parakeet'
            assert 'parakeet-tdt-1.1b' in result['model']
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    def test_parakeet_custom_model(self, service, sample_audio_file):
        """Test transcription with custom Parakeet model."""
        with patch.object(service, '_transcribe_with_parakeet') as mock_transcribe:
            expected_result = {
                'text': 'Custom model transcription.',
                'segments': [{'text': 'Custom model transcription.', 'start': 0.0, 'end': 2.0}],
                'timestamps': [(0.0, 2.0)],
                'provider': 'parakeet',
                'model': 'nvidia/parakeet-ctc-0.6b',
                'metadata': {'provider': 'parakeet', 'model': 'nvidia/parakeet-ctc-0.6b'}
            }
            mock_transcribe.return_value = expected_result
            
            # Perform transcription with custom model
            result = service.transcribe(
                sample_audio_file, 
                provider='parakeet',
                model='nvidia/parakeet-ctc-0.6b'
            )
            
            # Verify results
            assert result['model'] == 'nvidia/parakeet-ctc-0.6b'
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    def test_list_parakeet_models(self, service):
        """Test that Parakeet models are listed when NeMo is available."""
        models = service.list_available_models()
        
        assert 'parakeet' in models
        assert 'nvidia/parakeet-tdt-1.1b' in models['parakeet']
        assert 'nvidia/parakeet-rnnt-1.1b' in models['parakeet']
        assert 'nvidia/parakeet-ctc-1.1b' in models['parakeet']
        assert len(models['parakeet']) == 7  # 3 model types x 2 sizes + 1 v2 model
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', False)
    def test_list_models_without_nemo(self, service):
        """Test that Parakeet models are not listed when NeMo is unavailable."""
        models = service.list_available_models()
        assert 'parakeet' not in models
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.torch')
    def test_parakeet_cuda_device(self, mock_torch, service, sample_audio_file):
        """Test that model handles CUDA availability correctly."""
        # Test the transcription with CUDA available
        mock_torch.cuda.is_available.return_value = True
        
        with patch.object(service, '_transcribe_with_parakeet') as mock_transcribe:
            expected_result = {
                'text': 'CUDA test transcription.',
                'segments': [{'text': 'CUDA test transcription.', 'start': 0.0, 'end': 2.0}],
                'timestamps': [(0.0, 2.0)],
                'provider': 'parakeet',
                'model': 'nvidia/parakeet-tdt-1.1b',
                'metadata': {'device': 'cuda'}
            }
            mock_transcribe.return_value = expected_result
            
            result = service.transcribe(sample_audio_file, provider='parakeet')
            assert result['text'] == 'CUDA test transcription.'
    
    def test_parakeet_model_loading_error(self, service, sample_audio_file):
        """Test handling of model loading errors."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True):
            with patch.object(service, '_transcribe_with_parakeet') as mock_transcribe:
                mock_transcribe.side_effect = TranscriptionError("Failed to load model")
                
                with pytest.raises(TranscriptionError) as exc_info:
                    service.transcribe(sample_audio_file, provider='parakeet')
                assert "Failed to load model" in str(exc_info.value)
    
    def test_parakeet_transcription_error(self, service, sample_audio_file):
        """Test handling of transcription errors."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True):
            with patch.object(service, '_transcribe_with_parakeet') as mock_transcribe:
                # The internal method would wrap exceptions in TranscriptionError
                mock_transcribe.side_effect = TranscriptionError("Transcription failed")
                
                with pytest.raises(TranscriptionError) as exc_info:
                    service.transcribe(sample_audio_file, provider='parakeet')
                assert "Transcription failed" in str(exc_info.value)
    
    def test_parakeet_empty_transcription(self, service, sample_audio_file):
        """Test handling of empty transcription results."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True):
            with patch.object(service, '_transcribe_with_parakeet') as mock_transcribe:
                expected_result = {
                    'text': '',
                    'segments': [],
                    'timestamps': [],
                    'provider': 'parakeet',
                    'model': 'nvidia/parakeet-tdt-1.1b',
                    'metadata': {}
                }
                mock_transcribe.return_value = expected_result
                
                result = service.transcribe(sample_audio_file, provider='parakeet')
                assert result['text'] == ''
                assert len(result['segments']) == 0
    
    def test_parakeet_different_return_formats(self, service, sample_audio_file):
        """Test that different return formats are handled correctly."""
        with patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True):
            with patch.object(service, '_transcribe_with_parakeet') as mock_transcribe:
                # Test with timestamped words
                expected_result = {
                    'text': 'Hello world test.',
                    'segments': [
                        {'text': 'Hello', 'start': 0.0, 'end': 0.5},
                        {'text': 'world', 'start': 0.5, 'end': 1.0},
                        {'text': 'test.', 'start': 1.0, 'end': 1.5}
                    ],
                    'timestamps': [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)],
                    'provider': 'parakeet',
                    'model': 'nvidia/parakeet-tdt-1.1b',
                    'metadata': {},
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 0.5},
                        {'word': 'world', 'start': 0.5, 'end': 1.0},
                        {'word': 'test.', 'start': 1.0, 'end': 1.5}
                    ]
                }
                mock_transcribe.return_value = expected_result
                
                result = service.transcribe(
                    sample_audio_file, 
                    provider='parakeet',
                    return_format='words'
                )
                assert 'words' in result
                assert len(result['words']) == 3