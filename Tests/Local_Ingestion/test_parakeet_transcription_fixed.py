"""
Tests for NVIDIA Parakeet transcription support - Fixed version.
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
    
    def test_parakeet_transcription_mock(self, service, sample_audio_file):
        """Test successful transcription with Parakeet using direct method mocking."""
        # Mock the internal transcription method
        with patch.object(service, '_transcribe_with_parakeet') as mock_transcribe:
            mock_transcribe.return_value = {
                'text': 'This is a test transcription.',
                'segments': [{'text': 'This is a test transcription.', 'start': 0.0, 'end': 2.0}],
                'timestamps': [(0.0, 2.0)],
                'provider': 'parakeet',
                'model': 'nvidia/parakeet-tdt-1.1b',
                'metadata': {}
            }
            
            with patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True):
                result = service.transcribe(sample_audio_file, provider='parakeet')
            
            assert result['text'] == 'This is a test transcription.'
            assert len(result['segments']) == 1
            assert result['provider'] == 'parakeet'