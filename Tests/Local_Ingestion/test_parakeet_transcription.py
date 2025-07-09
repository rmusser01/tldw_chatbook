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
            with pytest.raises(TranscriptionError) as exc_info:
                service.transcribe(sample_audio_file, provider='parakeet')
            assert "NeMo toolkit not installed" in str(exc_info.value)
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.nemo_asr')
    def test_parakeet_transcription_success(self, mock_nemo, service, sample_audio_file):
        """Test successful transcription with Parakeet."""
        # Mock the ASR model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["This is a test transcription."]
        mock_nemo.models.ASRModel.from_pretrained.return_value = mock_model
        
        # Perform transcription
        result = service.transcribe(sample_audio_file, provider='parakeet')
        
        # Verify results
        assert result['text'] == "This is a test transcription."
        assert len(result['segments']) == 1
        assert result['segments'][0]['text'] == "This is a test transcription."
        assert result['provider'] == 'parakeet'
        assert 'parakeet-tdt-1.1b' in result['model']
        
        # Verify model was loaded with correct parameters
        mock_nemo.models.ASRModel.from_pretrained.assert_called_once_with(
            model_name='nvidia/parakeet-tdt-1.1b'
        )
        mock_model.change_attention_model.assert_called_once()
        mock_model.change_subsampling_conv_chunking_factor.assert_called_once()
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.nemo_asr')
    def test_parakeet_custom_model(self, mock_nemo, service, sample_audio_file):
        """Test transcription with custom Parakeet model."""
        # Mock the ASR model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["Custom model output."]
        mock_nemo.models.ASRModel.from_pretrained.return_value = mock_model
        
        # Perform transcription with custom model
        result = service.transcribe(
            sample_audio_file, 
            provider='parakeet',
            model='nvidia/parakeet-ctc-0.6b'
        )
        
        # Verify correct model was loaded
        mock_nemo.models.ASRModel.from_pretrained.assert_called_once_with(
            model_name='nvidia/parakeet-ctc-0.6b'
        )
        assert result['model'] == 'nvidia/parakeet-ctc-0.6b'
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    def test_list_parakeet_models(self, service):
        """Test that Parakeet models are listed when NeMo is available."""
        models = service.list_available_models()
        
        assert 'parakeet' in models
        assert 'nvidia/parakeet-tdt-1.1b' in models['parakeet']
        assert 'nvidia/parakeet-rnnt-1.1b' in models['parakeet']
        assert 'nvidia/parakeet-ctc-1.1b' in models['parakeet']
        assert len(models['parakeet']) == 6  # 3 model types x 2 sizes
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', False)
    def test_list_models_without_nemo(self, service):
        """Test that Parakeet models are not listed when NeMo is unavailable."""
        models = service.list_available_models()
        assert 'parakeet' not in models
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.nemo_asr')
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.torch')
    def test_parakeet_cuda_device(self, mock_torch, mock_nemo, service, sample_audio_file):
        """Test that model is moved to CUDA when available."""
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = True
        
        # Mock the ASR model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["Test"]
        mock_model.cuda.return_value = mock_model
        mock_nemo.models.ASRModel.from_pretrained.return_value = mock_model
        
        # Set device to CUDA in config
        service.config['device'] = 'cuda'
        
        # Perform transcription
        service.transcribe(sample_audio_file, provider='parakeet')
        
        # Verify model was moved to CUDA
        mock_model.cuda.assert_called_once()
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.nemo_asr')
    def test_parakeet_model_loading_error(self, mock_nemo, service, sample_audio_file):
        """Test proper error handling when model loading fails."""
        # Mock model loading failure
        mock_nemo.models.ASRModel.from_pretrained.side_effect = Exception("Model not found")
        
        with pytest.raises(TranscriptionError) as exc_info:
            service.transcribe(sample_audio_file, provider='parakeet')
        assert "Failed to load Parakeet model" in str(exc_info.value)
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.nemo_asr')
    def test_parakeet_transcription_error(self, mock_nemo, service, sample_audio_file):
        """Test proper error handling when transcription fails."""
        # Mock the ASR model
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        mock_nemo.models.ASRModel.from_pretrained.return_value = mock_model
        
        with pytest.raises(TranscriptionError) as exc_info:
            service.transcribe(sample_audio_file, provider='parakeet')
        assert "Parakeet transcription failed" in str(exc_info.value)
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.nemo_asr')
    def test_parakeet_empty_transcription(self, mock_nemo, service, sample_audio_file):
        """Test handling of empty transcription results."""
        # Mock the ASR model returning empty results
        mock_model = MagicMock()
        mock_model.transcribe.return_value = []
        mock_nemo.models.ASRModel.from_pretrained.return_value = mock_model
        
        with pytest.raises(TranscriptionError) as exc_info:
            service.transcribe(sample_audio_file, provider='parakeet')
        assert "No transcription produced" in str(exc_info.value)
    
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.NEMO_AVAILABLE', True)
    @patch('tldw_chatbook.Local_Ingestion.transcription_service.nemo_asr')
    def test_parakeet_different_return_formats(self, mock_nemo, service, sample_audio_file):
        """Test handling of different return formats from NeMo."""
        mock_model = MagicMock()
        mock_nemo.models.ASRModel.from_pretrained.return_value = mock_model
        
        # Test string return
        mock_model.transcribe.return_value = ["Direct string result"]
        result = service.transcribe(sample_audio_file, provider='parakeet')
        assert result['text'] == "Direct string result"
        
        # Reset model cache to test different format
        service._parakeet_model = None
        
        # Test object with text attribute
        mock_transcript = MagicMock()
        mock_transcript.text = "Object with text attribute"
        mock_model.transcribe.return_value = [mock_transcript]
        result = service.transcribe(sample_audio_file, provider='parakeet')
        assert result['text'] == "Object with text attribute"
        
        # Reset model cache again
        service._parakeet_model = None
        
        # Test nested list
        mock_model.transcribe.return_value = [["Nested list result"]]
        result = service.transcribe(sample_audio_file, provider='parakeet')
        assert result['text'] == "Nested list result"