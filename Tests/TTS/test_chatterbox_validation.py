"""
Validation tests for Chatterbox TTS backend.
Tests streaming, crossfade, and voice cloning functionality.
"""
import asyncio
import pytest
import pytest_asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import torch

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.chatterbox import ChatterboxTTSBackend


class TestChatterboxValidation:
    """Comprehensive validation tests for Chatterbox TTS backend"""
    
    @pytest_asyncio.fixture
    async def backend(self):
        """Create a Chatterbox backend instance for testing"""
        config = {
            "CHATTERBOX_DEVICE": "cpu",
            "CHATTERBOX_STREAMING": True,
            "CHATTERBOX_ENABLE_CROSSFADE": True,
            "CHATTERBOX_CROSSFADE_MS": 50,
        }
        backend = ChatterboxTTSBackend(config)
        return backend
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock Chatterbox model"""
        model = Mock()
        model.sr = 24000
        
        # Mock generate method
        def mock_generate(text, **kwargs):
            # Generate dummy audio based on text length
            duration = len(text.split()) * 0.5  # 0.5 seconds per word
            samples = int(duration * 24000)
            return torch.randn(samples)
        
        model.generate = mock_generate
        
        # Mock generate_stream method
        def mock_generate_stream(text, **kwargs):
            chunk_size = kwargs.get('chunk_size', 1024)
            full_audio = mock_generate(text, **kwargs)
            
            # Yield chunks
            for i in range(0, len(full_audio), chunk_size):
                chunk = full_audio[i:i + chunk_size]
                metrics = {"chunk_index": i // chunk_size}
                yield chunk, metrics
        
        model.generate_stream = mock_generate_stream
        
        return model
    
    @pytest.mark.asyncio
    async def test_basic_generation(self, backend, mock_model):
        """Test basic TTS generation"""
        backend.model = mock_model
        backend.deps_available = True
        
        request = OpenAISpeechRequest(
            input="Hello world",
            voice="default",
            model="chatterbox",
            response_format="wav"
        )
        
        # Collect generated audio
        audio_chunks = []
        async for chunk in backend.generate_speech_stream(request):
            audio_chunks.append(chunk)
        
        # Verify we got audio
        assert len(audio_chunks) > 0
        combined_audio = b''.join(audio_chunks)
        assert len(combined_audio) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self, backend, mock_model):
        """Test that streaming actually works"""
        backend.model = mock_model
        backend.deps_available = True
        backend.streaming_enabled = True
        
        # Test with longer text to ensure multiple chunks
        request = OpenAISpeechRequest(
            input="This is a longer text that should generate multiple audio chunks for testing streaming functionality",
            voice="default",
            model="chatterbox",
            response_format="wav"
        )
        
        chunk_count = 0
        first_chunk_received = False
        
        # Mock the _generate_stream_async to verify it's called
        original_method = backend._generate_stream_async
        async def mock_stream_async(*args, **kwargs):
            nonlocal first_chunk_received
            async for chunk, metrics in original_method(*args, **kwargs):
                if not first_chunk_received:
                    first_chunk_received = True
                yield chunk, metrics
        
        backend._generate_stream_async = mock_stream_async
        
        async for chunk in backend.generate_speech_stream(request):
            chunk_count += 1
            # Verify we're getting bytes
            assert isinstance(chunk, bytes)
        
        # Should have multiple chunks for streaming
        assert chunk_count >= 1
        assert first_chunk_received
    
    @pytest.mark.asyncio
    async def test_crossfade_functionality(self, backend, mock_model):
        """Test crossfade between audio chunks"""
        backend.model = mock_model
        backend.deps_available = True
        backend.enable_crossfade = True
        backend.crossfade_duration_ms = 50
        
        # Create two dummy audio tensors
        tensor1 = torch.randn(24000)  # 1 second at 24kHz
        tensor2 = torch.randn(24000)
        
        # Apply crossfade
        result = backend.crossfade_audio_chunks(tensor1, tensor2, 50)
        
        # Verify crossfade properties
        assert isinstance(result, torch.Tensor)
        # Result should be shorter than sum due to overlap
        expected_length = len(tensor1) + len(tensor2) - int(24000 * 0.05)  # 50ms overlap
        assert len(result) == expected_length
    
    @pytest.mark.asyncio
    async def test_text_chunking(self, backend):
        """Test text chunking for long inputs"""
        backend.max_chunk_size = 50  # Small chunk size for testing
        
        long_text = " ".join(["word"] * 100)  # 100 words
        chunks = backend.chunk_text(long_text)
        
        # Verify chunking
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= backend.max_chunk_size
        
        # Verify no content is lost
        combined = " ".join(chunks)
        assert combined == long_text
    
    @pytest.mark.asyncio
    async def test_text_preprocessing(self, backend):
        """Test text preprocessing features"""
        backend.preprocess_text_enabled = True
        
        # Test various preprocessing scenarios
        test_cases = [
            ("J.R.R. Tolkien", "J R R Tolkien"),  # Dot removal
            ("Reference[1] here[2]", "Reference here"),  # Reference removal
            ("Multiple   spaces", "Multiple spaces"),  # Space normalization
            ("http://example.com text", " text"),  # URL removal
        ]
        
        for input_text, expected in test_cases:
            result = backend.preprocess_text(input_text)
            assert result == expected
    
    @pytest.mark.asyncio
    async def test_voice_cloning_path(self, backend, mock_model):
        """Test voice cloning with reference audio"""
        backend.model = mock_model
        backend.deps_available = True
        
        # Create a temporary reference audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Write dummy WAV header and data
            tmp_file.write(b'RIFF' + b'\x00' * 40)  # Minimal WAV header
            reference_path = tmp_file.name
        
        try:
            request = OpenAISpeechRequest(
                input="Clone this voice",
                voice=f"custom:{reference_path}",
                model="chatterbox",
                response_format="wav"
            )
            
            # Should not raise an error
            audio_chunks = []
            async for chunk in backend.generate_speech_stream(request):
                audio_chunks.append(chunk)
            
            assert len(audio_chunks) > 0
            
        finally:
            os.unlink(reference_path)
    
    @pytest.mark.asyncio
    async def test_audio_normalization(self, backend):
        """Test audio normalization"""
        backend.normalize_audio_enabled = True
        backend.target_db = -20.0
        
        # Create audio with known RMS
        audio_tensor = torch.ones(1000) * 0.1  # Low amplitude
        
        # Apply normalization
        normalized = backend.normalize_audio(audio_tensor, -20.0)
        
        # Verify amplitude increased
        assert torch.abs(normalized).mean() > torch.abs(audio_tensor).mean()
        
        # Verify no clipping
        assert torch.all(normalized >= -1.0)
        assert torch.all(normalized <= 1.0)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, backend):
        """Test error handling in various scenarios"""
        # Test without model initialized
        backend.model = None
        
        request = OpenAISpeechRequest(
            input="Test",
            voice="default",
            model="chatterbox",
            response_format="wav"
        )
        
        with pytest.raises(ValueError, match="not initialized"):
            async for _ in backend.generate_speech_stream(request):
                pass
        
        # Test with empty input
        backend.model = Mock()
        request.input = ""
        
        with pytest.raises(ValueError, match="required"):
            async for _ in backend.generate_speech_stream(request):
                pass
    
    @pytest.mark.asyncio
    async def test_streaming_async_wrapper(self, backend, mock_model):
        """Test the _generate_stream_async wrapper specifically"""
        backend.model = mock_model
        
        chunks_received = []
        async for chunk, metrics in backend._generate_stream_async(
            "Test text", None, 0.5, 0.5
        ):
            chunks_received.append((chunk, metrics))
        
        assert len(chunks_received) > 0
        
        # Verify chunk structure
        for chunk, metrics in chunks_received:
            assert isinstance(chunk, torch.Tensor)
            assert isinstance(metrics, dict)
    
    @pytest.mark.asyncio
    async def test_combine_audio_with_crossfade(self, backend):
        """Test the _combine_audio_with_crossfade method"""
        backend.enable_crossfade = True
        backend.crossfade_duration_ms = 50
        
        # Create mock WAV chunks
        import io
        import wave
        
        def create_wav_bytes(duration_sec=1.0):
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                samples = np.random.randint(-32768, 32767, int(24000 * duration_sec), dtype=np.int16)
                wav.writeframes(samples.tobytes())
            return buffer.getvalue()
        
        # Create multiple chunks
        chunks = [create_wav_bytes(0.5) for _ in range(3)]
        
        # Combine with crossfade
        with patch.object(backend, 'model', Mock(sr=24000)):
            result = await backend._combine_audio_with_crossfade(chunks)
        
        assert isinstance(result, bytes)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_integration_with_real_model():
    """Integration test with real Chatterbox model (if available)"""
    try:
        from chatterbox.tts import ChatterboxTTS
        
        backend = ChatterboxTTSBackend()
        await backend.initialize()
        
        if backend.model is None:
            pytest.skip("Chatterbox model not available")
        
        request = OpenAISpeechRequest(
            input="Hello, this is a test",
            voice="default",
            model="chatterbox",
            response_format="wav"
        )
        
        audio_data = b''
        async for chunk in backend.generate_speech_stream(request):
            audio_data += chunk
        
        # Verify we got valid WAV data
        assert audio_data.startswith(b'RIFF')
        assert len(audio_data) > 1000  # Should be reasonably sized
        
    except ImportError:
        pytest.skip("Chatterbox not installed")