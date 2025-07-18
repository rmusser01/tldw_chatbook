"""
Validation tests for Kokoro TTS backend.
Tests ONNX/PyTorch backends, streaming, voice mixing, and new features.
"""
import asyncio
import pytest
import pytest_asyncio
import tempfile
import os
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend, map_voice_to_kokoro


class TestKokoroValidation:
    """Comprehensive validation tests for Kokoro TTS backend"""
    
    @pytest_asyncio.fixture
    async def backend_onnx(self):
        """Create a Kokoro backend instance configured for ONNX"""
        config = {
            "KOKORO_USE_ONNX": True,
            "KOKORO_DEVICE": "cpu",
            "KOKORO_MAX_TOKENS": 500,
            "KOKORO_ENABLE_VOICE_MIXING": True,
            "KOKORO_TRACK_PERFORMANCE": True,
        }
        backend = KokoroTTSBackend(config)
        return backend
    
    @pytest_asyncio.fixture
    async def backend_pytorch(self):
        """Create a Kokoro backend instance configured for PyTorch"""
        config = {
            "KOKORO_USE_ONNX": False,
            "KOKORO_DEVICE": "cpu",
            "KOKORO_MAX_TOKENS": 500,
        }
        backend = KokoroTTSBackend(config)
        return backend
    
    @pytest.fixture
    def mock_kokoro_instance(self):
        """Create a mock kokoro_onnx instance"""
        instance = Mock()
        
        # Mock create_stream method
        async def mock_create_stream(text, voice="af_bella", speed=1.0, lang="en"):
            # Generate chunks based on text length
            words = text.split()
            chunk_size = 2048
            total_samples = len(words) * 12000  # ~0.5s per word at 24kHz
            
            for i in range(0, total_samples, chunk_size):
                chunk_samples = min(chunk_size, total_samples - i)
                samples = np.random.randn(chunk_samples).astype(np.float32) * 0.1
                yield samples, 24000
        
        instance.create_stream = mock_create_stream
        return instance
    
    @pytest.mark.asyncio
    async def test_onnx_streaming(self, backend_onnx, mock_kokoro_instance):
        """Test ONNX backend streaming functionality"""
        backend_onnx.kokoro_instance = mock_kokoro_instance
        backend_onnx.model_loaded = True
        
        request = OpenAISpeechRequest(
            input="Hello world from Kokoro",
            voice="af_bella",
            model="kokoro",
            response_format="pcm"
        )
        
        chunks_received = 0
        first_chunk_time = None
        start_time = time.time()
        
        async for chunk in backend_onnx.generate_speech_stream(request):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            chunks_received += 1
            assert isinstance(chunk, bytes)
            assert len(chunk) > 0
        
        # Verify streaming behavior
        assert chunks_received > 1  # Should receive multiple chunks
        assert first_chunk_time < 1.0  # First chunk should arrive quickly
    
    @pytest.mark.asyncio
    async def test_voice_mixing(self, backend_onnx, mock_kokoro_instance):
        """Test voice mixing functionality"""
        backend_onnx.kokoro_instance = mock_kokoro_instance
        backend_onnx.model_loaded = True
        backend_onnx.enable_voice_mixing = True
        
        # Test mixed voice format
        request = OpenAISpeechRequest(
            input="Mixed voice test",
            voice="af_bella:0.7,af_sarah:0.3",  # 70% bella, 30% sarah
            model="kokoro",
            response_format="pcm"
        )
        
        # Verify voice config parsing
        voice_config = backend_onnx._parse_voice_config(request.voice)
        assert voice_config['is_mixed'] == True
        assert len(voice_config['voices']) == 2
        assert voice_config['voices'][0] == ('af_bella', 0.7)
        assert voice_config['voices'][1] == ('af_sarah', 0.3)
        
        # Generate audio
        audio_data = b''
        async for chunk in backend_onnx.generate_speech_stream(request):
            audio_data += chunk
        
        assert len(audio_data) > 0
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self, backend_onnx, mock_kokoro_instance):
        """Test progress reporting functionality"""
        backend_onnx.kokoro_instance = mock_kokoro_instance
        backend_onnx.model_loaded = True
        
        progress_updates = []
        
        async def progress_callback(progress_info):
            progress_updates.append(progress_info)
        
        backend_onnx.progress_callback = progress_callback
        
        request = OpenAISpeechRequest(
            input="This is a test for progress tracking",
            voice="af_bella",
            model="kokoro",
            response_format="mp3"
        )
        
        async for chunk in backend_onnx.generate_speech_stream(request):
            pass
        
        # Verify progress updates
        assert len(progress_updates) > 0
        
        # Check first and last updates
        assert progress_updates[0]['progress'] >= 0.0
        assert progress_updates[-1]['progress'] == 1.0
        assert 'status' in progress_updates[-1]
    
    @pytest.mark.asyncio
    async def test_format_conversion_streaming(self, backend_onnx, mock_kokoro_instance):
        """Test streaming with format conversion"""
        backend_onnx.kokoro_instance = mock_kokoro_instance
        backend_onnx.model_loaded = True
        
        # Mock audio service
        mock_audio_service = Mock()
        async def mock_convert(audio_array, format, source_format="pcm", sample_rate=24000):
            # Return dummy converted data
            return b'converted_audio_chunk'
        
        mock_audio_service.convert_audio = mock_convert
        backend_onnx.audio_service = mock_audio_service
        
        request = OpenAISpeechRequest(
            input="Format conversion test",
            voice="af_bella",
            model="kokoro",
            response_format="mp3"  # Requires conversion
        )
        
        chunks = []
        async for chunk in backend_onnx.generate_speech_stream(request):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert all(chunk == b'converted_audio_chunk' for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_word_timestamps_onnx(self, backend_onnx, mock_kokoro_instance):
        """Test word-level timestamp generation for ONNX"""
        backend_onnx.kokoro_instance = mock_kokoro_instance
        backend_onnx.model_loaded = True
        
        text = "Hello world from Kokoro"
        audio_bytes, timestamps = await backend_onnx.generate_with_timestamps(text)
        
        # Verify audio generation
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0
        
        # Verify timestamps
        assert len(timestamps) == 4  # 4 words
        for i, ts in enumerate(timestamps):
            assert 'word' in ts
            assert 'start' in ts
            assert 'end' in ts
            assert 'confidence' in ts
            assert ts['start'] < ts['end']
            if i > 0:
                assert ts['start'] >= timestamps[i-1]['end']
    
    @pytest.mark.asyncio
    async def test_phoneme_generation(self, backend_onnx):
        """Test phoneme-based generation"""
        backend_onnx.use_onnx = True
        
        # Mock kokoro instance with phoneme support
        mock_instance = Mock()
        mock_instance.generate_from_phonemes = Mock(return_value=np.random.randn(24000).astype(np.float32))
        backend_onnx.kokoro_instance = mock_instance
        
        phonemes = "HH AH0 L OW1"
        audio_bytes = await backend_onnx.generate_from_phonemes(phonemes)
        
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0
        mock_instance.generate_from_phonemes.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_phoneme_not_supported(self, backend_pytorch):
        """Test phoneme generation error for PyTorch backend"""
        with pytest.raises(NotImplementedError, match="only supported with ONNX"):
            await backend_pytorch.generate_from_phonemes("HH AH0 L OW1")
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, backend_onnx, mock_kokoro_instance):
        """Test performance metrics tracking"""
        backend_onnx.kokoro_instance = mock_kokoro_instance
        backend_onnx.model_loaded = True
        backend_onnx.track_performance = True
        
        # Generate some audio
        request = OpenAISpeechRequest(
            input="Performance test",
            voice="af_bella",
            model="kokoro",
            response_format="pcm"
        )
        
        async for chunk in backend_onnx.generate_speech_stream(request):
            pass
        
        # Check metrics
        stats = backend_onnx.get_performance_stats()
        assert stats['total_generations'] == 1
        assert stats['total_time'] > 0
        assert stats['average_tokens_per_second'] > 0
    
    @pytest.mark.asyncio
    async def test_text_chunking(self, backend_onnx):
        """Test text chunking for long inputs"""
        backend_onnx.max_tokens = 10  # Small limit for testing
        
        long_text = " ".join(["word"] * 50)
        chunks = backend_onnx.text_chunker.chunk_text(long_text)
        
        assert len(chunks) > 1
        # Verify chunks respect token limit
        for chunk in chunks:
            assert len(chunk.split()) <= 10
    
    @pytest.mark.asyncio
    async def test_voice_mapping(self):
        """Test voice name mapping"""
        # Test OpenAI-style names
        assert map_voice_to_kokoro("alloy") == "af_bella"
        assert map_voice_to_kokoro("echo") == "af_sarah"
        assert map_voice_to_kokoro("onyx") == "am_michael"
        
        # Test direct names
        assert map_voice_to_kokoro("af_bella") == "af_bella"
        assert map_voice_to_kokoro("bella") == "af_bella"
        
        # Test unknown names (should return as-is)
        assert map_voice_to_kokoro("unknown_voice") == "unknown_voice"
    
    @pytest.mark.asyncio
    async def test_pytorch_text_splitting(self, backend_pytorch):
        """Test PyTorch-specific text splitting"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs: text.split())
        backend_pytorch.tokenizer = mock_tokenizer
        
        text = "This is a long text that needs to be split into multiple chunks for processing"
        chunks = backend_pytorch._split_text_for_pytorch(text, max_tokens=5)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 5
    
    @pytest.mark.asyncio
    async def test_model_download_handling(self, backend_onnx):
        """Test model download error handling"""
        backend_onnx.model_path = "/nonexistent/path/model.onnx"
        backend_onnx.voices_json = "/nonexistent/path/voices.json"
        
        # Mock failed download
        with patch('requests.get', side_effect=Exception("Network error")):
            await backend_onnx._initialize_onnx()
            
        # Should fall back to non-ONNX
        assert backend_onnx.use_onnx == False
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, backend_onnx, mock_kokoro_instance):
        """Test error recovery during streaming"""
        backend_onnx.kokoro_instance = mock_kokoro_instance
        backend_onnx.model_loaded = True
        
        # Mock audio service that fails on first conversion
        mock_audio_service = Mock()
        call_count = 0
        async def mock_convert(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Conversion failed")
            return b'converted_chunk'
        
        mock_audio_service.convert_audio = mock_convert
        backend_onnx.audio_service = mock_audio_service
        
        request = OpenAISpeechRequest(
            input="Error recovery test",
            voice="af_bella",
            model="kokoro",
            response_format="mp3"
        )
        
        chunks = []
        async for chunk in backend_onnx.generate_speech_stream(request):
            chunks.append(chunk)
        
        # Should recover and continue
        assert len(chunks) > 0


@pytest.mark.asyncio
async def test_integration_with_real_kokoro():
    """Integration test with real Kokoro model (if available)"""
    try:
        # Test ONNX backend
        config = {"KOKORO_USE_ONNX": True}
        backend = KokoroTTSBackend(config)
        await backend.initialize()
        
        if not backend.kokoro_instance:
            pytest.skip("Kokoro ONNX not available")
        
        request = OpenAISpeechRequest(
            input="Hello from Kokoro",
            voice="af_bella",
            model="kokoro",
            response_format="pcm"
        )
        
        audio_data = b''
        chunk_count = 0
        start_time = time.time()
        first_chunk_time = None
        
        async for chunk in backend.generate_speech_stream(request):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            audio_data += chunk
            chunk_count += 1
        
        # Verify streaming behavior
        assert chunk_count > 1
        assert first_chunk_time < 2.0  # Should start streaming quickly
        assert len(audio_data) > 1000
        
        # Test timestamp generation
        audio_bytes, timestamps = await backend.generate_with_timestamps(
            "Test word timestamps",
            voice="af_bella"
        )
        
        assert len(timestamps) == 3  # 3 words
        assert all('word' in ts for ts in timestamps)
        
    except ImportError:
        pytest.skip("Kokoro dependencies not installed")
    except Exception as e:
        pytest.skip(f"Kokoro initialization failed: {e}")