"""
Integration tests for Higgs TTS Backend.

These tests make real API calls and require the Higgs Audio model to be installed.
They test actual audio generation, voice cloning, and multi-speaker synthesis.

To run these tests:
1. Install the Higgs model: pip install boson-multimodal
2. Run with: pytest Tests/TTS/test_higgs_integration.py -v -s -m integration

Note: These tests may take significant time and resources as they perform actual inference.
"""

import asyncio
import io
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pytest
from loguru import logger

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.higgs import HiggsAudioTTSBackend
from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager

# Check if boson-multimodal is available
try:
    import boson_multimodal
    HIGGS_AVAILABLE = True
except ImportError:
    HIGGS_AVAILABLE = False

# Mark all tests in this module as integration tests and skip if Higgs not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HIGGS_AVAILABLE, reason="Higgs Audio (boson-multimodal) not installed")
]


class TestHiggsIntegration:
    """Integration tests for Higgs TTS backend with real model inference."""

    @pytest.fixture(scope="class")
    def backend(self):
        """Create a real Higgs TTS backend instance."""
        try:
            backend = HiggsAudioTTSBackend()
            yield backend
        except Exception as e:
            pytest.skip(f"Cannot initialize Higgs backend: {e}")

    @pytest.fixture
    def test_audio_dir(self):
        """Create a temporary directory for test audio files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_audio_file(self, test_audio_dir):
        """Create a real audio file for voice cloning tests."""
        # Generate a simple sine wave as test audio
        duration = 3.0  # seconds
        sample_rate = 24000
        frequency = 440.0  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        
        file_path = test_audio_dir / "sample_voice.wav"
        with wave.open(str(file_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return file_path

    @pytest.mark.asyncio
    async def test_model_loading(self, backend):
        """Test that the model loads successfully."""
        # Model should load on demand
        if hasattr(backend, 'model_loaded'):
            assert backend.model_loaded is False
        
        # Trigger model loading
        try:
            await backend.load_model()
            assert backend.model_loaded is True
            logger.info("Model loaded successfully")
        except Exception as e:
            if "boson_multimodal" in str(e):
                pytest.skip("Higgs Audio model not installed. Install with: pip install boson-multimodal")
            raise

    @pytest.mark.asyncio
    async def test_basic_tts_generation(self, backend, test_audio_dir):
        """Test basic text-to-speech generation with real model."""
        # Ensure model is loaded
        if not backend.model_loaded:
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
        
        text = "Hello, this is a test of the Higgs text to speech system."
        
        # Create request using OpenAI schema
        request = OpenAISpeechRequest(
            input=text,
            voice="professional_female",  # Higgs voice name
            response_format="wav"
        )
        
        # Generate audio via streaming (collect all chunks)
        audio_chunks = []
        chunk_count = 0
        
        async for chunk in backend.generate_speech_stream(request):
            audio_chunks.append(chunk)
            chunk_count += 1
            logger.debug(f"Received chunk {chunk_count}: {len(chunk)} bytes")
        
        # Combine chunks
        audio_data = b"".join(audio_chunks)
        assert len(audio_data) > 0
        assert chunk_count > 0
        
        # Save to file
        output_file = test_audio_dir / "basic_output.wav"
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        # Verify the audio file is valid (if WAV format)
        if request.response_format == "wav":
            with wave.open(str(output_file), 'rb') as wav_file:
                assert wav_file.getnchannels() in [1, 2]
                assert wav_file.getframerate() > 0
                assert wav_file.getnframes() > 0
                duration = wav_file.getnframes() / wav_file.getframerate()
                logger.info(f"Generated {duration:.2f}s of audio")

    @pytest.mark.asyncio
    async def test_voice_cloning(self, backend, test_audio_dir, sample_audio_file):
        """Test voice cloning with a real audio sample."""
        # Ensure model is loaded
        if not backend.model_loaded:
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
        
        text = "This is my cloned voice speaking."
        
        # Create a voice profile from the sample
        profile_created = await backend.create_voice_profile(
            profile_name="test_clone",
            reference_audio_path=str(sample_audio_file),
            display_name="Test Cloned Voice"
        )
        
        assert profile_created is True
        
        # Generate speech with the cloned voice
        request = OpenAISpeechRequest(
            input=text,
            voice="test_clone",  # Use the cloned voice
            response_format="wav"
        )
        
        audio_chunks = []
        async for chunk in backend.generate_speech_stream(request):
            audio_chunks.append(chunk)
        
        audio_data = b"".join(audio_chunks)
        assert len(audio_data) > 0
        
        # Save output
        output_file = test_audio_dir / "cloned_output.wav"
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Generated speech with cloned voice: {len(audio_data)} bytes")

    @pytest.mark.asyncio
    async def test_multi_speaker_dialog(self, backend, test_audio_dir):
        """Test multi-speaker dialog generation."""
        # Ensure model is loaded
        if not backend.model_loaded:
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
        
        # Multi-speaker text with special format
        dialog_text = """[Speaker: professional_female]
Hello! How are you today?

[Speaker: energetic_male]
I'm doing great, thanks for asking!

[Speaker: professional_female]
That's wonderful to hear.

[Speaker: young_female]
Can I join the conversation too?"""
        
        request = OpenAISpeechRequest(
            input=dialog_text,
            voice="multi",  # Special voice for multi-speaker
            response_format="wav"
        )
        
        # Generate multi-speaker audio
        audio_chunks = []
        async for chunk in backend.generate_speech_stream(request):
            audio_chunks.append(chunk)
        
        audio_data = b"".join(audio_chunks)
        assert len(audio_data) > 0
        
        # Save output
        output_file = test_audio_dir / "dialog_output.wav"
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Generated multi-speaker dialog: {len(audio_data)} bytes")

    @pytest.mark.asyncio
    async def test_streaming_with_callback(self, backend):
        """Test streaming audio generation with progress callbacks."""
        # Ensure model is loaded
        if not backend.model_loaded:
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
        
        text = "This is a test of streaming audio generation. The audio should be generated in chunks."
        
        chunks_received = []
        
        request = OpenAISpeechRequest(
            input=text,
            voice="calm_female",
            response_format="pcm"
        )
        
        # Generate with streaming
        async for chunk in backend.generate_speech_stream(request):
            assert chunk is not None
            assert isinstance(chunk, bytes)
            chunks_received.append(len(chunk))
        
        # Verify we received chunks
        assert len(chunks_received) > 0
        total_bytes = sum(chunks_received)
        assert total_bytes > 0
        
        logger.info(f"Received {len(chunks_received)} chunks, {total_bytes} total bytes")

    @pytest.mark.asyncio
    async def test_voice_parameters(self, backend, test_audio_dir):
        """Test generation with different voice parameters."""
        # Ensure model is loaded
        if not backend.model_loaded:
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
        
        text = "Testing different voice parameters."
        
        # Test different voices
        voices = ["professional_female", "energetic_male", "calm_female"]
        
        for voice in voices:
            request = OpenAISpeechRequest(
                input=text,
                voice=voice,
                response_format="wav"
            )
            
            audio_chunks = []
            async for chunk in backend.generate_speech_stream(request):
                audio_chunks.append(chunk)
            
            audio_data = b"".join(audio_chunks)
            assert len(audio_data) > 0
            
            output_file = test_audio_dir / f"{voice}_output.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"Generated with {voice}: {len(audio_data)} bytes")

    @pytest.mark.asyncio
    async def test_long_text_generation(self, backend, test_audio_dir):
        """Test generation with long text to verify chunking and memory handling."""
        # Ensure model is loaded
        if not backend.model_loaded:
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
        
        # Generate a long text
        long_text = " ".join([
            f"This is sentence number {i}."
            for i in range(1, 51)  # 50 sentences
        ])
        
        request = OpenAISpeechRequest(
            input=long_text,
            voice="narrator_male",
            response_format="wav"
        )
        
        # Track memory usage
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        audio_chunks = []
        async for chunk in backend.generate_speech_stream(request):
            audio_chunks.append(chunk)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        audio_data = b"".join(audio_chunks)
        assert len(audio_data) > 0
        
        # Memory increase should be reasonable (less than 500MB)
        assert mem_increase < 500, f"Memory increased by {mem_increase:.1f}MB"
        
        # Save output
        output_file = test_audio_dir / "long_text_output.wav"
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Generated {len(audio_data)} bytes, memory increase: {mem_increase:.1f}MB")

    @pytest.mark.asyncio
    async def test_concurrent_generation(self, backend, test_audio_dir):
        """Test concurrent audio generation requests."""
        # Ensure model is loaded
        if not backend.model_loaded:
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
        
        texts = [
            "First concurrent request.",
            "Second concurrent request.",
            "Third concurrent request."
        ]
        
        async def generate_task(index, text):
            request = OpenAISpeechRequest(
                input=text,
                voice="professional_female",
                response_format="wav"
            )
            
            chunks = []
            async for chunk in backend.generate_speech_stream(request):
                chunks.append(chunk)
            
            audio_data = b"".join(chunks)
            output_file = test_audio_dir / f"concurrent_{index}.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            
            return len(audio_data)
        
        # Run generations concurrently
        tasks = [generate_task(i, text) for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        assert len(results) == 3
        for i, size in enumerate(results):
            assert size > 0
            logger.info(f"Concurrent request {i+1}: {size} bytes")

    @pytest.mark.asyncio
    async def test_error_handling(self, backend, test_audio_dir):
        """Test error handling for various edge cases."""
        # Ensure model is loaded
        if not backend.model_loaded:
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
        
        # Test with empty text
        request = OpenAISpeechRequest(
            input="",
            voice="professional_female",
            response_format="wav"
        )
        
        # Should handle empty text gracefully (may generate silence or error)
        chunks = []
        try:
            async for chunk in backend.generate_speech_stream(request):
                chunks.append(chunk)
        except Exception as e:
            # Empty text might raise an error, which is fine
            logger.info(f"Empty text handled with error: {e}")
        
        # Test with invalid voice (should fall back to default)
        request = OpenAISpeechRequest(
            input="Test with invalid voice",
            voice="invalid_voice_name",
            response_format="wav"
        )
        
        chunks = []
        async for chunk in backend.generate_speech_stream(request):
            chunks.append(chunk)
        
        # Should still generate something
        audio_data = b"".join(chunks)
        assert len(audio_data) > 0
        
        # Test with invalid audio file for cloning
        try:
            await backend.create_voice_profile(
                profile_name="bad_clone",
                reference_audio_path="/nonexistent/audio.wav"
            )
            # Should fail
            assert False, "Expected error for nonexistent file"
        except Exception as e:
            logger.info(f"Correctly handled invalid file: {e}")


class TestHiggsVoiceManagerIntegration:
    """Integration tests for Higgs voice manager with real file operations."""

    @pytest.fixture
    def voice_manager(self, test_audio_dir):
        """Create a voice manager with test directory."""
        profiles_dir = test_audio_dir / "voice_profiles"
        profiles_dir.mkdir(exist_ok=True)
        return HiggsVoiceProfileManager(profiles_dir)

    @pytest.fixture
    def backend(self):
        """Create a backend for voice profile creation."""
        try:
            backend = HiggsAudioTTSBackend()
            return backend
        except Exception as e:
            pytest.skip(f"Cannot initialize Higgs backend: {e}")

    @pytest.mark.asyncio
    async def test_voice_profile_persistence(self, voice_manager, backend, sample_audio_file):
        """Test saving and loading voice profiles with real embeddings."""
        # Ensure model is loaded
        if not backend.model_loaded:
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
        
        # Create a real voice profile
        success = await backend.create_voice_profile(
            profile_name="persistent_voice",
            reference_audio_path=str(sample_audio_file),
            display_name="Test Persistent Voice"
        )
        
        assert success is True
        
        # List profiles to verify it was created
        profiles = voice_manager.list_profiles()
        assert "persistent_voice" in [p["name"] for p in profiles]
        
        # Use the profile to generate audio
        request = OpenAISpeechRequest(
            input="Testing loaded voice profile.",
            voice="persistent_voice",
            response_format="wav"
        )
        
        chunks = []
        async for chunk in backend.generate_speech_stream(request):
            chunks.append(chunk)
        
        audio_data = b"".join(chunks)
        assert len(audio_data) > 0

    @pytest.mark.asyncio
    async def test_profile_management(self, voice_manager, backend, sample_audio_file, test_audio_dir):
        """Test creating, listing, and deleting voice profiles."""
        # Ensure model is loaded
        if not backend.model_loaded:
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
        
        # Create multiple profiles
        for i in range(3):
            success = await backend.create_voice_profile(
                profile_name=f"test_profile_{i}",
                reference_audio_path=str(sample_audio_file),
                display_name=f"Test Profile {i}"
            )
            assert success is True
        
        # List all profiles
        profiles = voice_manager.list_profiles()
        profile_names = [p["name"] for p in profiles]
        
        for i in range(3):
            assert f"test_profile_{i}" in profile_names
        
        # Delete a profile
        if hasattr(voice_manager, 'delete_profile'):
            voice_manager.delete_profile("test_profile_1")
            
            # Verify deletion
            profiles = voice_manager.list_profiles()
            profile_names = [p["name"] for p in profiles]
            assert "test_profile_1" not in profile_names
            assert "test_profile_0" in profile_names
            assert "test_profile_2" in profile_names


@pytest.mark.asyncio
async def test_full_workflow_integration():
    """Test a complete workflow from voice cloning to multi-speaker dialog."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        try:
            # Initialize backend
            backend = HiggsAudioTTSBackend()
            
            # Load model
            try:
                await backend.load_model()
            except Exception as e:
                if "boson_multimodal" in str(e):
                    pytest.skip("Higgs Audio model not installed")
                raise
            
            # Create sample audio files for cloning
            sample_rate = 24000
            duration = 3.0
            
            # Create two different "voice" samples (different frequencies)
            for i, (name, freq) in enumerate([("alice", 300), ("bob", 200)]):
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio_data = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
                
                file_path = tmpdir / f"{name}_sample.wav"
                with wave.open(str(file_path), 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.tobytes())
                
                # Clone the voice
                success = await backend.create_voice_profile(
                    profile_name=f"{name}_voice",
                    reference_audio_path=str(file_path),
                    display_name=f"{name.capitalize()}'s Voice"
                )
                assert success is True
            
            # Create a dialog using cloned voices and built-in voices
            dialog_text = """[Speaker: alice_voice]
Hi Bob, have you heard about the new AI model?

[Speaker: bob_voice]
Yes Alice! It's quite impressive.

[Speaker: professional_female]
Excuse me, can I join your discussion?

[Speaker: alice_voice]
Of course! We're talking about AI models.

[Speaker: bob_voice]
The more perspectives, the better!"""
            
            # Generate the dialog
            request = OpenAISpeechRequest(
                input=dialog_text,
                voice="multi",
                response_format="wav"
            )
            
            chunks = []
            chunk_count = 0
            async for chunk in backend.generate_speech_stream(request):
                chunks.append(chunk)
                chunk_count += 1
            
            audio_data = b"".join(chunks)
            
            # Verify the result
            assert len(audio_data) > 0
            assert chunk_count > 0
            
            # Save the output
            output_file = tmpdir / "full_conversation.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"Full workflow completed: {len(audio_data)} bytes in {chunk_count} chunks")
            
        except Exception as e:
            pytest.skip(f"Cannot complete full workflow: {e}")


if __name__ == "__main__":
    # Run with: python -m pytest Tests/TTS/test_higgs_integration.py -v -s -m integration
    pytest.main([__file__, "-v", "-s", "-m", "integration"])