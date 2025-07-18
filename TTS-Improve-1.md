# TTS Implementation Review and Completion Plan

## Executive Summary

This document provides a comprehensive review of the TTS (Text-to-Speech) implementation in tldw_chatbook, identifying what is fully implemented, partially implemented, and what remains to be done for Kokoro (ONNX and non-ONNX), Chatterbox, OpenAI TTS, and ElevenLabs.

## Current Implementation Status

### ✅ Fully Implemented

#### 1. **OpenAI TTS Backend** (`tldw_chatbook/TTS/backends/openai.py`)
- Complete streaming implementation with proper chunking
- All voices supported: alloy, echo, fable, onyx, nova, shimmer
- All formats supported: mp3, opus, aac, flac, wav, pcm
- Comprehensive error handling with user-friendly messages
- Rate limiting and proper status code handling
- API key configuration from multiple sources (env, config.toml, app_tts)
- Input validation (4096 character limit, voice/format validation)

#### 2. **ElevenLabs Backend** (`tldw_chatbook/TTS/backends/elevenlabs.py`)
- Complete streaming implementation
- Voice ID mapping for common names and custom voice IDs
- Advanced voice settings:
  - Stability (0.0-1.0)
  - Similarity boost (0.0-1.0)
  - Style (0.0-1.0)
  - Speaker boost (true/false)
- Multiple output format support with intelligent mapping
- Cost tracking ready (per-character pricing)
- 5000 character limit per request
- Proper error handling and API key validation

#### 3. **Core Architecture**
- Well-structured backend system with proper abstractions:
  - `TTSBackendBase`: Abstract base for all backends
  - `APITTSBackend`: Base for API-based backends
  - `LocalTTSBackend`: Base for local model backends
- Backend registry pattern for easy extension
- Streaming support with `StreamingTTSMixin`
- Event-driven architecture with dedicated TTS events
- Global rate limiting (4 concurrent generations)
- Singleton pattern for TTS service

### ⚠️ Partially Implemented

#### 1. **Kokoro Backend** (`tldw_chatbook/TTS/backends/kokoro.py`)

**What's Implemented:**
- ✅ Both ONNX and PyTorch support
- ✅ Automatic model downloading with checksum verification
- ✅ Voice mixing with weighted combinations
- ✅ Performance metrics tracking
- ✅ Text chunking for long content
- ✅ Language detection
- ✅ Multiple voice support (10 voices)

**What's Missing:**
- ❌ **True streaming**: Currently collects all audio before yielding
- ❌ **Real-time chunking**: Should yield audio chunks as generated
- ❌ **Word-level timestamps**: Phoneme data not exposed
- ❌ **Progress callbacks**: No generation progress updates
- ❌ **Streaming for PCM format**: Only non-PCM formats stream properly

**Performance Gap:**
- Current: Generates full audio then streams
- Target (from Kokoro-FastAPI): 35x-100x realtime with <1s latency

#### 2. **Chatterbox Backend** (`tldw_chatbook/TTS/backends/chatterbox.py`)

**What's Implemented:**
- ✅ Zero-shot voice cloning support
- ✅ Advanced text preprocessing
- ✅ Multi-candidate generation with Whisper validation
- ✅ Audio normalization and post-processing
- ✅ Voice management with metadata
- ✅ Fallback strategies for robust generation

**What's Missing:**
- ❌ **Async streaming**: `_generate_stream_async` not properly implemented
- ❌ **Progress tracking**: No chunk/progress updates
- ❌ **Crossfade**: Multi-chunk audio needs smooth transitions
- ❌ **True streaming**: Collects audio before yielding

### ❌ Not Implemented

#### 1. **AllTalk TTS Backend**

**Configuration Exists:**
```toml
ALLTALK_TTS_URL_DEFAULT = "http://127.0.0.1:7851"
ALLTALK_TTS_VOICE_DEFAULT = "female_01.wav"
ALLTALK_TTS_LANGUAGE_DEFAULT = "en"
ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT = "wav"
```

**What's Missing:**
- No backend implementation file
- No registration in `TTSBackendManager`
- No event handler support
- No UI integration

## Detailed Implementation Plan

### Phase 1: Complete Kokoro Streaming (High Priority)

#### 1.1 Implement True Streaming
```python
# In kokoro.py - Replace current implementation with:
async def _generate_onnx_stream(self, request):
    # For PCM format, stream directly
    if request.response_format == "pcm":
        async for samples, sample_rate in self.kokoro_instance.create_stream(...):
            # Yield chunks immediately
            int16_samples = np.int16(samples * 32767)
            yield int16_samples.tobytes()
    else:
        # For other formats, use chunked processing
        chunk_buffer = []
        chunk_size = 8192  # samples
        
        async for samples, sr in self.kokoro_instance.create_stream(...):
            chunk_buffer.extend(samples)
            
            while len(chunk_buffer) >= chunk_size:
                chunk = chunk_buffer[:chunk_size]
                chunk_buffer = chunk_buffer[chunk_size:]
                
                # Convert and yield
                audio_bytes = self.audio_service.convert_audio_sync(
                    np.array(chunk), request.response_format, sr
                )
                yield audio_bytes
```

#### 1.2 Add Progress Tracking
```python
# Add progress callback support
class KokoroTTSBackend:
    def __init__(self, config):
        super().__init__(config)
        self.progress_callback = config.get('progress_callback')
    
    async def _generate_with_progress(self, text, **kwargs):
        total_tokens = len(self.tokenizer.encode(text))
        processed_tokens = 0
        
        async for samples, sr in self.kokoro_instance.create_stream(text, **kwargs):
            processed_tokens += len(samples) // 256  # Approximate
            
            if self.progress_callback:
                await self.progress_callback({
                    'processed': processed_tokens,
                    'total': total_tokens,
                    'percentage': (processed_tokens / total_tokens) * 100
                })
            
            yield samples, sr
```

#### 1.3 Implement Word-Level Timestamps
```python
# Expose phoneme data for word timestamps
async def generate_with_timestamps(self, text, **kwargs):
    # Generate phonemes first
    phonemes = self.kokoro_instance.generate_phonemes(text)
    
    # Generate audio with phoneme alignment
    audio_chunks = []
    word_timestamps = []
    current_time = 0.0
    
    for word, phoneme_data in phonemes:
        samples, duration = await self._generate_word(phoneme_data, **kwargs)
        audio_chunks.append(samples)
        
        word_timestamps.append({
            'word': word,
            'start': current_time,
            'end': current_time + duration
        })
        current_time += duration
    
    return audio_chunks, word_timestamps
```

### Phase 2: Complete Chatterbox Streaming (Medium Priority)

#### 2.1 Fix Async Streaming
```python
# Properly implement async streaming
async def _generate_stream_async(self, text, audio_prompt_path, exaggeration, cfg_weight):
    # Create async generator from sync model
    def sync_gen():
        return self.model.generate_stream(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            chunk_size=self.chunk_size
        )
    
    # Run in executor and yield results
    loop = asyncio.get_event_loop()
    executor = None
    
    # Create queue for thread-safe communication
    queue = asyncio.Queue()
    
    def producer():
        try:
            for chunk, metrics in sync_gen():
                asyncio.run_coroutine_threadsafe(
                    queue.put((chunk, metrics)), loop
                )
            asyncio.run_coroutine_threadsafe(
                queue.put(None), loop  # Sentinel
            )
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                queue.put(e), loop
            )
    
    # Start producer thread
    await loop.run_in_executor(executor, producer)
    
    # Consume from queue
    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item
```

#### 2.2 Add Crossfade Support
```python
# Implement audio crossfade for smooth transitions
def crossfade_audio(self, audio1, audio2, overlap_duration_ms=50):
    import torch
    
    sample_rate = getattr(self.model, 'sr', 24000)
    overlap_samples = int(sample_rate * overlap_duration_ms / 1000)
    
    if overlap_samples >= min(len(audio1), len(audio2)):
        # No overlap possible, just concatenate
        return torch.cat([audio1, audio2])
    
    # Create fade curves
    fade_out = torch.linspace(1, 0, overlap_samples)
    fade_in = torch.linspace(0, 1, overlap_samples)
    
    # Apply crossfade
    audio1_fade = audio1.clone()
    audio1_fade[-overlap_samples:] *= fade_out
    
    audio2_fade = audio2.clone()
    audio2_fade[:overlap_samples] *= fade_in
    
    # Combine
    result = torch.zeros(len(audio1) + len(audio2) - overlap_samples)
    result[:len(audio1)] = audio1_fade
    result[len(audio1)-overlap_samples:] += audio2_fade
    
    return result
```

### Phase 3: Implement AllTalk Backend (High Priority)

#### 3.1 Create AllTalk Backend
```python
# New file: tldw_chatbook/TTS/backends/alltalk.py
from typing import AsyncGenerator, Optional, Dict, Any
import httpx
from loguru import logger

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.base_backends import APITTSBackend
from tldw_chatbook.config import get_cli_setting

class AllTalkTTSBackend(APITTSBackend):
    """AllTalk Text-to-Speech API backend (OpenAI-compatible)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Get configuration
        self.base_url = self.config.get("ALLTALK_TTS_URL",
            get_cli_setting("app_tts", "ALLTALK_TTS_URL_DEFAULT", 
                          "http://127.0.0.1:7851"))
        
        self.default_voice = self.config.get("ALLTALK_TTS_VOICE",
            get_cli_setting("app_tts", "ALLTALK_TTS_VOICE_DEFAULT", 
                          "female_01.wav"))
        
        self.default_language = self.config.get("ALLTALK_TTS_LANGUAGE",
            get_cli_setting("app_tts", "ALLTALK_TTS_LANGUAGE_DEFAULT", "en"))
        
        self.output_format = self.config.get("ALLTALK_TTS_OUTPUT_FORMAT",
            get_cli_setting("app_tts", "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT", "wav"))
        
        # AllTalk uses OpenAI-compatible endpoint
        self.endpoint = f"{self.base_url}/v1/audio/speech"
        
    async def initialize(self):
        """Initialize the backend"""
        logger.info(f"AllTalkTTSBackend initialized with URL: {self.base_url}")
        
        # Test connection
        try:
            async with self.client.get(f"{self.base_url}/api/voices") as response:
                if response.status_code == 200:
                    voices = response.json()
                    logger.info(f"AllTalk available voices: {voices}")
                else:
                    logger.warning(f"AllTalk server returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to AllTalk server: {e}")
    
    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """Generate speech using AllTalk API"""
        
        # Map voice to AllTalk format
        voice = request.voice
        if not voice.endswith('.wav'):
            voice = f"{voice}.wav"
        
        # AllTalk OpenAI-compatible payload
        payload = {
            "model": "tts-1",  # AllTalk ignores this but requires it
            "input": request.input,
            "voice": voice,
            "response_format": self.output_format,
            "speed": request.speed,
            "language": self.default_language  # AllTalk extension
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        logger.info(f"AllTalkTTSBackend: Generating TTS for {len(request.input)} chars")
        
        try:
            async with self.client.stream("POST", self.endpoint, 
                                        headers=headers, json=payload) as response:
                response.raise_for_status()
                
                # Stream the audio data
                chunk_size = 8192
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    yield chunk
                    
            logger.info("AllTalkTTSBackend: Generation complete")
            
        except httpx.HTTPStatusError as e:
            logger.error(f"AllTalk API error {e.response.status_code}")
            raise ValueError(f"AllTalk TTS request failed: HTTP {e.response.status_code}")
        except Exception as e:
            logger.error(f"AllTalkTTSBackend error: {e}")
            raise ValueError("AllTalk TTS generation failed")
```

#### 3.2 Register AllTalk Backend
```python
# In TTS_Backends.py - Add to _register_builtin_backends():
try:
    from tldw_chatbook.TTS.backends.alltalk import AllTalkTTSBackend
    BackendRegistry.register("alltalk_*", AllTalkTTSBackend)
except ImportError:
    logger.warning("AllTalk TTS backend not available")
```

### Phase 4: Cross-Backend Improvements

#### 4.1 Unified Progress Interface
```python
# Add to base_backends.py
class ProgressCallback(Protocol):
    async def __call__(self, progress: Dict[str, Any]) -> None:
        """Progress callback with info dict containing:
        - processed: int (tokens/samples processed)
        - total: int (total tokens/samples)
        - percentage: float (0-100)
        - eta_seconds: Optional[float]
        - current_text: Optional[str] (current chunk being processed)
        """
        ...

class TTSBackendBase(ABC):
    def set_progress_callback(self, callback: Optional[ProgressCallback]):
        """Set progress callback for generation tracking"""
        self.progress_callback = callback
```

#### 4.2 Cost Estimation
```python
# Add to each backend
def estimate_cost(self, text: str, model: str) -> Dict[str, float]:
    """Estimate generation cost"""
    char_count = len(text)
    
    # Provider-specific pricing
    if isinstance(self, OpenAITTSBackend):
        # $15.00 per 1M characters for tts-1
        # $30.00 per 1M characters for tts-1-hd
        rate = 0.015 if model == "tts-1" else 0.030
        cost = (char_count / 1_000_000) * rate
    elif isinstance(self, ElevenLabsTTSBackend):
        # ~$0.30 per 1K characters (varies by plan)
        cost = (char_count / 1_000) * 0.30
    else:
        cost = 0.0  # Local models
    
    return {
        'estimated_cost': cost,
        'character_count': char_count,
        'provider': self.__class__.__name__
    }
```

### Phase 5: Testing and Validation

#### 5.1 Streaming Performance Tests
```python
# Create test_tts_streaming.py
async def test_kokoro_streaming_performance():
    """Verify Kokoro streams in real-time"""
    backend = KokoroTTSBackend()
    await backend.initialize()
    
    text = "Hello world, this is a streaming test."
    request = OpenAISpeechRequest(
        input=text,
        voice="af_bella",
        response_format="pcm"
    )
    
    first_chunk_time = None
    chunk_count = 0
    
    start = time.time()
    async for chunk in backend.generate_speech_stream(request):
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
        chunk_count += 1
    
    total_time = time.time() - start
    
    assert first_chunk_time < 1.0, f"First chunk took {first_chunk_time}s (should be <1s)"
    assert chunk_count > 1, "Should receive multiple chunks"
    print(f"Streaming stats: First chunk: {first_chunk_time}s, Total: {total_time}s, Chunks: {chunk_count}")
```

#### 5.2 Backend Comparison Tests
```python
# Compare all backends
async def compare_backends():
    text = "The quick brown fox jumps over the lazy dog."
    
    backends = [
        ("OpenAI", OpenAITTSBackend),
        ("ElevenLabs", ElevenLabsTTSBackend),
        ("Kokoro", KokoroTTSBackend),
        ("Chatterbox", ChatterboxTTSBackend),
        ("AllTalk", AllTalkTTSBackend),
    ]
    
    for name, backend_class in backends:
        try:
            backend = backend_class()
            await backend.initialize()
            
            # Test generation
            start = time.time()
            audio_size = 0
            
            request = OpenAISpeechRequest(
                input=text,
                voice="default",
                response_format="mp3"
            )
            
            async for chunk in backend.generate_speech_stream(request):
                audio_size += len(chunk)
            
            duration = time.time() - start
            print(f"{name}: {duration:.2f}s, {audio_size/1024:.1f}KB")
            
        except Exception as e:
            print(f"{name}: Failed - {e}")
```

## Implementation Priority

### Week 1 (High Priority)
1. Fix Kokoro streaming to support true real-time generation
2. Implement AllTalk backend (configuration already exists)
3. Add unified progress tracking interface

### Week 2 (Medium Priority)
1. Fix Chatterbox async streaming issues
2. Implement audio caching system
3. Add cost estimation for all providers
4. Complete crossfade for multi-chunk audio

### Week 3 (Low Priority)
1. Add word-level timestamps for Kokoro
2. Implement advanced SSML support
3. Add connection retry logic
4. Performance optimization and benchmarking

## Success Metrics

1. **Streaming Latency**: First audio chunk < 1 second for all backends
2. **Real-time Factor**: Local models achieve 35x+ realtime generation
3. **Memory Usage**: Streaming should not accumulate full audio in memory
4. **Error Recovery**: All backends gracefully handle interruptions
5. **Cost Accuracy**: Cost estimates within 10% of actual charges

## Notes

- All backends should follow the same streaming pattern for consistency
- Progress callbacks should be non-blocking
- Temporary files must be cleaned up properly
- API keys should never appear in logs
- All user-facing errors should be actionable

---

Last Updated: 2025-07-18