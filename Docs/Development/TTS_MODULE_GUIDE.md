# TTS (Text-to-Speech) Module Guide

## Overview

The TTS module in tldw_chatbook provides a flexible, extensible system for generating speech from text using multiple providers. It supports both cloud-based APIs (OpenAI, ElevenLabs) and local models (Kokoro), with features like streaming audio generation, format conversion, and text normalization.

## Architecture

### Module Structure

```
tldw_chatbook/TTS/
├── __init__.py              # Module exports
├── audio_schemas.py         # Pydantic schemas for requests/responses
├── TTS_Generation.py        # Main TTS service orchestration
├── TTS_Backends.py          # Backend manager and base class
├── audio_service.py         # Audio format conversion service
├── text_processing.py       # Text normalization and chunking
├── backends/                # Backend implementations
│   ├── __init__.py
│   ├── openai.py           # OpenAI TTS API
│   ├── kokoro.py           # Local Kokoro model
│   └── elevenlabs.py       # ElevenLabs API
└── utils/                   # Utility modules
    ├── __init__.py
    ├── download_models.py   # Model download utilities
    ├── voice_utils.py       # Voice mixing utilities
    └── performance.py       # Performance tracking
```

### Core Components

#### 1. TTSService (TTS_Generation.py)
The main orchestration layer that:
- Manages backend selection based on model/provider
- Handles streaming audio generation
- Coordinates text processing and audio conversion
- Provides a unified interface for all backends

#### 2. TTSBackendBase (TTS_Backends.py)
Abstract base class that all backends must implement:
```python
class TTSBackendBase(ABC):
    async def initialize(self)
    async def generate_speech_stream(request: OpenAISpeechRequest) -> AsyncGenerator[bytes, None]
    async def close(self)
```

#### 3. Audio Service (audio_service.py)
Handles audio format conversion with:
- `StreamingAudioWriter`: Real-time encoding for streaming
- Support for MP3, Opus, AAC, FLAC, WAV, PCM
- Async and sync conversion methods

#### 4. Text Processing (text_processing.py)
Provides text preparation for TTS:
- `TextNormalizer`: Handles URLs, emails, phone numbers, units
- `TextChunker`: Splits long texts respecting sentence boundaries
- Language detection based on voice selection

## Backend Implementations

### OpenAI Backend

**Features:**
- Supports tts-1 and tts-1-hd models
- Multiple voices: alloy, echo, fable, onyx, nova, shimmer
- Streaming response support
- All OpenAI audio formats

**Configuration:**
```toml
[app_tts]
OPENAI_API_KEY_fallback = "sk-your-api-key"
```

### Kokoro Backend

**Features:**
- Local text-to-speech using Kokoro-82M model
- ONNX runtime support (PyTorch planned)
- Multiple voice packs
- Voice mixing capabilities (planned)
- No internet connection required

**Configuration:**
```toml
[app_tts]
KOKORO_ONNX_MODEL_PATH_DEFAULT = "models/kokoro-v0_19.onnx"
KOKORO_ONNX_VOICES_JSON_DEFAULT = "models/voices.json"
KOKORO_DEVICE_DEFAULT = "cpu"  # or "cuda" for GPU
KOKORO_MAX_TOKENS = 500
```

**Voices:**
- Female: af_bella, af_nicole, af_sarah, af_sky, bf_emma, bf_isabella
- Male: am_adam, am_michael, bm_george, bm_lewis

### ElevenLabs Backend

**Features:**
- High-quality voice synthesis
- Advanced voice settings (stability, similarity boost, style)
- Multiple languages and accents
- Speaker boost for enhanced clarity
- Multiple output formats

**Configuration:**
```toml
[app_tts]
ELEVENLABS_API_KEY_fallback = "your-api-key"
ELEVENLABS_DEFAULT_VOICE = "voice-id"
ELEVENLABS_DEFAULT_MODEL = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_192"
ELEVENLABS_VOICE_STABILITY = 0.5
ELEVENLABS_SIMILARITY_BOOST = 0.8
ELEVENLABS_STYLE = 0.0
ELEVENLABS_USE_SPEAKER_BOOST = true
```

## Installation

### Basic Installation
The core TTS functionality (OpenAI, ElevenLabs) is included with the base installation:
```bash
pip install tldw_chatbook
```

### Local TTS Support
For local TTS models like Kokoro, install the optional dependencies:
```bash
pip install tldw_chatbook[local_tts]
```

This installs:
- kokoro-onnx: ONNX runtime for Kokoro
- scipy: Audio processing
- nltk: Text tokenization
- pyaudio/pydub: Audio playback
- transformers: Advanced tokenization
- torch: PyTorch support (for future backends)
- onnxruntime: ONNX model inference

### Kokoro Model Setup
1. Download the model files:
   - Model: `kokoro-v0_19.onnx` (~300MB)
   - Voices: `voices.json`

2. Place them in your configured paths or use the download utility:
   ```python
   from tldw_chatbook.TTS.utils.download_models import download_kokoro_model
   await download_kokoro_model()
   ```

## Usage

### Basic Usage in the App

1. Click the speak button (🔊) on any chat message
2. The TTS service will:
   - Use the configured default provider
   - Generate audio with the default voice
   - Play the audio automatically

### Programmatic Usage

```python
from tldw_chatbook.TTS import get_tts_service, OpenAISpeechRequest

# Initialize service
tts_service = await get_tts_service(app_config)

# Create request
request = OpenAISpeechRequest(
    model="tts-1",
    input="Hello, world!",
    voice="alloy",
    response_format="mp3",
    speed=1.0
)

# Generate audio
internal_model_id = "openai_official_tts-1"
async for chunk in tts_service.generate_audio_stream(request, internal_model_id):
    # Process audio chunks
    audio_file.write(chunk)
```

### Event System Integration

The TTS module integrates with Textual's event system:

```python
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSRequestEvent

# Request TTS generation
app.post_message(TTSRequestEvent(
    text="Text to speak",
    message_id="msg_123",
    voice="alloy"  # Optional voice override
))

# Handle completion
@on(TTSCompleteEvent)
async def handle_tts_complete(self, event: TTSCompleteEvent):
    if event.error:
        self.notify(f"TTS failed: {event.error}")
    else:
        # Audio file available at event.audio_file
        play_audio_file(event.audio_file)
```

## Configuration Reference

### Global Settings
```toml
[app_tts]
default_provider = "openai"  # openai, kokoro, elevenlabs
default_voice = "alloy"      # Provider-specific voice
default_model = "tts-1"      # Provider-specific model
default_format = "mp3"       # Audio output format
default_speed = 1.0          # Speech speed (0.25-4.0)
```

### Provider Selection
The system automatically selects backends based on the model:
- `tts-1`, `tts-1-hd` → OpenAI backend
- `kokoro` → Local Kokoro backend
- `eleven_*` models → ElevenLabs backend

### Audio Formats
Supported formats vary by provider:
- **All providers**: mp3, wav, pcm
- **OpenAI**: opus, aac, flac
- **ElevenLabs**: Various bitrate/quality options
- **Kokoro**: Best with wav/pcm for quality

## Advanced Features

### Text Normalization
Configure text preprocessing:
```python
normalization_options = NormalizationOptions(
    normalize=True,
    unit_normalization=True,      # 10KB → 10 kilobytes
    url_normalization=True,       # https://example.com → example dot com
    email_normalization=True,     # user@example.com → user at example dot com
    phone_normalization=True,     # 555-1234 → 5 5 5 1 2 3 4
)
```

### Voice Mixing (Kokoro)
Combine multiple voice characteristics:
```python
# Future enhancement
voice = "af_bella:0.6,af_sarah:0.4"  # 60% bella, 40% sarah
```

### Streaming with Chunk Processing
```python
async for chunk in backend.generate_speech_stream(request):
    # Process chunks in real-time
    await websocket.send(chunk)
    
    # Or accumulate for post-processing
    chunks.append(chunk)
```

## Performance Considerations

### Kokoro Performance
- **CPU**: ~3.5s latency for first token
- **GPU**: ~0.3s latency for first token
- **Generation speed**: 35-100x realtime
- **Token rate**: ~140 tokens/second

### Optimization Tips
1. **Use streaming** for better perceived performance
2. **Pre-download models** for Kokoro to avoid first-run delays
3. **Cache frequently used phrases** (future enhancement)
4. **Adjust chunk size** based on network conditions
5. **Use appropriate format**:
   - PCM for lowest latency
   - MP3 for compatibility
   - Opus for best compression

### Memory Usage
- Kokoro model: ~300MB when loaded
- Audio buffers: Minimal with streaming
- Text processing: Negligible

## Troubleshooting

### Common Issues

#### "TTS service not available"
- Check if TTS was initialized successfully
- Verify API keys are configured
- Check logs for initialization errors

#### "No audio output"
- Verify audio playback system is working
- Check file permissions for temp directory
- Ensure audio format is supported by system

#### "Kokoro model not found"
- Download model files to configured paths
- Check file permissions
- Verify ONNX runtime is installed

#### "API key errors"
- Check key format and validity
- Verify key has required permissions
- Check API quotas/limits

### Debug Logging
Enable debug logging for detailed information:
```toml
[logging]
level = "DEBUG"
```

Check logs at: `~/.share/tldw_cli/logs/`

### Performance Issues
1. **Slow generation**:
   - Use streaming for better UX
   - Consider using faster models (tts-1 vs tts-1-hd)
   - Check network latency for API calls

2. **High memory usage**:
   - Unload models when not in use
   - Use streaming instead of full generation
   - Monitor with Stats tab

## API Reference

### Schemas

#### OpenAISpeechRequest
```python
class OpenAISpeechRequest(BaseModel):
    model: str                    # Model identifier
    input: str                    # Text to synthesize
    voice: str                    # Voice selection
    response_format: str          # Audio format
    speed: float = 1.0           # Speed adjustment (0.25-4.0)
    stream: bool = True          # Enable streaming
    lang_code: Optional[str]     # Language hint
    normalization_options: Optional[NormalizationOptions]
```

### Backend Methods

#### initialize()
Async initialization of the backend. Called once before first use.

#### generate_speech_stream()
Main generation method returning an async generator of audio bytes.

#### close()
Cleanup method for releasing resources.

## Future Enhancements

### Planned Features
1. **Voice Cloning**: Support for custom voices
2. **SSML Support**: Advanced speech markup
3. **Caching System**: Reduce repeated generations
4. **Batch Processing**: Multiple texts in one request
5. **Real-time Streaming**: WebSocket-based streaming
6. **More Backends**: Edge-TTS, Coqui, Piper

### Experimental Features
- **Emotion Control**: Adjust emotional tone
- **Prosody Tuning**: Fine-tune speech characteristics
- **Multi-speaker**: Different voices in one text
- **Audio Effects**: Post-processing effects

## Contributing

### Adding a New Backend

1. Create new file in `backends/`:
```python
from tldw_chatbook.TTS.TTS_Backends import TTSBackendBase

class MyBackend(TTSBackendBase):
    async def initialize(self):
        # Setup code
        pass
    
    async def generate_speech_stream(self, request):
        # Generation logic
        yield audio_bytes
```

2. Register in `TTS_Backends.py`:
```python
elif backend_id == "my_backend":
    self._backends[backend_id] = MyBackend(config)
```

3. Add configuration schema
4. Update documentation

### Testing
Run TTS-specific tests:
```bash
pytest Tests/TTS/
```

## Security Considerations

1. **API Keys**: Never log or display API keys
2. **Input Validation**: All text inputs are sanitized
3. **File Paths**: Temporary files use secure generation
4. **Network**: HTTPS only for API calls
5. **Local Models**: Verify model file integrity

## License

The TTS module follows the main project's AGPL-3.0+ license. Individual model licenses:
- Kokoro: Apache 2.0
- API providers: Subject to their respective terms of service