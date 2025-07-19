# TTS Playground Implementation Fix Plan

## Executive Summary

The TTS Playground has a solid UI foundation but critical backend functionality is missing or broken. This plan details the fixes needed to make it fully operational, prioritized by user impact.

## Current Implementation Status

### ✅ Working Features
- TTS Playground UI with all controls
- Basic TTS generation for OpenAI provider
- Voice Blend Dialog UI (creates/saves blends)
- Settings persistence to config.toml
- Event system connectivity

### ❌ Broken Features
- Provider-specific settings ignored at runtime
- AudioBook generation non-functional (backend incomplete)
- Voice blends not integrated with playground
- Import/Export uses hardcoded paths
- Missing audio processing capabilities

### ⚠️ Partially Working
- ElevenLabs/Kokoro TTS (works but ignores dynamic settings)
- File picker integration (works elsewhere, not used here)

## Priority 1: Critical Fixes (Required for Basic Functionality)

### 1.1 Fix Provider-Specific Settings Pass-Through

**Problem**: ElevenLabs voice settings (stability, similarity_boost, etc.) are collected from UI but ignored by backend.

**Root Cause**: The backend uses class-level defaults instead of request parameters.

**Fix Implementation**:

```python
# In tldw_chatbook/TTS/backends/elevenlabs.py, modify generate_audio method:

async def generate_audio(self, request: OpenAISpeechRequest, internal_model_id: str = None) -> AsyncGenerator[bytes, None]:
    # ... existing code ...
    
    # Override default voice settings if provided in request
    voice_settings = {
        "stability": self.voice_stability,
        "similarity_boost": self.similarity_boost,
        "style": self.style,
        "use_speaker_boost": self.use_speaker_boost
    }
    
    # Check for extra_params in request
    if hasattr(request, 'extra_params') and request.extra_params:
        if 'stability' in request.extra_params:
            voice_settings['stability'] = float(request.extra_params['stability'])
        if 'similarity_boost' in request.extra_params:
            voice_settings['similarity_boost'] = float(request.extra_params['similarity_boost'])
        if 'style' in request.extra_params:
            voice_settings['style'] = float(request.extra_params['style'])
        if 'use_speaker_boost' in request.extra_params:
            voice_settings['use_speaker_boost'] = bool(request.extra_params['use_speaker_boost'])
    
    payload = {
        "text": request.input,
        "model_id": model_id,
        "output_format": output_format,
        "voice_settings": voice_settings
    }
```

**Also update the event handler to properly pass parameters**:

```python
# In tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py, line 165-172:

# Create a modified request with extra params
if event.provider == "elevenlabs" and event.extra_params:
    # Add extra_params as an attribute to the request
    request.extra_params = event.extra_params
elif event.provider == "kokoro" and event.extra_params:
    request.extra_params = event.extra_params

# Pass the request with extra_params
async for chunk in self._stts_service.generate_audio_stream(request, internal_model_id):
    audio_data += chunk
```

### 1.2 Fix Voice Blend Integration

**Problem**: Voice blends are saved but not shown in the playground dropdown.

**Fix Implementation**:

```python
# In tldw_chatbook/UI/STTS_Window.py, modify _update_provider_options method:

def _update_provider_options(self, provider: str) -> None:
    """Update options based on selected provider"""
    # ... existing code ...
    
    elif provider == "kokoro":
        # Standard voices
        voice_options = [
            ("af_bella", "Bella (American Female)"),
            # ... existing voices ...
        ]
        
        # Add saved voice blends
        blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
        if blend_file.exists():
            try:
                with open(blend_file, 'r') as f:
                    blends = json.load(f)
                    for blend_name, blend_data in blends.items():
                        # Add blend to dropdown with special prefix
                        display_name = f"🎭 {blend_name} (Blend)"
                        voice_options.append((f"blend:{blend_name}", display_name))
            except Exception as e:
                logger.error(f"Failed to load voice blends: {e}")
        
        voice_select.set_options(voice_options)
```

### 1.3 Fix File Picker for Import/Export

**Problem**: Hardcoded paths to Downloads folder.

**Fix Implementation**:

```python
# In tldw_chatbook/UI/STTS_Window.py, replace _import_voice_blends method:

def _import_voice_blends(self) -> None:
    """Import voice blends from file"""
    try:
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
        from tldw_chatbook.Third_Party.textual_fspicker import Filters
        
        filters = Filters(
            ("JSON Files", lambda p: p.suffix.lower() == ".json"),
            ("All Files", lambda p: True)
        )
        
        file_picker = FileOpen(
            title="Import Voice Blends",
            filters=filters,
            context="voice_blends_import"
        )
        
        self.app.push_screen(file_picker, self._handle_import_file)
        
    except Exception as e:
        logger.error(f"Failed to show import dialog: {e}")
        self.app.notify(f"Error showing import dialog: {e}", severity="error")

def _handle_import_file(self, path: str | None) -> None:
    """Handle the imported file"""
    if not path:
        return
    # ... rest of import logic ...
```

## Priority 2: AudioBook Generation Backend

### 2.1 Implement Audio Processing Functions

**Add dependencies to requirements**:
```toml
# In pyproject.toml under [project.optional-dependencies]
audiobook = [
    "pydub>=0.25.1",
    "numpy>=1.24.0",
    "mutagen>=1.47.0",
]
```

**Implement missing functions**:

```python
# In tldw_chatbook/TTS/audiobook_generator.py:

async def _generate_silence(self, duration: float, format: str) -> bytes:
    """Generate silence of specified duration"""
    try:
        from pydub import AudioSegment
        
        # Create silence
        silence = AudioSegment.silent(duration=int(duration * 1000))  # Convert to milliseconds
        
        # Export to requested format
        buffer = io.BytesIO()
        silence.export(buffer, format=format)
        return buffer.getvalue()
        
    except ImportError:
        logger.error("pydub not installed. Install with: pip install pydub")
        # Fallback: return minimal valid audio data
        if format == "mp3":
            # Minimal MP3 frame (silence)
            return b'\xff\xfb\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        else:
            return b''  # Will cause issues but won't crash

async def _normalize_audio(self, audio_data: bytes, format: str, target_db: float) -> bytes:
    """Normalize audio to target dB level"""
    try:
        from pydub import AudioSegment
        import numpy as np
        
        # Load audio
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format)
        
        # Calculate current dBFS
        current_db = audio.dBFS
        
        # Calculate gain needed
        gain_db = target_db - current_db
        
        # Apply gain
        normalized = audio.apply_gain(gain_db)
        
        # Export
        buffer = io.BytesIO()
        normalized.export(buffer, format=format)
        return buffer.getvalue()
        
    except ImportError:
        logger.warning("Audio normalization unavailable. Install pydub and numpy.")
        return audio_data  # Return unchanged

async def _get_audio_duration(self, audio_path: Path) -> float:
    """Get duration of audio file in seconds"""
    try:
        from mutagen import File
        
        audio_file = File(str(audio_path))
        if audio_file is not None and hasattr(audio_file.info, 'length'):
            return float(audio_file.info.length)
        else:
            # Fallback to pydub
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert from milliseconds
            
    except ImportError:
        logger.warning("Cannot determine audio duration. Install mutagen or pydub.")
        # Rough estimate based on file size and format
        file_size = audio_path.stat().st_size
        # Assume 128kbps for MP3
        return (file_size * 8) / (128 * 1000)
```

### 2.2 Add Format Conversion Support

```python
# In tldw_chatbook/TTS/audiobook_generator.py, add method:

async def _convert_audio_format(self, audio_path: Path, target_format: str) -> Path:
    """Convert audio to target format"""
    if audio_path.suffix[1:].lower() == target_format.lower():
        return audio_path
    
    try:
        from pydub import AudioSegment
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Create output path
        output_path = audio_path.with_suffix(f'.{target_format}')
        
        # Export with metadata
        tags = {
            'title': self.metadata.title,
            'artist': self.metadata.narrator,
            'album': self.metadata.title,
            'date': self.metadata.publication_date or str(datetime.now().year),
            'genre': self.metadata.genre or 'Audiobook'
        }
        
        audio.export(output_path, format=target_format, tags=tags)
        
        # Remove original if different
        if audio_path != output_path:
            audio_path.unlink()
        
        return output_path
        
    except ImportError:
        logger.error(f"Cannot convert to {target_format}. Install pydub and ffmpeg.")
        raise RuntimeError(f"Audio format conversion to {target_format} not available")
```

## Priority 3: Enhanced Features

### 3.1 Add Cost Estimation

```python
# In tldw_chatbook/UI/STTS_Window.py, add method to AudioBookGenerationWidget:

def _estimate_cost(self) -> Optional[float]:
    """Estimate cost for audiobook generation"""
    if not self.content_text:
        return None
    
    provider = self.query_one("#audiobook-provider-select", Select).value
    model = self.query_one("#audiobook-model-select", Select).value
    
    # Rough character count
    char_count = len(self.content_text)
    
    # Cost per 1M characters (example rates)
    costs = {
        ("openai", "tts-1"): 15.0,
        ("openai", "tts-1-hd"): 30.0,
        ("elevenlabs", "eleven_monolingual_v1"): 100.0,
        ("elevenlabs", "eleven_turbo_v2"): 50.0,
    }
    
    cost_per_million = costs.get((provider, model), 20.0)  # Default
    estimated_cost = (char_count / 1_000_000) * cost_per_million
    
    return estimated_cost

# Add to UI before generate button:
yield Label("Estimated Cost: $0.00", id="audiobook-cost-estimate", classes="cost-estimate")
```

### 3.2 Add Progress Bar

```python
# In tldw_chatbook/UI/STTS_Window.py, add to AudioBookGenerationWidget compose:

# After the log widget
yield ProgressBar(
    total=100,
    show_eta=True,
    show_percentage=True,
    id="audiobook-progress-bar",
    classes="progress-bar"
)
```

### 3.3 Voice Preview

```python
# Add preview button next to voice selection:
yield Button("🔊 Preview", id="preview-voice-btn", variant="default", disabled=False)

# Add handler:
def _preview_voice(self) -> None:
    """Generate a short preview of the selected voice"""
    voice = self.query_one("#audiobook-narrator-select", Select).value
    provider = self.query_one("#audiobook-provider-select", Select).value
    
    preview_text = "Hello, this is a preview of the selected voice for your audiobook."
    
    # Post mini TTS event
    self.app.post_message(STTSPlaygroundGenerateEvent(
        text=preview_text,
        provider=provider,
        voice=voice,
        model=self.query_one("#audiobook-model-select", Select).value,
        speed=1.0,
        format="mp3",
        extra_params={"preview": True}
    ))
```

## Priority 4: Testing & Validation

### 4.1 Add Unit Tests

```python
# Create Tests/TTS/test_tts_playground_fixes.py:

import pytest
from unittest.mock import Mock, patch, AsyncMock
from tldw_chatbook.TTS.backends.elevenlabs import ElevenLabsTTSBackend
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest

@pytest.mark.asyncio
async def test_elevenlabs_uses_dynamic_settings():
    """Test that ElevenLabs backend uses request-specific settings"""
    backend = ElevenLabsTTSBackend({"ELEVENLABS_API_KEY": "test"})
    
    request = OpenAISpeechRequest(
        input="Test text",
        voice="test-voice",
        response_format="mp3"
    )
    request.extra_params = {
        "stability": 0.9,
        "similarity_boost": 0.3,
        "style": 0.7,
        "use_speaker_boost": False
    }
    
    with patch.object(backend, 'client') as mock_client:
        mock_response = AsyncMock()
        mock_client.stream.return_value.__aenter__.return_value = mock_response
        
        # Capture the payload
        captured_payload = None
        async def capture_payload(*args, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get('json', {})
            return mock_response
        
        mock_client.stream.side_effect = capture_payload
        
        # Generate audio
        async for _ in backend.generate_audio(request):
            break
        
        # Verify settings were used
        assert captured_payload is not None
        assert captured_payload['voice_settings']['stability'] == 0.9
        assert captured_payload['voice_settings']['similarity_boost'] == 0.3
        assert captured_payload['voice_settings']['style'] == 0.7
        assert captured_payload['voice_settings']['use_speaker_boost'] is False
```

### 4.2 Integration Tests

```python
# Add to Tests/integration/test_tts_playground_integration.py:

@pytest.mark.asyncio
async def test_voice_blend_appears_in_dropdown(app):
    """Test that saved voice blends appear in the Kokoro dropdown"""
    # Create a test blend
    blend_data = {
        "test_blend": {
            "voices": [("af_bella", 0.6), ("af_sarah", 0.4)],
            "description": "Test blend"
        }
    }
    
    blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
    blend_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(blend_file, 'w') as f:
        json.dump(blend_data, f)
    
    # Open TTS playground
    await app.push_screen("stts_window")
    playground = app.query_one("TTSPlaygroundWidget")
    
    # Select Kokoro provider
    provider_select = playground.query_one("#tts-provider-select")
    provider_select.value = "kokoro"
    
    # Check voice options
    voice_select = playground.query_one("#tts-voice-select")
    voice_options = [opt[0] for opt in voice_select._options]
    
    assert "blend:test_blend" in voice_options
    
    # Cleanup
    blend_file.unlink()
```

## Implementation Timeline

### Week 1: Critical Fixes
- Day 1-2: Fix provider-specific settings pass-through
- Day 3: Fix voice blend integration  
- Day 4: Fix file picker usage
- Day 5: Testing and validation

### Week 2: AudioBook Backend
- Day 1-2: Implement audio processing functions
- Day 3: Add format conversion support
- Day 4: Complete audiobook generation flow
- Day 5: Testing with real audiobook generation

### Week 3: Enhanced Features
- Day 1: Add cost estimation
- Day 2: Implement progress tracking
- Day 3: Add voice preview functionality
- Day 4-5: Final testing and bug fixes

## Success Criteria

1. All provider-specific settings are applied correctly
2. Voice blends appear in and work from the playground
3. AudioBook generation completes successfully
4. Import/Export uses proper file dialogs
5. All tests pass
6. No hardcoded paths remain
7. Progress feedback is accurate
8. Cost estimation is within 10% of actual

## Risk Mitigation

1. **FFmpeg Dependency**: Provide fallback for systems without FFmpeg
2. **Large File Handling**: Implement streaming for audiobooks over 1GB
3. **API Rate Limits**: Add retry logic with exponential backoff
4. **Memory Usage**: Process audiobooks in chunks, not all at once
5. **Cross-Platform**: Test on Windows, macOS, and Linux

## Notes for Developers

- Always validate numeric inputs before passing to backends
- Use proper async/await patterns for long-running operations
- Log all errors with context for debugging
- Update documentation when adding new features
- Consider adding telemetry for usage patterns
- Keep UI responsive during generation