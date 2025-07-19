# TTS Module Implementation Summary

## Overview
This document summarizes the fixes and enhancements made to the TTS (Text-to-Speech) module in the tldw_chatbook application.

## Completed Tasks

### 1. Enhanced Audio Player with Pause/Resume Support
**Files Modified:** `/tldw_chatbook/TTS/audio_player.py`

- Added `PAUSED` state to `PlaybackState` enum
- Implemented `pause()` and `resume()` methods in `SimpleAudioPlayer`
- Added platform-specific pause support:
  - **macOS**: Stop/restart fallback (afplay doesn't support pause)
  - **Linux**: Native pause for mpv/mplayer, fallback for others
  - **Windows**: COM automation support (future enhancement)
- Added position tracking for pause/resume functionality
- Implemented `AsyncAudioPlayer` wrapper with pause/resume methods

### 2. TTS Playground UI Enhancements
**Files Modified:** `/tldw_chatbook/UI/STTS_Window.py`

#### Audio Control Buttons
- Added pause/resume button with dynamic label switching
- Added stop button for immediate playback termination
- Added export button with file picker dialog
- Fixed button ID consistency (`pause-audio-btn`, `stop-audio-btn`, `export-audio-btn`)

#### Playback Progress Tracking
- Added `ProgressBar` widget for visual playback progress
- Added time display showing current/total duration (MM:SS format)
- Implemented `_update_progress_timer()` method for real-time updates
- Added CSS classes for showing/hiding progress elements

### 3. Provider-Specific Settings Integration
**Files Modified:** `/tldw_chatbook/TTS/backends/elevenlabs.py`, `/tldw_chatbook/TTS/backends/kokoro.py`

#### ElevenLabs Backend
- Modified to accept dynamic voice settings from `request.extra_params`
- Supports dynamic `stability` and `similarity_boost` parameters
- Settings now flow from UI sliders to backend API calls

#### Kokoro Backend
- Added support for language parameter from `extra_params`
- Fixed voice blend integration
- Language selection now properly propagates to backend

### 4. Voice Blend Integration
**Files Modified:** `/tldw_chatbook/UI/STTS_Window.py`, `/tldw_chatbook/Widgets/voice_blend_dialog.py`

- Implemented voice blend loading from JSON file
- Added voice blends to all dropdown locations:
  - TTS Playground
  - TTS Settings
  - AudioBook Generation
- Voice blends display with "blend:" prefix and 🎭 emoji
- Created `VoiceBlendDialog` for creating/editing blends

### 5. File Operation Improvements
**Files Modified:** `/tldw_chatbook/UI/STTS_Window.py`

- Replaced hardcoded Downloads folder paths with file picker dialogs
- Added proper file import/export with callbacks
- Implemented error handling for file operations
- Added file validation and user feedback

### 6. Audio Processing Functions
**Files Modified:** `/tldw_chatbook/TTS/audiobook_generator.py`

- Implemented `_generate_silence()` with pydub and fallback methods
- Implemented `_normalize_audio()` with target dB level support
- Implemented `_get_audio_duration()` with multiple detection methods
- Added proper error handling and logging

### 7. Content Import Dialogs
**New Files Created:**
- `/tldw_chatbook/Widgets/note_selection_dialog.py`
- `/tldw_chatbook/Widgets/conversation_selection_dialog.py`

#### NoteSelectionDialog
- Multi-select interface for notes
- Search functionality
- Preview of note content
- Select all/clear all functionality

#### ConversationSelectionDialog
- Single-select interface for conversations
- Message filtering options (all/user/assistant)
- Speaker name formatting options
- Search functionality

## Architecture Overview

### Event Flow
1. User interaction in UI triggers event (e.g., `STTSPlaygroundGenerateEvent`)
2. Event handler in `stts_events.py` processes request
3. TTS service generates audio through appropriate backend
4. Audio file saved to temporary location
5. UI updated with generation status and file path
6. Audio player initialized for playback control

### Key Components
- **UI Layer**: `STTS_Window.py` with three main widgets:
  - `TTSPlaygroundWidget`: Testing and experimentation
  - `TTSSettingsWidget`: Configuration management
  - `AudioBookGenerationWidget`: Long-form content generation
  
- **Event System**: 
  - `STTSPlaygroundGenerateEvent`: TTS generation requests
  - `STTSSettingsSaveEvent`: Settings persistence
  - `STTSAudioBookGenerateEvent`: AudioBook generation
  
- **Audio Backend**:
  - `SimpleAudioPlayer`: Cross-platform audio playback
  - `AsyncAudioPlayer`: Async wrapper for Textual integration
  
- **TTS Backends**:
  - OpenAI, ElevenLabs, Kokoro, Chatterbox, AllTalk
  - Each with specific configuration options

## Usage Guide

### TTS Playground
1. Enter text in the input area
2. Select provider, model, voice, and format
3. Adjust provider-specific settings (if available)
4. Click "Generate" to create audio
5. Use playback controls to play/pause/stop
6. Export audio to desired location

### Voice Blends (Kokoro)
1. Click "Manage Voice Blends" in settings
2. Create new blend or edit existing
3. Add multiple voices with weights
4. Weights are automatically normalized
5. Save blend for use in any TTS generation

### AudioBook Generation
1. Import content from:
   - File (txt, md, etc.)
   - Notes (multi-select)
   - Conversation (with filtering)
   - Clipboard paste
2. Configure chapter detection
3. Set voice and format options
4. Generate complete audiobook

## Configuration

### Settings Storage
- Main config: `~/.config/tldw_cli/config.toml`
- Voice blends: `~/.config/tldw_cli/kokoro_voice_blends.json`

### Environment Variables
- `ELEVENLABS_API_KEY`: ElevenLabs API key
- `OPENAI_API_KEY`: OpenAI API key
- `KOKORO_MODEL_PATH`: Path to Kokoro models
- `CHATTERBOX_VOICE_DIR`: Directory for voice samples

## Performance Considerations

### Audio Playback
- Playback progress updates every 100ms
- Audio files stored in system temp directory
- Automatic cleanup of temp files on app exit

### TTS Generation
- Streaming generation for large texts
- Chunk size optimization per provider
- Background worker threads for non-blocking UI

## Security Notes

- API keys stored in config file (consider encryption)
- Temp audio files have restricted permissions
- Input validation on all user-provided text
- Path validation for file operations

## Future Enhancements

1. **Playback Features**
   - Playback speed control
   - Seek bar for position control
   - Volume control integration

2. **TTS Features**
   - Batch processing for multiple texts
   - Voice cloning integration
   - Real-time streaming TTS

3. **UI Improvements**
   - Waveform visualization
   - Generation queue management
   - Preset management system

## Troubleshooting

### Common Issues

1. **No audio playback**
   - Check system audio player availability
   - Verify file permissions in temp directory
   - Ensure audio format compatibility

2. **TTS generation fails**
   - Verify API keys are set correctly
   - Check network connectivity
   - Ensure sufficient API credits

3. **Voice blends not appearing**
   - Check JSON file exists and is valid
   - Verify file permissions
   - Restart app after creating blends

### Debug Mode
Enable debug logging by setting log level in config:
```toml
[logging]
level = "DEBUG"
```

## Testing

### Manual Testing Checklist
- [ ] Generate audio with each provider
- [ ] Test pause/resume on each platform
- [ ] Export audio in different formats
- [ ] Create and use voice blends
- [ ] Import content from all sources
- [ ] Generate audiobook with chapters

### Automated Tests
Located in `/Tests/TTS/` and `/Tests/UI/`:
- `test_audio_player.py`: Audio playback functionality
- `test_voice_blend_dialog.py`: Voice blend UI
- Integration tests for TTS backends

## Dependencies

### Required
- `pydub`: Audio processing and format conversion
- `textual`: TUI framework
- `httpx`: Async HTTP client
- `loguru`: Logging

### Optional
- `elevenlabs`: ElevenLabs API client
- `openai`: OpenAI API client
- `onnxruntime`: Kokoro model inference

## API Documentation

See backend-specific documentation:
- OpenAI TTS: https://platform.openai.com/docs/guides/text-to-speech
- ElevenLabs: https://docs.elevenlabs.io/
- Kokoro: Local model documentation