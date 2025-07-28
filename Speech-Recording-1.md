# Speech Recording & Live Dictation Implementation Plan

## Overview

This document outlines the plan to implement a live speech recording and dictation library for the tldw_chatbook project. The goal is to enable real-time voice transcription and recording capabilities that can be integrated into the existing S/TT/S tab and potentially used throughout the application for voice input.

## Current State Analysis

### Existing Infrastructure
1. **STTS Tab**: Already exists (`TAB_STTS`) with UI window (`UI/STTS_Window.py`)
2. **Transcription Service**: Existing `transcription_service.py` with:
   - Support for multiple backends (faster-whisper, parakeet-mlx, etc.)
   - `ParakeetMLXStreamingTranscriber` class for streaming transcription
   - Real-time ASR support for parakeet-mlx on Apple Silicon
3. **Audio Dependencies**: 
   - `pyaudio` is already listed in optional dependencies
   - Support for soundfile, scipy, numpy for audio processing
4. **UI Infrastructure**: 
   - CSS styling for `.mic-button` already exists in Constants.py
   - Event handling system in place for STTS events

### Gaps to Fill
1. No actual microphone capture/recording implementation
2. No live dictation UI component
3. No integration between microphone input and streaming transcription
4. No cross-platform audio recording solution

## Proposed Architecture

### Core Components

#### 1. Audio Recording Service (`tldw_chatbook/Audio/recording_service.py`)
```python
class AudioRecordingService:
    """Cross-platform audio recording service with streaming support"""
    
    def __init__(self):
        # Initialize audio backend (pyaudio, sounddevice, etc.)
        # Configure audio parameters (sample rate, channels, etc.)
        
    def start_recording(self, callback=None):
        """Start recording audio from microphone"""
        
    def stop_recording(self):
        """Stop recording and return audio data"""
        
    def get_audio_devices(self):
        """List available audio input devices"""
        
    def set_device(self, device_id):
        """Set the active recording device"""
```

#### 2. Live Dictation Service (`tldw_chatbook/Audio/dictation_service.py`)
```python
class LiveDictationService:
    """Combines audio recording with real-time transcription"""
    
    def __init__(self, transcription_provider='auto'):
        # Initialize recording service
        # Initialize transcription service
        # Set up streaming pipeline
        
    def start_dictation(self, callback):
        """Start live dictation with transcription callback"""
        
    def stop_dictation(self):
        """Stop dictation and return final transcript"""
        
    def pause_dictation(self):
        """Pause dictation temporarily"""
        
    def resume_dictation(self):
        """Resume paused dictation"""
```

#### 3. UI Components

##### a. Voice Input Widget (`tldw_chatbook/Widgets/voice_input_widget.py`)
```python
class VoiceInputWidget(Widget):
    """Reusable voice input widget with visual feedback"""
    
    - Record button with states (idle, recording, processing)
    - Audio level indicator
    - Transcription preview
    - Device selection dropdown
```

##### b. Dictation Window (`tldw_chatbook/UI/Dictation_Window.py`)
```python
class DictationWindow(Container):
    """Full dictation interface for the STTS tab"""
    
    - Live transcription display
    - Recording controls
    - Settings (language, model, device)
    - Export options
```

#### 4. Event Handlers (`tldw_chatbook/Event_Handlers/Audio_Events/`)
- `recording_events.py`: Handle recording start/stop/error events
- `dictation_events.py`: Handle transcription updates, final results

### Implementation Details

#### Audio Backend Strategy
1. **Primary**: PyAudio for cross-platform support
2. **Fallback**: sounddevice (more modern, better error handling)
3. **Platform-specific optimizations**:
   - macOS: Use Core Audio for lower latency
   - Windows: WASAPI for better performance
   - Linux: ALSA/PulseAudio support

#### Streaming Pipeline
```
Microphone → Audio Buffer → VAD Filter → Transcription Service → UI Update
     ↓             ↓              ↓                ↓                  ↓
  PyAudio    Ring Buffer    WebRTCVAD      Parakeet/Whisper    Textual Event
```

#### Key Features
1. **Voice Activity Detection (VAD)**: Filter silence to optimize transcription
2. **Automatic punctuation**: Add punctuation to transcribed text
3. **Command detection**: Recognize voice commands (e.g., "new paragraph", "comma")
4. **Multi-language support**: Leverage existing transcription service capabilities
5. **Keyboard shortcuts**: Push-to-talk, toggle recording, etc.

### Integration Points

#### 1. Chat Window Integration
- Add microphone button next to send button (CSS already exists)
- Voice input replaces/appends to text input
- Optional continuous dictation mode

#### 2. Notes Window Integration
- Voice notes feature
- Dictate directly into notes
- Voice commands for formatting

#### 3. STTS Tab Enhancement
- Dedicated dictation section
- Advanced settings and controls
- Transcription history

### Cross-Platform Considerations

#### macOS
- Leverage parakeet-mlx for optimal performance on Apple Silicon
- Request microphone permissions via Info.plist
- Use Core Audio for lower latency

#### Windows
- Use Windows audio APIs for better integration
- Handle Windows Defender/firewall prompts
- Test with various audio drivers

#### Linux
- Support both ALSA and PulseAudio
- Handle permission issues gracefully
- Test on major distributions

### Error Handling

1. **Permission Denied**: Clear user guidance for enabling microphone access
2. **No Audio Device**: Graceful fallback and device selection UI
3. **Network Issues**: For cloud-based transcription, queue and retry
4. **Resource Constraints**: Adaptive quality based on system performance

### Testing Strategy

1. **Unit Tests**:
   - Audio buffer management
   - VAD accuracy
   - Transcription pipeline

2. **Integration Tests**:
   - End-to-end dictation flow
   - Multi-provider transcription
   - UI responsiveness

3. **Platform Tests**:
   - Cross-platform audio capture
   - Permission handling
   - Performance benchmarks

### Performance Optimization

1. **Audio Processing**:
   - Use numpy for efficient audio manipulation
   - Implement circular buffers for streaming
   - Optimize sample rate conversion

2. **Transcription**:
   - Batch audio chunks intelligently
   - Use VAD to skip silence
   - Cache frequently used models

3. **UI Updates**:
   - Debounce transcription updates
   - Use Textual's worker threads
   - Implement virtual scrolling for long transcripts

### Security & Privacy

1. **Local Processing**: Prioritize on-device transcription
2. **Audio Storage**: Temporary buffers only, no persistent storage by default
3. **User Consent**: Clear indication when recording is active
4. **Data Handling**: Follow GDPR/privacy best practices

### Future Extensions

1. **Meeting Recording**: 
   - System audio capture for Zoom/Teams
   - Speaker diarization
   - Meeting summaries

2. **Voice Commands**:
   - Control application via voice
   - Custom command recognition
   - Workflow automation

3. **Audio Enhancement**:
   - Noise reduction
   - Echo cancellation
   - Automatic gain control

## Implementation Phases

### Phase 1: Core Recording (Week 1)
- [ ] Implement AudioRecordingService
- [ ] Cross-platform audio capture
- [ ] Basic error handling
- [ ] Unit tests

### Phase 2: Transcription Integration (Week 2)
- [ ] Implement LiveDictationService
- [ ] Connect to existing transcription service
- [ ] Streaming pipeline
- [ ] VAD integration

### Phase 3: UI Components (Week 3)
- [ ] Create VoiceInputWidget
- [ ] Implement DictationWindow
- [ ] Add to STTS tab
- [ ] Event handlers

### Phase 4: Application Integration (Week 4)
- [ ] Chat window microphone button
- [ ] Notes voice input
- [ ] Keyboard shortcuts
- [ ] Settings/configuration

### Phase 5: Polish & Testing (Week 5)
- [ ] Cross-platform testing
- [ ] Performance optimization
- [ ] Documentation
- [ ] Example scripts

## Dependencies to Add

```toml
[project.optional-dependencies]
speech_recording = [
    "pyaudio>=0.2.14",  # Primary audio backend
    "sounddevice>=0.4.6",  # Fallback audio backend
    "webrtcvad>=2.0.10",  # Voice activity detection
    "numpy>=1.24.0",  # Audio processing (already included)
    "scipy>=1.10.0",  # Signal processing (already included)
]
```

## Configuration Schema

```toml
[speech_recording]
# Audio settings
sample_rate = 16000
channels = 1
chunk_size = 1024
device_id = "default"  # or specific device ID

# Recording settings
vad_enabled = true
vad_aggressiveness = 2  # 0-3, higher is more aggressive
auto_adjust_gain = true
noise_reduction = true

# Transcription settings
transcription_provider = "auto"  # auto, parakeet-mlx, faster-whisper, etc.
transcription_model = "auto"
language = "en"
enable_punctuation = true

# UI settings
show_waveform = true
show_transcription_preview = true
push_to_talk = false  # vs continuous recording
```

## Risks & Mitigation

1. **Risk**: PyAudio installation issues
   - **Mitigation**: Provide pre-built wheels, fallback to sounddevice

2. **Risk**: Real-time performance on low-end hardware
   - **Mitigation**: Adaptive quality settings, option to disable real-time transcription

3. **Risk**: Microphone permission complexities
   - **Mitigation**: Clear documentation, OS-specific guides, graceful degradation

## Success Criteria

1. **Functionality**:
   - ✓ Can record audio from microphone across all platforms
   - ✓ Real-time transcription with <2 second latency
   - ✓ Seamless integration with existing UI

2. **Performance**:
   - ✓ <5% CPU usage during idle recording
   - ✓ <100MB memory footprint
   - ✓ No UI lag during transcription

3. **User Experience**:
   - ✓ One-click recording start/stop
   - ✓ Clear visual feedback
   - ✓ Intuitive keyboard shortcuts

## Post-Implementation Concerns

1. **Maintenance**: Keep audio backends updated for OS compatibility
2. **Feature Creep**: Resist adding too many audio processing features initially
3. **Documentation**: Ensure clear user guides for setup and troubleshooting
4. **Community**: Encourage contributions for additional language models