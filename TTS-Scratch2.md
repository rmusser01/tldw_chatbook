# TTS Implementation Scratch Notes

## Current State Analysis

### Working Components:
1. UI Layout and Controls
2. Event System Wiring
3. Basic TTS Generation (OpenAI works)
4. Voice Blend Dialog UI
5. Settings persistence

### Broken/Incomplete:
1. Provider-specific settings not applied at runtime
2. AudioBook backend has TODOs
3. Voice blends not integrated with playground
4. Import/Export uses hardcoded paths
5. Missing audio processing capabilities

## Code Issues Found:

### 1. ElevenLabs Backend (backends/elevenlabs.py)
- Line 119-123: Uses class-level settings, ignores request params
- Need to check if request has extra_params and override defaults

### 2. Event Handler (STTS_Events/stts_events.py)
- Line 169-170: Attempts to update request_dict but doesn't pass to backend
- The `generate_audio_stream` call doesn't accept the modified dict

### 3. AudioBook Generator (audiobook_generator.py)
- Line 711: TODO - silence generation
- Line 721: TODO - audio normalization  
- Line 727: TODO - duration detection
- Missing FFmpeg integration for format conversion

### 4. Voice Blend Integration
- Kokoro backend supports "blend:name" format (line 533-541)
- UI saves blends but doesn't populate them in dropdown
- Need to load blends and add to voice selection

### 5. File Paths
- Line 1441, 1500: Hardcoded Downloads paths
- Should use enhanced file picker

## Technical Debt:
- No error recovery in audiobook generation
- No cost estimation before generation
- No batch processing support
- Limited progress feedback
- No audio preview capability

## Dependencies Check:
- FFmpeg needed for audio processing
- pydub for audio manipulation
- mutagen for metadata
- numpy for audio normalization