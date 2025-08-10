# TTS/Dictation Improvements Plan

## Overview
This document outlines the planned improvements for the TTS (Text-to-Speech) and Dictation (Speech-to-Text) features in tldw_chatbook, based on architectural review conducted on 2025-07-30.

## Current State Analysis

### Key Issues Identified

1. **Architectural Clarity Issue**
   - Dictation (STT) functionality is incorrectly integrated within the STTS tab
   - STTS naming is ambiguous - contains both TTS and STT features
   - Conceptual confusion between inverse operations (TTS vs STT)

2. **Resource Management**
   - Continuous audio recording threads even when idle
   - No automatic cleanup of audio buffers
   - Processing threads continue until explicitly stopped

3. **Error Handling**
   - Audio backend initialization failures not gracefully handled at UI level
   - Missing user-friendly fallback messages
   - No guidance for common issues (permissions, device selection)

4. **Feature Completeness**
   - Limited voice command vocabulary
   - No integration between dictation and other app features
   - Basic export functionality (text/markdown only)

5. **Performance Considerations**
   - Fixed buffer duration (500ms) may not be optimal
   - Continuous buffering during silence
   - No adaptive quality settings

6. **Security/Privacy**
   - Transcription history saved unencrypted
   - No option to disable history
   - Unclear audio data cleanup timing

## Recommended Improvements

### 1. UI/UX Reorganization
- [ ] Create dedicated "Dictation" or "Speech Input" tab
- [ ] Rename STTS tab to "Speech Services" or split into separate tabs
- [ ] Add visual indicators for microphone activity
- [ ] Add transcription status indicators

### 2. Architecture Improvements
- [ ] Implement lazy initialization of audio backends
- [ ] Add graceful degradation for unavailable hardware
- [ ] Simplify threading model for single-user app
- [ ] Add configurable audio buffer sizes

### 3. Feature Enhancements
- [ ] Add integration points for dictated text insertion
- [ ] Implement audio recording export
- [ ] Expand voice command vocabulary
- [ ] Support custom voice command definitions

### 4. Privacy & Security
- [ ] Add option to disable transcription history
- [ ] Implement optional encryption for saved transcriptions
- [ ] Clear audio buffers immediately after processing
- [ ] Add privacy mode (local processing only)

### 5. Error Handling
- [ ] Implement user-friendly error messages
- [ ] Add microphone permission checks
- [ ] Provide fallback options for backend failures
- [ ] Add audio device troubleshooting UI

### 6. Performance Optimization
- [ ] Implement adaptive buffering based on VAD
- [ ] Add idle period processing suspension
- [ ] Optimize thread usage
- [ ] Add performance metrics/tuning

### 7. Integration Improvements
- [ ] Create bidirectional workflow (Dictate → Edit → TTS)
- [ ] Enable dictation into chat conversations
- [ ] Support dictation in notes
- [ ] Voice commands for app control

## Implementation Priority

### Phase 1: Critical Fixes (High Priority)
1. Fix architectural confusion (separate tabs/clear naming)
2. Improve error handling and user feedback
3. Add privacy options for history

### Phase 2: Core Improvements (Medium Priority)
1. Optimize resource management
2. Enhance voice commands
3. Add integration points

### Phase 3: Advanced Features (Low Priority)
1. Bidirectional workflows
2. Performance tuning options
3. Advanced privacy features

## Architecture Decision Records (ADRs)

### ADR-001: Separate Dictation from TTS UI
**Date**: 2025-07-30
**Status**: Proposed
**Context**: The current STTS tab combines both TTS and STT functionality, creating user confusion.
**Decision**: Create separate tabs for TTS and Dictation functionality.
**Consequences**: 
- Clearer user mental model
- More focused UI for each feature
- Potential code duplication for shared components

### ADR-002: Lazy Audio Backend Initialization
**Date**: 2025-07-30
**Status**: Proposed
**Context**: Audio backends are initialized on service creation, even if never used.
**Decision**: Initialize audio backends only when first needed.
**Consequences**:
- Reduced startup time
- Lower resource usage when features not used
- Slight delay on first use

### ADR-003: Privacy-First Transcription History
**Date**: 2025-07-30
**Status**: Proposed
**Context**: Transcription history is always saved, unencrypted.
**Decision**: Make history opt-in with encryption option.
**Consequences**:
- Better privacy for users
- Additional complexity in settings
- Potential loss of convenience features

## Technical Notes

### Current Architecture
```
STTS_Window
├── TTSPlaygroundWidget
├── TTSSettingsWidget
├── AudioBookGenerationWidget
└── DictationWindow (misplaced?)
```

### Proposed Architecture
```
SpeechServicesTab
├── TTS Tab
│   ├── TTSPlaygroundWidget
│   ├── TTSSettingsWidget
│   └── AudioBookGenerationWidget
└── Dictation Tab
    ├── DictationWindow
    ├── TranscriptionHistory
    └── VoiceCommandSettings
```

### Key Files to Modify
- `/tldw_chatbook/UI/STTS_Window.py` - Split or rename
- `/tldw_chatbook/UI/Dictation_Window.py` - Enhance and relocate
- `/tldw_chatbook/Audio/dictation_service.py` - Optimize resource usage
- `/tldw_chatbook/Constants.py` - Add new tab constants

### ADR-004: Implementation Approach for Tab Separation
**Date**: 2025-07-30
**Status**: Accepted
**Context**: Need to separate TTS and Dictation functionality with minimal disruption.
**Decision**: Keep the existing STTS tab but reorganize it internally with clear separation:
1. Rename STTS to "Speech Services" in UI
2. Add clear navigation between TTS and Dictation sections
3. Keep unified tab to avoid breaking existing user workflows
**Consequences**:
- Less disruptive than creating entirely new tabs
- Maintains backward compatibility
- Still provides clear separation of concerns
- Easier migration path

## Testing Requirements
- Cross-platform audio device testing
- Performance benchmarks before/after optimization
- Privacy mode verification
- Error scenario coverage

## Documentation Updates Needed
- User guide for new tab structure
- Privacy settings documentation
- Voice command reference
- Troubleshooting guide

## Implementation Progress

### Phase 1: UI Reorganization (In Progress)

#### Created Speech_Services_Window.py
- New window class that clearly separates TTS and Dictation
- Improved sidebar organization with distinct sections
- Mode indicator showing current service (TTS or STT)
- Descriptive text for each service type
- Keyboard shortcuts for quick switching (Ctrl+T, Ctrl+D)
- Placeholder buttons for future features

#### Key Improvements:
1. **Clear Visual Separation**: TTS and Dictation are in separate sections with dividers
2. **Mode Awareness**: Current mode is prominently displayed
3. **Service Descriptions**: Each section explains what it does
4. **Future-Ready**: Disabled buttons show planned features
5. **Better Navigation**: Logical grouping of related functions

### Phase 2: Core Improvements (Completed)

#### Implemented Lazy Initialization (dictation_service_lazy.py)
- Audio and transcription services initialize only when first needed
- Graceful error handling with user-friendly messages
- No resource consumption when features not in use
- Clear error messages guide users to solutions

#### Implemented Privacy Settings (Dictation_Window_Improved.py)
- Privacy-first defaults (local processing, no history)
- Toggle for saving transcription history
- Option for local-only processing (no cloud services)
- Auto-clear audio buffers for privacy
- Visual indicators for privacy status

#### Key Features Added:
1. **Resource Optimization**:
   - Services start only when dictation begins
   - Configurable buffer sizes (100-2000ms)
   - Automatic cleanup of audio data
   - Simplified threading for single-user app

2. **Error Handling**:
   - User-friendly error messages
   - Specific guidance for common issues
   - Graceful degradation when hardware unavailable
   - Status indicators in UI

3. **Privacy Controls**:
   - Local-only mode restricts to on-device providers
   - Optional history with encryption capability
   - Immediate audio buffer clearing
   - Clear privacy status indicators

### ADR-005: Privacy-First Design
**Date**: 2025-07-30
**Status**: Implemented
**Context**: Users need control over their voice data and transcriptions.
**Decision**: Implement privacy-first defaults with opt-in for features that store data.
**Consequences**:
- Better user trust
- Compliance with privacy expectations
- Some convenience features require explicit opt-in

### Next Steps:
1. Update app.py to use new improved components
2. Add integration points for dictation output to other features
3. Implement audio device troubleshooting UI
4. Create comprehensive documentation

### Summary of Improvements:
- ✅ Clear separation of TTS and Dictation
- ✅ Lazy initialization for better resource usage
- ✅ Privacy-first settings with user control
- ✅ Improved error handling with guidance
- ✅ Performance optimization options
- ✅ Integration points for voice input across app
- ✅ Audio device troubleshooting UI
- ✅ Expanded voice command system
- ✅ Encrypted transcription history
- ✅ Performance monitoring and metrics

## Complete Implementation Summary

### Files Created/Modified

#### UI Components
1. **Speech_Services_Window.py** - Main window with clear TTS/Dictation separation
2. **STTS_Window_Updated.py** - Backward-compatible wrapper
3. **Dictation_Window_Improved.py** - Enhanced dictation UI with privacy controls

#### Core Services
1. **dictation_service_lazy.py** - Lazy-loading dictation service
2. **voice_commands.py** - Expanded voice command system
3. **transcription_history.py** - Encrypted history management
4. **dictation_metrics.py** - Performance monitoring

#### Widgets
1. **voice_input_button.py** - Reusable voice input for any text field
2. **audio_troubleshooting_dialog.py** - Device diagnostics and testing
3. **voice_command_dialog.py** - Command configuration UI
4. **transcription_history_viewer.py** - History browser with search
5. **dictation_performance_widget.py** - Performance dashboard

#### Events
1. **dictation_integration_events.py** - Cross-app integration events

#### Documentation
1. **Speech-Services-Guide.md** - Comprehensive user guide
2. **TTS-Improve-1.md** - This implementation document

### Key Achievements

1. **Better Architecture**
   - Clear separation of concerns
   - Lazy initialization reduces resource usage
   - Event-driven integration with rest of app

2. **Privacy & Security**
   - Privacy-first defaults
   - Optional encrypted history
   - Local-only processing mode
   - Automatic audio buffer clearing

3. **User Experience**
   - Intuitive UI organization
   - Built-in troubleshooting
   - Voice input anywhere
   - Comprehensive help

4. **Performance**
   - Configurable buffer sizes
   - Performance monitoring
   - Provider comparison metrics
   - Resource usage tracking

5. **Extensibility**
   - Custom voice commands
   - Integration events
   - Reusable components
   - Well-documented APIs

### Usage Examples

#### Quick Voice Input
```python
# Add to any widget
yield VoiceInputButton(
    on_result=lambda text: self.insert_text(text)
)
```

#### Integration with Chat
```python
# In chat window
def on_dictation_output_event(self, event):
    if event.target == "chat":
        self.add_message(event.text)
```

#### Custom Commands
```python
# User can add custom commands like:
command = VoiceCommand(
    phrase="insert timestamp",
    action="insert_timestamp",
    parameters={"format": "%Y-%m-%d %H:%M"}
)
```

### Testing Recommendations

1. **Cross-platform Testing**
   - Test on macOS, Linux, Windows
   - Verify audio device detection
   - Check permission prompts

2. **Privacy Testing**
   - Verify local-only mode restrictions
   - Test history encryption
   - Check buffer clearing

3. **Performance Testing**
   - Monitor resource usage
   - Test with long dictation sessions
   - Compare provider performance

4. **Integration Testing**
   - Test voice input in all text fields
   - Verify event propagation
   - Test command execution

---
**Document Created**: 2025-07-30
**Last Updated**: 2025-07-30
**Author**: Claude (AI Assistant)