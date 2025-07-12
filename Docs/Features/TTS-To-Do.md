# TTS (Text-to-Speech) To-Do List

This document tracks the remaining tasks and future enhancements for the TTS/S/TT/S (Speech/Text-to-Speech) system in tldw_chatbook.

## ✅ Completed Features

### Core TTS Functionality
- [x] OpenAI TTS backend implementation with streaming support
- [x] ElevenLabs TTS backend with voice mapping and format conversion
- [x] Kokoro local TTS backend with both ONNX and PyTorch support
- [x] Automatic model and voice pack downloading for Kokoro
- [x] Audio format conversion service (MP3, WAV, OPUS, AAC, FLAC, PCM)
- [x] Text processing and chunking for long texts
- [x] Secure temporary file handling for audio files
- [x] **Simple cross-platform audio player** (NEW)
- [x] **Audio export with custom naming and metadata** (NEW)
- [x] **Progress tracking for TTS generation** (NEW)
- [x] **Cost tracking and usage statistics** (NEW)

### S/TT/S Tab UI
- [x] Main S/TT/S tab with navigation sidebar
- [x] TTS Playground for testing different providers and settings
- [x] TTS Settings window for global configuration
- [x] AudioBook/Podcast Generation window UI
- [x] Dynamic voice/model selection based on provider
- [x] Real-time generation progress logging
- [x] Audio playback controls (play, pause, stop, export)
- [x] Settings persistence to config file
- [x] **Progress bar and time estimation display** (NEW)
- [x] **Cost estimation before generation** (NEW)

### Integration
- [x] Event-driven architecture for TTS operations
- [x] Integration with main app event system
- [x] Rate limiting and cooldown for TTS requests
- [x] Error handling and user notifications
- [x] Performance metrics tracking for Kokoro
- [x] **Automatic cooldown cleanup to prevent memory leaks** (NEW)
- [x] **Simplified singleton pattern for single-user app** (NEW)

### Security & Stability
- [x] **Fixed PowerShell command injection vulnerability** (NEW)
- [x] **Fixed HTTPx response handling errors** (NEW)
- [x] **Removed sensitive data from error logs** (NEW)
- [x] **Improved configuration override ordering** (NEW)

## 📋 Remaining Tasks

### High Priority

#### 1. **AudioBook/Podcast Generation Backend** 🎙️
- [ ] Implement chapter detection from text content
- [ ] Add support for multiple voices per chapter/character
- [ ] Implement batch processing for long documents
- [ ] Add progress tracking for multi-chapter generation
- [ ] Create metadata embedding (title, author, chapters)
- [ ] Add background music/sound effects support
- [ ] Implement pause/resume for long generation tasks

#### 2. **Export and File Management** 💾
- [x] ~~Implement audio file export with custom naming~~ ✅ COMPLETED
- [ ] Add support for M4B audiobook format
- [ ] Create playlist generation for multi-file exports
- [ ] Add audio file compression options
- [ ] Implement batch export functionality

#### 3. **Advanced Text Processing** 📝
- [ ] SSML (Speech Synthesis Markup Language) support
- [ ] Automatic punctuation and formatting cleanup
- [ ] Intelligent sentence breaking for natural pauses
- [ ] Pronunciation dictionary/corrections
- [ ] Language detection and switching
- [ ] Emotion and emphasis markup support

### Medium Priority

#### 4. **Voice Cloning (ElevenLabs)** 🎭
- [ ] UI for uploading voice samples
- [ ] Voice training progress tracking
- [ ] Custom voice management (list, delete, rename)
- [ ] Voice sharing/export functionality
- [ ] Voice preview before generation

#### 5. **Speech Recognition (STT)** 🎤
- [ ] Implement Whisper integration for local STT
- [ ] Add real-time transcription support
- [ ] Create audio file import and transcription
- [ ] Implement transcription editing UI
- [ ] Add speaker diarization support
- [ ] Export transcriptions in various formats

#### 6. **Audio Effects and Post-Processing** 🎵
- [ ] Volume normalization across segments
- [ ] Noise reduction filters
- [ ] Echo/reverb effects
- [ ] Speed adjustment without pitch change
- [ ] Silence trimming and padding
- [ ] Audio crossfading between segments

#### 7. **Cost Tracking and Limits** 💰
- [x] ~~Implement per-provider cost estimation~~ ✅ COMPLETED
- [x] ~~Add usage tracking and statistics~~ ✅ COMPLETED
- [ ] Create budget limits and warnings
- [x] ~~Show cost before generation~~ ✅ COMPLETED
- [ ] Monthly/daily usage reports UI
- [ ] Cost optimization suggestions

### Low Priority

#### 8. **Additional TTS Providers** 🔌
- [ ] Google Cloud Text-to-Speech
- [ ] Amazon Polly
- [ ] Microsoft Azure TTS
- [ ] Coqui TTS (local)
- [ ] Piper TTS (local)
- [ ] Custom OpenAI-compatible endpoints

#### 9. **Advanced Features** ⚡
- [ ] Real-time streaming to audio devices
- [ ] Live preview during typing
- [ ] Voice mixing and layering
- [ ] Automatic language translation before TTS
- [ ] Batch job scheduling
- [ ] TTS templates and presets
- [ ] Keyboard shortcuts for common operations

#### 10. **Performance Optimizations** 🚀
- [ ] Implement audio caching to avoid regeneration
- [ ] Add CDN support for cloud providers
- [ ] Optimize chunk size for streaming
- [ ] Parallel processing for multi-voice generation
- [ ] GPU acceleration for local models
- [ ] Memory usage optimization for large texts

## 🐛 Known Issues

1. **Audio Playback**
   - [ ] Cross-platform audio playback needs testing
   - [ ] Some audio formats may not play on all systems
   - [x] ~~Pause/resume functionality not fully implemented~~ ✅ FIXED - Using simple system audio players

2. **Kokoro Backend**
   - [ ] PyTorch backend needs more testing
   - [ ] Model download progress not shown
   - [ ] Some voices may not download correctly

3. **UI/UX**
   - [ ] Settings window needs input validation
   - [ ] No visual waveform display
   - [ ] Export dialog needs file browser integration

## 📚 Documentation Needs

- [ ] User guide for TTS features
- [ ] API documentation for extending TTS
- [ ] Provider comparison guide
- [ ] Troubleshooting guide
- [ ] Best practices for audiobook creation
- [ ] SSML markup reference

## 🧪 Testing Requirements

- [x] ~~Unit tests for all TTS backends~~ ✅ COMPLETED - Basic tests added
- [ ] Integration tests for event system
- [ ] Cross-platform audio playback tests
- [ ] Performance benchmarks for providers
- [ ] Stress tests for long text generation
- [ ] API key validation tests

## 💡 Future Ideas

- **AI Voice Director**: Use LLM to automatically assign voices and emotions based on text content
- **Voice Marketplace**: Share and download custom voices
- **Collaborative Audiobooks**: Multiple users contributing voices
- **Real-time Translation**: Speak in one language, output in another
- **Accessibility Features**: Screen reader integration, high contrast audio waveforms
- **Mobile Companion App**: Control TTS generation from phone
- **Podcast Publishing**: Direct upload to podcast platforms
- **Voice Authentication**: Use voice cloning for security

## 🏗️ Architecture Improvements

- [ ] Separate TTS service into microservice
- [ ] Add message queue for batch processing
- [ ] Implement webhook support for long tasks
- [ ] Create TTS plugin system
- [ ] Add GraphQL API for TTS operations
- [ ] Implement distributed processing

---

**Note**: This list is maintained as part of the ongoing development of the TTS system. Items may be added, modified, or removed as the project evolves.

**Last Updated**: 2025-07-10 - Added security fixes, audio player, cost tracking, export functionality, and progress tracking