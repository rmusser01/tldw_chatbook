# ADR-001: Higgs Audio Backend Architecture

**Date**: 2025-01-25  
**Status**: Accepted  
**Context**: Implementing Higgs Audio V2 as a TTS backend for tldw_chatbook

## Decision

We will implement the Higgs Audio backend by:
1. Inheriting from `LocalTTSBackend` rather than `APITTSBackend`
2. Using a comprehensive voice profile management system
3. Supporting multi-speaker dialog generation
4. Implementing zero-shot voice cloning capabilities

## Context

Higgs Audio V2 is an open-source audio foundation model trained on 10M hours of speech data. It offers:
- Zero-shot voice cloning from short reference audio
- Multi-speaker dialog generation
- Multilingual speech synthesis
- Background music support
- Local model execution without API dependencies

## Rationale

### 1. LocalTTSBackend Inheritance

**Decision**: Inherit from `LocalTTSBackend` instead of `APITTSBackend`

**Reasons**:
- Higgs Audio runs locally using PyTorch models, not through an API
- LocalTTSBackend provides appropriate abstractions for model loading/unloading
- No need for HTTP client or API key management
- Better performance with direct model access

### 2. Voice Profile Management System

**Decision**: Implement a JSON-based voice profile storage system

**Structure**:
```json
{
  "profile_name": {
    "display_name": "Friendly Voice",
    "reference_audio": "/path/to/audio.wav",
    "language": "en",
    "metadata": {
      "description": "Warm, friendly voice",
      "style": "casual",
      "analyzed_pitch": 220.5,
      "analyzed_tempo": 120.0
    }
  }
}
```

**Reasons**:
- Persistent storage of cloned voices across sessions
- Metadata allows for voice characteristics documentation
- Easy to extend with additional properties
- Human-readable format for debugging

### 3. Multi-Speaker Dialog Implementation

**Decision**: Use configurable delimiter (default: "|||") to separate speakers

**Example**:
```
Speaker1|||Hello there! Speaker2|||Hi, how are you?
```

**Reasons**:
- Simple text-based format that doesn't require structured input
- Compatible with existing chat interfaces
- Easy to parse and validate
- Delimiter can be customized to avoid conflicts

### 4. Model Loading Strategy

**Decision**: Lazy-load the Higgs model on first use

**Implementation**:
- Import `boson-multimodal` only when needed
- Create `HiggsAudioServeEngine` instance on demand
- Support configurable device (CPU/CUDA) and dtype

**Reasons**:
- Reduces startup time when TTS isn't needed
- Allows graceful degradation if dependencies missing
- Memory efficient - model only loaded when required

### 5. Audio Processing Pipeline

**Decision**: Process audio through these stages:
1. Text normalization (optional)
2. Language detection or specification
3. Voice configuration preparation
4. Message formatting for Higgs engine
5. Audio generation with progress tracking
6. Format conversion to requested output

**Reasons**:
- Modular pipeline allows for easy debugging
- Progress tracking enables UI feedback
- Format conversion provides flexibility
- Normalization improves output quality

### 6. Performance Tracking

**Decision**: Built-in performance metrics collection

**Tracked Metrics**:
- Total tokens processed
- Generation time
- Voice cloning operations
- Average generation speed

**Reasons**:
- Helps identify performance bottlenecks
- Useful for comparing different models/settings
- Provides data for optimization decisions

## Consequences

### Positive
- Full control over model execution and optimization
- No API costs or rate limits
- Voice profiles persist across sessions
- Rich feature set including voice cloning
- Streaming generation with progress feedback
- Multi-speaker support for dialog generation

### Negative
- Requires significant local resources (GPU recommended)
- Large model downloads needed (~6GB)
- Complex dependency management (PyTorch, torchaudio, etc.)
- Higher implementation complexity than API backends

### Neutral
- Voice profile management adds UI complexity
- Multi-speaker parsing requires text preprocessing
- Model initialization time on first use

## Implementation Notes

1. **Dependencies**: Core dependencies are PyTorch and boson-multimodal. Optional dependencies include librosa for audio analysis and torchaudio for audio processing.

2. **Error Handling**: Graceful degradation when optional dependencies missing. Clear error messages for missing required dependencies.

3. **Configuration**: Extensive configuration options through config.toml including model path, device selection, voice sample directory, and performance settings.

4. **Security**: Voice profiles stored locally with paths validated. No external API calls or data transmission.

## Related Decisions

- Could extend to support voice style transfer in future
- May add voice profile import/export functionality
- Could implement voice profile sharing mechanism

## References

- [Higgs Audio GitHub](https://github.com/boson-ai/higgs-audio)
- [Boson Multimodal Documentation](https://github.com/boson-ai/boson-multimodal)
- tldw_chatbook TTS Backend Architecture (base_backends.py)