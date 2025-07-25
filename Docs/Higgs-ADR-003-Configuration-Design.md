# ADR-003: Higgs Audio Configuration Design

**Date**: 2025-01-25  
**Status**: Accepted  
**Context**: Defining configuration parameters for Higgs Audio TTS backend

## Decision

Implement comprehensive configuration with:
1. Model and hardware settings
2. Voice management configuration
3. Feature toggles
4. Generation parameters
5. Sensible defaults for all settings

## Context

Higgs Audio is a sophisticated TTS model with many configurable parameters. The configuration needs to:
- Support both simple and advanced use cases
- Provide good defaults for immediate use
- Allow fine-tuning for optimal performance
- Be consistent with other TTS backends

## Rationale

### 1. Model Configuration

**Parameters**:
- `HIGGS_MODEL_PATH`: Model identifier or local path
- `HIGGS_DEVICE`: Computing device selection
- `HIGGS_DTYPE`: Model precision
- `HIGGS_ENABLE_FLASH_ATTN`: Optimization toggle

**Defaults**:
```toml
HIGGS_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
HIGGS_DEVICE = "auto"  # Auto-detects CUDA
HIGGS_DTYPE = "bfloat16"
HIGGS_ENABLE_FLASH_ATTN = true
```

**Reasons**:
- Default model is the recommended 3B parameter version
- Auto device selection removes setup friction
- bfloat16 balances quality and performance
- Flash attention enabled for better performance

### 2. Voice Management

**Parameters**:
- `HIGGS_VOICE_SAMPLES_DIR`: Profile storage location
- `HIGGS_ENABLE_VOICE_CLONING`: Feature toggle
- `HIGGS_MAX_REFERENCE_DURATION`: Reference audio limit
- `HIGGS_DEFAULT_LANGUAGE`: Fallback language

**Defaults**:
```toml
HIGGS_VOICE_SAMPLES_DIR = "~/.config/tldw_cli/higgs_voices"
HIGGS_ENABLE_VOICE_CLONING = true
HIGGS_MAX_REFERENCE_DURATION = 30
HIGGS_DEFAULT_LANGUAGE = "en"
```

**Reasons**:
- Consistent directory with other tldw_cli data
- Voice cloning is a key feature, enabled by default
- 30-second limit prevents excessive memory usage
- English default matches most common use case

### 3. Feature Configuration

**Parameters**:
- `HIGGS_ENABLE_MULTI_SPEAKER`: Multi-speaker support
- `HIGGS_SPEAKER_DELIMITER`: Speaker separator
- `HIGGS_ENABLE_BACKGROUND_MUSIC`: Music mixing
- `HIGGS_TRACK_PERFORMANCE`: Metrics collection

**Defaults**:
```toml
HIGGS_ENABLE_MULTI_SPEAKER = true
HIGGS_SPEAKER_DELIMITER = "|||"
HIGGS_ENABLE_BACKGROUND_MUSIC = false
HIGGS_TRACK_PERFORMANCE = true
```

**Reasons**:
- Multi-speaker enabled for dialog generation
- "|||" delimiter unlikely to appear in normal text
- Background music off by default (requires setup)
- Performance tracking helps optimization

### 4. Generation Parameters

**Parameters**:
- `HIGGS_MAX_NEW_TOKENS`: Generation length limit
- `HIGGS_TEMPERATURE`: Sampling randomness
- `HIGGS_TOP_P`: Nucleus sampling
- `HIGGS_REPETITION_PENALTY`: Repetition reduction

**Defaults**:
```toml
HIGGS_MAX_NEW_TOKENS = 4096
HIGGS_TEMPERATURE = 0.7
HIGGS_TOP_P = 0.9
HIGGS_REPETITION_PENALTY = 1.1
```

**Reasons**:
- 4096 tokens sufficient for most audio generation
- 0.7 temperature balances consistency and variety
- 0.9 top_p standard for nucleus sampling
- 1.1 penalty reduces repetitive patterns

### 5. Integration with TTSSettings

**Decision**: Add "higgs" to provider options

```toml
default_tts_provider = "kokoro"  # Options: ..., "higgs"
```

**Reasons**:
- Makes Higgs discoverable in UI
- Consistent with other backend listings
- Allows setting as default provider

## Consequences

### Positive
- Comprehensive control over model behavior
- Good defaults enable immediate use
- Performance tuning options available
- Feature flags allow gradual adoption

### Negative
- Many parameters may overwhelm new users
- Some settings require understanding of ML concepts
- Configuration file becomes longer

### Neutral
- Similar complexity to other local TTS backends
- Advanced features optional through toggles

## Configuration Priority

1. **Essential**: Model path, device, voice directory
2. **Important**: Voice cloning, multi-speaker, language
3. **Advanced**: Generation parameters, dtype, flash attention
4. **Optional**: Background music, performance tracking

## Migration Notes

For users upgrading:
1. No action required - defaults work immediately
2. Existing voice profiles compatible
3. Can gradually adopt advanced features

## Future Considerations

1. **Presets**: Could add configuration presets (quality/speed)
2. **UI Integration**: Settings could be exposed in UI
3. **Auto-tuning**: Could detect optimal settings
4. **Model Management**: Could add model download/update

## References

- Higgs Audio documentation
- config.toml structure (lines 141-209)
- Other TTS backend configurations for consistency