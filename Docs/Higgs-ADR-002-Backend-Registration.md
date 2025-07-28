# ADR-002: Higgs Audio Backend Registration and Configuration

**Date**: 2025-01-25  
**Status**: Accepted  
**Context**: Integrating Higgs Audio backend into tldw_chatbook's TTS system

## Decision

Register Higgs Audio backend with:
1. Backend ID pattern: `local_higgs_*`
2. Comprehensive configuration defaults in TTSBackendManager
3. Environment variable and config.toml support
4. Automatic CUDA detection

## Context

The tldw_chatbook TTS system uses a registry pattern where backends are:
- Registered with glob patterns (e.g., `local_kokoro_*`, `openai_official_*`)
- Configured through TTSBackendManager's `_prepare_backend_config` method
- Support multiple configuration sources with precedence

## Rationale

### 1. Backend ID Pattern

**Decision**: Use `local_higgs_*` pattern

**Reasons**:
- Consistent with other local backends (`local_kokoro_*`, `local_chatterbox_*`)
- The `local_` prefix clearly indicates no API dependency
- Wildcard allows variants (e.g., `local_higgs_3b`, `local_higgs_7b`)

### 2. Configuration Hierarchy

**Decision**: Configuration precedence (highest to lowest):
1. Backend-specific config (`app_config[backend_id]`)
2. Environment variables (e.g., `HIGGS_MODEL_PATH`)
3. Global app config values
4. Hard-coded defaults

**Implementation**:
```python
higgs_defaults = {
    "HIGGS_MODEL_PATH": os.getenv("HIGGS_MODEL_PATH",
        self.app_config.get("HIGGS_MODEL_PATH", "bosonai/higgs-audio-v2-generation-3B-base")),
    # ... other settings
}
```

**Reasons**:
- Flexibility for different deployment scenarios
- Environment variables for containerization
- Sensible defaults for immediate use

### 3. Configuration Parameters

**Core Settings**:
- `HIGGS_MODEL_PATH`: Model identifier or local path
- `HIGGS_DEVICE`: Computing device (auto-detects CUDA)
- `HIGGS_ENABLE_FLASH_ATTN`: Flash attention optimization
- `HIGGS_DTYPE`: Model precision (bfloat16 default)

**Voice Settings**:
- `HIGGS_VOICE_SAMPLES_DIR`: Voice profile storage location
- `HIGGS_ENABLE_VOICE_CLONING`: Toggle voice cloning feature
- `HIGGS_MAX_REFERENCE_DURATION`: Limit reference audio length
- `HIGGS_DEFAULT_LANGUAGE`: Fallback language

**Feature Settings**:
- `HIGGS_ENABLE_MULTI_SPEAKER`: Multi-speaker dialog support
- `HIGGS_SPEAKER_DELIMITER`: Speaker separation marker
- `HIGGS_ENABLE_BACKGROUND_MUSIC`: Background audio mixing
- `HIGGS_TRACK_PERFORMANCE`: Performance metrics collection

**Generation Settings**:
- `HIGGS_MAX_NEW_TOKENS`: Maximum generation length
- `HIGGS_TEMPERATURE`: Sampling temperature
- `HIGGS_TOP_P`: Nucleus sampling parameter
- `HIGGS_REPETITION_PENALTY`: Repetition reduction

### 4. CUDA Auto-Detection

**Decision**: Automatically detect CUDA availability

**Implementation**:
```python
def _check_cuda_available(self) -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
```

**Reasons**:
- User-friendly - no manual device configuration needed
- Graceful fallback to CPU if CUDA unavailable
- Handles missing PyTorch gracefully

## Consequences

### Positive
- Zero configuration needed for basic usage
- Flexible configuration for advanced users
- Consistent with existing backend patterns
- Environment-based configuration for deployment

### Negative
- Many configuration parameters to document
- Potential confusion with similar settings across backends
- CUDA detection adds startup overhead

### Neutral
- Configuration complexity matches feature richness
- Similar pattern to other local backends

## Implementation Notes

1. **Registration Order**: Higgs registered after other backends to ensure dependencies loaded first

2. **Import Safety**: Registration wrapped in try-except to handle missing dependencies gracefully

3. **Path Expansion**: Voice samples directory uses `os.path.expanduser()` for `~` expansion

4. **Type Safety**: Device detection returns boolean, converted to string in config

## Future Considerations

1. **Model Variants**: The wildcard pattern allows future model size variants
2. **Configuration Profiles**: Could add preset configurations for common use cases
3. **Dynamic Settings**: Some settings could become runtime-adjustable

## References

- TTSBackendManager implementation (TTS_Backends.py:70-241)
- Backend registry pattern (TTS_Backends.py:39-67)
- Existing backend configurations for patterns