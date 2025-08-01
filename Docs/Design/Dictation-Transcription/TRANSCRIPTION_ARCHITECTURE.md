# Transcription Service Architecture

This document describes the internal architecture and implementation details of the transcription service in tldw_chatbook.

## Overview

The transcription service (`tldw_chatbook/Local_Ingestion/transcription_service.py`) provides a unified interface for multiple transcription backends, handling audio format conversion, model management, and result normalization.

## Architecture

### Core Components

```
TranscriptionService
├── Configuration Management
├── Model Cache (lazy loading)
├── Provider Implementations
│   ├── Faster-Whisper
│   ├── Qwen2Audio  
│   ├── Parakeet (NeMo)
│   └── Canary (NeMo)
└── Audio Processing
    ├── Format Conversion (FFmpeg)
    └── WAV Normalization
```

### Class Structure

```python
class TranscriptionService:
    def __init__(self):
        # Configuration from config.toml
        self.config = {
            'default_provider': str,
            'default_model': str,
            'default_language': str,
            'default_source_language': str,
            'default_target_language': str,
            'device': str,
            'compute_type': str,
            'chunk_length_seconds': float
        }
        
        # Model caches (lazy loaded)
        self._model_cache = {}  # Faster-Whisper models
        self._qwen_processor = None
        self._qwen_model = None
        self._parakeet_model = None
        self._canary_model = None
        self._canary_decoding = None
```

## Provider Implementations

### Faster-Whisper Integration

**Key Features:**
- Uses `faster-whisper` library (OpenAI Whisper optimized with CTranslate2)
- Model caching by (model, device, compute_type) tuple
- Supports transcription and translation to English
- Automatic language detection

**Implementation Details:**
```python
def _transcribe_with_faster_whisper(self, audio_path, model, language, vad_filter, 
                                   source_lang, target_lang, **kwargs):
    # Model caching for performance
    cache_key = (model, self.config['device'], self.config['compute_type'])
    
    # Task determination
    task = 'translate' if target_lang == 'en' and source_lang != 'en' else 'transcribe'
    
    # Streaming transcription with segment collection
    segments_generator, info = whisper_model.transcribe(audio_path, **options)
```

### Parakeet Integration

**Key Features:**
- Uses NVIDIA NeMo ASR models
- Optimized for real-time/streaming applications
- Supports multiple architectures (TDT, RNNT, CTC)
- GPU acceleration with automatic device selection

**Implementation Details:**
```python
def _transcribe_with_parakeet(self, audio_path, model, language, **kwargs):
    # Lazy model loading
    if self._parakeet_model is None:
        self._parakeet_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        
        # Streaming optimizations
        self._parakeet_model.change_attention_model("rel_pos_local_attn", 
                                                   context_sizes=[128, 128])
        self._parakeet_model.change_subsampling_conv_chunking_factor(1)
```

### Canary Integration

**Key Features:**
- Multilingual ASR and translation (en, de, es, fr)
- Chunked inference for long audio files
- Uses NeMo's EncDecMultiTaskModel
- Automatic handling of long-form audio

**Implementation Details:**
```python
def _transcribe_with_canary(self, audio_path, model, source_lang, target_lang, 
                           chunk_len_in_secs, **kwargs):
    # Task determination
    taskname = "s2t_translation" if target_lang != source_lang else "asr"
    
    # Chunked processing for long audio
    if duration > chunk_len_in_secs:
        hyps = get_buffered_pred_feat_multitaskAED(
            self._canary_model.cfg.preprocessor,
            [manifest_entry],
            self._canary_model,
            chunk_len_in_secs,
            tokens_per_chunk,
            self._canary_model.device
        )
```

**Chunked Inference Algorithm:**
1. Audio is split into non-overlapping chunks
2. Each chunk is processed with context from previous chunks
3. Results are concatenated to form final transcription
4. Optimized for memory efficiency and long audio files

### Qwen2Audio Integration

**Key Features:**
- Multimodal audio understanding
- Uses Transformers library
- Supports conversational prompts
- Advanced audio analysis capabilities

## Audio Processing Pipeline

### Format Conversion

The service automatically converts various audio formats to WAV:

```python
def _ensure_wav_format(self, audio_path: str) -> str:
    # Skip if already WAV
    if audio_path.lower().endswith('.wav'):
        return audio_path
    
    # Convert using FFmpeg
    output_path = tempfile.mktemp(suffix='.wav')
    self._convert_to_wav(audio_path, output_path)
    return output_path
```

**FFmpeg Integration:**
- Auto-detection of FFmpeg installation
- Platform-specific path resolution
- Fallback to config-specified path

### Temporary File Management

- Automatic cleanup of converted WAV files
- Exception-safe cleanup in finally blocks
- Respects system temp directory settings

## Language and Translation Handling

### Language Parameter Resolution

The service uses a precedence hierarchy for language settings:

1. Explicit `source_lang` parameter
2. Config `default_source_language`
3. Legacy `language` parameter
4. Config `default_language`
5. Provider defaults

### Translation Support Matrix

| Provider | Source Languages | Target Languages | Translation Type |
|----------|-----------------|------------------|------------------|
| Faster-Whisper | Any | English only | Built-in Whisper |
| Canary | en, de, es, fr | en, de, es, fr | S2T Translation |
| Parakeet | English | N/A | None |
| Qwen2Audio | Multiple | N/A | None |

### Result Normalization

All providers return a standardized result format:

```python
{
    "text": str,                    # Full transcription
    "segments": List[Dict],         # Time-aligned segments
    "language": str,                # Detected/specified language
    "provider": str,                # Provider used
    "model": str,                   # Model used
    "duration": float,              # Audio duration (optional)
    "task": str,                    # "transcription" or "translation" (optional)
    "source_language": str,         # For translations (optional)
    "target_language": str,         # For translations (optional)
    "translation": str              # Translation text (optional)
}
```

## Performance Optimizations

### Model Caching

- Models are loaded once and reused across requests
- Cache key includes all parameters affecting model initialization
- Thread-safe lazy loading pattern

### Memory Management

- Chunked processing for long audio (Canary)
- Streaming transcription (Faster-Whisper)
- Automatic GPU memory management

### Device Selection

```python
# Automatic device selection based on availability
if self.config['device'] == 'cuda' and torch.cuda.is_available():
    model = model.cuda()
elif self.config['device'] == 'mps' and torch.backends.mps.is_available():
    model = model.to('mps')
```

## Error Handling

### Exception Hierarchy

```python
TranscriptionError (base)
├── ConversionError     # Audio format conversion failures
├── Model load errors   # Model initialization failures
└── Provider errors     # Provider-specific failures
```

### Error Recovery

- Graceful degradation when optional features unavailable
- Detailed error messages with remediation steps
- Automatic cleanup on failure

## Integration Points

### Configuration System

The service integrates with tldw_chatbook's configuration system:
- Reads from `config.toml` via `get_cli_setting()`
- Supports environment variable overrides
- Default fallbacks for all settings

### UI Integration

Called from multiple UI components:
- `Ingest_Window.py` - Main ingestion interface
- `IngestLocalAudioWindow.py` - Audio file processing
- `IngestLocalVideoWindow.py` - Video file processing
- `audio_processing.py` - Batch processing logic

### Legacy Compatibility

The service maintains compatibility with the legacy `Audio_Transcription_Lib.py`:
- Legacy code delegates to new service for Parakeet
- Maintains same result format
- Preserves parameter names for backward compatibility

## Extension Points

### Adding New Providers

1. Add provider check in `__init__`:
```python
try:
    import new_provider
    NEW_PROVIDER_AVAILABLE = True
except ImportError:
    NEW_PROVIDER_AVAILABLE = False
```

2. Add lazy-loaded model attribute:
```python
self._new_provider_model = None
```

3. Implement provider method:
```python
def _transcribe_with_new_provider(self, audio_path, model, source_lang, 
                                 target_lang, **kwargs):
    # Implementation
```

4. Add to transcribe() method:
```python
elif provider == 'new_provider':
    return self._transcribe_with_new_provider(...)
```

5. Update list_available_models():
```python
if NEW_PROVIDER_AVAILABLE:
    models['new_provider'] = ['model1', 'model2']
```

### Adding Translation Support

For providers with translation capabilities:

1. Detect translation request:
```python
taskname = "translate" if target_lang and target_lang != source_lang else "transcribe"
```

2. Configure provider for translation:
```python
# Provider-specific translation setup
```

3. Include translation metadata in results:
```python
if taskname == "translate":
    result["task"] = "translation"
    result["source_language"] = source_lang
    result["target_language"] = target_lang
    result["translation"] = translated_text
```

## Testing Considerations

### Unit Testing

- Mock external dependencies (models, FFmpeg)
- Test language parameter resolution
- Verify result format consistency
- Test error handling paths

### Integration Testing

- Test with real audio files
- Verify model loading and caching
- Test translation workflows
- Validate chunked processing

### Performance Testing

- Benchmark different providers
- Memory usage profiling
- GPU utilization monitoring
- Chunking efficiency for long audio

## Future Enhancements

### Planned Features

1. **Speaker Diarization**: Integration with pyannote or NeMo diarization
2. **Streaming Support**: Real-time transcription for live audio
3. **Batch Processing**: Parallel processing of multiple files
4. **Custom Vocabularies**: Provider-specific vocabulary customization
5. **Confidence Scores**: Per-word confidence metrics

### Architecture Improvements

1. **Plugin System**: Dynamic provider loading
2. **Result Caching**: Cache transcriptions by file hash
3. **Distributed Processing**: Support for remote transcription workers
4. **Fine-tuning Support**: Custom model fine-tuning integration