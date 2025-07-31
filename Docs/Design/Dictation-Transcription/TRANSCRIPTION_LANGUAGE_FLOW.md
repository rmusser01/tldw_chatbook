# Transcription Language Flow Documentation

This document details how language parameters flow through the transcription system, addressing the integration between UI components and the transcription service.

## Overview

The transcription system supports multiple language-related parameters:
- **Source Language**: The language of the audio being transcribed
- **Target Language**: The language to translate to (if translation is desired)
- **Legacy Language Parameter**: For backward compatibility

## Parameter Flow

### 1. UI Layer (Ingestion Windows)

The UI collects language information from users through input fields:

```python
# In IngestLocalAudioWindow / IngestLocalVideoWindow
transcription_language = self.query_one("#local-transcription-language-audio", Input).value.strip() or "en"
```

### 2. Processing Layer (audio_processing.py)

The processor receives the language parameter and passes it to transcription:

```python
def process_audio_files(
    self,
    transcription_language: Optional[str] = 'en',  # UI parameter name
    # ... other parameters
):
    # Internal processing
    transcription_result = self._transcribe_audio(
        audio_path,
        language=transcription_language,  # Maps to 'language' parameter
        # ... other parameters
    )
```

### 3. Transcription Service Layer

The transcription service now supports multiple language parameters:

```python
def transcribe(
    self,
    audio_path: str,
    language: Optional[str] = None,      # Legacy/backward compatibility
    source_lang: Optional[str] = None,   # Explicit source language
    target_lang: Optional[str] = None,   # Translation target
    # ... other parameters
)
```

## Parameter Resolution

The service uses a precedence hierarchy:

1. **Explicit source_lang** (highest priority)
2. **Config default_source_language**
3. **Legacy language parameter**
4. **Config default_language**
5. **Provider defaults** (lowest priority)

```python
# Resolution logic in transcription_service.py
source_lang = (
    source_lang or 
    self.config['default_source_language'] or 
    language or 
    self.config['default_language']
)
```

## Integration Points

### Audio Transcription Library (Legacy)

The legacy `Audio_Transcription_Lib.py` uses different parameter names:

```python
def speech_to_text(
    audio_file_path: str,
    selected_source_lang: str = 'en',  # Different parameter name
    # ... other parameters
)
```

This is handled by the integration layer that maps parameters appropriately.

### Direct Service Usage

When using the transcription service directly:

```python
from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService

service = TranscriptionService()

# Option 1: Using legacy parameter
result = service.transcribe(
    audio_path="audio.wav",
    language="es"  # Spanish
)

# Option 2: Using explicit parameters
result = service.transcribe(
    audio_path="audio.wav",
    source_lang="es",  # Spanish source
    target_lang="en"   # Translate to English
)
```

## Translation Support

### Providers with Translation

1. **Faster-Whisper**: Translates to English only
   ```python
   # Automatic translation when target_lang='en' and source_lang != 'en'
   task = 'translate' if target_lang == 'en' and source_lang != 'en' else 'transcribe'
   ```

2. **Canary**: Multilingual translation
   ```python
   # Supports translation between en, de, es, fr
   taskname = "s2t_translation" if target_lang != source_lang else "asr"
   ```

### Translation Results

When translation is performed, the result includes additional fields:

```python
{
    "text": "Translated text",
    "task": "translation",
    "source_language": "es",
    "target_language": "en",
    "translation": "Translated text",
    # ... other fields
}
```

## Configuration

### User Configuration (config.toml)

```toml
[transcription]
# Default source language for transcription
default_language = "en"

# Explicit source language (overrides default_language)
default_source_language = ""

# Target language for translation
default_target_language = ""
```

### Programmatic Configuration

```python
# Override configuration programmatically
service = TranscriptionService()
service.config['default_source_language'] = 'es'
service.config['default_target_language'] = 'en'
```

## Best Practices

### For UI Developers

1. **Use consistent parameter names** across UI components
2. **Provide language selection** for users
3. **Show translation options** only for supported providers
4. **Display detected language** in results

### For API Users

1. **Prefer explicit parameters** over legacy ones:
   ```python
   # Good
   service.transcribe(audio_path, source_lang='es', target_lang='en')
   
   # Less clear
   service.transcribe(audio_path, language='es')
   ```

2. **Check provider capabilities** before requesting translation:
   ```python
   models = service.list_available_models()
   if 'canary' in models:
       # Canary supports multilingual translation
       result = service.transcribe(
           audio_path, 
           provider='canary',
           source_lang='de', 
           target_lang='en'
       )
   ```

3. **Handle translation results** appropriately:
   ```python
   result = service.transcribe(...)
   if result.get('task') == 'translation':
       translated_text = result['translation']
       source_lang = result['source_language']
       target_lang = result['target_language']
   ```

## Migration Guide

### From Legacy Parameters

If you're using the old parameter names:

```python
# Old way
speech_to_text(
    audio_file_path="audio.wav",
    selected_source_lang="es"
)

# New way
service.transcribe(
    audio_path="audio.wav",
    source_lang="es"
)
```

### Adding Translation

To add translation to existing code:

```python
# Before: Transcription only
result = service.transcribe(audio_path, language='es')

# After: Transcription + Translation
result = service.transcribe(
    audio_path, 
    source_lang='es',
    target_lang='en',
    provider='canary'  # or 'faster-whisper' for English translation
)
```

## Error Handling

### Common Issues

1. **Unsupported Language Combination**
   ```python
   try:
       result = service.transcribe(
           audio_path,
           provider='canary',
           source_lang='ja',  # Japanese not supported by Canary
           target_lang='en'
       )
   except TranscriptionError as e:
       # Handle unsupported language
       logger.error(f"Language not supported: {e}")
   ```

2. **Translation Not Available**
   ```python
   # Parakeet doesn't support translation
   result = service.transcribe(
       audio_path,
       provider='parakeet',
       target_lang='es'  # Will be ignored
   )
   # Check if translation actually happened
   if 'translation' not in result:
       logger.warning("Translation not performed")
   ```

## Testing

### Unit Tests

Test language parameter resolution:

```python
def test_language_parameter_precedence():
    service = TranscriptionService()
    service.config['default_language'] = 'en'
    service.config['default_source_language'] = 'es'
    
    # Should use source_lang over all defaults
    result = service._resolve_language_params(
        language='fr',
        source_lang='de'
    )
    assert result['source'] == 'de'
```

### Integration Tests

Test end-to-end language flow:

```python
def test_ui_to_transcription_flow():
    # Simulate UI input
    ui_language = "es"
    
    # Process through audio processor
    processor = LocalAudioProcessor()
    result = processor.process_audio_files(
        inputs=["test.wav"],
        transcription_language=ui_language
    )
    
    # Verify language was used
    assert result['results'][0]['language'] == 'es'
```