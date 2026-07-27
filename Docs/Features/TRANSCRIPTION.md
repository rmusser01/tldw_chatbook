# Transcription Features

This document describes the transcription capabilities available in tldw_chatbook, including support for multiple providers, languages, and translation features.

## Overview

tldw_chatbook provides a unified transcription service that supports multiple backend providers, each with their own strengths:

- **Faster-Whisper**: Fast, accurate transcription with support for many languages
- **Parakeet ONNX**: Installed, local-only Parakeet v2/v3 INT8 for Library batch
  transcription
- **Qwen2Audio**: Advanced multimodal understanding
- **Parakeet (legacy NeMo)**: NVIDIA's older optimized provider for real-time
  transcription
- **Parakeet-MLX**: The separate Apple Silicon real-time provider
- **Canary**: NVIDIA's multilingual model with translation capabilities

## Installation

### Basic Audio Support
```bash
pip install -e ".[audio]"
```

### Console Microphone Dictation
```bash
pip install -e ".[speech_recording,transcription_parakeet_onnx]"
```

Install the verified **Parakeet v2 INT8** bundle from **Library → Models**
before using the Console microphone. The Mic action never downloads a model.

### NVIDIA Models (Parakeet & Canary)
```bash
pip install -e ".[nemo]"
```

## Configuration

Transcription settings are configured in `~/.config/tldw_cli/config.toml`:

```toml
[transcription]
# Default transcription provider
# Options: "faster-whisper", "qwen2audio", "parakeet", "canary"
default_provider = "faster-whisper"

# Default model for each provider
default_model = "base"

# Language settings
default_language = "en"              # Default language for transcription
default_source_language = ""         # Explicit source language (overrides default_language)
default_target_language = ""         # Target language for translation (if supported)

# Hardware settings
device = "cpu"                       # Options: "cpu", "cuda", "mps" (Apple Silicon)
compute_type = "int8"               # For faster-whisper: "int8", "float16", "float32"

# Advanced settings
use_vad_by_default = false          # Voice Activity Detection
chunk_length_seconds = 40.0         # For Canary long audio processing
```

## Library Batch Routing

For local audio and video ingestion, the Library's **Transcription provider**
menu exposes `default`, `parakeet-onnx`, and `faster-whisper`. The visible
`default` value is a semantic default, not an alias for Parakeet. While the
Parakeet promotion gate is closed, it resolves to faster-whisper for every
request. `en` remains the default requested language.

| Selected provider | Request | Current batch result |
|---|---|---|
| `default` | Transcription, `auto`, unsupported language, or translation to `en` | faster-whisper INT8 while the promotion gate is closed |
| `default` | Translation to any target other than `en` | Fails; faster-whisper translates only to English |
| `parakeet-onnx` | Missing language or `en` | `nemo-parakeet-tdt-0.6b-v2`, INT8 |
| `parakeet-onnx` | Supported non-English code | `nemo-parakeet-tdt-0.6b-v3`, INT8 |
| `parakeet-onnx` | `auto`, an unsupported language, or any translation | Fails with `Retry with faster-whisper` guidance |
| `faster-whisper` | Transcription, or translation with target `en` | faster-whisper INT8 |
| `faster-whisper` | Translation to any target other than `en` | Fails; faster-whisper translates only to English |

The exact supported non-English Parakeet v3 codes are `bg`, `hr`, `cs`, `da`,
`nl`, `et`, `fi`, `fr`, `de`, `el`, `hu`, `it`, `lv`, `lt`, `mt`, `pl`, `pt`,
`ro`, `sk`, `sl`, `es`, `sv`, `ru`, and `uk`.

For v3, the requested language selects the route only; it is not passed to the
decoder as a constraint. Results therefore report `requested_language` as the
selected code, `effective_language=auto`, `detected_language=null`, and the
`requested_language_not_enforced` warning. They do not claim that v3 detected
or was forced to the requested language.

Batch transcription never downloads a model in an ingestion worker. Exact
Parakeet requires the matching existing local bundle. Selecting v3 with a
known verified v2 receipt is rejected; a receipt does not verify an arbitrary
v3 directory. faster-whisper is also loaded with `local_files_only=True`, so a
missing cache fails clearly instead of starting a worker download.

This Parakeet ONNX batch path is distinct from the legacy NeMo `parakeet`
provider and the macOS-only `parakeet-mlx` provider described below. Full
managed v3 artifacts and the interactive **Retry with faster-whisper** action
remain deferred.

## Available Models

### Parakeet ONNX
- **English model**: `nemo-parakeet-tdt-0.6b-v2` (INT8)
- **Supported non-English model**: `nemo-parakeet-tdt-0.6b-v3` (INT8)
- **Runtime**: `onnx-asr[cpu]==0.12.0`
- **Translation**: Not supported
- **Best for**: Explicit, installed/local Library batch transcription

### Faster-Whisper
- **Models**: tiny, base, small, medium, large-v1, large-v2, large-v3, distil-large-v3
- **Languages**: 100+ languages with automatic detection
- **Translation**: To English only
- **Best for**: General-purpose transcription with good accuracy/speed balance

### Qwen2Audio
- **Models**: Qwen2-Audio-7B-Instruct
- **Languages**: Multiple languages
- **Translation**: Not supported
- **Best for**: Advanced audio understanding and context

### Parakeet (Legacy NeMo Provider)
- **Models**: 
  - nvidia/parakeet-tdt-1.1b (Transducer)
  - nvidia/parakeet-rnnt-1.1b (RNN-Transducer)
  - nvidia/parakeet-ctc-1.1b (CTC)
  - nvidia/parakeet-tdt-0.6b (Smaller variants)
  - nvidia/parakeet-tdt-0.6b-v2 (Latest small model)
- **Languages**: Primarily English
- **Translation**: Not supported
- **Best for**: Real-time transcription, streaming applications

### Canary
- **Models**: 
  - nvidia/canary-1b-flash (Optimized for speed)
  - nvidia/canary-1b (Standard model)
- **Languages**: English, German, Spanish, French
- **Translation**: Between all supported languages
- **Best for**: Multilingual transcription and translation

## Usage in the Application

### Console Dictation

1. Open **Console** and expand the composer.
2. Select **Mic** to record from the default microphone.
3. Select **Rec ●** to stop, or wait for the 60-second limit.
4. After the local **STT…** step, the English transcript is inserted at the
   current caret. It is not sent automatically.

Console dictation uses explicit `en`, Parakeet v2 INT8, and either
`transcription.parakeet_onnx_model_dir` or the verified Library-installed
bundle. Missing dependencies, model files, microphone access, empty audio, and
transcription failures leave the draft unchanged and display an error.

### Basic Transcription

1. Navigate to **Library**
2. Select **Local Audio** or **Local Video**
3. Choose your transcription settings:
   - **Provider**: Keep the semantic default or select an exact provider
   - **Model**: Choose model size/variant
   - **Language**: `en` by default; use `auto` only with faster-whisper
4. Select your audio/video files
5. Click **Process Files**

### Translation

Translation is available when using supported providers:

#### Faster-Whisper (English Translation Only)
1. Select a non-English audio file
2. Set the translation target to `en`
3. Process the file for transcription and translation to English

#### Canary (Multilingual Translation)
1. Select **canary** as the provider
2. Set the **Source Language** (en, de, es, or fr)
3. Set the **Target Language** (en, de, es, or fr)
4. Process the file for transcription + translation

## Language Codes

Common language codes for transcription:

| Language | Code | Supported By |
|----------|------|--------------|
| English | en | All providers |
| Spanish | es | Faster-Whisper, Canary |
| French | fr | Faster-Whisper, Canary |
| German | de | Faster-Whisper, Canary |
| Chinese | zh | Faster-Whisper, Qwen2Audio |
| Japanese | ja | Faster-Whisper, Qwen2Audio |
| Russian | ru | Faster-Whisper |
| Arabic | ar | Faster-Whisper |
| Hindi | hi | Faster-Whisper |
| Portuguese | pt | Faster-Whisper |

Use "auto" for automatic language detection (Faster-Whisper only).

## Performance Tips

### Model Selection
- **For speed**: Use smaller models (tiny, base) or Parakeet
- **For accuracy**: Use larger models (large-v3) or Canary
- **For non-English**: Use Faster-Whisper or Canary

### Hardware Acceleration
- **NVIDIA GPU**: Set `device = "cuda"` for 5-10x speedup
- **Apple Silicon**: Set `device = "mps"` for Metal acceleration
- **CPU**: Use `compute_type = "int8"` for better performance

### Long Audio Files
- **Canary**: Automatically uses chunked processing for files > 40 seconds
- Adjust `chunk_length_seconds` for different chunk sizes
- Smaller chunks (10s) provide better timestamp accuracy

## Advanced Features

### Voice Activity Detection (VAD)
Enable VAD to filter out silence and non-speech segments:
```toml
use_vad_by_default = true
```

### Custom Chunk Sizes
For Canary model, adjust chunk size for long audio:
```toml
chunk_length_seconds = 10.0  # Better for podcasts with timestamps
chunk_length_seconds = 60.0  # Better for long lectures
```

### Batch Processing
The Local Ingestion window supports batch processing:
1. Select multiple audio/video files
2. All files will be processed with the same settings
3. Progress is shown for each file

## Troubleshooting

### Common Issues

**"Model not found"**
- Ensure you've installed the required dependencies
- For Library batch transcription, install or cache the model before ingesting;
  the worker will not download it
- For exact Parakeet ONNX, select the local bundle for the routed v2 or v3 model
- For faster-whisper, confirm the selected model is already in the local cache

**"CUDA out of memory"**
- Use a smaller model
- Reduce batch size to 1
- Set `device = "cpu"` as fallback

**"NeMo toolkit not installed"**
- Install with: `pip install -e ".[nemo]"`
- Requires Python 3.8+ and compatible PyTorch

**Poor transcription quality**
- Try a larger model
- Ensure audio quality is good (clear speech, minimal background noise)
- Specify the correct source language instead of using "auto"

### Language Detection Issues
- Faster-Whisper's "auto" detection works best with >30 seconds of audio
- For short clips, manually specify the language
- Canary requires explicit language specification

## API Integration

For programmatic access, use the TLDW API integration:
1. Configure TLDW API settings in Tools & Settings
2. Use the API ingestion windows for remote processing
3. Supports the same transcription options as local processing
