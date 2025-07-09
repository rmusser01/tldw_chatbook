# Transcription Features

This document describes the transcription capabilities available in tldw_chatbook, including support for multiple providers, languages, and translation features.

## Overview

tldw_chatbook provides a unified transcription service that supports multiple backend providers, each with their own strengths:

- **Faster-Whisper**: Fast, accurate transcription with support for many languages
- **Qwen2Audio**: Advanced multimodal understanding
- **Parakeet**: NVIDIA's optimized models for real-time transcription
- **Canary**: NVIDIA's multilingual model with translation capabilities

## Installation

### Basic Audio Support
```bash
pip install -e ".[audio]"
```

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

## Available Models

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

### Parakeet
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

### Basic Transcription

1. Navigate to the **Ingest** tab
2. Select **Local Audio** or **Local Video**
3. Choose your transcription settings:
   - **Provider**: Select from available providers
   - **Model**: Choose model size/variant
   - **Language**: Set source language (or "auto" for detection)
4. Select your audio/video files
5. Click **Process Files**

### Translation

Translation is available when using supported providers:

#### Faster-Whisper (English Translation Only)
1. Select a non-English audio file
2. The system will automatically translate to English during transcription

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
- First run may take time to download models
- Check internet connection for model downloads

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