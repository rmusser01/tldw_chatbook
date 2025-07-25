# Higgs Audio TTS Backend Guide

This guide covers the installation, configuration, and usage of the Higgs Audio V2 text-to-speech backend in tldw_chatbook.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Basic Usage](#basic-usage)
5. [Voice Cloning](#voice-cloning)
6. [Multi-Speaker Dialog](#multi-speaker-dialog)
7. [Voice Profile Management](#voice-profile-management)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Overview

Higgs Audio V2 is an open-source audio foundation model developed by Boson AI, trained on over 10 million hours of multilingual speech data. Key features include:

- **Zero-shot voice cloning**: Clone any voice from a short audio sample
- **Multi-speaker dialog**: Generate conversations with different voices
- **Multilingual support**: Generate speech in multiple languages
- **Local execution**: No API costs or internet dependency
- **High quality**: State-of-the-art speech synthesis

## Installation

### Prerequisites

1. Python 3.11 or higher
2. PyTorch (CPU or CUDA)
3. 8GB+ RAM (16GB+ recommended)
4. ~6GB disk space for models

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio

# Higgs Audio
pip install boson-multimodal

# Optional but recommended
pip install numpy librosa soundfile
```

### Step 2: Install tldw_chatbook with TTS support

```bash
# From the tldw_chatbook directory
pip install -e ".[tts]"
```

### Step 3: Download Model (Optional)

The model will be downloaded automatically on first use, but you can pre-download:

```python
from huggingface_hub import snapshot_download
snapshot_download("bosonai/higgs-audio-v2-generation-3B-base")
```

## Configuration

### Basic Configuration

Add to your `~/.config/tldw_cli/config.toml`:

```toml
[TTSSettings]
default_tts_provider = "higgs"  # Set Higgs as default

# Higgs Audio settings
HIGGS_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
HIGGS_DEVICE = "auto"  # auto, cpu, cuda, cuda:0
HIGGS_VOICE_SAMPLES_DIR = "~/.config/tldw_cli/higgs_voices"
```

### Advanced Configuration

```toml
# Performance settings
HIGGS_ENABLE_FLASH_ATTN = true  # Faster attention (requires compatible GPU)
HIGGS_DTYPE = "bfloat16"  # float32, float16, bfloat16
HIGGS_MAX_TOKENS = 500  # Chunk size for long texts

# Voice settings
HIGGS_ENABLE_VOICE_CLONING = true
HIGGS_MAX_REFERENCE_DURATION = 30  # Max seconds for reference audio
HIGGS_DEFAULT_LANGUAGE = "en"

# Features
HIGGS_ENABLE_MULTI_SPEAKER = true
HIGGS_SPEAKER_DELIMITER = "|||"
HIGGS_ENABLE_BACKGROUND_MUSIC = false
HIGGS_TRACK_PERFORMANCE = true

# Generation parameters
HIGGS_MAX_NEW_TOKENS = 4096
HIGGS_TEMPERATURE = 0.7  # 0.0-2.0, higher = more variation
HIGGS_TOP_P = 0.9  # 0.0-1.0, nucleus sampling
HIGGS_REPETITION_PENALTY = 1.1  # 1.0 = no penalty
```

### Environment Variables

You can override config with environment variables:

```bash
export HIGGS_MODEL_PATH="/path/to/local/model"
export HIGGS_DEVICE="cuda:1"
```

## Basic Usage

### In tldw_chatbook TUI

1. Start the application:
   ```bash
   python -m tldw_chatbook.app
   ```

2. In the Chat tab:
   - Press `Ctrl+T` to open TTS settings
   - Select "higgs" as the provider
   - Choose a voice or use default
   - Enable TTS for messages

### Programmatic Usage

```python
from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest

# Initialize backend manager
config = {"HIGGS_DEVICE": "cuda"}
manager = TTSBackendManager(config)

# Get Higgs backend
backend = await manager.get_backend("local_higgs_v2")

# Generate speech
request = OpenAISpeechRequest(
    input="Hello, this is a test of Higgs Audio.",
    voice="default",
    response_format="mp3"
)

# Stream audio
async for chunk in backend.generate_speech_stream(request):
    # Process audio chunk
    pass
```

## Voice Cloning

### Creating a Voice Profile

1. **From Reference Audio**:
   ```python
   from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager
   
   manager = HiggsVoiceProfileManager(Path("~/.config/tldw_cli/higgs_voices"))
   
   success, message = manager.create_profile(
       profile_name="john_doe",
       reference_audio_path="/path/to/john_sample.wav",
       display_name="John's Voice",
       description="Professional male voice",
       language="en",
       tags=["professional", "male"]
   )
   ```

2. **Using Pre-defined Voices**:
   
   Higgs includes several default voice profiles:
   - `professional_female`: Clear, professional female voice
   - `warm_female`: Friendly, conversational female voice
   - `storyteller_male`: Expressive male narrator
   - `deep_male`: Deep, authoritative male voice
   - `energetic_female`: Upbeat, dynamic female voice
   - `soft_female`: Gentle, soothing female voice

### Zero-Shot Voice Cloning

Clone a voice on-the-fly without creating a profile:

```python
request = OpenAISpeechRequest(
    input="This will sound like the reference audio.",
    voice="/path/to/reference_audio.wav",  # Direct path to audio
    response_format="mp3"
)
```

### Voice Requirements

- **Format**: WAV, MP3, FLAC, OGG, M4A
- **Duration**: 3-30 seconds recommended
- **Quality**: Clear speech, minimal background noise
- **Content**: Natural speech, avoid singing or effects

## Multi-Speaker Dialog

### Basic Dialog Generation

Use the `|||` delimiter to separate speakers:

```python
dialog_text = """
Narrator|||In a quiet coffee shop, two friends meet after years apart.
Sarah|||Oh my goodness, is that really you?
John|||Sarah! I can't believe it's been so long!
Sarah|||How have you been? You look great!
Narrator|||They embraced warmly, years melting away in an instant.
"""

request = OpenAISpeechRequest(
    input=dialog_text,
    voice="storyteller_male",  # Default/narrator voice
    response_format="mp3"
)
```

### Custom Speaker Mapping

Create profiles for each character:

```python
# Create character voices
manager.create_profile("sarah", "/audio/sarah_sample.wav", "Sarah's Voice")
manager.create_profile("john", "/audio/john_sample.wav", "John's Voice")

# Use in dialog
dialog = """
sarah|||Oh my goodness, is that really you?
john|||Sarah! I can't believe it's been so long!
"""
```

### Advanced Dialog Options

```toml
# In config.toml
HIGGS_SPEAKER_DELIMITER = ":::"  # Custom delimiter
HIGGS_ENABLE_MULTI_SPEAKER = true
```

## Voice Profile Management

### Command-Line Tools

```bash
# List all profiles
python -m tldw_chatbook.TTS.backends.higgs_voice_manager list

# Create profile
python -m tldw_chatbook.TTS.backends.higgs_voice_manager create \
    --name "podcast_host" \
    --audio "/path/to/sample.wav" \
    --display "Podcast Host Voice" \
    --description "Energetic podcast host"

# Export profile
python -m tldw_chatbook.TTS.backends.higgs_voice_manager export \
    --profile "podcast_host" \
    --output "/path/to/export"

# Import profile
python -m tldw_chatbook.TTS.backends.higgs_voice_manager import \
    --package "/path/to/higgs_voice_podcast_host"
```

### Profile Management API

```python
from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager

manager = HiggsVoiceProfileManager(voice_samples_dir)

# List profiles
profiles = manager.list_profiles(tags=["professional"])

# Get specific profile
profile = manager.get_profile("john_doe")

# Update profile
success, msg = manager.update_profile(
    "john_doe",
    description="Updated description",
    tags=["professional", "narrator"]
)

# Delete profile
success, msg = manager.delete_profile("old_profile")

# Backup and restore
success, msg = manager.restore_from_backup()  # Latest backup
```

## Advanced Features

### Background Music (Experimental)

```toml
HIGGS_ENABLE_BACKGROUND_MUSIC = true
HIGGS_BACKGROUND_MUSIC_PATH = "/path/to/background.mp3"
```

### Performance Tracking

```python
# Get performance statistics
stats = backend.get_performance_stats()
print(f"Generated {stats['total_generations']} times")
print(f"Average time: {stats['average_generation_time']:.2f}s")
```

### Streaming with Progress

```python
# Set progress callback
async def progress_callback(info):
    print(f"Progress: {info['progress']:.1%} - {info['status']}")

backend.set_progress_callback(progress_callback)

# Generate with progress tracking
async for chunk in backend.generate_speech_stream(request):
    # Progress updates will be printed
    pass
```

### Language Support

Higgs supports multiple languages:

```python
# Explicit language
request = OpenAISpeechRequest(
    input="Bonjour, comment allez-vous?",
    voice="professional_female",
    response_format="mp3",
    extra_params={"language": "fr"}
)

# Auto-detection (if language not specified)
# Higgs will attempt to detect from text
```

Supported languages include:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```toml
   HIGGS_DTYPE = "float16"  # Reduce memory usage
   HIGGS_DEVICE = "cpu"  # Use CPU if GPU OOM
   ```

2. **Slow Generation**
   - Enable flash attention: `HIGGS_ENABLE_FLASH_ATTN = true`
   - Use GPU: `HIGGS_DEVICE = "cuda"`
   - Reduce quality: `HIGGS_DTYPE = "float16"`

3. **Model Download Issues**
   ```bash
   # Manual download
   huggingface-cli download bosonai/higgs-audio-v2-generation-3B-base
   ```

4. **Voice Cloning Quality**
   - Use longer reference audio (10-30 seconds)
   - Ensure clear, noise-free samples
   - Match language of reference and target

### Debug Mode

```python
import logging
logging.getLogger("tldw_chatbook.TTS").setLevel(logging.DEBUG)
```

## Examples

### Example 1: Audiobook Generation

```python
async def generate_audiobook(text_file, output_file):
    # Initialize
    manager = TTSBackendManager({})
    backend = await manager.get_backend("local_higgs_v2")
    
    # Read text
    with open(text_file, 'r') as f:
        text = f.read()
    
    # Generate with narrator voice
    request = OpenAISpeechRequest(
        input=text,
        voice="storyteller_male",
        response_format="mp3"
    )
    
    # Save audio
    audio_chunks = []
    async for chunk in backend.generate_speech_stream(request):
        audio_chunks.append(chunk)
    
    with open(output_file, 'wb') as f:
        f.write(b''.join(audio_chunks))
```

### Example 2: Podcast Conversation

```python
podcast_script = """
host|||Welcome to Tech Talk! Today we're discussing AI voice synthesis.
guest|||Thanks for having me! It's an exciting topic.
host|||Let's start with the basics. What exactly is voice synthesis?
guest|||Voice synthesis is the artificial production of human speech.
host|||And how has AI changed this field?
guest|||AI has revolutionized it with neural networks that can clone voices.
"""

request = OpenAISpeechRequest(
    input=podcast_script,
    voice="professional_female",  # Default host voice
    response_format="mp3"
)
```

### Example 3: Voice Profile Batch Creation

```python
async def create_voice_library(samples_dir):
    manager = HiggsVoiceProfileManager(higgs_voices_dir)
    
    for audio_file in Path(samples_dir).glob("*.wav"):
        profile_name = audio_file.stem
        success, msg = manager.create_profile(
            profile_name=profile_name,
            reference_audio_path=str(audio_file),
            display_name=profile_name.replace("_", " ").title(),
            tags=["batch_import"]
        )
        print(f"{profile_name}: {msg}")
```

## Best Practices

1. **Voice Cloning**
   - Use high-quality reference audio
   - Keep references between 5-20 seconds
   - Match the speaking style you want

2. **Performance**
   - Pre-load models for production use
   - Use appropriate dtype for your hardware
   - Enable flash attention on compatible GPUs

3. **Multi-Speaker**
   - Create profiles for recurring characters
   - Use consistent delimiter formatting
   - Test voice combinations for clarity

4. **Storage**
   - Regularly backup voice profiles
   - Organize with meaningful tags
   - Document voice characteristics

## Resources

- [Higgs Audio GitHub](https://github.com/boson-ai/higgs-audio)
- [Boson Multimodal Docs](https://github.com/boson-ai/boson-multimodal)
- [Model on Hugging Face](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base)

## License

Higgs Audio is released under an open-source license. Check the official repository for details.