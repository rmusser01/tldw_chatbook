# NVIDIA Parakeet Transcription Support

This document describes the NVIDIA Parakeet ASR (Automatic Speech Recognition) integration in tldw_chatbook, providing high-performance, low-latency transcription capabilities.

## Overview

NVIDIA Parakeet is a family of efficient ASR models designed for real-time transcription with excellent accuracy. The integration supports multiple Parakeet model variants optimized for different use cases.

## Features

- **Low Latency**: Optimized for real-time and streaming transcription
- **Multiple Model Variants**: TDT, CTC, and RNN-T architectures
- **GPU Acceleration**: Full CUDA support for faster processing
- **Efficient Memory Usage**: Handles long audio files efficiently
- **Production Ready**: Designed for deployment at scale

## Installation

To use Parakeet transcription, install tldw_chatbook with audio dependencies:

```bash
pip install -e ".[audio]"
```

This will install the NeMo toolkit and other required dependencies.

## Available Models

### Parakeet TDT (Transducer)
- `nvidia/parakeet-tdt-1.1b` - Large model, best accuracy
- `nvidia/parakeet-tdt-0.6b` - Smaller model, faster inference

**Best for**: Real-time streaming applications, live transcription

### Parakeet CTC
- `nvidia/parakeet-ctc-1.1b` - Large CTC model
- `nvidia/parakeet-ctc-0.6b` - Smaller CTC model

**Best for**: Batch processing, when timestamps aren't critical

### Parakeet RNN-T
- `nvidia/parakeet-rnnt-1.1b` - Large RNN-Transducer
- `nvidia/parakeet-rnnt-0.6b` - Smaller RNN-Transducer

**Best for**: Balance between accuracy and speed

## Usage

### Basic Transcription

```python
from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService

service = TranscriptionService()
result = service.transcribe(
    audio_path="path/to/audio.wav",
    provider="parakeet",
    model="nvidia/parakeet-tdt-1.1b",
    language="en"
)

print(result['text'])
```

### Configuration

Set Parakeet as your default transcription provider in `~/.config/tldw_cli/config.toml`:

```toml
[transcription]
default_provider = "parakeet"
default_model = "nvidia/parakeet-tdt-1.1b"
default_language = "en"
device = "cuda"  # Use GPU if available
```

### Through Audio Processing

```python
from tldw_chatbook.Local_Ingestion.audio_processing import LocalAudioProcessor

processor = LocalAudioProcessor()
results = processor.process_audio_files(
    inputs=["audio1.mp3", "audio2.wav"],
    transcription_model="nvidia/parakeet-tdt-1.1b",
    transcription_language="en",
    perform_chunking=True,
    perform_analysis=True
)
```

## Performance Comparison

| Model | Latency | Accuracy | Memory Usage | Best Use Case |
|-------|---------|----------|--------------|---------------|
| Parakeet TDT 1.1B | Very Low | High | Moderate | Real-time streaming |
| Parakeet CTC 1.1B | Low | High | Low | Batch processing |
| Parakeet RNN-T 1.1B | Low | Very High | Moderate | General purpose |
| Whisper Large-v3 | High | Very High | High | Offline processing |
| Whisper Base | Moderate | Moderate | Low | Quick drafts |

## Technical Details

### Architecture
- **TDT (Transformer-Transducer)**: Uses transformer architecture for acoustic modeling with transducer for alignment-free decoding
- **CTC (Connectionist Temporal Classification)**: Simple and fast, outputs characters/tokens directly
- **RNN-T (RNN-Transducer)**: Combines RNN acoustic model with transducer decoder

### Optimizations
- Local attention mechanisms for efficient long-form audio processing
- Chunked convolution for streaming applications
- Mixed precision inference (FP16) on supported GPUs
- Automatic batch size optimization

### Audio Requirements
- Supported formats: WAV, MP3, M4A, FLAC (auto-converted to WAV)
- Recommended sample rate: 16kHz (auto-resampled if different)
- Mono audio (stereo is auto-converted)

## Troubleshooting

### NeMo Installation Issues
If you encounter issues installing NeMo:
```bash
# Install with specific CUDA version
pip install nemo-toolkit[asr]==1.20.0 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Model Download Issues
Models are downloaded from NVIDIA NGC on first use. Ensure you have:
- Stable internet connection
- Sufficient disk space (~4GB per model)
- Write permissions to model cache directory

### GPU Memory Issues
If you run out of GPU memory:
- Use smaller models (0.6B variants)
- Reduce batch size to 1
- Use CPU inference (slower but works)

### Fallback Behavior
If Parakeet fails, the system can fall back to other providers:
```python
try:
    result = service.transcribe(audio_path, provider="parakeet")
except TranscriptionError:
    # Fallback to whisper
    result = service.transcribe(audio_path, provider="faster-whisper")
```

## Development

### Adding New Parakeet Models
To add support for new Parakeet models:

1. Add model name to `list_available_models()` in `transcription_service.py`
2. Test model loading and inference
3. Update documentation

### Custom Model Configurations
```python
# Example: Custom attention window for very long audio
if self._parakeet_model is None:
    self._parakeet_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    # Custom configuration
    self._parakeet_model.change_attention_model(
        "rel_pos_local_attn",
        context_sizes=[256, 256]  # Larger context for long audio
    )
```

## Future Enhancements

- [ ] Timestamp extraction from Parakeet models
- [ ] Speaker diarization integration
- [ ] Streaming transcription API
- [ ] Fine-tuning support for domain-specific models
- [ ] Multi-language Parakeet models
- [ ] Punctuation and capitalization models

## References

- [NVIDIA Parakeet Models](https://catalog.ngc.nvidia.com/models?query=parakeet)
- [NeMo ASR Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/intro.html)
- [Parakeet Paper](https://arxiv.org/abs/2305.05084)