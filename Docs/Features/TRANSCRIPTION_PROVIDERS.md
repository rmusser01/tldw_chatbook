# Transcription Provider Comparison Guide

This guide helps you choose the right transcription provider for your use case.

## Quick Comparison Table

| Feature | Faster-Whisper | Qwen2Audio | Parakeet | Canary |
|---------|----------------|------------|----------|---------|
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Accuracy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Language Support** | 100+ | Multiple | English | 4 (en,de,es,fr) |
| **Translation** | To English | ❌ | ❌ | Any direction |
| **Model Sizes** | 39MB-1.5GB | 15GB | 600MB-1.1GB | 1GB |
| **GPU Required** | Optional | Recommended | Optional | Recommended |
| **Real-time** | ❌ | ❌ | ✅ | ✅ |
| **Long Audio** | ✅ | ✅ | ✅ | ✅ (chunked) |

## Detailed Provider Profiles

### Faster-Whisper

**Best for:** General-purpose transcription with broad language support

**Strengths:**
- Excellent accuracy across many languages
- Automatic language detection
- Multiple model sizes for speed/accuracy trade-off
- Translation to English from any language
- Works well on CPU

**Limitations:**
- Translation only to English
- Not optimized for real-time
- Larger models can be slow without GPU

**Recommended Models:**
- `tiny` or `base`: Quick transcription, good for drafts
- `small` or `medium`: Balanced speed and accuracy
- `large-v3`: Best accuracy, slower processing
- `distil-large-v3`: Good accuracy, 6x faster than large-v3

**Use Cases:**
- Podcast transcription
- Meeting notes
- Multi-language content
- Quick drafts with smaller models

### Qwen2Audio

**Best for:** Advanced audio understanding and analysis

**Strengths:**
- State-of-the-art audio understanding
- Handles complex audio scenes
- Can understand context and intent
- Multimodal capabilities

**Limitations:**
- Large model size (15GB)
- Requires significant GPU memory
- No translation support
- Slower processing

**Use Cases:**
- Audio content analysis
- Complex audio scenes
- Research applications
- When context understanding is critical

### Parakeet

**Best for:** Real-time English transcription

**Strengths:**
- Optimized for streaming/real-time
- Very fast inference
- Low latency
- Multiple architecture options (TDT, RNNT, CTC)
- Efficient memory usage

**Limitations:**
- English only
- No translation
- Less accurate than larger models
- Limited punctuation

**Recommended Models:**
- `parakeet-tdt-0.6b-v2`: Fastest, good for real-time
- `parakeet-tdt-1.1b`: Better accuracy, still fast
- `parakeet-rnnt-*`: Good for streaming applications
- `parakeet-ctc-*`: Traditional ASR approach

**Use Cases:**
- Live transcription
- Voice assistants
- Real-time captioning
- Low-latency applications

### Canary

**Best for:** Multilingual transcription and translation

**Strengths:**
- Supports 4 languages (en, de, es, fr)
- Translation between all supported languages
- Chunked processing for long audio
- Good accuracy
- Punctuation and capitalization

**Limitations:**
- Limited to 4 languages
- Requires GPU for best performance
- Larger model size

**Recommended Models:**
- `canary-1b-flash`: Optimized for speed
- `canary-1b`: Standard model, better accuracy

**Use Cases:**
- Multilingual meetings
- Content translation
- European language content
- Long-form content with translation needs

## Choosing the Right Provider

### By Use Case

**Podcasts/Long Audio:**
- English only: Faster-Whisper (medium or large)
- Multiple languages: Faster-Whisper
- Need translation: Canary (for supported languages) or Faster-Whisper (to English)

**Real-time/Streaming:**
- First choice: Parakeet (tdt models)
- Alternative: Canary with small chunks

**Multilingual Content:**
- Many languages: Faster-Whisper
- European languages with translation: Canary

**Quick Drafts:**
- Faster-Whisper (tiny or base)
- Parakeet (0.6b models)

**Best Quality:**
- Faster-Whisper (large-v3)
- Qwen2Audio (for complex audio)
- Canary (for supported languages)

### By Hardware

**CPU Only:**
- Faster-Whisper with int8 compute
- Small Parakeet models
- Avoid Qwen2Audio

**Limited GPU (4-8GB):**
- Faster-Whisper (all models)
- Parakeet (all models)
- Canary (may need reduced batch size)

**Powerful GPU (16GB+):**
- All providers work well
- Qwen2Audio becomes viable
- Can use largest models

### By Language

**English Only:**
- Parakeet for speed
- Faster-Whisper for accuracy

**European Languages:**
- Canary for en/de/es/fr with translation
- Faster-Whisper for broader support

**Asian Languages:**
- Faster-Whisper
- Qwen2Audio

**Rare Languages:**
- Faster-Whisper (check language list)

## Performance Tuning Tips

### For Speed
1. Use smaller models (tiny, base, 0.6b variants)
2. Enable GPU acceleration if available
3. Use int8 compute type for CPU
4. Disable VAD if not needed
5. Use Parakeet for English-only content

### for Accuracy
1. Use larger models (large-v3, 1.1b variants)
2. Specify source language (avoid auto-detection)
3. Enable VAD for noisy audio
4. Use Canary for supported languages
5. Pre-process audio (noise reduction)

### For Long Audio
1. Canary with appropriate chunk size
2. Faster-Whisper handles well by default
3. Adjust chunk_length_seconds in config:
   - 10s for timestamps accuracy
   - 40s for general use
   - 60s+ for lectures

## Common Combinations

### "Fast and Good Enough"
- Provider: Faster-Whisper
- Model: base or small
- Settings: CPU with int8

### "Best Quality"
- Provider: Faster-Whisper  
- Model: large-v3
- Settings: GPU acceleration

### "Real-time English"
- Provider: Parakeet
- Model: parakeet-tdt-0.6b-v2
- Settings: GPU if available

### "Multilingual with Translation"
- Provider: Canary
- Model: canary-1b-flash
- Settings: GPU recommended

## Troubleshooting Guide

### "Transcription is too slow"
- Switch to smaller model
- Enable GPU acceleration
- Use Parakeet for English
- Check CPU/GPU usage

### "Poor accuracy"
- Use larger model
- Specify correct source language
- Enable VAD for noisy audio
- Check audio quality

### "Out of memory"
- Use smaller model
- Reduce batch size
- Switch to CPU processing
- Use chunked processing (Canary)

### "Language not detected correctly"
- Manually specify language
- Use longer audio sample
- Ensure clear speech in first 30s

### "Translation not working"
- Check provider supports translation
- Verify source/target language combo
- Faster-Whisper only translates to English
- Canary only supports en/de/es/fr