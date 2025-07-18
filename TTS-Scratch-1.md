# TTS Implementation Scratch Pad

## Working Notes and Issues

### Current Task: AllTalk TTS Backend Implementation [2025-07-18]

**Goal**: Create AllTalk TTS backend using the existing configuration and OpenAI-compatible API.

**Previous Task**: ✅ Kokoro True Streaming Implementation - Successfully implemented chunk-based streaming for both ONNX and PyTorch backends.

---

### Implementation Checklist

- [x] Kokoro true streaming implementation ✅
  - ONNX: Chunk-based streaming for all formats
  - PyTorch: Per-chunk streaming instead of collecting all
  - Fixed convert_audio_sync issue by using async properly
  - Added first chunk latency tracking
- [ ] Kokoro progress callbacks
- [ ] Kokoro word-level timestamps
- [ ] Chatterbox async streaming fix
- [ ] Chatterbox crossfade implementation
- [ ] AllTalk backend creation
- [ ] AllTalk backend registration
- [ ] Progress interface standardization
- [ ] Cost estimation implementation
- [ ] Testing framework setup

---

### Issues Encountered

#### Issue #1: convert_audio_sync method doesn't exist
**Problem**: Kokoro backend calls `self.audio_service.convert_audio_sync()` but AudioService only has async `convert_audio()`
**Solution**: Either create sync wrapper or properly use async version in streaming
**Status**: In progress - will fix by using async properly in chunked streaming

#### Issue #2: Non-PCM formats collect all audio before yielding
**Problem**: Current implementation for non-PCM formats (mp3, opus, etc.) collects all samples in a list before converting and yielding
**Solution**: Implement chunk-based conversion to enable true streaming
**Status**: Working on solution 

---

### Code Snippets and Tests

```python
# Test code and snippets will be added here during implementation
```

---

### Performance Benchmarks

| Backend | First Chunk Latency | Total Generation Time | Real-time Factor |
|---------|-------------------|---------------------|------------------|
| Kokoro ONNX | TBD | TBD | TBD |
| Kokoro PyTorch | TBD | TBD | TBD |
| Chatterbox | TBD | TBD | TBD |
| OpenAI | TBD | TBD | TBD |
| ElevenLabs | TBD | TBD | TBD |
| AllTalk | TBD | TBD | TBD |

---

### API Endpoints and Formats

#### AllTalk API
- Base URL: `http://127.0.0.1:7851`
- Speech endpoint: `/v1/audio/speech`
- Voices endpoint: `/api/voices`
- Format: OpenAI-compatible with language extension

---

### Dependencies and Imports

```python
# Track any new dependencies needed
# kokoro-onnx - already listed
# chatterbox-tts - already listed
# No new dependencies for AllTalk (uses httpx)
```

---

### Configuration Changes

```toml
# Any config.toml changes needed
# AllTalk config already exists
```

---

### Testing Commands

```bash
# Commands to test each backend
# python -m pytest Tests/TTS/test_streaming.py
# python -m pytest Tests/TTS/test_alltalk.py
```

---

### Notes and Observations

1. Kokoro currently uses `create_stream()` but collects all samples before yielding
2. Chatterbox has a `generate_stream()` method but async wrapper is incomplete
3. AllTalk should be straightforward since it's OpenAI-compatible
4. Progress tracking needs to be standardized across all backends
5. Consider adding a streaming test utility for all backends

---

### TODO During Implementation

1. Check if kokoro_instance.create_stream is truly async or needs asyncio.to_thread
2. Verify Chatterbox thread safety for streaming
3. Test AllTalk with different voice files
4. Benchmark memory usage during streaming
5. Create unified error handling strategy

---

### References

- Kokoro-FastAPI: https://github.com/remsky/Kokoro-FastAPI
- AllTalk GitHub: [Need to find official repo]
- Chatterbox Docs: [Need to find official docs]

---

Last Updated: 2025-07-18 (Created)