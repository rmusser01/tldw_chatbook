# TTS Implementation Scratch Pad

## Working Notes and Issues

### Current Task: Add Progress Callbacks to Kokoro Backend [2025-07-18]

**Goal**: Implement progress tracking callbacks in Kokoro backend to provide real-time generation progress updates.

**Previous Tasks**: 
- ✅ Kokoro True Streaming Implementation - Successfully implemented chunk-based streaming for both ONNX and PyTorch backends.
- ✅ AllTalk TTS Backend Implementation - Created and integrated AllTalk backend with UI support.

---

### Implementation Checklist

- [x] Kokoro true streaming implementation ✅
  - ONNX: Chunk-based streaming for all formats
  - PyTorch: Per-chunk streaming instead of collecting all
  - Fixed convert_audio_sync issue by using async properly
  - Added first chunk latency tracking
- [x] Kokoro progress callbacks ✅
  - Added ProgressCallback protocol to base_backends.py
  - Implemented progress reporting in both ONNX and PyTorch paths
  - Reports progress, tokens processed, status, and metrics
  - Includes completion reporting with final metrics
- [ ] Kokoro word-level timestamps
- [ ] Chatterbox async streaming fix
- [ ] Chatterbox crossfade implementation
- [x] AllTalk backend creation ✅
  - Created alltalk.py with full OpenAI-compatible implementation
  - Supports voice mapping and language selection
  - Proper error handling for local server connection
- [x] AllTalk backend registration ✅
  - Added to TTSBackendManager._register_builtin_backends()
  - Registered with pattern "alltalk_*"
- [x] AllTalk UI Integration ✅
  - Added to provider list in STTS_Window.py
  - Added model-to-backend mapping in tts_events.py
  - Added voice options for AllTalk
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
**Status**: ✅ RESOLVED - Implemented chunk-based streaming for all formats

#### Issue #3: Progress tracking standardization
**Problem**: No standardized way to track and report TTS generation progress across backends
**Solution**: Created ProgressCallback protocol in base_backends.py
**Status**: ✅ RESOLVED - Implemented for Kokoro, ready to extend to other backends 

---

### Code Snippets and Tests

```python
# Test code and snippets will be added here during implementation
```

---

### Summary of Completed High-Priority Tasks

1. **Kokoro True Streaming** ✅
   - Implemented real-time chunk-based streaming for all formats
   - Fixed async audio conversion issue
   - Added first chunk latency tracking
   - Both ONNX and PyTorch backends now stream properly

2. **AllTalk Backend** ✅
   - Created complete OpenAI-compatible backend
   - Registered in backend manager
   - Added UI integration (provider, model, voice options)
   - Added to event handler model mapping

3. **Progress Callbacks** ✅
   - Created standardized ProgressCallback protocol
   - Implemented in Kokoro backend (both ONNX and PyTorch)
   - Reports progress, tokens, status, and detailed metrics
   - Ready to extend to other backends

### Performance Benchmarks

| Backend | First Chunk Latency | Total Generation Time | Real-time Factor |
|---------|-------------------|---------------------|------------------|
| Kokoro ONNX | <1s expected | TBD | Target: 35x-100x |
| Kokoro PyTorch | <1s expected | TBD | Target: 35x-100x |
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