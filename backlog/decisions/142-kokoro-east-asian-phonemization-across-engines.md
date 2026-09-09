# ADR-142: Kokoro East Asian phonemization across engines

Status: Accepted
Date: 2026-09-09
Related task: TASK-32149
Extends: [ADR-140](140-official-kokoro-pytorch-runtime.md) and
[ADR-023](023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)

## Context

The real language matrix exposed a frontend mismatch. ONNX's generic eSpeak
path rejects the application aliases `fr` and `zh`. With Japanese text it can
produce audio describing Chinese characters instead of speaking the requested
Japanese reply. The official Kokoro v1 PyTorch pipeline uses Misaki's Japanese
and Chinese frontends for those languages. Its Japanese extras install UniDic's
Python package separately from the dictionary data; missing data currently
becomes an unhelpful retryable generation error.

## Decision

Use the same optional upstream Misaki `JAG2P` and v1 `ZHG2P` frontends for ONNX
Japanese and Mandarin, then pass their phonemes through ONNX's existing
`is_phonemes` interface. Keep the selected ONNX model, voice, device, speed and
output format. Normalize the French locale to eSpeak's `fr-fr`. Other languages
continue through the existing ONNX frontend; explicit phoneme input bypasses
text conversion.

Load language dependencies only when the corresponding language is requested.
English ONNX must remain independent of Kokoro's PyTorch package and Japanese
or Chinese extras. Missing extras or Japanese dictionary data produce typed,
fixed installation guidance. Users provision `misaki[ja]`/`misaki[zh]` and the
UniDic dictionary in their chosen TTS environment; this change neither installs
dependencies nor downloads dictionaries automatically.

Cache language frontends only within the backend instance. Phonemization runs
inside the retained ONNX worker, outside the application event loop. Stop joins
any active frontend work and prevents it from starting inference afterward.
Close releases frontend caches after retained work finishes. Global/Studio
settings, profile authority and asset ownership remain unchanged.

The official PyTorch pipeline also phonemizes inside its generator before
entering the model. Pass a request-local callable through its existing optional
model argument to check Stop immediately before model entry. This preserves
upstream chunking and leaves the shared model and pipeline unchanged; a call
already in progress still joins before its owner is released.

## Alternatives and qualification limits

Generic eSpeak's Japanese character descriptions cannot establish Japanese
speech. Silently switching engines hides the user's selected runtime; rejecting
all Japanese ONNX requests would discard a supported phoneme-input path.
Reimplementing language tokenization duplicates the model's maintained frontend.

Qualification names the exact language, voice, runtime and dictionary versions.
ASR spelling/script variants and pronunciation uncertainty remain visible;
successful transport alone cannot prove the spoken content is correct.

References: [official Kokoro pipeline](https://github.com/hexgrad/kokoro),
[Misaki frontends](https://github.com/hexgrad/misaki),
[Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx).
