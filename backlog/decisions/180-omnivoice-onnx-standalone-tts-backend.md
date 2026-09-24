# ADR-180: OmniVoice ONNX standalone TTS backend (alongside audio.cpp)

Status: Accepted (2026-09-23)
Date: 2026-09-23
Companion spec: [2026-09-23 OmniVoice ONNX TTS backend design](../../Docs/superpowers/specs/2026-09-23-omnivoice-onnx-tts-backend-design.md)
Related: [ADR-023](023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md) (TTS
adapter registry — followed, not amended), [ADR-039](039-global-and-studio-tts-settings-ownership.md)
(TTS settings ownership), [ADR-051](051-private-tts-clone-reference-assets.md)
(clone reference assets), [ADR-080](080-model-machine-memory-fit-estimation.md)
(machine-memory fit)

## Decision

Add `omnivoice` as the 8th built-in TTS provider via the legacy bridge
(kokoro/higgs shape), not a native ADR-023 adapter and not an external runtime:

- In-process engine: onnxruntime sessions + numpy diffusion sampler + `tokenizers`.
- Sampler/prompt logic ported from the Apache-2.0 k2-fsa/omnivoice upstream with
  attribution. The AFun9/Omnivoice-onnx reference repo is unlicensed — never copied.
- Managed acquisition through the Model_Artifacts curated registry (CC-BY-NC weights
  + Boson tokenizer license surfaced at consent), config-path fallback.
- Cloning via ADR-051 canonical reference assets (audio + transcript).

## Context

Chatbook already runs OmniVoice through the audio.cpp runtime (GGUF/safetensors
recipes). The user wants the ct03/omnivoice-onnx-int8hq export as a standalone
backend: a pip-installable onnxruntime engine with no native-binary supervision.
RTF ~3–7 on CPU makes it batch-only; it never joins streaming/duplex paths.

## Consequences

- A second OmniVoice runtime coexists with audio.cpp recipes by explicit user
  choice; audio.cpp remains the native-runtime lane, this is the pip-only lane.
- New `omnivoice_tts` extra (onnxruntime, tokenizers).
- No changes to ADR-023/039/051/080 — they are followed, not amended.
