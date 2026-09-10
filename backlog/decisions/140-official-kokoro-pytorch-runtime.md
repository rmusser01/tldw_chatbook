# ADR-140: Official Kokoro PyTorch Runtime

Status: Accepted
Date: 2026-09-09
Related task: TASK-32111
Extends: [ADR-023](023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md) and
[ADR-039](039-global-and-studio-tts-settings-ownership.md)

## Context

Real validation of the advertised PyTorch option loaded the official Kokoro v1
checkpoint (SHA-256 `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`).
The app's placeholder transformer cannot accept its component state dictionaries.
The subsequent generation helpers also ignore the model and synthesize random
noise. Fixture-based delivery tests did not establish real model compatibility.

Upstream Kokoro supplies the actual model, vocoder, phoneme vocabulary and
language pipelines. Its published 0.9.4 runtime requires Python >=3.10,<3.13;
Chatbook continues to support Python >=3.11 and has a separately working ONNX path.

## Decision

- Use the optional upstream `kokoro` 0.9.x runtime for PyTorch synthesis. Keep it
  behind the existing lazy Kokoro backend; importing or using another provider
  must not require it. Declare it in local TTS installation extras only on
  Python <3.13. Report installation guidance and the ONNX alternative when absent.
- Replace the placeholder architecture, tokenizer and random waveform helpers
  with one upstream `KModel` shared by language-specific `KPipeline` instances.
  Use the configured checkpoint and an adjacent `config.json` when present;
  otherwise obtain the official v1 configuration at pinned model revision
  `f3ff3571791e39611d31c381e3a41a3af07b4987` through the upstream artifact cache.
  Model and voice deserialization must use restricted weights-only loading.
  Named voice packs must remain inside the configured directory after central
  filename/path validation, including backend download and timestamp paths.
  Asset downloads use exclusively created temporary files in the validated
  destination directory and remove only their own partial file after failure.
- Pass text, normalized language, the complete float32 voice pack and speed to
  the upstream pipeline. Concatenate every returned waveform segment. Do not
  resample a second time to apply speed, or synthesize replacement audio after a
  dependency, model, phonemization or inference failure.
- Keep model initialization and inference off the event loop. Preserve existing
  Global/Studio/profile ownership, backend selection, codecs and encoded-audio
  limits. Shared model/config/G2P assets remain shared artifacts under ADR-040;
  validation isolates those caches and does not change user settings.
- On MPS, delegate only the upstream STFT transform and inverse to CPU and return
  their tensors to the original device. Install the wrapper after checkpoint
  loading, keeping learned weights and the upstream reconstruction math intact.
  Do not enable a process-wide fallback or select the upstream approximate
  complex-free inverse. Neural inference remains on MPS.
- Retain and join native worker tasks through cancellation and close, including
  initialization, downloads, voice loading and timestamp generation. Cancelled
  initialization must not publish a model after the backend has closed.
- Default reply requests inherit the applied immutable Global Kokoro engine
  setting. An explicit request engine overrides it; Speech Lab's separate ONNX
  switch remains an explicit choice. Resolve the route before taking the backend
  lock so one native engine cannot be used under another engine's lock.
- Report dependency/setup and oversized non-English input errors with typed,
  fixed recovery guidance. Reject paragraphs above the upstream 510-phoneme
  limit before synthesis and preserve explicit newline boundaries; upstream
  0.9.4 otherwise silently truncates some non-English input.

## Alternatives and consequences

Implementing Kokoro's architecture/vocoder locally would duplicate a maintained
runtime and repeat the missing-model failure. Silently using ONNX when the user
explicitly selects PyTorch would hide whether that choice works. Neither is used.
The PyTorch option currently needs Python 3.11 or 3.12; Python 3.13+ remains
supported through ONNX. Future upstream Python support can relax the dependency
marker after real installation and playback validation.

The runtime may fetch small model configuration and language assets on first
deliberate use. Offline users must provision those caches (or adjacent model
configuration) in addition to checkpoint and voice files. Japanese and Chinese
require the corresponding upstream Misaki language extras. Live qualification
must name the exercised language/device/runtime instead of claiming every
upstream language or platform.

Sources: [official runtime](https://github.com/hexgrad/kokoro),
[runtime metadata](https://pypi.org/pypi/kokoro/0.9.4/json),
[official weights](https://huggingface.co/hexgrad/Kokoro-82M).
