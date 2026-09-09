---
id: TASK-32111
title: Replace placeholder Kokoro PyTorch synthesis with the official runtime
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 00:37'
updated_date: '2026-09-09 02:04'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real checkpoint validation found that the advertised Kokoro PyTorch path loads an incompatible placeholder architecture and contains random-noise generation. Restore intelligible local speech while preserving existing delivery and settings behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Official Kokoro v1 weights generate intelligible complete speech through Speech Lab and Speak replies on macOS CPU and MPS.
- [x] #2 Missing or unsupported PyTorch runtime dependencies produce actionable errors without breaking ONNX or other TTS backends.
- [x] #3 Language selection, voice packs and blends, speed, encoded audio limits, and inference failure paths have targeted regression coverage.
- [x] #4 Pinned runtime/model provenance, Python compatibility limits, real playback evidence, and relevant documentation are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/140-official-kokoro-pytorch-runtime.md
Reason: replace the incompatible placeholder with the upstream optional runtime and record Python compatibility/dependency ownership. Existing ADR-023 and ADR-039 provider and settings boundaries remain unchanged.

1. Record official checkpoint load failure and upstream runtime/config/voice contracts.
2. Add focused failing tests for real upstream output delegation, language normalization, tensor voice packs/blends, missing dependencies, and off-loop model loading.
3. Replace placeholder synthesis and architecture with the official Kokoro runtime; preserve existing backend delivery limits and settings ownership.
4. Run targeted regression/packaging checks and real CPU/MPS Speech Lab plus Speak replies playback with independent transcription.
5. Document provenance, Python limits and remaining coverage, review, then integrate through the authorized dev PR workflow.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced Kokoro's placeholder architecture and random waveforms with official KModel/KPipeline synthesis under ADR-140. Fixed Global engine inheritance, configured checkpoint/fallback routing, deferred initialization, full voice-pack blends, MPS Fourier compatibility, retained native cancellation/close, and typed safe dependency/language/length recovery.

Validation: 301 targeted regressions passed (one optional ONNX skip), then all 65 final runtime/diagnostic tests passed. Fifteen real CPU/MPS/ONNX clips across source and fresh Python 3.12/3.13 wheels passed complete-content checks. Fresh Torch 2.14/Transformers 5.16.1 playback, British bf_emma speed 1.25 MP3, and Python 3.13 ONNX all passed. New/replaced-file Ruff and changed-file formatting pass; existing-file findings decreased 129 to 126 with no new code/message findings. Independent review has no outstanding findings. No full suite was run.

Updated runtime/backend/bridge, optional dependency markers, UI error copy, regression tests, user guide, ADR index and testing lesson. Exact provenance and qualifications/limits are in Docs/QA/tts-runtime-recovery-2026-09-09/README.md and evidence-summary.json. User configuration hash is unchanged. ADR: backlog/decisions/140-official-kokoro-pytorch-runtime.md.
<!-- SECTION:NOTES:END -->
