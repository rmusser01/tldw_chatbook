---
id: TASK-32111
title: Replace placeholder Kokoro PyTorch synthesis with the official runtime
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 00:37'
updated_date: '2026-09-09 02:48'
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
- [x] #5 Named voice lookup rejects traversal, absolute names, and symlinks escaping the configured voice directory.
- [x] #6 Streaming and timestamp generation use the same language for Hindi, Italian, and Brazilian Portuguese voices in both Kokoro engines.
- [x] #7 The PR keyboard evidence waits for actual focus and selector state before sending keys on Windows.
- [x] #8 Kokoro asset downloads cannot follow a pre-existing partial-file symlink and clean up only the temporary file they create.
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
6. Address Qodo review with central voice-path validation, one shared backend language map, public API documentation, and regression tests. Synchronize the unchanged Windows GGUF keyboard test using its existing bounded state helper, then rerun targeted CI. Rebuild and revalidate the final installed runtime before integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced Kokoro's incompatible placeholder architecture and random waveforms with official KModel/KPipeline synthesis under ADR-140. Fixed Global engine inheritance, configured checkpoint/fallback routing, deferred initialization, full voice-pack blends, MPS Fourier compatibility, retained native cancellation/close, and typed dependency/language/length recovery.

Review fixes confine named voices in the wrapper and real backend download/load routes, reject traversal/absolute names/outside links, and use exclusive temporary download files after reproducing partial-file symlink corruption. Both engines and both generation APIs share one language map. Public API contracts are documented. The MPS review allegation was disproved against supported upstream TorchSTFT and a real MPS numerical test including recursive device movement. Independent final review found no remaining actionable findings.

Final validation: 340 targeted tests passed, one optional ONNX skip; no full suite. Nine final installed-wheel playback clips passed complete-content checks across MPS WAV, British CPU MP3 at speed 1.25, and Python 3.13 ONNX, in addition to the 15 initial clips. Speech Lab and trusted Speak replies crossed real device/player boundaries and joined cleanup. Installed Python 3.13 dependency guidance passed. Final production hashes match the qualified wheel; user configuration is unchanged. Fresh environments use Torch 2.14 and Transformers 5.16.1. Exact raw transcripts and limitations remain in the QA evidence.

Fixed the Windows GGUF keyboard evidence's deferred-focus/overlay synchronization without changing production UI. Reviewed diagnostic statement changes under TASK-494/ADR-029: six fewer calls, one fewer model-path candidate, no new sinks; replacement logging records only validated compute device. Regenerated inventory passes its source guard. New/replaced-file Ruff and changed-file formatting pass; existing-file findings have no additions.

Updated runtime/backend/bridge, dependency markers, UI guidance, tests, user guide, ADR index, diagnostic inventory and testing lesson. Evidence: Docs/QA/tts-runtime-recovery-2026-09-09/README.md, evidence-summary.json and review-validation.json. ADR: backlog/decisions/140-official-kokoro-pytorch-runtime.md. Native model/recovery qualification and unresolved upstream limitations are recorded separately under TASK-32112/TASK-32113.
<!-- SECTION:NOTES:END -->
