---
id: TASK-32148
title: Add repeatable opt-in live TTS cancellation and playback validation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 05:34'
updated_date: '2026-09-09 08:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the saved speech validation probes into a reproducible developer tool, and close real Kokoro cancellation and sustained-playback coverage gaps on macOS ARM.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The opt-in command accepts explicit local assets, runtime and new private output paths; help and cheap tests cause no inference, audio, network downloads or user-configuration writes.
- [x] #2 Actual delegated PyTorch and ONNX native calls expose monotonic entry and exit evidence so cancellation is proved to overlap inference, followed by successful successor playback.
- [x] #3 Repeated synthesis and playback retain complete audio and record bounded observer memory, resource snapshots and cleanup assertions rather than unconditional success flags.
- [x] #4 Validation artifacts identify tested application, packages, models, audio bytes and physical playback outcomes; full-content transcription remains separate from transport and cleanup evidence.
- [x] #5 Targeted harness regressions and real macOS CPU, MPS and ONNX runs pass; any unavailable cases have concrete backlog follow-ups and prerequisite evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing ADR-023 TTS provider ownership, ADR-039 Global/Studio ownership and ADR-040 Lab audition ownership apply.
Reason: Test-only promotion of existing live validation seams; production fixes discovered by the probe are tracked separately before implementation.

1. Add cheap failing negative-control tests for opt-in admission, real inference overlap, playback duration/content evidence and joined ownership.
2. Promote the existing validators into import-safe local-asset runner and separate local-ASR verifier, with an isolated private worker, bounded observations and explicit package provenance; provide a concise runbook.
3. Observe unchanged real PyTorch forward and ONNX session calls, prove cancellation falls between entry and exit, and verify terminal lifecycle plus successor playback and bounded repeated runs.
4. Execute serialized macOS CPU, MPS and ONNX qualification with complete source/playback artifacts and separate content results. Record any production defect before fixing it.
5. Run targeted harness and affected lifecycle tests, verify real cleanup and unchanged user configuration, and file remaining hardware/provider prerequisites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added an opt-in local-asset validation runner and separate full-content ASR verifier, with a documented private-profile workflow. The mounted Speech Lab and trusted Console path use real inference, codecs and physical playback. Negative controls cover admission, actual inference overlap, full-duration playback, bounded observations, source/asset identity and retained cleanup; imports/help remain inert. Existing ADR-023/039/040 apply; no new architecture decision is needed for the test-only tool.

The original CPU/MPS/ONNX English matrix passed all scenarios with 33 normalized-exact complete clips, three observed native-overlap Stops, successor recovery and bounded repeats. The final installed wheel passed another nine complete clips and three Stop/recovery controls. Failed earlier probes remain distinct. Targeted harness and affected integration regressions pass; the user config hash is unchanged and all recorded owned processes exited. Evidence and scope limits are in Docs/QA/tts-macos-burndown-2026-09-09/{kokoro-english,integration}/README.md and Docs/Development/TTS/Live_Validation.md. This does not claim acoustic loopback, full-shell navigation or indefinite memory-leak freedom.
<!-- SECTION:NOTES:END -->
