---
id: TASK-32096
title: Validate real Chatterbox and audio.cpp speech on macOS ARM
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-08 22:15'
updated_date: '2026-09-08 23:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the live inference gaps remaining after TASK-32076 by exercising the real local models and audio device on Apple Silicon.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatterbox CPU and MPS generation and playback have recorded real-model outcomes.
- [x] #2 Pinned audio.cpp CPU and Metal generation and playback have recorded real-model outcomes.
- [x] #3 Repeated Speak replies and Speech Lab paths are exercised with full-content audio verification, or a concrete runtime blocker is documented.
- [x] #4 Any confirmed application defects have focused regression evidence and are repaired before successful paths are claimed.
- [x] #5 Runtime, model, configuration isolation, and remaining limitations are documented without modifying the user profile.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin current dev and isolated runtime/model/config paths.
2. Install the released Chatterbox runtime and build the compatible audio.cpp release for CPU and Metal.
3. Exercise real Speech Lab and Speak replies admission, repeated generation, full decoding, actual device playback, and independent transcription for each available compute path.
4. Reproduce any application defect, add focused regression coverage, repair its cause, and repeat the affected live run.
5. Record exact runtime/model identities and outcomes, including unavailable or failed paths.
ADR required: no
ADR path: N/A; existing backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md and backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md apply.
Reason: verify and, if necessary, repair existing inference and playback contracts without changing architectural boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Validated real Chatterbox CPU/MPS and pinned audio.cpp CPU/Metal synthesis through mounted Speech Lab and repeated Speak replies, with full decode, actual device completion and independent transcription. A further installed-wheel MPS/MP3 run validates setuptools 81; 15 real clips passed. Fixed missing native capability wiring in Console destination resolution and repaired Chatterbox extra dependency bounds after fresh-wheel installation failures. The two native regressions failed before and passed after; 181 Console/resolver and 215 backend cases pass. Fresh-wheel import/resource/isolation checks and all six preflight checks pass. Updated recovery guidance and evidence lessons. Exact runtimes, hashes, model limits and evidence are in Docs/superpowers/qa/tts-macos-live-2026-09-08/verification.md. Existing ADRs 023 and 050 apply; no new boundary or ADR required. PR integration and hosted review remain pending.
<!-- SECTION:NOTES:END -->
