---
id: TASK-32162
title: Stop Kokoro PyTorch between internal synthesis segments
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 06:25'
updated_date: '2026-09-09 08:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A real CPU cancellation run joined active inference but still started another upstream model segment after Stop. Preserve native ownership while preventing subsequent segment work and discarded audio.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stop or close during a native PyTorch segment joins that segment and does not start a later upstream segment across streaming and timestamp paths.
- [x] #2 Repeated cancellation preserves ownership and successor requests still produce complete speech.
- [x] #3 Targeted failure-first regressions and real CPU and MPS cancellation evidence pass with no audio emitted for the cancelled request.
- [x] #4 Stop during language phonemization prevents the next model forward call; request-scoped cancellation leaves the shared model and a successor request usable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing ADR-023 ownership, ADR-140 official Kokoro runtime and ADR-142 language frontends apply.
Reason: Routine cancellation repair inside the existing provider/runtime boundary; the official per-call model override carries a request-local Stop gate without mutating the shared model.

1. Reproduce the real CPU extra-segment failure using a controlled two-segment official-pipeline boundary.
2. Signal cooperative stop promptly on caller cancellation and backend close, check it between upstream segments, and retain actual native work through cleanup.
3. Verify normal full output, streaming/timestamps, repeated cancellation, successor requests and real CPU/MPS inference intervals.
4. Record the incident and exact runtime evidence; run affected tests and review before completion.
5. Reproduce Stop during pipeline G2P before model entry, then guard the official per-request model call so completed phonemization cannot start inference after Stop. Verify both public paths and an unaffected successor, without shared pipeline/model mutation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a request-local cooperative Stop event to retained PyTorch generation and check it around each upstream segment. A request-local model callable passed through the official pipeline model argument also prevents model entry after Stop during G2P. Upstream chunking, the shared model/pipeline and normal successor requests remain unchanged; current native work is still joined before model release. Existing ADR-023/140 and the language integration in ADR-142 apply.

Preserved the real CPU extra-segment failure and fake G2P post-Stop-forward reproduction. Failure-first PCM/WAV/timestamp cancellation/close tests pass, including repeated cancellation and unchanged shared-state successors. Independent probes against the actual KPipeline call contract passed all ten review cases. Real CPU and MPS mid-inference Stop, complete successor playback and final installed-wheel controls pass with zero cancelled output or later native calls. Evidence: Docs/QA/tts-macos-burndown-2026-09-09/{kokoro-english,kokoro-languages,integration}/README.md.
<!-- SECTION:NOTES:END -->
