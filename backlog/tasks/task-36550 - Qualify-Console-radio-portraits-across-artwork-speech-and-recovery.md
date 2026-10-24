---
id: TASK-36550
title: Qualify Console radio portraits across artwork speech and recovery
status: To Do
created_date: 2026-10-23 22:56
dependencies:
- TASK-36547
- TASK-36548
- TASK-36549
labels:
- console
- qa
- documentation
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
- Docs/superpowers/reviews/2026-10-23-console-radio-portraits-implementation-review.md
updated_date: 2026-10-23 23:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the complete first-version portrait experience with imported and generated artwork, real speech output and profile recovery, recording the limits of actual device and provider coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A mounted editor-to-Console journey imports, previews, applies, reopens and removes a set; generation uses a real configured image-edit backend and demonstrates reviewed per-slot replacement.
- [ ] #2 Real TTS qualification separately covers streaming and supported WAV artifact output with onset, nonzero quiet pauses, continuous updates without test-only publication, final-syllable/device-tail completion, stop/replacement during pending output, no early sentence/listening handoff and conversation changes; hardware latency and untested players are reported explicitly.
- [ ] #3 Known waveform and measured visible output prove the 100 ms onset/silence target on a qualified local path; timing-unavailable external playback retains truthful fallback without claiming synchronization.
- [ ] #4 Full-profile backup/restore preserves frames and retires stale presentation owners; original artwork, ordinary export/sync formats and soft-delete/restore behavior remain unchanged.
- [ ] #5 Targeted integration, static, migration packaging, design-token, CSS and Backlog checks pass or unrelated baseline failures are reproduced and reported; no full test sweep runs without explicit user request.
- [ ] #6 User and architecture documentation describe preparation, six-request generation, profile-local storage, motion settings and capability fallback; evidence and each task's implementation notes match what was actually exercised.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
