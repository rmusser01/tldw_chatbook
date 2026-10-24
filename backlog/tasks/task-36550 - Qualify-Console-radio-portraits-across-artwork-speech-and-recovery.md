---
id: TASK-36550
title: Qualify Console radio portraits across artwork speech and recovery
status: In Progress
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
- Docs/superpowers/qa/console-radio-portraits/2026-10-23/qualification.md
- Docs/superpowers/reviews/2026-10-23-task-36550-review.md
updated_date: 2026-10-24 04:11
assignee:
- '@codex'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the complete first-version portrait experience with imported and generated artwork, real speech output and profile recovery, recording the limits of actual device and provider coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A mounted editor-to-Console journey imports, previews, applies, reopens and removes a set; generation uses a real configured image-edit backend and demonstrates reviewed per-slot replacement.
- [ ] #2 Real TTS qualification separately covers streaming and supported WAV artifact output with onset, nonzero quiet pauses, continuous updates without test-only publication, final-syllable/device-tail completion, stop/replacement during pending output, no early sentence/listening handoff and conversation changes; hardware latency and untested players are reported explicitly.
- [x] #3 Known waveform and measured visible output prove the 100 ms onset/silence target on a qualified local path; timing-unavailable external playback retains truthful fallback without claiming synchronization.
- [x] #4 Full-profile backup/restore preserves frames and retires stale presentation owners; original artwork, ordinary export/sync formats and soft-delete/restore behavior remain unchanged.
- [x] #5 Targeted integration, static, migration packaging, design-token, CSS and Backlog checks pass or unrelated baseline failures are reproduced and reported; no full test sweep runs without explicit user request.
- [ ] #6 User and architecture documentation describe preparation, six-request generation, profile-local storage, motion settings and capability fallback; evidence and each task's implementation notes match what was actually exercised.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Qualify the approved profile-local attachment and speech/UI contracts; no new provider boundary is part of this unit.

1. Read current qualification acceptance criteria and existing production-app, backup/recovery and speech probes.
2. Add an owned mounted editor-to-Console import/apply/reopen/remove journey and full-profile backup/restore attachment coverage, preserving normal export/sync boundaries.
3. Exercise the real available image-edit/TTS services and local device paths with recorded inputs, outputs, timing, cancellation/tail and ownership evidence. Inspect user-provided ComfyUI/MOSS capabilities first; keep unavailable paths explicitly pending.
4. Measure known waveform onset/silence against actual presentation paint and device timestamps; document measurement scope and hardware loopback limits.
5. Run only relevant targeted integration/static/packaging/style checks; reproduce unrelated baseline failures separately.
6. Update user/architecture guidance and dated evidence, review this unit, and mark Done only for acceptance criteria actually proved.
Live CoreAudio qualification exposed an early finished acknowledgement (~24 ms before final scheduled DAC sample). Before completing this unit, add deterministic tests with/without portrait observation; fence natural completion on both native acknowledgement and the validated final output interval, retaining immediate scoped Stop/replacement. Keep timing extraction bounded and optional amplitude analysis out of audio completion authority. Re-run affected sink/speech races and real PCM/WAV calibration, then review the corrective diff. ADR-242 already requires completed device tails; clarify the measured implementation implication in its notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->


## Qualification checkpoint

Installed artwork journey, native full-profile restore/fresh open and original F9
control pass. Portable exports: 10 checks pass. Storage/authoring/ownership gate:
136 pass; final reviewed sink/speech gate: 226 pass. Real PCM/WAV/afplay probes:
three pass; exact waveform/terminal-flush scoring stays below 100 ms and verifies
completion after the scheduled tail. CoreAudio's early finished ACK exposed and
led to a scalar tail-fence correction independent of portrait analysis; ADR-242
and the live-verification lesson record it. See the linked dated evidence and
partial review for measurements, runtime hashes and baseline limits.

**Still In Progress:** no MOSS image request was submitted. The user clarified
MOSS image creation on 192.168.2.216:8188, but direct system_stats/object_info
requests still time out. The current ComfyUI adapter is packaged H3-only; the
reachable MOSS workflow/node/reference contract is required. Real TTS synthesis,
perceptual/final-syllable capture, live hands-free/interrupt/context-change coverage
also remains unqualified. Known-waveform/device and mocked ownership checks do not
complete those requirements. Implementation Notes and final completion remain
pending; this checkpoint is not a Done claim.
