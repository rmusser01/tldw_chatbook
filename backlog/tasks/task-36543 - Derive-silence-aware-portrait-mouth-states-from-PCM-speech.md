---
id: TASK-36543
title: Derive silence-aware portrait mouth states from PCM speech
status: To Do
created_date: 2026-10-23 22:55
dependencies:
- TASK-36509
labels:
- audio
- tts
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn real speech levels into stable closed, small and open mouth states that close during natural pauses, with bounded process-local presentation data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PCM16 energy uses 20 ms windows and preserves correct levels across chunk/sample boundaries, supported channel counts and sample rates without importing Textual or device libraries.
- [ ] #2 Digital zero forces closed; two raw windows below -45 dBFS close within 40 ms independently of smoothing; speech reopens at -42 dBFS.
- [ ] #3 Finite dBFS smoothing uses 40/100 ms attack/release and -24/-27 dBFS small/open hysteresis; gate closure resets the filter without infinity or NaN leakage.
- [ ] #4 Presentation snapshots retain one immutable owner identity, transport state, level, timing validity and monotonic freshness without audio bytes, persistent envelope history or unbounded reader backlog.
- [ ] #5 Deterministic tests cover loud-to-quiet nonzero background, exact silence, rejected/malformed PCM, odd chunks, stereo anti-phase input, finite reopen and stale owner generations.
- [ ] #6 Applicable targeted tests and static/import-boundary checks pass without installing a dependency or adding user tuning controls.
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
