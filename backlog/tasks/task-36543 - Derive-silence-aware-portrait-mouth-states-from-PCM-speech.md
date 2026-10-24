---
id: TASK-36543
title: Derive silence-aware portrait mouth states from PCM speech
status: Done
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
- Docs/superpowers/reviews/2026-10-23-task-36543-review.md
assignee:
- '@codex'
updated_date: 2026-10-24 00:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn real speech levels into stable closed, small and open mouth states that close during natural pauses, with bounded process-local presentation data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PCM16 energy uses 20 ms windows and preserves correct levels across chunk/sample boundaries, supported channel counts and sample rates without importing Textual or device libraries.
- [x] #2 Digital zero forces closed; two raw windows below -45 dBFS close within 40 ms independently of smoothing; speech reopens at -42 dBFS.
- [x] #3 Finite dBFS smoothing uses 40/100 ms attack/release and -24/-27 dBFS small/open hysteresis; gate closure resets the filter without infinity or NaN leakage.
- [x] #4 Presentation snapshots retain one immutable owner identity, transport state, level, timing validity and monotonic freshness without audio bytes, persistent envelope history or unbounded reader backlog.
- [x] #5 Deterministic tests cover loud-to-quiet nonzero background, exact silence, rejected/malformed PCM, odd chunks, stereo anti-phase input, finite reopen and stale owner generations.
- [x] #6 Applicable targeted tests and static/import-boundary checks pass without installing a dependency or adding user tuning controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Implement ADR-242's ephemeral, bounded speech presentation contract without new dependencies or a second speech coordinator.

1. Add deterministic failing PCM/filter/snapshot tests in Tests/Audio/test_speech_presentation.py, using injected time and exact known PCM values.
2. Verify the missing/wrong behavior red, then implement stdlib-only Audio/speech_presentation.py with the shared plan's exact public contracts.
3. Preserve incomplete PCM frames/windows across accepted chunks, use energy across channels rather than phase-cancelling averaging, and flush a bounded partial final window while rejecting malformed final samples.
4. Implement raw silence gating before finite dBFS smoothing, exact reopen/hysteresis boundaries, zero/invalid reset, immutable owner-fenced latest snapshots and >100 ms freshness fallback without altering transport terminal state.
5. Run targeted deterministic and existing audio-boundary controls, Ruff/formatter and fresh-process import checks. Verify no Textual, sounddevice or TTS import and no clock sleeps/device/provider side effects.
6. Self-review the atomic change and its consumer contracts, correct findings, record evidence/notes and commit before the next atomic unit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented stdlib-only Audio/speech_presentation.py: source-frame PCM16 RMS windows, split/interleaved sample carry, rational boundaries and partial-final flush; raw two-window silence gate, digital-zero reset and finite 40/100 ms smoothing with -24/-27 dBFS hysteresis. Added immutable owner/snapshot values and one-latest-value observation with read-only exact owner identity, generation/timestamp fencing, confirmed pause/terminal protection and >100 ms stale fallback. Qualified silence empties the meter without falsely losing timing. No device/UI/provider import, new dependency, thread or tuning UI.

Tests first failed with the missing module, then explicit failing owner/transport/silence and fractional-window regressions drove corrections. Final targeted PCM/presentation/import/sink/pump run: 90 passed, including 38 new contract cases; no hardware or sleeps. Ruff/formatter, AST, fresh-process import boundary and diff checks pass. Updated Docs/Architecture/speech.md, the shared plan contract and the recorded fractional-window lesson. Self-review at Docs/superpowers/reviews/2026-10-23-task-36543-review.md has no unresolved blocking finding.

ADR-242 applies; no new architectural decision or scope expansion. Real device timing/publication, hardware latency and Console rendering remain subsequent integration obligations. Earlier Actor Pack golden-byte baseline failures are unchanged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
