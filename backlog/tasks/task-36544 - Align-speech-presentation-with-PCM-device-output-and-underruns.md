---
id: TASK-36544
title: Align speech presentation with PCM device output and underruns
status: Done
created_date: 2026-10-23 22:55
dependencies:
- TASK-36543
labels:
- audio
- tts
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
- Docs/superpowers/reviews/2026-10-23-console-radio-portraits-implementation-review.md
- Docs/superpowers/reviews/2026-10-23-task-36544-review.md
updated_date: 2026-10-24 01:14
assignee:
- '@codex'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep portrait speech levels aligned with the audio actually scheduled to the output device, including buffering, inserted silence and supported WAV artifacts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Optional presentation observation preserves public sink call compatibility and audio samples when disabled or when observation fails; backend-confirmed natural completion applies equally to observed and unobserved callers.
- [x] #2 Source-frame consumption and device-output positions remain distinct; validated output-DAC timing maps both to monotonic presentation time, including partial-block zero padding and resumed audio after underruns.
- [x] #3 Rejected or retried feed chunks never advance envelope position; device-scheduled metadata survives until its output time and remains bounded by admitted audio plus the one-second tail reserve.
- [x] #4 Slow or hidden readers cannot block audio, drop samples or accumulate presentation events; the realtime callback performs no image, database, UI, provider or blocking observer work.
- [x] #5 Eligible PCM16 WAV artifacts reuse the existing sink; unqualified external timing retains working audio with status-only presentation, preserving one-voice ownership and fallback without double playback.
- [x] #6 Fake-device tests prove delayed DAC output, final-block device drain with and without observation or valid DAC timing, full/partial underruns, backpressure, stop/replacement, invalid timestamps, observer failure and bounded cleanup; affected sink, pump, format and spoken-feedback regression checks pass.
- [x] #7 Natural completion retains the live sink, abortable stream and speech ownership until backend output-finished acknowledgement; source-queue exhaustion cannot publish drain or release pump completion, explicit Stop/replacement still meets the two-block silence contract, late/duplicate acknowledgements cannot override interruption, and a missing acknowledgement reaches bounded failure/cleanup.
- [x] #8 Qualified presentation updates continuously on the existing notification thread at a 20 ms sampling interval without UI reads, direct test calls to the publication helper or per-block publication-job accumulation; loud/pause/loud and pending-tail tests prove fresh observations, expired mapped output loses timing validity, and terminal cleanup stops publication.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Implement the revised output-timed observation and backend-confirmed completion contract on the existing PCM sink; no new player, worker framework or dependency.

1. Read the installed sink/pump/factory callers and eligible raw/WAV routing. Existing sink/import controls passed in the prior atomic unit; preserve them while deliberately correcting natural drain semantics.
2. Add failing fake-device cases for a delayed final block, native finish acknowledgement, Stop/replacement and periodic output-timed observations; extend the internal factory seam and all Audio fakes together.
3. Keep natural drain nonterminal until sounddevice CallbackStop/finished_callback acknowledgement, retain abortable owner/stream, preserve immediate Stop and bounded watchdog, and reject late/duplicate/unexpected completion.
4. Prepare/commit envelope analysis only for accepted PCM off the realtime callback; map source ranges and device silence to validated DAC/monotonic intervals. Publish coalesced latest snapshots on the existing notification thread at a nominal 20 ms cadence; enforce queued-audio plus one-second-tail bounds and stale/invalid timing fallback.
5. Route eligible complete PCM16 WAV artifacts through the same sink without a second player or duplicate fallback/completion. Exercise delayed DAC, partial/full underrun, odd chunks, rejected feed, blocked readers/listeners, backend failure and file success/fallback.
6. Run affected sink/pump/format/spoken-feedback tests and static/import checks, update speech architecture notes, self-review/correct the atomic change, record evidence and commit before the next unit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented optional bounded output-timed PCM observation on the existing notification thread and corrected natural drain to await native finished acknowledgement, including unobserved callers. Reused existing raw/PCM16-WAV routing with optional presentation forwarding. Stop/failure still abort; retirement epochs reject late/in-flight output, malformed frames fail closed, and coalesced underruns preserve bounded handoff. ADR-242 applies. Reviewed source, factories, pump and callers; no blocker remains. Verification: 122 targeted sink/pump/presentation/format/spoken-feedback cases plus 3 lazy-import cases passed; new tests Ruff/format clean, zero introduced legacy lint findings, AST and diff checks pass. Physical audio latency and device-clock qualification remain for integration. Files and corrections/evidence: Docs/superpowers/reviews/2026-10-23-task-36544-review.md; architecture and incident-backed lesson updated.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
