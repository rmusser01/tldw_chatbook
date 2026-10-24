---
id: TASK-36545
title: Bind portrait speech observation to Console and preview owners
status: Done
created_date: 2026-10-23 22:56
dependencies:
- TASK-36544
labels:
- console
- tts
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
- Docs/superpowers/reviews/2026-10-23-console-radio-portraits-implementation-review.md
- Docs/superpowers/reviews/2026-10-23-task-36545-review.md
updated_date: 2026-10-24 01:29
assignee:
- '@codex'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose consistent speech presentation for manual Speak, automatic replies, hands-free sentences and guarded editor samples without weakening existing speech admission or stop behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Manual and automatic Speak publish only for their existing validated message/request/session owner; stale session/profile/character signals cannot animate a new owner.
- [x] #2 Hands-free receives optional start/progress/terminal observation with its exact reply/session/character/sequencer token while retaining current voice selection, cooldown bypass, quiet failures and exactly-once completion; the next sentence or listening starts only after backend-confirmed natural drain or settled interruption.
- [x] #3 Guarded app-owned editor speech can expose its own presentation owner and uses selected-character resolution plus existing destination checks without fabricating a conversation message.
- [x] #4 Preview close/cancel/supersession stops only its exact request; late cleanup and terminal acknowledgements cannot stop or clear newer speech.
- [x] #5 Confirmed pause/resume is presentation transport state and does not introduce unsupported values into existing speech-admission state machines or add new player capabilities.
- [x] #6 Targeted ownership tests cover older-message replay, unknown/cross-character provenance, delayed final-device-block completion, sentence handoff, stale validators, absent observers and cancellation during device drain; late completion cannot clear newer speech or fire completion twice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Extend existing speech admission/request ownership with optional disposable presentation, retaining provider selection and native completion boundaries.

1. Read message, utterance, guarded preview, hands-free and player lifecycle callers plus existing ownership tests.
2. Add failing tests through actual public forwarding entries for exact owner, exactly-once completion, stale/replacement cleanup, status-only fallback and native-tail handoff.
3. Thread optional presentation through existing events/generation/sink/file paths; project confirmed transport without changing admission states or current hands-free voice resolution.
4. Capture/revalidate Console scope/session/character/message or sequencer ownership, expose the current presentation read seam, and stop guarded preview cancellation by exact lifecycle only.
5. Run targeted new and affected speech/Console/Buddy tests, lint delta and self-review; document evidence and commit before the next atomic unit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Bound optional SpeechPresentation to existing request lifecycles/events and forwarded through all generation, sentence, sink, artifact and app auto-play edges. Console captures profile/session/message ownership and exposes one validated current channel; hands-free captures reply/character/sequencer identity before worker dispatch and revalidates without changing voice resolution, quiet failures or cooldown bypass. Guarded previews retain exact lifecycle Stop. Confirmed legacy pause/resume is status-only, outside admission states. ADR-242 applies. Atomic review found and corrected cancellation status, lifecycle/channel pairing and duplicate sentence counters. Verification: 172 targeted speech cases passed; 13 combined owner/mounted/stale-worker cases passed. New tests Ruff/format clean, zero introduced legacy lint findings, AST/diff checks pass. Files, corrections, harness context and limits: Docs/superpowers/reviews/2026-10-23-task-36545-review.md; speech architecture updated. Physical latency and visible playback remain for integration.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
