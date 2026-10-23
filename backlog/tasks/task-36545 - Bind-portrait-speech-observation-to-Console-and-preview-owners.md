---
id: TASK-36545
title: Bind portrait speech observation to Console and preview owners
status: To Do
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose consistent speech presentation for manual Speak, automatic replies, hands-free sentences and guarded editor samples without weakening existing speech admission or stop behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Manual and automatic Speak publish only for their existing validated message/request/session owner; stale session/profile/character signals cannot animate a new owner.
- [ ] #2 Hands-free receives optional start/progress/terminal observation with its exact reply/session/character/sequencer token while retaining current voice selection, cooldown bypass, quiet failures and exactly-once completion.
- [ ] #3 Guarded app-owned editor speech can expose its own presentation owner and uses selected-character resolution plus existing destination checks without fabricating a conversation message.
- [ ] #4 Preview close/cancel/supersession stops only its exact request; late cleanup and terminal acknowledgements cannot stop or clear newer speech.
- [ ] #5 Confirmed pause/resume is presentation transport state and does not introduce unsupported values into existing speech-admission state machines or add new player capabilities.
- [ ] #6 Targeted ownership tests cover older-message replay, unknown/cross-character provenance, sentence handoff, stale validators, absent observers and cancellation without changing existing public-call behavior.
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
