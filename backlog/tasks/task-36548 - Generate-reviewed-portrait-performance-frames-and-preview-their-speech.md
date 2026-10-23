---
id: TASK-36548
title: Generate reviewed portrait performance frames and preview their speech
status: To Do
created_date: 2026-10-23 22:56
dependencies:
- TASK-36545
- TASK-36547
labels:
- personas
- image-generation
- tts
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare mouth and blink artwork from an existing character portrait through configured image editing, then test it with explicitly requested TTS before activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Generation uses one captured still reference from persisted artwork, existing reference-image capability/request validation and the chosen configured provider/model, with no text-only or provider fallback.
- [ ] #2 The batch action discloses six image-edit requests and the source destination; selected-slot regeneration submits one request and preserves other successful draft slots.
- [ ] #3 Sequential generation retains partial successes, rejects stale authority before new submissions and after results, cancels remaining work on draft changes and never retries uncertain paid outcomes automatically.
- [ ] #4 Generated images populate the import/review draft and undergo identical normalization, budget and alignment preview requirements; no generation result activates before Apply.
- [ ] #5 Explicit TTS sample playback uses the guarded preview owner, existing voice/destination resolution and exact scoped cancellation; it creates no message and cannot animate an unrelated Console portrait.
- [ ] #6 Targeted workflow tests cover animated-source frame extraction, absent/incompatible backends, partial failure, slot retry, source/profile switch, uncancellable late results, missing TTS, destination refusal and newer-audio protection.
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
