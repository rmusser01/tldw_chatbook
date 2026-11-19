---
id: TASK-36548
title: Generate reviewed portrait performance frames and preview their speech
status: Done
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
- Docs/superpowers/reviews/2026-10-23-task-36548-review.md
assignee:
- '@codex'
updated_date: 2026-10-24 03:12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare mouth and blink artwork from an existing character portrait through configured image editing, then test it with explicitly requested TTS before activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generation uses one captured still reference from persisted artwork, existing reference-image capability/request validation and the chosen configured provider/model, with no text-only or provider fallback.
- [x] #2 The batch action discloses six image-edit requests and the source destination; selected-slot regeneration submits one request and preserves other successful draft slots.
- [x] #3 Sequential generation retains partial successes, rejects stale authority before new submissions and after results, cancels remaining work on draft changes and never retries uncertain paid outcomes automatically.
- [x] #4 Generated images populate the import/review draft and undergo identical normalization, budget and alignment preview requirements; no generation result activates before Apply.
- [x] #5 Explicit TTS sample playback uses the guarded preview owner, existing voice/destination resolution and exact scoped cancellation; it creates no message and cannot animate an unrelated Console portrait.
- [x] #6 Targeted workflow tests cover animated-source frame extraction, absent/incompatible backends, partial failure, slot retry, source/profile switch, uncancellable late results, missing TTS, destination refusal and newer-audio protection.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Implement approved explicit reference-image editing and exact-owner speech preview through existing generation and guarded TTS boundaries.

1. Read existing image request/capability validation, still extraction, guarded selected-character TTS/destination ownership and generation fixtures. Add failing pure prompt/reference and actual mounted action tests.
2. Extend the existing screen-owned portrait controller in UI/Persona_Modules rather than growing PersonasScreen beyond its ratchet; keep the same disposable draft, source capture and parent message route, without adding a service/adapter.
3. Disclose the chosen backend/model, captured source destination and six explicit requests before a batch; selected regeneration admits one request. Extract one bounded still reference and use existing capability/build/run validation, sequentially, with no provider fallback or automatic retry.
4. Revalidate actual source/profile/editor before every submission and after results; retain successful slots on partial failures, fence uncancellable late results, and stop remaining requests on discard.
5. Add fixed-copy explicitly requested TTS using selected-character voice/destination resolution, guarded exact-owner cancellation and draft-only speech presentation. Cancellation/close/source switch drains the owned sample without messages or unrelated Console animation.
6. Verify pure/mounted generation and guarded speech cases, update guidance/architecture/plan, run relevant static/CSS checks, self-review and commit before the next unit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented disclosed sequential six-pose image editing and explicit selected-slot regeneration in the existing screen-owned portrait controller under ADR-242. One metadata-free frame-zero reference retains the captured original digest, including animated sources. Existing configured-backend/capability/build/run validation is preserved; no provider/text-only fallback or feature-level paid retry was added. Source/profile/editor and local authority are checked around every submission/result; partial successes stay in the same import draft, while stale/uncancellable results cannot stage or trigger remaining requests. Optional preview errors never suggest another paid request for a successful result.

Added an explicitly confirmed fixed-copy TTS sample through speak_guarded_utterance using the selected local CharacterRef, rechecked destination fingerprint and a draft-only presentation channel. No messages, Console speech binding or auto-speak preference are created. Stop/Cancel target the exact sample; Cancel waits for Stop acknowledgement, and retirement drains the owning task. Existing motion preferences apply. Missing/refused/changed TTS keeps imports/generation/Apply usable. Busy selectors remain stable and deferred focus stays within its draft epoch.

Verification: 49 new/affected generation and guarded-speech cases passed (radio-36548-verified); subsequent destination-change and combined import/generation/portrait checks passed 45 cases (radio-36548-final-requests). Five affected sample cases passed after selector locking (radio-36548-controls). A descriptor warning in the longer run was traced with lsof to the reused Personas fixture's legacy executor-thread SQLite handles. After native quiescent teardown, 29 new/affected mounted cases passed and lsof reported no fixture database handles (radio-36548-fd-fixed). Final keyboard focus check passed (radio-36548-focus). 44 CSS/token/build checks passed (radio-36548-css). Production/new-test Ruff, format checks and git diff --check pass. Style-template tests now use the established bootstrap-profile marker rather than bypassing recovery guards.

Review: Docs/superpowers/reviews/2026-10-23-task-36548-review.md. Updated media-generation architecture, user guide, implementation plan and concrete fixture-retirement lesson. Persona screen size was unchanged; orchestration extends the focused controller introduced by the import unit, without a service/adapter/dependency. No full suite, paid image editing, real TTS/device qualification, push or merge was performed. ADR required: yes; backlog/decisions/242-console-radio-portrait-performance.md applies.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
