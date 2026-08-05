---
id: TASK-617.5
title: Add character TTS assignment UI and roleplay speech runtime
status: Done
assignee:
  - '@codex'
created_date: '2026-07-31 05:11'
updated_date: '2026-07-31 07:58'
labels:
  - tts
  - profiles
  - roleplay
dependencies:
  - TASK-617.1
  - TASK-617.2
  - TASK-617.3
  - TASK-617.4
  - TASK-710
  - TASK-763
  - TASK-951
references:
  - backlog/decisions/028-character-tts-generation-profile-ownership.md
  - >-
    backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
documentation:
  - >-
    Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md
  - >-
    Docs/superpowers/specs/2026-07-28-tts-character-identity-persona-separation-design.md
parent_task_id: TASK-617
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver TTS Slice 3B so a user can visibly assign an existing audio.cpp generation profile to one exact local or authority-scoped server character and have manual Console Speak use that assignment through the trusted speech snapshot path. Broken assignment state must remain visible and fail closed without silently falling back.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A character editor exposes one **Voice & Speech** section showing global-default or the exact current assignment, profile availability, and actions to assign, replace, repair, preview, create, edit, and detach using the existing profile service.
- [x] #2 Manual Console **Speak** resolves a trusted completed character-authored message to its exact `CharacterRef` and uses the immutable assigned profile revision; an unassigned character continues through current global preferences.
- [x] #3 Invalid or stale speech snapshots, missing authority, unavailable assigned profiles, profile-store failures, and mutation conflicts fail closed with bounded actionable errors and no silent provider, model, voice, global, or legacy fallback.
- [x] #4 A user can explicitly choose a one-message global override after assigned-profile resolution fails; the override does not mutate assignment state.
- [x] #5 Assignments survive the current character soft-delete and restore lifecycle. Because Personas exposes no permanent character-delete operation, this PR adds no speculative cleanup path; cleanup never runs solely because a target is soft-deleted, temporarily unavailable, or unverified.
- [x] #6 Textual workers keep repository, capability, and synthesis work off the event loop; stale or unmounted UI work cannot publish.
- [x] #7 Deterministic service, Console, and Textual tests cover assignment, replacement, detach, assigned and unassigned speech, fail-closed recovery, explicit override, authority separation, and soft-delete lifecycle preservation.
- [x] #8 The task adds no automatic speech, managed audio.cpp process behavior, Persona TTS inheritance, Sync contract changes, or character-card portability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: `backlog/decisions/028-character-tts-generation-profile-ownership.md` and `backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`

Reason: The existing ADRs already define exact `CharacterRef` ownership, trusted message-authorship admission, native audio.cpp profile resolution, fail-closed behavior, and the Slice 3B boundary. This task implements those decisions without changing storage, ownership, provider, or service contracts beyond the already-approved extension points.

Detailed plan: `Docs/superpowers/plans/2026-07-31-tts-character-assignment-runtime.md`

1. Add a service-owned exact assignment read and a pure character-speech resolver, with tests for authority separation, assigned/unassigned selection, immutable exact requests, invalid joined state, and profile-store failures.
2. Route trusted Console Speak requests through that resolver before cooldown admission, reuse the existing complete-audio artifact path for exact audio.cpp synthesis, and add a bounded one-message global-override recovery event.
3. Add compact Voice & Speech controls to the existing local editor and local/server character card, using screen-owned workers and freshness fences for identity, profile reads, availability, assignment mutations, preview, edit, and detach.
4. Reuse the existing Speech Playground/profile editor surfaces for create, preview, and repair actions without adding a second profile-management implementation.
5. Verify soft-delete preservation, runtime fail-closed behavior, Textual stale-result suppression, existing global/legacy behavior, and the focused cumulative regression suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the service-owned exact assignment read, fail-closed character request resolver, trusted Console Speak admission, and a single-use explicit global override without fallback or assignment mutation.
- Added local/server character Voice & Speech controls using exact authority, repository-generation, capability-revision, selection, and mounted-state fences. Local authority is re-read immediately before publication, preview, and mutation.
- Reused the existing Speech profile editor and Playground. Applying a profile to an already-mounted Playground now retires prior handler audio, stops playback, clears transport state, and fences late generation completion.
- Preserved assignments across soft delete/restore. No permanent-delete cleanup, managed audio.cpp process behavior, automatic speech, Persona inheritance, Sync change, or card-portability work was added.
- ADR check: ADR-028 and ADR-037 already govern this implementation; no new ADR was required.
- Verification: focused resolver/navigation/generation tests passed (69); focused character-TTS Personas tests passed (13); static, compile, lint, and scoped formatting gates passed. The cumulative suite recorded 1,077 passes plus one unchanged Personas import-copy baseline failure. The broad TTS/Console suite recorded 2,188 passes and 14 skips plus 11 unchanged backup/preferences baseline failures.
<!-- SECTION:NOTES:END -->
