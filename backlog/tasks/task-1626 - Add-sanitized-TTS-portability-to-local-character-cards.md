---
id: TASK-1626
title: Add sanitized TTS portability to local character cards
status: Done
assignee:
  - '@codex'
created_date: '2026-07-31 19:33'
updated_date: '2026-07-31 22:17'
labels:
  - tts
  - profiles
  - characters
  - portability
dependencies:
  - TASK-617.5
  - TASK-951
references:
  - >-
    Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md
  - backlog/decisions/028-character-tts-generation-profile-ownership.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users explicitly carry a sanitized local TTS generation profile with an existing local character card while keeping ordinary exports private, imported attachments untrusted, and character/profile stores safely independent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ordinary JSON and PNG character-card exports contain no TTS payload; explicit Include TTS profile and standalone profile export emit the exact sanitized version-1 payload without mutating stored cards and preserve unrelated extensions.
- [x] #2 Explicit export fails safely when the reserved Chatbook TTS namespace is malformed or already populated, and exported data excludes authority, origins, credentials, paths, health, timestamps, revisions, message text, and other local-only state.
- [x] #3 Character-card import bounds, validates, and strips the TTS attachment before character persistence; unknown, malformed, oversized, too-deep, unsupported-provider, or unsupported audio.cpp payloads warn and never block the character import or persist untrusted TTS data.
- [x] #4 Valid audio.cpp attachments use the approved UUID/name/generation-tuple collision matrix with explicit reuse-or-copy decisions, collision-safe names and UUIDs, and no silent mutation of existing profiles.
- [x] #5 Character persistence returns a structured created-versus-reused outcome; new characters receive an assignment only for a currently available profile, while reused characters require explicit confirmation and retain their existing assignment when the imported profile is unavailable.
- [x] #6 Cancellation and profile-transaction failure leave no partial profile or assignment; a newly imported character remains unassigned and a reused character retains its prior assignment, with partial success reported for repair.
- [x] #7 The Personas local-card UI exposes explicit inclusion, collision, and existing-character confirmations; standalone profile import, server card persistence/synchronization, and managed audio.cpp lifecycle remain out of scope.
- [x] #8 Focused portability, security, compensation, UI, and ordinary import/export regression tests pass.
- [x] #9 Portability logs, events, notifications, exports, and metrics exclude message text, character authority, credentials, provider origins, and full filesystem paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend ADR-028 before implementation with the sanitized portability, hostile-input, collision-authority, and cross-store compensation decision.
2. Add a strict version-1 sanitized profile codec and hostile-input tests.
3. Add a structured local character import outcome that strips the reserved attachment before persistence while preserving the legacy ID-returning wrapper and sanitizing touched import logging.
4. Add generation-fenced collision reads and atomic profile-plus-assignment persistence, then expose typed inspect/commit operations through the profile service.
5. Extend existing JSON/PNG exports with an explicit transient payload and add standalone profile export using the same codec; defaults stay unchanged.
6. Evaluate structural validity and current availability before character persistence, then add Personas inclusion/collision/existing-character prompts, commit-time revalidation, compensation, privacy tests, and focused UAT.
7. Run focused regression/static checks, self-review privacy boundaries, and close the task.

ADR required: yes
ADR path: `backlog/decisions/028-character-tts-generation-profile-ownership.md`
Reason: Slice 4 activates the previously deferred sanitized portability boundary and defines hostile import, collision, local ownership, and cross-database compensation behavior.

Detailed plan: `Docs/superpowers/plans/2026-07-31-character-tts-portability.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added one strict, deterministic version-1 audio.cpp portability codec shared by character-card and standalone profile export. Ordinary JSON/PNG exports remain TTS-free, and opt-in export works on transient copies only.
- Added typed character import inspection/outcomes that validate and strip the reserved extension before persistence, plus explicit Personas inclusion, collision, reused-character confirmation, unavailable-profile repair, and partial-success behavior.
- Added generation-fenced collision reads, atomic profile-plus-assignment creation, and exact selected-profile snapshot comparison so delete/recreate ABA races fail as conflicts. Portability-only repository capabilities are checked lazily, preserving the existing profile-service construction contract.
- Hardened card parser and export-path logging so hostile card values, message text, credentials, origins, authorities, and full paths are not surfaced. No dependency, managed audio.cpp lifecycle, server sync, or standalone import surface was added.
- Updated ADR-028 and added focused codec, JSON/PNG, repository/service, mounted UI, privacy, compatibility, and first-time roleplay complete-WAV UAT coverage.
- Verification: 991 focused and adjacent regression tests passed in 406.74 seconds; Ruff passed all changed Python files; mypy passed `profile_portability.py` and `profile_service.py`; `git diff --check` passed; independent re-review reported no remaining findings.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked.
- [x] #2 Focused unit, integration, and UI tests pass.
- [x] #3 Changed Python files pass applicable static analysis and git diff --check.
- [x] #4 ADR-028 and relevant documentation are updated.
- [x] #5 Implementation Notes summarize approach, trade-offs, files, and verification.
- [x] #6 Self-review confirms no privacy, security, license, or ordinary card import/export regression.
<!-- DOD:END -->
