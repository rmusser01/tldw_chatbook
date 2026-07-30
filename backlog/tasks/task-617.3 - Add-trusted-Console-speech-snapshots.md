---
id: TASK-617.3
title: Add trusted Console speech snapshots
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-30 19:28'
updated_date: '2026-07-30 19:32'
labels:
  - tts
  - console
  - security
dependencies: []
references:
  - >-
    backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
documentation:
  - >-
    Docs/superpowers/specs/2026-07-28-tts-character-identity-persona-separation-design.md
parent_task_id: TASK-617
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bind Console Speak to an immutable app-issued message snapshot and reject stale or mismatched requests before normalization, cooldown, profile lookup, or provider work while preserving existing global TTS selection and playback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Console store issues a frozen snapshot containing native and optional persisted identity, exact visible content, selected variant and monotonic speech revision, persisted row version, role, completion state, assistant kind, and a CharacterRef only when authority is complete.
- [ ] #2 Speech revisions advance on every content, status, variant-addition, variant-selection, and variant-content mutation so edit-then-revert remains stale; revisions are never serialized across process restart.
- [ ] #3 Admission rejects deleted, moved, incomplete, non-assistant, stale-variant, stale-revision, stale-persisted-version, and mismatched-authorship snapshots before normalization, cooldown, or provider work.
- [ ] #4 An unchanged valid Console snapshot follows the existing global TTS provider, model, voice, format, speed, complete-WAV playback, and error behavior without assigned-profile resolution.
- [ ] #5 Non-Console callers retain a distinct explicit trusted global-speech request path and do not construct Console snapshots.
- [ ] #6 Diagnostics use bounded outcome codes and never log snapshot text, authority, credentials, origins, routing IDs, or paths.
- [ ] #7 Deterministic store, event-handler, UI, and native audio.cpp regression tests cover valid and rejected requests including controlled post-click mutations.
- [ ] #8 The task adds no profile assignment mutation, assignment UI, character-specific resolver, automatic speech, Persona voice inheritance, portability, or managed audio.cpp behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
Reason: ADR-037 already governs immutable Console speech snapshots, authorship validation, pre-cooldown rejection, privacy, and deferral of assigned-profile resolution; no new ADR is required.

1. Define the immutable privacy-safe snapshot and bounded rejection contract with TDD.
2. Add Console-owned process-local speech revisions, persisted-version reads, snapshot issuance, and exact validation with TDD.
3. Add the dedicated snapshot event and validate it before any cooldown mutation while preserving the explicit global request path.
4. Wire Console and application routing and align Speak availability to completed assistant messages.
5. Prove unchanged native audio.cpp and legacy/global synthesis and document the boundary.
6. Run focused and broader verification, independent review, and Backlog closeout.

Full plan: Docs/superpowers/plans/2026-07-30-trusted-console-speech-snapshots.md
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked and concise Implementation Notes record the delivered behavior and deviations.
- [ ] #2 Focused store, event-handler, Console UI, native audio.cpp, complete-WAV, and legacy TTS tests pass.
- [ ] #3 Task-scoped Ruff, formatting, compile, typing, and git diff checks pass or exact unchanged baselines are documented.
- [ ] #4 ADR-037 and the approved Slice 3A design remain current and are linked from the task and plan.
- [ ] #5 Independent review finds no unresolved Critical, Important, or Minor issue before merge.
<!-- DOD:END -->
