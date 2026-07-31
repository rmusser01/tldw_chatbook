---
id: TASK-617.4
title: Add exact character TTS assignment mutation service
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-31 00:58'
updated_date: '2026-07-31 01:09'
labels:
  - tts
  - profiles
  - roleplay
dependencies:
  - TASK-617.2
  - TASK-617.3
  - TASK-763
  - TASK-951
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
Complete TTS Slice 3A by exposing exact source-aware character assignment mutations over the existing profile service and repository so Slice 3B can add visible assignment and assigned-profile speech without weakening lifecycle or capability authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A caller can set or replace one exact `CharacterRef` assignment using a caller-held `LoadedTTSProfile` and its repository generation.
- [ ] #2 A set or replace request validates the exact loaded profile revision against a fresh authoritative capability observation before repository mutation.
- [ ] #3 The repository's final transaction checks expected lifecycle generation, selected profile revision, and expected current assignment state before mutation.
- [ ] #4 Detach uses the caller-held assignment generation and exact assigned profile ID; it is idempotent only when already absent and refuses to remove a replacement.
- [ ] #5 Stale restore, profile edit, catalog movement, assignment races, missing authority, and malformed repository results fail closed with bounded errors and no partial mutation.
- [ ] #6 Deterministic service and repository tests cover success, replacement, detach, lifecycle races, capability races, and compare-and-set conflicts.
- [ ] #7 The task adds no assignment UI, speech resolver, automatic speech, Persona TTS, portability, Sync changes, or managed audio.cpp behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
Reason: ADR-037 already governs lifecycle generation profile revision expected-current-assignment compare-and-set semantics and the Slice 3B deferrals; this task implements that accepted boundary without a schema or ownership change.

1. Pin mandatory repository assignment expectations and transaction-boundary lifecycle races with failing tests.
2. Implement transactional generation profile-revision expected-current-assignment and exact-detach checks.
3. Pin profile-service capability ordering forwarded expectations stale-state handling and bounded failures with failing tests.
4. Implement minimal exact set/replace and detach service operations over existing domain values.
5. Update the developer guide run focused and broad verification complete TASK-617.4 request review rebase and open one PR.

Full plan: Docs/superpowers/plans/2026-07-31-tts-assignment-mutation-service.md
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked and concise implementation notes record the delivered behavior and any plan deviations.
- [ ] #2 Focused repository, service, lifecycle, capability, ownership, Console snapshot, and native audio.cpp regression tests pass.
- [ ] #3 Task-scoped Ruff, formatting, compile, typing, and git diff checks pass or exact unchanged baselines are documented.
- [ ] #4 ADR-037, the approved Slice 3A design, and the TTS developer guide remain current and linked.
- [ ] #5 Independent review finds no unresolved Critical, Important, or Minor issue before the PR is merged.
<!-- DOD:END -->
