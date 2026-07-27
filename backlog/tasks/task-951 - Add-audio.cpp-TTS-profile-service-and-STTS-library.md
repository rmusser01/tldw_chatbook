---
id: TASK-951
title: Add audio.cpp TTS profile service and STTS library
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-27 17:15'
updated_date: '2026-07-27 17:23'
labels:
  - tts
  - profiles
  - stts
dependencies:
  - TASK-763
references:
  - TASK-763
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
  - backlog/decisions/028-character-tts-generation-profile-ownership.md
documentation:
  - >-
    Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the reusable profile-management layer for successful native audio.cpp Playground generations so users can save exact selections and manage them safely before character assignment is introduced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A successful native audio.cpp Playground result can be saved as a named reusable profile and managed through bounded list search edit duplicate preview repair and protected-delete flows.
- [ ] #2 Native audio.cpp artifacts carry immutable text-free requested-selection and configuration-revision provenance while legacy artifacts expose no profile provenance or save action and retain existing behavior.
- [ ] #3 Structured audio.cpp voice discovery preserves complete model-missing and unverified authority before compatibility projection; caller cancellation is neither cached nor published.
- [ ] #4 A 50-row profile page renders repository data before bounded capability enrichment using at most four voice observations one ten-second aggregate deadline one moving-catalog retry and no server-default voice query.
- [ ] #5 Late UI repository catalog configuration and unmount results cannot publish and search coalesces to one active page pipeline plus one latest pending query.
- [ ] #6 Loaded-profile update delete duplicate and availability flows reject stale pre-restore repository generations even when a replacement store reuses the same UUID and revision.
- [ ] #7 Profile preview survives STTS remount with exact persisted model and voice values and never substitutes a first model or server-default voice; unavailable generation is blocked and unverified generation requires a warned explicit attempt.
- [ ] #8 Save-from-success uses the admitted immutable snapshot without a redundant catalog or voice request and has one configuration-revision decision point before repository work without a cross-resource transaction.
- [ ] #9 Rename-only edits are established by service comparison against the exact stored revision while generation edits and duplicates require fresh authoritative capability validation.
- [ ] #10 Executable profiles accept only audio_cpp WAV speed 1.0 and empty options and reject every other provider including unreviewed native descriptors.
- [ ] #11 One lazy app-owned profile service reuses the app-owned repository TTS service admission coordinator Playground generation player artifact and cleanup lifecycles.
- [ ] #12 Profile-store failure disables only profile consumers while ordinary Playground settings and legacy speech remain usable.
- [ ] #13 Deterministic domain service adapter and Textual tests cover success conflict cancellation stale-generation restore timeout catalog movement remount no-fallback legacy regression and persistence across restart.
- [ ] #14 The slice adds no character identity assignment UI request resolver roleplay routing portability standalone import or export managed audio.cpp process behavior or legacy-profile execution.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Commit the reviewed task scope design amendments and supported-interpreter baseline.
2. Add adapter-boundary structured audio.cpp voice status and status-aware caching with TDD.
3. Add bounded revision-coherent capability snapshots exact native admission and text-free requested-selection provenance with TDD.
4. Fence profile-derived repository mutations by the loaded lifecycle generation with restore-race coverage.
5. Add the native-only profile service for save list availability edit duplicate count delete and exact preview presets.
6. Bind one lazy profile service to the existing app-owned repository and TTS service without a new lifecycle owner.
7. Add the focused bounded STTS profile library editor coalesced search and stale-result suppression.
8. Reuse the existing Playground for one-shot exact preview and native-only save-result-as-profile.
9. Update guides run focused and broad verification perform isolated external-server UAT request independent review rebase and record final task evidence.

Full plan: Docs/superpowers/plans/2026-07-27-audio-cpp-tts-profile-service-stts-library.md
ADR required: yes
ADR paths: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md and backlog/decisions/028-character-tts-generation-profile-ownership.md
Reason: Existing ADRs already govern exact native provenance structured capability observation app-owned profile service and lifecycle-generation mutation admission; no new ADR schema migration store dependency or process-runtime decision is needed.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked and concise implementation notes document the delivered behavior and deviations.
- [ ] #2 Focused unit integration concurrency and Textual tests plus relevant legacy TTS regressions pass.
- [ ] #3 Task-scoped Ruff format compileall focused typing and git diff checks pass.
- [ ] #4 ADR-023 ADR-028 the approved design and relevant user or developer documentation remain current.
- [ ] #5 Independent code and scope review finds no unresolved Critical Important or Minor issue.
- [ ] #6 The task is rebased on current dev and marked Done only after all required evidence is recorded.
<!-- DOD:END -->
