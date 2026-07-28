---
id: TASK-951
title: Add audio.cpp TTS profile service and STTS library
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 17:15'
updated_date: '2026-07-28 14:22'
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
- [x] #1 A successful native audio.cpp Playground result can be saved as a named reusable profile and managed through bounded list search edit duplicate preview repair and protected-delete flows.
- [x] #2 Native audio.cpp artifacts carry immutable text-free requested-selection and configuration-revision provenance while legacy artifacts expose no profile provenance or save action and retain existing behavior.
- [x] #3 Structured audio.cpp voice discovery preserves complete model-missing and unverified authority before compatibility projection; caller cancellation is neither cached nor published.
- [x] #4 A 50-row profile page renders repository data before bounded capability enrichment using at most four voice observations one ten-second aggregate deadline one moving-catalog retry and no server-default voice query.
- [x] #5 Late UI repository catalog configuration and unmount results cannot publish and search coalesces to one active page pipeline plus one latest pending query.
- [x] #6 Loaded-profile update delete duplicate and availability flows reject stale pre-restore repository generations even when a replacement store reuses the same UUID and revision.
- [x] #7 Profile preview survives STTS remount with exact persisted model and voice values and never substitutes a first model or server-default voice; unavailable generation is blocked and unverified generation requires a warned explicit attempt.
- [x] #8 Save-from-success uses the admitted immutable snapshot without a redundant catalog or voice request and has one configuration-revision decision point before repository work without a cross-resource transaction.
- [x] #9 Rename-only edits are established by service comparison against the exact stored revision while generation edits and duplicates require fresh authoritative capability validation.
- [x] #10 Executable profiles accept only audio_cpp WAV speed 1.0 and empty options and reject every other provider including unreviewed native descriptors.
- [x] #11 One lazy app-owned profile service reuses the app-owned repository TTS service admission coordinator Playground generation player artifact and cleanup lifecycles.
- [x] #12 Profile-store failure disables only profile consumers while ordinary Playground settings and legacy speech remain usable.
- [x] #13 Deterministic domain service adapter and Textual tests cover success conflict cancellation stale-generation restore timeout catalog movement remount no-fallback legacy regression and persistence across restart.
- [x] #14 The slice adds no character identity assignment UI request resolver roleplay routing portability standalone import or export managed audio.cpp process behavior or legacy-profile execution.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Delivered the native audio.cpp reusable profile service and bounded STTS profile library: structured voice authority, immutable text-free admitted-selection provenance, lifecycle-generation mutation fences, one lazy app-owned service, bounded capability enrichment and coalesced search, safe edit/duplicate/delete flows, exact remount-safe preview, and save-from-success while preserving legacy behavior and Playground failure isolation.

Verification after the final rebase: the focused Task 951 gate passed 1132 tests with 3 warnings. The broader TTS/STTS gate passed 1987 tests with 14 expected skips; its 11 failures were reproduced exactly on untouched origin/dev ef09085b and are pre-existing backup/preferences baseline failures. Task-scoped Ruff, Ruff format, fatal-error lint, compileall, mypy on 7 production modules, git diff, scope, and privacy checks passed. Independent specification and quality reviews approved the final diff with no unresolved Critical, Important, or Minor finding.

External-server UAT against user-started audio.cpp at 127.0.0.1:8080 with supertonic-3 verified first-run configuration, one-model readiness, complete-WAV generation and playback, save/search/refresh, M1-to-M2 edit at revision 2, duplicate, exact M2 preview with a second complete WAV, protected deletion, and persistence plus exact preview rehydration across a real app restart. The first WAV was 603056-byte mono PCM16 44.1 kHz audio lasting 6.837 seconds; the second was 383038 bytes lasting 4.342 seconds. The profile database remained mode 0600 and stored no submitted text or endpoint/credential data.

ADR check: ADR-023 and ADR-028 remain applicable and current; no new ADR was required. The branch is rebased on origin/dev 3f297856d. No deviation expanded scope: character assignment, roleplay routing, portability/import/export, managed audio.cpp process behavior, and legacy-profile execution remain excluded.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked and concise implementation notes document the delivered behavior and deviations.
- [x] #2 Focused unit integration concurrency and Textual tests plus relevant legacy TTS regressions pass.
- [x] #3 Task-scoped Ruff format compileall focused typing and git diff checks pass.
- [x] #4 ADR-023 ADR-028 the approved design and relevant user or developer documentation remain current.
- [x] #5 Independent code and scope review finds no unresolved Critical Important or Minor issue.
- [x] #6 The task is rebased on current dev and marked Done only after all required evidence is recorded.
<!-- DOD:END -->
