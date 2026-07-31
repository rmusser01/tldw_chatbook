---
id: TASK-617.3
title: Add trusted Console speech snapshots
status: Done
assignee:
  - '@codex'
created_date: '2026-07-30 19:28'
updated_date: '2026-07-30 20:27'
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
- [x] #1 The Console store issues a frozen snapshot containing native and optional persisted identity, exact visible content, selected variant and monotonic speech revision, persisted row version, role, completion state, assistant kind, and a CharacterRef only when authority is complete.
- [x] #2 Speech revisions advance on every content, status, variant-addition, variant-selection, and variant-content mutation so edit-then-revert remains stale; revisions are never serialized across process restart.
- [x] #3 Admission rejects deleted, moved, incomplete, non-assistant, stale-variant, stale-revision, stale-persisted-version, and mismatched-authorship snapshots before normalization, cooldown, or provider work.
- [x] #4 An unchanged valid Console snapshot follows the existing global TTS provider, model, voice, format, speed, complete-WAV playback, and error behavior without assigned-profile resolution.
- [x] #5 Non-Console callers retain a distinct explicit trusted global-speech request path and do not construct Console snapshots.
- [x] #6 Diagnostics use bounded outcome codes and never log snapshot text, authority, credentials, origins, routing IDs, or paths.
- [x] #7 Deterministic store, event-handler, UI, and native audio.cpp regression tests cover valid and rejected requests including controlled post-click mutations.
- [x] #8 The task adds no profile assignment mutation, assignment UI, character-specific resolver, automatic speech, Persona voice inheritance, portability, or managed audio.cpp behavior.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented immutable trusted Console speech snapshots with store-owned issuance and validation across active session/path, exact selected content and variant, monotonic process-local revision, persisted row version, completed-assistant state, and assistant authorship. Added a dedicated snapshot event that validates before cooldown, normalization, or provider work; Console Speak now posts only store-issued snapshots and remains limited to completed assistant messages. Preserved the explicit non-Console global request path, saved global TTS selection, native audio.cpp complete-WAV lifecycle, legacy bridge, autoplay, and external-process ownership. Added deterministic store, handler, UI, privacy, native audio.cpp, and global-path regressions plus user/developer documentation. ADR-037 remains the governing decision; no new ADR was required.

Verification: 294 task-critical tests passed in the final green run. The complete focused union recorded 561 passes and three unrelated mounted workspace-browser timing failures; two passed immediately when rerun serially and the remaining stale-search assertion reproduced on untouched base dev. The broader Tests/TTS + Tests/Chat union recorded 4,681 passes and 74 skips; two socket errors disappeared outside the sandbox and every remaining failing node reproduced on untouched base dev. Ruff, scoped format, compileall, and diff checks passed. The new snapshot module passes mypy; three TTSEventHandler.notify diagnostics are unchanged from base dev. Independent review found no Critical, Important, or Minor issues.

Plan deviation: the native snapshot integration regression passed on first execution because the dedicated event support had already been completed and verified in the preceding plan step; no additional production change was needed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked and concise Implementation Notes record the delivered behavior and deviations.
- [x] #2 Focused store, event-handler, Console UI, native audio.cpp, complete-WAV, and legacy TTS tests pass.
- [x] #3 Task-scoped Ruff, formatting, compile, typing, and git diff checks pass or exact unchanged baselines are documented.
- [x] #4 ADR-037 and the approved Slice 3A design remain current and are linked from the task and plan.
- [x] #5 Independent review finds no unresolved Critical, Important, or Minor issue before merge.
<!-- DOD:END -->
