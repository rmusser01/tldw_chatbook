---
id: TASK-1981
title: Establish Speech and TTS settings ownership contracts
status: Done
assignee: []
created_date: '2026-08-01 06:01'
updated_date: '2026-08-01 07:03'
labels:
  - tts
  - settings
  - architecture
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
  - 'https://github.com/rmusser01/tldw_chatbook/pull/1159'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a single testable contract for which scope owns every existing Speech and TTS field or action, and for the safe state and navigation values shared by global Settings and the Speech Lab. This foundation prevents later UI slices from dropping controls, conflating saved configuration with runtime readiness, or leaking request and credential data across scopes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every current shared default and built-in OpenAI, audio.cpp, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk control, action, and readout is classified exactly once as global configuration, Studio preference, Voice Profile operation, runtime operation/readout, or explicitly retired with a reason (OWN-001 through OWN-005).
- [x] #2 The ownership inventory preserves canonical provider IDs, keeps Default TTS Provider separate from Configure Provider, rejects duplicate and unknown classifications, and adds no dynamic plugin or generic form framework (OWN-005 and OWN-006).
- [x] #3 Shared configuration state accepts only Inherited, Default, Saved, Unsaved, Incomplete, and Invalid, while runtime state independently accepts only Not checked, Checking, Ready, Stale, Unavailable, and Reconfiguring (STATE-010 and STATE-011).
- [x] #4 A bounded cross-screen navigation value carries only a canonical provider ID and an allowed configure, test, refresh-models, or refresh-voices intent; it cannot carry credentials, field values, synthesis text, or arbitrary widget selectors (IA-005, SEC-001, and SEC-004).
- [x] #5 A revisioned safe status value can identify provider, saved configuration revision, runtime revision, optional catalog revision, observation time, freshness, bounded diagnostic category, and recovery action without raw URLs, exception text, secrets, or submitted content (STATE-013 and SEC-004).
- [x] #6 Automated completeness tests fail when a current Speech control is unclassified, multiply classified, or assigned to a scope that contradicts ADR-039, and cover all seven built-in providers plus shared defaults.
- [x] #7 This task changes no persisted setting, visible Settings or Lab ownership, provider network behavior, character assignment, adapter routing, or managed audio.cpp behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md
Reason: TASK-1981 directly implements the accepted ownership, state, runtime-status, and bounded-navigation contracts; no new ADR is needed.

Detailed plan: Docs/superpowers/plans/2026-07-31-task-1981-speech-tts-settings-ownership-contracts.md

1. Add failing completeness and ADR-039 scope-partition tests for every current built-in Speech control.
2. Add the explicit immutable ownership manifest and strict validator without wiring it into live UI behavior.
3. Add failing tests and minimal bounded DTOs for configuration/runtime state, cross-screen intent, and revisioned safe status.
4. Run focused neighboring regressions, static checks, and a diff-only non-behavioral audit.
5. Record verification and ADR conformance in Implementation Notes, check the acceptance criteria, and mark Done only after all gates pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented an explicit immutable ownership inventory for all current shared and built-in Speech/TTS controls, strict completeness/ADR-039 validation, bounded configuration and runtime state enums, safe cross-screen navigation targets, and revisioned runtime-status values. Studio ownership is intentionally limited to the five controls proven to reach current request DTOs end to end; all other provider configuration remains global or read-only until a request-local contract exists. No persistence, live UI ownership, network behavior, character assignment, adapter routing, or managed audio.cpp behavior changed. ADR: conforms to backlog/decisions/039-global-and-studio-tts-settings-ownership.md; no new ADR. Files: tldw_chatbook/UI/Speech/speech_settings_contracts.py, Tests/UI/test_speech_settings_contracts.py, the detailed Superpowers plan, and this task. Verification: focused and neighboring gate 95 passed; Ruff and git diff --check passed; independent code review found no remaining issues. A full-suite attempt reached approximately 11 percent with only the already accepted persistent-diagnostic-inventory baseline failure observed before the redundant long run was stopped; the baseline test was then reproduced directly as 1 failed and is unrelated to this task's files.
<!-- SECTION:NOTES:END -->
