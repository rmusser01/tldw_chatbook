---
id: TASK-1692
title: Establish Speech and TTS settings ownership contracts
status: To Do
assignee: []
created_date: '2026-08-01 06:01'
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
- [ ] #1 Every current shared default and built-in OpenAI, audio.cpp, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk control, action, and readout is classified exactly once as global configuration, Studio preference, Voice Profile operation, runtime operation/readout, or explicitly retired with a reason (OWN-001 through OWN-005).
- [ ] #2 The ownership inventory preserves canonical provider IDs, keeps Default TTS Provider separate from Configure Provider, rejects duplicate and unknown classifications, and adds no dynamic plugin or generic form framework (OWN-005 and OWN-006).
- [ ] #3 Shared configuration state accepts only Inherited, Default, Saved, Unsaved, Incomplete, and Invalid, while runtime state independently accepts only Not checked, Checking, Ready, Stale, Unavailable, and Reconfiguring (STATE-010 and STATE-011).
- [ ] #4 A bounded cross-screen navigation value carries only a canonical provider ID and an allowed configure, test, refresh-models, or refresh-voices intent; it cannot carry credentials, field values, synthesis text, or arbitrary widget selectors (IA-005, SEC-001, and SEC-004).
- [ ] #5 A revisioned safe status value can identify provider, saved configuration revision, runtime revision, optional catalog revision, observation time, freshness, bounded diagnostic category, and recovery action without raw URLs, exception text, secrets, or submitted content (STATE-013 and SEC-004).
- [ ] #6 Automated completeness tests fail when a current Speech control is unclassified, multiply classified, or assigned to a scope that contradicts ADR-039, and cover all seven built-in providers plus shared defaults.
- [ ] #7 This task changes no persisted setting, visible Settings or Lab ownership, provider network behavior, character assignment, adapter routing, or managed audio.cpp behavior.
<!-- AC:END -->
