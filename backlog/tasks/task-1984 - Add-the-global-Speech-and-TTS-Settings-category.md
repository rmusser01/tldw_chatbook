---
id: TASK-1984
title: Add the global Speech and TTS Settings category
status: Done
assignee: []
created_date: '2026-08-01 06:03'
updated_date: '2026-08-01 11:57'
labels:
  - tts
  - settings
  - ui
dependencies:
  - TASK-1981
  - TASK-1983
references:
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
  - backlog/decisions/012-provider-credential-settings-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make application-wide TTS defaults and provider setup discoverable in the primary Settings destination, with one truthful write owner for connection, credential, initialization, and safety configuration. Existing providers must keep their behavior while users gain a coherent global setup path that does not perform hidden runtime work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Main Settings contains a keyboard-reachable Speech & TTS category whose search index matches speech, TTS, voice, audio.cpp, audio_cpp, OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk and opens the named provider when applicable (IA-001 and IA-002).
- [x] #2 The category persistently states that it edits application-wide defaults, explains that Studio preferences are separate, and provides an Open Speech Lab action without running a test or refresh (IA-003 and OWN-001).
- [x] #3 Global defaults expose provider, model policy and value, voice policy and value, format, and speed; Default TTS Provider and Configure Provider are separate, and only the selected configuration form is mounted or expanded (CFG-001 through CFG-003 and OWN-006).
- [x] #4 The selected-provider forms expose every globally owned field from TASK-1981: all external audio.cpp connection, timeout, and safety fields; OpenAI key, base URL, and organization; ElevenLabs key; Kokoro device and ONNX resources; Chatterbox device and voice resources; Higgs model and voice resources, device, flash attention, and dtype; and AllTalk URL. Path-picker affordances remain beside their global path fields (OWN-005).
- [x] #5 Credential Set, Replace, and Clear saved credential are explicit operations outside ordinary Save; environment sources are read-only, masked placeholders are never payloads, shadowed local fallbacks are explained, and local secret storage is labeled (CFG-008, SEC-001, and SEC-002).
- [x] #6 Save performs local shape, range, constraint, URL, and path-syntax validation and atomic persistence without connecting, discovering, initializing a model, or synthesizing; Revert and Restore Non-secret Defaults have the approved non-secret draft semantics (CFG-005, CFG-006, and CFG-010).
- [x] #7 A successful save reconfigures only providers whose effective adapter-affecting global inputs changed, while selection-only and unrelated changes avoid adapter recreation; persistence success and runtime reconfiguration results remain distinct and no fallback provider is selected (CFG-007, CFG-009, STATE-021, and STATE-022).
- [x] #8 Global-owned fields are no longer writable through the Lab editor once the Settings replacement is available; Lab shows their effective values or a scoped link without becoming a second persistence owner, and no current field disappears during the transition (OWN-001 through OWN-005).
- [x] #9 Automated Textual and persistence tests cover search, provider preselection, keyboard flow, all-provider field completeness, credential intent, local validation, zero network calls on normal Settings actions, exact saved mutations, targeted reconfiguration, and unchanged legacy-provider configuration behavior.
- [x] #10 The category exposes no binary, server.json, bind, launch, adoption, restart, supervision, stop, or managed audio.cpp control.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md and backlog/decisions/012-provider-credential-settings-boundary.md
Reason: TASK-1984 directly implements the accepted durable global owner, credential mutation boundary, and provider-runtime handoff; no new decision is introduced.

Detailed plan: Docs/superpowers/plans/2026-08-01-task-1984-global-speech-tts-settings.md

1. Add failing pure tests for inventory, validation, non-secret defaults, credential intent, and affected-provider classification.
2. Implement the bounded global Speech & TTS settings model and atomic mutation proposals.
3. Add failing Textual tests for discoverability, provider preselection, keyboard flow, scope copy, one selected form, field completeness, path pickers, and managed-control exclusion.
4. Wire the global panel into Settings and reuse the established TTS publication path for persistence plus targeted reconfiguration.
5. Make global-owned Lab controls read-only and exclude them from Lab save payloads without dropping any existing control.
6. Run focused and neighboring regression gates, static checks, and independent review before completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the searchable global Speech & TTS Settings category, bounded provider-specific forms, explicit credential operations, non-secret draft actions, and targeted runtime publication.
- Moved global-owned Lab controls to read-only effective-value presentation while retaining the existing request-scoped compatibility controls for the later Studio migration.
- Persisted OpenAI and ElevenLabs credentials through canonical `api_settings` sections, removed legacy aliases on clear, and refreshed the long-lived application config snapshot after successful cache publication.
- Added pure model, Textual layout/interaction, persistence, privacy, and legacy-adapter regressions. Final independent review reported 537 passing relevant/neighboring tests; the known pre-existing TTS export-baseline assertion remains outside this task.
- ADR check: implemented [ADR-039](../decisions/039-global-and-studio-tts-settings-ownership.md) and [ADR-012](../decisions/012-provider-credential-settings-boundary.md); no new ADR was required.
<!-- SECTION:NOTES:END -->
