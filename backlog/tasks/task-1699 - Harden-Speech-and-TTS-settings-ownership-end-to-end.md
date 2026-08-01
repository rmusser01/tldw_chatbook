---
id: TASK-1699
title: Harden Speech and TTS settings ownership end to end
status: To Do
assignee: []
created_date: '2026-08-01 06:05'
updated_date: '2026-08-01 06:08'
labels:
  - tts
  - settings
  - accessibility
  - testing
dependencies:
  - TASK-1693
  - TASK-1694
  - TASK-1695
  - TASK-1696
  - TASK-1697
  - TASK-1698
references:
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove the completed global, Studio, character, and runtime ownership model as one accessible and privacy-safe user journey before live external-server acceptance. This closeout gate must catch cross-slice regressions that focused unit tests cannot, while preserving all legacy providers and the existing complete-WAV playback contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Global Settings and Studio pass programmatic-label, keyboard-only focus order, non-color state, disabled-reason, status-announcement, and supported narrow-terminal layout gates; primary setup, Save, recovery, and Cancel controls remain reachable without horizontal scrolling (A11Y-001 through A11Y-006).
- [ ] #2 A deterministic first-time harness finds Speech & TTS through Settings search, saves a fake external audio.cpp URL without network access, enters Lab through the scoped link, explicitly tests and refreshes against fakes, returns with exact context, and hands a valid complete WAV artifact to the existing Console or Roleplay playback control.
- [ ] #3 Cross-surface tests prove a Studio save produces no global config mutation and no adapter reconfiguration, Reset to Global deletes overrides and follows later global changes, and character preview and assignment flows never mutate the wrong store.
- [ ] #4 Privacy tests prove credentials, masked placeholders, environment values, submitted synthesis text, raw provider bodies, and arbitrary exceptions do not enter Studio storage, character stores, navigation context, status or catalog snapshots, diagnostics, metrics, caches, migrations, or artifact provenance (SEC-001 through SEC-005).
- [ ] #5 Migration and rollback tests prove repeated migration is a no-op, malformed Studio data is isolated, legacy keys remain readable, an older reader can ignore speech_studio, and disabling the Studio reader restores prior global behavior without destructive down-migration (MIG-001 through MIG-006).
- [ ] #6 OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk retain their accepted configuration and generation request shapes, and approximate legacy catalogs cannot invalidate exact selections merely by omission.
- [ ] #7 Race coverage exercises concurrent global save, provider reconfiguration, Studio save, catalog and voice refresh, navigation, generation, and playback so stale revisions cannot mutate current UI or invalidate a completed artifact.
- [ ] #8 Normal CI uses only fakes and pinned fixtures, downloads no model or server, starts no audio.cpp process, contacts no provider, and validates complete WAV structure and playback handoff without claiming audible output or incremental streaming.
- [ ] #9 No managed audio.cpp setting or lifecycle behavior, new provider, native legacy-adapter migration, character-profile redesign, hidden discovery, or automatic speech is added.
- [ ] #10 Release evidence maps every IA, OWN, CFG, CAT, STATE, MIG, SEC, and A11Y requirement in the approved PRD to at least one passing focused test, end-to-end test, or explicitly identified manual UAT journey, and no requirement is left without evidence.
<!-- AC:END -->
