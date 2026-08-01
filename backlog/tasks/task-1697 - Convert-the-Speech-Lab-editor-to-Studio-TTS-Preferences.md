---
id: TASK-1697
title: Convert the Speech Lab editor to Studio TTS Preferences
status: Done
assignee: []
created_date: '2026-08-01 06:04'
updated_date: '2026-08-01 15:49'
labels:
  - tts
  - studio
  - ui
dependencies:
  - TASK-1693
  - TASK-1694
  - TASK-1695
  - TASK-1696
references:
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
  - backlog/decisions/028-character-tts-generation-profile-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Lab Speech editor an explicitly Studio-only workspace backed by the separate Studio preference store. Users must be able to experiment and persist supported request tuning without changing global setup or character profiles, and character selections must remain safe previews until deliberately adopted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Lab subview is named Studio TTS Preferences, persistently states that edits affect only Studio, and links to global Speech & TTS Settings with the canonical selected-provider context (IA-004).
- [x] #2 The Studio editor exposes shared provider, model, voice, format, and speed overrides plus only the request-scoped provider tuning classified by TASK-1692; credentials, endpoints, initialization resources, safety limits, and unsupported runtime-global options are absent or read-only links to their owner (OWN-002, CFG-022, and CFG-026).
- [x] #3 On mount and provider switch, saved Studio overrides are restored for that canonical provider while absent values visibly inherit the current global source without copying it into Studio storage (CFG-021 and CFG-022).
- [x] #4 Current validated Studio controls take precedence for the next Studio generation without saving automatically, and generation failure or cancellation does not persist the draft (CFG-024).
- [x] #5 Save Studio Preferences changes only the Studio store and triggers no global mutation, credential operation, character mutation, catalog refresh, or provider reconfiguration; Revert reloads the last saved Studio snapshot and Reset to Global deletes all Studio overrides (CFG-020, CFG-021, and CFG-023).
- [x] #6 Opening a character TTS profile creates a clearly labeled non-persistent preview usable by the current Studio generation; Save alone cannot absorb it, and only explicit Adopt as Studio Preferences followed by a successful Studio save makes compatible values durable (CFG-025).
- [x] #7 Changing provider, leaving the pane, following a global-settings link, or dismissing the screen with a dirty draft offers Save and continue, Discard and continue, and Cancel; failed Save and Cancel preserve the current form and focus, and hidden unsaved provider drafts are not retained (CFG-012).
- [x] #8 Voice-blend add, import, and export are presented as Voice Profile library operations rather than global or Studio preference persistence, while existing generation and playback actions remain runtime operations (OWN-004 and OWN-005).
- [x] #9 Unknown provider IDs and unsupported option keys fail closed with field-specific safe errors, and a corrupt Studio record offers Studio-only reset without suggesting global or character reset (STATE-020 and STATE-024).
- [x] #10 Automated Textual, storage, and generation tests cover scope copy, keyboard flow, per-provider restoration, inheritance, current-draft precedence, Save/Revert/Reset isolation, zero reconfiguration on Studio save, provider switching, dirty navigation, preview generation, explicit adoption, unadopted preview dismissal, unsupported tuning, and unchanged global and character stores.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md and backlog/decisions/028-character-tts-generation-profile-ownership.md
Reason: TASK-1697 implements the accepted Studio persistence/editor, request-precedence, and character-preview ownership boundaries without changing their architecture.

1. Add failing pure and Textual tests for sparse inherited Studio fields, supported provider tuning, validation, scope copy, action inventory, and the absence of global connection/credential controls.
2. Replace the transitional mixed Lab settings pane with a compact Studio TTS Preferences editor backed only by `StudioTTSPreferenceStore`, including Save, Revert, Reset to Global, corrupt-record recovery, and per-provider restoration.
3. Seed Playground controls from persisted Studio overrides with visible global inheritance, freeze the current validated Playground controls into `TTSStudioDraftSelection`, and route Studio generation through the existing effective-settings admission path without implicit persistence.
4. Preserve character profiles as labeled non-persistent Playground previews and add the explicit Adopt as Studio Preferences handoff; ordinary Studio Save must ignore an unadopted preview.
5. Add dirty-draft guards for provider changes, Lab view changes, global-Settings navigation, and dismissal with Save/Discard/Cancel semantics and retained focus on failure or cancellation.
6. Keep Voice Profile library operations and generation/playback actions separate, then run focused storage, Textual, generation, navigation, legacy-provider, and isolation regressions plus static checks and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Studio-only Speech preference editor and wired its sparse, revisioned snapshot through Playground request admission without adding any global, credential, character, catalog, or provider-reconfiguration write path. The editor restores provider-scoped overrides with visible inheritance; supports isolated Save, Revert, and Reset to Global; protects dirty navigation with Save/Discard/Cancel and focus retention; and keeps audio.cpp WAV/speed constraints read-only.

Character TTS profiles now enter Playground as exact non-persistent previews and require explicit adoption plus a successful Studio save. Review also found and fixed a catalog race that could replace a settled audio.cpp preview with the global provider while leaving stale adoption state; exact profile selection is now preserved through catalog/voice discovery and is detached on user axis edits.

ADR: implements existing ADR-039 and ADR-028; no new decision was introduced. Verification: 307 affected storage, generation, Textual, navigation, profile, and admission tests passed; the broader Speech/STTS/TTS run reached 2,489 passed with 14 expected optional-dependency skips and one stale prior portability-export allowlist, whose corrected contract test passes. Ruff check, Ruff format check, `git diff --check`, self-review, and independent re-review all pass.
<!-- SECTION:NOTES:END -->
