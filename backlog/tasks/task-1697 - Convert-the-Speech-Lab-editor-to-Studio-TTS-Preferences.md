---
id: TASK-1697
title: Convert the Speech Lab editor to Studio TTS Preferences
status: To Do
assignee: []
created_date: '2026-08-01 06:04'
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
- [ ] #1 The Lab subview is named Studio TTS Preferences, persistently states that edits affect only Studio, and links to global Speech & TTS Settings with the canonical selected-provider context (IA-004).
- [ ] #2 The Studio editor exposes shared provider, model, voice, format, and speed overrides plus only the request-scoped provider tuning classified by TASK-1692; credentials, endpoints, initialization resources, safety limits, and unsupported runtime-global options are absent or read-only links to their owner (OWN-002, CFG-022, and CFG-026).
- [ ] #3 On mount and provider switch, saved Studio overrides are restored for that canonical provider while absent values visibly inherit the current global source without copying it into Studio storage (CFG-021 and CFG-022).
- [ ] #4 Current validated Studio controls take precedence for the next Studio generation without saving automatically, and generation failure or cancellation does not persist the draft (CFG-024).
- [ ] #5 Save Studio Preferences changes only the Studio store and triggers no global mutation, credential operation, character mutation, catalog refresh, or provider reconfiguration; Revert reloads the last saved Studio snapshot and Reset to Global deletes all Studio overrides (CFG-020, CFG-021, and CFG-023).
- [ ] #6 Opening a character TTS profile creates a clearly labeled non-persistent preview usable by the current Studio generation; Save alone cannot absorb it, and only explicit Adopt as Studio Preferences followed by a successful Studio save makes compatible values durable (CFG-025).
- [ ] #7 Changing provider, leaving the pane, following a global-settings link, or dismissing the screen with a dirty draft offers Save and continue, Discard and continue, and Cancel; failed Save and Cancel preserve the current form and focus, and hidden unsaved provider drafts are not retained (CFG-012).
- [ ] #8 Voice-blend add, import, and export are presented as Voice Profile library operations rather than global or Studio preference persistence, while existing generation and playback actions remain runtime operations (OWN-004 and OWN-005).
- [ ] #9 Unknown provider IDs and unsupported option keys fail closed with field-specific safe errors, and a corrupt Studio record offers Studio-only reset without suggesting global or character reset (STATE-020 and STATE-024).
- [ ] #10 Automated Textual, storage, and generation tests cover scope copy, keyboard flow, per-provider restoration, inheritance, current-draft precedence, Save/Revert/Reset isolation, zero reconfiguration on Studio save, provider switching, dirty navigation, preview generation, explicit adoption, unadopted preview dismissal, unsupported tuning, and unchanged global and character stores.
<!-- AC:END -->
