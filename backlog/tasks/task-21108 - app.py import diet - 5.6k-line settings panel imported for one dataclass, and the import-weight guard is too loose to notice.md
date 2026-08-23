---
id: TASK-21108
title: >-
  app.py import diet - 5.6k-line settings panel imported for one dataclass, and the import-weight guard is too loose to notice
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
  - imports
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21108).

app.py's top-level imports grew 194 -> 220 since the 2026-08-11 audit pin. Concrete deferrable
whales verified: `Widgets/Settings_Widgets/speech_tts_settings_panel` (5,618-line widget module
imported for the single `SpeechTTSPanelDraftSnapshot` payload class used in isinstance checks,
app.py:329-331); `TTS/voice_bundle_service` (1,857); the `Notes/notes_sync_runtime` chain;
Notifications package init. None is needed before first paint. Meanwhile
`Tests/Performance/test_app_import_weight.py:85-86` allows 8.0 s / 4,000 modules - far above
any real drift signal.

## Acceptance Criteria

- [ ] `SpeechTTSPanelDraftSnapshot` lives in a small types module; the 5,618-line panel module is no longer on the app import path (sys.modules assertion)
- [ ] voice_bundle_service and the notes_sync_runtime import chain are deferred to first use
- [ ] The import-weight guardrail budgets are tightened to sit just above the measured post-diet reality, so the next regression of this class fails a test
