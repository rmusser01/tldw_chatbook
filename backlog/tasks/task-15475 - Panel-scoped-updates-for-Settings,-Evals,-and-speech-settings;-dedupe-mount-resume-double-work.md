---
id: TASK-15475
title: Panel-scoped updates for Settings, Evals, and speech settings; dedupe mount/resume double-work
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - settings
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: Settings' `active_category` is a screen-level recompose reactive (`settings_screen.py:1930`) — each rail click rebuilds the category buttons plus a 60-150-widget detail pane, and entering Overview triggers a SECOND full-screen recompose via the sync-rows refresh (`:6659-6660`); "Sync preview"/"Run" full-screen-recompose twice per click to change three Statics (`:14970/:15002`). Evals rail selection recomposes 150-300 widgets per click (`UI/Evals/evals_screen.py:409`). The speech/TTS settings panel rebuilds ~200 widgets on every provider/policy dropdown change (`Widgets/Settings_Widgets/speech_tts_settings_panel.py:3741-3787`; `speech_playground_pane.py:700` shows the correct region-swap pattern in the same feature). Also duplicated per-visit work: Console dispatches `_refresh_console_skill_candidates` twice per visit (`chat_screen.py:13956` and `:19165`, exclusive=False) and syncs task-resume state twice; Settings runs `_queue_sync_rows_refresh` on both on_mount and on_screen_resume (`:2398/:2444`).

Fix direction: detail-pane-scoped swaps and targeted Static patches; per-instance flags to dedupe the mount/resume double-dispatch. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings rail clicks and Evals selections rebuild only the detail region (evidence per surface); sync preview updates statics in place
- [ ] #2 Speech panel dropdown changes rebuild only the provider-form subsection
- [ ] #3 The duplicated mount/resume workers run once per visit (evidence); all touched surfaces behaviorally unchanged (tests)
<!-- AC:END -->
