---
id: TASK-25906
title: 'Doctor: aggregate health-check surface'
status: To Do
assignee: []
created_date: '2026-08-31 15:09'
updated_date: '2026-08-31 15:11'
labels:
  - ops
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chatbook is local-first, so the user owns the whole stack - local inference servers, optional extras, API keys, DB integrity, config validity - and today has no way to ask what is broken. Verified on origin/dev: a named grep for doctor, healthcheck, diagnostic and self_test across tldw_chatbook, Packaging and scripts returns only a log sink (Utils/persistent_diagnostics.py), TTS abbreviation expansions, and TTS supervisor health intervals; the nearest surface is Settings > Diagnostics, which parses and reloads the TOML (UI/Screens/settings_screen.py:9030-9042). Every ingredient already exists and is separately proven - this task is aggregation and presentation, not new probing logic.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single surface reports pass/fail per check with a reason, covering at minimum: config load status, optional-dependency availability, DB integrity, configured-provider readiness, and private-path posture
- [ ] #2 Each check reuses the existing implementation rather than reimplementing it (Utils/optional_deps, DB/base_db.check_integrity, config.get_config_load_failure, config.get_detected_api_providers)
- [ ] #3 Checks that require network calls are opt-in and clearly labeled, never run by default
- [ ] #4 A failing check names the specific remediation where one is known, and says so honestly where none is
- [ ] #5 No secret values appear in the output; provider readiness reports configured/not-configured, never the key
- [ ] #6 The surface is reachable without a working config - it must still run and report when config load has failed
<!-- AC:END -->
