---
id: TASK-432
title: Provider test reports draft values and the Test control is clickable
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - settings
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live in Settings > Providers & Models: with an unsaved draft endpoint (:9099) the test correctly exercised the DRAFT (endpoint reachable, 1 model - nothing was listening on the displayed URL) but the evidence line printed the stale saved value api_settings.llama_cpp.api_url=http://localhost:8080/completion, so the proof contradicts what was tested. Separately, "Test Provider" renders like a control but clicking it does nothing; the test only runs via the 't' category hotkey and only when focus is outside an input, and nothing explains this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Test results display the exact endpoint/model/key-source values that were used in the test (draft values when testing a draft)
- [ ] #2 The Test control is activatable by mouse click
- [ ] #3 The 't' hotkey behavior and its focus requirement are discoverable or removed as the sole path
<!-- AC:END -->
