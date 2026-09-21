---
id: TASK-32861
title: One StatusLine helper for the duplicated _set_status implementations
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies: []
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Twenty-two `def _set_status` implementations exist with no shared status-line helper (grep for a StatusLine/StatusBar/StatusMixin base: zero hits). Twenty-one are UI status lines with divergent shapes — plain message; message + `error: bool`; `copy` + severity + app-announcement channel (`UI/Speech/speech_settings_pane.py:1255`); mounted-guard + `query_one("#id", Static).update(...)` (`UI/Research_Window.py:799`, `UI/Writing_Window.py:257`, 7 in Widgets/Console, others) — and one is not UI at all (`Audio/diarizer_local.py:556`, a backend progress callback; excluded). One helper/mixin (~40 LOC) absorbing "guard `is_mounted`, update the Static by id, optional severity/announcement" collapses ~18-21 sites of 5-12 lines each: ~150-250 LOC plus severity-rendering consistency. ADR required: no.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One status-line helper/mixin owns the mounted-guard, update-by-id, severity, and announcement variants
- [ ] #2 The ~21 UI sites adopt it; `Audio/diarizer_local.py` is explicitly excluded
- [ ] #3 Severity/announcement behavior is consistent across surfaces that adopt it (or per-surface deltas are intentional and noted)
- [ ] #4 Net ~150-250 LOC deleted; touched widget suites pass
<!-- AC:END -->
