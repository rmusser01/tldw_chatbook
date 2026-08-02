---
id: TASK-1979
title: 'Change review: Settings surface, per-workspace toggle, git-absent gating'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - settings
  - change-review
  - workspaces
dependencies:
  - TASK-1971
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User control and honest availability: flat [change_review] config section (enabled, max_file_bytes, max_files, max_total_bytes, retention_days, diff_display_max_lines) with env overrides; a per-workspace toggle in Settings beside folder roots; feature-absent states with honest copy — no git binary ('Change review needs git — install git to enable'), no folder roots configured. Toggles take effect without restart (poke the live config tree — the app_config-captured-once trap).

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Disabling per-workspace stops snapshots for that workspace's roots on the NEXT run without restart
- [ ] #2 git absent -> Settings and card copy state the reason; runs behave exactly as with the feature off
- [ ] #3 Every knob is read live from config with the documented env-var override
- [ ] #4 Settings copy passes the monochrome/persistence-badge conventions of the Settings screen
<!-- AC:END -->
