---
id: TASK-33244
title: >-
  Dreams settings surface - enable Dreams and edit the interest profile in
  Settings
status: To Do
assignee: []
created_date: '2026-09-29 03:18'
labels:
  - dreams
  - settings
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-filed (the original task-32903 was lost when dev reassigned the number; original filed 2026-09-23 from Phase 1 final review ruling R20). Dreams currently requires hand-editing config.toml to enable, and the interest profile (topics/region) has no editing UI - the profile only populates from notes/media/Personal Context signals. Add a Settings screen section: enable toggle, provider/model pickers, region field, topic list editor (user-seeded topics addable/removable with weight), goals are edited via the existing DreamsGoalsModal (g from the story modal) - cross-link rather than duplicate. Fold the queued stack polish: rename the stale footer-hints test (says ten actions, modal now has eleven).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dreams can be enabled and disabled from the Settings screen without editing config.toml by hand,Interest profile topics and region are viewable and editable; user topics addable and removable with weight,Section follows ADR-150 tokens and lives in the canonical settings_screen.py (no legacy settings windows),Stale footer-hints test name corrected to the current action count,dreams.md enable-path section points at the Settings screen first, config.toml as the escape hatch
<!-- AC:END -->
