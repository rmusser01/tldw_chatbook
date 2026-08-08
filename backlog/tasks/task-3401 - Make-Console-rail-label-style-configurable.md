---
id: TASK-3401
title: Make Console rail label style configurable
status: To Do
assignee: []
created_date: '2026-08-08 05:44'
updated_date: '2026-08-08 05:46'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-08-07-console-rail-label-setting-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users keep the existing horizontal collapsed Console rail labels or opt into the compact stacked vertical presentation from the canonical Settings screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Horizontal rail labels remain the default when no valid preference is saved
- [ ] #2 Settings > Console Behavior lets users opt into stacked vertical collapsed rail labels
- [ ] #3 The preference persists and reloads across app sessions
- [ ] #4 Expanded headers, tooltips, badges, focus behavior, and non-Console rails remain unchanged
- [ ] #5 Targeted Settings, Console rail, and configuration tests pass
- [ ] #6 Saving immediately updates the in-memory preference; returning to Console shows both handle styles without restarting Chatbook
<!-- AC:END -->
