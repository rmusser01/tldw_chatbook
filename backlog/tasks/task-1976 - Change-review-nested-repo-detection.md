---
id: TASK-1976
title: 'Change review: detect nested repos and disclose the tracking hole'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - workspaces
  - change-review
dependencies:
  - TASK-1971
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
git records a child repo as a gitlink: uncommitted changes INSIDE a nested clone are invisible to the snapshot — silently violating the feature's core promise for the common ~/projects root shape. v1 detects nested repos during the registration scan and discloses honestly: card/inspector and Review screen banners state 'N nested repositories inside this root are not tracked', naming them on the Review screen. (Auto-registering them as sub-roots is TASK-1977.)

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A root containing a child repo reports the child by path in the Review screen banner
- [ ] #2 An edit inside the child repo produces NO diff rows AND the banner is present (the hole is disclosed, not hidden)
- [ ] #3 A root that IS a repo (no children) shows no banner and tracks normally
- [ ] #4 Detection runs in the registration scan, not per-turn
<!-- AC:END -->
