---
id: TASK-32826
title: Add manual unread and meaningful conversation attention actions
status: To Do
assignee: []
created_date: '2026-09-19 01:04'
labels: []
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-09-18-console-conversation-review-and-attention-design.md
  - backlog/decisions/171-console-conversation-review-and-attention.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users mark Console conversations unread as reminders and recognize each attention state through a representative icon and one consistent action menu.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mark as unread and Mark as read persist locally; marking the current chat survives until explicit successful reopening after leaving; automatic restore and failed navigation do not clear it.
- [ ] #2 One right-hand action icon replaces the separate appearance button and asterisk; its menu includes Change icon and colour and preserves existing actions.
- [ ] #3 Approval, blocked, failed, running, stopped, unseen-result and unread states have meaningful distinguishable indicators, text explanations and ASCII fallbacks; urgent states override unread without erasing it.
- [ ] #4 Manual unread never changes operational receipt acknowledgement; stale async callbacks cannot clear a newer mark or affect another conversation/profile.
- [ ] #5 Targeted persistence, activation, action-menu and mounted keyboard tests pass; existing custom icon and colour restore after overrides clear.
<!-- AC:END -->

## ID allocation

CLI initially allocated TASK-32773; corrected immediately before work to TASK-32826 after surveying all local refs and 31 worktrees (maximum observed 32825).
