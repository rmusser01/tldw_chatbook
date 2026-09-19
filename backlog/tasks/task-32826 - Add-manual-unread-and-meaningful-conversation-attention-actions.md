---
id: TASK-32826
title: Add manual unread and meaningful conversation attention actions
status: To Do
assignee: []
created_date: '2026-09-19 01:04'
updated_date: '2026-09-19 01:29'
labels: []
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-09-18-console-conversation-review-and-attention-design.md
  - backlog/decisions/171-console-conversation-review-and-attention.md
  - Docs/superpowers/plans/2026-09-18-console-manual-unread-and-attention.md
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
- [ ] #6 Visible/page-ID unread enrichment remains correct beyond 100 marked chats; compare-and-clear serializes with writers and rejects stale generations even when timestamps repeat.
- [ ] #7 Coarse background-unseen evidence with an unknown outcome never displays a success check; semantic projection preserves simultaneous and hidden/capped-row attention.
- [ ] #8 The enlarged action menu clamps to its actual content height and remains keyboard/pointer usable near viewport edges.
<!-- AC:END -->
