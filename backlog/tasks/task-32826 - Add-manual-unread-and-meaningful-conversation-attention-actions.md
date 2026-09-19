---
id: TASK-32826
title: Add manual unread and meaningful conversation attention actions
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 01:04'
updated_date: '2026-09-19 03:56'
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
- [x] #1 Mark as unread and Mark as read persist locally; marking the current chat survives until explicit successful reopening after leaving; automatic restore and failed navigation do not clear it.
- [x] #2 One right-hand action icon replaces the separate appearance button and asterisk; its menu includes Change icon and colour and preserves existing actions.
- [x] #3 Approval, blocked, failed, running, stopped, unseen-result and unread states have meaningful distinguishable indicators, text explanations and ASCII fallbacks; urgent states override unread without erasing it.
- [x] #4 Manual unread never changes operational receipt acknowledgement; stale async callbacks cannot clear a newer mark or affect another conversation/profile.
- [x] #5 Targeted persistence, activation, action-menu and mounted keyboard tests pass; existing custom icon and colour restore after overrides clear.
- [x] #6 Visible/page-ID unread enrichment remains correct beyond 100 marked chats; compare-and-clear serializes with writers and rejects stale generations even when timestamps repeat.
- [x] #7 Coarse background-unseen evidence with an unknown outcome never displays a success check; semantic projection preserves simultaneous and hidden/capped-row attention.
- [x] #8 The enlarged action menu clamps to its actual content height and remains keyboard/pointer usable near viewport edges.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-09-18-console-manual-unread-and-attention.md.
ADR required: yes
ADR path: backlog/decisions/171-console-conversation-review-and-attention.md
Reason: direct implementation of the approved durable-read and long-lived Console UX contracts; no new ADR required.
Preserve the Inspector sidebar and existing ownership/privacy boundaries; targeted tests only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented serialized durable manual unread, explicit successful revisit acknowledgement after paint, semantic attention/receipt projection, and one combined right-hand action menu. Unread remains independent of operational receipts; stale marks, navigation and profile callbacks are fenced.

ADR: backlog/decisions/171-console-conversation-review-and-attention.md. Updated the Console user guides and production-style tests; rebuilt CSS from token-backed source modules. Fresh review findings were reproduced and fixed (stuck receipts, cost-entry estimates, narrow reader focus).

Verification: final focused qualification 131 passed; broad feature regression 698 passed with nine independently reproduced baseline failures and three separately proven baseline exclusions. Full suite not run. See Docs/superpowers/reports/2026-09-18-console-conversation-ux-verification.md for scope and limitations.

Remains In Progress: required native terminal qualification is unavailable because computer-use access to iTerm was denied and Windows Terminal is not available. Headless production-style checks are not claimed as native evidence.

Final flat-list owner/recompose qualification: 58 passed. Final focused qualification: 131 passed. New files pass Ruff; modified legacy code has no changed-line diagnostics. See the linked verification report for the exact baseline exclusions and native-check limitation.
<!-- SECTION:NOTES:END -->
