---
id: TASK-1501
title: 'Wizard tools step: broken switch row layout and bare tool labels'
status: Done
assignee: []
created_date: '2026-07-31 00:22'
updated_date: '2026-07-31 01:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: switch borders collide into fragments against labels; rows are lowercase internals ('glob files') with no description or risk badge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Switch rows render cleanly aligned at 120x40 and 80x24
- [ ] #2 Each tool row has a one-line plain-language description
- [ ] #3 Risk-tagged tools carry a visible badge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rows restyled: 3-row height, switch + bold name + muted plain-language description (static _TOOL_COPY map; ⚠ marks disk/note-mutating tools). Region-overlap regression test added.
<!-- SECTION:NOTES:END -->
