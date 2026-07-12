---
id: TASK-171
title: Simplify redundant opt(exception=False) logging sites
status: Done
assignee: []
created_date: '2026-07-11 22:03'
updated_date: '2026-07-11 23:52'
labels:
  - follow-up
  - logging
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The repo-wide exc_info codemod (PR #592) left ~6 sites using logger.opt(exception=False).<level>(...), which is equivalent to a plain logger.<level>(...) call. Simplify them for readability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 opt(exception=False) sites reduced to plain logger calls
- [ ] #2 No traceback-capturing behavior changed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in the quick-wins batch (branch claude/followups-quickwins). See Docs/superpowers/plans/2026-07-11-followups-quickwins.md.
<!-- SECTION:NOTES:END -->
