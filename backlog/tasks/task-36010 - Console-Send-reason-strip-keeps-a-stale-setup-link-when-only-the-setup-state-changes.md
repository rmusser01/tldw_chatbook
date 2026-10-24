---
id: TASK-36010
title: >-
  Console: the Send-reason strip keeps a stale setup link when only the setup
  state changes
status: To Do
assignee: []
created_date: '2026-10-21 07:23'
labels:
  - console
  - bug
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-33620.15.7 (2026-10-21); it predates that branch. The composer's Send-reason strip turns the whole reason into a link that opens the setup wizard while the composer is setup-blocked, and shows plain text otherwise. Its update is skipped when the new content equals the old, but Textual's `Content` equality compares plain text only, not styles or actions. So when the setup-blocked state flips and the reason text stays the same, the strip keeps the old presentation: a link that should have gone stays clickable, or a needed link to setup never appears.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the composer becomes setup-blocked with an unchanged reason, the strip's reason opens the setup wizard
- [ ] #2 When the setup block clears with an unchanged reason, the strip's reason is plain text with no action
- [ ] #3 An unchanged reason and setup state still requests no layout (the TASK-33620.15.7 pin stays green)
- [ ] #4 A test covers both flips and fails on the current code
<!-- AC:END -->
