---
id: TASK-38230
title: 'Modal wide-tier reverse sweep is red at its first assert'
status: To Do
assignee: []
created_date: '2026-10-30 18:43'
labels:
  - tests
dependencies: []
references:
  - Tests/UI/test_modal_wide_tier.py
  - Tests/UI/modal_wide_tier_registry.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_every_modalscreen_class_is_tiered_or_explicitly_skipped fails at its first assert on dev today (stale skip names), which masks its "missing" assert. Roleplay frame B2b added a skip row for its RoleplayChoiceModal and does not fix the stale names. Found during Roleplay frame B2b (TASK-33910.4, plan Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The reverse sweep passes on dev with every stale skip name removed or re-pointed
- [ ] #2 A new ModalScreen subclass with no tier and no skip row turns it red
<!-- AC:END -->
