---
id: TASK-42011
title: "The reviewed-diagnostic metadata test is red on dev"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- tests
priority: low
references:
- Tests/Architecture/test_persistent_diagnostic_inventory.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41711 and TASK-41714 (round 17). `test_reviewed_diagnostic_changes_are_metadata_only` fails on rows in `Agents/agent_service.py` ('on_child_settled consumer raised') and `Chat/console_chat_store.py`, neither touched by those branches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rows are reviewed and the inventory updated, or the code restored, and the test passes
<!-- AC:END -->
