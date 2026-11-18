---
id: TASK-41725
title: "The persistent diagnostic inventory is red on dev for agent_service.py's on_child_settled consumer message"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- tests
- architecture
priority: high
references:
- Tests/Architecture/test_persistent_diagnostic_inventory.py
- tldw_chatbook/Agents/agent_service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-16 restore implementer and re-run by the coordinator (2026-11-18): `test_reviewed_diagnostic_changes_are_metadata_only` fails on the `agent_service.py` diagnostic "on_child_settled consumer raised", at dev 45b1949ca3 and at dev 97f8c58c0c. No round-16 branch touches that file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The diagnostic is reviewed: it logs metadata only and is recorded in the inventory, or it is changed to do so
- [ ] #2 The test passes, and the commit that made it red is named in the notes
<!-- AC:END -->
