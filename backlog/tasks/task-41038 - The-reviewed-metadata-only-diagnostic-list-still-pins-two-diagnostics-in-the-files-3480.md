---
id: TASK-41038
title: "The reviewed metadata-only diagnostic list still pins two diagnostics in the files #3480 moved them out of"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- architecture
- diagnostics
priority: medium
references:
- Tests/Architecture/test_persistent_diagnostic_inventory.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only` fails on dev ecad404bd4 (2026-11-15). `REVIEWED_METADATA_ONLY_DIAGNOSTICS` expects two diagnostics in files that no longer contain them:
- "on_child_settled consumer raised" is expected in `Agents/agent_service.py`, but now lives in `Agents/agent_native_execution.py:1515`.
- "Failed to flush Console roleplay context on first persist" is expected in `Chat/console_chat_store.py`, but now lives in `Chat/console_store_first_persistence.py:142`.

Both diagnostics moved in 6125fbc5b8 (2026-11-08, #3480). The test is one of the three Tests/Architecture failures on dev's own baseline. Every architecture-review PR has subtracted it by hand since 2026-11-12, and while it is red it cannot catch a reviewed diagnostic that starts logging more than metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The reviewed list names each diagnostic's current owning file, with its metadata-only fields still checked
- [ ] #2 The test passes on dev serially
<!-- AC:END -->
