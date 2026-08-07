---
id: TASK-2768
title: Persistent-diagnostic inventory is stale on dev
status: To Do
assignee: []
created_date: '2026-08-07 06:42'
labels:
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
scripts/check_persistent_diagnostic_inventory.py fails on clean dev (verified at 22c08f958), so Tests/Architecture/test_persistent_diagnostic_inventory.py is red there. The regeneration diff is ~199 insertions spanning new diagnostic owners in Agents/local_tool_provider.py and Chat/prompt_history.py plus six UI/Console_Modules entries that the decomposition waves moved. Wave 3 deliberately did NOT run --write: the checker prints 'review the diff before running --write' because this is a security artifact, and signing off on two unrelated modules' new diagnostic owners is exactly the rubber-stamp that gate exists to prevent. Someone who owns those diagnostics should review and regenerate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each new diagnostic owner in the regeneration diff has been reviewed by someone who can vouch for it
- [ ] #2 The inventory is regenerated and Tests/Architecture/test_persistent_diagnostic_inventory.py passes on dev
<!-- AC:END -->
