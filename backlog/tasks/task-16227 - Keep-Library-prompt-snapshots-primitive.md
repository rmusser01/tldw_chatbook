---
id: TASK-16227
title: Keep Library prompt snapshots primitive
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 08:52'
updated_date: '2026-08-14 08:56'
labels:
  - library
  - state
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Library Prompt browse scope snapshots within the app state store's reviewed built-in-container boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library saves Prompt browse scope as a primitive mapping rather than a domain object
- [x] #2 Restore validates the mapping and reconstructs the exact PromptBrowseScope
- [x] #3 Legacy object snapshots and malformed mappings degrade safely
- [x] #4 Focused Library snapshot and ProductionApp privacy tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED evidence that Prompt browse scope snapshots are primitive and round-trip exactly.
2. Serialize the frozen scope to a dict and validate dict restoration while retaining in-memory legacy compatibility.
3. Add malformed-state characterization and run focused Library/ProductionApp/static checks.

ADR required: no
ADR path: N/A
Reason: This restores the existing screen-state privacy boundary without changing state ownership or persistence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Library now serializes PromptBrowseScope with dataclasses.asdict before entering the screen-state store, validates primitive dict snapshots through the domain constructor on restore, preserves legacy in-memory object compatibility, and degrades malformed mappings to the default scope. RED proved the old domain object crossed the snapshot boundary; focused Prompt round-trip tests and the full ProductionApp route/privacy tour are green. Ruff lint and py_compile pass. Ruff format remains the identical pre-existing whole-file drift in both legacy files; unrelated formatting churn was intentionally avoided. ADR required: no (state ownership and persistence are unchanged).
<!-- SECTION:NOTES:END -->
