---
id: TASK-22033
title: Migrate Library Prompts to the adaptive reader shell
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 23:26'
updated_date: '2026-08-26 19:17'
labels:
  - library
  - ui
dependencies:
  - TASK-22032
references:
  - >-
    Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md
  - backlog/decisions/086-library-adaptive-reader-shell.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move Prompts into the shared Library adaptive reader structure while preserving browse paging collections import history provenance validation optimistic updates and lifecycle behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Prompts list remains mounted beside a permanent work pane with independent list collapse and destination-specific geometry
- [x] #2 Basic is the default mode and Basic Advanced and Info operate on one lossless item-owned draft
- [x] #3 Saving from Basic preserves every Advanced-only field and validation can focus the owning mode
- [x] #4 Create import history collections provenance lifecycle and destructive actions remain reachable without unmounting the list
- [x] #5 Selection loading draft navigation stale workers conflicts deletion and retry follow the approved identity and recovery contracts
- [x] #6 Existing Prompt capability and backend ownership remain unchanged
- [x] #7 Automated browse editor hidden-field history geometry focus and capability tests pass with a representative live TUI walkthrough
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory Prompt capabilities and draft authority
2. Add one lossless reader projection
3. Split persistent list and work pane
4. Verify hidden fields, workflows, geometry, and focus

ADR required: yes
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md
Reason: consumes the accepted Library structural boundary without changing Prompt authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Migrated Library Prompts into the shared adaptive reader with retained Items and Work panes, independent collapse geometry, and Basic as the default projection. Added one screen-owned lossless Prompt draft shared by Basic, Advanced, and Info; explicit validation ownership; selected-versus-loaded detail fencing; truthful read-only recovery; retained browse, bulk, import, history, collection, conflict, lifecycle, and delete/Undo authority without adding a parallel persistence path. Hardened editor-origin import and browse/detail retry behavior after independent review. Verified after rebasing onto origin/dev with 15 focused retained-reader tests, 340 Prompt canvas tests, 1,126 broader Prompt state/controller/widget/service tests, Ruff, compileall, diff checks, and the complete isolated production-CSS live matrix. Full-repository collection remains independently blocked on the existing unregistered filterwarnings marker baseline in Tests/Agents/test_mcp_tool_provider.py. ADR required: yes. ADR path: backlog/decisions/086-library-adaptive-reader-shell.md. Reason: directly implements the accepted long-lived Library adaptive-reader boundary; no new ADR was required.

Post-Qodo hardening: removed the unused test-only PromptReaderState module so LibraryScreen remains the sole mutable reader authority; transferred hidden-field preservation and validation-focus coverage to the mounted production reader; fixed Info fallback so only unavailable Basic routes to Advanced; routed invalid outer saves to the owning Advanced block control; and confined config, data, app-data, and all XDG evidence-driver paths through centralized validation. Verified with 17 mounted reader tests, 10 seam/authority tests, focused mode and isolation regressions, Ruff, compileall, diff checks, and a fresh complete isolated live matrix. ADR required: yes. ADR path: backlog/decisions/086-library-adaptive-reader-shell.md. Reason: hardens the existing accepted boundary without adding a new architectural decision.
<!-- SECTION:NOTES:END -->
