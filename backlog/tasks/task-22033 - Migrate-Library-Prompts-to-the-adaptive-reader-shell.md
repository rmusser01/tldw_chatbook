---
id: TASK-22033
title: Migrate Library Prompts to the adaptive reader shell
status: To Do
assignee: []
created_date: '2026-08-24 23:26'
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
- [ ] #1 The Prompts list remains mounted beside a permanent work pane with independent list collapse and destination-specific geometry
- [ ] #2 Basic is the default mode and Basic Advanced and Info operate on one lossless item-owned draft
- [ ] #3 Saving from Basic preserves every Advanced-only field and validation can focus the owning mode
- [ ] #4 Create import history collections provenance lifecycle and destructive actions remain reachable without unmounting the list
- [ ] #5 Selection loading draft navigation stale workers conflicts deletion and retry follow the approved identity and recovery contracts
- [ ] #6 Existing Prompt capability and backend ownership remain unchanged
- [ ] #7 Automated browse editor hidden-field history geometry focus and capability tests pass with a representative live TUI walkthrough
<!-- AC:END -->
