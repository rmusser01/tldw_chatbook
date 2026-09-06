---
id: TASK-31935
title: Vendor and validate the Canvas Mermaid grammar subset
status: To Do
assignee: []
created_date: '2026-09-06 22:11'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31934
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide reproducible offline grammar and Unicode inputs with explicit semantic admission inside QuickJS.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pinned input and output integrity, member allowlists, licenses and reproducible builds exclude lifecycle scripts and undeclared downloads.
- [ ] #2 The approved syntax produces closed semantic models and every excluded construct is rejected explicitly with bounded source-free errors.
- [ ] #3 Unicode segmentation and width inputs are pinned, and the parser executes in QuickJS without native globals or a module loader.
<!-- AC:END -->
