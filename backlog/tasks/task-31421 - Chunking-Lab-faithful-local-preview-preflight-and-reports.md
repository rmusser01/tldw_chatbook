---
id: TASK-31421
title: Chunking Lab - faithful local preview preflight and reports
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:10'
updated_date: '2026-09-04 23:25'
labels:
  - chunking
  - chunking-lab
dependencies: []
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable faithful unsaved-template experiments on the completed ADR-078 runtime without reviving the retired pipeline or changing the server-parity validator. Implements Chunking Lab spec sections 5-7 and AC 6-9, 14, 18-19. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: local execution and structured result contract. Execution baseline is current dev with TASK-19801 through TASK-19806 completed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Run and Lab Save use a named local capability preflight: unknown executable fields, unavailable assets, legacy shapes, and implicit network or LLM work are refused while metadata and classifier selection rules survive.
- [ ] #2 Unsaved preview and applying the same saved flat template share the existing runtime seam and produce equivalent full pre/chunk/post outputs; valid empty outputs never invoke default chunking.
- [ ] #3 Structured results retain supported engine and operation metadata with authoritative fields protected; source alignment is exact and verified or explicitly unavailable, including repeated text and transformations.
- [ ] #4 Real deterministic execution fixtures cover supported filtering, merging, context, dict-output behavior, and clear refusal of unsupported combinations; existing parity validation and vendor protections remain unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: faithful local execution and structured reports. Follow Task 1 of Docs/superpowers/plans/2026-09-04-chunking-lab.md: write failing capability and execution tests; implement immutable report models and separate Lab preflight; extend the shared non-vendored runtime seam; verify focused runtime/parity regressions; self-review and independent review before completion.
<!-- SECTION:PLAN:END -->
