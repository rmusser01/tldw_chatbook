---
id: TASK-34783
title: Correct architecture references against implemented runtime contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-10-10 19:27'
updated_date: '2026-10-10 19:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the new engineering reference accurate about live hooks, workflow execution, RAG provider costs, storage and module ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hooks documentation describes the wired v2 runtime and its permission/checkpoint requirements
- [x] #2 Workflow execution scope and reranking request counts match actual source
- [x] #3 Schema, module paths and storage descriptions avoid known stale or inaccurate claims
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-163/138/126 contracts apply. Reason: documentation correction preserves implemented behavior. 1. Verify reviewer findings against source. 2. Correct the affected references and AGENTS drift. 3. Check local links, preflight and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected live v2 hook ownership/effects/checkpoints, workflow session execution, uncached reranking operation counts and local-provider support; removed volatile schema numbers and inaccurate storage/module claims. Resolved AGENTS conflict with current tool-transcript ownership. Independent source review cleared the scoped changes after one local-provider wording correction; all37 local Markdown links resolve and derived-artifact preflight passes. No runtime behavior changes or new ADR; existingADR163/138/126 contracts remain. Documentation-only verification used source and link checks, without new runtime tests.
<!-- SECTION:NOTES:END -->
