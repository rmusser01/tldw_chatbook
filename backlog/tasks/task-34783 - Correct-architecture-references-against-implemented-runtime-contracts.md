---
id: TASK-34783
title: Correct architecture references against implemented runtime contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-10-10 19:27'
updated_date: '2026-10-10 20:40'
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

Stacked PR #3018 source-accuracy follow-up (2026-10-10): verify the Eval default/private override selectors, sparse YAML lifecycle and eval.definitions recovery owner; correct stale configuration, schema and persistence claims plus the database-path override sentence; validate relative links and source references before restoring AC #3 and Done.
ADR required: no
ADR path: backlog/decisions/220-evaluation-defaults-and-private-overrides.md; backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: copy-only alignment with the already implemented and reviewed configuration/recovery contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected live v2 hook ownership/effects/checkpoints, workflow session execution, uncached reranking operation counts and local-provider support; removed volatile schema numbers and inaccurate storage/module claims. Resolved AGENTS conflict with current tool-transcript ownership. Independent source review cleared the scoped changes after one local-provider wording correction; all37 local Markdown links resolve and derived-artifact preflight passes. No runtime behavior changes or new ADR; existingADR163/138/126 contracts remain. Documentation-only verification used source and link checks, without new runtime tests.

Stacked-source accuracy follow-up (2026-10-10, reviewed predecessors through fdb0db32364c778e047acead8cbad90a04458077): corrected the Eval reference to distinguish shipped defaults from sparse eval_overrides.yaml beside the effective config.toml, standalone inputs/full exports, raw YAML persistence and eval.definitions backup ownership. Genuine absence remains unused; invalid/unreadable override paths block complete backup, and retained legacy definitions stay inactive until authorized activation. Replaced stale Eval schema numbers with SCHEMA_VERSION, corrected word-bench HTTP ownership, and clarified explicit database path overrides. Verified against selectors, loader/save/recovery adapter, participant binding, EvalsDB and capture-client source under existing ADR-220/126. All83 relative Markdown links resolve and newly named source files exist; whitespace and Backlog file readability checks pass. Copy-only change; no runtime tests, full suite or new platform qualification. Root will run the final stacked preflight.
<!-- SECTION:NOTES:END -->
