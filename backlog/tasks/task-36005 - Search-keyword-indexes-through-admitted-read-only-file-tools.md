---
id: TASK-36005
title: Search keyword indexes through admitted read-only file tools
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-21 14:32'
updated_date: '2026-10-21 18:08'
labels: []
dependencies:
  - TASK-36004
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-keyword-index.md
  - backlog/decisions/240-workspace-project-context-and-folder-bound-index.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Accelerate project discovery while keeping current files and the requesting run authoritative.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Only compatible readable local bindings advertise fs_search_index; strict bounded inputs return at most10 live hits and16KiB output with workspace-file provenance.
- [ ] #2 Run exclusion floors and current exclusions filter names and counts before ranking; current identity/hash and final cross-process authority acceptance prevent stale disclosures.
- [ ] #3 Index Allowed cannot bypass fs_read Blocked or Ask, existing path/instruction/review/budget guards, or a removed binding.
- [ ] #4 Saved Library choices/scopes and ordinary file tools remain unchanged; keyword lookup needs no embeddings/network and cached candidate p95 is measured against500ms.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement approved K4 service search and optional per-alias read-only local tool with immutable off-thread exact-run scope projection. Filter requesting frozen/accumulated and live exclusions before FTS ranking/LIMIT; capture current identity/hash/text via existing pinned private reader after both search and effective fs_read review, then final registry/store disclosure fence. Enforce strict inputs,100candidate/10hit/16KiB and smaller run budgets, no cached excerpts or instruction promotion. Preserve ordinary path/instruction/tool schema hashes and all saved Library policy/scope combinations. Add conditional fs_read Ask floor with existing approve_once/deny rows and exact run/call-ID ephemeral approval provenance via narrow console_interrupt_rounds hook; block broad/forged/replayed/sibling grants without persisting read/search grants. TDD real store/worker/provider and production controller composition, late mutation/exclusion races, aliases and permission tests; targeted covering selection and cached candidate p95<=500ms/live-validation separate profiling. ADR required: yes, existing accepted ADR. ADR path: backlog/decisions/240-workspace-project-context-and-folder-bound-index.md. Reason: direct implementation of its folder-bound file authority/tool disclosure boundary, alongside ADR101/102/174 and mandatory storage admission; no new ADR or semantic/Library policy changes.
<!-- SECTION:PLAN:END -->
