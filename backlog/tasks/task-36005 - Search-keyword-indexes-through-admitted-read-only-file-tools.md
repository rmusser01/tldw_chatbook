---
id: TASK-36005
title: Search keyword indexes through admitted read-only file tools
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 14:32'
updated_date: '2026-10-21 21:57'
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
- [x] #1 Only compatible readable local bindings advertise fs_search_index; strict bounded inputs return at most10 live hits and16KiB output with workspace-file provenance.
- [x] #2 Run exclusion floors and current exclusions filter names and counts before ranking; current identity/hash and final cross-process authority acceptance prevent stale disclosures.
- [x] #3 Index Allowed cannot bypass fs_read Blocked or Ask, existing path/instruction/review/budget guards, or a removed binding.
- [x] #4 Saved Library choices/scopes and ordinary file tools remain unchanged; keyword lookup needs no embeddings/network and cached candidate p95 is measured against500ms.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement approved K4 service search and optional per-alias read-only local tool with immutable off-thread exact-run scope projection. Filter requesting frozen/accumulated and live exclusions before FTS ranking/LIMIT; capture current identity/hash/text via existing pinned private reader after both search and effective fs_read review, then final registry/store disclosure fence. Enforce strict inputs,100candidate/10hit/16KiB and smaller run budgets, no cached excerpts or instruction promotion. Preserve ordinary path/instruction/tool schema hashes and all saved Library policy/scope combinations. Add conditional fs_read Ask floor with existing approve_once/deny rows and exact run/call-ID ephemeral approval provenance via narrow console_interrupt_rounds hook; block broad/forged/replayed/sibling grants without persisting read/search grants. TDD real store/worker/provider and production controller composition, late mutation/exclusion races, aliases and permission tests; targeted covering selection and cached candidate p95<=500ms/live-validation separate profiling. ADR required: yes, existing accepted ADR. ADR path: backlog/decisions/240-workspace-project-context-and-folder-bound-index.md. Reason: direct implementation of its folder-bound file authority/tool disclosure boundary, alongside ADR101/102/174 and mandatory storage admission; no new ADR or semantic/Library policy changes. Measured common-query p95 missed500ms under concurrent and idle load; narrow query-local DB eligibility optimization plus exact suffix/prefix/exclusion equivalence and affected provider regressions is authorized before code. Preserve mandatory admission, rank/LIMIT filtering and raw failed timings; reuse stable cache for post-change query-only timing, no new scan or repeated unchanged Library suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented run-bound optional fs_search_index with pre-ranking exclusion floors, current-file identity/hash validation, final cross-process acceptance and bounded output. Search and effective fs_read review use exact one-call provenance across floor changes without changing ordinary approvals or saved Library controls/scopes. Shared service read admission bounds writer plus query operations; failed native retirement retains ownership. Final stale-result serialization stays within16KiB. Independent spec/quality review approved after fix1 (8ea34f516f ->4f744897ff); source base9e6d127a0d. Required59passed, DB22passed and affected9passed; fix covering122passed plus corrected fixture/final9approval/runtime passes and1retirement pass qualify final source. Unchanged Windows symlink setup failure, inherited lint/warnings and unexecuted platforms are retained, not passing evidence. Cached lexical final p95 common/rare/miss308.57/242.75/221.79ms after narrow query-local optimization; prior misses and cache-only reconstruction distinction retained. Actual production query-pin restore and full literal-scale qualification remain explicit release integration gates. Modified provider/composition/review/executor/read-root seams, service/models, narrow lexical filtering and targeted tests. ADR required: yes, existing accepted ADR240 (backlog/decisions/240-workspace-project-context-and-folder-bound-index.md); implements file-authority/disclosure/lifetime boundaries with ADR101/102/174, no semantic work or Library migration. Reports, raw receipts, source hashes and review are retained in Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/.
<!-- SECTION:NOTES:END -->
