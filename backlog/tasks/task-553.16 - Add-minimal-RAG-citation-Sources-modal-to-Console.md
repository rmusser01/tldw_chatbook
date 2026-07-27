---
id: TASK-553.16
title: Add minimal RAG citation Sources modal to Console
status: Done
assignee: []
created_date: '2026-07-27 15:34'
updated_date: '2026-07-27 17:39'
labels:
  - rag
  - citations
  - console
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-27-citation-evidence-inspector-design.md
  - Docs/superpowers/plans/2026-07-27-minimal-console-rag-citations.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
  - >-
    backlog/tasks/task-839 -
    Prevent-optional-MLX-imports-from-aborting-test-collection.md
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Console users see the exact local RAG chunks cited by a persisted answer and open supported original Library items without adding another provenance or resolution system.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected local RAG answers persist every eligible citation occurrence and retain valid cited-source mappings across restart.
- [x] #2 Persisted cited answers show a deduplicated Sources count; streaming, uncited, stale, and legacy-empty traces show no footer.
- [x] #3 Users can open one Sources modal and read the exact cited chunks as literal text.
- [x] #4 Supported local media, note, and conversation sources open through Library by exact stored identity; missing or unsupported items fail safely.
- [x] #5 Footer and modal reads run outside composition, discard stale results, and avoid loading chunk bodies until the modal opens.
- [x] #6 Scoped automated tests and touched-file static checks pass without expanding into unrelated baseline repair.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed plan: Docs/superpowers/plans/2026-07-27-minimal-console-rag-citations.md

1. Persist every eligible selected-answer marker occurrence in the existing trace.
2. Add one repository-owned current-message active-trace lookup.
3. Discover deduplicated footer counts outside composition and render Sources (N).
4. Lazily hydrate exact chunks in one revalidated modal.
5. Pass supported source identities to Library's existing exact-ID opener.

ADR required: no
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: Reuses existing citation and Library boundaries without changing architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Persisted selected-answer marker occurrences in the existing citation trace, added repository-owned current-message verification, and rendered deduplicated `Sources (N)` footers only for available persisted evidence. Added one lazy, all-or-nothing modal that revalidates least-privilege hydration, preserves exact stored chunks and identities, and renders untrusted text literally. Supported media, note, and conversation rows now pass an exact bounded two-key context to Library's existing exact-ID opener; unsupported and missing items fail through existing behavior.

ADR required: no. Reused `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`; no schema, provenance model, resolver framework, server dependency, or responsive inspector was added. The unrelated Console runtime-backend fixture baseline remains tracked separately as TASK-944.

Scoped post-rebase verification: 69 citation/Console tests passed; 8 filtered Library navigation tests passed; touched-file Ruff, CSS bundle sync, and `git diff --check` passed. Final cross-slice review findings (unavailable footer suppression and `Open in Library` copy) were fixed and re-reviewed.
<!-- SECTION:NOTES:END -->
