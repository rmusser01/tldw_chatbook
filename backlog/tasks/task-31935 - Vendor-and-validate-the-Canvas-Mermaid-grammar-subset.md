---
id: TASK-31935
title: Vendor and validate the Canvas Mermaid grammar subset
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 22:11'
updated_date: '2026-09-06 23:52'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31934
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - >-
    Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide reproducible offline grammar and Unicode inputs with explicit semantic admission inside QuickJS.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pinned input and output integrity, member allowlists, licenses and reproducible builds exclude lifecycle scripts and undeclared downloads.
- [x] #2 The approved syntax produces closed semantic models and every excluded construct is rejected explicitly with bounded source-free errors.
- [x] #3 Unicode segmentation and width inputs are pinned, and the parser executes in QuickJS without native globals or a module loader.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing QuickJS-only semantic and vendoring integrity tests.
2. Pin authenticated Mermaid grammar maps, Unicode rules/data and notices, then implement closed semantic admission and shared parse budgets with reproducible inert output.
3. Register the real non-executable candidate and package data; verify semantic/integrity/static/reproducibility checks and obtain independent review.
ADR required: yes
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md (existing, extends ADR-121)
Reason: Direct implementation of the accepted offline library and pinned Unicode/runtime boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the pinned Mermaid 11.17.2 Jison grammar subset, Unicode 16.0.0 / UAX29 revision45 tables, closed semantic admission and shared document parse budgets. The stdlib-only Python 3.12.11 vendor authenticates the full archive and extracts only the declared maps, metadata and license; independent builds match every output byte. The real candidate retains verified assets and separate source-byte accounting while execution remains disabled and the diagram default remains null. Added explicit package data, separate notices and the test-only real-hash candidate fixture. Validation: 112 final Mermaid semantic/vendor tests passed (including all 1093 official Unicode rows and two offline rebuilds), plus 2 branch/rejoin cases; preceding affected run had 166 passes with one subsequently fixed input-preprocessing interruption, including successful legacy runtime rebuild and loopback checks. Scoped Ruff, formatter and node syntax checks pass; 3 existing lint findings outside the edited runtime-test range and the existing RequestsDependencyWarning are unchanged. ADR: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md. Compatibility and the comment-preprocessing testing lesson are updated. Parent review and status completion remain pending.

Review fix round 1: reproduced and fixed the production-47 trailing-whitespace compound-edge bypass; sequence comments now use upstream token boundaries and flow preprocessing tracks pipe labels, preserving literal percent pairs; emphasis refusal includes punctuation boundaries. Added exact-label, whitespace, formatting, directive and long-comment controls. Affected checks: 187 passed, 1 existing offline-runtime-cache skip, 1 unchanged loopback test deselected; 4 additional directive/long-sequence-comment controls passed. Candidate assets/catalog regenerated and two-build identity check passed; no budget, input-version or release-policy changes. Task remains In Progress for re-review.

Review fix round 2: underscore emphasis boundaries now include Unicode symbol categories as well as punctuation. RED: 12 symbol-boundary failures and 2 ordinary-label positives. GREEN: 147 Mermaid semantic/integrity/reproducibility checks passed with only the existing warning; scoped Ruff, formatter, node syntax and diff checks passed. Previous compound-edge and comment-label fixes are untouched; candidate remains non-executable. In Progress for scoped re-review.
<!-- SECTION:NOTES:END -->
